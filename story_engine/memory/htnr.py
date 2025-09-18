from __future__ import annotations

import json
import math
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import re


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    if denom <= 0.0:
        return 0.0
    return float(a.dot(b) / denom)


def _sent_tokenize_ko(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _mmr_select(query_vec: np.ndarray, sentences: List[str], encode_fn, topk: int = 2, lambda_mmr: float = 0.7) -> List[str]:
    if not sentences:
        return []
    try:
        sent_embs = encode_fn(sentences)
    except Exception:
        return sentences[:topk]
    sent_embs = np.asarray(sent_embs, dtype=np.float32)
    if sent_embs.ndim == 1:
        sent_embs = sent_embs.reshape(1, -1)
    sent_embs = np.array([_normalize(e) for e in sent_embs], dtype=np.float32)
    query_vec = _normalize(query_vec)
    selected: List[int] = []
    candidates = list(range(len(sentences)))
    sims = sent_embs @ query_vec
    while candidates and len(selected) < topk:
        mmr_scores = []
        for idx in candidates:
            relevance = float(sims[idx])
            redundancy = 0.0
            if selected:
                redundancy = max(float(sent_embs[idx].dot(sent_embs[j])) for j in selected)
            score = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy
            mmr_scores.append((score, idx))
        mmr_scores.sort(key=lambda x: x[0], reverse=True)
        best = mmr_scores[0][1]
        selected.append(best)
        candidates.remove(best)
    return [sentences[i] for i in selected]


@dataclass
class HTNRNode:
    id: int
    content: str
    emb: np.ndarray
    V: float
    D: float
    S: float
    R: float
    parent: Optional[int]
    children: List[int]
    birth_step: int
    chapter: int = 0
    canonical: bool = True
    source: str = "Unknown"
    tags: tuple = ()


class HTNRMemoryV2:
    def __init__(
        self,
        embedder: Any,
        *,
        embedding_model_name: str = "text-embedding-3-small",
        max_leaves: int = 10,
        alpha: float = 0.9,
        beta: float = 1.0,
        gamma: float = 0.6,
        delta: float = 1.0,
        nn_k: int = 4,
        anchor_recent: int = 4,
        descend_margin: float = 0.02,
        tau_w: float = 0.2,
        tau_R: float = 0.65,
        shrink_lambda: float = 0.10,
        bridge_zeta: float = 2.0,
        score_alpha: float = 1.0,
        K_cap: int = 12,
        leaf_sample: int = 32,
        segmenter: Optional[Any] = None,
    ) -> None:
        self.model = embedder
        self.model_name = embedding_model_name
        dim = self.model.get_sentence_embedding_dimension()
        self.alpha = alpha
        self.nn_k = nn_k
        self.anchor_recent = anchor_recent
        self.descend_margin = descend_margin
        self.max_leaves = max_leaves
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.tau_w = tau_w
        self.tau_R = tau_R
        self.shrink_lambda = shrink_lambda
        self.bridge_zeta = bridge_zeta
        self.score_alpha = score_alpha
        self.K_cap = K_cap
        self.leaf_sample = max(4, int(leaf_sample))
        self.M = np.zeros(dim, dtype=np.float32)
        self.prevM = np.zeros(dim, dtype=np.float32)
        self.smoothed_lambda = 0.05
        self.nodes: Dict[int, HTNRNode] = {}
        self.leaves: List[int] = []
        self.next_id = 0
        self.step = 0
        self._leaf_cache_mat: Optional[np.ndarray] = None
        self._leaf_cache_ids: Optional[List[int]] = None
        self.segment = segmenter if segmenter else _sent_tokenize_ko

    def _lambda_t(self) -> float:
        drift = 1.0 - _cos(self.M, self.prevM)
        lam = max(0.005, min(0.5 * max(0.0, drift), 0.3))
        self.smoothed_lambda = 0.1 * lam + 0.9 * self.smoothed_lambda
        return lam

    def _update_V_for_leaves(self, lam: float) -> None:
        if not self.leaves:
            return
        for nid in self.leaves:
            node = self.nodes[nid]
            node.R = _cos(node.emb, self.M)
            node.V = (1 - lam) * node.V + lam * node.R

    def _new_node(
        self,
        content: str,
        emb: np.ndarray,
        *,
        V: float,
        D: float,
        parent: Optional[int],
        children: Optional[List[int]],
        S: float = 1.0,
        R: float = 1.0,
        chapter: int = 0,
        canonical: bool = True,
        source: str = "Unknown",
        tags: tuple = (),
    ) -> int:
        nid = self.next_id
        self.next_id += 1
        self.nodes[nid] = HTNRNode(
            id=nid,
            content=content,
            emb=emb,
            V=V,
            D=D,
            S=S,
            R=R,
            parent=parent,
            children=list(children or []),
            birth_step=self.step,
            chapter=chapter,
            canonical=canonical,
            source=source,
            tags=tuple(tags),
        )
        if parent is not None and parent in self.nodes:
            self.nodes[parent].children.append(nid)
        return nid

    def _eligible_leaves(self) -> List[int]:
        if len(self.leaves) <= self.anchor_recent:
            return []
        return self.leaves[:-self.anchor_recent]

    def _leaf_bridge_S(self, idx_i: int, embs: np.ndarray) -> float:
        vi = embs[idx_i]
        sims = embs @ vi
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(vi)
        cosines = sims / np.maximum(norms, 1e-8)
        if len(cosines) > 1:
            cosines = np.delete(cosines, idx_i)
        mu = float(np.mean(cosines)) if len(cosines) else 0.0
        var = float(np.var(cosines)) if len(cosines) else 0.0
        bridge = (1.0 - mu) + var
        return 1.0 + self.bridge_zeta * max(0.0, bridge)

    def _parent_S_from_children(self, child_ids: List[int], parent_emb: np.ndarray) -> float:
        if not child_ids:
            return 1.0
        count = len(child_ids)
        if count == 1:
            return 1.0
        child_embs = np.stack([self.nodes[c].emb for c in child_ids], axis=0)
        child_embs = child_embs / (np.linalg.norm(child_embs, axis=1, keepdims=True) + 1e-8)
        sims = child_embs @ child_embs.T
        triu = np.triu_indices(count, k=1)
        pair_mean_dist = float(np.mean(1.0 - sims[triu])) if triu[0].size > 0 else 0.0
        gini_simpson = 1.0 - (1.0 / count)
        size_factor = math.log1p(count)
        return 1.0 + size_factor * (0.5 * pair_mean_dist + 0.5 * gini_simpson)

    def _pair_priority_score(
        self,
        i: int,
        j: int,
        Vi: float,
        Vj: float,
        Si: float,
        Sj: float,
        Di: float,
        Dj: float,
        Wij: float,
    ) -> float:
        if Wij < self.tau_w:
            return -1.0
        num = 1.0 / (1.0 + math.exp(-self.score_alpha * Wij))
        den = 1e-3 + self.beta * (Vi + Vj) + (Si + Sj) + self.delta * (Di + Dj)
        return float(num / den)

    def _refresh_leaf_cache(self) -> None:
        if not self.leaves:
            self._leaf_cache_mat = None
            self._leaf_cache_ids = None
            return
        embs = np.stack([self.nodes[c].emb for c in self.leaves], axis=0)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self._leaf_cache_mat = embs.astype(np.float32)
        self._leaf_cache_ids = list(self.leaves)

    def _leaf_knn_pairs(self, candidates: List[int]) -> List[Tuple[float, int, int]]:
        if len(candidates) < 2:
            return []
        if (self._leaf_cache_ids is None) or (self._leaf_cache_ids != self.leaves):
            self._refresh_leaf_cache()
        if not self._leaf_cache_ids or self._leaf_cache_mat is None:
            return []
        id2pos = {nid: idx for idx, nid in enumerate(self._leaf_cache_ids)}
        if len(candidates) > self.leaf_sample:
            candidates = random.sample(candidates, self.leaf_sample)
        mat = self._leaf_cache_mat
        selected_idx = [id2pos[nid] for nid in candidates if nid in id2pos]
        pairs: List[Tuple[float, int, int]] = []
        for idx_i in selected_idx:
            vi = mat[idx_i]
            cosines = mat @ vi
            order = np.argsort(-cosines)
            Si = self._leaf_bridge_S(idx_i, mat)
            i = self._leaf_cache_ids[idx_i]
            cnt = 0
            for rank in order:
                if rank == idx_i:
                    continue
                j = self._leaf_cache_ids[rank]
                if j not in candidates:
                    continue
                Wij = float(cosines[rank])
                Sj = self._leaf_bridge_S(rank, mat)
                ni, nj = self.nodes[i], self.nodes[j]
                score = self._pair_priority_score(i, j, ni.V, nj.V, Si, Sj, ni.D, nj.D, Wij)
                if score > 0:
                    pairs.append((score, i, j))
                    cnt += 1
                if cnt >= self.nn_k:
                    break
        pairs.sort(key=lambda x: x[0], reverse=True)
        return pairs

    def _try_build_parent(self, i: int, j: int) -> Optional[int]:
        ni, nj = self.nodes[i], self.nodes[j]
        weights = np.array([max(1e-6, ni.V), max(1e-6, nj.V)], dtype=np.float32)
        weights = weights / (np.sum(weights) + 1e-8)
        base = (1.0 - self.shrink_lambda) * (weights[0] * ni.emb + weights[1] * nj.emb) + self.shrink_lambda * self.M
        new_emb = _normalize(base)
        R = float(np.linalg.norm(weights[0] * ni.emb + weights[1] * nj.emb))
        if R < self.tau_R:
            return None
        R_align = _cos(new_emb, self.M)
        new_V = 0.7 * 0.5 * (ni.V + nj.V) + 0.3 * R_align
        new_D = max(ni.D, nj.D)
        sentences: List[str] = []
        for text in [ni.content, nj.content]:
            if text:
                try:
                    sentences.extend(self.segment(text))
                except Exception:
                    sentences.extend(_sent_tokenize_ko(text))
        try:
            proxy_lines = _mmr_select(new_emb, sentences, self.model.encode, topk=2, lambda_mmr=0.7)
        except Exception:
            proxy_lines = []
        parent_content = " / ".join(proxy_lines) if proxy_lines else (ni.content or nj.content or "[합성]")
        S_parent = self._parent_S_from_children([i, j], new_emb)
        new_id = self._new_node(
            content=parent_content,
            emb=new_emb,
            V=new_V,
            D=new_D,
            parent=None,
            children=[i, j],
            S=S_parent,
            R=R,
            chapter=max(ni.chapter, nj.chapter),
            canonical=(ni.canonical and nj.canonical),
            source="Merge",
            tags=tuple(set(ni.tags + nj.tags)),
        )
        self.nodes[i].parent = new_id
        self.nodes[j].parent = new_id
        return new_id

    def _merge_once(self) -> bool:
        candidates = self._eligible_leaves()
        if len(candidates) < 2:
            return False
        pairs = self._leaf_knn_pairs(candidates)
        if not pairs:
            return False
        merged = False
        used: set[int] = set()
        for _, i, j in pairs:
            if i in used or j in used:
                continue
            new_id = self._try_build_parent(i, j)
            if new_id is None:
                continue
            self.leaves = [x for x in self.leaves if x not in (i, j)]
            self.leaves.append(new_id)
            used.update({i, j, new_id})
            merged = True
        if merged:
            self._refresh_leaf_cache()
        return merged

    def process_and_add(
        self,
        text_chunk: str,
        source: str = "Unknown",
        *,
        chapter: int = 0,
        canonical: bool = True,
        tags: tuple = (),
    ) -> None:
        if not text_chunk:
            return
        emb = self.model.encode(text_chunk)
        if isinstance(emb, np.ndarray) and emb.ndim > 1:
            emb = emb[0]
        emb = _normalize(emb)
        self.prevM = self.M.copy()
        self.M = _normalize(self.alpha * self.M + (1 - self.alpha) * emb)
        lam = self._lambda_t()
        self._update_V_for_leaves(lam)
        D_birth = 1.0 - _cos(self.M, self.prevM)
        nid = self._new_node(
            text_chunk,
            emb,
            V=1.0,
            D=D_birth,
            parent=None,
            children=None,
            S=1.0,
            R=1.0,
            chapter=chapter,
            canonical=canonical,
            source=source,
            tags=tags,
        )
        self.leaves.append(nid)
        self.step += 1
        safety = 0
        while len(self.leaves) > self.max_leaves and safety < 8:
            if not self._merge_once():
                break
            safety += 1
        self._refresh_leaf_cache()

    def retrieve_for_query(
        self,
        query: str,
        *,
        chapter_limit: Optional[int] = None,
        canonical_only: bool = True,
    ) -> List[str]:
        if not self.leaves:
            return []
        try:
            q = self.model.encode(query)
        except Exception:
            return []
        if isinstance(q, np.ndarray) and q.ndim > 1:
            q = q[0]
        q = _normalize(q)

        def eligible(nid: int) -> bool:
            node = self.nodes[nid]
            if canonical_only and not node.canonical:
                return False
            if chapter_limit is not None and node.chapter > chapter_limit:
                return False
            return True

        pool_ids = [nid for nid in self.leaves if eligible(nid)]
        if not pool_ids:
            return []

        def score(nid: int) -> float:
            node = self.nodes[nid]
            return self.gamma * _cos(q, node.emb) + (1 - self.gamma) * _cos(self.M, node.emb)

        pool = sorted(((score(nid), nid) for nid in pool_ids), key=lambda x: -x[0])
        K = min(8, self.K_cap, len(pool))
        out: List[str] = []
        visited: set[int] = set()
        while pool and len(out) < K:
            sc, nid = pool.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = self.nodes[nid]
            must_descend = bool(node.children) and node.R < self.tau_R
            if not node.children or not must_descend:
                if node.children:
                    child_scores = sorted(((score(c), c) for c in node.children), key=lambda x: -x[0])
                    best_child_score = child_scores[0][0] if child_scores else -1e9
                    if best_child_score >= sc - self.descend_margin:
                        pool = child_scores + pool
                        continue
                out.append(node.content if node.content else "[요약 없음]")
                continue
            child_scores = sorted(((score(c), c) for c in node.children), key=lambda x: -x[0])
            pool = child_scores + pool
        return out


def _to_blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def _ensure_htnr_schema(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS htnr_meta(
            id INTEGER PRIMARY KEY CHECK (id=1),
            model_name TEXT,
            dim INTEGER,
            alpha REAL,
            beta REAL,
            gamma REAL,
            delta REAL,
            tau_w REAL,
            tau_R REAL,
            shrink_lambda REAL,
            bridge_zeta REAL,
            score_alpha REAL,
            K_cap INTEGER,
            max_leaves INTEGER,
            nn_k INTEGER,
            anchor_recent INTEGER,
            descend_margin REAL,
            step INTEGER,
            next_id INTEGER,
            smoothed_lambda REAL,
            M BLOB,
            prevM BLOB
        )"""
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS htnr_nodes(
            id INTEGER PRIMARY KEY,
            content TEXT,
            V REAL,
            D REAL,
            S REAL,
            R REAL,
            parent INTEGER,
            children_json TEXT,
            birth_step INTEGER,
            emb BLOB,
            chapter INTEGER DEFAULT 0,
            canonical INTEGER DEFAULT 1,
            source TEXT DEFAULT 'Unknown',
            tags_json TEXT DEFAULT '[]'
        )"""
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS htnr_leaves(
            pos INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id INTEGER
        )"""
    )
    con.commit()


def _ensure_and_migrate_htnr_schema(con: sqlite3.Connection) -> None:
    _ensure_htnr_schema(con)
    cur = con.cursor()
    cols = {row[1] for row in cur.execute("PRAGMA table_info(htnr_nodes)").fetchall()}
    if "chapter" not in cols:
        cur.execute("ALTER TABLE htnr_nodes ADD COLUMN chapter INTEGER DEFAULT 0")
    if "canonical" not in cols:
        cur.execute("ALTER TABLE htnr_nodes ADD COLUMN canonical INTEGER DEFAULT 1")
    if "source" not in cols:
        cur.execute("ALTER TABLE htnr_nodes ADD COLUMN source TEXT DEFAULT 'Unknown'")
    if "tags_json" not in cols:
        cur.execute("ALTER TABLE htnr_nodes ADD COLUMN tags_json TEXT DEFAULT '[]'")
    cur.execute("UPDATE htnr_nodes SET chapter=0 WHERE chapter IS NULL")
    cur.execute("UPDATE htnr_nodes SET canonical=1 WHERE canonical IS NULL")
    cur.execute("UPDATE htnr_nodes SET source='Unknown' WHERE source IS NULL")
    cur.execute("UPDATE htnr_nodes SET tags_json='[]' WHERE tags_json IS NULL")
    con.commit()


def htnr_save(memory: HTNRMemoryV2, db_path: str) -> None:
    con = sqlite3.connect(db_path)
    _ensure_and_migrate_htnr_schema(con)
    cur = con.cursor()
    cur.execute("DELETE FROM htnr_meta")
    cur.execute("DELETE FROM htnr_nodes")
    cur.execute("DELETE FROM htnr_leaves")
    dim = memory.M.shape[0]
    cur.execute(
        """
        INSERT INTO htnr_meta(
            id, model_name, dim, alpha, beta, gamma, delta, tau_w, tau_R,
            shrink_lambda, bridge_zeta, score_alpha, K_cap, max_leaves, nn_k,
            anchor_recent, descend_margin, step, next_id, smoothed_lambda, M, prevM
        ) VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            memory.model_name,
            dim,
            memory.alpha,
            memory.beta,
            memory.gamma,
            memory.delta,
            memory.tau_w,
            memory.tau_R,
            memory.shrink_lambda,
            memory.bridge_zeta,
            memory.score_alpha,
            memory.K_cap,
            memory.max_leaves,
            memory.nn_k,
            memory.anchor_recent,
            memory.descend_margin,
            memory.step,
            memory.next_id,
            float(memory.smoothed_lambda),
            _to_blob(memory.M),
            _to_blob(memory.prevM),
        ),
    )
    for node in memory.nodes.values():
        cur.execute(
            """
            INSERT INTO htnr_nodes(
                id, content, V, D, S, R, parent, children_json, birth_step,
                emb, chapter, canonical, source, tags_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                node.id,
                node.content,
                float(node.V),
                float(node.D),
                float(node.S),
                float(node.R),
                (node.parent if node.parent is not None else -1),
                json.dumps(node.children or []),
                int(node.birth_step),
                _to_blob(node.emb),
                int(node.chapter),
                1 if node.canonical else 0,
                node.source,
                json.dumps(list(node.tags)),
            ),
        )
    for nid in memory.leaves:
        cur.execute("INSERT INTO htnr_leaves(node_id) VALUES (?)", (nid,))
    con.commit()
    con.close()


def htnr_load(db_path: str, embedder: Any) -> Optional[HTNRMemoryV2]:
    if not os.path.exists(db_path):
        return None
    con = sqlite3.connect(db_path)
    _ensure_and_migrate_htnr_schema(con)
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT model_name,dim,alpha,beta,gamma,delta,tau_w,tau_R,shrink_lambda,
               bridge_zeta,score_alpha,K_cap,max_leaves,nn_k,anchor_recent,
               descend_margin,step,next_id,smoothed_lambda,M,prevM
        FROM htnr_meta WHERE id=1
        """
    ).fetchone()
    if not row:
        con.close()
        return None
    (
        model_name,
        dim,
        alpha,
        beta,
        gamma,
        delta,
        tau_w,
        tau_R,
        shrink_lambda,
        bridge_zeta,
        score_alpha,
        K_cap,
        max_leaves,
        nn_k,
        anchor_recent,
        descend_margin,
        step,
        next_id,
        smoothed_lambda,
        M_blob,
        prevM_blob,
    ) = row
    current_dim = embedder.get_sentence_embedding_dimension()
    if current_dim != dim:
        con.close()
        return HTNRMemoryV2(
            embedder,
            embedding_model_name=model_name,
            max_leaves=max_leaves,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            nn_k=nn_k,
            anchor_recent=anchor_recent,
            descend_margin=descend_margin,
            tau_w=tau_w,
            tau_R=tau_R,
            shrink_lambda=shrink_lambda,
            bridge_zeta=bridge_zeta,
            score_alpha=score_alpha,
            K_cap=K_cap,
        )
    memory = HTNRMemoryV2(
        embedder,
        embedding_model_name=model_name,
        max_leaves=max_leaves,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        nn_k=nn_k,
        anchor_recent=anchor_recent,
        descend_margin=descend_margin,
        tau_w=tau_w,
        tau_R=tau_R,
        shrink_lambda=shrink_lambda,
        bridge_zeta=bridge_zeta,
        score_alpha=score_alpha,
        K_cap=K_cap,
    )
    memory.M = _from_blob(M_blob, dim)
    memory.prevM = _from_blob(prevM_blob, dim)
    memory.smoothed_lambda = float(smoothed_lambda)
    memory.nodes.clear()
    memory.leaves.clear()
    memory.next_id = int(next_id)
    memory.step = int(step)
    rows = cur.execute(
        """
        SELECT id,content,V,D,S,R,parent,children_json,birth_step,emb,chapter,canonical,source,tags_json
        FROM htnr_nodes
        """
    ).fetchall()
    for (
        node_id,
        content,
        V,
        D,
        S,
        R,
        parent,
        children_json,
        birth_step,
        emb_blob,
        chapter,
        canonical,
        source,
        tags_json,
    ) in rows:
        emb = _from_blob(emb_blob, memory.M.shape[0])
        memory.nodes[int(node_id)] = HTNRNode(
            id=int(node_id),
            content=content,
            emb=emb,
            V=float(V),
            D=float(D),
            S=float(S),
            R=float(R),
            parent=None if parent == -1 else int(parent),
            children=list(json.loads(children_json or "[]")),
            birth_step=int(birth_step),
            chapter=int(chapter or 0),
            canonical=bool(canonical),
            source=source or "Unknown",
            tags=tuple(json.loads(tags_json or "[]")),
        )
    for (nid,) in cur.execute("SELECT node_id FROM htnr_leaves ORDER BY pos ASC").fetchall():
        memory.leaves.append(int(nid))
    con.close()
    return memory
