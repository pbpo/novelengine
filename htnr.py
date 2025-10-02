# -*- coding: utf-8 -*-
"""
HTNRMemory V54P+ (TurboQA 패치 포함)
- 적응형 쿼리 블렌딩(auto)
- 제약 병합(교차 문서 금지, 부모에 doc_id 전파)
- 부모 요약 MMR top‑k 하이퍼(동적)
- GPU 가속(torch 존재시 leaf@q, leaf@leafᵀ)
- Batch‑Freeze 컨텍스트 갱신 (배치 단위 컨텍스트 고정)

드롭‑인 교체 파일: 기존 compare_htnr_rag_narrative.py와 호환.
"""

from __future__ import annotations

import os
import io
import re
import math
import time
import json
import random
import logging
import hashlib
import threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Protocol, Deque, Sequence
from collections import deque, OrderedDict, defaultdict

import numpy as np

# ---------- Optional ANN ----------
try:
    import hnswlib as _hnswlib
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

# ---------- Optional Torch (GPU) ----------
try:
    import torch
    _HAS_TORCH = True
    _TORCH_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH = False
    _TORCH_CUDA = False

# ---------- Logging ----------
logger = logging.getLogger("HTNRV54P+")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# ---------- Numerics / Hyper ----------
@dataclass
class Numerics:
    eps_norm: float = 1e-8
    eps_cov: float = 1e-6
    ema_gamma_fast: float = 0.2
    ema_gamma_slow: float = 0.85
    max_rate_tau_w: float = 0.03
    max_rate_tau_R: float = 0.02
    cos_lower: float = -0.999
    cos_upper: float = 0.999


@dataclass
class Hyper:
    # Capacity / buffer / heads
    max_leaves: int = 48
    buffer_size: int = 16
    K_contexts: int = 3
    anchor_recent: int = 6
    eventlog_enabled: bool = False
    eventlog_buffer: int = 128
    eventlog_truncate: int = 16384
    # Candidate / merge / maintenance budgets
    cand_prefilter: int = 256
    knn_topk_pairs: int = 4
    max_merges_per_cycle: int = 4
    max_phaseR_per_cycle: int = 3

    # Embedding shrink toward context
    shrink_lambda: float = 0.08

    # Interference / homeostasis
    interference_q: float = 0.95
    homeo_interval: int = 50

    # Chaptering
    min_chapter_gap: int = 24

    # RAG fallback
    rag_topk: int = 6
    rag_entropy_tau: float = 1.0
    rag_entropy_thr: float = 1.2
    rag_top1_thr: float = 0.65

    # ANN options
    ann_backend: str = "auto"   # {"auto","hnsw","exact","none"}
    ann_min_leaves: int = 128
    ann_M: int = 32
    ann_efC: int = 200
    ann_efS: int = 64

    # ---- 프로파일 프리셋 ----
    profile: str = "QA"  # {"QA","Streaming"}

    # ---- 쿼리 블렌딩 ----
    query_blend: str = "auto"        # {"none","fixed","auto"}
    query_blend_fixed: float = 0.40  # fixed 모드에서 M_agg 가중치
    query_blend_cos_low: float = 0.20
    query_blend_cos_high: float = 0.80

    # ---- 요약 품질 ----
    mmr_topk_parent: int = 6
    mmr_topk_parent_dynamic: bool = True

    # ---- 병합 정책 ----
    opportunistic_merge: bool = False        # QA: False 권장
    merge_cross_doc: bool = False            # QA: False 권장

    # ---- GPU 가속 ----
    use_torch: bool = False
    torch_force_cpu: bool = False
    gpu_max_pairwise: int = 12000  # Exact 유사도 행렬 GPU 계산 허용 최대 L

    # ---- Batch‑Freeze ----
    batch_freeze_enable: bool = False
    batch_freeze_size: int = 128  # 64~512 권장


# ---------- Utils ----------
def _normalize(x: np.ndarray, eps: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n <= eps:
            logger.debug("normalize: zero-norm vector encountered")
            return arr
        return arr / n
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    zeros = np.sum(n <= eps)
    if zeros:
        logger.debug(f"normalize: {int(zeros)} zero-norm rows")
    out = np.divide(arr, (n + eps), out=np.zeros_like(arr), where=(n > eps))
    return out


def _cos(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    if den <= 0.0:
        return 0.0
    v = float(a.dot(b) / den)
    return max(-1.0, min(1.0, v))


def _stable_hash(text: str) -> int:
    h = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _status_code_of(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        for attr in ("status_code", "status"):
            val = getattr(resp, attr, None)
            if isinstance(val, int):
                return val
    return None


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n\u3002\uFF01\uFF1F\"])[\s\)\]\u3001\u3000]*")


def sent_tokenize(text: str) -> List[str]:
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s and s.strip()]


def jaccard(a: str, b: str) -> float:
    A, B = set(a.lower().split()), set(b.lower().split())
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _normalize_rows(matrix: Any) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm == 0.0:
            return arr
        return arr / norm
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass
class ContextItem:
    text: str
    score: float
    source: str
    doc_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class SimpleVectorIndex:
    def __init__(self, embedder: "EmbedderProtocol", items: Sequence[Chunk]) -> None:
        if not items:
            raise ValueError("벡터 인덱스에 항목이 없습니다.")
        self.embedder = embedder
        self.items = list(items)
        texts = [chunk.text for chunk in self.items]
        self.embeddings = _normalize_rows(self.embedder.encode(texts))

    def search(self, query: str, limit: int) -> List[ContextItem]:
        if limit <= 0:
            return []
        q_vec = _normalize_rows(self.embedder.encode([query]))[0]
        sims = self.embeddings @ q_vec
        order = np.argsort(-sims)[:limit]
        contexts: List[ContextItem] = []
        for rank, idx in enumerate(order, start=1):
            chunk = self.items[int(idx)]
            contexts.append(
                ContextItem(
                    text=chunk.text,
                    score=float(sims[int(idx)]),
                    source="RAG",
                    doc_id=chunk.doc_id,
                    meta={"chunk_id": chunk.chunk_id, "rank": rank},
                )
            )
        return contexts


class DocVectorIndex(SimpleVectorIndex):
    def __init__(self, embedder: "EmbedderProtocol", items: Sequence[Chunk]) -> None:
        super().__init__(embedder, items)
        self.by_doc: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(self.items):
            self.by_doc[chunk.doc_id].append(idx)

    def search_in_doc(self, query: str, doc_id: str, limit: int = 3) -> List[ContextItem]:
        ids = self.by_doc.get(doc_id, [])
        if not ids:
            return []
        q_vec = _normalize_rows(self.embedder.encode([query]))[0]
        sims = (self.embeddings[ids] @ q_vec)
        order = np.argsort(-sims)[:limit]
        out: List[ContextItem] = []
        for rank, loc in enumerate(order, start=1):
            item_idx = ids[int(loc)]
            chunk = self.items[item_idx]
            out.append(
                ContextItem(
                    text=chunk.text,
                    score=float(sims[int(loc)]),
                    source="DOC",
                    doc_id=chunk.doc_id,
                    meta={"chunk_id": chunk.chunk_id, "rank_in_doc": rank},
                )
            )
        return out


def _ctx_doc_id(ctx: Any) -> Optional[str]:
    if isinstance(ctx, ContextItem):
        return ctx.doc_id
    if isinstance(ctx, dict):
        doc_id = ctx.get("doc_id")
        if doc_id:
            return str(doc_id)
        meta = ctx.get("meta")
        if isinstance(meta, dict):
            maybe = meta.get("doc_id") or meta.get("document_id")
            if maybe:
                return str(maybe)
        source = ctx.get("source")
    else:
        return None
    if isinstance(source, str) and source.startswith("doc:"):
        tail = source.split(":", 1)[1]
        return tail.split("#", 1)[0] if tail else None
    return None


def _clone_context_item(ctx: Any) -> ContextItem:
    if isinstance(ctx, ContextItem):
        meta = dict(ctx.meta) if ctx.meta else {}
        return ContextItem(text=ctx.text, score=ctx.score, source=ctx.source, doc_id=ctx.doc_id, meta=meta)
    if isinstance(ctx, dict):
        meta = ctx.get("meta")
        meta = dict(meta) if isinstance(meta, dict) else {}
        text = ctx.get("text") or ctx.get("content") or ""
        score = float(ctx.get("score", 0.0))
        doc_id = ctx.get("doc_id") or meta.get("doc_id")
        source = ctx.get("source", "HTNR")
        if isinstance(source, str) and source.startswith("doc:") and not doc_id:
            tail = source.split(":", 1)[1]
            if tail:
                doc_id = tail.split("#", 1)[0]
                meta.setdefault("chunk_id", tail)
        return ContextItem(text=text, score=score, source=str(source), doc_id=(str(doc_id) if doc_id else None), meta=meta)
    raise TypeError("unsupported context type")


def build_evidence_from_htnr(
    question: str,
    htnr_contexts: Sequence[Any],
    doc_index: DocVectorIndex,
    *,
    per_doc: int = 3,
    max_total: int = 8,
    adjacent: int = 1,
) -> List[ContextItem]:
    if doc_index is None:
        raise ValueError("doc_index must not be None")

    base_per_doc = max(1, int(per_doc))
    adjacent = max(0, int(adjacent))
    q_lower = (question or "").lower()
    need_more = any(t in q_lower for t in ("why", "how", "explain", "reason", "because"))
    ask_when = any(t in q_lower for t in ("when", "date", "year"))
    ask_where = any(t in q_lower for t in ("where", "place", "location"))
    ask_who = q_lower.strip().startswith("who")
    if need_more:
        per_doc_eff = max(base_per_doc, 6)
        adjacent_eff = max(adjacent, 2)
    elif ask_when or ask_where or ask_who:
        per_doc_eff = max(base_per_doc, 4)
        adjacent_eff = max(adjacent, 2 if ask_when else 1)
    else:
        per_doc_eff = max(base_per_doc, 4)
        adjacent_eff = max(adjacent, 1)

    out: List[ContextItem] = []
    seen: set[Tuple[Optional[str], Optional[str]]] = set()

    by_doc: Dict[str, List[int]] = getattr(doc_index, "by_doc", {})  # type: ignore[attr-defined]
    items: Sequence[Chunk] = getattr(doc_index, "items", [])  # type: ignore[attr-defined]
    positions_cache: Dict[str, Dict[str, int]] = {}

    def _mark_and_add(ci: ContextItem) -> None:
        chunk_key = ci.meta.get("chunk_id") if ci.meta else None
        key = (ci.doc_id, chunk_key)
        if key in seen:
            return
        seen.add(key)
        out.append(ci)

    def _positions_for(doc_id: str) -> Dict[str, int]:
        cached = positions_cache.get(doc_id)
        if cached is not None:
            return cached
        mapping: Dict[str, int] = {}
        doc_indices = by_doc.get(doc_id, [])
        for pos, item_idx in enumerate(doc_indices):
            if 0 <= item_idx < len(items):
                mapping[items[item_idx].chunk_id] = pos
        positions_cache[doc_id] = mapping
        return mapping

    def _neighbor_context(doc_id: str, pos: int) -> Optional[ContextItem]:
        doc_indices = by_doc.get(doc_id, [])
        if not doc_indices or pos < 0 or pos >= len(doc_indices):
            return None
        idx = doc_indices[pos]
        if idx < 0 or idx >= len(items):
            return None
        chunk = items[idx]
        return ContextItem(
            text=chunk.text,
            score=0.0,
            source="DOC",
            doc_id=chunk.doc_id,
            meta={"chunk_id": chunk.chunk_id, "adjacent": True},
        )

    for ctx in htnr_contexts:
        doc_id = _ctx_doc_id(ctx)
        if not doc_id:
            continue
        extras = doc_index.search_in_doc(question, doc_id, limit=per_doc_eff)
        pos_map = _positions_for(doc_id)
        doc_indices = by_doc.get(doc_id, [])
        for extra in extras:
            meta = extra.meta or {}
            chunk_key = meta.get("chunk_id")
            _mark_and_add(extra)
            if len(out) >= max_total:
                break
            if adjacent_eff <= 0 or not chunk_key or not doc_indices:
                continue
            base_pos = pos_map.get(str(chunk_key))
            if base_pos is None:
                # chunk_id may include doc prefix; ensure we try raw chunk_id
                if isinstance(chunk_key, str) and "#" not in chunk_key:
                    base_pos = pos_map.get(f"{doc_id}#{chunk_key}")
            if base_pos is None:
                continue
            # derive score baseline for neighbors
            neighbor_score = float(extra.score) if extra.score is not None else 0.0
            for delta in range(1, adjacent_eff + 1):
                for nb_pos in (base_pos - delta, base_pos + delta):
                    if nb_pos < 0 or nb_pos >= len(doc_indices):
                        continue
                    neighbor = _neighbor_context(doc_id, nb_pos)
                    if neighbor is None:
                        continue
                    # decay neighbor scores to avoid swamping top hits
                    neighbor.meta["adjacent_offset"] = delta
                    neighbor.score = neighbor_score * (0.8 ** delta)
                    _mark_and_add(neighbor)
                    if len(out) >= max_total:
                        break
                if len(out) >= max_total:
                    break
            if len(out) >= max_total:
                break
        if len(out) >= max_total:
            break

    if len(out) < max_total:
        carry = max(1, max_total // 4)
        for ctx in htnr_contexts[:carry]:
            clone = _clone_context_item(ctx)
            _mark_and_add(clone)
            if len(out) >= max_total:
                break

    return out[:max_total]


# ---------- LRU Embedding Cache ----------
class LRUCache:
    def __init__(self, capacity: int = 20000):
        self.capacity = int(capacity)
        self.store: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self.store:
                val = self.store.pop(key)
                self.store[key] = val
                self.hits += 1
                return val.copy()
            self.misses += 1
            return None

    def put(self, key: str, val: np.ndarray) -> None:
        with self._lock:
            arr = np.asarray(val, dtype=np.float32).copy()
            arr.setflags(write=False)
            if key in self.store:
                self.store.pop(key)
            self.store[key] = arr
            if len(self.store) > self.capacity:
                self.store.popitem(last=False)


class SentCache(OrderedDict):
    def __init__(self, capacity: int = 5000):
        super().__init__()
        self.capacity = int(max(1, capacity))

    def get_or_none(self, key: str) -> Optional[np.ndarray]:
        val = super().get(key)
        if val is not None:
            self.move_to_end(key)
        return val

    def put(self, key: str, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32).copy()
        arr.setflags(write=False)
        if key in self:
            self.move_to_end(key)
        self[key] = arr
        if len(self) > self.capacity:
            self.popitem(last=False)

    def __getitem__(self, key: str) -> np.ndarray:  # type: ignore[override]
        val = super().__getitem__(key)
        self.move_to_end(key)
        return val


# ---------- RAG protocol ----------
@dataclass
class RAGResult:
    content: str
    score: float
    metadata: Dict[str, Any]


class ExternalRAGProtocol(Protocol):
    def search(self, query: str, top_k: int) -> List[RAGResult]: ...


class FunctionRAG(ExternalRAGProtocol):
    """Inject any function that returns [{'title','snippet','score','url'}...] or RAGResult list."""
    def __init__(self, fn):
        self.fn = fn

    def search(self, query: str, top_k: int) -> List[RAGResult]:
        try:
            items = self.fn(query, top_k)
            out = []
            for i, it in enumerate(items[:top_k]):
                if isinstance(it, RAGResult):
                    out.append(it)
                    continue
                title = it.get("title") if isinstance(it, dict) else str(it)
                snippet = it.get("snippet") if isinstance(it, dict) else str(it)
                score = float(it.get("score", 1.0 / (i + 1))) if isinstance(it, dict) else 1.0 / (i + 1)
                url = it.get("url") if isinstance(it, dict) else None
                out.append(RAGResult(
                    content=f"[{title}] {snippet}",
                    score=score,
                    metadata={"url": url, "type": "rag"}
                ))
            return out
        except Exception as e:
            logger.warning(f"External RAG error: {e}")
            return []


# ---------- Online change & AutoCal (EVT-lite + PI) ----------
class PageHinkley:
    def __init__(self, delta: float = 0.0, lam: float = 3.0):
        self.delta = float(delta)
        self.lam = float(lam)
        self.mean = 0.0
        self.gp = 0.0
        self.gn = 0.0
        self.stable_run = 0

    def update(self, x: float) -> bool:
        self.mean = 0.99 * self.mean + 0.01 * x
        y = x - self.mean - self.delta
        self.gp = max(0.0, self.gp + y)
        self.gn = min(0.0, self.gn + y)
        boundary = (self.gp > self.lam) or (abs(self.gn) > self.lam)
        if boundary:
            self.gp = self.gn = 0.0
            self.stable_run = 0
            return True
        self.stable_run += 1
        return False


class AutoCal:
    def __init__(self, numerics: Numerics):
        self.num = numerics
        self.tau_w: float = 0.20
        self.tau_R: float = math.sqrt((1.0 + self.tau_w) / 2.0)
        self.alpha_t: float = 0.90
        self.tail_buf: Deque[float] = deque(maxlen=8192)
        self.min_tail: int = 80
        self.ph_alpha = PageHinkley(delta=0.0, lam=4.0)

    def add_tail_samples(self, sims_flat: np.ndarray) -> None:
        sims = np.asarray(sims_flat, dtype=np.float32).ravel()
        sims = np.clip(sims, self.num.cos_lower, self.num.cos_upper)
        for v in sims.tolist():
            if np.isfinite(v):
                self.tail_buf.append(float(v))

    def _evt_tau(self) -> Optional[float]:
        n = len(self.tail_buf)
        if n < self.min_tail:
            return None
        q = 0.90 + 0.05 * (1.0 / (1.0 + math.exp(-0.01 * (n - 400))))  # 0.90→0.95
        arr = np.array(self.tail_buf, dtype=np.float32)
        return float(np.quantile(arr, q))

    def update_alpha(self, pe: float) -> float:
        _ = self.ph_alpha.update(max(0.0, pe))
        sr = max(5.0, float(self.ph_alpha.stable_run))
        self.alpha_t = float(np.clip(math.exp(-1.0 / sr), 0.60, 0.98))
        return self.alpha_t

    def update_thresholds(self, rl: float) -> Tuple[float, float]:
        hat_evt = self._evt_tau()
        hat_pi = float(np.clip(self.tau_w + 0.05 * (rl - 0.5), 0.10, 0.98))
        # Smooth mixing
        w_evt = 1.0 / (1.0 + math.exp(-0.01 * (len(self.tail_buf) - 400)))
        w_pi = 1.0 - 0.5 * w_evt
        Z = max(1e-6, w_evt + w_pi)
        w_evt /= Z
        w_pi /= Z
        blended = w_evt * (hat_evt if hat_evt is not None else self.tau_w) + w_pi * hat_pi
        # EMA + rate limit
        prev = self.tau_w
        tau_w_new = self.num.ema_gamma_slow * prev + (1 - self.num.ema_gamma_slow) * blended
        tau_w_new = float(np.clip(tau_w_new, prev - self.num.max_rate_tau_w, prev + self.num.max_rate_tau_w))
        tau_w_new = float(np.clip(tau_w_new, 0.10, 0.98))
        self.tau_w = tau_w_new
        prevR = self.tau_R
        target_R = math.sqrt((1.0 + self.tau_w) / 2.0)
        tau_R_new = self.num.ema_gamma_fast * prevR + (1 - self.num.ema_gamma_fast) * target_R
        tau_R_new = float(np.clip(tau_R_new, prevR - self.num.max_rate_tau_R, prevR + self.num.max_rate_tau_R))
        self.tau_R = tau_R_new
        return self.tau_w, self.tau_R


# ---------- Data model / Log ----------
@dataclass
class HTNRNode:
    id: int
    content: str
    emb: np.ndarray
    V: float
    D: float
    S: float
    R: float
    E: float
    Chi: float
    parent: Optional[int]
    children: List[int]
    birth_step: int
    chapter: int = 0
    source: str = "Unknown"


@dataclass
class BufferItem:
    content: str
    emb: np.ndarray
    E_init: float
    source: str
    chapter_hint: Optional[int]


class EventLog:
    def __init__(self, to_file: Optional[str] = None, buffer_size: int = 128,
                 truncate_len: int = 16384, enabled: bool = True):
        self.enabled = bool(enabled)
        self.to_file = to_file
        self.buffer_size = max(8, int(buffer_size))
        self.truncate_len = int(truncate_len)
        self._buf: List[str] = []
        self._lock = threading.RLock()
        self._fh: Optional[io.TextIOBase] = None
        if not self.enabled:
            return
        if to_file:
            try:
                parent = os.path.dirname(os.path.abspath(to_file))
                if parent and parent != ".":
                    os.makedirs(parent, exist_ok=True)
                self._fh = open(to_file, "a", encoding="utf-8")
            except Exception:
                logging.exception("EventLog open failed; logs disabled")
                self.enabled = False
                self._fh = None

    def write(self, rec: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = json.dumps(rec, ensure_ascii=False)
        if len(line) > self.truncate_len:
            line = line[: self.truncate_len] + "...(truncated)"
        do_flush = False
        with self._lock:
            self._buf.append(line)
            if len(self._buf) >= self.buffer_size:
                do_flush = True
        if do_flush:
            self.flush()

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._buf:
                return
            data = "\n".join(self._buf) + "\n"
            self._buf.clear()
            if self._fh:
                self._fh.write(data)
                self._fh.flush()

    def close(self) -> None:
        if not self.enabled:
            return
        self.flush()
        with self._lock:
            if self._fh:
                self._fh.close()
                self._fh = None



# ---------- Summarization (MMR) ----------
def mmr_select(query_vec: np.ndarray, sentences: List[str], encode_fn,
               *, topk: int = 2, lambda_mmr: float = 0.7, max_sentences: int = 64) -> List[str]:
    if not sentences:
        return []
    if len(sentences) > max_sentences:
        sentences = sorted(sentences, key=len, reverse=True)[:max_sentences]
    try:
        embs = encode_fn(sentences)
    except Exception:
        return sentences[:topk]
    embs = _normalize(np.asarray(embs, dtype=np.float32), 1e-8)
    q = _normalize(query_vec, 1e-8)
    sims = embs @ q
    selected: List[int] = []
    cand = list(range(len(sentences)))
    while cand and len(selected) < topk:
        rel = sims[cand]
        red = np.max(embs[cand] @ embs[selected].T, axis=1) if selected else np.zeros_like(rel)
        score = lambda_mmr * rel - (1 - lambda_mmr) * red
        j = int(cand[int(np.argmax(score))])
        selected.append(j)
        cand.remove(j)
    # Jaccard suppression
    out = []
    for s in [sentences[i] for i in selected]:
        if not out or jaccard(out[-1], s) < 0.7:
            out.append(s)
    return out[:topk]


# ---------- Main ----------
class HTNRMemoryV54P:
    def __init__(self, embedder: "EmbedderProtocol", *,
                 external_rag: Optional[ExternalRAGProtocol] = None,
                 hyper: Optional[Hyper] = None,
                 numerics: Optional[Numerics] = None,
                 log_path: Optional[str] = None,
                 seed: Optional[int] = 42):
        self.model = embedder
        self.dim = self.model.get_sentence_embedding_dimension()
        self.hyper = hyper or Hyper()
        self.num = numerics or Numerics()
        self.external_rag = external_rag

        self._sanity_check_hyper()

        # ---- 프로파일 기본값 적용 ----
        self._apply_profile_defaults()

        # Context (MoC)
        H = self.hyper.K_contexts
        self.M_heads = np.zeros((H, self.dim), dtype=np.float32)
        self.M_weights = np.ones(H, dtype=np.float32) / H
        self.M_agg = np.zeros(self.dim, dtype=np.float32)
        self.M_long = np.zeros(self.dim, dtype=np.float32)

        # State
        self.nodes: Dict[int, HTNRNode] = {}
        self.leaves: List[int] = []
        self.next_id = 0
        self.step = 0
        self.current_chapter = 0
        self.last_chapter_step = -10**9
        self.smoothed_lambda = 0.05

        # Cache
        self._leaf_mat: Optional[np.ndarray] = None
        self._leaf_ids: Optional[List[int]] = None

        # ANN
        self._ann_index = None
        self._ann_dirty = True

        # Buffer / AutoCal / Chapter-PH
        self.buffer: Deque[BufferItem] = deque(maxlen=self.hyper.buffer_size)
        self.auto = AutoCal(self.num)
        self.ph_chapter = PageHinkley(delta=0.0, lam=3.0)

        # Log / lock / rng
        self.log = EventLog(
    log_path,
    buffer_size=self.hyper.eventlog_buffer,
    truncate_len=self.hyper.eventlog_truncate,
    enabled=self.hyper.eventlog_enabled,
)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)

        # Whitening stats
        self._w_mu = np.zeros(4, dtype=np.float32)
        self._w_cov = np.eye(4, dtype=np.float32)
        self._w_n = 0

        # PhaseR cooldown
        self._phaseR_cooldown_until = -10**9

        # sentence embedding cache for MMR (optional; embedder already caches)
        self._sent_cache = SentCache(capacity=5000)

        # Torch / Device
        self._use_torch = bool(self.hyper.use_torch and _HAS_TORCH and not self.hyper.torch_force_cpu)
        if self._use_torch:
            if _TORCH_CUDA:
                self._device = torch.device("cuda")
                torch.set_float32_matmul_precision("high")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = None
        self._leaf_mat_t = None  # torch.Tensor | None
        self._M_agg_t = None     # torch.Tensor | None
        self._gpu_dirty = True

    # ----- Profile defaults -----
    def _apply_profile_defaults(self) -> None:
        p = (self.hyper.profile or "QA").lower()
        if p == "qa":
            self.hyper.query_blend = "auto"
            self.hyper.opportunistic_merge = False
            self.hyper.merge_cross_doc = False
            self.hyper.mmr_topk_parent = max(4, self.hyper.mmr_topk_parent)
        elif p == "streaming":
            if self.hyper.query_blend == "auto":
                self.hyper.query_blend_fixed = 0.20
            self.hyper.opportunistic_merge = True

        if self.hyper.batch_freeze_enable:
            self.hyper.buffer_size = max(self.hyper.buffer_size, self.hyper.batch_freeze_size)

    # ----- Cache & ANN -----
    def _refresh_leaf_cache(self) -> None:
        if not self.leaves:
            self._leaf_mat = None
            self._leaf_ids = None
            self._ann_dirty = True
            self._gpu_dirty = True
            return
        embs = np.stack([self.nodes[i].emb for i in self.leaves], axis=0)
        norms = np.linalg.norm(embs, axis=1)
        zero_rows = int(np.sum(norms <= self.num.eps_norm))
        if zero_rows > 0:
            self.log.write({"t": "warn", "zero_rows_in_cache": zero_rows})
        self._leaf_mat = _normalize(embs, self.num.eps_norm)
        self._leaf_ids = list(self.leaves)
        self._ann_dirty = True
        self._gpu_dirty = True

    def _ann_mark_dirty_for_ids(self, added: List[int], removed: List[int]) -> None:
        if not added and not removed:
            return
        if self.hyper.ann_backend in ("none", "exact"):
            self._ann_dirty = True
            self._ann_index = None
            self._gpu_dirty = True
            return
        if not _HAS_HNSW or len(self.leaves) < self.hyper.ann_min_leaves:
            self._ann_dirty = True
            self._gpu_dirty = True
            return
        with self._lock:
            if self._ann_index is None:
                self._ann_dirty = True
                self._gpu_dirty = True
                return
            try:
                for nid in removed:
                    try:
                        self._ann_index.mark_deleted(int(nid))
                    except Exception:
                        self._ann_dirty = True
                        self._gpu_dirty = True
                        return
                if added:
                    labels = np.array(added, dtype=np.int64)
                    embs = np.stack([self.nodes[n].emb for n in added], axis=0)
                    embs = _normalize(embs, self.num.eps_norm)
                    self._ann_index.add_items(embs, labels)
                self._ann_dirty = False
                self._gpu_dirty = True
            except Exception:
                self._ann_dirty = True
                self._gpu_dirty = True

    def _ensure_ann_index(self) -> None:
        if self.hyper.ann_backend == "hnsw" and not _HAS_HNSW:
            raise RuntimeError("ann_backend='hnsw'인데 hnswlib가 설치되어 있지 않습니다.")
        if self.hyper.ann_backend in ("none", "exact"):
            self._ann_index = None
            return
        if self.hyper.ann_backend == "auto" and not _HAS_HNSW:
            self._ann_index = None
            return
        if len(self.leaves) < self.hyper.ann_min_leaves:
            self._ann_index = None
            return
        if self._leaf_mat is None or self._leaf_ids is None:
            return
        if (self._ann_index is None) or self._ann_dirty:
            with self._lock:
                if (self._ann_index is None) or self._ann_dirty:
                    if len(self.leaves) < self.hyper.ann_min_leaves:
                        self._ann_index = None
                        self._ann_dirty = False
                        return
                    if self._leaf_mat is None or self._leaf_ids is None:
                        return
                    try:
                        idx = _hnswlib.Index(space="cosine", dim=self.dim)
                        idx.init_index(max_elements=len(self.leaves),
                                       ef_construction=self.hyper.ann_efC, M=self.hyper.ann_M)
                        labels = np.array(self._leaf_ids, dtype=np.int64)
                        idx.add_items(self._leaf_mat, labels)
                        idx.set_ef(self.hyper.ann_efS)
                        self._ann_index = idx
                        self._ann_dirty = False
                        self.log.write({"t": "ann_build", "L": len(self.leaves)})
                    except Exception as e:
                        logger.warning(f"HNSW build failed: {e}")
                        self._ann_index = None

    # ----- GPU sync -----
    def _sync_gpu_buffers(self) -> None:
        if not self._use_torch:
            return
        if self._gpu_dirty:
            if self._leaf_mat is not None and self._leaf_mat.size:
                t = torch.from_numpy(np.ascontiguousarray(self._leaf_mat))
                if self._device.type == "cuda":
                    t = t.pin_memory()
                self._leaf_mat_t = t.to(self._device, non_blocking=(self._device.type == "cuda"))
            else:
                self._leaf_mat_t = None
            if self.M_agg is not None and self.M_agg.size:
                mt = torch.from_numpy(np.ascontiguousarray(self.M_agg))
                if self._device.type == "cuda":
                    mt = mt.pin_memory()
                self._M_agg_t = mt.to(self._device, non_blocking=(self._device.type == "cuda"))
            else:
                self._M_agg_t = None
            self._gpu_dirty = False

    # ----- Context update (sequential) -----
    def _update_contexts(self, emb: np.ndarray) -> Tuple[float, float]:
        sims = self.M_heads @ emb
        temp = 1.0 + 5.0 * self.smoothed_lambda
        ex = np.exp(sims / max(1e-6, temp))
        gates = ex / (np.sum(ex) + 1e-8)
        PE = 1.0 - _cos(emb, self.M_agg, self.num.eps_norm)
        alpha = self.auto.update_alpha(PE)
        self.M_heads = alpha * self.M_heads + (1 - alpha) * (gates[:, None] * emb)
        self.M_heads = _normalize(self.M_heads, self.num.eps_norm)
        prev = self.M_agg.copy()
        self.M_weights = 0.95 * self.M_weights + 0.05 * gates
        self.M_weights /= (np.sum(self.M_weights) + 1e-8)
        self.M_agg = _normalize(np.sum(self.M_heads * self.M_weights[:, None], axis=0), self.num.eps_norm)
        self.M_long = _normalize(0.99 * self.M_long + 0.01 * self.M_agg, self.num.eps_norm)
        D_birth = 1.0 - _cos(self.M_agg, prev, self.num.eps_norm)
        lam = max(0.005, min(0.5 * D_birth, 0.3))
        self.smoothed_lambda = 0.1 * lam + 0.9 * self.smoothed_lambda
        self._gpu_dirty = True
        return PE, D_birth

    # ----- Batch‑Freeze ingest (NEW) -----
    def _ingest_batch_freeze(self, batch: List[BufferItem]) -> None:
        if not batch:
            return

        added_ids: List[int] = []

        if self.step == 0 and float(np.linalg.norm(self.M_agg)) <= self.num.eps_norm:
            first = batch[0]
            self.M_heads[:] = first.emb
            self.M_agg = first.emb.copy()
            self.M_long = first.emb.copy()
            nid0 = self._add_node_from_item(first, PE=0.0, D_birth=0.0, chapter_hint=first.chapter_hint)
            added_ids.append(nid0)
            batch = batch[1:]
            if not batch:
                self._refresh_leaf_cache()
                self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
                self._maintenance()
                return

        H = self.M_heads.shape[0]
        M_heads_0 = self.M_heads.copy()
        M_weights_0 = self.M_weights.copy()
        M_agg_0 = self.M_agg.copy()
        temp = 1.0 + 5.0 * self.smoothed_lambda

        E = np.stack([it.emb for it in batch], axis=0)
        norms = np.linalg.norm(E, axis=1) * (np.linalg.norm(M_agg_0) + self.num.eps_norm) + self.num.eps_norm
        PE_vec = (1.0 - (E @ M_agg_0) / norms).astype(np.float32)
        PE_mean = float(np.mean(PE_vec)) if PE_vec.size else 0.0

        sims = (M_heads_0 @ E.T) / max(1e-6, temp)
        ex = np.exp(sims - np.max(sims, axis=0, keepdims=True))
        gates = ex / (np.sum(ex, axis=0, keepdims=True) + 1e-8)

        num = gates @ E
        den = np.sum(gates, axis=1, keepdims=True) + 1e-8
        avg_emb = num / den

        alpha_t = self.auto.update_alpha(PE_mean)
        M_heads_1 = alpha_t * M_heads_0 + (1 - alpha_t) * avg_emb
        M_heads_1 = _normalize(M_heads_1, self.num.eps_norm)

        gate_mean = np.mean(gates, axis=1).astype(np.float32)
        M_weights_1 = 0.95 * M_weights_0 + 0.05 * gate_mean
        M_weights_1 = M_weights_1 / (np.sum(M_weights_1) + 1e-8)

        M_agg_1 = _normalize(np.sum(M_heads_1 * M_weights_1[:, None], axis=0), self.num.eps_norm)
        self.M_long = _normalize(0.99 * self.M_long + 0.01 * M_agg_1, self.num.eps_norm)

        D_birth_batch = 1.0 - _cos(M_agg_1, M_agg_0, self.num.eps_norm)
        lam = max(0.005, min(0.5 * D_birth_batch, 0.3))
        self.smoothed_lambda = 0.1 * lam + 0.9 * self.smoothed_lambda

        self.M_heads = M_heads_1
        self.M_weights = M_weights_1
        self.M_agg = M_agg_1
        self._gpu_dirty = True

        for it, pe in zip(batch, PE_vec.tolist()):
            nid = self._add_node_from_item(it, PE=pe, D_birth=D_birth_batch, chapter_hint=it.chapter_hint)
            added_ids.append(nid)

        self._refresh_leaf_cache()
        self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
        self._maintenance()

    def _add_node_from_item(self, item: BufferItem, *, PE: float, D_birth: float, chapter_hint: Optional[int]) -> int:
        # 챕터 경계 감지
        sig = 0.4 * D_birth + 0.4 * PE + 0.2 * (1.0 - _cos(item.emb, self.M_long, self.num.eps_norm))
        boundary = self.ph_chapter.update(sig)
        if boundary and (self.step - self.last_chapter_step >= self.hyper.min_chapter_gap) and (self.ph_chapter.stable_run == 0):
            self.current_chapter += 1
            self.last_chapter_step = self.step

        node_ch = self.current_chapter if chapter_hint is None else int(chapter_hint)
        R_init = _cos(item.emb, self.M_agg, self.num.eps_norm)  # 주: Batch‑Freeze에서는 직전 배치 커밋 이후의 M_agg
        V_init = float(np.clip(R_init + item.E_init + 0.7 * PE, 0.0, 2.0))

        nid = self.next_id
        self.next_id += 1
        node = HTNRNode(
            id=nid, content=item.content, emb=item.emb, V=V_init, D=D_birth, S=1.0, R=R_init,
            E=item.E_init, Chi=0.0, parent=None, children=[], birth_step=self.step,
            chapter=node_ch, source=item.source
        )
        self.nodes[nid] = node
        self.leaves.append(nid)
        self.step += 1
        self.log.write({"t": "add", "id": nid, "V": V_init, "R": R_init, "E": item.E_init, "ch": node_ch})
        return nid

    # ----- Add / Buffer -----
    def process_and_add(self, text: str, *, source: str = "Unknown",
                        chapter: Optional[int] = None, salience_boost: float = 0.0) -> None:
        if not text:
            return
        vec = self.model.encode([text])[0]
        vec = _normalize(vec, self.num.eps_norm)
        item = BufferItem(text, vec, max(0.0, float(salience_boost)), source, chapter)
        with self._lock:
            self.buffer.append(item)
            self._replay_if_ready()

    def flush_buffer(self) -> None:
        with self._lock:
            self._replay(force=True)

    def _replay_if_ready(self) -> None:
        if self.hyper.batch_freeze_enable:
            if len(self.buffer) >= min(self.buffer.maxlen, self.hyper.batch_freeze_size):
                self._replay(force=False)
        else:
            if len(self.buffer) >= self.buffer.maxlen or self.auto.ph_alpha.stable_run > 20:
                self._replay(force=False)

    def _replay(self, *, force: bool) -> None:
        if not self.buffer:
            return

        processed = 0
        added_ids: List[int] = []
        max_per_tick = max(4, self.hyper.buffer_size)
        did_batch = False

        if self.hyper.batch_freeze_enable:
            # 배치 단위 처리
            while self.buffer and (processed < max_per_tick or force):
                B = min(len(self.buffer), self.hyper.batch_freeze_size)
                batch = [self.buffer.popleft() for _ in range(B)]
                self._ingest_batch_freeze(batch)
                processed += B
                did_batch = True
        else:
            # 기존(순차) 처리 경로
            while self.buffer and processed < max_per_tick:
                item = self.buffer.popleft()
                if self.step == 0 and float(np.linalg.norm(self.M_agg)) <= self.num.eps_norm:
                    self.M_heads[:] = item.emb
                    self.M_agg = item.emb.copy()
                    self.M_long = item.emb.copy()
                    PE, D_birth = 0.0, 0.0
                else:
                    PE, D_birth = self._update_contexts(item.emb)

                # 노드화
                nid = self._add_node_from_item(item, PE=PE, D_birth=D_birth, chapter_hint=item.chapter_hint)
                added_ids.append(nid)
                processed += 1

        if did_batch:
            return

        self._refresh_leaf_cache()
        self._ann_mark_dirty_for_ids(added=added_ids, removed=[])
        self._maintenance()

    # ----- Maintenance -----
    def _maintenance(self) -> None:
        if not self.leaves:
            return

        # Homeostasis (IQR + logistic)
        if self.step % self.hyper.homeo_interval == 0:
            V_int = np.array([max(0.0, self.nodes[i].V - self.nodes[i].E) for i in self.leaves], dtype=np.float32)
            if V_int.size >= 4:
                q1, q3 = np.percentile(V_int, [25, 75])
                iqr = max(1e-6, q3 - q1)
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                W = np.clip(V_int, lo, hi)
                mu, sd = float(np.mean(W)), float(np.std(W) + 1e-6)
                z = (W - mu) / sd
                s = 2.0 * (1.0 / (1.0 + np.exp(-z)))
                for idx, nid in enumerate(self.leaves):
                    self.nodes[nid].V = min(2.0, float(s[idx] + self.nodes[nid].E))

        # Interference (local top‑K; density = count+avg-sim)
        if self._leaf_mat is None:
            self._refresh_leaf_cache()
        L = len(self.leaves)
        if L >= 8 and self._leaf_mat is not None:
            use_torch_similarity = False
            S_t = None
            if self._use_torch and L <= self.hyper.gpu_max_pairwise:
                self._sync_gpu_buffers()
                if self._leaf_mat_t is not None:
                    S_t = torch.matmul(self._leaf_mat_t, self._leaf_mat_t.t())
                    use_torch_similarity = True
            if use_torch_similarity and S_t is not None:
                tri = torch.triu_indices(L, L, 1, device=S_t.device)
                tail_vals = S_t[tri[0], tri[1]]
                q = float(torch.quantile(tail_vals, self.hyper.interference_q).item())
                tail_base = tail_vals.detach().cpu().numpy().astype(np.float32)
                K = max(1, int(round(math.log(L + 1))))
                topk_vals, topk_idx = torch.topk(S_t, k=min(K + 1, L), dim=1)
                for r in range(L):
                    strong_vals = []
                    for val, idx in zip(topk_vals[r].tolist(), topk_idx[r].tolist()):
                        if idx == r:
                            continue
                        if val >= q:
                            strong_vals.append(float(val))
                    avg = float(np.mean(strong_vals)) if strong_vals else 0.0
                    density = 0.5 * math.log(1.0 + len(strong_vals)) + 0.5 * avg
                    nid = self.leaves[r]
                    protection = math.tanh(self.nodes[nid].E)
                    dec = max(0.0, 0.05 * density - 0.01 * protection)
                    if dec > 0:
                        self.nodes[nid].V = max(0.0, self.nodes[nid].V - dec)
                self.auto.add_tail_samples(tail_base)
            else:
                S = (self._leaf_mat @ self._leaf_mat.T).astype(np.float32)
                iu = np.triu_indices(L, k=1)
                tail_base = S[iu]
                q = float(np.quantile(tail_base, self.hyper.interference_q))
                K = max(1, int(round(math.log(L + 1))))
                for r in range(L):
                    row = S[r]
                    idx = _topk_indices(row, min(K + 1, L))
                    nbrs = [j for j in idx if j != r]
                    strong_vals = [row[j] for j in nbrs if row[j] >= q]
                    avg = float(np.mean(strong_vals)) if strong_vals else 0.0
                    density = 0.5 * math.log(1.0 + len(strong_vals)) + 0.5 * avg
                    nid = self.leaves[r]
                    protection = math.tanh(self.nodes[nid].E)
                    dec = max(0.0, 0.05 * density - 0.01 * protection)
                    if dec > 0:
                        self.nodes[nid].V = max(0.0, self.nodes[nid].V - dec)
                self.auto.add_tail_samples(tail_base)

        # Thresholds
        occupancy = float(len(self.leaves) / max(1, self.hyper.max_leaves))
        self.auto.update_thresholds(rl=np.clip(occupancy, 0.0, 2.0))

        # Over-capacity merges
        merges = 0
        while len(self.leaves) > self.hyper.max_leaves and merges < self.hyper.max_merges_per_cycle:
            if not self._merge_once():
                break
            merges += 1

        # Phase‑R
        self._phase_reorg()

        # Opportunistic merge (옵션)
        if self.hyper.opportunistic_merge and len(self.leaves) >= self.hyper.anchor_recent + 2:
            self._merge_once()

        # Metrics
        parents_in_leaves = sum(1 for nid in self.leaves if self.nodes[nid].children)
        self.log.write({"t": "metrics", "leaves": len(self.leaves),
                        "parents_in_leaves": parents_in_leaves,
                        "tau_w": self.auto.tau_w, "tau_R": self.auto.tau_R})

    def _sanity_check_hyper(self) -> None:
        h = self.hyper
        def _ensure(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)
        _ensure(1 <= h.K_contexts <= 8, "hyper.K_contexts must be within [1,8]")
        _ensure(8 <= h.buffer_size <= 4096, "hyper.buffer_size must be within [8,4096]")
        _ensure(16 <= h.max_leaves <= 4096, "hyper.max_leaves must be within [16,4096]")
        _ensure(0.0 <= h.shrink_lambda <= 0.5, "hyper.shrink_lambda must be within [0.0,0.5]")
        _ensure(h.ann_backend in {"auto", "hnsw", "exact", "none"}, "hyper.ann_backend invalid")
        _ensure(0.0 < h.rag_entropy_tau <= 5.0, "hyper.rag_entropy_tau must be within (0,5]")

    # ----- Merge -----
    def _dendritic_gating(self, na: HTNRNode, nb: HTNRNode) -> np.ndarray:
        Ri = _cos(na.emb, self.M_agg, self.num.eps_norm)
        Rj = _cos(nb.emb, self.M_agg, self.num.eps_norm)
        diff = 3.0 * (Ri - Rj)
        gate_i = 1.0 / (1.0 + math.exp(-diff))
        gate_j = 1.0 - gate_i
        gated = gate_i * na.emb + gate_j * nb.emb
        s = max(1e-6, na.V + nb.V)
        vw = (na.V / s) * na.emb + (nb.V / s) * nb.emb
        new = 0.5 * gated + 0.5 * vw
        new = (1 - self.hyper.shrink_lambda) * new + self.hyper.shrink_lambda * self.M_agg
        return _normalize(new, self.num.eps_norm)

    def _encode_cached_sentences(self, sents: List[str]) -> np.ndarray:
        miss = [s for s in sents if self._sent_cache.get_or_none(s) is None]
        if miss:
            embs = self.model.encode(miss)
            for s, e in zip(miss, embs):
                self._sent_cache.put(s, e)
        return np.stack([self._sent_cache[s] for s in sents], axis=0)

    def _try_build_parent(self, ida: int, idb: int) -> bool:
        na, nb = self.nodes[ida], self.nodes[idb]

        def _extract_doc_id(source: Any) -> Optional[str]:
            if isinstance(source, str) and source.startswith("doc:"):
                return source.split(":", 1)[1]
            return None

        doc_a = _extract_doc_id(na.source)
        doc_b = _extract_doc_id(nb.source)
        if not self.hyper.merge_cross_doc and doc_a and doc_b and doc_a != doc_b:
            return False

        merged_emb = self._dendritic_gating(na, nb)
        R_align = _cos(merged_emb, self.M_agg, self.num.eps_norm)
        if R_align < self.auto.tau_R:
            return False

        sents = sent_tokenize((na.content or "") + "\n" + (nb.content or ""))
        if self.hyper.mmr_topk_parent_dynamic:
            tk = int(np.interp(len(sents), [0, 8, 20, 50], [2, 3, 5, 8]))
            tk = max(2, min(tk, max(2, self.hyper.mmr_topk_parent)))
        else:
            tk = max(2, self.hyper.mmr_topk_parent)
        lines = mmr_select(merged_emb, sents, self._encode_cached_sentences,
                           topk=tk, lambda_mmr=0.7, max_sentences=64)
        parent_content = " / ".join(lines) if lines else ((na.content or "")[:80] + " / " + (nb.content or "")[:80])

        new_E = 0.5 * (na.E + nb.E)
        new_V = min(2.0, 0.35 * (na.V + nb.V) + 0.65 * (R_align + new_E))
        new_D = max(na.D, nb.D)
        new_S = 1.0 + math.log1p(len(na.children) + len(nb.children))
        new_Chi = 0.5 * (na.Chi + nb.Chi) + 0.5 * R_align

        pid = self.next_id
        self.next_id += 1
        if doc_a and doc_b and doc_a == doc_b:
            parent_source = f"doc:{doc_a}"
        elif doc_a:
            parent_source = f"doc:{doc_a}"
        elif doc_b:
            parent_source = f"doc:{doc_b}"
        else:
            parent_source = "Merge"

        node = HTNRNode(
            id=pid, content=parent_content, emb=merged_emb, V=new_V, D=new_D, S=new_S, R=R_align,
            E=new_E, Chi=new_Chi, parent=None, children=[ida, idb], birth_step=self.step,
            chapter=max(na.chapter, nb.chapter), source=parent_source
        )
        self.nodes[pid] = node
        self.nodes[ida].parent = pid
        self.nodes[idb].parent = pid
        removed = []
        if ida in self.leaves:
            self.leaves.remove(ida)
            removed.append(ida)
        if idb in self.leaves:
            self.leaves.remove(idb)
            removed.append(idb)
        self.leaves.append(pid)
        self._ann_mark_dirty_for_ids(added=[pid], removed=removed)
        self.log.write({"t": "merge", "p": pid, "a": ida, "b": idb, "R": R_align, "V": new_V})
        return True

    def _merge_once(self) -> bool:
        cand_ids = self._eligible_leaves()
        if len(cand_ids) < 2 or self._leaf_mat is None:
            return False

        self._ensure_ann_index()

        scored = [(max(0.0, self.nodes[i].V) * max(0.0, self.nodes[i].R), i) for i in cand_ids]
        scored.sort(key=lambda z: z[0], reverse=True)
        cand_ids = [i for _, i in scored[: min(self.hyper.cand_prefilter, len(scored))]]
        cand_set = set(cand_ids)

        pairs: List[Tuple[float, int, int]] = []
        used_set = set()
        if self._ann_index is not None:
            for a in cand_ids:
                try:
                    labels, dists = self._ann_index.knn_query(self.nodes[a].emb, k=self.hyper.knn_topk_pairs + 8)
                    for lab, dist in zip(labels[0], dists[0]):
                        b = int(lab)
                        if a == b or b not in cand_set:
                            continue
                        key = (min(a, b), max(a, b))
                        if key in used_set:
                            continue
                        w = float(1.0 - float(dist))  # cosine → similarity
                        if w >= self.auto.tau_w:
                            pairs.append((w, a, b))
                            used_set.add(key)
                except Exception:
                    pairs = []
                    break

        if not pairs:
            idx_map = {nid: k for k, nid in enumerate(self.leaves)}
            rows = [idx_map[i] for i in cand_ids]

            # GPU 가속 가능한 경우
            if self._use_torch and len(rows) <= self.hyper.gpu_max_pairwise:
                self._sync_gpu_buffers()
                if self._leaf_mat_t is not None:
                    sub_t = self._leaf_mat_t[rows, :]
                    S_t = torch.matmul(sub_t, sub_t.t())
                    S = S_t.detach().cpu().numpy().astype(np.float32)
                else:
                    sub = self._leaf_mat[rows]
                    S = (sub @ sub.T).astype(np.float32)
            else:
                sub = self._leaf_mat[rows]
                S = (sub @ sub.T).astype(np.float32)

            m = S.shape[0]
            k = min(self.hyper.knn_topk_pairs + 1, m)
            for i in range(m):
                idx = _topk_indices(S[i], k)
                for j in idx:
                    if j == i:
                        continue
                    a, b = cand_ids[min(i, j)], cand_ids[max(i, j)]
                    if a == b:
                        continue
                    key = (a, b)
                    if key in used_set:
                        continue
                    w = float(S[i, j])
                    if w >= self.auto.tau_w:
                        pairs.append((w, a, b))
                        used_set.add(key)

        if not pairs:
            return False

        # Whitening features + L2 denominator
        feats = []
        for _, a, b in pairs:
            na, nb = self.nodes[a], self.nodes[b]
            feats.append([na.S + nb.S, na.V + nb.V, na.D + nb.D, na.E + nb.E])
        feats = np.array(feats, dtype=np.float32)
        W, bvec = self._update_whitening(feats)
        z = (feats - bvec) @ W.T
        den = 1e-3 + np.linalg.norm(z, axis=1)
        w_sig = 1.0 / (1.0 + np.exp(-3.0 * np.array([p[0] for p in pairs], dtype=np.float32)))
        scores = w_sig / den
        order = np.argsort(-scores)

        merged = False
        used_leaf: set[int] = set()
        took = 0
        max_do = self.hyper.max_merges_per_cycle
        for idx in order:
            _, a, b = pairs[idx]
            if a in used_leaf or b in used_leaf:
                continue
            if self._try_build_parent(a, b):
                used_leaf.add(a)
                used_leaf.add(b)
                merged = True
                took += 1
                if took >= max_do:
                    break

        if merged:
            self._refresh_leaf_cache()
            self._ensure_ann_index()
        return merged

    def _eligible_leaves(self) -> List[int]:
        L = len(self.leaves)
        if L <= self.hyper.anchor_recent:
            return list(self.leaves)[:-1] if L > 1 else list(self.leaves)
        return self.leaves[:-self.hyper.anchor_recent]

    def _update_whitening(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = feats.astype(np.float32)
        mu = np.mean(x, axis=0)
        xc = x - mu
        cov = (xc.T @ xc) / max(1.0, x.shape[0] - 1)
        if self._w_n < 50:
            self._w_mu = mu
            self._w_cov = cov + self.num.eps_cov * np.eye(4, dtype=np.float32)
            self._w_n += x.shape[0]
        else:
            g = 0.8
            self._w_mu = g * self._w_mu + (1 - g) * mu
            self._w_cov = g * self._w_cov + (1 - g) * cov
            self._w_n += x.shape[0]
        if self._w_n < 200:
            d = np.sqrt(np.clip(np.diag(self._w_cov), self.num.eps_cov, None))
            W = np.diag(1.0 / d)
            b = self._w_mu
            return W, b
        evals, evecs = np.linalg.eigh(self._w_cov + self.num.eps_cov * np.eye(4, dtype=np.float32))
        inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(evals, 1e-5, None)))
        W = evecs @ inv_sqrt @ evecs.T
        b = self._w_mu
        return W, b

    # ----- Phase‑R -----
    def _phase_reorg(self) -> None:
        if not self.leaves:
            return
        if self.step < self._phaseR_cooldown_until:
            return
        dissolved = 0
        budget = self.hyper.max_phaseR_per_cycle
        removed_ids: List[int] = []
        added_ids: List[int] = []

        for nid in list(self.leaves):
            if dissolved >= budget:
                break
            node = self.nodes.get(nid)
            if not node or not node.children:
                continue
            child_scores = [_cos(self.nodes[c].emb, self.M_agg, self.num.eps_norm) for c in node.children]
            if not child_scores:
                continue
            best_child = max(child_scores)
            var = float(np.var(child_scores))
            margin = 0.05 + 0.10 * min(1.0, math.sqrt(var))
            if (node.R + 1e-6) < (best_child - margin) and (self.step - node.birth_step) > 20:
                for c in node.children:
                    self.nodes[c].parent = None
                    self.nodes[c].V = max(0.0, self.nodes[c].V * 0.97)
                    if c not in self.leaves:
                        self.leaves.append(c)
                        added_ids.append(c)
                if nid in self.leaves:
                    self.leaves.remove(nid)
                removed_ids.append(nid)
                self.nodes.pop(nid, None)
                dissolved += 1

        if dissolved > 0:
            self._phaseR_cooldown_until = self.step + 120
            self._ann_mark_dirty_for_ids(added=added_ids, removed=removed_ids)
            self.log.write({"t": "phaseR", "count": dissolved})
            while len(self.leaves) > self.hyper.max_leaves and self._merge_once():
                pass
            self._refresh_leaf_cache()
            self._ensure_ann_index()

    # ----- Retrieval (+ RAG) -----
    def _uncertainty(self, scores: np.ndarray, k: int) -> Tuple[float, float]:
        if scores.size == 0:
            return float("inf"), 0.0
        idx = _topk_indices(scores, min(k, scores.size))
        s = scores[idx]
        tau = self.hyper.rag_entropy_tau
        ex = np.exp(s / max(1e-6, tau))
        Z = float(np.sum(ex))
        if not np.isfinite(Z) or Z <= 1e-12:
            return 0.0, float(np.max(s) if s.size else 0.0)
        p = ex / (Z + 1e-8)
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        return ent, float(np.max(s))

    def retrieve_for_query(self, query: str, *, K_cap: int = 8, mutate: bool = False,
                           flush: bool = False, return_meta: bool = False, use_rag: bool = True) -> List[Any]:
        if flush:
            self.flush_buffer()

        with self._lock:
            leaf_mat = None if self._leaf_mat is None else self._leaf_mat.copy()
            leaf_ids = None if self._leaf_ids is None else list(self._leaf_ids)
            M_agg = self.M_agg.copy()

        if leaf_mat is None or not leaf_ids:
            return self._do_rag(query, K_cap, return_meta) if (use_rag and self.external_rag) else []

        # --- 쿼리 임베딩 ---
        q = self.model.encode([query])[0]
        q = _normalize(q, self.num.eps_norm)

        # --- 적응형 블렌딩 ---
        comp = q
        mode = (self.hyper.query_blend or "none").lower()
        if mode != "none":
            cos_qm = _cos(q, M_agg, self.num.eps_norm)
            if mode == "fixed":
                w = float(np.clip(self.hyper.query_blend_fixed, 0.0, 0.95))
            else:  # "auto"
                lo, hi = self.hyper.query_blend_cos_low, self.hyper.query_blend_cos_high
                t = (cos_qm - lo) / max(1e-6, (hi - lo))
                w = float(np.clip(t, 0.0, 1.0)) * float(np.clip(self.hyper.query_blend_fixed, 0.0, 0.95))
            comp = _normalize((1.0 - w) * q + w * M_agg, self.num.eps_norm)

        # --- 스코어 계산 (GPU 우선) ---
        if self._use_torch:
            self._sync_gpu_buffers()
            if self._leaf_mat_t is not None:
                comp_t = torch.from_numpy(np.ascontiguousarray(comp))
                if self._device.type == "cuda":
                    comp_t = comp_t.pin_memory()
                comp_t = comp_t.to(self._device, non_blocking=(self._device.type == "cuda"))
                scores_t = torch.matmul(self._leaf_mat_t, comp_t)
                scores = scores_t.detach().cpu().numpy().astype(np.float32)
            else:
                scores = (leaf_mat @ comp).astype(np.float32)
        else:
            scores = (leaf_mat @ comp).astype(np.float32)

        chosen: List[Tuple[int, float]] = []
        if len(leaf_ids) >= 800:
            ch_max: Dict[int, float] = {}
            for idx_leaf, nid in enumerate(leaf_ids):
                node = self.nodes.get(nid)
                if not node:
                    continue
                chapter_id = node.chapter if node.chapter is not None else -1
                val = float(scores[idx_leaf])
                if (chapter_id not in ch_max) or (val > ch_max[chapter_id]):
                    ch_max[chapter_id] = val
            top_chapters = {ch for ch, _ in sorted(ch_max.items(), key=lambda z: -z[1])[:3]}
            mask_idx = [i for i, nid in enumerate(leaf_ids)
                        if (self.nodes.get(nid) and (self.nodes[nid].chapter if self.nodes[nid].chapter is not None else -1) in top_chapters)]
            if mask_idx:
                scores_sub = scores[mask_idx]
                leaf_ids_sub = [leaf_ids[i] for i in mask_idx]
                k = min(K_cap, len(scores_sub))
                idx_sel = _topk_indices(scores_sub, k)
                chosen = [(leaf_ids_sub[ii], float(scores_sub[ii])) for ii in idx_sel]

        if not chosen:
            k = min(K_cap, len(scores))
            idx = _topk_indices(scores, k)
            chosen = [(leaf_ids[ii], float(scores[ii])) for ii in idx]

        outs = []
        query_lower = (query or "").lower()
        needs_detail = any(token in query_lower for token in ("why", "how", "explain", "because", "reason"))
        for nid, base_score in chosen:
            node = self.nodes.get(nid)
            if not node:
                continue
            text = node.content or ""
            sents = sent_tokenize(text)
            if len(sents) > 4:
                try:
                    # 정답 스팬 유실 방지를 위해 요약된 문장 수를 늘립니다.
                    topk_sents = 6 if needs_detail else 4
                    selected = mmr_select(
                        comp,
                        sents,
                        self._encode_cached_sentences,
                        topk=topk_sents,
                        lambda_mmr=0.75,
                        max_sentences=80,
                    )
                    if selected:
                        text = " ".join(selected)
                except Exception:
                    pass
            vr = max(-1.0, min(1.0, node.V * node.R / 2.0))
            final_score = base_score * (1.0 + 0.25 * vr)
            raw_source = node.source if isinstance(node.source, str) else ""
            doc_id: Optional[str] = None
            chunk_anchor: Optional[str] = None
            if raw_source.startswith("doc:"):
                tail = raw_source.split(":", 1)[1]
                if tail:
                    chunk_anchor = tail
                    doc_id = tail.split("#", 1)[0]
            meta = {
                "node_id": nid,
                "tier": "L1",
                "chapter": node.chapter,
            }
            if doc_id:
                meta["doc_id"] = doc_id
            if chunk_anchor:
                meta["chunk_id"] = chunk_anchor
            outs.append({
                "content": text,
                "score": final_score,
                "node_id": nid,
                "chapter": node.chapter,
                "source": node.source,
                "tier": "L1",
                "doc_id": doc_id,
                "chunk_id": chunk_anchor,
                "meta": meta,
            })

        outs.sort(key=lambda x: x["score"], reverse=True)

        # RAG fallback
        if use_rag and self.external_rag:
            ent, top1 = self._uncertainty(scores, k=self.hyper.rag_topk)
            if (top1 < self.hyper.rag_top1_thr) or (ent > self.hyper.rag_entropy_thr):
                rag = self._do_rag(query, K_cap, return_meta=True)
                seen = set()
                final = []
                for r in outs + rag:
                    h = _stable_hash(r["content"][:200])
                    if h in seen:
                        continue
                    seen.add(h)
                    final.append(r)
                    if len(final) >= K_cap:
                        break
                if mutate:
                    with self._lock:
                        for r in outs:
                            nid = r["node_id"]
                            if nid in self.nodes:
                                self.nodes[nid].V = min(2.0, self.nodes[nid].V + 0.05 * float(r["score"]))
                return final if return_meta else [r["content"] for r in final]

        if mutate:
            with self._lock:
                for r in outs:
                    nid = r["node_id"]
                    if nid in self.nodes:
                        self.nodes[nid].V = min(2.0, self.nodes[nid].V + 0.05 * float(r["score"]))
        return outs if return_meta else [r["content"] for r in outs]

    def _do_rag(self, query: str, K_cap: int, return_meta: bool) -> List[Any]:
        try:
            rag_res = self.external_rag.search(query, top_k=min(self.hyper.rag_topk, K_cap))
        except Exception as e:
            logger.warning(f"RAG error: {e}")
            rag_res = []
        rag = [{
            "content": r.content, "score": float(r.score),
            "node_id": -1, "chapter": -1, "source": r.metadata.get("source", "RAG"),
            "url": r.metadata.get("url"), "tier": "L2"
        } for r in rag_res]
        return rag if return_meta else [r["content"] for r in rag]

    # ----- Snapshot -----
    def snapshot_save(self, path: str) -> None:
        with self._lock:
            data: Dict[str, Any] = {
                "version": "v5.4p+turbo",
                "dim": self.dim, "step": self.step, "next_id": self.next_id,
                "ctx": {
                    "M_heads": self.M_heads.tolist(), "M_weights": self.M_weights.tolist(),
                    "M_agg": self.M_agg.tolist(), "M_long": self.M_long.tolist(),
                    "smoothed_lambda": self.smoothed_lambda,
                    "current_chapter": self.current_chapter, "last_chapter_step": self.last_chapter_step,
                },
                "auto": {
                    "tau_w": self.auto.tau_w, "tau_R": self.auto.tau_R, "alpha_t": self.auto.alpha_t,
                    "tail_buf": list(self.auto.tail_buf), "ph_alpha_sr": self.auto.ph_alpha.stable_run,
                },
                "hyper": asdict(self.hyper), "numerics": asdict(self.num),
                "nodes": [{
                    "id": n.id, "content": n.content, "emb": n.emb.tolist(),
                    "V": n.V, "D": n.D, "S": n.S, "R": n.R, "E": n.E, "Chi": n.Chi,
                    "parent": n.parent, "children": n.children, "birth_step": n.birth_step,
                    "chapter": n.chapter, "source": n.source
                } for n in self.nodes.values()],
                "leaves": list(self.leaves),
            }
            parent = os.path.dirname(os.path.abspath(path))
            if parent and parent != ".":
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    @classmethod
    def snapshot_load(cls, embedder: "EmbedderProtocol", path: str,
                      external_rag: Optional[ExternalRAGProtocol] = None) -> "HTNRMemoryV54P":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dim = embedder.get_sentence_embedding_dimension()
        if dim != data.get("dim"):
            raise ValueError(f"Embedder dim {dim} != snapshot dim {data.get('dim')}")
        inst = cls(embedder, external_rag=external_rag,
                   hyper=Hyper(**data.get("hyper", {})),
                   numerics=Numerics(**data.get("numerics", {})))
        inst.step = int(data.get("step", 0))
        inst.next_id = int(data.get("next_id", 0))
        ctx = data.get("ctx", {})
        H = inst.hyper.K_contexts

        def _fit_heads(value: Any) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 2:
                arr = arr.reshape(-1, dim) if arr.size else np.zeros((0, dim), dtype=np.float32)
            out = np.zeros((H, dim), dtype=np.float32)
            h = min(H, arr.shape[0])
            d = min(dim, arr.shape[1])
            if h and d:
                out[:h, :d] = arr[:h, :d]
            return out

        def _fit_vec(value: Any) -> np.ndarray:
            arr = np.asarray(value, dtype=np.float32)
            out = np.zeros(dim, dtype=np.float32)
            length = min(dim, arr.size)
            if length:
                out[:length] = arr.flat[:length]
            return out

        inst.M_heads = _fit_heads(ctx.get("M_heads", np.zeros((H, dim), dtype=np.float32)))
        weights = np.asarray(ctx.get("M_weights", np.ones(H, dtype=np.float32) / max(1, H)), dtype=np.float32)
        if weights.size < H:
            weights = np.pad(weights, (0, H - weights.size), constant_values=0.0)
        else:
            weights = weights[:H]
        if not np.any(np.isfinite(weights)) or float(np.sum(np.abs(weights))) <= 1e-8:
            weights = np.ones(H, dtype=np.float32)
        inst.M_weights = weights.astype(np.float32)
        inst.M_weights = inst.M_weights / (np.sum(inst.M_weights) + 1e-8)
        inst.M_agg = _fit_vec(ctx.get("M_agg", np.zeros(dim, dtype=np.float32)))
        inst.M_long = _fit_vec(ctx.get("M_long", np.zeros(dim, dtype=np.float32)))
        inst.smoothed_lambda = float(ctx.get("smoothed_lambda", 0.05))
        inst.current_chapter = int(ctx.get("current_chapter", 0))
        inst.last_chapter_step = int(ctx.get("last_chapter_step", -10**9))
        auto = data.get("auto", {})
        inst.auto.tau_w = float(auto.get("tau_w", 0.2))
        inst.auto.tau_R = float(auto.get("tau_R", math.sqrt((1.0 + inst.auto.tau_w) / 2.0)))
        inst.auto.alpha_t = float(auto.get("alpha_t", 0.9))
        inst.auto.tail_buf.clear()
        for v in auto.get("tail_buf", []):
            inst.auto.tail_buf.append(float(v))
        inst.nodes.clear()
        inst.leaves.clear()
        for row in data.get("nodes", []):
            node = HTNRNode(
                id=int(row["id"]), content=row.get("content", ""),
                emb=np.array(row["emb"], dtype=np.float32),
                V=float(row["V"]), D=float(row["D"]), S=float(row["S"]),
                R=float(row["R"]), E=float(row["E"]), Chi=float(row["Chi"]),
                parent=(int(row["parent"]) if row["parent"] is not None else None),
                children=list(row.get("children", [])),
                birth_step=int(row.get("birth_step", 0)),
                chapter=int(row.get("chapter", 0)),
                source=row.get("source", "Unknown")
            )
            inst.nodes[node.id] = node
        inst.leaves = [int(x) for x in data.get("leaves", [])]
        inst._refresh_leaf_cache()
        inst._ensure_ann_index()
        logger.info(f"Loaded snapshot: {path} (nodes={len(inst.nodes)}, leaves={len(inst.leaves)})")
        return inst

    # ----- Close -----
    def close(self) -> None:
        self.log.close()


# ---------- Embedder protocol & implementations ----------
class EmbedderProtocol(Protocol):
    def get_sentence_embedding_dimension(self) -> int: ...
    def encode(self, texts: Any) -> np.ndarray: ...


class SentenceTransformerEmbedder(EmbedderProtocol):
    def __init__(self, model_name: str = "google/embeddinggemma-300m",
                 *, device: Optional[str] = None, batch_size: int = 32,
                 normalize_embeddings: bool = False, cache_capacity: int = 20000):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("pip install 'sentence-transformers' 필요") from e

        self._batch_size = int(max(1, batch_size))
        self._normalize = bool(normalize_embeddings)
        self._lock = threading.RLock()
        self.model = SentenceTransformer(model_name, device=device)
        dim = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(dim):
            self.dim = int(dim())
        else:
            raise RuntimeError("SentenceTransformer 모델이 임베딩 차원을 제공하지 않습니다.")
        self.cache = LRUCache(capacity=cache_capacity)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: Any) -> np.ndarray:
        if isinstance(texts, str):
            inputs = [texts]
        else:
            inputs = list(texts)

        if not inputs:
            return np.zeros((0, self.dim), dtype=np.float32)

        misses: List[str] = []
        order: List[Tuple[str, Optional[np.ndarray]]] = []
        for s in inputs:
            cached = self.cache.get(s)
            order.append((s, cached))
            if cached is None:
                misses.append(s)

        if misses:
            with self._lock:
                pending: List[str] = []
                for s in misses:
                    cached = self.cache.get(s)
                    if cached is None:
                        pending.append(s)
                if pending:
                    vectors = self.model.encode(
                        pending,
                        batch_size=self._batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize,
                    )
                    arr = np.asarray(vectors, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    for text, vec in zip(pending, arr):
                        self.cache.put(text, vec)

        out: List[np.ndarray] = []
        for s, cached in order:
            if cached is not None:
                out.append(cached)
                continue
            vec = self.cache.get(s)
            if vec is None:
                vec = np.zeros(self.dim, dtype=np.float32)
            out.append(vec)
        return np.stack(out, axis=0)
