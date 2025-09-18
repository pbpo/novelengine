from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _norm(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm_val = float(np.linalg.norm(arr))
    if norm_val <= 1e-8:
        return arr
    return arr / norm_val


@dataclass
class STCNode:
    id: int
    content: str
    w: float
    birth: int


class STC:
    """Story Thread Cache mirror supporting take/save operations."""

    def __init__(
        self,
        emb: Any,
        alpha: float = 0.9,
        cap: int = 40,
        j_th: float = 0.75,
        path: str = "stc.json",
    ) -> None:
        self.emb = emb
        self.alpha = alpha
        self.cap = cap
        self.j = j_th
        self.path = path
        self.nodes: Dict[int, STCNode] = {}
        self.order: List[int] = []
        self.next = 0
        self.M = np.zeros(self.emb.dim, dtype=np.float32)

    def _sh(self, text: str) -> set[str]:
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
        return {" ".join(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))}

    @staticmethod
    def _similar(a: str, b: str) -> bool:
        a = (a or "").strip().lower()
        b = (b or "").strip().lower()
        return bool(a and b and a == b)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "cap": self.cap,
            "j": self.j,
            "next": self.next,
            "order": list(self.order),
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "w": node.w,
                    "birth": node.birth,
                }
                for node in self.nodes.values()
            ],
            "mean": self.M.tolist(),
        }

    def restore(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        self.alpha = float(data.get("alpha", self.alpha))
        self.cap = int(data.get("cap", self.cap))
        self.j = float(data.get("j", self.j))
        self.next = int(data.get("next", 0))
        self.order = [int(item) for item in data.get("order", [])][-self.cap :]
        self.nodes = {}
        for node in data.get("nodes", []):
            node_id = int(node.get("id", 0))
            self.nodes[node_id] = STCNode(
                node_id,
                (node.get("content") or "").strip(),
                float(node.get("w", 1.0)),
                int(node.get("birth", 0)),
            )
        mean = data.get("mean")
        if isinstance(mean, list) and mean:
            try:
                self.M = _norm(np.array(mean, dtype=np.float32))
            except Exception:
                self.M = np.zeros(self.emb.dim, dtype=np.float32)

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.path
        if not target:
            return
        try:
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(self.snapshot(), fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.path
        if not target or not os.path.exists(target):
            return
        try:
            with open(target, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self.restore(data)
        except Exception:
            pass

    def _dist(self, vec: np.ndarray) -> float:
        return float(np.linalg.norm(self.M - vec))

    def _update(self, node_id: int, text: str) -> None:
        vec = _norm(self.emb.enc(text)[0])
        self.M = self.alpha * self.M + (1 - self.alpha) * vec
        node = self.nodes.get(node_id)
        if node:
            node.content = text
            node.w = self._dist(vec)

    def take(self, query: str, k: int = 5) -> List[str]:
        if not self.nodes:
            return []
        vec = _norm(self.emb.enc(query)[0])
        scored: List[Tuple[float, int]] = []
        for node_id in self.order:
            node = self.nodes.get(node_id)
            if not node:
                continue
            vec_node = _norm(self.emb.enc(node.content)[0])
            score = float(vec @ vec_node)
            scored.append((score, node_id))
        scored.sort(reverse=True)
        selected = [node_id for _, node_id in scored[:k]]
        return [self.nodes[node_id].content for node_id in selected if node_id in self.nodes]

    def put(self, text: str, birth: int) -> None:
        vec = _norm(self.emb.enc(text)[0])
        self.next += 1
        node_id = self.next
        self.nodes[node_id] = STCNode(node_id, text, self._dist(vec), birth)
        self.order.append(node_id)
        self.order = self.order[-self.cap :]
        self.M = self.alpha * self.M + (1 - self.alpha) * vec

    def trim(self) -> None:
        if len(self.order) <= self.cap:
            return
        for node_id in self.order[:-self.cap]:
            self.nodes.pop(node_id, None)
        self.order = self.order[-self.cap :]
