from __future__ import annotations

import numpy as np
import re
from typing import Any, List, Optional, Tuple


def _norm(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm_val = float(np.linalg.norm(arr))
    return arr if norm_val < eps else arr / norm_val


class AntiRepeat:
    """Detects repetitive scenes using shingles and cosine similarity."""

    def __init__(
        self,
        emb: Any,
        *,
        window: int = 8,
        shingle: int = 5,
        jaccard: float = 0.25,
        cos: float = 0.90,
        **kwargs: Any,
    ) -> None:
        self.emb = emb
        self.win = int(kwargs.get("window", window))
        self.k = int(kwargs.get("shingle", shingle))
        self.J = float(kwargs.get("jaccard", jaccard))
        self.C = float(kwargs.get("cos", cos))
        self.hist: List[Tuple[int, set[str], np.ndarray]] = []
        self.base_win = self.win
        self.base_J = self.J
        self.base_C = self.C

    def shingles(self, text: str) -> set[str]:
        tokens = re.findall(r"[가-힣A-Za-z0-9]+", text)
        return {
            " ".join(tokens[i : i + self.k])
            for i in range(max(0, len(tokens) - self.k + 1))
        }

    def repetitive(self, ch: int, text: str) -> bool:
        shingles = self.shingles(text)
        embedding = _norm(self.emb.enc(text)[0])
        for _, prev_shingles, prev_vec in self.hist[-self.win :]:
            jac = len(shingles & prev_shingles) / float(len(shingles | prev_shingles) or 1)
            if jac >= self.J or float(embedding @ prev_vec) >= self.C:
                return True
        return False

    def push(self, ch: int, text: str) -> None:
        entry = (
            ch,
            self.shingles(text),
            _norm(self.emb.enc(text)[0]),
        )
        self.hist.append(entry)
        self.hist = self.hist[-self.win :]
