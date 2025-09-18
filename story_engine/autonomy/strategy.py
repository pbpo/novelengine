from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional


class RhythmUCB:
    """Upper-Confidence-Bound selector for rhythm modes."""

    def __init__(self, modes: Optional[List[str]] = None) -> None:
        self.modes = list(dict.fromkeys(modes or ["image_lead", "dialogue_heavy"]))
        self.N: Dict[str, int] = {mode: 0 for mode in self.modes}
        self.Q: Dict[str, float] = {mode: 0.0 for mode in self.modes}
        self.t = 0

    def pick(self) -> str:
        self.t += 1
        cold = [mode for mode in self.modes if self.N.get(mode, 0) == 0]
        if cold:
            return random.choice(cold)

        def ucb(mode: str) -> float:
            visits = max(1, self.N.get(mode, 0))
            value = self.Q.get(mode, 0.0)
            return value + math.sqrt(2.0 * math.log(max(1, self.t)) / (visits + 1e-8))

        return max(self.modes, key=ucb)

    def update(self, mode: str, reward: float) -> None:
        if mode not in self.N:
            return
        self.N[mode] += 1
        visits = max(1, self.N[mode])
        prev = self.Q.get(mode, 0.0)
        self.Q[mode] = prev + (reward - prev) / visits


class CanonCouncil:
    """PVC workflow for canonical facts."""

    def __init__(
        self,
        base_facts: Optional[List[str]] = None,
        *,
        commit_threshold: int = 2,
        proposal_k: int = 2,
    ) -> None:
        self.base = {fact.strip() for fact in (base_facts or []) if fact and fact.strip()}
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.commit_threshold = max(1, int(commit_threshold))
        self.proposal_k = max(1, int(proposal_k))

    def propose(self, fact: str, chapter: int, evidence: str = "") -> None:
        fact = (fact or "").strip()
        if not fact or fact in self.base:
            return
        entry = self.candidates.setdefault(fact, {"count": 0, "chapters": set(), "evid": []})
        entry["count"] += 1
        entry["chapters"].add(int(chapter))
        evidence = (evidence or "").strip()
        if evidence:
            entry["evid"].append(evidence)

    def commit_ready(self) -> List[str]:
        ready: List[str] = []
        for fact, entry in self.candidates.items():
            if entry["count"] >= self.proposal_k and len(entry["chapters"]) >= self.commit_threshold:
                ready.append(fact)
        return ready

    def commit(self, facts: List[str]) -> None:
        for fact in facts:
            if fact in self.candidates:
                self.base.add(fact)
                self.candidates.pop(fact, None)

    def constraint(self, stage: str, limits: Dict[str, Any]) -> str:
        cap = int((limits or {}).get(stage, 0))
        if cap <= 0:
            return "- [캐논 확장] 금지: 이번 장면은 신사실 승격 없이 기존 사실만 활용."
        return (
            f"- [캐논 확장] 새 가설 사실 최대 {cap}개까지 제안. 가설 표현을 사용하고 근거 혹은 로그 1개를 동반할 것."
        )


class AutonomyScorer:
    """Weighted reward computation used for rollout scoring and rhythm feedback."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        default = {
            "novelty": 1.0,
            "qc": 0.8,
            "pace": 0.6,
            "transition": 0.3,
            "sense_fit": 0.2,
            "stc_hit": 0.4,
        }
        if weights:
            for key, value in weights.items():
                clean = key[2:] if key.startswith("w_") else key
                if clean in default:
                    default[clean] = float(value)
        self.weights = default

    def score(
        self,
        *,
        novelty_ok: bool,
        qc_ok: bool,
        pace_ok: bool,
        has_transition: bool,
        sense_ok: bool,
        stc_hit: bool,
    ) -> float:
        return (
            self.weights["novelty"] * (1.0 if novelty_ok else -1.0)
            + self.weights["qc"] * (1.0 if qc_ok else -1.0)
            + self.weights["pace"] * (1.0 if pace_ok else -1.0)
            + self.weights["transition"] * (1.0 if has_transition else 0.0)
            + self.weights["sense_fit"] * (1.0 if sense_ok else 0.0)
            + self.weights["stc_hit"] * (1.0 if stc_hit else 0.0)
        )
