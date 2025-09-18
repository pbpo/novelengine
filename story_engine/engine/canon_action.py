"""Rules for maintaining action canon consistency across scenes."""

from __future__ import annotations

import re
from typing import List, Tuple

from story_engine.continuity import ContinuityLedger


class ActionCanonTracker:
    """Enforces actor-specific action outcomes without prescribing phrasing."""

    def __init__(
        self,
        ledger: ContinuityLedger,
        watch_actors: List[str] | None = None,
        slot: str = "photo",
    ) -> None:
        self.ledger = ledger
        self.slot = slot
        self.watch = [actor for actor in (watch_actors or ["김대리"]) if actor]

        self._fail_pats = [
            re.compile(p, re.IGNORECASE)
            for p in (
                r"사진[^\n]{0,12}(못|실패|안\s*찍)",
                r"셔터[^\n]{0,10}(안|먹)",
                r"저장[^\n]{0,8}안\s*됐",
                r"흔들려[^\n]{0,6}쓸\s*수\s*없",
                r"캡처[^\n]{0,8}실패",
            )
        ]
        self._succ_pats = [
            re.compile(p, re.IGNORECASE)
            for p in (
                r"사진[^\n]{0,10}(찍었|찍어\s*놨|찍어\s*두|건져)",
                r"(캡처|스샷)[^\n]{0,8}(했|있)",
                r"증거\s*사진",
                r"앨범[^\n]{0,8}있",
                r"파일[^\n]{0,8}있",
            )
        ]
        self._neut_pats = [re.compile(p, re.IGNORECASE) for p in (r"목격했", r"봤다", r"보[았]다")]

    # ------------------------------------------------------------------
    def _actor_lines(self, text: str, actor: str) -> List[str]:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        idxs = [i for i, ln in enumerate(lines) if actor in ln]
        window: List[str] = []
        seen: set[str] = set()
        for idx in idxs:
            for pos in (idx - 1, idx, idx + 1):
                if 0 <= pos < len(lines):
                    ln = lines[pos]
                    if ln not in seen:
                        seen.add(ln)
                        window.append(ln)
        return window

    def _classify(self, line: str) -> str:
        text = line or ""
        if any(p.search(text) for p in self._fail_pats):
            return "failure"
        if any(p.search(text) for p in self._succ_pats):
            return "success"
        if any(p.search(text) for p in self._neut_pats):
            return "neutral"
        return "unknown"

    def extract_mentions(self, text: str) -> List[Tuple[str, str, str]]:
        mentions: List[Tuple[str, str, str]] = []
        for actor in self.watch:
            for line in self._actor_lines(text, actor):
                tag = self._classify(line)
                if tag != "unknown":
                    mentions.append((actor, self.slot, tag))
        return mentions

    # ------------------------------------------------------------------
    def constraint(self) -> str:
        lines: List[str] = []
        for actor in self.watch:
            canon = (self.ledger.recall_action(actor, self.slot) or "").lower()
            if canon == "failure":
                lines.append(
                    f"- [행동 캐논] {actor}의 '사진 촬영'은 **실패**로 확정됨. "
                    "성공/보유/증거사진 서술 금지. 이후 대사·회상은 이 실패를 전제로 자연스럽게 반영."
                )
        return "\n".join(lines)

    def check_conflict(self, text: str) -> List[str]:
        issues: List[str] = []
        for actor in self.watch:
            canon = (self.ledger.recall_action(actor, self.slot) or "").lower()
            if canon != "failure":
                continue
            lines = self._actor_lines(text, actor)
            joined = "\n".join(lines)
            if any(p.search(line) for line in lines for p in self._succ_pats):
                issues.append(f"{actor}: 행동 캐논 위반(사진=성공으로 들림)")
                continue
            has_photo_context = any(
                re.search(r"사진|셔터|캡처|증거", line) for line in lines
            )
            has_failure_marker = any(p.search(joined) for p in self._fail_pats)
            if lines and has_photo_context and not has_failure_marker:
                issues.append(f"{actor}: 실패 전제 누락(사진 맥락이나 실패 언급 없음)")
        return issues

    @staticmethod
    def rewrite_hint(issues: List[str]) -> str:
        if not issues:
            return ""
        return (
            "- [행동 캐논 수정] '사진 촬영'은 **실패**였음. "
            "성공/보유/증거사진 서술을 제거하고, 실패를 전제로 발화/회상을 1회 명시(표현 자유)."
        )

    def commit_from_text(self, text: str, chapter: int) -> None:
        priority = {"failure": 3, "success": 2, "neutral": 1, "unknown": 0}
        chosen: dict[str, Tuple[str, str, str]] = {}
        for actor, slot, outcome in self.extract_mentions(text):
            key = f"{actor}::{slot}"
            if (key not in chosen) or (priority[outcome] > priority[chosen[key][2]]):
                chosen[key] = (actor, slot, outcome)
        for actor, slot, outcome in chosen.values():
            self.ledger.record_action(actor, slot, outcome, chapter, evidence="[auto-extract]")

