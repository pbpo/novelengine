"""Modules that suppress structural repetition in generated scenes."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Dict, List

from story_engine.continuity import ContinuityLedger


PLACE_ALIASES: Dict[str, set[str]] = {
    "지하주차장": {"지하주차장", "지하 주차장", "주차장"},
    "엘리베이터": {"엘리베이터", "승강기", "리프트"},
    "복도": {"복도", "코리도", "홀"},
    "부장실": {"부장실", "본부장실", "실무실", "상사실"},
}


def norm_place(text: str) -> str:
    probe = (text or "").lower()
    for canon, aliases in PLACE_ALIASES.items():
        if any(alias in probe for alias in aliases):
            return canon
    return ""


class PlaceTimeGate:
    """Prevents hard loops back to recent locations without transitions."""

    def __init__(self, ledger: ContinuityLedger, cool_chapters: int = 2) -> None:
        self.ledger = ledger
        self.cool = int(cool_chapters)
        self.recent_places: deque[str] = deque(maxlen=3)
        seed = norm_place(self.ledger.last_place())
        if seed:
            self.recent_places.append(seed)

    def constraint(self, planned_place_hint: str) -> str:
        planned = norm_place(planned_place_hint)
        bans: List[str] = [place for place in self.recent_places if place]
        text: List[str] = []
        if planned:
            text.append(
                f"- 오프닝 장소 고정: {planned}. 첫 문단에 {planned}의 구체 디테일 2개로 시작."
            )
        if bans:
            text.append("- [장소 회귀 금지] 최근 장소 재사용 금지: " + ", ".join(bans))
            text.append(
                "- [이동 사유] 장소가 바뀌면 '…해서 올라갔다/내려갔다/옮겼다' 같은 전환문 1문장 필수."
            )
        return "\n".join(text)

    def update_after_scene(self, text: str) -> None:
        place = norm_place(text)
        if place:
            self.recent_places.append(place)

    def check_conflict(self, text: str) -> List[str]:
        issues: List[str] = []
        joined = text or ""
        bans = set(self.recent_places)
        for place in bans:
            if place and any(alias in joined for alias in PLACE_ALIASES.get(place, {place})):
                issues.append(f"장소 회귀: '{place}' 과다 재사용")
        if norm_place(joined) and not re.search(
            r"(그래서|그 결과로|이어서|직후|올라가|내려가|옮겨|이동)",
            joined,
        ):
            issues.append("장소 변경 전환문 누락")
        return issues


class BeatMutex:
    """Mutually exclusive beat tracker with cooldown windows."""

    GROUPS: Dict[str, List[str]] = {
        "ENVELOPE_HANDOFF": [
            r"봉투",
            r"모르는\s*게\s*약",
            r"복사본(이다|이래요|이랬다)?",
        ]
    }

    def __init__(self, cooldown: int = 2) -> None:
        self.cool = max(1, int(cooldown))
        self.locked: Dict[str, int] = {}

    def constraint(self, chapter: int) -> str:
        active = [group for group, until in self.locked.items() if chapter <= until]
        if not active:
            return ""
        return (
            "- [루프 금지] "
            + ", ".join(active)
            + f" 패턴은 {self.cool}장 쿨다운 중(회상 1문장만 허용)."
        )

    def lock(self, group: str, chapter: int) -> None:
        self.locked[group] = chapter + self.cool

    def detect_group(self, text: str) -> List[str]:
        hits: List[str] = []
        payload = text or ""
        for group, patterns in self.GROUPS.items():
            if any(re.search(pattern, payload) for pattern in patterns):
                hits.append(group)
        return hits

    def check_conflict(self, text: str, chapter: int) -> List[str]:
        issues: List[str] = []
        payload = text or ""
        active = [group for group, until in self.locked.items() if chapter <= until]
        for group in active:
            patterns = self.GROUPS.get(group, [])
            if any(re.search(pattern, payload) for pattern in patterns):
                issues.append(f"{group} 재연 금지 위반")
        return issues


class DialogueLoopBlocker:
    """Limits repeated dialogue templates per chapter."""

    LINES: List[str] = [
        r"그\s*봉투",
        r"모르는\s*게\s*약",
        r"복사본(이다|이래요|이랬다)?",
    ]

    def __init__(self, per_chapter_cap: int = 1) -> None:
        self.cap = max(1, int(per_chapter_cap))
        self.counts: Dict[tuple[int, str], int] = defaultdict(int)

    def constraint(self) -> str:
        return (
            "- [대사 루프 차단] '그 봉투…', '모르는 게 약', '복사본이다'는 장당 1회 이하. "
            "중복 시 재서술·압축."
        )

    def check_conflict(self, text: str, chapter: int) -> List[str]:
        issues: List[str] = []
        payload = text or ""
        for pattern in self.LINES:
            occurrences = len(re.findall(pattern, payload))
            key = (chapter, pattern)
            self.counts[key] = max(self.counts.get(key, 0), occurrences)
            if occurrences > self.cap:
                human_readable = re.sub(r"\\(.*?\\)", "", pattern)
                issues.append(f"대사 루프 과다: {human_readable} ×{occurrences}회")
        return issues

