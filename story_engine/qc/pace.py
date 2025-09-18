from __future__ import annotations

import re
from typing import Dict, List, Pattern


class PaceLimiter:
    """Limits number of distinct event categories per scene."""

    def __init__(self, max_events: int = 2, cats: Dict[str, List[str]] | None = None) -> None:
        self.max_events = int(max_events)
        self.compiled: Dict[str, List[Pattern[str]]] = {}
        self.last_overflow = False
        default = cats or {
            "call_or_sms": [r"전화", r"문자", r"메시지"],
            "break_in": [r"침입", r"문고리", r"현관", r"창문"],
            "chase": [r"추격", r"쫓", r"도망"],
            "summon": [r"소환", r"면담", r"감사팀"],
            "manipulate": [r"조작", r"삭제", r"위조"],
            "reveal": [r"폭로", r"밝혀졌", r"드러났"],
            "surveil": [r"감시", r"cctv", r"미행", r"도청"],
            "violence": [r"폭력", r"흉기", r"구타", r"총"],
        }
        self.update_patterns(default)

    def update_patterns(self, cats: Dict[str, List[str]]) -> None:
        compiled: Dict[str, List[Pattern[str]]] = {}
        for name, patterns in (cats or {}).items():
            bucket: List[Pattern[str]] = []
            for pat in patterns or []:
                if 0 < len(pat) <= 80:
                    try:
                        bucket.append(re.compile(pat, re.IGNORECASE))
                    except re.error:
                        continue
            if bucket:
                compiled[name] = bucket
        if compiled:
            self.compiled = compiled

    def count_categories(self, text: str) -> int:
        total = 0
        for regs in self.compiled.values():
            if any(reg.search(text) for reg in regs):
                total += 1
        return total

    def constraint(self) -> str:
        return f"- [페이싱 제한] 새 사건 유형 최대 {self.max_events}개. 나머지는 암시/예고로 남길 것."

    def check_overflow(self, text: str) -> bool:
        self.last_overflow = self.count_categories(text) > self.max_events
        return self.last_overflow
