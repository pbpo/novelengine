from __future__ import annotations

from typing import List


class CanonLock:
    """Tracks canon facts and forbidden elements with formatted constraints."""

    def __init__(self, facts: List[str] | None = None, must_not: List[str] | None = None) -> None:
        self.facts = [fact.strip() for fact in (facts or []) if fact and fact.strip()]
        self.must_not = [item.strip() for item in (must_not or []) if item and item.strip()]

    def constraints(self) -> List[str]:
        lines: List[str] = []
        if self.facts:
            lines.append("- [캐논 고정] 아래 사실만 반복적으로 유지. 표현은 바뀌어도 내용 변경 금지.")
            for fact in self.facts:
                lines.append(f"  • {fact}")
        if self.must_not:
            lines.append("- [금지] " + ", ".join(self.must_not))
        return lines

    def violates(self, text: str) -> List[str]:
        if not text:
            return []
        lower = text.lower()
        return [item for item in self.must_not if item.lower() in lower]
