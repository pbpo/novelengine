from __future__ import annotations

import random
from typing import List, Tuple


class RhythmMixer:
    """Cycles through predefined rhythm modes with optional external overrides."""

    MODES: List[Tuple[str, str]] = [
        ("telegraphic", "짧은 문장 위주(6~12자), 쉼표 최소, 동사·명사 절약"),
        ("periodic", "긴 문장 2개 + 짧은 문장 1개, 종속절 1회"),
        ("interjection", "대사/서술 사이 말끊김(—, …) 1~2회 허용"),
        ("image_lead", "사물·공간 디테일로 개시 후 대사 진입"),
        ("inner_lead", "내면 2~3문장으로 개시, 대사는 뒤로"),
        ("dialogue_heavy", "대사 비중 60% 이상, 설명 최소, 동작지시 포함"),
        ("ellipsis", "생략부호 …/— 1~2회로 단절감 부여"),
        ("catalog", "3~5개 병렬 나열문 1회로 정보 압축"),
    ]

    def __init__(self, seed: int = 17) -> None:
        self._index = 0
        self._rng = random.Random(seed)
        self._modes = list(self.MODES)
        self._rng.shuffle(self._modes)

    def _pick(self) -> Tuple[str, str]:
        name, hint = self._modes[self._index % len(self._modes)]
        self._index += 1
        return name, hint

    def force_mode(self, name: str) -> None:
        for idx, (mode_name, _) in enumerate(self._modes):
            if mode_name == name:
                self._index = idx
                break

    def scene_constraint(self) -> str:
        name, hint = self._pick()
        return (
            f"- [리듬 변주:{name}] {hint}\n"
            "- 균형 루틴(대사→내면→환경 반복) 금지, 이번 장면은 위 변주를 따를 것."
        )
