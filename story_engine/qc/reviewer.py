from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from story_engine.llm import LLM_OAI


class O3Reviewer:
    """LLM-based QC reviewer mirroring the legacy behaviour."""

    def __init__(self, model_env: str, default_model: str = "gpt-5-mini") -> None:
        model = os.environ.get(model_env, default_model)
        self.model = model
        self.lm = LLM_OAI(model)

    @staticmethod
    def _safe_json(payload: str) -> Dict[str, Any]:
        match = re.search(r"```(?:json)?\n(.*?)\n```", payload, re.DOTALL)
        payload = match.group(1) if match else payload
        if "{" in payload and "}" in payload:
            try:
                payload = payload[payload.find("{") : payload.rfind("}") + 1]
            except Exception:
                pass
        try:
            data = json.loads(payload)
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def review(
        self,
        ch: int,
        beat: str,
        text: str,
        constraints: str,
        stc_expect: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        expect_block = "\n".join(f"- {item}" for item in (stc_expect or []) if item)
        user = (
            f"장:{ch}, 비트:{beat}\n제약:\n{constraints}\n"
            "- 인과 표현(‘때문에/그래서/그 결과로/이어서/직후’) 최소 1회 이상 포함.\n"
            "- 인물이 카드와 모순되거나 무기력하게 행동하면 각각 ooc/passive_protagonist로 표시.\n"
            + ("[STC 기대 항목]\n" + expect_block + "\n" if expect_block else "")
            + "[장면]\n"
            + text
            + "\n[스키마]{\"ok\":true,\"violations\":[\"info_dump\"],\"notes\":\"\"}"
        )
        system = "소설 QC. JSON만."
        payload = self.lm.gen(system, user, max_tokens=800, temperature=0.1, force_json=True)
        data = self._safe_json(payload)
        data.setdefault("ok", True)
        data.setdefault("violations", [])
        data.setdefault("notes", "")
        return data
