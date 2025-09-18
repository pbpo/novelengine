from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from story_engine.llm import LLM, LLM_OAI


class TwoPassDirector:
    """Planner + stylist pair used to outline and write scenes."""

    def __init__(self, planner_model: str, stylist_model: str) -> None:
        self.planner = LLM_OAI(planner_model)
        self.stylist = LLM(stylist_model)

    @staticmethod
    def _crop(payload: str) -> str:
        match = re.search(r"```(?:json)?\n(.*?)\n```", payload, re.DOTALL)
        payload = match.group(1) if match else payload
        if "{" in payload and "}" in payload:
            return payload[payload.find("{") : payload.rfind("}") + 1]
        return "{}"

    def plan(
        self,
        ch: int,
        beat: str,
        extras: str,
        rag_excerpt: str,
        *,
        rag_semantic: str = "",
        stc_must: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        stc_lines = "\n".join(f"- {item}" for item in (stc_must or []) if item)
        user = (
            f"비트:{beat}\n장:{ch}\n제약:\n{extras}\n"
            + (f"[STC_MUST_COVER]\n{stc_lines}\n" if stc_lines else "")
            + "### 최근 맥락 ###\n"
            + rag_excerpt
            + ("\n### 잠재적 연결 ###\n" + rag_semantic if rag_semantic else "")
            + "\n스키마: {\"beats\":[],\"causality\":\"\",\"hook\":\"\",\"answer\":\"\",\"tone\":\"\",\"connection_point\":\"\",\"style_mode\":\"\",\"candidate_facts\":[\"...\"]}"
        )
        system = "논리 플롯 기획자. JSON만."
        try:
            raw = self.planner.gen(system, user, max_tokens=800, temperature=0.2, force_json=True)
            return json.loads(self._crop(raw))
        except Exception:
            return {}

    def write_segment(
        self,
        ch: int,
        beat: str,
        plan: Dict[str, Any],
        extras: str,
        prev_tail: str,
        final: bool,
        max_tokens: int,
        temperature: float,
        *,
        stc_focus: Optional[List[str]] = None,
    ) -> str:
        system = "소설가. 출력=본문만."
        guard = "<<END>>" if final else ""
        continuation = "\n[이어쓰기] 직전 문장 반복 금지, 즉시 진행." if prev_tail else ""
        focus_block = (
            "[STC_FOCUS]\n" + "\n".join(f"- {item}" for item in (stc_focus or []) if item) + "\n"
            if stc_focus
            else ""
        )
        style = ""
        if isinstance(plan, dict):
            style = (plan.get("style_mode") or "").strip()
        style_line = f"- [리듬 변주 강제] 스타일 모드='{style}'" if style else ""
        rules: List[str] = [
            "- [감정 중심 서술] 포커스 인물의 내면·감각을 우선 반영",
            "- [Show, Don't Tell] 설명 대신 행동·대사·감각 묘사로 정보 전달",
            "- [감정 직접 서술 금지] '불안하다/화난다' 대신 행동·생리 반응으로 표현",
            "- 과설명 자제(정보는 필요할 때만)",
            "- 감정선–인과 중심으로 자연스럽게 이어 쓰기",
            "- [STC_FOCUS] 항목을 장면 전개에 직접 반영",
            "- 시점 규칙 준수",
            "- 장 끝에 '" + guard + "'" if final else "- 장면 종료 금지",
        ]
        if style_line:
            rules.append(style_line)
        tail = prev_tail[-700:] if prev_tail else ""
        user = (
            f"[장 {ch}] 비트:{beat}{continuation}\n계획서: ```{json.dumps(plan, ensure_ascii=False)}```\n"
            + focus_block
            + f"제약:\n{extras}\n규칙:\n"
            + "\n".join(rules)
            + "\n[tail]\n"
            + tail
        )
        return self.stylist.gen(
            system,
            user,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.97,
        )

    def write_longform(
        self,
        ch: int,
        beat: str,
        plan: Dict[str, Any],
        extras: str,
        segments: int,
        seg_tokens: int,
        temp: float,
        *,
        stc_focus: Optional[List[str]] = None,
        segment_hints: Optional[List[str]] = None,
    ) -> str:
        text = ""
        focus_items = [item for item in (stc_focus or []) if item]
        for idx in range(segments):
            batch: List[str] = []
            if focus_items:
                start = (idx * 2) % len(focus_items)
                batch = focus_items[start : start + 2]
                need = min(2, len(focus_items)) - len(batch)
                if need > 0:
                    batch += focus_items[:need]
            seg_extras = extras
            if segment_hints:
                hint = segment_hints[idx] if idx < len(segment_hints) else ""
                if hint:
                    seg_extras = (extras + "\n[MICRO_INCIDENT]\n- " + hint).strip()
            out = self.write_segment(
                ch,
                beat,
                plan,
                seg_extras,
                text,
                idx == segments - 1,
                seg_tokens,
                temp,
                stc_focus=batch,
            )
            text += ("\n" if text else "") + (out.strip() if out else "")
            if out and "<<END>>" in out:
                break
        return text.replace("<<END>>", "")
