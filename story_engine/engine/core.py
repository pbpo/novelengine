"""Modern Story Engine runner that avoids the legacy `stc` dependency."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import textwrap
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from story_engine.autonomy.strategy import AutonomyScorer, RhythmUCB
from story_engine.canon.lock import CanonLock
from story_engine.config import EngineConfig
from story_engine.continuity import ContinuityLedger
from story_engine.engine.canon_action import ActionCanonTracker
from story_engine.engine.loop_breakers import (
    DialogueLoopBlocker,
    PlaceTimeGate,
    BeatMutex,
    norm_place,
)
from story_engine.planning.director import TwoPassDirector
from story_engine.qc.anti_repeat import AntiRepeat
from story_engine.qc.reviewer import O3Reviewer
from story_engine.writing.rhythm import RhythmMixer


@dataclass(slots=True)
class EngineServices:
    """Container for refactored service singletons used by the engine."""

    canon_lock: CanonLock
    canon_council: Any
    canon_limits: Dict[str, int]
    autonomy_scorer: AutonomyScorer
    autonomy_ucb: RhythmUCB
    autonomy_enabled: bool
    autonomy_temp_add: float
    rhythm_mixer: RhythmMixer


class DummyEmbedder:
    """Lightweight embedding shim so QC modules remain functional offline."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def enc(self, text: str) -> List[np.ndarray]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for idx, char in enumerate(text or ""):
            vec[idx % self.dim] += float(ord(char) % 97) / 100.0
        return [vec]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class RAGStore:
    """Minimal sqlite-backed context store used by the modern engine."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.con = sqlite3.connect(path)
        cur = self.con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rag (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                chapter INTEGER DEFAULT 0,
                beat TEXT DEFAULT '',
                content TEXT NOT NULL,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.con.commit()

    def store_bible(self, text: str) -> None:
        cur = self.con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('bible', ?)",
            (text,),
        )
        self.con.commit()

    def fetch_bible(self) -> str:
        cur = self.con.cursor()
        row = cur.execute("SELECT value FROM meta WHERE key='bible'").fetchone()
        return row[0] if row else ""

    def store_segment(self, chapter: int, beat: str, content: str) -> None:
        cur = self.con.cursor()
        cur.execute(
            "INSERT INTO rag(kind, chapter, beat, content) VALUES(?,?,?,?)",
            ("segment", chapter, beat, content),
        )
        self.con.commit()

    def recent_segments(self, limit: int = 3) -> List[str]:
        cur = self.con.cursor()
        rows = cur.execute(
            "SELECT content FROM rag WHERE kind='segment' ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [row[0] for row in rows[::-1]]

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


class Engine:
    """Async orchestration engine that uses refactored modules end-to-end."""

    def __init__(
        self,
        cfg: EngineConfig,
        rag_db: str,
        out_path: str,
        target_chars: int,
        segments: int,
        seg_tokens: int,
        *,
        services: EngineServices,
    ) -> None:
        self.cfg = cfg
        self.out_path = out_path
        self.target_chars = target_chars
        self.segments = segments
        self.seg_tokens = seg_tokens
        self.services = services
        self.rag = RAGStore(rag_db)
        planner_model = os.environ.get(cfg.planner_model_env, "gpt-4o-mini")
        stylist_model = cfg.model
        self.director = TwoPassDirector(planner_model, stylist_model)
        self.reviewer = O3Reviewer(cfg.reviewer_model_env)
        self.embedder = DummyEmbedder()
        self.anti_repeat = AntiRepeat(self.embedder, **asdict(cfg.anti_repeat))
        self.stage_order = list(services.canon_limits.keys()) or [
            "SETUP",
            "ASCENT",
            "CONVERGE",
            "RESOLVE",
        ]
        self.generated: List[str] = []
        self._bible_chars: str = ""
        self.ledger = ContinuityLedger()
        self.action_canon = ActionCanonTracker(self.ledger, watch_actors=["김대리"])
        self.place_gate = PlaceTimeGate(self.ledger, cool_chapters=2)
        self.beat_mutex = BeatMutex(cooldown=2)
        self.dialog_loop = DialogueLoopBlocker(per_chapter_cap=1)

    def ingest_bible(self, text: str) -> None:
        self._bible_chars = text or ""
        self.rag.store_bible(self._bible_chars)

    async def autorun(self, target_chars: Optional[int] = None) -> str:
        goal = int(target_chars or self.target_chars)
        total = 0
        output: List[str] = []
        beats_matrix = list(self.cfg.beats)
        for chapter_idx, beats in enumerate(beats_matrix, start=1):
            stage_name = self.stage_order[min(chapter_idx - 1, len(self.stage_order) - 1)]
            for beat in beats:
                extras_lines, picked_mode = self._build_constraints(stage_name, beat, chapter_idx)
                extras_for_plan = "\n".join(line for line in extras_lines if line)
                rag_excerpt = self._context_snippet()
                plan = self.director.plan(
                    chapter_idx,
                    beat,
                    extras_for_plan,
                    rag_excerpt,
                    rag_semantic=self._semantic_hint(),
                )
                self._record_plan_signals(plan, chapter_idx)
                self._augment_constraints_with_plan(extras_lines, plan)
                extras_str = "\n".join(line for line in extras_lines if line)
                temp = self._temperature_for_mode(picked_mode)
                segment_text = self.director.write_longform(
                    chapter_idx,
                    beat,
                    plan,
                    extras_str,
                    self.segments,
                    self.seg_tokens,
                    temp,
                ).strip() or self._fallback_text(chapter_idx, beat)

                rewrite_hints: List[str] = []
                action_issues = self.action_canon.check_conflict(segment_text)
                if action_issues:
                    rewrite_hints.append(ActionCanonTracker.rewrite_hint(action_issues))
                loop_issues = self._collect_loop_conflicts(segment_text, chapter_idx)
                if loop_issues:
                    rewrite_hints.append(self._loop_rewrite_hint(loop_issues))

                active_extras = extras_str
                if rewrite_hints:
                    revised_extras = (extras_str + "\n" + "\n".join(rewrite_hints)).strip()
                    regenerated = self.director.write_longform(
                        chapter_idx,
                        beat,
                        plan,
                        revised_extras,
                        self.segments,
                        self.seg_tokens,
                        temp,
                    ).strip()
                    if regenerated:
                        segment_text = regenerated
                        active_extras = revised_extras

                repetitive = self.anti_repeat.repetitive(chapter_idx, segment_text)
                if repetitive:
                    segment_text += "\n\n[주의] 반복 가능성 감지 — 내용 검토 권장."
                review = self.reviewer.review(chapter_idx, beat, segment_text, active_extras)
                self.anti_repeat.push(chapter_idx, segment_text)
                self._update_autonomy(picked_mode, repetitive, review)
                wrapped = self._wrap_output(chapter_idx, beat, segment_text, review)
                output.append(wrapped)
                self.rag.store_segment(chapter_idx, beat, segment_text)
                self.action_canon.commit_from_text(segment_text, chapter_idx)
                self.place_gate.update_after_scene(segment_text)
                for group in self.beat_mutex.detect_group(segment_text):
                    self.beat_mutex.lock(group, chapter_idx)
                if re.search(r"내일\s*점심", segment_text):
                    self.ledger.record_appointment("lunch_tomorrow", "내일 점심", chapter_idx)
                place_sig = norm_place(segment_text)
                envelope_tag = "ENVELOPE" if re.search(r"봉투", segment_text) else ""
                signature = (place_sig or "UNKNOWN") + (
                    f"|{envelope_tag}" if envelope_tag else ""
                )
                self.ledger.push_scene_signature(signature)
                self.ledger.update_last(place_sig or "", None, chapter_idx)
                total += len(segment_text)
                await asyncio.sleep(0)
                if total >= goal:
                    break
            if total >= goal:
                break
        final_text = "\n\n".join(output)
        with open(self.out_path, "w", encoding="utf-8") as fh:
            fh.write(final_text)
        self.generated = output
        return final_text

    def _build_constraints(self, stage: str, beat: str, chapter: int) -> tuple[List[str], str]:
        ucb = self.services.autonomy_ucb
        mode = ucb.pick()
        self.services.rhythm_mixer.force_mode(mode)
        rhythm = self.services.rhythm_mixer.scene_constraint()
        canon_lines = self.services.canon_lock.constraints()
        canon_lines.append(
            self.services.canon_council.constraint(stage, self.services.canon_limits)
        )
        extras: List[str] = []
        extras.append(f"- [비트] {beat}")
        bible_hint = self._bible_hint()
        if bible_hint:
            extras.append(bible_hint)
        extras.extend(canon_lines)
        extras.append(rhythm)
        action_line = self.action_canon.constraint()
        if action_line:
            extras.append(action_line)
        beat_line = self.beat_mutex.constraint(chapter)
        if beat_line:
            extras.append(beat_line)
        dialog_line = self.dialog_loop.constraint()
        if dialog_line:
            extras.append(dialog_line)
        return [line for line in extras if line], mode

    def _context_snippet(self) -> str:
        recent = self.rag.recent_segments()
        bible = self._bible_hint(inline=True)
        payload = "\n".join(filter(None, [bible] + recent))
        return payload[-1200:]

    def _semantic_hint(self) -> str:
        recent = self.rag.recent_segments(limit=1)
        return recent[0][-240:] if recent else ""

    def _bible_hint(self, *, inline: bool = False) -> str:
        source = self._bible_chars or self.rag.fetch_bible()
        if not source:
            return ""
        snippet = textwrap.shorten(source, width=280, placeholder="…")
        prefix = "[세계관]" if inline else "- [세계관 유지]"
        return f"{prefix} {snippet}"

    def _temperature_for_mode(self, mode: str) -> float:
        base = self.cfg.gen_temp_by_mode.get(mode, 0.82)
        return base + (self.services.autonomy_temp_add if self.services.autonomy_enabled else 0.0)

    def _fallback_text(self, chapter_idx: int, beat: str) -> str:
        return (
            f"[생성 비활성화] 장 {chapter_idx} 비트 '{beat}' 장면을 수동 작성하세요."
        )

    def _augment_constraints_with_plan(self, extras: List[str], plan: Dict[str, Any]) -> None:
        place_hint = self._extract_place_hint(plan)
        place_line = self.place_gate.constraint(place_hint)
        if place_line:
            extras.append(place_line)

    def _extract_place_hint(self, plan: Dict[str, Any]) -> str:
        if not isinstance(plan, dict):
            return ""
        candidate_keys = (
            "place",
            "location",
            "setting",
            "scene_location",
            "anchor",
            "scene",
        )
        for key in candidate_keys:
            value = plan.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return first.strip()
        return ""

    def _record_plan_signals(self, plan: Dict[str, Any], chapter: int) -> None:
        if not plan:
            return
        try:
            payload = json.dumps(plan, ensure_ascii=False)
        except Exception:
            payload = str(plan)
        if "내일 점심" in payload:
            self.ledger.record_appointment("lunch_tomorrow", "내일 점심", chapter)

    def _collect_loop_conflicts(self, text: str, chapter: int) -> List[str]:
        issues: List[str] = []
        issues.extend(self.place_gate.check_conflict(text))
        issues.extend(self.beat_mutex.check_conflict(text, chapter))
        issues.extend(self.dialog_loop.check_conflict(text, chapter))
        if self.ledger.has_appointment("lunch_tomorrow") and re.search(
            r"(지하주차장|봉투|모르는\s*게\s*약)",
            text or "",
        ):
            issues.append("예약 이후 동일 패턴 재연(약속-이전 루프)")
        return issues

    @staticmethod
    def _loop_rewrite_hint(issues: List[str]) -> str:
        if not issues:
            return ""
        return (
            "- [루프 차단/정합성 수정]\n"
            "  1) 직전 장소로의 회귀 금지(회상 1문장만 허용),\n"
            "  2) '봉투/모르는 게 약/복사본' 패턴은 쿨다운 중이므로 대체 행동/정보로 전개,\n"
            "  3) 대사 템플릿은 중복 시 재서술·압축.\n"
            "  4) 전환문 1문장으로 이동 사유를 명시."
        )

    def _wrap_output(
        self,
        chapter_idx: int,
        beat: str,
        text: str,
        review: Dict[str, Any],
    ) -> str:
        header = f"[장 {chapter_idx} | {beat}]"
        notes: List[str] = []
        if review.get("violations"):
            violations = ", ".join(review.get("violations", []))
            notes.append(f"QC: {violations}")
        if review.get("notes"):
            notes.append(str(review.get("notes")))
        note_block = ("\n" + "\n".join(notes)) if notes else ""
        return f"{header}\n{text}{note_block}"

    def _update_autonomy(self, mode: str, repetitive: bool, review: Dict[str, Any]) -> None:
        scorer = self.services.autonomy_scorer
        reward = scorer.score(
            novelty_ok=not repetitive,
            qc_ok=bool(review.get("ok", True)),
            pace_ok=True,
            has_transition=self._has_transition(review),
            sense_ok=True,
            stc_hit=not review.get("violations"),
        )
        self.services.autonomy_ucb.update(mode, reward)

    @staticmethod
    def _has_transition(review: Dict[str, Any]) -> bool:
        notes = (review.get("notes") or "").lower()
        return any(token in notes for token in ("transition", "flow", "caus"))

    def close(self) -> None:
        self.rag.close()


def build_engine_services(cfg: EngineConfig, resources: Dict[str, Any]) -> EngineServices:
    """Assemble the EngineServices dataclass from factory-produced resources."""

    canon = resources.get("canon", {})
    autonomy = resources.get("autonomy", {})
    rhythm: RhythmMixer = resources.get("rhythm")
    if not canon or not canon.get("lock") or not canon.get("council"):
        raise RuntimeError("Canon services must be initialised before building the engine.")
    if not autonomy or not autonomy.get("scorer") or not autonomy.get("ucb"):
        raise RuntimeError("Autonomy services must be initialised before building the engine.")
    if rhythm is None:
        raise RuntimeError("Rhythm mixer is missing; call create_rhythm_mixer() first.")

    return EngineServices(
        canon_lock=canon["lock"],
        canon_council=canon["council"],
        canon_limits=canon.get("stage_limits", {}),
        autonomy_scorer=autonomy["scorer"],
        autonomy_ucb=autonomy["ucb"],
        autonomy_enabled=bool(autonomy.get("enabled", True)),
        autonomy_temp_add=float(autonomy.get("temp_add", 0.0)),
        rhythm_mixer=rhythm,
    )
