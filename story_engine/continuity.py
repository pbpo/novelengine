"""Continuity tracking helpers for locations, actions, and appointments."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class ContinuityLedger:
    """Lightweight persistence layer for narrative continuity state."""

    def __init__(self, path: str = "continuity.json") -> None:
        self.path = path
        self.data: Dict[str, Any] = {
            "last": {"place": "", "time_min": None, "chapter": 0},
            "facts": {},
            "threads": {},
            "actions": {},
            "appointments": {},
            "scene_hashes": [],
        }
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self.data.update({k: v for k, v in loaded.items() if k in self.data})
                    for key in ("actions", "appointments", "scene_hashes"):
                        self.data.setdefault(key, loaded.get(key, self.data[key]))
            except Exception:
                # Keep defaults on read failure.
                pass
        # Ensure mandatory slots exist even if the stored file predates updates.
        self.data.setdefault("last", {"place": "", "time_min": None, "chapter": 0})
        self.data.setdefault("actions", {})
        self.data.setdefault("appointments", {})
        self.data.setdefault("scene_hashes", [])

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
        except Exception:
            # Persistence failures should not hard-stop generation.
            pass

    # ------------------------------------------------------------------
    # Last scene helpers
    def update_last(self, place: Optional[str], time_min: Optional[int], chapter: int) -> None:
        last = self.data.setdefault("last", {"place": "", "time_min": None, "chapter": 0})
        if place is not None:
            last["place"] = place
        if time_min is not None:
            last["time_min"] = time_min
        last["chapter"] = max(int(chapter), last.get("chapter", 0))
        self.save()

    def last_place(self) -> str:
        last = self.data.get("last") or {}
        return str(last.get("place", ""))

    def last_time_min(self) -> Optional[int]:
        last = self.data.get("last") or {}
        value = last.get("time_min")
        return int(value) if isinstance(value, int) else None

    def last_chapter(self) -> int:
        last = self.data.get("last") or {}
        value = last.get("chapter")
        return int(value) if isinstance(value, int) else 0

    # ------------------------------------------------------------------
    # Action canon helpers
    @staticmethod
    def _akey(actor: str, slot: str = "photo") -> str:
        a = (actor or "").strip().lower()
        s = (slot or "photo").strip().lower()
        return f"{a}::{s}" if a else s

    def record_action(
        self,
        actor: str,
        slot: str,
        outcome: str,
        chapter: int,
        evidence: str = "",
    ) -> None:
        actions: Dict[str, Dict[str, Any]] = self.data.setdefault("actions", {})
        key = self._akey(actor, slot)
        rec = actions.setdefault(
            key,
            {
                "actor": actor,
                "slot": slot,
                "outcome": "",
                "first_ch": int(chapter),
                "seen": [],
                "evidence": [],
            },
        )
        rec["outcome"] = (outcome or "").strip().lower()
        if int(chapter) not in rec["seen"]:
            rec["seen"].append(int(chapter))
        hint = (evidence or "").strip()
        if hint:
            rec.setdefault("evidence", [])
            rec["evidence"].append(hint[:160])
        self.save()

    def recall_action(self, actor: str, slot: str) -> str:
        key = self._akey(actor, slot)
        rec = self.data.get("actions", {}).get(key)
        return str((rec or {}).get("outcome", ""))

    def actions_summary(self) -> List[str]:
        out: List[str] = []
        for rec in self.data.get("actions", {}).values():
            out.append(
                f"{rec.get('actor', '?')}:{rec.get('slot', '?')}={rec.get('outcome', '')}"
            )
        return out

    # ------------------------------------------------------------------
    # Appointment helpers
    def record_appointment(self, key: str, when_label: str, chapter: int) -> None:
        appointments: Dict[str, Dict[str, Any]] = self.data.setdefault("appointments", {})
        entry = appointments.setdefault(key, {"when": when_label, "chapters": []})
        if int(chapter) not in entry["chapters"]:
            entry["chapters"].append(int(chapter))
        entry["when"] = when_label
        self.save()

    def has_appointment(self, key: str) -> bool:
        return key in (self.data.get("appointments") or {})

    # ------------------------------------------------------------------
    # Scene signature helpers
    def push_scene_signature(self, signature: str, cap: int = 10) -> None:
        arr: List[str] = self.data.setdefault("scene_hashes", [])
        if signature:
            arr.append(signature[:160])
            self.data["scene_hashes"] = arr[-cap:]
            self.save()

