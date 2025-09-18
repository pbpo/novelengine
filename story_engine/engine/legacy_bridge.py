"""Helpers to reuse legacy stc.Engine with refactored modules."""

from __future__ import annotations

from story_engine.autonomy.strategy import AutonomyScorer, CanonCouncil, RhythmUCB
from story_engine.canon.lock import CanonLock
from story_engine.memory.htnr import HTNRMemoryV2, htnr_load, htnr_save
from story_engine.qc.anti_repeat import AntiRepeat
from story_engine.qc.pace import PaceLimiter
from story_engine.qc.reviewer import O3Reviewer
from story_engine.writing.rhythm import RhythmMixer


def patch_legacy_engine() -> None:
    """Monkeypatch the legacy stc module so it uses the refactored classes."""
    import stc as legacy  # type: ignore

    legacy.AutonomyScorer = AutonomyScorer
    legacy.CanonCouncil = CanonCouncil
    legacy.RhythmUCB = RhythmUCB
    legacy.CanonLock = CanonLock
    legacy.RhythmMixer = RhythmMixer
    legacy.AntiRepeat = AntiRepeat
    legacy.PaceLimiter = PaceLimiter
    legacy.O3Reviewer = O3Reviewer
    legacy.HTNRMemoryV2 = HTNRMemoryV2
    legacy.htnr_load = htnr_load
    legacy.htnr_save = htnr_save
