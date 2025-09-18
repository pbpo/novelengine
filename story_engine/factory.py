from __future__ import annotations

from dataclasses import asdict
from typing import Dict

from story_engine.autonomy.strategy import AutonomyScorer, CanonCouncil, RhythmUCB
from story_engine.canon.lock import CanonLock
from story_engine.config import EngineConfig
from story_engine.llm.clients import LLMCache, set_global_cache
from story_engine.writing.rhythm import RhythmMixer


def create_llm_cache(cfg: EngineConfig) -> LLMCache:
    cache_cfg = cfg.cost
    cache = LLMCache(
        enable=bool(cache_cfg.cache),
        ttl_hours=float(cache_cfg.cache_ttl_hours),
    )
    set_global_cache(cache)
    return cache


def create_canon_services(cfg: EngineConfig) -> Dict[str, object]:
    canon_cfg = cfg.canon
    lock = CanonLock(canon_cfg.facts, canon_cfg.must_not)
    council = CanonCouncil(
        base_facts=list(canon_cfg.facts),
        commit_threshold=canon_cfg.commit_threshold,
        proposal_k=canon_cfg.proposal_k,
    )
    return {
        "lock": lock,
        "council": council,
        "stage_limits": dict(canon_cfg.max_new_per_stage),
    }


def create_autonomy_services(cfg: EngineConfig) -> Dict[str, object]:
    auto_cfg = cfg.autonomy
    scorer = AutonomyScorer(auto_cfg.reward.to_dict())
    ucb = RhythmUCB(auto_cfg.rhythm_modes)
    return {
        "scorer": scorer,
        "ucb": ucb,
        "enabled": auto_cfg.enabled,
        "rollouts": auto_cfg.rollouts,
        "temp_add": auto_cfg.temp_add,
    }


def create_rhythm_mixer(seed: int = 17) -> RhythmMixer:
    return RhythmMixer(seed=seed)
