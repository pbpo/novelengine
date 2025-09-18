from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, replace
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from .models import EngineConfig, RewardWeights


def _merge_dict(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | None) -> EngineConfig:
    """Load configuration file into EngineConfig dataclass."""

    cfg = EngineConfig()
    if not path:
        return cfg
    if not os.path.exists(path):
        return cfg
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        if yaml is None:
            raise RuntimeError("PyYAML not installed; cannot load non-JSON config")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        return cfg

    base_dict = cfg.as_dict()
    merged = _merge_dict(base_dict, raw)

    reward = RewardWeights.from_dict(merged.get("autonomy", {}).get("reward", {}))
    merged.setdefault("autonomy", {})["reward"] = reward.to_dict()

    return _dict_to_config(merged)


def _dict_to_config(data: Dict[str, Any]) -> EngineConfig:
    cfg = EngineConfig()
    cfg.embedding_model = data.get("embedding_model", cfg.embedding_model)
    cfg.model = data.get("model", cfg.model)
    cfg.planner_model_env = data.get("planner_model_env", cfg.planner_model_env)
    cfg.reviewer_model_env = data.get("reviewer_model_env", cfg.reviewer_model_env)
    cfg.beats = data.get("beats", cfg.beats)
    cfg.gen_temp_by_mode = data.get("gen_temp_by_mode", cfg.gen_temp_by_mode)

    anti = data.get("anti_repeat", {})
    cfg.anti_repeat = replace(cfg.anti_repeat, **anti)

    focus = data.get("focus", {})
    cfg.focus = replace(cfg.focus, **focus)

    review = data.get("review", {})
    fail_on = review.get("fail_on")
    if isinstance(fail_on, list):
        cfg.review.fail_on = [str(item) for item in fail_on if item]

    pov = data.get("pov_fixed", {})
    cfg.pov_fixed = replace(cfg.pov_fixed, **{k: v for k, v in pov.items() if k in {"mode", "name"}})

    canon = data.get("canon", {})
    cfg.canon.facts = [str(f) for f in canon.get("facts", cfg.canon.facts)]
    cfg.canon.must_not = [str(f) for f in canon.get("must_not", cfg.canon.must_not)]
    cfg.canon.proposal_k = int(canon.get("proposal_k", cfg.canon.proposal_k))
    cfg.canon.commit_threshold = int(canon.get("commit_threshold", cfg.canon.commit_threshold))
    cfg.canon.max_new_per_stage = dict(
        canon.get("max_new_per_stage", cfg.canon.max_new_per_stage)
    )

    autonomy = data.get("autonomy", {})
    cfg.autonomy.enabled = bool(autonomy.get("enabled", cfg.autonomy.enabled))
    cfg.autonomy.rollouts = int(autonomy.get("rollouts", cfg.autonomy.rollouts))
    cfg.autonomy.temp_add = float(autonomy.get("temp_add", cfg.autonomy.temp_add))
    cfg.autonomy.rhythm_ucb = bool(autonomy.get("rhythm_ucb", cfg.autonomy.rhythm_ucb))
    cfg.autonomy.rhythm_modes = list(
        autonomy.get("rhythm_modes", cfg.autonomy.rhythm_modes)
    )
    reward = autonomy.get("reward")
    if isinstance(reward, dict):
        cfg.autonomy.reward = RewardWeights.from_dict(reward)

    cfg.htnr = dict(data.get("htnr", cfg.htnr))

    expansion = data.get("expansion", {})
    cfg.expansion = replace(
        cfg.expansion,
        enabled=bool(expansion.get("enabled", cfg.expansion.enabled)),
        seg_tokens_mul=float(expansion.get("seg_tokens_mul", cfg.expansion.seg_tokens_mul)),
        segments_add=dict(expansion.get("segments_add", cfg.expansion.segments_add)),
        micro_per_segment=int(expansion.get("micro_per_segment", cfg.expansion.micro_per_segment)),
        temp_add=float(expansion.get("temp_add", cfg.expansion.temp_add)),
        unlock_scope=bool(expansion.get("unlock_scope", cfg.expansion.unlock_scope)),
    )

    cost = data.get("cost", {})
    cfg.cost = replace(cfg.cost, **{k: cost.get(k, getattr(cfg.cost, k)) for k in asdict(cfg.cost)})

    loop = data.get("loop_guard", {})
    cfg.loop_guard = replace(
        cfg.loop_guard,
        min_gap=int(loop.get("min_gap", cfg.loop_guard.min_gap)),
        max_per_arc=int(loop.get("max_per_arc", cfg.loop_guard.max_per_arc)),
    )

    scope_lock = data.get("scope_lock", cfg.scope_lock)
    if isinstance(scope_lock, dict):
        cfg.scope_lock = dict(scope_lock)

    pace = data.get("pace", {})
    cfg.pace = replace(
        cfg.pace,
        max_events_per_scene=int(pace.get("max_events_per_scene", cfg.pace.max_events_per_scene)),
    )

    return cfg
