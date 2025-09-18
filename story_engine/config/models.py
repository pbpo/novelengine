from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(slots=True)
class AntiRepeatConfig:
    window: int = 8
    shingle: int = 5
    jaccard: float = 0.30
    cos: float = 0.92


@dataclass(slots=True)
class FocusConfig:
    max_core_chars: int = 3
    max_new_facts: int = 2


@dataclass(slots=True)
class ReviewConfig:
    fail_on: List[str] = field(
        default_factory=lambda: [
            "info_dump",
            "tone_shift",
            "continuity_break",
            "repetition",
            "ooc",
            "pov_shift",
            "passive_protagonist",
        ]
    )


@dataclass(slots=True)
class POVConfig:
    mode: str = "third_limited"
    name: Optional[str] = None


@dataclass(slots=True)
class CanonStageLimits:
    SETUP: int = 0
    ASCENT: int = 1
    CONVERGE: int = 2
    RESOLVE: int = 1

    def as_dict(self) -> Dict[str, int]:
        return {
            "SETUP": self.SETUP,
            "ASCENT": self.ASCENT,
            "CONVERGE": self.CONVERGE,
            "RESOLVE": self.RESOLVE,
        }


@dataclass(slots=True)
class CanonConfig:
    facts: List[str] = field(default_factory=list)
    must_not: List[str] = field(default_factory=list)
    proposal_k: int = 2
    commit_threshold: int = 2
    max_new_per_stage: Dict[str, int] = field(default_factory=lambda: CanonStageLimits().as_dict())


@dataclass(slots=True)
class RewardWeights:
    novelty: float = 1.0
    qc: float = 0.8
    pace: float = 0.6
    transition: float = 0.3
    sense_fit: float = 0.2
    stc_hit: float = 0.4

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "RewardWeights":
        base = cls()
        for key, value in (data or {}).items():
            attr = key[2:] if key.startswith("w_") else key
            if hasattr(base, attr):
                setattr(base, attr, float(value))
        return base

    def to_dict(self) -> Dict[str, float]:
        return {
            "novelty": self.novelty,
            "qc": self.qc,
            "pace": self.pace,
            "transition": self.transition,
            "sense_fit": self.sense_fit,
            "stc_hit": self.stc_hit,
        }


@dataclass(slots=True)
class AutonomyConfig:
    enabled: bool = True
    rollouts: int = 3
    temp_add: float = 0.06
    rhythm_ucb: bool = True
    rhythm_modes: List[str] = field(default_factory=lambda: [
        "image_lead",
        "dialogue_heavy",
        "inner_lead",
        "periodic",
    ])
    reward: RewardWeights = field(default_factory=RewardWeights)


@dataclass(slots=True)
class ExpansionConfig:
    enabled: bool = False
    seg_tokens_mul: float = 1.0
    segments_add: Dict[str, int] = field(default_factory=dict)
    micro_per_segment: int = 1
    temp_add: float = 0.0
    unlock_scope: bool = False


@dataclass(slots=True)
class CostConfig:
    cache: bool = True
    cache_ttl_hours: int = 168
    qc_every: int = 3
    analyst_every: int = 2
    entity_extract_every: int = 3
    persona_max_per_ch: int = 1


@dataclass(slots=True)
class LoopGuardConfig:
    min_gap: int = 2
    max_per_arc: int = 3


@dataclass(slots=True)
class PaceConfig:
    max_events_per_scene: int = 2


@dataclass(slots=True)
class EngineConfig:
    embedding_model: str = "text-embedding-3-small"
    model: str = "claude-sonnet-4-20250514"
    planner_model_env: str = "OAI_PLANNER_MODEL"
    reviewer_model_env: str = "OAI_REVIEWER_MODEL"
    beats: List[List[str]] = field(
        default_factory=lambda: [
            ["도입", "촉발 사건", "1차 목표"],
            ["장애물", "중간 고비"],
            ["진실 접근", "최종 대비"],
            ["결전", "여운"],
        ]
    )
    gen_temp_by_mode: Dict[str, float] = field(
        default_factory=lambda: {
            "dialog-driven": 0.90,
            "tight-action": 0.85,
            "quiet-slice": 0.75,
        }
    )
    anti_repeat: AntiRepeatConfig = field(default_factory=AntiRepeatConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    pov_fixed: POVConfig = field(default_factory=POVConfig)
    canon: CanonConfig = field(default_factory=CanonConfig)
    autonomy: AutonomyConfig = field(default_factory=AutonomyConfig)
    htnr: Dict[str, int] = field(default_factory=lambda: {
        "enabled": 1,
        "max_leaves": 16,
        "leaf_sample": 32,
    })
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)
    scope_lock: Dict[str, object] = field(
        default_factory=lambda: {"locked": ["family"], "unban_stage": "CONVERGE"}
    )
    pace: PaceConfig = field(default_factory=PaceConfig)

    def as_dict(self) -> Dict[str, object]:
        """Convert to a nested dict roughly compatible with the legacy config"""
        return {
            "embedding_model": self.embedding_model,
            "model": self.model,
            "planner_model_env": self.planner_model_env,
            "reviewer_model_env": self.reviewer_model_env,
            "beats": self.beats,
            "gen_temp_by_mode": self.gen_temp_by_mode,
            "anti_repeat": vars(self.anti_repeat),
            "focus": vars(self.focus),
            "review": {"fail_on": list(self.review.fail_on)},
            "pov_fixed": {"mode": self.pov_fixed.mode, "name": self.pov_fixed.name},
            "canon": {
                "facts": list(self.canon.facts),
                "must_not": list(self.canon.must_not),
                "proposal_k": self.canon.proposal_k,
                "commit_threshold": self.canon.commit_threshold,
                "max_new_per_stage": dict(self.canon.max_new_per_stage),
            },
            "autonomy": {
                "enabled": self.autonomy.enabled,
                "rollouts": self.autonomy.rollouts,
                "temp_add": self.autonomy.temp_add,
                "rhythm_ucb": self.autonomy.rhythm_ucb,
                "rhythm_modes": list(self.autonomy.rhythm_modes),
                "reward": self.autonomy.reward.to_dict(),
            },
            "htnr": dict(self.htnr),
            "expansion": {
                "enabled": self.expansion.enabled,
                "seg_tokens_mul": self.expansion.seg_tokens_mul,
                "segments_add": dict(self.expansion.segments_add),
                "micro_per_segment": self.expansion.micro_per_segment,
                "temp_add": self.expansion.temp_add,
                "unlock_scope": self.expansion.unlock_scope,
            },
            "cost": vars(self.cost),
            "loop_guard": vars(self.loop_guard),
            "scope_lock": dict(self.scope_lock),
            "pace": {"max_events_per_scene": self.pace.max_events_per_scene},
        }
