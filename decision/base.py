"""FastDecisionEngine protocol + shared DecisionResult / config."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from data.schemas.episode import DecisionBackend
from features.extract import EpisodeFeatures

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decision.json"


@dataclass(frozen=True)
class DecisionConfig:
    """Knobs for ladder engines (no magic numbers in evaluate())."""

    random_seed: int = 42
    review_top_k: int = 5
    fail_path_tokens: tuple[str, ...] = ("fail", "failure")
    success_path_tokens: tuple[str, ...] = ("success", "success_cleaned", "stack_ok")
    type_keywords: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "position_deviation": ("position", "pos", "lateral", "offset"),
            "orientation_deviation": ("orientation", "orient", "rotation"),
            "step_omission": ("omission", "omit", "skipped", "missing_step"),
            "wrong_target_object": ("wrong_target", "wrong_object"),
            "timing_error": ("timing", "early", "late"),
            "grasping_error": ("grasp", "gripper", "slip"),
        }
    )
    default_fail_probability: float = 0.5
    # JEV stub cost–quality knobs (ignored by cheaper ladder rungs).
    jev_stub_cost_units: float = 10.0
    jev_stub_token_overhead: int = 16

    def __post_init__(self) -> None:
        if self.review_top_k < 1:
            raise ValueError("review_top_k must be >= 1")
        if not 0.0 <= self.default_fail_probability <= 1.0:
            raise ValueError("default_fail_probability must be in [0, 1]")
        if self.jev_stub_cost_units < 0:
            raise ValueError("jev_stub_cost_units must be >= 0")
        if self.jev_stub_token_overhead < 0:
            raise ValueError("jev_stub_token_overhead must be >= 0")


def load_decision_config(path: str | Path | None = None) -> DecisionConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return DecisionConfig()
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    known = {f.name for f in DecisionConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in known or k.startswith("_"):
            continue
        if k in ("fail_path_tokens", "success_path_tokens") and isinstance(v, list):
            kwargs[k] = tuple(v)
        elif k == "type_keywords" and isinstance(v, dict):
            kwargs[k] = {str(tk): tuple(words) for tk, words in v.items()}
        else:
            kwargs[k] = v
    return DecisionConfig(**kwargs)


def config_to_dict(cfg: DecisionConfig) -> dict:
    return asdict(cfg)


@dataclass(frozen=True)
class DecisionResult:
    """Prediction side-channel (attach to ModelOutput via helpers if needed)."""

    episode_id: str
    backend: DecisionBackend
    # success=True → predict OK; success=False → predict fail
    predicted_success: bool
    failure_probability: float
    failure_type: str | None
    confidence: float
    needs_deep_review: bool
    evidence: tuple[str, ...] = ()


class FastDecisionEngine(ABC):
    """Plan §6.2: evaluate(features, decision_schema) → DecisionResult."""

    name: str
    backend: DecisionBackend

    @abstractmethod
    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        ...

    def evaluate_many(
        self,
        features_list: list[EpisodeFeatures],
        decision_schema: DecisionConfig | None = None,
    ) -> list[DecisionResult]:
        return [self.evaluate(f, decision_schema) for f in features_list]
