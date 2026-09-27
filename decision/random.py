"""Seeded Random ladder rung — weak baseline."""

from __future__ import annotations

import random
from typing import Sequence

from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures

# Fallback type pool when no reference types provided.
_DEFAULT_TYPES: tuple[str, ...] = (
    "position_deviation",
    "step_omission",
    "grasping_error",
)


class RandomDecisionEngine(FastDecisionEngine):
    """Independent Bernoulli success + uniform type; seeded for reproducibility."""

    name = "random"
    backend = "random"  # type: ignore[assignment]

    def __init__(
        self,
        config: DecisionConfig | None = None,
        failure_types: Sequence[str] | None = None,
        fail_prior: float | None = None,
    ) -> None:
        self.config = config or DecisionConfig()
        self._types = list(failure_types) if failure_types else list(_DEFAULT_TYPES)
        self._fail_prior = (
            fail_prior if fail_prior is not None else self.config.default_fail_probability
        )
        self._rng = random.Random(self.config.random_seed)

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        cfg = decision_schema or self.config
        # Re-seed per episode_id so order-independent + reproducible.
        rng = random.Random(f"{cfg.random_seed}:{features.episode_id}")
        is_fail = rng.random() < self._fail_prior
        p_fail = float(is_fail)
        ftype = rng.choice(self._types) if is_fail and self._types else None
        return DecisionResult(
            episode_id=features.episode_id,
            backend="random",
            predicted_success=not is_fail,
            failure_probability=p_fail,
            failure_type=ftype,
            confidence=0.5,
            needs_deep_review=is_fail,
            evidence=("random_draw",),
        )
