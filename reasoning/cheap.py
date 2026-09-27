"""Cheap path: path-cue only success/fail (no type scan, no priors)."""

from __future__ import annotations

from decision.base import DecisionConfig, DecisionResult
from decision.jev import CostSnapshot
from features.extract import EpisodeFeatures


class CheapPathEngine:
    """Lowest rung: strong path tokens only; abstains (low conf) otherwise."""

    name = "cheap"
    backend = "baseline"  # type: ignore[assignment]

    def __init__(
        self,
        config: DecisionConfig | None = None,
        *,
        cost_units_per_call: float = 0.1,
    ) -> None:
        self.config = config or DecisionConfig()
        self.cost_units_per_call = float(cost_units_per_call)
        self._calls = 0
        self._stub_units = 0.0

    def reset_cost(self) -> None:
        self._calls = 0
        self._stub_units = 0.0

    def cost_snapshot(self) -> CostSnapshot:
        return CostSnapshot(calls=self._calls, tokens=0, stub_units=self._stub_units)

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        cfg = decision_schema or self.config
        self._calls += 1
        self._stub_units += self.cost_units_per_call
        text = features.path_text
        hit_fail = any(tok in text for tok in cfg.fail_path_tokens)
        hit_ok = any(tok in text for tok in cfg.success_path_tokens)

        if hit_fail and not hit_ok:
            return DecisionResult(
                episode_id=features.episode_id,
                backend="baseline",
                predicted_success=False,
                failure_probability=0.9,
                failure_type=None,
                confidence=0.9,
                needs_deep_review=False,
                evidence=("cheap_fail_path_cue",),
            )
        if hit_ok and not hit_fail:
            return DecisionResult(
                episode_id=features.episode_id,
                backend="baseline",
                predicted_success=True,
                failure_probability=0.1,
                failure_type=None,
                confidence=0.9,
                needs_deep_review=False,
                evidence=("cheap_success_path_cue",),
            )
        if hit_fail and hit_ok:
            return DecisionResult(
                episode_id=features.episode_id,
                backend="baseline",
                predicted_success=False,
                failure_probability=0.7,
                failure_type=None,
                confidence=0.55,
                needs_deep_review=True,
                evidence=("cheap_mixed_path_cue",),
            )
        # Abstain: low confidence so cascade escalates to fast.
        return DecisionResult(
            episode_id=features.episode_id,
            backend="baseline",
            predicted_success=True,
            failure_probability=0.5,
            failure_type=None,
            confidence=0.35,
            needs_deep_review=True,
            evidence=("cheap_abstain",),
        )
