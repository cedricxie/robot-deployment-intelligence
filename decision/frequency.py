"""Frequency-only rung: empirical majority success/fail + most-common fail type.

Fit on a **development** (or other non-hidden) reference corpus only.
At predict time, every episode gets the same majority-class prediction —
no per-episode features. Documented weak baseline for the ladder.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures


class FrequencyOnlyDecisionEngine(FastDecisionEngine):
    """Predict most common outcome from reference empirical frequencies."""

    name = "frequency"
    backend = "frequency"  # type: ignore[assignment]

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self._fitted = False
        self._predict_fail = True
        self._fail_prior = self.config.default_fail_probability
        self._majority_type: str | None = None
        self._n_ref = 0
        self._n_fail = 0

    def fit(self, reference: Iterable[Episode]) -> FrequencyOnlyDecisionEngine:
        """Learn fail prior + majority failure_type from reference GT labels.

        Reference should be development (or public_eval for analysis) — never
        hidden_eval in improvement-loop callers.
        """
        n = n_fail = 0
        type_counts: Counter[str] = Counter()
        for ep in reference:
            gt = ep.ground_truth
            if gt.success is None:
                continue
            n += 1
            if gt.success is False:
                n_fail += 1
                if gt.failure_type:
                    type_counts[gt.failure_type] += 1
        self._n_ref = n
        self._n_fail = n_fail
        self._fail_prior = (n_fail / n) if n else self.config.default_fail_probability
        # Majority class for success/fail: predict fail iff fail_prior >= 0.5
        self._predict_fail = self._fail_prior >= 0.5
        self._majority_type = type_counts.most_common(1)[0][0] if type_counts else None
        self._fitted = True
        return self

    @property
    def fail_prior(self) -> float:
        return self._fail_prior

    @property
    def majority_failure_type(self) -> str | None:
        return self._majority_type

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        cfg = decision_schema or self.config
        if not self._fitted:
            # Unfitted → use config prior as constant predictor.
            predict_fail = cfg.default_fail_probability >= 0.5
            p = cfg.default_fail_probability
            ftype = None
            evidence = ("frequency_unfitted_prior",)
        else:
            predict_fail = self._predict_fail
            p = self._fail_prior
            ftype = self._majority_type if predict_fail else None
            evidence = (
                f"fail_prior={self._fail_prior:.3f}",
                f"n_ref={self._n_ref}",
                f"majority_type={self._majority_type}",
            )
        return DecisionResult(
            episode_id=features.episode_id,
            backend="frequency",
            predicted_success=not predict_fail,
            failure_probability=float(p if predict_fail else 1.0 - p),
            failure_type=ftype,
            confidence=max(p, 1.0 - p),
            needs_deep_review=predict_fail,
            evidence=evidence,
        )
