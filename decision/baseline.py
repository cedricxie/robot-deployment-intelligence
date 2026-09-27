"""BaselineFastDecisionEngine — default ship path (GT-free heuristics).

Uses path/id keyword cues + optional development priors. Never reads
ground_truth at evaluate() time.

Beats Random on RoboFAC-shaped fixtures where paths contain success/fail
tokens. Miss modes (document in PR): renamed paths without cues; missing
type keywords → falls back to frequency prior / majority type.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures


class BaselineFastDecisionEngine(FastDecisionEngine):
    name = "baseline"
    backend = "baseline"  # type: ignore[assignment]

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()
        self._fail_prior = self.config.default_fail_probability
        self._majority_type: str | None = None
        self._fitted = False

    def fit(self, reference: Iterable[Episode]) -> BaselineFastDecisionEngine:
        """Optional: learn fail prior + majority type for cue-less episodes."""
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
        if n:
            self._fail_prior = n_fail / n
        self._majority_type = type_counts.most_common(1)[0][0] if type_counts else None
        self._fitted = True
        return self

    def _path_success_cue(self, text: str, cfg: DecisionConfig) -> bool | None:
        """Return True/False if a strong path cue fires, else None."""
        hit_fail = any(tok in text for tok in cfg.fail_path_tokens)
        hit_ok = any(tok in text for tok in cfg.success_path_tokens)
        if hit_fail and not hit_ok:
            return False
        if hit_ok and not hit_fail:
            return True
        if hit_fail and hit_ok:
            # Prefer fail token if both (fail_* often nested under success dirs)
            return False
        return None

    def _guess_type(self, features: EpisodeFeatures, cfg: DecisionConfig) -> str | None:
        blob = " ".join(
            filter(
                None,
                [
                    features.path_text,
                    (features.instruction or "").lower(),
                    (features.task or "").lower(),
                ],
            )
        )
        for ftype, words in cfg.type_keywords.items():
            if any(w in blob for w in words):
                return ftype
        return self._majority_type

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        cfg = decision_schema or self.config
        evidence: list[str] = []
        cue = self._path_success_cue(features.path_text, cfg)

        if cue is True:
            predicted_success = True
            p_fail = 0.15
            confidence = 0.85
            evidence.append("success_path_cue")
        elif cue is False:
            predicted_success = False
            p_fail = 0.85
            confidence = 0.85
            evidence.append("fail_path_cue")
        else:
            # No path cue → development prior
            predicted_success = self._fail_prior < 0.5
            p_fail = self._fail_prior
            confidence = max(p_fail, 1.0 - p_fail)
            evidence.append(f"prior_fail={self._fail_prior:.3f}")

        # Weak meta boost: failure_subtask in metadata suggests fail trajectory
        if features.has_failure_subtask_meta and predicted_success:
            predicted_success = False
            p_fail = max(p_fail, 0.75)
            confidence = max(confidence, 0.7)
            evidence.append("failure_subtask_meta")

        ftype = None if predicted_success else self._guess_type(features, cfg)
        if ftype:
            evidence.append(f"type={ftype}")

        return DecisionResult(
            episode_id=features.episode_id,
            backend="baseline",
            predicted_success=predicted_success,
            failure_probability=float(p_fail),
            failure_type=ftype,
            confidence=float(confidence),
            needs_deep_review=not predicted_success,
            evidence=tuple(evidence),
        )
