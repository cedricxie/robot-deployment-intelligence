"""JevDecisionEngine — stub/real switch via ``JEV_MODE`` (cost–quality rung).

JEV is **not** assumed superior to Baseline. The stub is deterministic and
GT-free (path + instruction cues only). ``real`` raises a clear error until a
credentialed adapter exists. ``off`` skips the ladder column.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures

JevMode = Literal["stub", "off", "real"]

# Extra instruction/task cues (GT-free). Weakly informed — may false-positive.
_FAIL_INSTR = ("fail", "failure", "error", "drop", "miss", "stuck", "collide", "slip")
_OK_INSTR = ("success", "successfully", "complete", "completed", "ok", "safe")


@dataclass(frozen=True)
class CostSnapshot:
    """Proxy cost for ladder cost–quality columns."""

    calls: int = 0
    tokens: int = 0
    stub_units: float = 0.0


def resolve_jev_mode(raw: str | None = None) -> JevMode:
    """Resolve ``JEV_MODE`` env (default ``stub``). Unknown → ``stub``."""
    value = (raw if raw is not None else os.environ.get("JEV_MODE", "stub")).strip().lower()
    if value in ("stub", "off", "real"):
        return value  # type: ignore[return-value]
    return "stub"


class JevDecisionEngine(FastDecisionEngine):
    """Stub mimics a dearer look: path cues + instruction scan + cost units.

    Never reads ``ground_truth``. Fit() only stores a development fail prior
    for cue-less episodes (same honesty bar as Baseline).
    """

    name = "jev"
    backend = "jev"  # type: ignore[assignment]

    def __init__(
        self,
        config: DecisionConfig | None = None,
        *,
        mode: JevMode | None = None,
        stub_cost_units: float | None = None,
        stub_token_overhead: int | None = None,
    ) -> None:
        self.config = config or DecisionConfig()
        self.mode: JevMode = mode if mode is not None else resolve_jev_mode()
        self.stub_cost_units = (
            float(stub_cost_units)
            if stub_cost_units is not None
            else float(getattr(self.config, "jev_stub_cost_units", 10.0))
        )
        self.stub_token_overhead = (
            int(stub_token_overhead)
            if stub_token_overhead is not None
            else int(getattr(self.config, "jev_stub_token_overhead", 16))
        )
        self._fail_prior = self.config.default_fail_probability
        self._majority_type: str | None = None
        self._calls = 0
        self._tokens = 0
        self._stub_units = 0.0

    def fit(self, reference: Iterable[Episode]) -> JevDecisionEngine:
        n = n_fail = 0
        from collections import Counter

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
        return self

    def reset_cost(self) -> None:
        self._calls = 0
        self._tokens = 0
        self._stub_units = 0.0

    def cost_snapshot(self) -> CostSnapshot:
        return CostSnapshot(
            calls=self._calls, tokens=self._tokens, stub_units=self._stub_units
        )

    def _charge(self, features: EpisodeFeatures) -> None:
        tok = self.stub_token_overhead + max(1, features.instruction_len // 4)
        self._calls += 1
        self._tokens += tok
        self._stub_units += self.stub_cost_units

    def _path_cue(self, text: str, cfg: DecisionConfig) -> bool | None:
        hit_fail = any(tok in text for tok in cfg.fail_path_tokens)
        hit_ok = any(tok in text for tok in cfg.success_path_tokens)
        if hit_fail and not hit_ok:
            return False
        if hit_ok and not hit_fail:
            return True
        if hit_fail and hit_ok:
            return False
        return None

    def _instr_cue(self, features: EpisodeFeatures) -> bool | None:
        blob = " ".join(
            filter(
                None,
                [
                    (features.instruction or "").lower(),
                    (features.task or "").lower(),
                ],
            )
        )
        if not blob.strip():
            return None
        hit_fail = any(w in blob for w in _FAIL_INSTR)
        hit_ok = any(w in blob for w in _OK_INSTR)
        if hit_fail and not hit_ok:
            return False
        if hit_ok and not hit_fail:
            return True
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
        if self.mode == "real":
            raise RuntimeError(
                "JEV real adapter not implemented. Set JEV_MODE=stub (default) "
                "or JEV_MODE=off to skip the JEV ladder column."
            )
        if self.mode == "off":
            raise RuntimeError(
                "JevDecisionEngine constructed with mode=off; harness should omit it."
            )

        cfg = decision_schema or self.config
        self._charge(features)
        evidence: list[str] = ["jev_stub"]

        path = self._path_cue(features.path_text, cfg)
        instr = self._instr_cue(features)

        if path is not None:
            predicted_success = path
            p_fail = 0.12 if path else 0.88
            confidence = 0.88
            evidence.append("success_path_cue" if path else "fail_path_cue")
        elif instr is not None:
            # Weaker than path — stub "pays" for an instruction look.
            predicted_success = instr
            p_fail = 0.28 if instr else 0.72
            confidence = 0.72
            evidence.append("success_instr_cue" if instr else "fail_instr_cue")
        else:
            predicted_success = self._fail_prior < 0.5
            p_fail = self._fail_prior
            confidence = max(p_fail, 1.0 - p_fail)
            evidence.append(f"prior_fail={self._fail_prior:.3f}")

        if features.has_failure_subtask_meta and predicted_success:
            predicted_success = False
            p_fail = max(p_fail, 0.78)
            confidence = max(confidence, 0.72)
            evidence.append("failure_subtask_meta")

        ftype = None if predicted_success else self._guess_type(features, cfg)
        if ftype:
            evidence.append(f"type={ftype}")

        return DecisionResult(
            episode_id=features.episode_id,
            backend="jev",
            predicted_success=predicted_success,
            failure_probability=float(p_fail),
            failure_type=ftype,
            confidence=float(confidence),
            needs_deep_review=not predicted_success,
            evidence=tuple(evidence),
        )

    def evaluate_many(
        self,
        features_list: list[EpisodeFeatures],
        decision_schema: DecisionConfig | None = None,
    ) -> list[DecisionResult]:
        self.reset_cost()
        return [self.evaluate(f, decision_schema) for f in features_list]


def build_jev_engine(
    config: DecisionConfig | None = None,
    *,
    mode: JevMode | None = None,
) -> JevDecisionEngine | None:
    """Factory: ``None`` when mode is ``off``; else a ``JevDecisionEngine``."""
    resolved = mode if mode is not None else resolve_jev_mode()
    if resolved == "off":
        return None
    return JevDecisionEngine(config, mode=resolved)
