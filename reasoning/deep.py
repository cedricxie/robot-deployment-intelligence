"""DeepReasoner behind VisionReasoningProvider (mock default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decision.base import DecisionConfig, DecisionResult
from decision.jev import CostSnapshot
from features.extract import EpisodeFeatures
from reasoning.providers import MockVisionReasoningProvider, VisionReasoningProvider

DEFAULT_PROMPT = (
    "Summarize whether this robot episode failed and which failure type fits."
)


@dataclass(frozen=True)
class StructuredDiagnosis:
    """Plan §6.2 DeepReasoner.reason output (+ raw / model / prompt versions)."""

    episode_id: str
    predicted_success: bool
    failure_probability: float
    failure_type: str | None
    confidence: float
    diagnosis: str
    evidence: tuple[str, ...] = ()
    raw: str = ""
    model: str = "mock"
    prompt_version: str = "v0"
    recommended_actions: tuple[dict[str, Any], ...] = ()


class DeepReasoner:
    """Strong path reserved for routed / all-deep cases. Mockable provider."""

    name = "deep"

    def __init__(
        self,
        provider: VisionReasoningProvider | None = None,
        *,
        decision_config: DecisionConfig | None = None,
        prompt: str = DEFAULT_PROMPT,
        prompt_version: str = "v0",
        cost_units_per_call: float = 50.0,
    ) -> None:
        self.provider = provider or MockVisionReasoningProvider(
            cost_units_per_call=min(cost_units_per_call, 5.0)
        )
        self.config = decision_config or DecisionConfig()
        self.prompt = prompt
        self.prompt_version = prompt_version
        self.cost_units_per_call = float(cost_units_per_call)
        self._calls = 0
        self._tokens = 0
        self._stub_units = 0.0

    def reset_cost(self) -> None:
        self._calls = 0
        self._tokens = 0
        self._stub_units = 0.0
        reset = getattr(self.provider, "reset_cost", None)
        if callable(reset):
            reset()

    def cost_snapshot(self) -> CostSnapshot:
        return CostSnapshot(
            calls=self._calls,
            tokens=self._tokens,
            stub_units=self._stub_units,
        )

    def reason(
        self,
        context: EpisodeFeatures,
        *,
        frames: list[str] | None = None,
        prior: DecisionResult | None = None,
    ) -> StructuredDiagnosis:
        """Produce a structured diagnosis from features (+ optional prior)."""
        if frames:
            raw = self.provider.summarize(frames, self.prompt)
        elif isinstance(self.provider, MockVisionReasoningProvider):
            raw = self.provider.summarize_features(context, self.prompt)
        else:
            raw = self.provider.summarize(
                [context.path_text, context.instruction or ""], self.prompt
            )

        # Provider may already charge; Deep adds deep-tier units on top.
        self._calls += 1
        extra_tokens = max(16, (context.instruction_len // 4) + 24)
        self._tokens += extra_tokens
        self._stub_units += self.cost_units_per_call

        predicted_success, p_fail, conf, ftype, evidence = self._parse(
            raw, context, prior
        )
        diagnosis = raw.strip() or "no diagnosis"
        return StructuredDiagnosis(
            episode_id=context.episode_id,
            predicted_success=predicted_success,
            failure_probability=p_fail,
            failure_type=ftype,
            confidence=conf,
            diagnosis=diagnosis,
            evidence=tuple(evidence),
            raw=raw,
            model=getattr(self.provider, "name", "provider"),
            prompt_version=self.prompt_version,
        )

    def to_decision(
        self,
        diag: StructuredDiagnosis,
        *,
        backend: str = "baseline",
    ) -> DecisionResult:
        return DecisionResult(
            episode_id=diag.episode_id,
            backend=backend,  # type: ignore[arg-type]
            predicted_success=diag.predicted_success,
            failure_probability=diag.failure_probability,
            failure_type=diag.failure_type,
            confidence=diag.confidence,
            needs_deep_review=False,
            evidence=diag.evidence + ("deep_reasoner",),
        )

    def _parse(
        self,
        raw: str,
        features: EpisodeFeatures,
        prior: DecisionResult | None,
    ) -> tuple[bool, float, float, str | None, list[str]]:
        text = raw.lower()
        evidence: list[str] = ["deep"]
        cfg = self.config

        if "looks failed" in text or "lean fail" in text:
            predicted_success = False
            p_fail = 0.9
            conf = 0.9
            evidence.append("mock_fail_summary")
        elif "looks successful" in text:
            predicted_success = True
            p_fail = 0.1
            conf = 0.9
            evidence.append("mock_ok_summary")
        elif prior is not None:
            predicted_success = prior.predicted_success
            p_fail = prior.failure_probability
            conf = max(prior.confidence, 0.7)
            evidence.append("prior_fast")
        else:
            # Fall back to path cues on features.
            hit_fail = any(t in features.path_text for t in cfg.fail_path_tokens)
            hit_ok = any(t in features.path_text for t in cfg.success_path_tokens)
            if hit_fail and not hit_ok:
                predicted_success, p_fail, conf = False, 0.88, 0.85
            elif hit_ok and not hit_fail:
                predicted_success, p_fail, conf = True, 0.12, 0.85
            else:
                predicted_success, p_fail, conf = True, 0.45, 0.55
            evidence.append("path_fallback")

        ftype: str | None = None
        if not predicted_success:
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
            for ft, words in cfg.type_keywords.items():
                if any(w in blob for w in words):
                    ftype = ft
                    evidence.append(f"type={ft}")
                    break
            if ftype is None and prior is not None:
                ftype = prior.failure_type
        return predicted_success, float(p_fail), float(conf), ftype, evidence
