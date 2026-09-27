"""JevDecisionEngine — stub / llm_proxy / real switch via ``JEV_MODE``.

JEV is **not** assumed superior to Baseline. Modes:
- ``stub``: deterministic path + instruction cues + stub cost units
- ``llm_proxy``: GT-free role-play of systemone answers (NOT live Jev API)
- ``real``: raises until a credentialed adapter exists
- ``off``: skip the ladder column

``llm_proxy`` is a reproducible assistant stand-in for eval — not real Jev
probs, billing, or a substitute for a live systemone call.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures

JevMode = Literal["stub", "off", "real", "llm_proxy"]

JudgeFn = Callable[[EpisodeFeatures, DecisionConfig], Mapping[str, Any]]

# Stub instruction/task cues (weak — may false-positive, e.g. SafeTask → "safe").
_FAIL_INSTR = ("fail", "failure", "error", "drop", "miss", "stuck", "collide", "slip")
_OK_INSTR = ("success", "successfully", "complete", "completed", "ok", "safe")

# LLM-proxy: richer path semantics + careful instruction phrases (avoid "safe").
_PATH_FAIL_SEM = (
    "fail",
    "failure",
    "gripper_error",
    "position_offset",
    "rotation_offset",
    "wrong_target",
    "misalign",
    "deviation",
    "omission",
    "collide",
    "slip",
    "dropped",
    "stuck",
    "offset",
    "error",
)
_PATH_OK_SEM = (
    "dataset_success",
    "success_cleaned",
    "stack_ok",
    "/success/",
    "_success_",
    "success/",
)
# Instruction-only phrases (do NOT scan task name — SafeTask false friend).
_INSTR_FAIL_PH = (
    "failed",
    "failure",
    "positional deviation",
    "orientation deviation",
    "could not grasp",
    "gripper",
    "slipped",
    "dropped",
    "misalign",
    "wrong object",
    "skipped",
    "omission",
)
_INSTR_OK_PH = (
    "successfully completed",
    "completed successfully",
    "task succeeded",
    "successfully",
)

# Extra path aliases → ontology types (GT-free folder cues).
_PATH_TYPE_ALIASES: tuple[tuple[str, str], ...] = (
    ("position_offset", "position_deviation"),
    ("rotation_offset", "orientation_deviation"),
    ("orient", "orientation_deviation"),
    ("gripper_error", "grasping_error"),
    ("grasp", "grasping_error"),
    ("wrong_target", "wrong_target_object"),
    ("omit", "step_omission"),
    ("omission", "step_omission"),
    ("timing", "timing_error"),
    ("position", "position_deviation"),
)

SYSTEMONE_QUESTIONS: dict[str, dict[str, Any]] = {
    "is_fail": {
        "type": "noul",
        "instructions": "Did this robot episode fail the task?",
        "criteria": {
            "true": "Task failed, dropped, incomplete, or error",
            "false": "Task completed successfully",
        },
    },
    "fail_type": {
        "type": "choice",
        "instructions": "If failed, which failure type best fits?",
        "criteria": {
            "position_deviation": "Wrong position or offset",
            "orientation_deviation": "Wrong orientation/rotation",
            "step_omission": "Skipped a required step",
            "wrong_target_object": "Wrong object targeted",
            "timing_error": "Too early/late",
            "grasping_error": "Grasp/slip/gripper failure",
            "other": "Other or success/unclear",
        },
    },
}


@dataclass(frozen=True)
class CostSnapshot:
    """Proxy cost for ladder cost–quality columns."""

    calls: int = 0
    tokens: int = 0
    stub_units: float = 0.0


def resolve_jev_mode(raw: str | None = None) -> JevMode:
    """Resolve ``JEV_MODE`` env (default ``stub``). Unknown → ``stub``."""
    value = (raw if raw is not None else os.environ.get("JEV_MODE", "stub")).strip().lower()
    if value in ("stub", "off", "real", "llm_proxy"):
        return value  # type: ignore[return-value]
    return "stub"


def features_to_state(features: EpisodeFeatures) -> str:
    """Build GT-free episode text for the ``state`` field (systemone-shaped)."""
    lines = [
        f"episode_id: {features.episode_id}",
        f"task: {features.task or ''}",
        f"instruction: {features.instruction or ''}",
        f"path_text: {features.path_text}",
        f"n_videos: {features.n_videos}",
        f"n_frames: {features.n_frames}",
        f"has_failure_subtask_meta: {features.has_failure_subtask_meta}",
    ]
    return "\n".join(lines)


def build_systemone_payload(features: EpisodeFeatures) -> dict[str, Any]:
    """Request body shape for systemone (used by proxy + future real adapter)."""
    return {
        "state": features_to_state(features),
        "questions": SYSTEMONE_QUESTIONS,
    }


def map_systemone_response(
    payload: Mapping[str, Any],
    *,
    episode_id: str,
    evidence_tag: str = "jev",
) -> tuple[DecisionResult, int]:
    """Map systemone-shaped JSON → DecisionResult + input_tokens.

    ``is_fail.noul`` → failure_probability; predicted_success = noul < 0.5.
    ``fail_type.choice`` → failure_type when predicted fail.
    """
    answers = payload.get("answers") or {}
    is_fail = answers.get("is_fail") or {}
    fail_type_ans = answers.get("fail_type") or {}

    noul_raw = is_fail.get("noul", 0.5)
    try:
        noul = float(noul_raw)
    except (TypeError, ValueError):
        noul = 0.5
    noul = min(1.0, max(0.0, noul))

    predicted_success = noul < 0.5
    confidence = abs(noul - 0.5) * 2.0

    ftype: str | None = None
    if not predicted_success:
        choice = fail_type_ans.get("choice")
        if isinstance(choice, str) and choice:
            ftype = choice
        conf_c = fail_type_ans.get("confidence")
        if conf_c is not None:
            try:
                confidence = max(confidence, float(conf_c))
            except (TypeError, ValueError):
                pass

    usage = payload.get("usage") or {}
    try:
        input_tokens = int(usage.get("input_tokens") or 0)
    except (TypeError, ValueError):
        input_tokens = 0

    evidence = [evidence_tag, f"noul={noul:.3f}"]
    model = payload.get("model")
    if model:
        evidence.append(f"model={model}")
    if ftype:
        evidence.append(f"type={ftype}")

    result = DecisionResult(
        episode_id=episode_id,
        backend="jev",
        predicted_success=predicted_success,
        failure_probability=float(noul),
        failure_type=ftype,
        confidence=float(min(1.0, max(0.0, confidence))),
        needs_deep_review=not predicted_success,
        evidence=tuple(evidence),
    )
    return result, input_tokens


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _soft_noul_from_score(score: float, prior: float) -> float:
    """Map evidence score in [-1, 1] (fail…ok flipped) to soft noul.

    Positive score → more fail; negative → more success. Soft blend with prior.
    """
    # Logistic-ish squash around prior.
    noul = prior + 0.42 * score
    return _clamp01(noul)


def _guess_type_proxy(
    path: str,
    instr: str,
    cfg: DecisionConfig,
    majority_type: str | None,
) -> tuple[str, float]:
    """Return (choice, confidence) from path/instruction keywords only."""
    blob = f"{path} {instr}"
    for alias, ftype in _PATH_TYPE_ALIASES:
        if alias in blob:
            return ftype, 0.78
    for ftype, words in cfg.type_keywords.items():
        if any(w in blob for w in words):
            return ftype, 0.72
    if majority_type:
        return majority_type, 0.45
    return "other", 0.35


def llm_proxy_judge(
    features: EpisodeFeatures,
    config: DecisionConfig,
    *,
    fail_prior: float = 0.5,
    majority_type: str | None = None,
) -> dict[str, Any]:
    """Deterministic GT-free role-play of systemone answers.

    Reads path/instruction/task more carefully than the stub (semantic path
    offsets/errors; avoids SafeTask→\"safe\" false friends). Soft noul, not
    only 0.12/0.88. **Not** the live Jev API.
    """
    path = features.path_text.lower()
    instr = (features.instruction or "").lower()
    # Task name is often a dataset label (SafeTask); do not treat as OK cue.
    state = features_to_state(features)
    approx_tokens = max(8, len(state) // 4)

    hit_ok_path = any(tok in path for tok in config.success_path_tokens) or any(
        tok in path for tok in _PATH_OK_SEM
    )
    hit_fail_literal = any(tok in path for tok in config.fail_path_tokens)
    # Semantic fail folders (offset/error) — weaker than dataset_success_* .
    hit_fail_sem = any(tok in path for tok in _PATH_FAIL_SEM)
    # Strong success dirs win over weak "error"/"offset" subfolder noise
    # (RoboFAC: dataset_success_cleaned/.../stack_error/...). Literal fail
    # token still overrides success nesting (Baseline honesty).
    if hit_ok_path and not hit_fail_literal:
        score = -0.95  # strong success
        evidence_note = "success_path_sem"
    elif hit_fail_literal:
        score = 0.95
        evidence_note = "fail_path_literal"
    elif hit_fail_sem:
        # Softer than literal "fail" — e.g. position_offset folders.
        score = 0.78 if ("offset" in path or "error" in path) else 0.65
        evidence_note = "fail_path_sem"
    else:
        # Instruction phrases only (not task label).
        hit_fail_i = any(p in instr for p in _INSTR_FAIL_PH)
        hit_ok_i = any(p in instr for p in _INSTR_OK_PH)
        if hit_fail_i and not hit_ok_i:
            score = 0.55
            evidence_note = "fail_instr_ph"
        elif hit_ok_i and not hit_fail_i:
            score = -0.55
            evidence_note = "ok_instr_ph"
        else:
            score = 0.0
            evidence_note = "prior"

    if features.has_failure_subtask_meta and score < 0.5:
        score = max(score, 0.55)
        evidence_note = evidence_note + "+subtask_meta"

    noul = _soft_noul_from_score(score, fail_prior)
    # When strong path evidence, snap closer to extremes (still soft).
    if evidence_note.startswith("success_path"):
        noul = min(noul, 0.18)
    elif evidence_note.startswith("fail_path_literal"):
        noul = max(noul, 0.82)
    elif evidence_note.startswith("fail_path_sem"):
        noul = max(noul, 0.68)

    noul = _clamp01(round(noul, 4))
    predicted_fail = noul >= 0.5

    if predicted_fail:
        choice, conf = _guess_type_proxy(path, instr, config, majority_type)
    else:
        choice, conf = "other", max(0.4, 1.0 - noul)

    return {
        "model": "llm_proxy_roleplay",
        "answers": {
            "is_fail": {"type": "noul", "noul": noul},
            "fail_type": {
                "type": "choice",
                "choice": choice,
                "confidence": conf,
            },
        },
        "usage": {"input_tokens": approx_tokens, "output_tokens": 24},
        "proxy_evidence": evidence_note,
    }


class JevDecisionEngine(FastDecisionEngine):
    """Stub / llm_proxy / real switch. Never reads ``ground_truth`` at evaluate.

    Fit() only stores a development fail prior for cue-less episodes.
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
        judge_fn: JudgeFn | None = None,
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
        self._judge_fn = judge_fn
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

    def _charge_stub(self, features: EpisodeFeatures) -> None:
        tok = self.stub_token_overhead + max(1, features.instruction_len // 4)
        self._calls += 1
        self._tokens += tok
        self._stub_units += self.stub_cost_units

    def _charge_proxy(self, input_tokens: int) -> None:
        """llm_proxy: count calls + approx tokens; stub_units stay 0 (no JEV bill)."""
        self._calls += 1
        self._tokens += max(0, int(input_tokens))

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

    def _evaluate_llm_proxy(
        self,
        features: EpisodeFeatures,
        cfg: DecisionConfig,
    ) -> DecisionResult:
        if self._judge_fn is not None:
            raw = dict(self._judge_fn(features, cfg))
        else:
            raw = llm_proxy_judge(
                features,
                cfg,
                fail_prior=self._fail_prior,
                majority_type=self._majority_type,
            )
        result, input_tokens = map_systemone_response(
            raw, episode_id=features.episode_id, evidence_tag="jev_llm_proxy"
        )
        # Attach proxy_evidence if present.
        note = raw.get("proxy_evidence")
        if isinstance(note, str) and note:
            result = DecisionResult(
                episode_id=result.episode_id,
                backend=result.backend,
                predicted_success=result.predicted_success,
                failure_probability=result.failure_probability,
                failure_type=result.failure_type,
                confidence=result.confidence,
                needs_deep_review=result.needs_deep_review,
                evidence=tuple(list(result.evidence) + [note]),
            )
        if input_tokens <= 0:
            input_tokens = max(8, len(features_to_state(features)) // 4)
        self._charge_proxy(input_tokens)
        return result

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        if self.mode == "real":
            raise RuntimeError(
                "JEV real adapter not implemented on this branch. Set "
                "JEV_MODE=stub, JEV_MODE=llm_proxy, or JEV_MODE=off."
            )
        if self.mode == "off":
            raise RuntimeError(
                "JevDecisionEngine constructed with mode=off; harness should omit it."
            )

        cfg = decision_schema or self.config
        if self.mode == "llm_proxy":
            return self._evaluate_llm_proxy(features, cfg)

        self._charge_stub(features)
        evidence: list[str] = ["jev_stub"]

        path = self._path_cue(features.path_text, cfg)
        instr = self._instr_cue(features)

        if path is not None:
            predicted_success = path
            p_fail = 0.12 if path else 0.88
            confidence = 0.88
            evidence.append("success_path_cue" if path else "fail_path_cue")
        elif instr is not None:
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
