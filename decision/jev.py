"""JevDecisionEngine — stub/real switch via ``JEV_MODE`` (cost–quality rung).

JEV is **not** assumed superior to Baseline. The stub is deterministic and
GT-free (path + instruction cues only). ``real`` POSTs to Jev Agent
``/api/v1/systemone`` with ``JEV_AGENT_KEY`` / ``JEV_API_KEY``. ``off`` skips
the ladder column.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping

from data.schemas.episode import Episode
from decision.base import DecisionConfig, DecisionResult, FastDecisionEngine
from features.extract import EpisodeFeatures

JevMode = Literal["stub", "off", "real"]

DEFAULT_JEV_BASE_URL = "https://jev-agent.com/api/v1/systemone"

# Extra instruction/task cues (GT-free). Weakly informed — may false-positive.
_FAIL_INSTR = ("fail", "failure", "error", "drop", "miss", "stuck", "collide", "slip")
_OK_INSTR = ("success", "successfully", "complete", "completed", "ok", "safe")

# Official systemone question bundle for robot episode failure triage.
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


@dataclass(frozen=True)
class QuotaSnapshot:
    """Last quota block from a real API response (if present)."""

    used: int | None = None
    limit: int | None = None
    remaining: int | None = None
    month: str | None = None


def resolve_jev_mode(raw: str | None = None) -> JevMode:
    """Resolve ``JEV_MODE`` env (default ``stub``). Unknown → ``stub``."""
    value = (raw if raw is not None else os.environ.get("JEV_MODE", "stub")).strip().lower()
    if value in ("stub", "off", "real"):
        return value  # type: ignore[return-value]
    return "stub"


def resolve_jev_api_key() -> str | None:
    """Return API key from ``JEV_AGENT_KEY`` or alias ``JEV_API_KEY``."""
    for name in ("JEV_AGENT_KEY", "JEV_API_KEY"):
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def resolve_jev_base_url() -> str:
    """``JEV_BASE_URL`` or default systemone endpoint."""
    return (
        os.environ.get("JEV_BASE_URL", DEFAULT_JEV_BASE_URL).strip() or DEFAULT_JEV_BASE_URL
    )


def features_to_state(features: EpisodeFeatures) -> str:
    """Build GT-free episode text for the ``state`` field."""
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
    """Request body for POST /api/v1/systemone."""
    return {
        "state": features_to_state(features),
        "questions": SYSTEMONE_QUESTIONS,
    }


def parse_quota(raw: Mapping[str, Any] | None) -> QuotaSnapshot:
    if not raw:
        return QuotaSnapshot()
    def _int(key: str) -> int | None:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    month = raw.get("month")
    return QuotaSnapshot(
        used=_int("used"),
        limit=_int("limit"),
        remaining=_int("remaining"),
        month=str(month) if month is not None else None,
    )


def map_systemone_response(
    payload: Mapping[str, Any],
    *,
    episode_id: str,
) -> tuple[DecisionResult, int, QuotaSnapshot]:
    """Map API JSON → DecisionResult + input_tokens + quota.

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
    p_fail = noul
    # Noul has no confidence field; use distance from 0.5.
    confidence = abs(noul - 0.5) * 2.0

    ftype: str | None = None
    if not predicted_success:
        choice = fail_type_ans.get("choice")
        if isinstance(choice, str) and choice and choice != "other":
            ftype = choice
        elif isinstance(choice, str) and choice == "other":
            ftype = "other"
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

    quota = parse_quota(payload.get("quota") if isinstance(payload.get("quota"), dict) else None)

    evidence = ["jev_real", f"noul={noul:.3f}"]
    model = payload.get("model")
    if model:
        evidence.append(f"model={model}")
    if ftype:
        evidence.append(f"type={ftype}")

    result = DecisionResult(
        episode_id=episode_id,
        backend="jev",
        predicted_success=predicted_success,
        failure_probability=float(p_fail),
        failure_type=ftype,
        confidence=float(min(1.0, max(0.0, confidence))),
        needs_deep_review=not predicted_success,
        evidence=tuple(evidence),
    )
    return result, input_tokens, quota


def post_systemone(
    payload: Mapping[str, Any],
    *,
    api_key: str,
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """POST JSON to systemone. Never logs the API key."""
    url = base_url or resolve_jev_base_url()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
        except Exception:  # noqa: BLE001
            detail = str(exc.reason)
        raise RuntimeError(
            f"JEV systemone HTTP {exc.code}: {detail or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"JEV systemone request failed: {exc.reason}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("JEV systemone returned non-JSON body") from exc
    if not isinstance(data, dict):
        raise RuntimeError("JEV systemone returned unexpected JSON type")
    return data


class JevDecisionEngine(FastDecisionEngine):
    """Stub mimics a dearer look; real mode calls Jev Agent systemone.

    Never reads ``ground_truth``. Fit() only stores a development fail prior
    for cue-less stub episodes (same honesty bar as Baseline).
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
        api_key: str | None = None,
        base_url: str | None = None,
        http_post: Any | None = None,
        timeout_s: float = 60.0,
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
        self._api_key = api_key  # optional override; never logged
        self._base_url = base_url
        self._http_post = http_post  # injectable for unit tests
        self._timeout_s = timeout_s
        self._fail_prior = self.config.default_fail_probability
        self._majority_type: str | None = None
        self._calls = 0
        self._tokens = 0
        self._stub_units = 0.0
        self._last_quota = QuotaSnapshot()

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

    def quota_snapshot(self) -> QuotaSnapshot:
        return self._last_quota

    def _charge_stub(self, features: EpisodeFeatures) -> None:
        tok = self.stub_token_overhead + max(1, features.instruction_len // 4)
        self._calls += 1
        self._tokens += tok
        self._stub_units += self.stub_cost_units

    def _charge_real(self, input_tokens: int) -> None:
        self._calls += 1
        self._tokens += max(0, int(input_tokens))
        # Real mode: stub_units stay 0; cost is calls + tokens only.

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

    def _evaluate_real(self, features: EpisodeFeatures) -> DecisionResult:
        key = self._api_key if self._api_key is not None else resolve_jev_api_key()
        if not key:
            raise RuntimeError(
                "JEV_MODE=real requires JEV_AGENT_KEY or JEV_API_KEY in the environment."
            )
        payload = build_systemone_payload(features)
        post = self._http_post or post_systemone
        if self._http_post is not None:
            raw = post(
                payload,
                api_key=key,
                base_url=self._base_url or resolve_jev_base_url(),
                timeout_s=self._timeout_s,
            )
        else:
            raw = post_systemone(
                payload,
                api_key=key,
                base_url=self._base_url,
                timeout_s=self._timeout_s,
            )
        result, input_tokens, quota = map_systemone_response(
            raw, episode_id=features.episode_id
        )
        self._charge_real(input_tokens)
        self._last_quota = quota
        return result

    def evaluate(
        self,
        features: EpisodeFeatures,
        decision_schema: DecisionConfig | None = None,
    ) -> DecisionResult:
        if self.mode == "real":
            return self._evaluate_real(features)
        if self.mode == "off":
            raise RuntimeError(
                "JevDecisionEngine constructed with mode=off; harness should omit it."
            )

        cfg = decision_schema or self.config
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
