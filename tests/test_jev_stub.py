"""JEV stub + real (mocked HTTP): deterministic, GT-free, mode switch, harness cost."""

from __future__ import annotations

import math
import os

import pytest

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig
from decision.jev import (
    JevDecisionEngine,
    build_jev_engine,
    build_systemone_payload,
    features_to_state,
    map_systemone_response,
    resolve_jev_api_key,
    resolve_jev_mode,
)
from evaluation.harness import EvaluationHarness
from features.extract import extract_features
from proxy_importance.config import ProxyImportanceConfig


def _ep(
    eid: str,
    success: bool,
    failure_type: str | None = None,
    video: str = "",
    instruction: str | None = None,
    meta: dict | None = None,
) -> Episode:
    return Episode(
        episode_id=eid,
        instruction=instruction,
        video_paths=[video] if video else [],
        metadata=meta or {},
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def test_resolve_jev_mode_default_and_aliases(monkeypatch):
    monkeypatch.delenv("JEV_MODE", raising=False)
    assert resolve_jev_mode() == "stub"
    assert resolve_jev_mode("OFF") == "off"
    assert resolve_jev_mode("real") == "real"
    assert resolve_jev_mode("weird") == "stub"


def test_resolve_jev_api_key_aliases(monkeypatch):
    monkeypatch.delenv("JEV_AGENT_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    assert resolve_jev_api_key() is None
    monkeypatch.setenv("JEV_API_KEY", "alias-key")
    assert resolve_jev_api_key() == "alias-key"
    monkeypatch.setenv("JEV_AGENT_KEY", "primary-key")
    assert resolve_jev_api_key() == "primary-key"


def test_build_jev_off_returns_none():
    assert build_jev_engine(mode="off") is None
    eng = build_jev_engine(mode="stub")
    assert eng is not None
    assert eng.mode == "stub"


def test_jev_real_missing_key_raises(monkeypatch):
    monkeypatch.delenv("JEV_AGENT_KEY", raising=False)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    eng = JevDecisionEngine(DecisionConfig(), mode="real")
    with pytest.raises(RuntimeError, match="JEV_AGENT_KEY or JEV_API_KEY"):
        eng.evaluate(extract_features(_ep("x", False, video="fail/x.mp4")))


def test_jev_real_mocked_http_maps_noul_and_choice(monkeypatch):
    monkeypatch.setenv("JEV_AGENT_KEY", "test-key-not-real")
    captured: dict = {}

    def fake_post(payload, *, api_key, base_url, timeout_s):
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return {
            "model": "jev-1.13.0",
            "answers": {
                "is_fail": {"type": "noul", "noul": 0.91},
                "fail_type": {
                    "type": "choice",
                    "choice": "grasping_error",
                    "confidence": 0.8,
                    "probabilities": {"grasping_error": 0.8, "other": 0.2},
                },
            },
            "usage": {"input_tokens": 420, "output_tokens": 40},
            "quota": {"used": 2, "limit": 50, "remaining": 48, "month": "2026-09"},
        }

    eng = JevDecisionEngine(
        DecisionConfig(), mode="real", http_post=fake_post, base_url="https://example.test/api"
    )
    feats = extract_features(
        _ep("f1", False, "grasping_error", "path/ep.mp4", instruction="pick cup")
    )
    result = eng.evaluate(feats)
    assert result.predicted_success is False
    assert result.failure_probability == pytest.approx(0.91)
    assert result.failure_type == "grasping_error"
    assert "jev_real" in result.evidence
    snap = eng.cost_snapshot()
    assert snap.calls == 1
    assert snap.tokens == 420
    assert snap.stub_units == 0.0
    q = eng.quota_snapshot()
    assert q.remaining == 48
    assert q.used == 2
    # Payload is GT-free
    state = captured["payload"]["state"]
    assert "pick cup" in state
    assert "ground_truth" not in state.lower()
    assert "success:" not in state.lower()
    assert captured["api_key"] == "test-key-not-real"


def test_jev_real_noul_below_half_is_success(monkeypatch):
    monkeypatch.setenv("JEV_AGENT_KEY", "test-key-not-real")

    def fake_post(payload, *, api_key, base_url, timeout_s):
        return {
            "model": "jev-1.13.0",
            "answers": {
                "is_fail": {"type": "noul", "noul": 0.12},
                "fail_type": {"type": "choice", "choice": "other", "confidence": 0.5},
            },
            "usage": {"input_tokens": 100, "output_tokens": 10},
        }

    eng = JevDecisionEngine(mode="real", http_post=fake_post)
    r = eng.evaluate(extract_features(_ep("ok", True, video="success/a.mp4")))
    assert r.predicted_success is True
    assert r.failure_type is None
    assert r.failure_probability == pytest.approx(0.12)


def test_map_systemone_and_state_helpers():
    feats = extract_features(_ep("e1", False, video="fail/x.mp4", instruction="stack"))
    state = features_to_state(feats)
    assert "episode_id: e1" in state
    assert "instruction: stack" in state
    payload = build_systemone_payload(feats)
    assert "is_fail" in payload["questions"]
    assert payload["questions"]["fail_type"]["type"] == "choice"

    result, toks, quota = map_systemone_response(
        {
            "answers": {
                "is_fail": {"type": "noul", "noul": 0.7},
                "fail_type": {"type": "choice", "choice": "step_omission"},
            },
            "usage": {"input_tokens": 55},
            "quota": {"remaining": 3},
        },
        episode_id="e1",
    )
    assert toks == 55
    assert quota.remaining == 3
    assert result.failure_type == "step_omission"
    assert result.predicted_success is False


def test_jev_stub_gt_free_path_and_instr_cues():
    eng = JevDecisionEngine(DecisionConfig(), mode="stub")
    fail = eng.evaluate(
        extract_features(_ep("f1", False, "position_deviation", "path/fail_pos.mp4"))
    )
    ok = eng.evaluate(
        extract_features(_ep("ok1", True, None, "dataset_success_cleaned/a.mp4"))
    )
    assert fail.predicted_success is False
    assert ok.predicted_success is True
    assert "jev_stub" in fail.evidence
    # Instruction cue when path is neutral
    instr_fail = eng.evaluate(
        extract_features(
            _ep("f2", False, "grasping_error", "neutral/ep.mp4", instruction="grip slip error")
        )
    )
    assert instr_fail.predicted_success is False
    assert "fail_instr_cue" in instr_fail.evidence


def test_jev_stub_deterministic_and_cost():
    cfg = DecisionConfig(jev_stub_cost_units=10.0, jev_stub_token_overhead=16)
    eng = JevDecisionEngine(cfg, mode="stub")
    feats = [
        extract_features(_ep("a", False, video="fail/a.mp4", instruction="abcd")),
        extract_features(_ep("b", True, video="success/b.mp4", instruction="")),
    ]
    r1 = eng.evaluate_many(feats)
    snap = eng.cost_snapshot()
    assert snap.calls == 2
    assert snap.stub_units == 20.0
    assert snap.tokens > 0
    r2 = eng.evaluate_many(feats)
    assert [x.predicted_success for x in r1] == [x.predicted_success for x in r2]
    assert eng.cost_snapshot().calls == 2  # reset on evaluate_many


def test_harness_includes_jev_cost_column():
    ref = [
        _ep("ref-f1", False, "position_deviation", "x/fail_pos.mp4"),
        _ep("ref-f2", False, "position_deviation", "x/fail_pos2.mp4"),
        _ep("ref-f3", False, "step_omission", "x/fail_omit.mp4"),
        _ep("ref-ok", True, None, "success/ok.mp4"),
        _ep("ref-f4", False, "position_deviation", "x/fail_pos3.mp4"),
    ]
    eval_eps = [
        _ep("ev-fail-pos", False, "position_deviation", "eval/fail_position.mp4"),
        _ep("ev-fail-omit", False, "step_omission", "eval/fail_omit.mp4"),
        _ep("ev-ok-1", True, None, "dataset_success_cleaned/1.mp4"),
        _ep("ev-ok-2", True, None, "stack_ok/2.mp4"),
    ]
    dcfg = DecisionConfig(random_seed=42, review_top_k=2, jev_stub_cost_units=10.0)
    pcfg = ProxyImportanceConfig(
        rare_freq_quantile=0.25, high_freq_min_count=3, novelty_min_score=1.0, top_k=2
    )
    report = EvaluationHarness(dcfg, pcfg, include_jev=True).evaluate(
        eval_eps, reference_episodes=ref, top_k=2
    )
    assert [c.backend for c in report.columns] == [
        "random",
        "frequency",
        "baseline",
        "jev",
    ]
    jev = report.column("jev")
    assert jev is not None
    assert jev.cost_calls == len(eval_eps)
    assert jev.cost_stub_units == 10.0 * len(eval_eps)
    assert jev.cost_tokens > 0
    # Cheaper rungs: zero stub cost
    for name in ("random", "frequency", "baseline"):
        col = report.column(name)
        assert col is not None
        assert col.cost_calls == 0
        assert col.cost_stub_units == 0.0
        assert math.isfinite(col.detection_f1)


def test_harness_skips_jev_when_off(monkeypatch):
    monkeypatch.setenv("JEV_MODE", "off")
    ref = [_ep("r", False, "position_deviation", "fail/r.mp4")]
    eval_eps = [_ep("e", False, "position_deviation", "fail/e.mp4")]
    report = EvaluationHarness(include_jev=None).evaluate(
        eval_eps, reference_episodes=ref
    )
    assert [c.backend for c in report.columns] == ["random", "frequency", "baseline"]
    assert report.column("jev") is None


def test_jev_does_not_read_ground_truth():
    """Wipe GT after feature extract — engine still uses path cues only."""
    ep = _ep("wipe", False, "position_deviation", "path/fail_pos.mp4")
    feats = extract_features(ep)
    ep.ground_truth = GroundTruth(success=None, failure_type=None)  # type: ignore[misc]
    # Episode/GroundTruth may be frozen — if so, just evaluate features alone.
    eng = JevDecisionEngine(mode="stub")
    r = eng.evaluate(feats)
    assert r.predicted_success is False
