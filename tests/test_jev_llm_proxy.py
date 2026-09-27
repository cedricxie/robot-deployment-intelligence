"""LLM-proxy JEV judge: deterministic, GT-free, systemone-shaped answers."""

from __future__ import annotations

import pytest

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig
from decision.jev import (
    JevDecisionEngine,
    build_systemone_payload,
    features_to_state,
    llm_proxy_judge,
    map_systemone_response,
    resolve_jev_mode,
)
from features.extract import extract_features


def _ep(
    eid: str,
    success: bool,
    failure_type: str | None = None,
    video: str = "",
    instruction: str | None = None,
    task: str | None = None,
    meta: dict | None = None,
) -> Episode:
    return Episode(
        episode_id=eid,
        task=task,
        instruction=instruction,
        video_paths=[video] if video else [],
        metadata=meta or {},
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def test_resolve_llm_proxy_mode():
    assert resolve_jev_mode("llm_proxy") == "llm_proxy"
    assert resolve_jev_mode("LLM_PROXY") == "llm_proxy"


def test_llm_proxy_judge_shape_and_soft_noul():
    cfg = DecisionConfig()
    feats = extract_features(
        _ep(
            "f1",
            False,
            "position_deviation",
            "rotation-v1/cube_grasp_view0_position_offset/f1.mp4",
            instruction="stack cubes",
        )
    )
    payload = llm_proxy_judge(feats, cfg, fail_prior=0.7)
    assert "answers" in payload
    assert "is_fail" in payload["answers"]
    assert "fail_type" in payload["answers"]
    noul = float(payload["answers"]["is_fail"]["noul"])
    assert 0.5 <= noul <= 1.0
    # Soft — not only the stub extremes.
    assert noul not in (0.12, 0.88)
    choice = payload["answers"]["fail_type"]["choice"]
    assert choice == "position_deviation"
    state = features_to_state(feats)
    assert "ground_truth" not in state.lower()
    assert "position_offset" in state


def test_llm_proxy_avoids_safetask_false_friend():
    """Stub treats task SafeTask as success via 'safe'; proxy must not."""
    cfg = DecisionConfig()
    # Episode id must not contain literal 'fail' (path_text includes id).
    feats = extract_features(
        _ep(
            "7351849e-6269-4511-8eea-0984fbddf4a6",
            False,
            None,
            "safetask-hammer/safe_knob_reach_view1/7351849e-6269-4511-8eea-0984fbddf4a6.mp4",
            task="SafeTask",
            instruction=None,
        )
    )
    stub = JevDecisionEngine(cfg, mode="stub")
    stub._fail_prior = 0.78
    stub_r = stub.evaluate(feats)
    # Stub historically false-positives success on SafeTask ('safe' in _OK_INSTR).
    assert stub_r.predicted_success is True
    assert "success_instr_cue" in stub_r.evidence

    proxy = JevDecisionEngine(cfg, mode="llm_proxy")
    proxy._fail_prior = 0.78
    proxy_r = proxy.evaluate(feats)
    assert proxy_r.predicted_success is False
    assert "jev_llm_proxy" in proxy_r.evidence


def test_llm_proxy_success_path_and_map():
    cfg = DecisionConfig()
    feats = extract_features(
        _ep("ok", True, None, "dataset_success_cleaned/view0/ok.mp4")
    )
    payload = llm_proxy_judge(feats, cfg, fail_prior=0.7)
    noul = float(payload["answers"]["is_fail"]["noul"])
    assert noul < 0.5
    result, toks = map_systemone_response(
        payload, episode_id="ok", evidence_tag="jev_llm_proxy"
    )
    assert result.predicted_success is True
    assert result.failure_type is None or result.predicted_success
    assert toks > 0
    assert build_systemone_payload(feats)["questions"]["is_fail"]["type"] == "noul"


def test_llm_proxy_injectable_judge_fn():
    cfg = DecisionConfig()

    def fake_judge(features, config):
        return {
            "model": "test_inject",
            "answers": {
                "is_fail": {"type": "noul", "noul": 0.91},
                "fail_type": {
                    "type": "choice",
                    "choice": "grasping_error",
                    "confidence": 0.8,
                },
            },
            "usage": {"input_tokens": 33},
        }

    eng = JevDecisionEngine(cfg, mode="llm_proxy", judge_fn=fake_judge)
    r = eng.evaluate(extract_features(_ep("x", False, video="neutral/x.mp4")))
    assert r.failure_probability == pytest.approx(0.91)
    assert r.failure_type == "grasping_error"
    snap = eng.cost_snapshot()
    assert snap.calls == 1
    assert snap.tokens == 33
    assert snap.stub_units == 0.0


def test_llm_proxy_gt_free_after_wipe():
    ep = _ep("wipe", False, "grasping_error", "sim/fail_grasp/a.mp4")
    feats = extract_features(ep)
    eng = JevDecisionEngine(mode="llm_proxy")
    r = eng.evaluate(feats)
    assert r.predicted_success is False
    assert "jev_llm_proxy" in r.evidence


def test_m1_llm_proxy_script_smoke_synthetic():
    from scripts.m1_jev_llm_proxy import run_llm_proxy

    result = run_llm_proxy(force_synthetic=True, n_episodes=40, seed=0, top_k=5)
    assert result.n_eval > 0
    assert result.harness.column("baseline") is not None
    assert result.harness.column("jev") is not None
    jev = result.harness.column("jev")
    assert jev is not None
    assert jev.cost_calls == result.n_eval
    assert jev.cost_stub_units == 0.0
