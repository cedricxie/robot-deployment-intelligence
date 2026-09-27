"""BaselineFastDecisionEngine: GT-free path cues; beats Random on fixtures."""

from __future__ import annotations

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig
from decision.baseline import BaselineFastDecisionEngine
from decision.random import RandomDecisionEngine
from evaluation.harness import detection_prf1
from features.extract import extract_features


def _ep(
    eid: str,
    success: bool,
    failure_type: str | None = None,
    video: str | None = None,
    meta: dict | None = None,
) -> Episode:
    return Episode(
        episode_id=eid,
        video_paths=[video] if video else [],
        metadata=meta or {},
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def test_baseline_path_cues_gt_free():
    eng = BaselineFastDecisionEngine(DecisionConfig())
    fail_ep = _ep(
        "fixture-sim-fail-pos-001",
        False,
        "position_deviation",
        video="MicrowaveTask/fixture-sim-fail-pos-001.mp4",
    )
    ok_ep = _ep(
        "fixture-sim-success-001",
        True,
        None,
        video="dataset_success_cleaned/SafeTask/stack_ok/ok.mp4",
    )
    # Ensure evaluate does not need GT: wipe would still work via path
    rf = eng.evaluate(extract_features(fail_ep))
    ro = eng.evaluate(extract_features(ok_ep))
    assert rf.predicted_success is False
    assert ro.predicted_success is True
    assert rf.failure_type == "position_deviation"
    assert "fail_path_cue" in rf.evidence
    assert "success_path_cue" in ro.evidence


def test_baseline_beats_random_on_fixture_smoke():
    """Acceptance: Baseline detection F1 > Random on path-cued fixtures."""
    eps = [
        _ep("fail-pos", False, "position_deviation", "path/fail_pos.mp4"),
        _ep("fail-omit", False, "step_omission", "path/fail_omit_step.mp4"),
        _ep("fail-grasp", False, "grasping_error", "path/fail_grasp.mp4"),
        _ep("ok-a", True, None, "dataset_success_cleaned/a.mp4"),
        _ep("ok-b", True, None, "stack_ok/b.mp4"),
        _ep("ok-c", True, None, "success/c.mp4"),
    ]
    cfg = DecisionConfig(random_seed=42, review_top_k=3)
    feats = [extract_features(e) for e in eps]
    base = BaselineFastDecisionEngine(cfg)
    rnd = RandomDecisionEngine(cfg)
    base_f1 = detection_prf1(eps, base.evaluate_many(feats)).f1
    rnd_f1 = detection_prf1(eps, rnd.evaluate_many(feats)).f1
    assert base_f1 > rnd_f1
    assert base_f1 == 1.0
