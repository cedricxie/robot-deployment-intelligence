"""Frequency-only: majority class from development reference."""

from __future__ import annotations

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig
from decision.frequency import FrequencyOnlyDecisionEngine
from features.extract import extract_features


def _ep(eid: str, success: bool, failure_type: str | None = None) -> Episode:
    return Episode(
        episode_id=eid,
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def test_frequency_majority_fail_and_type():
    ref = [
        _ep("f1", False, "position_deviation"),
        _ep("f2", False, "position_deviation"),
        _ep("f3", False, "step_omission"),
        _ep("ok", True, None),
    ]
    eng = FrequencyOnlyDecisionEngine(DecisionConfig()).fit(ref)
    assert eng.fail_prior == 0.75
    assert eng.majority_failure_type == "position_deviation"
    r = eng.evaluate(extract_features(_ep("query", True)))
    assert r.predicted_success is False
    assert r.failure_type == "position_deviation"
    assert r.backend == "frequency"
    assert 0.0 <= r.failure_probability <= 1.0


def test_frequency_majority_success():
    ref = [_ep("ok1", True), _ep("ok2", True), _ep("f1", False, "grasping_error")]
    eng = FrequencyOnlyDecisionEngine(DecisionConfig()).fit(ref)
    r = eng.evaluate(extract_features(_ep("q", False, "grasping_error")))
    assert r.predicted_success is True
    assert r.failure_type is None
