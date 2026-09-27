"""Harness skeleton: ladder columns finite; Baseline ≥ Random on smoke."""

from __future__ import annotations

import math

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig, load_decision_config
from evaluation.harness import EvaluationHarness, run_ladder
from proxy_importance.config import ProxyImportanceConfig


def _ep(
    eid: str,
    success: bool,
    failure_type: str | None = None,
    video: str = "",
) -> Episode:
    return Episode(
        episode_id=eid,
        video_paths=[video] if video else [],
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def _fixture_corpus() -> tuple[list[Episode], list[Episode]]:
    """Reference (dev-like) + eval set with path cues."""
    ref = [
        _ep("ref-f1", False, "position_deviation", "x/fail_pos.mp4"),
        _ep("ref-f2", False, "position_deviation", "x/fail_pos2.mp4"),
        _ep("ref-f3", False, "step_omission", "x/fail_omit.mp4"),
        _ep("ref-ok", True, None, "success/ok.mp4"),
    ]
    # Make position_deviation high-freq (≥3) for proxy C; add rare type later on eval
    ref.append(_ep("ref-f4", False, "position_deviation", "x/fail_pos3.mp4"))
    eval_eps = [
        _ep("ev-fail-pos", False, "position_deviation", "eval/fail_position.mp4"),
        _ep("ev-fail-omit", False, "step_omission", "eval/fail_omit.mp4"),
        _ep("ev-fail-novel", False, "timing_error", "eval/fail_timing.mp4"),
        _ep("ev-ok-1", True, None, "dataset_success_cleaned/1.mp4"),
        _ep("ev-ok-2", True, None, "stack_ok/2.mp4"),
    ]
    return ref, eval_eps


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def test_config_loads():
    cfg = load_decision_config()
    assert cfg.random_seed == 42
    assert cfg.review_top_k >= 1
    assert "fail" in cfg.fail_path_tokens


def test_harness_ladder_finite_metrics():
    ref, eval_eps = _fixture_corpus()
    dcfg = DecisionConfig(random_seed=42, review_top_k=2)
    pcfg = ProxyImportanceConfig(
        rare_freq_quantile=0.25, high_freq_min_count=3, novelty_min_score=1.0, top_k=2
    )
    report = EvaluationHarness(dcfg, pcfg).evaluate(
        eval_eps, reference_episodes=ref, top_k=2
    )
    assert [c.backend for c in report.columns] == ["random", "frequency", "baseline"]
    for col in report.columns:
        for val in (
            col.detection_precision,
            col.detection_recall,
            col.detection_f1,
            col.failure_type_agreement,
            col.proxy_retention,
            col.review_reduction,
        ):
            assert _finite(val)
            assert 0.0 <= val <= 1.0
        assert col.n_episodes == len(eval_eps)


def test_harness_baseline_beats_random_detection():
    ref, eval_eps = _fixture_corpus()
    dcfg = DecisionConfig(random_seed=0, review_top_k=2)
    report = EvaluationHarness(dcfg).evaluate(eval_eps, reference_episodes=ref)
    rnd = report.column("random")
    base = report.column("baseline")
    assert rnd is not None and base is not None
    assert base.detection_f1 > rnd.detection_f1


def test_run_ladder_rows_have_expected_keys():
    ref, eval_eps = _fixture_corpus()
    report = run_ladder(eval_eps, EvaluationHarness().default_engines(ref), reference_episodes=ref)
    rows = report.as_rows()
    assert set(rows[0]) >= {
        "backend",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "proxy_retention",
        "review_reduction",
        "failure_type_agreement",
    }
