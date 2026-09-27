"""Cascade routing + mock deep reasoner (no network)."""

from __future__ import annotations

import math
from pathlib import Path

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig, DecisionResult
from features.extract import FeatureExtractor, extract_features
from proxy_importance.config import ProxyImportanceConfig
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import build_frequency_table
from proxy_importance.rules import ProxyReviewPriorityResult, score_episodes
from reasoning.cascade import CascadeRunner
from reasoning.cheap import CheapPathEngine
from reasoning.deep import DeepReasoner
from reasoning.providers import MockVisionReasoningProvider
from reasoning.routing import (
    RoutingPolicy,
    cheap_can_stop,
    is_uncertain,
    load_routing_policy,
    select_deep_ids,
    should_escalate_candidate,
)
from scripts.m3_cascade_report import render_report, run_m3

ROOT = Path(__file__).resolve().parents[1]


def _ep(eid: str, success: bool, path: str, ftype: str | None = None) -> Episode:
    return Episode(
        episode_id=eid,
        task="CascadeTask",
        instruction="do the thing" if success else f"fail via {ftype or 'unknown'}",
        video_paths=[path],
        ground_truth=GroundTruth(success=success, failure_type=ftype),
    )


def _synth_slice(n: int = 40) -> tuple[list[Episode], list[Episode]]:
    """Balanced path-cued corpus for unit tests.

    Episode ids avoid success/fail tokens (those live only in video paths).
    """
    eps: list[Episode] = []
    types = (
        "position_deviation",
        "step_omission",
        "grasping_error",
        "orientation_deviation",
    )
    for i in range(n):
        if i % 4 == 0:
            eps.append(
                _ep(
                    f"ep-{i:03d}",
                    True,
                    f"dataset_success_cleaned/view0/ep-{i:03d}.mp4",
                )
            )
        elif i % 4 == 1:
            # Clear fail path cue
            ft = types[i % len(types)]
            eps.append(
                _ep(
                    f"ep-{i:03d}",
                    False,
                    f"sim/fail_{ft}/view0/ep-{i:03d}.mp4",
                    ft,
                )
            )
        else:
            # Cue-less fail (no success/fail path tokens) — uncertain for routing
            ft = types[i % len(types)]
            eps.append(
                _ep(
                    f"ep-{i:03d}",
                    False,
                    f"sim/traj/view0/ep-{i:03d}.mp4",
                    ft,
                )
            )
    # 70/30 split deterministic
    n_eval = max(1, n * 30 // 100)
    return eps[:-n_eval], eps[-n_eval:]


def test_mock_provider_no_network():
    p = MockVisionReasoningProvider()
    out = p.summarize(["dataset_success_cleaned/a.mp4"], "prompt")
    assert "successful" in out.lower() or "success" in out.lower()
    assert p.calls == 1


def test_deep_reasoner_structured_diagnosis():
    deep = DeepReasoner(MockVisionReasoningProvider(cost_units_per_call=0.0))
    feat = extract_features(
        _ep("e1", False, "sim/fail_position/view0/e1.mp4", "position_deviation")
    )
    diag = deep.reason(feat)
    assert diag.episode_id == "e1"
    assert diag.raw
    assert diag.model
    assert diag.prompt_version
    assert deep.cost_snapshot().calls == 1


def test_routing_policy_loads_json():
    pol = load_routing_policy(ROOT / "configs" / "cascade_routing.json")
    assert pol.max_deep_fraction == 0.20
    assert pol.escalate_if_uncertain is True


def test_cheap_stops_on_path_cue():
    pol = RoutingPolicy(cheap_stop_confidence=0.8)
    cheap = CheapPathEngine()
    ok = cheap.evaluate(
        extract_features(_ep("ok", True, "dataset_success_cleaned/x.mp4"))
    )
    assert cheap_can_stop(ok, pol)
    abstain = cheap.evaluate(
        extract_features(_ep("x", False, "sim/traj/view0/x.mp4", "timing_error"))
    )
    assert not cheap_can_stop(abstain, pol)
    assert "cheap_abstain" in abstain.evidence


def test_select_deep_ids_respects_budget():
    pol = RoutingPolicy(
        max_deep_fraction=0.25,
        escalate_if_uncertain=True,
        uncertain_if_no_path_cue=True,
        prefer_proxy_important_in_budget=True,
    )
    results = [
        DecisionResult(
            episode_id=f"e{i}",
            backend="baseline",
            predicted_success=False,
            failure_probability=0.7,
            failure_type="position_deviation",
            confidence=0.7,
            needs_deep_review=True,
            evidence=("prior_fail=0.700",),
        )
        for i in range(20)
    ]
    proxy = {
        r.episode_id: ProxyReviewPriorityResult(
            episode_id=r.episode_id,
            proxy_important=i < 10,
            rule_hits=("A", "B"),
            frequency_bucket=None,
            novelty_score=0.5,
            type_count=1,
            type_rel_freq=0.1,
        )
        for i, r in enumerate(results)
    }
    chosen = select_deep_ids(results, proxy, pol, n_episodes=20)
    assert len(chosen) == 5  # 0.25 * 20
    # Prefer proxy-important
    assert all(proxy[i].proxy_important for i in chosen)


def test_cascade_speedup_and_recall_guardrail_synthetic():
    development, eval_eps = _synth_slice(40)
    pcfg = ProxyImportanceConfig()
    freq = build_frequency_table(development, pcfg)
    bank = FailureBank.from_episodes(development)
    proxy = score_episodes(eval_eps, freq=freq, bank=bank, config=pcfg)

    policy = RoutingPolicy(
        max_deep_fraction=0.20,
        escalate_if_uncertain=True,
        uncertain_if_no_path_cue=True,
        escalate_if_needs_deep_review=False,
        escalate_if_proxy_important=False,
        cost_units_deep=50.0,
    )
    runner = CascadeRunner(policy=policy, decision_config=DecisionConfig())
    feats = FeatureExtractor().extract_many(eval_eps)
    a = runner.run_strategy(
        "all_deep",
        eval_eps,
        feats,
        reference=development,
        proxy_results=proxy,
        fast_backend="baseline",
    )
    b = runner.run_strategy(
        "cascade",
        eval_eps,
        feats,
        reference=development,
        proxy_results=proxy,
        fast_backend="baseline",
    )
    assert a.cost.deep_calls == len(eval_eps)
    assert b.cost.deep_calls > 0
    speedup = a.cost.deep_calls / b.cost.deep_calls
    assert speedup >= 3.0
    drop = a.proxy_important_recall - b.proxy_important_recall
    assert drop <= 0.05 + 1e-9
    assert math.isfinite(a.detection_f1) and math.isfinite(b.detection_f1)


def test_m3_report_synthetic_sections():
    result = run_m3(force_synthetic=True, n_episodes=40, seed=42)
    assert result.n_eval >= 1
    assert result.pairs
    md = render_report(result)
    for needle in (
        "Cascade economics",
        "all-deep",
        "proxy-important",
        "review-priority ≠ business importance",
        "Gate check",
        "Cost proxy",
    ):
        assert needle.lower() in md.lower() or needle in md
    a, b = result.pairs[0]
    assert a.strategy == "all_deep" and b.strategy == "cascade"
    assert a.cost.deep_calls >= b.cost.deep_calls


def test_should_escalate_flags():
    pol = RoutingPolicy(
        escalate_if_uncertain=False,
        escalate_if_needs_deep_review=True,
        escalate_if_proxy_important=False,
        uncertain_if_no_path_cue=False,
    )
    r = DecisionResult(
        episode_id="e",
        backend="baseline",
        predicted_success=False,
        failure_probability=0.9,
        failure_type=None,
        confidence=0.95,
        needs_deep_review=True,
        evidence=("fail_path_cue",),
    )
    assert should_escalate_candidate(r, proxy_important=False, policy=pol)
    assert not is_uncertain(r, pol)
