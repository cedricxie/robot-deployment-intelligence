"""Smoke + small integration for M0.5 thin-slice runner/report."""

from __future__ import annotations

import math
import re
from pathlib import Path

from scripts.m0_5_thin_slice import render_report, run_thin_slice

ROOT = Path(__file__).resolve().parents[1]


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def test_thin_slice_synthetic_small_finite_metrics():
    """Fast unit: n=24 synthetic → finite ladder metrics + required report sections."""
    cfg = {
        "n_episodes": 40,
        "seed": 42,
        "top_k": 5,
        "min_labeled": 9999,  # force synthetic path even if RoboFAC present
        "fixture_dir": "data/samples/robofac",
        "decision_config": "configs/decision.json",
        "proxy_config": "configs/proxy_importance.json",
        "robofac_sim_glob": "data/raw/robofac/test_qa_sim/annos_per_video_split*.json",
        "use_realworld": False,
    }
    result = run_thin_slice(cfg, force_synthetic=True, n_episodes=40, seed=42, top_k=5)
    assert result.data_source == "fixture_synthetic"
    assert result.n_episodes == 40
    assert result.n_eval >= 1
    assert [c.backend for c in result.harness.columns] == [
        "random",
        "frequency",
        "baseline",
    ]
    for col in result.harness.columns:
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

    md = render_report(result)
    for section in (
        "Ladder metrics",
        "Proxy-important",
        "Ranking examples",
        "What this tells us about rule-based effect",
        "review priority ≠ business importance",
    ):
        assert section.lower() in md.lower() or section in md
    # Ranking table has min(top_k, n_eval) episode rows
    n_rank = min(result.top_k, result.n_eval)
    assert md.count("| `") >= n_rank
    assert n_rank >= 1


def test_thin_slice_report_mentions_data_source_and_seed():
    result = run_thin_slice(
        force_synthetic=True, n_episodes=20, seed=7, top_k=4
    )
    md = render_report(result)
    assert "Seed" in md and "7" in md
    assert "fixture" in md.lower() or "synthetic" in md.lower()
    assert re.search(r"BaselineFast|Baseline|path", md, re.I)


def test_thin_slice_prefers_robofac_when_available():
    """Integration: if labeled RoboFAC ≥ min_labeled, source is robofac_real."""
    sim = list((ROOT / "data/raw/robofac/test_qa_sim").glob("annos_per_video_split*.json"))
    if len(sim) < 1:
        return  # skip silently when raw data absent (CI without download)
    result = run_thin_slice(
        {
            "n_episodes": 40,
            "seed": 42,
            "top_k": 8,
            "min_labeled": 30,
            "fixture_dir": "data/samples/robofac",
            "decision_config": "configs/decision.json",
            "proxy_config": "configs/proxy_importance.json",
            "robofac_sim_glob": "data/raw/robofac/test_qa_sim/annos_per_video_split*.json",
            "use_realworld": False,
        },
        n_episodes=40,
        seed=42,
        top_k=8,
    )
    assert result.data_source == "robofac_real"
    assert 30 <= result.n_episodes <= 40
    base = result.harness.column("baseline")
    assert base is not None and _finite(base.detection_f1)
