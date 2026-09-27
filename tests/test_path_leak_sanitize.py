"""Path-label leakage sanitizer + ablation wiring."""

from __future__ import annotations

import os

from data.schemas.episode import Episode, GroundTruth
from decision.base import DecisionConfig, with_path_leak_ablation
from decision.baseline import BaselineFastDecisionEngine
from features.extract import extract_features
from features.sanitize import (
    EXTRA_LEAK_TOKENS,
    collect_leak_tokens,
    path_leak_ablate_enabled,
    sanitize_path_text,
)


def test_sanitize_strips_success_fail_folder_cues():
    raw = (
        "uuid-1 dataset_success_cleaned/pickcube/stack_ok/"
        "uuid-1.mp4 fail_run/extra.mp4"
    )
    cleaned = sanitize_path_text(raw)
    assert "dataset_success_cleaned" not in cleaned
    assert "stack_ok" not in cleaned
    assert "success" not in cleaned
    assert "fail" not in cleaned
    assert "pickcube" in cleaned
    assert "uuid-1" in cleaned


def test_sanitize_longest_token_first():
    # Bare ``success`` must not leave ``dataset_`` + ``_cleaned`` fragments that
    # still match shorter leftovers incorrectly; whole folder cue wiped.
    cleaned = sanitize_path_text("foo/dataset_success_cleaned/bar.mp4")
    assert "dataset_success_cleaned" not in cleaned
    assert "success" not in cleaned
    assert "foo" in cleaned
    assert "bar" in cleaned or "bar.mp4" in cleaned


def test_collect_leak_tokens_includes_config_and_extras():
    toks = collect_leak_tokens(
        fail_path_tokens=("fail",),
        success_path_tokens=("success",),
    )
    assert "fail" in toks
    assert "success" in toks
    assert "dataset_success_cleaned" in toks
    assert toks[0] == max(toks, key=len)  # longest first
    for extra in EXTRA_LEAK_TOKENS:
        assert extra in toks


def test_extract_features_ablate_flag():
    ep = Episode(
        episode_id="e-ok",
        video_paths=["dataset_success_cleaned/SafeTask/stack_ok/ok.mp4"],
        ground_truth=GroundTruth(success=True),
    )
    leaked = extract_features(ep, ablate_path_leak=False)
    ablated = extract_features(ep, ablate_path_leak=True)
    assert "success" in leaked.path_text
    assert "stack_ok" in leaked.path_text
    assert "success" not in ablated.path_text
    assert "stack_ok" not in ablated.path_text


def test_path_leak_ablate_enabled_env(monkeypatch):
    monkeypatch.delenv("LEAK_ABLATE", raising=False)
    assert path_leak_ablate_enabled(None) is False
    monkeypatch.setenv("LEAK_ABLATE", "1")
    assert path_leak_ablate_enabled(None) is True
    assert path_leak_ablate_enabled(False) is False  # explicit wins
    assert path_leak_ablate_enabled(True) is True


def test_with_path_leak_ablation_empties_tokens():
    cfg = DecisionConfig()
    out = with_path_leak_ablation(cfg)
    assert out.ablate_path_leak is True
    assert out.fail_path_tokens == ()
    assert out.success_path_tokens == ()
    assert out.type_keywords == cfg.type_keywords


def test_baseline_loses_path_cue_after_ablation():
    ep = Episode(
        episode_id="fixture-sim-success-001",
        video_paths=["dataset_success_cleaned/SafeTask/stack_ok/ok.mp4"],
        ground_truth=GroundTruth(success=True),
    )
    eng = BaselineFastDecisionEngine(DecisionConfig(default_fail_probability=0.8))
    leaked = eng.evaluate(extract_features(ep, ablate_path_leak=False))
    ablated = eng.evaluate(extract_features(ep, ablate_path_leak=True))
    assert leaked.predicted_success is True
    assert "success_path_cue" in leaked.evidence
    # No path cue → high fail prior → predict fail
    assert ablated.predicted_success is False
    assert "success_path_cue" not in ablated.evidence
