"""Proxy review-priority A–D: conjunction, edges, determinism, hidden isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.schemas.episode import Episode, FrequencyBucket, GroundTruth
from data.split_access import HiddenLabelAccessError, assign_splits, write_split_manifests
from proxy_importance.access import (
    build_reference_for_tuning,
    score_for_improvement_loop,
)
from proxy_importance.config import ProxyImportanceConfig, load_config
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import build_frequency_table
from proxy_importance.rules import (
    apply_proxy_to_ground_truth,
    ranking_bonus_e_actionable_phrases,
    ranking_bonus_f_cascade_uncertainty,
    score_episode,
    score_episodes,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "proxy_importance.json"


def _ep(eid: str, success: bool | None, failure_type: str | None = None) -> Episode:
    return Episode(
        episode_id=eid,
        ground_truth=GroundTruth(success=success, failure_type=failure_type),
    )


def _ref_corpus() -> list[Episode]:
    """Reference fails: position_deviation×4, step_omission×2, grasping_error×1."""
    eps: list[Episode] = []
    for i in range(4):
        eps.append(_ep(f"ref-pos-{i}", False, "position_deviation"))
    for i in range(2):
        eps.append(_ep(f"ref-omit-{i}", False, "step_omission"))
    eps.append(_ep("ref-grasp-0", False, "grasping_error"))
    eps.append(_ep("ref-ok-0", True, None))
    return eps


@pytest.fixture
def cfg() -> ProxyImportanceConfig:
    # high_freq_min_count=3 → position_deviation is C; quantile 0.25 → grasping rare (B)
    return ProxyImportanceConfig(
        rare_freq_quantile=0.25,
        high_freq_min_count=3,
        novelty_min_score=1.0,
        top_k=5,
    )


@pytest.fixture
def freq_bank(cfg: ProxyImportanceConfig):
    ref = _ref_corpus()
    freq = build_frequency_table(ref, cfg)
    bank = FailureBank.from_episodes(ref)
    return freq, bank, ref


def test_config_defaults_load():
    cfg = load_config(REPO_CONFIG)
    assert cfg.rare_freq_quantile == 0.25
    assert cfg.high_freq_min_count == 3
    assert cfg.novelty_min_score == 1.0


def test_success_not_proxy_important(freq_bank, cfg):
    freq, bank, _ = freq_bank
    r = score_episode(_ep("ok", True, None), freq, bank, cfg)
    assert r.proxy_important is False
    assert "A" not in r.rule_hits
    assert r.rule_hits == ()


def test_rule_a_required_conjunction(freq_bank, cfg):
    """Success episodes never admit via B/C/D alone."""
    freq, bank, _ = freq_bank
    # Would be novel + rare-looking if it were a fail, but success → not important
    r = score_episode(_ep("ok-novel", True, "brand_new_type"), freq, bank, cfg)
    assert r.proxy_important is False


def test_rare_type_hits_b(freq_bank, cfg):
    freq, bank, _ = freq_bank
    assert freq.is_rare("grasping_error")
    r = score_episode(_ep("rare-1", False, "grasping_error"), freq, bank, cfg)
    assert "A" in r.rule_hits and "B" in r.rule_hits
    assert r.proxy_important is True
    assert r.frequency_bucket == FrequencyBucket.rare


def test_high_freq_hits_c(freq_bank, cfg):
    freq, bank, _ = freq_bank
    assert freq.is_high_freq("position_deviation")
    r = score_episode(_ep("common-1", False, "position_deviation"), freq, bank, cfg)
    assert "A" in r.rule_hits and "C" in r.rule_hits
    assert r.proxy_important is True
    assert r.frequency_bucket == FrequencyBucket.common


def test_novel_type_hits_d(freq_bank, cfg):
    freq, bank, _ = freq_bank
    assert "timing_error" not in bank.known_types
    r = score_episode(_ep("novel-1", False, "timing_error"), freq, bank, cfg)
    assert "A" in r.rule_hits and "D" in r.rule_hits
    assert r.novelty_score == 1.0
    assert r.proxy_important is True


def test_mid_known_type_not_important_without_bcd(freq_bank, cfg):
    """step_omission: count=2 → not rare (cutoff), not high-freq (≥3), in bank → no D."""
    freq, bank, _ = freq_bank
    assert not freq.is_rare("step_omission")
    # With quantile 0.25 on counts [1,2,4]: cutoff = sorted([1,2,4])[int(0.25*2)] = [1,2,4][0]=1
    # so only count≤1 is rare → grasping_error; step_omission count=2 is mid.
    assert freq.bucket("step_omission") == FrequencyBucket.mid
    assert not freq.is_high_freq("step_omission")
    assert "step_omission" in bank.known_types
    r = score_episode(_ep("mid-1", False, "step_omission"), freq, bank, cfg)
    assert r.proxy_important is False
    assert r.rule_hits == ("A",)


def test_conjunction_a_and_b_or_c_or_d(freq_bank, cfg):
    freq, bank, _ = freq_bank
    cases = [
        (_ep("b", False, "grasping_error"), True),
        (_ep("c", False, "position_deviation"), True),
        (_ep("d", False, "timing_error"), True),
        (_ep("mid", False, "step_omission"), False),
        (_ep("ok", True, "grasping_error"), False),
    ]
    for ep, expect in cases:
        assert score_episode(ep, freq, bank, cfg).proxy_important is expect


def test_deterministic_given_config(freq_bank, cfg):
    freq, bank, ref = freq_bank
    targets = [
        _ep("t0", False, "grasping_error"),
        _ep("t1", False, "position_deviation"),
        _ep("t2", True, None),
        _ep("t3", False, "timing_error"),
    ]
    a = score_episodes(targets, freq=freq, bank=bank, config=cfg)
    b = score_episodes(targets, freq=freq, bank=bank, config=cfg)
    assert a == b
    # Rebuilding freq/bank from same ref is deterministic
    freq2 = build_frequency_table(ref, cfg)
    bank2 = FailureBank.from_episodes(ref)
    c = score_episodes(targets, freq=freq2, bank=bank2, config=cfg)
    assert [r.proxy_important for r in a] == [r.proxy_important for r in c]


def test_apply_proxy_keeps_episode_lean(freq_bank, cfg):
    freq, bank, _ = freq_bank
    ep = _ep("x", False, "grasping_error")
    result = score_episode(ep, freq, bank, cfg)
    enriched = apply_proxy_to_ground_truth(ep, result)
    assert enriched.ground_truth.proxy_important is True
    assert enriched.ground_truth.proxy_rule_hits == list(result.rule_hits)
    # original unchanged
    assert ep.ground_truth.proxy_important is None


def test_ef_stubs_are_ranking_only(freq_bank, cfg):
    freq, bank, _ = freq_bank
    ep = _ep("x", False, "grasping_error")
    assert ranking_bonus_e_actionable_phrases(ep) == 0.0
    assert ranking_bonus_f_cascade_uncertainty(ep) == 0.0
    r = score_episode(ep, freq, bank, cfg)
    assert r.ranking_bonus == 0.0
    # E/F must not admit a mid known fail
    mid = score_episode(_ep("mid", False, "step_omission"), freq, bank, cfg)
    assert mid.proxy_important is False


def test_improvement_loop_cannot_use_hidden(tmp_path: Path, cfg):
    corpus = _ref_corpus() + [
        _ep("timing-new", False, "timing_error"),
        _ep("extra-ok", True, None),
    ]
    # Ensure enough ids
    ids = [e.episode_id for e in corpus]
    while len(ids) < 10:
        ids.append(f"pad-{len(ids)}")
        corpus.append(_ep(ids[-1], True, None))
    splits = assign_splits(ids, seed=42)
    write_split_manifests(splits, tmp_path)

    # Allowed path works
    results = score_for_improvement_loop(
        corpus,
        score_split="development",
        reference_splits=("development",),
        splits_dir=tmp_path,
        config=cfg,
    )
    assert isinstance(results, list)

    with pytest.raises(HiddenLabelAccessError):
        score_for_improvement_loop(
            corpus,
            score_split="hidden_eval",
            reference_splits=("development",),
            splits_dir=tmp_path,
            config=cfg,
        )

    with pytest.raises(HiddenLabelAccessError):
        build_reference_for_tuning(
            corpus,
            splits=("hidden_eval",),
            splits_dir=tmp_path,
            config=cfg,
        )


def test_empty_bank_everything_novel(cfg):
    bank = FailureBank()
    freq = build_frequency_table([], cfg)
    r = score_episode(_ep("cold", False, "anything"), freq, bank, cfg)
    assert "D" in r.rule_hits
    assert r.proxy_important is True
