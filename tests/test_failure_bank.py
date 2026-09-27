"""Failure bank JSONL round-trip + ranking A–D set / E–F bonuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.schemas.episode import Episode, GroundTruth, ModelOutput
from failure_bank import FailureBank, FailureCase
from failure_bank.store import case_from_episode
from proxy_importance.config import ProxyImportanceConfig
from proxy_importance.failure_bank import FailureBank as ProxyTypeBank
from proxy_importance.frequency import build_frequency_table
from ranking import Ranker, RankingWeights, compute_bonus_e, compute_bonus_f


def _ep(
    eid: str,
    *,
    success: bool = False,
    failure_type: str | None = "grasping_error",
    instruction: str | None = None,
    diagnosis: str | None = None,
    correction: str | None = None,
    confidence: float | None = None,
    needs_deep: bool | None = None,
    path: str = "sim/fail_grasp/view0/x.mp4",
) -> Episode:
    return Episode(
        episode_id=eid,
        task="PickPlace",
        instruction=instruction or f"do {failure_type}",
        video_paths=[path],
        ground_truth=GroundTruth(
            success=success,
            failure_type=failure_type if not success else None,
            diagnosis=diagnosis,
            correction=correction,
        ),
        model_output=ModelOutput(confidence=confidence, needs_deep_review=needs_deep),
    )


@pytest.fixture
def cfg() -> ProxyImportanceConfig:
    return ProxyImportanceConfig(
        rare_freq_quantile=0.25,
        high_freq_min_count=3,
        novelty_min_score=1.0,
    )


def test_jsonl_round_trip(tmp_path: Path, cfg: ProxyImportanceConfig):
    ref = [
        _ep("r0", failure_type="position_deviation"),
        _ep("r1", failure_type="position_deviation"),
        _ep("r2", failure_type="position_deviation"),
        _ep("r3", failure_type="step_omission"),
        _ep("r4", failure_type="grasping_error"),
        _ep("ok", success=True, failure_type=None),
    ]
    bank = FailureBank.from_episodes(ref, reference=ref, config=cfg, path=tmp_path / "bank.jsonl")
    assert len(bank) == 5  # successes skipped
    path = bank.save()
    assert path.is_file()

    loaded = FailureBank.from_jsonl(path)
    assert len(loaded) == 5
    assert loaded.known_types() == {
        "position_deviation",
        "step_omission",
        "grasping_error",
    }
    assert loaded.type_frequency("position_deviation") == 3
    assert loaded.novelty_score("timing_error") == 1.0
    assert loaded.novelty_score("grasping_error") == 0.0
    assert loaded.is_novel("timing_error")
    assert not loaded.is_novel("grasping_error")

    # Proxy bridge still exposes type-set novelty.
    proxy = loaded.to_proxy_bank()
    assert "grasping_error" in proxy.known_types
    assert proxy.novelty_score("timing_error") == 1.0


def test_upsert_overwrites(tmp_path: Path):
    bank = FailureBank(path=tmp_path / "b.jsonl")
    bank.upsert(
        FailureCase(case_id="a", episode_id="a", failure_type="x", novelty_score=1.0)
    )
    bank.upsert(
        FailureCase(case_id="a", episode_id="a", failure_type="y", novelty_score=0.0)
    )
    assert len(bank) == 1
    assert bank.get("a") is not None
    assert bank.get("a").failure_type == "y"


def test_proxy_type_bank_api_unchanged(cfg: ProxyImportanceConfig):
    """Existing proxy_importance FailureBank still works for rule D."""
    ref = [_ep("a", failure_type="position_deviation")]
    bank = ProxyTypeBank.from_episodes(ref)
    assert bank.novelty_score("position_deviation") == 0.0
    assert bank.novelty_score("new_type") == 1.0
    assert bank.is_novel("new_type")


def test_ranking_honors_ad_set_and_ef_bonuses(cfg: ProxyImportanceConfig):
    ref = [
        _ep(f"pos-{i}", failure_type="position_deviation") for i in range(4)
    ] + [
        _ep("omit-0", failure_type="step_omission"),
        _ep("omit-1", failure_type="step_omission"),
        _ep("grasp-0", failure_type="grasping_error"),
    ]
    freq = build_frequency_table(ref, cfg)
    proxy_bank = ProxyTypeBank.from_episodes(ref)

    rare = case_from_episode(_ep("rare", failure_type="grasping_error"), freq, proxy_bank, cfg)
    high = case_from_episode(
        _ep("high", failure_type="position_deviation"), freq, proxy_bank, cfg
    )
    novel = case_from_episode(_ep("novel", failure_type="timing_error"), freq, proxy_bank, cfg)
    mid = case_from_episode(_ep("mid", failure_type="step_omission"), freq, proxy_bank, cfg)
    ok = case_from_episode(_ep("ok", success=True, failure_type=None), freq, proxy_bank, cfg)

    assert rare.proxy_important and "B" in rare.rule_hits
    assert high.proxy_important and "C" in high.rule_hits
    assert novel.proxy_important and "D" in novel.rule_hits
    assert not mid.proxy_important
    assert not ok.proxy_important

    weights = RankingWeights(
        a=1.0, b=1.0, c=1.0, d=1.5, e=0.5, f=0.5,
        require_a=True, prefer_proxy_important=True, proxy_important_boost=2.0,
    )
    ranker = Ranker(weights)

    assert ranker.score(ok) == 0.0
    assert ranker.score(mid) > 0.0  # has A
    assert ranker.score(rare) > ranker.score(mid)
    assert ranker.score(novel) > ranker.score(mid)

    # E/F boost ranking but do not admit mid into proxy_important.
    mid_boosted = FailureCase(
        case_id="mid-ef",
        episode_id="mid-ef",
        failure_type="step_omission",
        rule_hits=("A",),
        proxy_important=False,
        bonus_e=1.0,
        bonus_f=1.0,
    )
    assert mid_boosted.proxy_important is False
    assert ranker.score(mid_boosted) > ranker.score(mid)
    # Still below a proxy-important case with boost.
    assert ranker.score(rare) > ranker.score(mid_boosted)


def test_bonus_e_actionable_phrases():
    assert compute_bonus_e(None, None, None) == 0.0
    assert compute_bonus_e("please retry the grasp", None) > 0.0
    assert compute_bonus_e(None, "reposition and adjust") >= 0.5


def test_bonus_f_uncertainty():
    hi = _ep("a", confidence=0.9)
    lo = _ep("b", confidence=0.2)
    deep = _ep("c", needs_deep=True)
    assert compute_bonus_f(hi) == 0.0
    assert compute_bonus_f(lo) > 0.0
    assert compute_bonus_f(deep) == 1.0


def test_no_streamlit_import_in_m2_modules():
    """Guard: M2 packages must not pull Streamlit."""
    import failure_bank
    import clustering
    import ranking
    import scripts.m2_cluster_report as m2

    for mod in (failure_bank, clustering, ranking, m2):
        assert "streamlit" not in dir(mod)
