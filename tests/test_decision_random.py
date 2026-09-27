"""RandomDecisionEngine: seeded, finite, order-independent."""

from __future__ import annotations

from decision.base import DecisionConfig
from decision.random import RandomDecisionEngine
from features.extract import EpisodeFeatures


def _feat(eid: str) -> EpisodeFeatures:
    return EpisodeFeatures(
        episode_id=eid,
        task=None,
        instruction=None,
        instruction_len=0,
        n_videos=0,
        n_frames=0,
        path_text=eid.lower(),
        has_failure_subtask_meta=False,
    )


def test_random_reproducible_and_finite():
    cfg = DecisionConfig(random_seed=7, default_fail_probability=0.5)
    eng = RandomDecisionEngine(cfg, failure_types=["position_deviation", "step_omission"])
    a = eng.evaluate(_feat("ep-a"))
    b = eng.evaluate(_feat("ep-a"))
    assert a == b
    assert 0.0 <= a.failure_probability <= 1.0
    assert a.backend == "random"
    assert a.confidence == 0.5


def test_random_order_independent():
    cfg = DecisionConfig(random_seed=42)
    eng = RandomDecisionEngine(cfg)
    ids = [f"e{i}" for i in range(20)]
    forward = {r.episode_id: r for r in eng.evaluate_many([_feat(i) for i in ids])}
    reverse = {r.episode_id: r for r in eng.evaluate_many([_feat(i) for i in reversed(ids)])}
    assert forward == reverse
