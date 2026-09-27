"""RoboFAC GT mapping: canonical types, idempotent remap, fixture agreement."""

from pathlib import Path

from data.adapters.robofac import RoboFACAdapter, load_episodes
from data.gt_mapping import apply_gt_mapping, map_ground_truth
from data.schemas.episode import Episode, GroundTruth
from ontology.robofac_type_map import (
    OTHER,
    ROBOFAC_CANONICAL_TYPES,
    map_robofac_failure_type,
    map_robofac_success,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples" / "robofac"


def test_six_class_aliases_map_to_canonical():
    cases = {
        "Position deviation.": "position_deviation",
        "orientation deviation": "orientation_deviation",
        "Step omission.": "step_omission",
        "Wrong target object.": "wrong_target_object",
        "Timing error.": "timing_error",
        "Grasping error.": "grasping_error",
    }
    for raw, expected in cases.items():
        assert map_robofac_failure_type(raw) == expected
        assert expected in ROBOFAC_CANONICAL_TYPES


def test_remapping_idempotent():
    for canon in ROBOFAC_CANONICAL_TYPES:
        assert map_robofac_failure_type(canon) == canon
        assert map_robofac_failure_type(canon.upper()) == canon

    gt = GroundTruth(success=False, failure_type="Position deviation.")
    once = map_ground_truth(gt)
    twice = map_ground_truth(once)
    assert once.failure_type == "position_deviation"
    assert twice.failure_type == once.failure_type
    assert twice.success == once.success

    ep = Episode(episode_id="x", ground_truth=gt)
    assert apply_gt_mapping(apply_gt_mapping(ep)).ground_truth == apply_gt_mapping(ep).ground_truth


def test_unknown_raw_becomes_other_blank_stays_none():
    assert map_robofac_failure_type("Totally novel failure") == OTHER
    assert map_robofac_failure_type(None) is None
    assert map_robofac_failure_type("  ") is None
    assert map_robofac_success(True) is True
    assert map_robofac_success(None) is None


def test_fixture_mapped_type_agreement():
    pos = list(RoboFACAdapter(SAMPLES / "sim_failure_position_deviation.json").load())[0]
    omit = list(RoboFACAdapter(SAMPLES / "sim_failure_step_omission.json").load())[0]
    ok = list(RoboFACAdapter(SAMPLES / "sim_success.json").load())[0]

    assert pos.ground_truth.success is False
    assert pos.ground_truth.failure_type == "position_deviation"
    assert omit.ground_truth.failure_type == "step_omission"
    assert ok.ground_truth.success is True
    assert ok.ground_truth.failure_type is None

    # apply_gt_mapping is a no-op on already-mapped adapter output
    for ep in (pos, omit, ok):
        assert apply_gt_mapping(ep).ground_truth == ep.ground_truth


def test_load_all_fixtures_canonical():
    eps = load_episodes(sorted(SAMPLES.glob("*.json")))
    for ep in eps:
        ft = ep.ground_truth.failure_type
        if ft is not None:
            assert ft in ROBOFAC_CANONICAL_TYPES or ft == OTHER
