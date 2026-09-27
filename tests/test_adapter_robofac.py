"""RoboFACAdapter maps fixtures → Episode without leaking QA section names."""

from pathlib import Path

from data.adapters.robofac import RoboFACAdapter, load_episodes
from data.schemas.episode import Episode

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples" / "robofac"

# Core Episode fields / nested keys must not expose RoboFAC QA section titles.
_LEAK_MARKERS = (
    "Failure detection",
    "Failure identification",
    "Failure explanation",
    "Failure locating",
    "Task identification",
    "High-level correction",
    "annos",
)


def _assert_no_leak(ep: Episode) -> None:
    blob = ep.model_dump()
    flat_keys = set(blob) | set(blob.get("ground_truth", {})) | set(blob.get("model_output", {}))
    for marker in _LEAK_MARKERS:
        assert marker not in flat_keys
        assert marker not in ep.metadata


def test_adapter_success_fixture():
    eps = list(RoboFACAdapter(SAMPLES / "sim_success.json").load())
    assert len(eps) == 1
    ep = eps[0]
    assert ep.schema_version == "0"
    assert ep.episode_id == "fixture-sim-success-001"
    assert ep.task == "SafeTask"
    assert ep.ground_truth.success is True
    assert ep.ground_truth.failure_type is None
    assert ep.instruction
    assert ep.video_paths
    assert ep.metadata["source"] == "robofac"
    _assert_no_leak(ep)


def test_adapter_failure_position_deviation():
    eps = list(RoboFACAdapter(SAMPLES / "sim_failure_position_deviation.json").load())
    assert len(eps) == 1
    ep = eps[0]
    assert ep.ground_truth.success is False
    assert ep.ground_truth.failure_type == "position_deviation"
    assert ep.ground_truth.diagnosis
    assert ep.ground_truth.correction
    assert ep.metadata.get("failure_subtask")
    _assert_no_leak(ep)


def test_adapter_failure_step_omission():
    eps = list(RoboFACAdapter(SAMPLES / "sim_failure_step_omission.json").load())
    assert len(eps) == 1
    ep = eps[0]
    assert ep.ground_truth.success is False
    assert ep.ground_truth.failure_type == "step_omission"
    _assert_no_leak(ep)


def test_load_all_sample_fixtures():
    paths = sorted(SAMPLES.glob("*.json"))
    assert len(paths) >= 2
    eps = load_episodes(paths)
    assert len(eps) >= 3
    successes = [e for e in eps if e.ground_truth.success is True]
    failures = [e for e in eps if e.ground_truth.success is False]
    assert successes
    assert len(failures) >= 1
    types = {e.ground_truth.failure_type for e in failures}
    assert "position_deviation" in types
