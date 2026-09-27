"""Episode schema validation."""

from data.schemas.episode import SCHEMA_VERSION, Episode, GroundTruth, ModelOutput


def test_minimal_episode_defaults():
    ep = Episode(episode_id="ep-1")
    assert ep.schema_version == SCHEMA_VERSION == "0"
    assert ep.ground_truth.success is None
    assert ep.model_output.decision_backend is None
    assert ep.model_output.evidence == []


def test_episode_roundtrip_with_gt_and_backend():
    ep = Episode(
        episode_id="ep-2",
        task="MicrowaveTask",
        instruction="Put the spoon in the mug",
        video_paths=["a.mp4"],
        metadata={"source": "fixture"},
        ground_truth=GroundTruth(
            success=False,
            failure_type="position_deviation",
            diagnosis="missed grasp",
        ),
        model_output=ModelOutput(decision_backend="baseline", evidence=["frame:3"]),
    )
    data = ep.model_dump()
    restored = Episode.model_validate(data)
    assert restored.ground_truth.failure_type == "position_deviation"
    assert restored.model_output.decision_backend == "baseline"


def test_decision_backend_ladder_values():
    for backend in ("random", "frequency", "baseline", "jev"):
        ep = Episode(
            episode_id=f"ep-{backend}",
            model_output=ModelOutput(decision_backend=backend),
        )
        assert ep.model_output.decision_backend == backend
