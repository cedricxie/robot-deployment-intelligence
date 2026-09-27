"""Feature extraction stays GT-free."""

from data.schemas.episode import Episode, GroundTruth
from features.extract import FeatureExtractor, extract_features


def test_extract_ignores_ground_truth():
    ep = Episode(
        episode_id="id-fail-hint",
        instruction="do thing",
        video_paths=["a/fail.mp4"],
        metadata={"failure_subtask": "reach"},
        ground_truth=GroundTruth(success=False, failure_type="position_deviation"),
    )
    feats = extract_features(ep)
    assert feats.episode_id == "id-fail-hint"
    assert feats.instruction_len == len("do thing")
    assert feats.n_videos == 1
    assert "fail" in feats.path_text
    assert feats.has_failure_subtask_meta is True
    # Same features if GT flipped
    ep2 = ep.model_copy(
        update={"ground_truth": GroundTruth(success=True, failure_type=None)}
    )
    assert extract_features(ep2).path_text == feats.path_text


def test_feature_extractor_many():
    eps = [Episode(episode_id=f"e{i}") for i in range(3)]
    out = FeatureExtractor().extract_many(eps)
    assert [f.episode_id for f in out] == ["e0", "e1", "e2"]
