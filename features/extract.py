"""GT-free EpisodeFeatures — never reads ground_truth for prediction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data.schemas.episode import Episode


@dataclass(frozen=True)
class EpisodeFeatures:
    """Lightweight signals derived from non-GT Episode fields only."""

    episode_id: str
    task: str | None
    instruction: str | None
    instruction_len: int
    n_videos: int
    n_frames: int
    path_text: str  # lowercased id + paths for keyword heuristics
    has_failure_subtask_meta: bool


def extract_features(episode: Episode) -> EpisodeFeatures:
    """Extract features without reading ``ground_truth`` / ``model_output``."""
    paths = [p or "" for p in episode.video_paths]
    frames = episode.frames or []
    instruction = episode.instruction
    path_text = " ".join([episode.episode_id, *paths]).lower()
    meta = episode.metadata or {}
    return EpisodeFeatures(
        episode_id=episode.episode_id,
        task=episode.task,
        instruction=instruction,
        instruction_len=len(instruction) if instruction else 0,
        n_videos=len(paths),
        n_frames=len(frames),
        path_text=path_text,
        has_failure_subtask_meta=bool(meta.get("failure_subtask")),
    )


class FeatureExtractor:
    """Thin wrapper matching plan §6.2 ``FeatureExtractor.extract``."""

    def extract(self, episode: Episode) -> EpisodeFeatures:
        return extract_features(episode)

    def extract_many(self, episodes: Iterable[Episode]) -> list[EpisodeFeatures]:
        return [self.extract(ep) for ep in episodes]
