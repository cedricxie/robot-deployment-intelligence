"""GT-free EpisodeFeatures — never reads ground_truth for prediction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from data.schemas.episode import Episode
from features.sanitize import (
    collect_leak_tokens,
    path_leak_ablate_enabled,
    sanitize_path_text,
)


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


def extract_features(
    episode: Episode,
    *,
    ablate_path_leak: bool | None = None,
    leak_tokens: Iterable[str] | None = None,
) -> EpisodeFeatures:
    """Extract features without reading ``ground_truth`` / ``model_output``.

    When path-leak ablation is on (``ablate_path_leak=True`` or ``LEAK_ABLATE=1``),
    success/fail directory cues are stripped from ``path_text``.
    """
    paths = [p or "" for p in episode.video_paths]
    frames = episode.frames or []
    instruction = episode.instruction
    path_text = " ".join([episode.episode_id, *paths]).lower()
    if path_leak_ablate_enabled(ablate_path_leak):
        tokens = (
            tuple(leak_tokens)
            if leak_tokens is not None
            else collect_leak_tokens()
        )
        path_text = sanitize_path_text(path_text, tokens)
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

    def __init__(self, *, ablate_path_leak: bool | None = None) -> None:
        self.ablate_path_leak = ablate_path_leak

    def extract(self, episode: Episode) -> EpisodeFeatures:
        return extract_features(episode, ablate_path_leak=self.ablate_path_leak)

    def extract_many(self, episodes: Iterable[Episode]) -> list[EpisodeFeatures]:
        return [self.extract(ep) for ep in episodes]
