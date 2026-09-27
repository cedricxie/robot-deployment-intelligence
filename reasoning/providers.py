"""VisionReasoningProvider — mock default (no network / API keys)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from features.extract import EpisodeFeatures


class VisionReasoningProvider(ABC):
    """Plan §6.2: summarize(frames, prompt) -> str."""

    name: str

    @abstractmethod
    def summarize(self, frames: Sequence[str] | None, prompt: str) -> str:
        ...


class MockVisionReasoningProvider(VisionReasoningProvider):
    """Deterministic stand-in: derives a short summary from features/path text.

    No LLM, no network. CI-safe. Optional ``features`` side-channel via
    ``summarize_features`` used by DeepReasoner when frames are empty.
    """

    name = "mock_vision"

    def __init__(self, *, cost_units_per_call: float = 5.0) -> None:
        self.cost_units_per_call = float(cost_units_per_call)
        self.calls = 0
        self.tokens = 0
        self.stub_units = 0.0

    def reset_cost(self) -> None:
        self.calls = 0
        self.tokens = 0
        self.stub_units = 0.0

    def summarize(self, frames: Sequence[str] | None, prompt: str) -> str:
        self.calls += 1
        blob = " ".join(frames or ())
        text = f"{prompt}\n{blob}".lower()
        tokens = max(8, len(text) // 4)
        self.tokens += tokens
        self.stub_units += self.cost_units_per_call
        return self._summary_from_text(text)

    def summarize_features(self, features: EpisodeFeatures, prompt: str) -> str:
        """GT-free summary from path/instruction when no real frames exist."""
        frames = [
            features.path_text,
            features.instruction or "",
            features.task or "",
        ]
        return self.summarize(frames, prompt)

    @staticmethod
    def _summary_from_text(text: str) -> str:
        fail_toks = ("fail", "failure", "error", "slip", "omit", "offset", "wrong")
        ok_toks = ("success", "success_cleaned", "stack_ok", "successfully")
        hit_fail = any(t in text for t in fail_toks)
        hit_ok = any(t in text for t in ok_toks)
        if hit_fail and not hit_ok:
            return "mock_vision: trajectory looks failed (path/instruction cues)."
        if hit_ok and not hit_fail:
            return "mock_vision: trajectory looks successful (path/instruction cues)."
        if hit_fail and hit_ok:
            return "mock_vision: mixed cues; lean fail."
        return "mock_vision: unclear; no strong path cues."
