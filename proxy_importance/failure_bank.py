"""Minimal Failure Bank for novelty (D): known failure_type set.

Full bank / clustering lands in later PRs. Cold-start: empty bank → all
unseen types score novelty 1.0 (seed from development failures in reports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from data.schemas.episode import Episode


@dataclass
class FailureBank:
    """Known failure patterns keyed by canonical failure_type."""

    known_types: set[str] = field(default_factory=set)

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode]) -> FailureBank:
        types: set[str] = set()
        for ep in episodes:
            if ep.ground_truth.success is False and ep.ground_truth.failure_type:
                types.add(ep.ground_truth.failure_type)
        return cls(known_types=types)

    def novelty_score(self, failure_type: str | None) -> float:
        """1.0 if type is missing/unknown to the bank; else 0.0."""
        if failure_type is None:
            return 1.0
        if failure_type not in self.known_types:
            return 1.0
        return 0.0

    def is_novel(self, failure_type: str | None, min_score: float = 1.0) -> bool:
        return self.novelty_score(failure_type) >= min_score
