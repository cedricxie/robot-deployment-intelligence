"""E/F ranking bonuses only — never admit into proxy-important set."""

from __future__ import annotations

from data.schemas.episode import Episode

# Simple actionable-phrase heuristic (diagnosis / correction / instruction).
_ACTIONABLE_PHRASES = (
    "retry",
    "regrasp",
    "re-grasp",
    "adjust",
    "slow down",
    "reposition",
    "increase",
    "decrease",
    "check",
    "realign",
    "open gripper",
    "close gripper",
)


def compute_bonus_e(
    diagnosis: str | None,
    correction: str | None,
    instruction: str | None = None,
) -> float:
    """E: actionable-phrase hits in text → [0, 1]. Ranking only."""
    text = " ".join(t for t in (diagnosis, correction, instruction) if t).lower()
    if not text:
        return 0.0
    hits = sum(1 for p in _ACTIONABLE_PHRASES if p in text)
    return min(1.0, 0.5 * hits)


def compute_bonus_f(episode: Episode, low_conf: float = 0.5) -> float:
    """F: cascade-uncertainty / low confidence / needs_deep_review → [0, 1]."""
    mo = episode.model_output
    if mo.needs_deep_review:
        return 1.0
    if mo.confidence is not None and mo.confidence < low_conf:
        return max(0.0, min(1.0, 1.0 - float(mo.confidence)))
    return 0.0
