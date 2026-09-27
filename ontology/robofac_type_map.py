"""RoboFAC six-class failure_type → canonical GroundTruth.failure_type.

Canonical ids are snake_case. Prefer this map over inventing new types;
extend only with ``other`` / ``unknown`` (plan §7 freeze #10).
"""

from __future__ import annotations

# RoboFAC paper predefined identification set (punctuation-stripped keys).
ROBOFAC_CANONICAL_TYPES: frozenset[str] = frozenset(
    {
        "position_deviation",
        "orientation_deviation",
        "step_omission",
        "wrong_target_object",
        "timing_error",
        "grasping_error",
    }
)

OTHER = "other"
UNKNOWN = "unknown"

# Surface strings seen in RoboFAC QA (and already-canonical ids) → canonical.
_ALIASES: dict[str, str] = {
    "position deviation": "position_deviation",
    "orientation deviation": "orientation_deviation",
    "step omission": "step_omission",
    "wrong target object": "wrong_target_object",
    "timing error": "timing_error",
    "grasping error": "grasping_error",
    # idempotent: already-canonical
    "position_deviation": "position_deviation",
    "orientation_deviation": "orientation_deviation",
    "step_omission": "step_omission",
    "wrong_target_object": "wrong_target_object",
    "timing_error": "timing_error",
    "grasping_error": "grasping_error",
    "other": OTHER,
    "unknown": UNKNOWN,
}


def _normalize_key(text: str) -> str:
    return text.rstrip(".").strip().lower()


def map_robofac_failure_type(raw: str | None) -> str | None:
    """Map a RoboFAC failure-identification string to a canonical id.

    - ``None`` / blank → ``None`` (typical for successful episodes)
    - known six-class (any punctuation/case) → snake_case id
    - already-canonical id → same id (idempotent)
    - anything else → ``other``
    """
    if raw is None:
        return None
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    return OTHER


def map_robofac_success(raw: bool | None) -> bool | None:
    """Pass-through for RoboFAC success/fail (already boolean when parsed)."""
    return raw
