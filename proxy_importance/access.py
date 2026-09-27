"""Improvement-loop-safe helpers: never load hidden_eval for proxy tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from data.schemas.episode import Episode
from data.split_access import (
    AccessRole,
    HiddenLabelAccessError,
    assert_split_allowed,
    load_split_ids_for_role,
)
from proxy_importance.config import ProxyImportanceConfig, load_config
from proxy_importance.failure_bank import FailureBank
from proxy_importance.frequency import FrequencyTable, build_frequency_table
from proxy_importance.rules import ProxyReviewPriorityResult, score_episodes

# Splits allowed when building frequency / bank for improvement-loop / pipeline.
TUNING_SPLITS = frozenset({"development", "public_eval"})


def assert_episodes_split_allowed(split: str, role: AccessRole = AccessRole.IMPROVEMENT_LOOP) -> None:
    """Refuse hidden_eval for improvement-loop / pipeline proxy helpers."""
    assert_split_allowed(split, role)


def filter_episodes_by_split(
    episodes: Iterable[Episode],
    split: str,
    role: AccessRole = AccessRole.IMPROVEMENT_LOOP,
    splits_dir: str | Path | None = None,
) -> list[Episode]:
    """Keep episodes whose ids are in ``split``, after role check."""
    assert_split_allowed(split, role)
    allowed = set(load_split_ids_for_role(split, role, splits_dir=splits_dir))
    return [ep for ep in episodes if ep.episode_id in allowed]


def build_reference_for_tuning(
    episodes: Iterable[Episode],
    splits: Iterable[str] = ("development",),
    role: AccessRole = AccessRole.IMPROVEMENT_LOOP,
    splits_dir: str | Path | None = None,
    config: ProxyImportanceConfig | None = None,
) -> tuple[FrequencyTable, FailureBank]:
    """Build frequency + bank from tuning splits only (never hidden_eval)."""
    cfg = config or load_config()
    eps = list(episodes)
    ref: list[Episode] = []
    for split in splits:
        if split not in TUNING_SPLITS:
            raise HiddenLabelAccessError(
                f"proxy tuning cannot use split {split!r}; allowed={sorted(TUNING_SPLITS)}"
            )
        assert_split_allowed(split, role)
        ref.extend(filter_episodes_by_split(eps, split, role=role, splits_dir=splits_dir))
    return build_frequency_table(ref, cfg), FailureBank.from_episodes(ref)


def score_for_improvement_loop(
    episodes: Iterable[Episode],
    score_split: str,
    reference_splits: Iterable[str] = ("development",),
    splits_dir: str | Path | None = None,
    config: ProxyImportanceConfig | None = None,
) -> list[ProxyReviewPriorityResult]:
    """Score one non-hidden split; reference stats from development (and optional public)."""
    cfg = config or load_config()
    eps = list(episodes)
    freq, bank = build_reference_for_tuning(
        eps, splits=reference_splits, role=AccessRole.IMPROVEMENT_LOOP, splits_dir=splits_dir, config=cfg
    )
    target = filter_episodes_by_split(
        eps, score_split, role=AccessRole.IMPROVEMENT_LOOP, splits_dir=splits_dir
    )
    return score_episodes(target, freq=freq, bank=bank, config=cfg)
