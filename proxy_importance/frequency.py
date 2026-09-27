"""Failure-type frequency table for rare (B) / high-freq (C) buckets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from data.schemas.episode import Episode, FrequencyBucket
from proxy_importance.config import ProxyImportanceConfig


@dataclass(frozen=True)
class FrequencyTable:
    """Counts of failure_type among reference *failures* only."""

    counts: dict[str, int]
    n_fails: int
    rare_count_cutoff: int
    high_freq_min_count: int

    def count_of(self, failure_type: str | None) -> int:
        if failure_type is None:
            return 0
        return self.counts.get(failure_type, 0)

    def rel_freq(self, failure_type: str | None) -> float:
        if self.n_fails == 0 or failure_type is None:
            return 0.0
        return self.count_of(failure_type) / self.n_fails

    def is_rare(self, failure_type: str | None) -> bool:
        """B: type appears in corpus and count ≤ rare quantile cutoff."""
        if failure_type is None or failure_type not in self.counts:
            return False
        return self.counts[failure_type] <= self.rare_count_cutoff

    def is_high_freq(self, failure_type: str | None) -> bool:
        """C: type absolute count ≥ high_freq_min_count."""
        if failure_type is None:
            return False
        return self.count_of(failure_type) >= self.high_freq_min_count

    def bucket(self, failure_type: str | None) -> FrequencyBucket | None:
        if failure_type is None:
            return None
        if self.is_rare(failure_type):
            return FrequencyBucket.rare
        if self.is_high_freq(failure_type):
            return FrequencyBucket.common
        if failure_type in self.counts:
            return FrequencyBucket.mid
        return None


def _quantile_cutoff(values: list[int], q: float) -> int:
    """Inclusive empirical quantile on a non-empty sorted list of ints."""
    if not values:
        return 0
    ordered = sorted(values)
    # nearest-rank: index = ceil(q * n) - 1, clamped
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    idx = max(0, min(len(ordered) - 1, int(q * len(ordered) + 1e-12) - 1))
    # Use classic inclusive: floor(q * (n-1))
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


def build_frequency_table(
    episodes: Iterable[Episode],
    config: ProxyImportanceConfig | None = None,
) -> FrequencyTable:
    """Build type counts from episodes where success is False."""
    cfg = config or ProxyImportanceConfig()
    counter: Counter[str] = Counter()
    n_fails = 0
    for ep in episodes:
        if ep.ground_truth.success is not False:
            continue
        n_fails += 1
        ft = ep.ground_truth.failure_type
        if ft is not None:
            counter[ft] += 1
    counts = dict(counter)
    type_counts = list(counts.values())
    cutoff = _quantile_cutoff(type_counts, cfg.rare_freq_quantile) if type_counts else 0
    return FrequencyTable(
        counts=counts,
        n_fails=n_fails,
        rare_count_cutoff=cutoff,
        high_freq_min_count=cfg.high_freq_min_count,
    )
