"""Auto cluster purity vs failure_type = label consistency only.

Does **not** measure engineering usefulness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence


def cluster_purity(
    cluster_ids: Sequence[int],
    labels: Sequence[str | None],
    *,
    ignore_none: bool = False,
) -> float:
    """Weighted purity: majority-label fraction per cluster.

    When ``ignore_none`` is False (default), ``None`` labels do not vote for
    majority but still count in the denominator (cannot be "correct").

    When ``ignore_none`` is True, only points with a non-None label are scored
    (recommended when many fails lack mapped ``failure_type``).
    """
    if len(cluster_ids) != len(labels):
        raise ValueError("cluster_ids and labels length mismatch")
    if ignore_none:
        pairs = [(c, l) for c, l in zip(cluster_ids, labels) if l is not None]
        if not pairs:
            return 0.0
        cluster_ids = [c for c, _ in pairs]
        labels = [l for _, l in pairs]

    n = len(cluster_ids)
    if n == 0:
        return 0.0

    by_cluster: dict[int, list[str | None]] = defaultdict(list)
    for cid, lab in zip(cluster_ids, labels):
        by_cluster[cid].append(lab)

    correct = 0
    for labs in by_cluster.values():
        votes = Counter(x for x in labs if x is not None)
        if not votes:
            continue
        correct += votes.most_common(1)[0][1]
    return correct / n


def cluster_size_distribution(cluster_ids: Sequence[int]) -> dict[int, int]:
    """Map cluster_id → size."""
    return dict(Counter(cluster_ids))


def compression_ratio(n_cases: int, n_clusters: int) -> float:
    """n_cases / n_clusters (how many fails per cluster)."""
    if n_clusters <= 0:
        return 0.0
    return n_cases / n_clusters
