"""Pure-Python agglomerative clustering (average linkage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from clustering.embed import cosine_distance, embed_cases
from failure_bank.case import FailureCase


@dataclass(frozen=True)
class ClusterResult:
    labels: list[int]  # cluster id per case (0..k-1 contiguous)
    n_clusters: int
    distances_merged: list[float]


def _cluster_distance(
    members_a: list[int],
    members_b: list[int],
    dist: list[list[float]],
    linkage: str,
) -> float:
    vals = [dist[i][j] for i in members_a for j in members_b]
    if not vals:
        return 0.0
    if linkage == "single":
        return min(vals)
    if linkage == "complete":
        return max(vals)
    # average
    return sum(vals) / len(vals)


def agglomerative_fit_predict(
    vectors: Sequence[Sequence[float]],
    *,
    n_clusters: int | None = None,
    distance_threshold: float | None = 0.55,
    linkage: str = "average",
) -> ClusterResult:
    """Agglomerative clustering on precomputed vectors.

    Stop when ``n_clusters`` reached **or** next merge distance exceeds
    ``distance_threshold`` (whichever applies). If both None → one cluster.
    """
    n = len(vectors)
    if n == 0:
        return ClusterResult(labels=[], n_clusters=0, distances_merged=[])
    if n == 1:
        return ClusterResult(labels=[0], n_clusters=1, distances_merged=[])

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_distance(vectors[i], vectors[j])
            dist[i][j] = dist[j][i] = d

    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    next_id = n
    merged_dists: list[float] = []
    target = n_clusters if n_clusters is not None else 1

    while len(clusters) > target:
        ids = sorted(clusters.keys())
        best_d = float("inf")
        best_pair: tuple[int, int] | None = None
        for ai, a in enumerate(ids):
            for b in ids[ai + 1 :]:
                d = _cluster_distance(clusters[a], clusters[b], dist, linkage)
                if d < best_d:
                    best_d = d
                    best_pair = (a, b)
        if best_pair is None:
            break
        if (
            n_clusters is None
            and distance_threshold is not None
            and best_d > distance_threshold
        ):
            break
        a, b = best_pair
        merged_dists.append(best_d)
        clusters[next_id] = clusters.pop(a) + clusters.pop(b)
        next_id += 1

    # Assign contiguous labels in stable member-min order.
    ordered = sorted(clusters.values(), key=lambda members: min(members))
    labels = [0] * n
    for cid, members in enumerate(ordered):
        for m in members:
            labels[m] = cid
    return ClusterResult(
        labels=labels,
        n_clusters=len(ordered),
        distances_merged=merged_dists,
    )


class Clusterer:
    """``Clusterer.fit_predict(cases) -> ClusterResult`` (plan §6.2)."""

    def __init__(
        self,
        *,
        dim: int = 64,
        n_clusters: int | None = None,
        distance_threshold: float | None = 0.55,
        linkage: str = "average",
    ) -> None:
        self.dim = dim
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage

    def fit_predict(self, cases: Sequence[FailureCase]) -> ClusterResult:
        vectors = embed_cases(cases, dim=self.dim)
        return agglomerative_fit_predict(
            vectors,
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
        )
