"""Agglomerative clustering + purity (label consistency only)."""

from __future__ import annotations

from clustering import (
    Clusterer,
    agglomerative_fit_predict,
    cluster_purity,
    compression_ratio,
    hash_embed,
)
from failure_bank import FailureCase


def _case(cid: str, path: str, instruction: str, ftype: str) -> FailureCase:
    return FailureCase(
        case_id=cid,
        episode_id=cid,
        failure_type=ftype,
        task="T",
        instruction=instruction,
        path_text=path.lower(),
    )


def test_purity_perfect_when_labels_match_clusters():
    labels = ["a", "a", "b", "b", "c"]
    clusters = [0, 0, 1, 1, 2]
    assert cluster_purity(clusters, labels) == 1.0


def test_purity_mixed_cluster():
    # One impure cluster of 4: majority a (3/4) + singleton b.
    labels = ["a", "a", "a", "b", "b"]
    clusters = [0, 0, 0, 0, 1]
    # correct = 3 (cluster0) + 1 (cluster1) = 4 / 5
    assert abs(cluster_purity(clusters, labels) - 0.8) < 1e-9


def test_purity_none_labels_do_not_inflate():
    labels = ["a", None, "a"]
    clusters = [0, 0, 0]
    # majority a with 2 votes; correct=2 / 3
    assert abs(cluster_purity(clusters, labels) - 2 / 3) < 1e-9


def test_purity_empty():
    assert cluster_purity([], []) == 0.0


def test_hash_embed_deterministic():
    a = hash_embed(["fail grasp left"], dim=32)
    b = hash_embed(["fail grasp left"], dim=32)
    assert a == b
    assert len(a[0]) == 32


def test_agglomerative_identical_vectors_merge():
    # Same text → distance 0 → one cluster under any reasonable threshold.
    vecs = hash_embed(["same path fail grasp", "same path fail grasp", "same path fail grasp"])
    res = agglomerative_fit_predict(vecs, n_clusters=None, distance_threshold=0.1)
    assert res.n_clusters == 1
    assert res.labels == [0, 0, 0]


def test_agglomerative_respects_n_clusters():
    vecs = hash_embed(
        [
            "path fail grasp alpha",
            "path fail grasp beta",
            "path omit step gamma",
            "path omit step delta",
            "path timing error epsilon",
        ],
        dim=64,
    )
    res = agglomerative_fit_predict(vecs, n_clusters=2, distance_threshold=None)
    assert res.n_clusters == 2
    assert sorted(set(res.labels)) == [0, 1]


def test_clusterer_on_synthetic_typed_cases():
    """Same-type cases sharing path tokens should form label-consistent groups."""
    cases = [
        _case("g0", "sim/fail_grasp/v0/a.mp4", "grasp object", "grasping_error"),
        _case("g1", "sim/fail_grasp/v0/b.mp4", "grasp cup", "grasping_error"),
        _case("g2", "sim/fail_grasp/v1/c.mp4", "grasp bowl", "grasping_error"),
        _case("o0", "sim/fail_omit/v0/a.mp4", "omit place step", "step_omission"),
        _case("o1", "sim/fail_omit/v0/b.mp4", "omit reach step", "step_omission"),
        _case("t0", "sim/fail_timing/v0/a.mp4", "timing late close", "timing_error"),
        _case("t1", "sim/fail_timing/v0/b.mp4", "timing early open", "timing_error"),
    ]
    # Force fewer clusters than cases so compression happens.
    cres = Clusterer(dim=64, n_clusters=3, distance_threshold=None).fit_predict(cases)
    assert cres.n_clusters == 3
    purity = cluster_purity(cres.labels, [c.failure_type for c in cases])
    # With path tokens aligned to types, purity should be strong.
    assert purity >= 0.7
    assert compression_ratio(len(cases), cres.n_clusters) == len(cases) / 3


def test_compression_ratio():
    assert compression_ratio(10, 2) == 5.0
    assert compression_ratio(0, 0) == 0.0


def test_purity_ignore_none_scores_labeled_only():
    labels = ["a", None, "a", "b", None]
    clusters = [0, 0, 0, 1, 1]
    # all-denom: majority a gets 2 in c0; c1 has one b → correct=3 / 5
    assert abs(cluster_purity(clusters, labels, ignore_none=False) - 0.6) < 1e-9
    # labeled-only: points (a,a,b) in clusters (0,0,1) → correct=3 / 3
    assert cluster_purity(clusters, labels, ignore_none=True) == 1.0
