"""Split writer: 60/20/20 seed 42, no ID overlap, hidden isolation."""

from pathlib import Path

import pytest

from data.split_access import (
    DEFAULT_SEED,
    AccessRole,
    HiddenLabelAccessError,
    SPLIT_NAMES,
    assign_splits,
    improvement_loop_load_split,
    load_split_ids_for_role,
    read_split_ids,
    write_split_manifests,
)

REPO_SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def test_assign_splits_ratios_and_no_overlap():
    ids = [f"ep-{i:03d}" for i in range(100)]
    splits = assign_splits(ids, seed=DEFAULT_SEED)
    assert set(splits) == set(SPLIT_NAMES)
    assert len(splits["development"]) == 60
    assert len(splits["public_eval"]) == 20
    assert len(splits["hidden_eval"]) == 20

    all_ids = (
        splits["development"] + splits["public_eval"] + splits["hidden_eval"]
    )
    assert len(all_ids) == 100
    assert len(set(all_ids)) == 100
    assert set(all_ids) == set(ids)

    # disjoint pairwise
    assert not set(splits["development"]) & set(splits["public_eval"])
    assert not set(splits["development"]) & set(splits["hidden_eval"])
    assert not set(splits["public_eval"]) & set(splits["hidden_eval"])


def test_assign_splits_deterministic():
    ids = [f"ep-{i}" for i in range(50)]
    a = assign_splits(ids, seed=42)
    b = assign_splits(ids, seed=42)
    c = assign_splits(ids, seed=0)
    assert a == b
    assert a != c


def test_write_and_read_manifests(tmp_path: Path):
    ids = [f"ep-{i:02d}" for i in range(20)]
    splits = assign_splits(ids, seed=42)
    write_split_manifests(splits, tmp_path)
    for name in SPLIT_NAMES:
        assert read_split_ids(name, tmp_path) == splits[name]


def test_committed_manifests_no_overlap():
    """Committed data/splits manifests are disjoint and non-empty."""
    loaded = {name: read_split_ids(name, REPO_SPLITS) for name in SPLIT_NAMES}
    assert all(loaded[n] for n in SPLIT_NAMES)
    all_ids = loaded["development"] + loaded["public_eval"] + loaded["hidden_eval"]
    assert len(all_ids) == len(set(all_ids))
    n = len(all_ids)
    # within tolerance of 60/20/20 (int truncation)
    assert abs(len(loaded["development"]) / n - 0.60) <= 0.05
    assert abs(len(loaded["public_eval"]) / n - 0.20) <= 0.05
    assert abs(len(loaded["hidden_eval"]) / n - 0.20) <= 0.05


def test_improvement_loop_cannot_load_hidden_labels(tmp_path: Path):
    ids = [f"ep-{i}" for i in range(10)]
    splits = assign_splits(ids, seed=42)
    write_split_manifests(splits, tmp_path)

    # Allowed splits work.
    assert improvement_loop_load_split("development", tmp_path) == splits["development"]
    assert improvement_loop_load_split("public_eval", tmp_path) == splits["public_eval"]

    with pytest.raises(HiddenLabelAccessError, match="hidden_eval"):
        improvement_loop_load_split("hidden_eval", tmp_path)

    with pytest.raises(HiddenLabelAccessError):
        load_split_ids_for_role("hidden_eval", AccessRole.IMPROVEMENT_LOOP, tmp_path)

    with pytest.raises(HiddenLabelAccessError):
        load_split_ids_for_role("hidden_eval", AccessRole.PIPELINE, tmp_path)

    # Harness final scoring may load hidden ids.
    assert (
        load_split_ids_for_role("hidden_eval", AccessRole.HARNESS_FINAL, tmp_path)
        == splits["hidden_eval"]
    )
