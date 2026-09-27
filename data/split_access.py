"""Split manifest I/O + hidden-label isolation for improvement-loop helpers."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Iterable

SPLIT_NAMES = ("development", "public_eval", "hidden_eval")
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.60, 0.20, 0.20)

DEFAULT_SPLITS_DIR = Path(__file__).resolve().parent / "splits"


class AccessRole(str, Enum):
    """Who is asking to read a split (labels)."""

    PIPELINE = "pipeline"
    IMPROVEMENT_LOOP = "improvement_loop"
    HARNESS_FINAL = "harness_final"
    MILESTONE_REPORT = "milestone_report"


# improvement loop: tune on development; may inspect public_eval for selection;
# must NEVER load hidden_eval labels.
_IMPROVEMENT_LOOP_ALLOWED = frozenset({"development", "public_eval"})


class HiddenLabelAccessError(PermissionError):
    """Raised when a non-harness role tries to load hidden_eval labels."""


def assert_split_allowed(split: str, role: AccessRole) -> None:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    if role == AccessRole.IMPROVEMENT_LOOP and split == "hidden_eval":
        raise HiddenLabelAccessError(
            "improvement-loop helpers cannot load hidden_eval labels "
            "(harness final scoring only)"
        )
    if role == AccessRole.PIPELINE and split == "hidden_eval":
        raise HiddenLabelAccessError(
            "pipeline / tuning paths cannot load hidden_eval labels"
        )


def split_manifest_path(split: str, splits_dir: str | Path | None = None) -> Path:
    root = Path(splits_dir) if splits_dir is not None else DEFAULT_SPLITS_DIR
    return root / f"{split}.jsonl"


def read_split_ids(split: str, splits_dir: str | Path | None = None) -> list[str]:
    path = split_manifest_path(split, splits_dir)
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ids.append(str(row["episode_id"] if isinstance(row, dict) else row))
    return ids


def load_split_ids_for_role(
    split: str,
    role: AccessRole,
    splits_dir: str | Path | None = None,
) -> list[str]:
    """Load episode ids for ``split`` after enforcing role isolation."""
    assert_split_allowed(split, role)
    return read_split_ids(split, splits_dir)


def improvement_loop_load_split(
    split: str,
    splits_dir: str | Path | None = None,
) -> list[str]:
    """Improvement-loop entry point: refuses hidden_eval labels."""
    return load_split_ids_for_role(
        split, AccessRole.IMPROVEMENT_LOOP, splits_dir=splits_dir
    )


def write_split_manifests(
    split_ids: dict[str, Iterable[str]],
    splits_dir: str | Path,
) -> dict[str, Path]:
    """Write ``{split: [episode_id, ...]}`` as JSONL manifests."""
    root = Path(splits_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, ids in split_ids.items():
        if name not in SPLIT_NAMES:
            raise ValueError(f"unknown split {name!r}")
        path = root / f"{name}.jsonl"
        lines = [json.dumps({"episode_id": eid}, ensure_ascii=False) for eid in ids]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written[name] = path
    return written


def assign_splits(
    episode_ids: list[str],
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, list[str]]:
    """Deterministic 60/20/20 (by default) shuffle-split of episode ids."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    # Local RNG — no global random.seed side effects.
    import random

    rng = random.Random(seed)
    ids = list(dict.fromkeys(episode_ids))  # stable unique, preserve first-seen order
    rng.shuffle(ids)
    n = len(ids)
    n_dev = int(n * ratios[0])
    n_pub = int(n * ratios[1])
    # remainder → hidden so sizes sum to n
    n_hid = n - n_dev - n_pub
    return {
        "development": ids[:n_dev],
        "public_eval": ids[n_dev : n_dev + n_pub],
        "hidden_eval": ids[n_dev + n_pub : n_dev + n_pub + n_hid],
    }
