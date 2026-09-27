#!/usr/bin/env python3
"""Write deterministic 60/20/20 split manifests (seed=42) under data/splits/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.robofac import load_episodes  # noqa: E402
from data.split_access import (  # noqa: E402
    DEFAULT_SEED,
    assign_splits,
    write_split_manifests,
)

SAMPLES = ROOT / "data" / "samples" / "robofac"
DEFAULT_OUT = ROOT / "data" / "splits"


def collect_episode_ids(extra_n: int = 0) -> list[str]:
    """Fixture episode ids, optionally padded with synthetic ids for sizing demos."""
    paths = sorted(SAMPLES.glob("*.json"))
    ids = [ep.episode_id for ep in load_episodes(paths)]
    for i in range(extra_n):
        ids.append(f"synthetic-{i:04d}")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for development/public_eval/hidden_eval.jsonl",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--pad",
        type=int,
        default=17,
        help="Synthetic episode ids to append (fixtures alone are too few for 60/20/20)",
    )
    args = parser.parse_args(argv)

    ids = collect_episode_ids(extra_n=args.pad)
    splits = assign_splits(ids, seed=args.seed)
    written = write_split_manifests(splits, args.out)
    for name, path in written.items():
        print(f"{name}: n={len(splits[name])} -> {path}")


if __name__ == "__main__":
    main()
