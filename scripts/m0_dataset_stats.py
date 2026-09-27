#!/usr/bin/env python3
"""Summarize Episode fixtures + GT label coverage for M0 smoke."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.robofac import RoboFACAdapter  # noqa: E402
from data.gt_mapping import apply_gt_mapping  # noqa: E402
from data.split_access import DEFAULT_SPLITS_DIR, SPLIT_NAMES, read_split_ids  # noqa: E402

SAMPLES = ROOT / "data" / "samples" / "robofac"


def label_coverage(episodes) -> dict[str, float]:
    n = len(episodes) or 1
    success_mapped = sum(1 for e in episodes if e.ground_truth.success is not None)
    type_mapped = sum(1 for e in episodes if e.ground_truth.failure_type is not None)
    fails = [e for e in episodes if e.ground_truth.success is False]
    type_on_fails = sum(1 for e in fails if e.ground_truth.failure_type is not None)
    return {
        "n_episodes": len(episodes),
        "success_mapped_frac": success_mapped / n,
        "failure_type_mapped_frac": type_mapped / n,
        "failure_type_on_fails_frac": (type_on_fails / len(fails)) if fails else 1.0,
        "n_fails": len(fails),
    }


def main() -> None:
    episodes = []
    for path in sorted(SAMPLES.glob("*.json")):
        episodes.extend(RoboFACAdapter(path).load())
    # Remap is idempotent; coverage reflects canonical GT.
    episodes = [apply_gt_mapping(e) for e in episodes]

    success = Counter(ep.ground_truth.success for ep in episodes)
    types = Counter(
        ep.ground_truth.failure_type for ep in episodes if ep.ground_truth.failure_type
    )
    cov = label_coverage(episodes)
    print(f"n_episodes={cov['n_episodes']}")
    print(f"success_counts={dict(success)}")
    print(f"failure_types={dict(types)}")
    print(
        "label_coverage="
        f"success={cov['success_mapped_frac']:.2f} "
        f"type={cov['failure_type_mapped_frac']:.2f} "
        f"type_on_fails={cov['failure_type_on_fails_frac']:.2f}"
    )

    if DEFAULT_SPLITS_DIR.exists():
        sizes = {s: len(read_split_ids(s)) for s in SPLIT_NAMES}
        print(f"split_sizes={sizes}")


if __name__ == "__main__":
    main()
