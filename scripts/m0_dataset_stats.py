#!/usr/bin/env python3
"""Stub: summarize Episode fixtures for M0 smoke."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.robofac import RoboFACAdapter  # noqa: E402

SAMPLES = ROOT / "data" / "samples" / "robofac"


def main() -> None:
    episodes = []
    for path in sorted(SAMPLES.glob("*.json")):
        episodes.extend(RoboFACAdapter(path).load())
    success = Counter(ep.ground_truth.success for ep in episodes)
    types = Counter(ep.ground_truth.failure_type for ep in episodes if ep.ground_truth.failure_type)
    print(f"n_episodes={len(episodes)}")
    print(f"success_counts={dict(success)}")
    print(f"failure_types={dict(types)}")


if __name__ == "__main__":
    main()
