#!/usr/bin/env python3
"""Write reports/m0_proxy_importance.md — proxy review-priority A–D.

Offline demo corpus = RoboFAC fixtures + synthetic fails.
Frequency + Failure Bank seeded from the *reference* half only (dev-style);
scores the full corpus and a public_eval-style holdout. Never loads hidden_eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.robofac import load_episodes  # noqa: E402
from data.schemas.episode import Episode, GroundTruth  # noqa: E402
from proxy_importance.config import config_to_dict, load_config  # noqa: E402
from proxy_importance.failure_bank import FailureBank  # noqa: E402
from proxy_importance.frequency import build_frequency_table  # noqa: E402
from proxy_importance.rules import ProxyReviewPriorityResult, score_episodes  # noqa: E402

SAMPLES = ROOT / "data" / "samples" / "robofac"
DEFAULT_OUT = ROOT / "reports" / "m0_proxy_importance.md"

# Deterministic pad so B (rare) / C (high-freq) have enough mass offline.
_SYNTH_FAIL_TYPES = (
    "position_deviation",  # common
    "position_deviation",
    "position_deviation",
    "position_deviation",
    "step_omission",  # mid / borderline
    "step_omission",
    "grasping_error",  # rare-ish
    "timing_error",  # rare / potentially novel if held out of bank
)


def build_demo_corpus() -> list[Episode]:
    fixtures = load_episodes(sorted(SAMPLES.glob("*.json")))
    synth: list[Episode] = []
    for i, ft in enumerate(_SYNTH_FAIL_TYPES):
        synth.append(
            Episode(
                episode_id=f"synthetic-proxy-{i:04d}",
                metadata={"source": "synthetic_proxy"},
                ground_truth=GroundTruth(success=False, failure_type=ft),
            )
        )
    for i in range(2):
        synth.append(
            Episode(
                episode_id=f"synthetic-proxy-ok-{i:04d}",
                metadata={"source": "synthetic_proxy"},
                ground_truth=GroundTruth(success=True),
            )
        )
    return fixtures + synth


def _split_demo(corpus: list[Episode]) -> tuple[list[Episode], list[Episode]]:
    """Deterministic 70/30-ish: odd index → public_eval-style holdout."""
    development: list[Episode] = []
    public_eval: list[Episode] = []
    for i, ep in enumerate(corpus):
        (public_eval if i % 3 == 0 else development).append(ep)
    return development, public_eval


def _hit_counts(results: list[ProxyReviewPriorityResult]) -> dict[str, int]:
    return {
        "A": sum(1 for r in results if "A" in r.rule_hits),
        "B": sum(1 for r in results if "B" in r.rule_hits),
        "C": sum(1 for r in results if "C" in r.rule_hits),
        "D": sum(1 for r in results if "D" in r.rule_hits),
    }


def render(
    by_split: dict[str, list[ProxyReviewPriorityResult]],
    freq_counts: dict[str, int],
    rare_cutoff: int,
    cfg,
) -> str:
    lines: list[str] = [
        "# M0 proxy review-priority report",
        "",
        "> **Disclaimer:** proxy importance is rule-based; "
        "**review priority ≠ business importance**; not business-critical GT.",
        "",
        "## Formula (frozen)",
        "",
        "```text",
        "proxy_important ⇔ A ∧ (B ∨ C ∨ D)",
        "A = fail (success=false)",
        "B = rare failure_type (count ≤ rare_freq_quantile of per-type counts)",
        "C = high-frequency recurrence (count ≥ high_freq_min_count)",
        "D = novelty vs Failure Bank (unseen type → novelty_score=1.0)",
        "E/F = ranking bonuses only (stubs; not admission)",
        "```",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config_to_dict(cfg), indent=2),
        "```",
        "",
        f"Reference frequency (development failures): counts={freq_counts}, "
        f"rare_count_cutoff={rare_cutoff}",
        "",
        "## Split summaries",
        "",
        "| Split | n | proxy_important | prevalence | A | B | C | D |",
        "|-------|---|-----------------|------------|---|---|---|---|",
    ]
    for split, results in by_split.items():
        n = len(results)
        n_imp = sum(1 for r in results if r.proxy_important)
        prev = (n_imp / n) if n else 0.0
        h = _hit_counts(results)
        lines.append(
            f"| {split} | {n} | {n_imp} | {prev:.2%} | {h['A']} | {h['B']} | {h['C']} | {h['D']} |"
        )

    lines += ["", "## Top-K proxy-important", ""]
    for split, results in by_split.items():
        imp = sorted(
            (r for r in results if r.proxy_important),
            key=lambda r: (-len(r.rule_hits), -r.novelty_score, r.episode_id),
        )[: cfg.top_k]
        lines.append(f"### {split}")
        lines.append("")
        if not imp:
            lines.append("_none_")
        else:
            for r in imp:
                lines.append(
                    f"- `{r.episode_id}` hits={list(r.rule_hits)} "
                    f"bucket={r.frequency_bucket.value if r.frequency_bucket else None} novelty={r.novelty_score:.1f}"
                )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Frequency + Failure Bank seeded from **development**-style reference only.",
        "- `hidden_eval` is never loaded (harness final scoring only).",
        "- Cold-start: types absent from the bank are novel (D).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    corpus = build_demo_corpus()
    development, public_eval = _split_demo(corpus)

    freq = build_frequency_table(development, cfg)
    bank = FailureBank.from_episodes(development)

    by_split = {
        "development": score_episodes(development, freq=freq, bank=bank, config=cfg),
        "public_eval": score_episodes(public_eval, freq=freq, bank=bank, config=cfg),
    }

    text = render(by_split, freq.counts, freq.rare_count_cutoff, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    for split, results in by_split.items():
        n_imp = sum(1 for r in results if r.proxy_important)
        print(f"{split}: n={len(results)} proxy_important={n_imp}")


if __name__ == "__main__":
    main()
