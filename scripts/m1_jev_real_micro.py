#!/usr/bin/env python3
"""M1 micro: Baseline vs real JEV on ≤4 eval episodes (free-tier safe).

Requires ``JEV_MODE=real`` and ``JEV_AGENT_KEY`` (or ``JEV_API_KEY``).
Writes ``reports/m1_jev_real_micro.md``. Never prints the API key.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision.base import DecisionConfig, DecisionResult, load_decision_config  # noqa: E402
from decision.baseline import BaselineFastDecisionEngine  # noqa: E402
from decision.jev import (  # noqa: E402
    JevDecisionEngine,
    resolve_jev_api_key,
    resolve_jev_base_url,
    resolve_jev_mode,
)
from evaluation.harness import detection_prf1, failure_type_agreement  # noqa: E402
from features.extract import FeatureExtractor  # noqa: E402
from scripts.m0_5_thin_slice import (  # noqa: E402
    build_fixture_synthetic,
    sample_corpus,
    split_dev_eval,
    try_load_robofac,
)
from scripts.m1_failure_intelligence import load_m1_config  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "m1_jev_real_micro.md"


@dataclass
class MicroRow:
    episode_id: str
    gt_success: bool | None
    gt_failure_type: str | None
    baseline_success: bool
    baseline_p_fail: float
    baseline_type: str | None
    jev_success: bool
    jev_p_fail: float
    jev_type: str | None
    jev_tokens: int


@dataclass
class MicroResult:
    data_source: str
    seed: int
    n_eval: int
    max_real_calls: int
    base_url: str
    baseline_prf1: Any
    jev_prf1: Any
    baseline_type_agree: float
    jev_type_agree: float
    cost_calls: int
    cost_tokens: int
    quota_remaining: int | None
    quota_used: int | None
    quota_limit: int | None
    quota_month: str | None
    rows: list[MicroRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def run_micro(
    *,
    max_real_calls: int = 4,
    seed: int | None = None,
    force_synthetic: bool = False,
    n_corpus: int = 40,
) -> MicroResult:
    if max_real_calls < 1:
        raise ValueError("max_real_calls must be >= 1")
    mode = resolve_jev_mode()
    if mode != "real":
        raise RuntimeError(
            f"m1_jev_real_micro requires JEV_MODE=real (got {mode!r})."
        )
    if not resolve_jev_api_key():
        raise RuntimeError(
            "JEV_MODE=real requires JEV_AGENT_KEY or JEV_API_KEY in the environment."
        )

    base = load_m1_config()
    sd = int(seed if seed is not None else base.get("seed", 42))
    notes: list[str] = []

    data_source = "fixture_synthetic"
    if force_synthetic:
        corpus = build_fixture_synthetic(
            n_corpus, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
        )
        notes.append("Forced fixture/synthetic corpus.")
    else:
        thin_keys = {
            "robofac_sim_glob": base.get("robofac_sim_glob"),
            "robofac_real_glob": base.get("robofac_real_glob"),
            "use_realworld": base.get("use_realworld", False),
        }
        robofac = try_load_robofac(thin_keys)
        min_labeled = int(base.get("min_labeled", 100))
        if len(robofac) >= min_labeled:
            corpus = sample_corpus(robofac, n_corpus, sd)
            data_source = "robofac_real"
            notes.append(
                f"Sampled n={len(corpus)} from {len(robofac)} labeled RoboFAC episodes."
            )
        else:
            corpus = build_fixture_synthetic(
                n_corpus, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
            )
            notes.append(
                f"RoboFAC labeled={len(robofac)} < {min_labeled}; using fixture/synthetic."
            )

    development, eval_eps = split_dev_eval(corpus, sd)
    eval_slice = eval_eps[:max_real_calls]
    notes.append(
        f"Eval holdout slice n={len(eval_slice)} (max_real_calls={max_real_calls}); "
        f"development reference n={len(development)} for Baseline fit only."
    )
    notes.append(
        "JEV real calls are GT-free (path/instruction/meta state text only). "
        "Free-tier credits are tiny — this micro run is not a ship gate."
    )

    dcfg = load_decision_config(
        ROOT / base["decision_config"] if "decision_config" in base else None
    )
    dcfg = DecisionConfig(
        random_seed=dcfg.random_seed,
        review_top_k=max(1, min(dcfg.review_top_k, len(eval_slice) or 1)),
        fail_path_tokens=dcfg.fail_path_tokens,
        success_path_tokens=dcfg.success_path_tokens,
        type_keywords=dcfg.type_keywords,
        default_fail_probability=dcfg.default_fail_probability,
        jev_stub_cost_units=dcfg.jev_stub_cost_units,
        jev_stub_token_overhead=dcfg.jev_stub_token_overhead,
    )

    extractor = FeatureExtractor()
    feats = extractor.extract_many(eval_slice)

    baseline = BaselineFastDecisionEngine(dcfg).fit(development)
    base_results: list[DecisionResult] = [
        baseline.evaluate(f, dcfg) for f in feats
    ]

    jev = JevDecisionEngine(dcfg, mode="real")
    jev.reset_cost()
    jev_results: list[DecisionResult] = []
    tokens_before = 0
    per_call_tokens: list[int] = []
    for f in feats:
        r = jev.evaluate(f, dcfg)
        snap = jev.cost_snapshot()
        per_call_tokens.append(snap.tokens - tokens_before)
        tokens_before = snap.tokens
        jev_results.append(r)

    cost = jev.cost_snapshot()
    quota = jev.quota_snapshot()

    rows: list[MicroRow] = []
    for ep, br, jr, tok in zip(eval_slice, base_results, jev_results, per_call_tokens):
        rows.append(
            MicroRow(
                episode_id=ep.episode_id,
                gt_success=ep.ground_truth.success,
                gt_failure_type=ep.ground_truth.failure_type,
                baseline_success=br.predicted_success,
                baseline_p_fail=br.failure_probability,
                baseline_type=br.failure_type,
                jev_success=jr.predicted_success,
                jev_p_fail=jr.failure_probability,
                jev_type=jr.failure_type,
                jev_tokens=tok,
            )
        )

    return MicroResult(
        data_source=data_source,
        seed=sd,
        n_eval=len(eval_slice),
        max_real_calls=max_real_calls,
        base_url=resolve_jev_base_url(),
        baseline_prf1=detection_prf1(eval_slice, base_results),
        jev_prf1=detection_prf1(eval_slice, jev_results),
        baseline_type_agree=failure_type_agreement(eval_slice, base_results),
        jev_type_agree=failure_type_agreement(eval_slice, jev_results),
        cost_calls=cost.calls,
        cost_tokens=cost.tokens,
        quota_remaining=quota.remaining,
        quota_used=quota.used,
        quota_limit=quota.limit,
        quota_month=quota.month,
        rows=rows,
        notes=notes,
    )


def render_report(result: MicroResult) -> str:
    src = (
        "real RoboFAC (sampled)"
        if result.data_source == "robofac_real"
        else "fixtures + synthetic (path cues)"
    )
    b, j = result.baseline_prf1, result.jev_prf1
    quota_line = "not returned"
    if result.quota_remaining is not None:
        parts = [f"remaining={result.quota_remaining}"]
        if result.quota_used is not None:
            parts.append(f"used={result.quota_used}")
        if result.quota_limit is not None:
            parts.append(f"limit={result.quota_limit}")
        if result.quota_month:
            parts.append(f"month={result.quota_month}")
        quota_line = ", ".join(parts)

    lines: list[str] = [
        "# M1 micro — Baseline vs real JEV (n≤4)",
        "",
        "> ## HUGE DISCLAIMER",
        ">",
        "> **n is tiny (≤4).** Free-tier credits are scarce (~single-digit remaining). "
        "This is a **plumbing / smoke** check that the real adapter maps "
        "`noul`/`choice`/`usage` correctly — **not** a ship gate, **not** evidence "
        "that JEV beats Baseline, **not** a statistically meaningful F1 comparison.",
        ">",
        "> Do not cite these F1 numbers as product claims. Re-run on a paid quota "
        "with a proper eval split before any cost–quality decision.",
        "",
        "## Run metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Data source | {src} |",
        f"| Seed | {result.seed} |",
        f"| Eval n | {result.n_eval} (max_real_calls={result.max_real_calls}) |",
        f"| JEV mode | `real` |",
        f"| Endpoint | `{result.base_url}` |",
        f"| Cost calls | {result.cost_calls} |",
        f"| Cost tokens (input) | {result.cost_tokens} |",
        f"| Stub cost units | 0 (real) |",
        f"| Quota (API) | {quota_line} |",
        "",
        "### Notes",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")

    lines += [
        "",
        "## Metrics (failure detection)",
        "",
        "| Backend | Det P | Det R | Det F1 | Type agree | Calls | Tokens |",
        "|---------|-------|-------|--------|------------|-------|--------|",
        f"| baseline | {b.precision:.3f} | {b.recall:.3f} | {b.f1:.3f} | "
        f"{result.baseline_type_agree:.3f} | 0 | 0 |",
        f"| jev (real) | {j.precision:.3f} | {j.recall:.3f} | {j.f1:.3f} | "
        f"{result.jev_type_agree:.3f} | {result.cost_calls} | {result.cost_tokens} |",
        "",
        "## Per-episode rows",
        "",
        "| episode_id | GT success | GT type | Baseline pred | Base p_fail | Base type | "
        "JEV pred | JEV p_fail (noul) | JEV type | JEV tokens |",
        "|------------|------------|---------|---------------|-------------|-----------|"
        "----------|-------------------|----------|------------|",
    ]
    for r in result.rows:
        lines.append(
            f"| `{r.episode_id}` | {r.gt_success} | {r.gt_failure_type or '—'} | "
            f"{'ok' if r.baseline_success else 'fail'} | {r.baseline_p_fail:.3f} | "
            f"{r.baseline_type or '—'} | "
            f"{'ok' if r.jev_success else 'fail'} | {r.jev_p_fail:.3f} | "
            f"{r.jev_type or '—'} | {r.jev_tokens} |"
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "# Key must already be in the environment — never echo it.",
        "export JEV_MODE=real",
        "python scripts/m1_jev_real_micro.py --max-real-calls 4",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-real-calls", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-corpus", type=int, default=40)
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    # Hard cap: never exceed 4 unless caller explicitly set higher — still warn.
    max_calls = args.max_real_calls
    if max_calls > 4:
        print(
            f"warning: max_real_calls={max_calls} > 4 — free tier may exhaust",
            file=sys.stderr,
        )

    result = run_micro(
        max_real_calls=max_calls,
        seed=args.seed,
        force_synthetic=args.force_synthetic,
        n_corpus=args.n_corpus,
    )
    text = render_report(result)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(
        f"n={result.n_eval} baseline_F1={result.baseline_prf1.f1:.3f} "
        f"jev_F1={result.jev_prf1.f1:.3f} calls={result.cost_calls} "
        f"tokens={result.cost_tokens} quota_remaining={result.quota_remaining}"
    )


if __name__ == "__main__":
    main()
