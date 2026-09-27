#!/usr/bin/env python3
"""M1 failure intelligence: full ladder + cost proxy → markdown report.

Reuses the thin-slice data path (RoboFAC labeled when present; else synthetic).
JEV is framed as a **cost–quality** experiment, not an assumed winner.
Never loads hidden_eval. No API keys required for stub mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision.base import DecisionConfig, load_decision_config  # noqa: E402
from decision.jev import JevDecisionEngine, resolve_jev_mode  # noqa: E402
from evaluation.harness import EvaluationHarness, HarnessReport  # noqa: E402
from proxy_importance.config import ProxyImportanceConfig, load_config as load_proxy_config  # noqa: E402
from scripts.m0_5_thin_slice import (  # noqa: E402
    build_fixture_synthetic,
    sample_corpus,
    split_dev_eval,
    try_load_robofac,
)

DEFAULT_CONFIG = ROOT / "configs" / "m1_failure_intelligence.json"


@dataclass
class M1Result:
    data_source: str
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    top_k: int
    jev_mode: str
    eval_split_name: str
    harness: HarnessReport
    n_gt_fails_eval: int = 0
    notes: list[str] = field(default_factory=list)


def load_m1_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def run_m1(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
    top_k: int | None = None,
    jev_mode: str | None = None,
) -> M1Result:
    base = load_m1_config() if cfg is None else dict(cfg)
    n = int(n_episodes if n_episodes is not None else base.get("n_episodes", 150))
    sd = int(seed if seed is not None else base.get("seed", 42))
    k = int(top_k if top_k is not None else base.get("top_k", 15))
    min_labeled = int(base.get("min_labeled", 100))
    mode = resolve_jev_mode(
        jev_mode if jev_mode is not None else base.get("jev_mode", os.environ.get("JEV_MODE", "stub"))
    )
    notes: list[str] = []

    data_source = "fixture_synthetic"
    if force_synthetic:
        corpus = build_fixture_synthetic(
            n, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
        )
        notes.append("Forced fixture/synthetic corpus (path cues + typed fails).")
    else:
        # Reuse thin-slice RoboFAC loader (same globs / labeled filter).
        thin_keys = {
            "robofac_sim_glob": base.get("robofac_sim_glob"),
            "robofac_real_glob": base.get("robofac_real_glob"),
            "use_realworld": base.get("use_realworld", False),
        }
        robofac = try_load_robofac(thin_keys)
        if len(robofac) >= min_labeled:
            corpus = sample_corpus(robofac, n, sd)
            data_source = "robofac_real"
            notes.append(
                f"Loaded {len(robofac)} labeled RoboFAC test_qa episodes; "
                f"sampled n={len(corpus)} (seed={sd})."
            )
        else:
            corpus = build_fixture_synthetic(
                n, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
            )
            notes.append(
                f"RoboFAC labeled count={len(robofac)} < min_labeled={min_labeled}; "
                "using fixture/synthetic expansion."
            )

    development, eval_eps = split_dev_eval(corpus, sd)
    notes.append(
        "Reference = development (Frequency/Baseline/JEV fit + proxy bank). "
        "Eval = public_eval-style holdout. hidden_eval never loaded."
    )
    notes.append(
        f"JEV_MODE={mode}. JEV is a cost–quality experiment — not assumed better than Baseline."
    )

    dcfg = load_decision_config(
        ROOT / base["decision_config"] if "decision_config" in base else None
    )
    dcfg = DecisionConfig(
        random_seed=dcfg.random_seed,
        review_top_k=k,
        fail_path_tokens=dcfg.fail_path_tokens,
        success_path_tokens=dcfg.success_path_tokens,
        type_keywords=dcfg.type_keywords,
        default_fail_probability=dcfg.default_fail_probability,
        jev_stub_cost_units=dcfg.jev_stub_cost_units,
        jev_stub_token_overhead=dcfg.jev_stub_token_overhead,
    )
    pcfg = load_proxy_config(ROOT / base["proxy_config"] if "proxy_config" in base else None)
    pcfg = ProxyImportanceConfig(
        rare_freq_quantile=pcfg.rare_freq_quantile,
        high_freq_min_count=pcfg.high_freq_min_count,
        novelty_min_score=pcfg.novelty_min_score,
        top_k=k,
    )

    harness = EvaluationHarness(dcfg, pcfg, include_jev=False)
    engines = harness.default_engines(development)
    report_mode = mode
    if mode == "real":
        notes.append(
            "JEV_MODE=real requested but real adapter is not implemented; "
            "report uses stub (acceptance: stub path always runnable)."
        )
        report_mode = "stub"
    if report_mode != "off":
        engines.append(JevDecisionEngine(dcfg, mode="stub"))

    report = harness.evaluate(
        eval_eps, reference_episodes=development, engines=engines, top_k=k
    )
    n_gt_fails = sum(1 for e in eval_eps if e.ground_truth.success is False)

    return M1Result(
        data_source=data_source,
        seed=sd,
        n_episodes=len(corpus),
        n_development=len(development),
        n_eval=len(eval_eps),
        top_k=k,
        jev_mode=report_mode if mode != "real" else "stub (real unavailable)",
        eval_split_name="public_eval",
        harness=report,
        n_gt_fails_eval=n_gt_fails,
        notes=notes,
    )


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_report(result: M1Result) -> str:
    src_label = (
        "real RoboFAC (test_qa_sim labeled subset)"
        if result.data_source == "robofac_real"
        else "fixtures + deterministic synthetic (path cues)"
    )
    lines: list[str] = [
        "# M1 failure intelligence — ladder + JEV cost–quality",
        "",
        "> **Disclaimer:** **review priority ≠ business importance.** "
        "`proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business "
        "criticality. **JEV is a cost–quality experiment**, not an assumed winner over "
        "BaselineFast. Stub is GT-free (path + instruction cues) with stub cost units.",
        "",
        "## Run metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Data source | {src_label} |",
        f"| Dataset size (corpus) | {result.n_episodes} |",
        f"| Seed | {result.seed} |",
        f"| Development (reference) | {result.n_development} |",
        f"| Eval split | `{result.eval_split_name}` (holdout) n={result.n_eval} |",
        f"| Top-K | {result.top_k} |",
        f"| JEV mode | `{result.jev_mode}` |",
        f"| API keys | none required (stub) |",
        "",
        "### Notes",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")

    lines += [
        "",
        "## Side-by-side ladder (eval holdout)",
        "",
        "| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention | Review reduction | Cost calls | Cost tokens | Stub cost units | n_proxy_imp |",
        "|---------|-------|-------|--------|------------|-----------------|------------------|------------|-------------|-----------------|-------------|",
    ]
    for c in result.harness.columns:
        lines.append(
            f"| {c.backend} | {c.detection_precision:.3f} | {c.detection_recall:.3f} | "
            f"{c.detection_f1:.3f} | {c.failure_type_agreement:.3f} | "
            f"{c.proxy_retention:.3f} | {c.review_reduction:.3f} | "
            f"{c.cost_calls} | {c.cost_tokens} | {c.cost_stub_units:.1f} | "
            f"{c.n_proxy_important} |"
        )

    rnd = result.harness.column("random")
    freq = result.harness.column("frequency")
    base = result.harness.column("baseline")
    jev = result.harness.column("jev")

    lines += [
        "",
        "## Detection gate",
        "",
    ]
    if base:
        if base.detection_f1 >= 0.70:
            lines.append(
                f"Baseline detection F1={base.detection_f1:.3f} **≥ 0.70** (M1 gate met on this slice)."
            )
        else:
            lines.append(
                f"Baseline detection F1={base.detection_f1:.3f} **< 0.70** — miss modes: "
                "renamed paths without `fail`/`success` tokens; cue-less episodes lean on "
                "development fail prior; type keywords absent from path/instruction."
            )
    if jev:
        if jev.detection_f1 >= 0.70:
            lines.append(
                f"JEV stub detection F1={jev.detection_f1:.3f} ≥ 0.70 on this slice "
                f"(cost={jev.cost_stub_units:.1f} stub units / {jev.cost_calls} calls / "
                f"{jev.cost_tokens} tokens)."
            )
        else:
            lines.append(
                f"JEV stub detection F1={jev.detection_f1:.3f} < 0.70 — same path miss modes "
                "plus instruction-keyword false positives/negatives on cue-less text."
            )

    n_imp = base.n_proxy_important if base else (jev.n_proxy_important if jev else 0)
    lines += [
        "",
        "## Proxy retention + review reduction (all rungs)",
        "",
        f"Proxy-important count on eval: **{n_imp}** (shared rule labels A∧(B∨C∨D)).",
        "",
        "| Backend | Proxy retention | Review reduction | vs all GT fails |",
        "|---------|-----------------|------------------|-----------------|",
    ]
    for c in result.harness.columns:
        vs_fails = (
            max(0.0, 1.0 - (result.top_k / result.n_gt_fails_eval))
            if result.n_gt_fails_eval
            else 0.0
        )
        lines.append(
            f"| {c.backend} | {_fmt_pct(c.proxy_retention)} | {_fmt_pct(c.review_reduction)} | "
            f"{_fmt_pct(vs_fails)} |"
        )

    lines += [
        "",
        "## Cost–quality framing (JEV vs Baseline)",
        "",
    ]
    if base and jev:
        df1 = jev.detection_f1 - base.detection_f1
        dret = jev.proxy_retention - base.proxy_retention
        lines.append(
            f"- Quality delta (JEV − Baseline): detection F1 **{df1:+.3f}**, "
            f"proxy retention **{dret:+.3f}**."
        )
        lines.append(
            f"- Cost (JEV stub): **{jev.cost_calls}** calls, **{jev.cost_tokens}** tokens, "
            f"**{jev.cost_stub_units:.1f}** stub cost units; Baseline/Random/Frequency "
            "local heuristics → **0** stub units."
        )
        if abs(df1) < 1e-9 and abs(dret) < 1e-9:
            lines.append(
                "- On this slice JEV stub quality ≈ Baseline while charging stub cost — "
                "**does not justify** stub spend; real adapter must beat Baseline on "
                "quality or unlock cascade savings to pass a cost–quality bar."
            )
        elif df1 > 0 or dret > 0:
            lines.append(
                "- Stub shows a quality edge on this slice; still a heuristic stand-in — "
                "re-run with a real adapter before claiming JEV superiority."
            )
        else:
            lines.append(
                "- Stub does not beat Baseline here; cost–quality hypothesis **not** supported "
                "by this stub run (honest negative / tie leaning Baseline)."
            )
    elif not jev:
        lines.append("- JEV column omitted (`JEV_MODE=off`).")

    lines += [
        "",
        "## What this tells us",
        "",
    ]
    if base and rnd and freq:
        lines.append(
            f"Ladder on source={result.data_source}: "
            f"Random F1={rnd.detection_f1:.3f}, Frequency F1={freq.detection_f1:.3f}, "
            f"Baseline F1={base.detection_f1:.3f}"
            + (f", JEV stub F1={jev.detection_f1:.3f}" if jev else "")
            + "."
        )
        if result.data_source == "robofac_real":
            lines.append(
                "RoboFAC success paths often include `dataset_success_cleaned` / `stack_ok` "
                "(strong success cue). Many fails lack `fail` path tokens, so Baseline/JEV "
                "lean on the development fail prior for cue-less fails. Frequency (majority "
                "fail) also scores high when the slice is fail-heavy. Type agreement is "
                "limited when QA `failure_type` is null or keywords miss."
            )
        else:
            lines.append(
                "Fixture/synthetic embeds explicit `fail_*` / `success` path tokens so "
                "Baseline/JEV beat Random on detection. This proves ladder plumbing + cost "
                "columns, not SOTA perception."
            )
        lines.append(
            "Proxy retention at Top-K is a **review-priority** sketch only. "
            "Shipping may stay Baseline-only; the ladder comparison remains in-scope."
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "JEV_MODE=stub python scripts/m1_failure_intelligence.py "
        "--config configs/m1_failure_intelligence.json",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--jev-mode", type=str, default=None, choices=["stub", "off", "real"])
    parser.add_argument("--force-synthetic", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_m1_config(args.config)
    result = run_m1(
        cfg,
        force_synthetic=args.force_synthetic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        top_k=args.top_k,
        jev_mode=args.jev_mode,
    )
    text = render_report(result)
    out = args.out or ROOT / cfg.get("report_out", "reports/m1_failure_intelligence.md")
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(
        f"source={result.data_source} n={result.n_episodes} "
        f"dev={result.n_development} eval={result.n_eval} "
        f"top_k={result.top_k} jev_mode={result.jev_mode}"
    )
    for c in result.harness.columns:
        print(
            f"  {c.backend}: F1={c.detection_f1:.3f} "
            f"proxy_ret={c.proxy_retention:.3f} rev_red={c.review_reduction:.3f} "
            f"cost=({c.cost_calls},{c.cost_tokens},{c.cost_stub_units:.1f})"
        )


if __name__ == "__main__":
    main()
