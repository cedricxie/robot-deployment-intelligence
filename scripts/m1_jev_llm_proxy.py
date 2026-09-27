#!/usr/bin/env python3
"""M1 JEV llm_proxy: Baseline vs role-play systemone judge on holdout-45.

Deterministic GT-free assistant stand-in for Jev systemone answers.
**Not** the live Jev API — no credits, no real probs. Same thin-slice path
as M1 (n=150 seed=42 → holdout 45, Top-K=15). Never loads hidden_eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decision.base import DecisionConfig, load_decision_config  # noqa: E402
from decision.baseline import BaselineFastDecisionEngine  # noqa: E402
from decision.frequency import FrequencyOnlyDecisionEngine  # noqa: E402
from decision.jev import JevDecisionEngine  # noqa: E402
from evaluation.harness import EvaluationHarness, HarnessReport  # noqa: E402
from proxy_importance.config import (  # noqa: E402
    ProxyImportanceConfig,
    load_config as load_proxy_config,
)
from scripts.m0_5_thin_slice import (  # noqa: E402
    build_fixture_synthetic,
    sample_corpus,
    split_dev_eval,
    try_load_robofac,
)
from scripts.m1_failure_intelligence import load_m1_config  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "m1_failure_intelligence.json"
DEFAULT_OUT = ROOT / "reports" / "m1_jev_llm_proxy.md"

# Stub numbers from reports/m1_failure_intelligence.md (RoboFAC holdout-45) for context.
_STUB_CONTEXT = {
    "detection_f1": 0.944,
    "failure_type_agreement": 0.857,
    "proxy_retention": 0.395,
    "cost_calls": 45,
    "cost_tokens": 842,
    "cost_stub_units": 450.0,
}


@dataclass
class LlmProxyResult:
    data_source: str
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    top_k: int
    eval_split_name: str
    harness: HarnessReport
    n_gt_fails_eval: int = 0
    notes: list[str] = field(default_factory=list)


def run_llm_proxy(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
    top_k: int | None = None,
) -> LlmProxyResult:
    base = load_m1_config() if cfg is None else dict(cfg)
    n = int(n_episodes if n_episodes is not None else base.get("n_episodes", 150))
    sd = int(seed if seed is not None else base.get("seed", 42))
    k = int(top_k if top_k is not None else base.get("top_k", 15))
    min_labeled = int(base.get("min_labeled", 100))
    notes: list[str] = []

    data_source = "fixture_synthetic"
    if force_synthetic:
        corpus = build_fixture_synthetic(
            n, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
        )
        notes.append("Forced fixture/synthetic corpus (path cues + typed fails).")
    else:
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
        "JEV_MODE=llm_proxy — deterministic role-play of systemone answers. "
        "**Not** live Jev API; no credits; not a substitute for real Jev probs."
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
    # Baseline + Frequency (+ Random via default) then llm_proxy JEV.
    engines = harness.default_engines(development)
    engines.append(JevDecisionEngine(dcfg, mode="llm_proxy"))

    report = harness.evaluate(
        eval_eps, reference_episodes=development, engines=engines, top_k=k
    )
    n_gt_fails = sum(1 for e in eval_eps if e.ground_truth.success is False)

    return LlmProxyResult(
        data_source=data_source,
        seed=sd,
        n_episodes=len(corpus),
        n_development=len(development),
        n_eval=len(eval_eps),
        top_k=k,
        eval_split_name="public_eval",
        harness=report,
        n_gt_fails_eval=n_gt_fails,
        notes=notes,
    )


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_report(result: LlmProxyResult) -> str:
    src_label = (
        "real RoboFAC (test_qa_sim labeled subset)"
        if result.data_source == "robofac_real"
        else "fixtures + deterministic synthetic (path cues)"
    )
    lines: list[str] = [
        "# M1 JEV llm_proxy — Baseline vs role-play systemone",
        "",
        "> **Disclaimer:** **review priority ≠ business importance.** "
        "`proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business "
        "criticality. **LLM-proxy ≠ real JEV** — this is a deterministic assistant "
        "role-play of systemone `{answers: is_fail.noul, fail_type.choice}` on "
        "GT-free `features_to_state` text. **Not** live Jev API, **not** real "
        "probs/billing, **not** a substitute for credentialed systemone. "
        "Baseline may still win when path cues leak.",
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
        "| JEV mode | `llm_proxy` |",
        "| API keys / JEV credits | none (local role-play) |",
        "",
        "### Notes",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")

    lines += [
        "",
        "## Side-by-side (eval holdout)",
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

    base = result.harness.column("baseline")
    jev = result.harness.column("jev")
    freq = result.harness.column("frequency")

    lines += [
        "",
        "## Stub context (prior M1 report)",
        "",
        "From `reports/m1_failure_intelligence.md` on the same RoboFAC holdout-45 "
        f"(seed=42): stub Det F1={_STUB_CONTEXT['detection_f1']:.3f}, "
        f"type agree={_STUB_CONTEXT['failure_type_agreement']:.3f}, "
        f"proxy retention={_STUB_CONTEXT['proxy_retention']:.3f}, "
        f"cost=({_STUB_CONTEXT['cost_calls']}, {_STUB_CONTEXT['cost_tokens']}, "
        f"{_STUB_CONTEXT['cost_stub_units']:.1f} stub units).",
        "",
        "## Cost–quality framing (llm_proxy vs Baseline)",
        "",
    ]
    if base and jev:
        df1 = jev.detection_f1 - base.detection_f1
        dret = jev.proxy_retention - base.proxy_retention
        dagree = jev.failure_type_agreement - base.failure_type_agreement
        lines.append(
            f"- Quality delta (llm_proxy − Baseline): detection F1 **{df1:+.3f}**, "
            f"type agree **{dagree:+.3f}**, proxy retention **{dret:+.3f}**."
        )
        lines.append(
            f"- Cost (llm_proxy): **{jev.cost_calls}** calls, **{jev.cost_tokens}** "
            "approx tokens (≈ state length/4), **0** stub units / **0** JEV credits; "
            "Baseline local heuristic → 0 calls."
        )
        if abs(df1) < 1e-9 and abs(dret) < 1e-9:
            lines.append(
                "- Quality ≈ Baseline with extra call/token accounting — does **not** "
                "justify spend; real Jev must beat Baseline on quality or unlock cascade "
                "savings."
            )
        elif df1 > 0 or dret > 0:
            lines.append(
                "- Proxy shows a quality edge on this slice vs Baseline; still a "
                "heuristic role-play — re-run with live systemone before claiming JEV "
                "superiority."
            )
        else:
            lines.append(
                "- Proxy does not beat Baseline here (path cues often leak to "
                "Baseline). Cost–quality hypothesis **not** supported by this "
                "llm_proxy run alone."
            )

    n_imp = base.n_proxy_important if base else (jev.n_proxy_important if jev else 0)
    lines += [
        "",
        "## Proxy retention + review reduction",
        "",
        f"Proxy-important count on eval: **{n_imp}** (shared rule labels A∧(B∨C∨D)).",
        "",
        "| Backend | Proxy retention | Review reduction |",
        "|---------|-----------------|------------------|",
    ]
    for c in result.harness.columns:
        if c.backend in ("baseline", "jev", "frequency"):
            lines.append(
                f"| {c.backend} | {_fmt_pct(c.proxy_retention)} | "
                f"{_fmt_pct(c.review_reduction)} |"
            )

    lines += [
        "",
        "## What this tells us",
        "",
    ]
    if base and jev:
        lines.append(
            f"Holdout n={result.n_eval} source={result.data_source}: "
            f"Baseline F1={base.detection_f1:.3f}, "
            f"llm_proxy F1={jev.detection_f1:.3f}"
            + (f", Frequency F1={freq.detection_f1:.3f}" if freq else "")
            + f"; type agree Baseline={base.failure_type_agreement:.3f} / "
            f"llm_proxy={jev.failure_type_agreement:.3f}."
        )
        lines.append(
            "llm_proxy parses semantic path folders (`position_offset`, "
            "`gripper_error`, …) with soft noul and avoids SafeTask→`safe` "
            "false friends that hurt the stub. When success/fail path tokens "
            "already leak, Baseline remains hard to beat on detection."
        )
        lines.append(
            "**LLM-proxy ≠ real JEV.** Treat this as a reproducible ceiling "
            "sketch for text-only cues, not billing-backed systemone quality."
        )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "JEV_MODE=llm_proxy python scripts/m1_jev_llm_proxy.py",
        "# optional: --force-synthetic --out reports/m1_jev_llm_proxy.md",
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
    parser.add_argument("--force-synthetic", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_m1_config(args.config)
    result = run_llm_proxy(
        cfg,
        force_synthetic=args.force_synthetic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        top_k=args.top_k,
    )
    text = render_report(result)
    out = args.out or DEFAULT_OUT
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(
        f"source={result.data_source} n={result.n_episodes} "
        f"dev={result.n_development} eval={result.n_eval} top_k={result.top_k}"
    )
    for c in result.harness.columns:
        print(
            f"  {c.backend}: F1={c.detection_f1:.3f} type={c.failure_type_agreement:.3f} "
            f"proxy_ret={c.proxy_retention:.3f} rev_red={c.review_reduction:.3f} "
            f"cost=({c.cost_calls},{c.cost_tokens},{c.cost_stub_units:.1f})"
        )


if __name__ == "__main__":
    main()
