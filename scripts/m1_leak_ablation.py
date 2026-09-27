#!/usr/bin/env python3
"""Path-label leakage ablation: leaked vs sanitized path_text on the M1 slice.

High Baseline / llm_proxy F1 on RoboFAC is driven by success/fail *directory*
cues (``dataset_success_cleaned``, ``stack_ok``, ``fail``, …). This script
runs the decision ladder twice on the same n=150 seed=42 holdout-45 Top-K=15
slice — once with cues intact, once with them stripped — and writes
``reports/m1_leak_ablation.md``.

Never loads hidden_eval. Never calls live Jev API / spends credits.
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

from decision.base import (  # noqa: E402
    DecisionConfig,
    load_decision_config,
    with_path_leak_ablation,
)
from decision.jev import JevDecisionEngine  # noqa: E402
from evaluation.harness import EvaluationHarness, HarnessReport, LadderColumn  # noqa: E402
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
DEFAULT_OUT = ROOT / "reports" / "m1_leak_ablation.md"


@dataclass
class AblationRun:
    label: str  # "leaked" | "ablated"
    data_source: str
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    top_k: int
    harness: HarnessReport
    n_gt_fails_eval: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class AblationResult:
    leaked: AblationRun
    ablated: AblationRun
    notes: list[str] = field(default_factory=list)


def _build_corpus(
    base: dict[str, Any],
    *,
    force_synthetic: bool,
    n: int,
    sd: int,
) -> tuple[list, str, list[str]]:
    notes: list[str] = []
    min_labeled = int(base.get("min_labeled", 100))
    data_source = "fixture_synthetic"
    if force_synthetic:
        corpus = build_fixture_synthetic(
            n, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
        )
        notes.append("Forced fixture/synthetic corpus (path cues + typed fails).")
        return corpus, data_source, notes

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
    return corpus, data_source, notes


def _make_dcfg(base: dict[str, Any], k: int, *, ablate: bool) -> DecisionConfig:
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
        ablate_path_leak=False,
    )
    return with_path_leak_ablation(dcfg) if ablate else dcfg


def _run_ladder(
    development,
    eval_eps,
    dcfg: DecisionConfig,
    pcfg: ProxyImportanceConfig,
    k: int,
) -> HarnessReport:
    harness = EvaluationHarness(dcfg, pcfg, include_jev=False)
    engines = harness.default_engines(development)
    stub = JevDecisionEngine(dcfg, mode="stub")
    stub.name = "jev_stub"
    proxy = JevDecisionEngine(dcfg, mode="llm_proxy")
    proxy.name = "llm_proxy"
    engines.extend([stub, proxy])
    return harness.evaluate(
        eval_eps, reference_episodes=development, engines=engines, top_k=k
    )


def run_ablation(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
    top_k: int | None = None,
) -> AblationResult:
    base = load_m1_config() if cfg is None else dict(cfg)
    n = int(n_episodes if n_episodes is not None else base.get("n_episodes", 150))
    sd = int(seed if seed is not None else base.get("seed", 42))
    k = int(top_k if top_k is not None else base.get("top_k", 15))

    corpus, data_source, load_notes = _build_corpus(
        base, force_synthetic=force_synthetic, n=n, sd=sd
    )
    development, eval_eps = split_dev_eval(corpus, sd)
    n_gt_fails = sum(1 for e in eval_eps if e.ground_truth.success is False)

    shared_notes = list(load_notes) + [
        "Reference = development (Frequency/Baseline/JEV fit + proxy bank). "
        "Eval = public_eval-style holdout. hidden_eval never loaded.",
        "JEV stub + llm_proxy are local heuristics — **not** live Jev API; "
        "no credits spent.",
    ]

    pcfg = load_proxy_config(ROOT / base["proxy_config"] if "proxy_config" in base else None)
    pcfg = ProxyImportanceConfig(
        rare_freq_quantile=pcfg.rare_freq_quantile,
        high_freq_min_count=pcfg.high_freq_min_count,
        novelty_min_score=pcfg.novelty_min_score,
        top_k=k,
    )

    runs: dict[str, AblationRun] = {}
    for label, ablate in (("leaked", False), ("ablated", True)):
        dcfg = _make_dcfg(base, k, ablate=ablate)
        report = _run_ladder(development, eval_eps, dcfg, pcfg, k)
        note = (
            "path cues intact (current / leaked baseline)."
            if not ablate
            else "path cues stripped via sanitize_path_text + emptied "
            "fail/success_path_tokens (LEAK_ABLATE / ablate_path_leak)."
        )
        runs[label] = AblationRun(
            label=label,
            data_source=data_source,
            seed=sd,
            n_episodes=len(corpus),
            n_development=len(development),
            n_eval=len(eval_eps),
            top_k=k,
            harness=report,
            n_gt_fails_eval=n_gt_fails,
            notes=shared_notes + [note],
        )

    return AblationResult(
        leaked=runs["leaked"],
        ablated=runs["ablated"],
        notes=shared_notes
        + [
            "Ablated run is the fairer detection ceiling; leaked F1≈1 was path leakage.",
        ],
    )


def _col(run: AblationRun, backend: str) -> LadderColumn | None:
    return run.harness.column(backend)


def render_report(result: AblationResult) -> str:
    leaked, ablated = result.leaked, result.ablated
    src_label = (
        "real RoboFAC (test_qa_sim labeled subset)"
        if leaked.data_source == "robofac_real"
        else "fixtures + deterministic synthetic (path cues)"
    )
    backends = ["random", "frequency", "baseline", "jev_stub", "llm_proxy"]

    lines: list[str] = [
        "# M1 path-label leakage ablation",
        "",
        "> **Honesty:** High Baseline / llm_proxy detection F1 on RoboFAC was "
        "driven by **success/fail path directory cues** "
        "(`dataset_success_cleaned`, `stack_ok`, `fail`, …) embedded in "
        "`path_text`. The **ablated** column strips those cues "
        "(`features.sanitize` + emptied `fail_path_tokens` / "
        "`success_path_tokens`). **Ablated is the fairer detection ceiling**; "
        "leaked F1=1 was path leakage, not perception. "
        "**review priority ≠ business importance.** "
        "llm_proxy / stub ≠ live Jev API (no credits).",
        "",
        "## Run metadata",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Data source | {src_label} |",
        f"| Corpus / seed | n={leaked.n_episodes}, seed={leaked.seed} |",
        f"| Development / eval | {leaked.n_development} / holdout n={leaked.n_eval} |",
        f"| Top-K | {leaked.top_k} |",
        f"| GT fails on eval | {leaked.n_gt_fails_eval} |",
        "| Hidden eval | never loaded |",
        "| Jev API | not called |",
        "",
        "### Notes",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")

    lines += [
        "",
        "## Side-by-side: leaked vs ablated",
        "",
        "| Backend | Leaked F1 | Ablated F1 | Leaked type | Ablated type | "
        "Leaked proxy ret | Ablated proxy ret |",
        "|---------|-----------|------------|-------------|--------------|"
        "------------------|-------------------|",
    ]
    for b in backends:
        lc = _col(leaked, b)
        ac = _col(ablated, b)
        if not lc or not ac:
            continue
        lines.append(
            f"| {b} | {lc.detection_f1:.3f} | {ac.detection_f1:.3f} | "
            f"{lc.failure_type_agreement:.3f} | {ac.failure_type_agreement:.3f} | "
            f"{lc.proxy_retention:.3f} | {ac.proxy_retention:.3f} |"
        )

    lines += [
        "",
        "## Full leaked ladder",
        "",
        "| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention |",
        "|---------|-------|-------|--------|------------|-----------------|",
    ]
    for c in leaked.harness.columns:
        lines.append(
            f"| {c.backend} | {c.detection_precision:.3f} | {c.detection_recall:.3f} | "
            f"{c.detection_f1:.3f} | {c.failure_type_agreement:.3f} | "
            f"{c.proxy_retention:.3f} |"
        )

    lines += [
        "",
        "## Full ablated ladder",
        "",
        "| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention |",
        "|---------|-------|-------|--------|------------|-----------------|",
    ]
    for c in ablated.harness.columns:
        lines.append(
            f"| {c.backend} | {c.detection_precision:.3f} | {c.detection_recall:.3f} | "
            f"{c.detection_f1:.3f} | {c.failure_type_agreement:.3f} | "
            f"{c.proxy_retention:.3f} |"
        )

    base_l, base_a = _col(leaked, "baseline"), _col(ablated, "baseline")
    proxy_l, proxy_a = _col(leaked, "llm_proxy"), _col(ablated, "llm_proxy")
    stub_l, stub_a = _col(leaked, "jev_stub"), _col(ablated, "jev_stub")

    lines += ["", "## What this tells us", ""]
    if base_l and base_a:
        lines.append(
            f"- **Baseline** detection F1 {base_l.detection_f1:.3f} → "
            f"{base_a.detection_f1:.3f} after stripping path-label cues "
            f"(Δ {base_a.detection_f1 - base_l.detection_f1:+.3f}). "
            "Leaked F1≈1 was directory leakage."
        )
    if proxy_l and proxy_a:
        lines.append(
            f"- **llm_proxy** detection F1 {proxy_l.detection_f1:.3f} → "
            f"{proxy_a.detection_f1:.3f} (Δ "
            f"{proxy_a.detection_f1 - proxy_l.detection_f1:+.3f}); "
            f"proxy retention {proxy_l.proxy_retention:.3f} → "
            f"{proxy_a.proxy_retention:.3f}."
        )
    if stub_l and stub_a:
        lines.append(
            f"- **jev_stub** detection F1 {stub_l.detection_f1:.3f} → "
            f"{stub_a.detection_f1:.3f}."
        )
    lines += [
        "- Random / Frequency ignore path tokens, so leaked≈ablated (sanity).",
        "- Residual ablated signal may remain from type-semantic folders "
        "(`position_offset`, `gripper_error`, …) or instruction text — that is "
        "content, not success/fail directory labels. `stack_error` under success "
        "trees can still confuse llm_proxy's softer `error` cue.",
        "- Prefer **ablated** metrics when quoting a detection ceiling.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/m1_leak_ablation.py",
        "# or: LEAK_ABLATE=1 on any M1 runner after this change",
        "# optional: --force-synthetic --out reports/m1_leak_ablation.md",
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
    result = run_ablation(
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
        f"source={result.leaked.data_source} n={result.leaked.n_episodes} "
        f"eval={result.leaked.n_eval} top_k={result.leaked.top_k}"
    )
    for b in ("baseline", "llm_proxy", "jev_stub", "frequency", "random"):
        lc = result.leaked.harness.column(b)
        ac = result.ablated.harness.column(b)
        if lc and ac:
            print(
                f"  {b}: leaked F1={lc.detection_f1:.3f} → ablated F1={ac.detection_f1:.3f} "
                f"(type {lc.failure_type_agreement:.3f}→{ac.failure_type_agreement:.3f}, "
                f"proxy_ret {lc.proxy_retention:.3f}→{ac.proxy_retention:.3f})"
            )


if __name__ == "__main__":
    main()
