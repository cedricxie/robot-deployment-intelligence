#!/usr/bin/env python3
"""M3 cascade economics: A (all-deep) vs B (cheap→fast→deep) → markdown.

Reuses M0.5/M1/M2 thin-slice loaders. Never loads hidden_eval.
Mock providers only — no API keys / network required.
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
from proxy_importance.config import load_config as load_proxy_config  # noqa: E402
from proxy_importance.failure_bank import FailureBank  # noqa: E402
from proxy_importance.frequency import build_frequency_table  # noqa: E402
from proxy_importance.rules import score_episodes  # noqa: E402
from reasoning.cascade import CascadeRunner, StrategyResult  # noqa: E402
from reasoning.routing import (  # noqa: E402
    RoutingPolicy,
    load_routing_policy,
    policy_to_dict,
)
from scripts.m0_5_thin_slice import (  # noqa: E402
    build_fixture_synthetic,
    sample_corpus,
    split_dev_eval,
    try_load_robofac,
)

DEFAULT_CONFIG = ROOT / "configs" / "m3_cascade.json"


@dataclass
class M3Result:
    data_source: str
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    policy: RoutingPolicy
    pairs: list[tuple[StrategyResult, StrategyResult]]
    notes: list[str] = field(default_factory=list)


def load_m3_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def run_m3(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
) -> M3Result:
    base = load_m3_config() if cfg is None else dict(cfg)
    n = int(n_episodes if n_episodes is not None else base.get("n_episodes", 150))
    sd = int(seed if seed is not None else base.get("seed", 42))
    min_labeled = int(base.get("min_labeled", 100))
    notes: list[str] = []

    data_source = "fixture_synthetic"
    if force_synthetic:
        corpus = build_fixture_synthetic(
            n, sd, ROOT / base.get("fixture_dir", "data/samples/robofac")
        )
        notes.append("Forced fixture/synthetic corpus.")
    else:
        robofac = try_load_robofac(base)
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
        "Reference = development (fast fit + proxy bank). "
        "Eval = public_eval-style holdout. hidden_eval never loaded."
    )

    dcfg = load_decision_config(
        ROOT / base["decision_config"] if "decision_config" in base else None
    )
    pcfg = load_proxy_config(
        ROOT / base["proxy_config"] if "proxy_config" in base else None
    )
    policy = load_routing_policy(
        ROOT / base["routing_config"] if "routing_config" in base else None
    )

    freq = build_frequency_table(development, pcfg)
    bank = FailureBank.from_episodes(development)
    proxy_results = score_episodes(eval_eps, freq=freq, bank=bank, config=pcfg)

    backends = tuple(base.get("fast_backends") or ["baseline", "jev"])
    runner = CascadeRunner(policy=policy, decision_config=dcfg)
    pairs = runner.compare(
        eval_eps,
        reference=development,
        proxy_results=proxy_results,
        fast_backends=backends,  # type: ignore[arg-type]
    )
    notes.append(
        "Strategies: A=all-deep; B=cheap→fast→deep. "
        "Mock VisionReasoningProvider / DeepReasoner (no network)."
    )
    notes.append(
        "Proxy-important recall = fail-detection recall on proxy-important subset. "
        "Review-priority ≠ business importance."
    )
    return M3Result(
        data_source=data_source,
        seed=sd,
        n_episodes=len(corpus),
        n_development=len(development),
        n_eval=len(eval_eps),
        policy=policy,
        pairs=pairs,
        notes=notes,
    )


def _speedup(a: StrategyResult, b: StrategyResult) -> float:
    if b.cost.deep_calls <= 0:
        return float("inf") if a.cost.deep_calls > 0 else 1.0
    return a.cost.deep_calls / b.cost.deep_calls


def _recall_drop(a: StrategyResult, b: StrategyResult) -> float:
    return a.proxy_important_recall - b.proxy_important_recall


def render_report(result: M3Result) -> str:
    pol = result.policy
    lines: list[str] = [
        "# M3 — Cascade economics (cheap → fast → deep)",
        "",
        "**Status:** auto-generated by `scripts/m3_cascade_report.py`",
        f"**Data:** `{result.data_source}` · n={result.n_episodes} · seed={result.seed}",
        f"**Split:** development={result.n_development} · eval={result.n_eval} "
        "(hidden_eval **never** loaded)",
        "",
        "## Honesty (read first)",
        "",
        "- **Review-priority / proxy-important ≠ business importance** "
        "(rules A∧(B∨C∨D)).",
        "- Mock providers only — no live LLM / API keys.",
        "- Routing policy is configurable JSON; deep budget caps escalations.",
        "",
        "## Routing policy",
        "",
        "```json",
        json.dumps(policy_to_dict(pol), indent=2),
        "```",
        "",
        "## A (all-deep) vs B (cascade)",
        "",
        "| Fast backend | Strat | Deep calls | Cheap calls | Fast calls | "
        "Cost units (Σ) | Det F1 | Type agree | Proxy-imp recall | n_proxy |",
        "|--------------|-------|-----------:|------------:|-----------:|"
        "---------------:|-------:|-----------:|-----------------:|--------:|",
    ]
    for a, b in result.pairs:
        for s in (a, b):
            label = "A all-deep" if s.strategy == "all_deep" else "B cascade"
            lines.append(
                f"| {s.fast_backend} | {label} | {s.cost.deep_calls} | "
                f"{s.cost.cheap_calls} | {s.cost.fast_calls} | "
                f"{s.cost.total_units:.1f} | {s.detection_f1:.3f} | "
                f"{s.type_agreement:.3f} | {s.proxy_important_recall:.3f} | "
                f"{s.n_proxy_important} |"
            )

    lines.extend(["", "## Gate check (M3 exit)", ""])
    lines.append(
        "| Fast backend | Deep A | Deep B | Speedup (A/B) | "
        "Proxy recall A | Proxy recall B | Recall drop (A−B) | ≥3×? | ≤5% drop? |"
    )
    lines.append(
        "|--------------|-------:|-------:|--------------:|"
        "---------------:|---------------:|------------------:|------:|----------:|"
    )
    for a, b in result.pairs:
        sp = _speedup(a, b)
        drop = _recall_drop(a, b)
        sp_s = f"{sp:.2f}×" if sp != float("inf") else "∞"
        ok_sp = "yes" if sp >= 3.0 else "no"
        ok_drop = "yes" if drop <= 0.05 + 1e-9 else "no"
        lines.append(
            f"| {a.fast_backend} | {a.cost.deep_calls} | {b.cost.deep_calls} | "
            f"{sp_s} | {a.proxy_important_recall:.3f} | "
            f"{b.proxy_important_recall:.3f} | {drop:+.3f} | {ok_sp} | {ok_drop} |"
        )

    lines.extend(["", "## Cost proxy detail (per ladder backend)", ""])
    lines.append(
        "| Fast backend | Strat | Cheap u | Fast u | Deep u | Fast tokens | Deep tokens |"
    )
    lines.append(
        "|--------------|-------|--------:|-------:|-------:|------------:|------------:|"
    )
    for a, b in result.pairs:
        for s in (a, b):
            label = "A" if s.strategy == "all_deep" else "B"
            lines.append(
                f"| {s.fast_backend} | {label} | {s.cost.cheap_units:.1f} | "
                f"{s.cost.fast_units:.1f} | {s.cost.deep_units:.1f} | "
                f"{s.cost.fast_tokens} | {s.cost.deep_tokens} |"
            )

    # Cascade stop breakdown for primary backend (baseline if present)
    primary = next((p for p in result.pairs if p[0].fast_backend == "baseline"), result.pairs[0])
    _, b0 = primary
    lines.extend(
        [
            "",
            "## Cascade B stop breakdown (baseline fast)",
            "",
            f"- Stopped at cheap (path cue): **{len(b0.stopped_at_cheap)}**",
            f"- Stopped at fast (no deep): **{len(b0.stopped_at_fast)}**",
            f"- Escalated to deep: **{len(b0.deep_ids)}**",
            "",
            "## Notes",
            "",
        ]
    )
    for note in result.notes:
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/m3_cascade_report.py --config configs/m3_cascade.json",
            "# optional: --force-synthetic",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--force-synthetic", action="store_true")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_m3_config(args.config)
    result = run_m3(
        cfg,
        force_synthetic=args.force_synthetic,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    md = render_report(result)
    out = args.out or ROOT / cfg.get("report_out", "reports/m3_cascade.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out}")
    for a, b in result.pairs:
        print(
            f"  {a.fast_backend}: deep A={a.cost.deep_calls} B={b.cost.deep_calls} "
            f"speedup={_speedup(a, b):.2f}× recall_drop={_recall_drop(a, b):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
