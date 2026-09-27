#!/usr/bin/env python3
"""M0.5 thin slice: Episode → features → ladder → proxy → Top-K → markdown.

Runs without API keys. Prefer real RoboFAC test_qa (labeled) when present;
else expand fixtures/synthetic with path cues. Never loads hidden_eval.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.adapters.robofac import load_episodes  # noqa: E402
from data.schemas.episode import Episode, GroundTruth  # noqa: E402
from decision.base import DecisionConfig, DecisionResult, load_decision_config  # noqa: E402
from decision.baseline import BaselineFastDecisionEngine  # noqa: E402
from evaluation.harness import EvaluationHarness, HarnessReport  # noqa: E402
from features.extract import FeatureExtractor  # noqa: E402
from proxy_importance.config import ProxyImportanceConfig, load_config as load_proxy_config  # noqa: E402
from proxy_importance.failure_bank import FailureBank  # noqa: E402
from proxy_importance.frequency import build_frequency_table  # noqa: E402
from proxy_importance.rules import ProxyReviewPriorityResult, score_episodes  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "m0_5_thin_slice.json"

# Failure types for synthetic expansion (path tokens match Baseline heuristics).
_SYNTH_TYPES = (
    "position_deviation",
    "position_deviation",
    "position_deviation",
    "step_omission",
    "step_omission",
    "orientation_deviation",
    "grasping_error",
    "timing_error",
    "wrong_target_object",
)


@dataclass
class ThinSliceResult:
    """Artifacts from one thin-slice run (for report + tests)."""

    data_source: str  # "robofac_real" | "fixture_synthetic"
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    top_k: int
    eval_split_name: str
    harness: HarnessReport
    baseline_results: list[DecisionResult]
    proxy_results: list[ProxyReviewPriorityResult]
    eval_episodes: list[Episode]
    development: list[Episode]
    n_gt_fails_eval: int = 0
    notes: list[str] = field(default_factory=list)


def load_thin_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _labeled_only(eps: list[Episode]) -> list[Episode]:
    return [e for e in eps if e.ground_truth.success is not None]


def try_load_robofac(cfg: dict[str, Any]) -> list[Episode]:
    """Load labeled episodes from on-disk RoboFAC test_qa (adapter format)."""
    paths: list[Path] = []
    for key in ("robofac_sim_glob", "robofac_real_glob"):
        if key == "robofac_real_glob" and not cfg.get("use_realworld", False):
            continue
        pattern = cfg.get(key)
        if not pattern:
            continue
        paths.extend(sorted(ROOT.glob(pattern)))
    if not paths:
        return []
    eps = _labeled_only(load_episodes(paths))
    # Dedup by episode_id (stable first-seen).
    seen: set[str] = set()
    out: list[Episode] = []
    for ep in eps:
        if ep.episode_id in seen:
            continue
        seen.add(ep.episode_id)
        out.append(ep)
    return out


def build_fixture_synthetic(n: int, seed: int, fixture_dir: Path) -> list[Episode]:
    """Deterministic fixture + synthetic corpus with path cues / typed fails."""
    fixtures = load_episodes(sorted(fixture_dir.glob("*.json"))) if fixture_dir.is_dir() else []
    rng = random.Random(seed)
    synth: list[Episode] = []
    # Ensure enough success + fail with Baseline path tokens.
    target = max(n, 100)
    i = 0
    while len(fixtures) + len(synth) < target:
        if i % 5 == 0:
            # success with success path cue
            synth.append(
                Episode(
                    episode_id=f"synth-ok-{i:04d}",
                    task="ThinSliceTask",
                    instruction="Pick and place successfully",
                    video_paths=[f"dataset_success_cleaned/view0/synth-ok-{i:04d}.mp4"],
                    metadata={"source": "synthetic_thin_slice"},
                    ground_truth=GroundTruth(success=True),
                )
            )
        else:
            ft = _SYNTH_TYPES[i % len(_SYNTH_TYPES)]
            token = {
                "position_deviation": "fail_position",
                "step_omission": "fail_omit",
                "orientation_deviation": "fail_orient",
                "grasping_error": "fail_grasp",
                "timing_error": "fail_timing",
                "wrong_target_object": "fail_wrong_target",
            }[ft]
            synth.append(
                Episode(
                    episode_id=f"synth-fail-{i:04d}",
                    task="ThinSliceTask",
                    instruction=f"Task with {ft.replace('_', ' ')}",
                    video_paths=[f"sim/{token}/view0/synth-fail-{i:04d}.mp4"],
                    metadata={"source": "synthetic_thin_slice", "failure_subtask": "reach"},
                    ground_truth=GroundTruth(success=False, failure_type=ft),
                )
            )
        i += 1
    corpus = list(fixtures) + synth
    rng.shuffle(corpus)
    return corpus[:n]


def sample_corpus(episodes: list[Episode], n: int, seed: int) -> list[Episode]:
    """Deterministic sample; stratify success/fail when both exist."""
    if len(episodes) <= n:
        return list(episodes)
    rng = random.Random(seed)
    ok = [e for e in episodes if e.ground_truth.success is True]
    bad = [e for e in episodes if e.ground_truth.success is False]
    other = [e for e in episodes if e.ground_truth.success is None]
    if not ok or not bad:
        pool = list(episodes)
        rng.shuffle(pool)
        return pool[:n]
    # Target ~proportionate mix, but keep ≥20% of the minority class when possible.
    n_ok_nat = max(1, round(n * len(ok) / len(episodes)))
    n_ok = min(len(ok), max(n_ok_nat, min(len(ok), max(1, n // 5))))
    n_bad = min(len(bad), n - n_ok)
    if n_ok + n_bad < n and other:
        n_other = min(len(other), n - n_ok - n_bad)
    else:
        n_other = 0
        # top up from majority if short
        short = n - n_ok - n_bad
        if short > 0 and len(bad) > n_bad:
            n_bad = min(len(bad), n_bad + short)
        short = n - n_ok - n_bad
        if short > 0 and len(ok) > n_ok:
            n_ok = min(len(ok), n_ok + short)
    rng.shuffle(ok)
    rng.shuffle(bad)
    rng.shuffle(other)
    out = ok[:n_ok] + bad[:n_bad] + other[:n_other]
    rng.shuffle(out)
    return out[:n]


def split_dev_eval(
    corpus: list[Episode], seed: int
) -> tuple[list[Episode], list[Episode]]:
    """Thin-slice 70/30 development vs public_eval-style; never uses hidden_eval."""
    by_id = {ep.episode_id: ep for ep in corpus}
    ids = list(by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    # ≥25% eval, at least 1; for tiny n keep ~30%
    n_eval = max(1, min(n - 1, max(n * 30 // 100, 1))) if n >= 2 else n
    n_dev = n - n_eval
    development = [by_id[i] for i in ids[:n_dev]]
    public_eval = [by_id[i] for i in ids[n_dev:]]
    return development, public_eval


def run_thin_slice(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
    top_k: int | None = None,
) -> ThinSliceResult:
    """Core runner: load → split → ladder → proxy → ranking artifacts."""
    base = load_thin_config() if cfg is None else dict(cfg)
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
        "Reference = development (Frequency/Baseline fit + proxy bank). "
        "Eval = public_eval-style holdout. hidden_eval never loaded."
    )

    dcfg = load_decision_config(ROOT / base["decision_config"] if "decision_config" in base else None)
    # Override review_top_k from thin-slice config for this run.
    dcfg = DecisionConfig(
        random_seed=dcfg.random_seed,
        review_top_k=k,
        fail_path_tokens=dcfg.fail_path_tokens,
        success_path_tokens=dcfg.success_path_tokens,
        type_keywords=dcfg.type_keywords,
        default_fail_probability=dcfg.default_fail_probability,
    )
    pcfg = load_proxy_config(ROOT / base["proxy_config"] if "proxy_config" in base else None)
    pcfg = ProxyImportanceConfig(
        rare_freq_quantile=pcfg.rare_freq_quantile,
        high_freq_min_count=pcfg.high_freq_min_count,
        novelty_min_score=pcfg.novelty_min_score,
        top_k=k,
    )

    harness = EvaluationHarness(dcfg, pcfg)
    report = harness.evaluate(eval_eps, reference_episodes=development, top_k=k)

    # Baseline results + proxy for ranking examples.
    extractor = FeatureExtractor()
    feats = extractor.extract_many(eval_eps)
    baseline = BaselineFastDecisionEngine(dcfg).fit(development)
    baseline_results = baseline.evaluate_many(feats, dcfg)
    freq_table = build_frequency_table(development, pcfg)
    bank = FailureBank.from_episodes(development)
    proxy_results = score_episodes(eval_eps, freq=freq_table, bank=bank, config=pcfg)

    n_gt_fails = sum(1 for e in eval_eps if e.ground_truth.success is False)

    return ThinSliceResult(
        data_source=data_source,
        seed=sd,
        n_episodes=len(corpus),
        n_development=len(development),
        n_eval=len(eval_eps),
        top_k=k,
        eval_split_name="public_eval",
        harness=report,
        baseline_results=baseline_results,
        proxy_results=proxy_results,
        eval_episodes=eval_eps,
        development=development,
        n_gt_fails_eval=n_gt_fails,
        notes=notes,
    )


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_report(result: ThinSliceResult) -> str:
    """Markdown report with required M0.5 sections."""
    src_label = (
        "real RoboFAC (test_qa_sim labeled subset)"
        if result.data_source == "robofac_real"
        else "fixtures + deterministic synthetic (path cues)"
    )
    lines: list[str] = [
        "# M0.5 thin-slice end-to-end report",
        "",
        "> **Disclaimer:** **review priority ≠ business importance.** "
        "`proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business "
        "criticality or confirmed model-iteration impact. "
        "BaselineFast is GT-free path/id heuristics — miss modes: renamed paths "
        "without `fail`/`success` tokens; missing type keywords → development prior / "
        "majority type (may tie Frequency).",
        "",
        "## Run metadata",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Data source | {src_label} |",
        f"| Dataset size (thin corpus) | {result.n_episodes} |",
        f"| Seed | {result.seed} |",
        f"| Development (reference) | {result.n_development} |",
        f"| Eval split | `{result.eval_split_name}` (holdout) n={result.n_eval} |",
        f"| Top-K | {result.top_k} |",
        f"| API keys | none required |",
        "",
        "### Notes",
        "",
    ]
    for note in result.notes:
        lines.append(f"- {note}")
    lines += ["", "## Ladder metrics (eval holdout)", ""]
    lines += [
        "| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention | Review reduction | n_proxy_imp |",
        "|---------|-------|-------|--------|------------|-----------------|------------------|-------------|",
    ]
    for c in result.harness.columns:
        lines.append(
            f"| {c.backend} | {c.detection_precision:.3f} | {c.detection_recall:.3f} | "
            f"{c.detection_f1:.3f} | {c.failure_type_agreement:.3f} | "
            f"{c.proxy_retention:.3f} | {c.review_reduction:.3f} | {c.n_proxy_important} |"
        )

    base_col = result.harness.column("baseline")
    n_imp = base_col.n_proxy_important if base_col else 0
    retention = base_col.proxy_retention if base_col else 0.0
    reduction_vs_all = base_col.review_reduction if base_col else 0.0
    # vs reviewing all GT fails on eval
    reduction_vs_fails = (
        1.0 - (result.top_k / result.n_gt_fails_eval) if result.n_gt_fails_eval else 0.0
    )
    reduction_vs_fails = max(0.0, reduction_vs_fails)

    lines += [
        "",
        "## Proxy-important / attention sketch (Baseline Top-K)",
        "",
        f"- Proxy-important count on eval: **{n_imp}**",
        f"- Retention of proxy-important in Top-{result.top_k}: **{_fmt_pct(retention)}**",
        f"- Review reduction vs reviewing all eval episodes: **{_fmt_pct(reduction_vs_all)}** "
        f"(review {result.top_k}/{result.n_eval})",
        f"- Review reduction vs reviewing all GT fails on eval: **{_fmt_pct(reduction_vs_fails)}** "
        f"(review {result.top_k}/{result.n_gt_fails_eval} fails)",
        "",
        "## Ranking examples (Baseline Top-K)",
        "",
        "| Rank | episode_id | p_fail | proxy | rule hits | GT success | GT type |",
        "|------|------------|--------|-------|-----------|------------|---------|",
    ]
    by_proxy = {p.episode_id: p for p in result.proxy_results}
    by_ep = {e.episode_id: e for e in result.eval_episodes}
    ranked = sorted(
        result.baseline_results,
        key=lambda r: (-r.failure_probability, r.episode_id),
    )[: result.top_k]
    for i, r in enumerate(ranked, 1):
        px = by_proxy.get(r.episode_id)
        ep = by_ep.get(r.episode_id)
        gt_s = ep.ground_truth.success if ep else None
        gt_t = ep.ground_truth.failure_type if ep else None
        hits = list(px.rule_hits) if px else []
        imp = "Y" if (px and px.proxy_important) else "N"
        lines.append(
            f"| {i} | `{r.episode_id}` | {r.failure_probability:.2f} | {imp} | "
            f"{hits} | {gt_s} | {gt_t} |"
        )

    # Honest section on rule-based effect
    rnd = result.harness.column("random")
    freq = result.harness.column("frequency")
    base = result.harness.column("baseline")
    lines += [
        "",
        "## What this tells us about rule-based effect",
        "",
    ]
    if base and rnd and freq:
        lines.append(
            f"On this slice (source={result.data_source}), Baseline detection F1="
            f"{base.detection_f1:.3f} vs Frequency={freq.detection_f1:.3f} vs "
            f"Random={rnd.detection_f1:.3f}."
        )
        if result.data_source == "robofac_real":
            lines.append(
                "RoboFAC success trajectories often live under `dataset_success_cleaned` "
                "(strong success path cue); many fails lack `fail`/`failure` path tokens, "
                "so Baseline leans on the development fail prior for cue-less fails. "
                "That works when the prior is fail-heavy (typical here) but is still a "
                "**heuristic**, not perception. Type agreement stays limited: many "
                "RoboFAC fails have null `failure_type` in QA, and keyword→type mapping "
                "only fires when path/instruction carries type tokens."
            )
        else:
            lines.append(
                "Fixture/synthetic corpus embeds explicit `fail_*` / `success` path "
                "tokens so Baseline can beat Random on detection; Frequency (majority "
                "fail) also scores high when the slice is fail-heavy. This proves the "
                "plumbing (Episode→features→ladder→proxy→Top-K→report), not SOTA "
                "perception."
            )
        lines.append(
            f"Proxy retention {_fmt_pct(retention)} at Top-{result.top_k} is a "
            "**review-priority** sketch only — ranking by predicted fail probability "
            "does not encode business importance. Rule-based miss modes remain: "
            "renamed paths, cue collisions, and types absent from keyword tables."
        )
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/m0_5_thin_slice.py --config configs/m0_5_thin_slice.json",
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
    parser.add_argument("--top-k", type=int, default=None, help="Ranking / review Top-K")
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Skip RoboFAC and use fixture/synthetic corpus",
    )
    args = parser.parse_args(argv)

    cfg = load_thin_config(args.config)
    result = run_thin_slice(
        cfg,
        force_synthetic=args.force_synthetic,
        n_episodes=args.n_episodes,
        seed=args.seed,
        top_k=args.top_k,
    )
    text = render_report(result)
    out = args.out or ROOT / cfg.get("report_out", "reports/m0_5_thin_slice.md")
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
            f"  {c.backend}: F1={c.detection_f1:.3f} "
            f"proxy_ret={c.proxy_retention:.3f} rev_red={c.review_reduction:.3f}"
        )


if __name__ == "__main__":
    main()
