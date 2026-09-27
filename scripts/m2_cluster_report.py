#!/usr/bin/env python3
"""M2 failure bank + clustering → markdown report.

Same RoboFAC thin-slice path as M1 (n≈150, seed=42). Seeds the bank from
**development** fails; clusters **eval** fails. Never loads hidden_eval.
No Streamlit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clustering import (  # noqa: E402
    Clusterer,
    cluster_purity,
    cluster_size_distribution,
    compression_ratio,
)
from failure_bank import FailureBank, FailureCase  # noqa: E402
from proxy_importance.config import load_config as load_proxy_config  # noqa: E402
from proxy_importance.failure_bank import FailureBank as ProxyTypeBank  # noqa: E402
from proxy_importance.frequency import build_frequency_table  # noqa: E402
from ranking import Ranker, load_weights  # noqa: E402
from scripts.m0_5_thin_slice import (  # noqa: E402
    build_fixture_synthetic,
    sample_corpus,
    split_dev_eval,
    try_load_robofac,
)

DEFAULT_CONFIG = ROOT / "configs" / "m2_clusters.json"


@dataclass
class M2Result:
    data_source: str
    seed: int
    n_episodes: int
    n_development: int
    n_eval: int
    n_bank_cases: int
    n_clustered: int
    n_clusters: int
    purity: float  # labeled-only (ignore_none) — primary gate metric
    purity_all: float  # includes untyped fails in denominator
    n_typed: int
    compression: float
    size_dist: dict[int, int]
    top_ranked: list[tuple[str, float, tuple[str, ...]]]
    notes: list[str] = field(default_factory=list)
    bank_path: str = ""
    cluster_source: str = "eval_fails"


def load_m2_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or DEFAULT_CONFIG
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _fails(episodes: list) -> list:
    return [e for e in episodes if e.ground_truth.success is False]


def run_m2(
    cfg: dict[str, Any] | None = None,
    *,
    force_synthetic: bool = False,
    n_episodes: int | None = None,
    seed: int | None = None,
) -> M2Result:
    base = load_m2_config() if cfg is None else dict(cfg)
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
        "Bank seeded from **development** fails; clustering on **eval** fails "
        "(public_eval-style holdout). hidden_eval never loaded."
    )

    pcfg = load_proxy_config(ROOT / base["proxy_config"] if "proxy_config" in base else None)
    # Reference for B/C/D = development (same isolation as proxy / M1).
    freq = build_frequency_table(development, pcfg)
    proxy_bank = ProxyTypeBank.from_episodes(development)

    bank = FailureBank.from_episodes(
        development,
        reference=development,
        config=pcfg,
        path=ROOT / base.get("bank_out", "data/artifacts/failure_bank.jsonl"),
    )
    # Also upsert eval fails into the artifact (full bank snapshot).
    from failure_bank.store import case_from_episode

    eval_fail_eps = _fails(eval_eps)
    eval_cases: list[FailureCase] = []
    for ep in eval_fail_eps:
        case = case_from_episode(ep, freq, proxy_bank, pcfg)
        bank.upsert(case)
        eval_cases.append(case)

    bank_path = bank.save()
    notes.append(f"Wrote failure bank JSONL → {bank_path.relative_to(ROOT)} ({len(bank)} cases).")

    cluster_source = str(base.get("cluster_source", "eval_fails"))
    if cluster_source == "all_bank":
        to_cluster = bank.all_cases()
    else:
        to_cluster = eval_cases
        cluster_source = "eval_fails"

    n_clusters_cfg = base.get("n_clusters")
    n_clusters_arg = int(n_clusters_cfg) if n_clusters_cfg is not None else None
    dist_thr = base.get("distance_threshold", 0.55)
    dist_thr_arg = float(dist_thr) if dist_thr is not None else None

    if not to_cluster:
        notes.append("No fail cases to cluster.")
        return M2Result(
            data_source=data_source,
            seed=sd,
            n_episodes=n,
            n_development=len(development),
            n_eval=len(eval_eps),
            n_bank_cases=len(bank),
            n_clustered=0,
            n_clusters=0,
            purity=0.0,
            purity_all=0.0,
            n_typed=0,
            compression=0.0,
            size_dist={},
            top_ranked=[],
            notes=notes,
            bank_path=str(bank_path.relative_to(ROOT)),
            cluster_source=cluster_source,
        )

    clusterer = Clusterer(
        dim=int(base.get("embed_dim", 64)),
        n_clusters=n_clusters_arg,
        distance_threshold=dist_thr_arg if n_clusters_arg is None else None,
        linkage=str(base.get("linkage", "average")),
    )
    cres = clusterer.fit_predict(to_cluster)
    labels = [c.failure_type for c in to_cluster]
    purity_all = cluster_purity(cres.labels, labels, ignore_none=False)
    purity = cluster_purity(cres.labels, labels, ignore_none=True)
    n_typed = sum(1 for lab in labels if lab is not None)
    size_dist = cluster_size_distribution(cres.labels)
    comp = compression_ratio(len(to_cluster), cres.n_clusters)

    wcfg = load_weights(ROOT / base["ranking_config"] if "ranking_config" in base else None)
    ranker = Ranker(wcfg)
    ranked = ranker.rank(to_cluster)
    top_ranked = [(r.case_id, r.score, r.rule_hits) for r in ranked[:15]]

    notes.append(
        "Embedding = deterministic bag-of-tokens hashing on path/instruction/task "
        "(GT-free; failure_type excluded from vectors)."
    )
    notes.append(
        "Purity = label consistency vs failure_type only — **not** engineering usefulness. "
        "Review-priority ranking ≠ business importance."
    )

    notes.append(
        f"Typed fails (failure_type set) among clustered: {n_typed}/{len(to_cluster)}. "
        "Primary purity uses labeled-only (ignore_none)."
    )

    return M2Result(
        data_source=data_source,
        seed=sd,
        n_episodes=n,
        n_development=len(development),
        n_eval=len(eval_eps),
        n_bank_cases=len(bank),
        n_clustered=len(to_cluster),
        n_clusters=cres.n_clusters,
        purity=purity,
        purity_all=purity_all,
        n_typed=n_typed,
        compression=comp,
        size_dist=size_dist,
        top_ranked=top_ranked,
        notes=notes,
        bank_path=str(bank_path.relative_to(ROOT)),
        cluster_source=cluster_source,
    )


def render_report(result: M2Result) -> str:
    sizes = sorted(result.size_dist.values(), reverse=True)
    size_hist = Counter(sizes)
    size_lines = ", ".join(f"size={s}×{cnt}" for s, cnt in sorted(size_hist.items(), reverse=True))
    ranked_rows = "\n".join(
        f"| `{cid}` | {score:.3f} | {','.join(hits) or '—'} |"
        for cid, score, hits in result.top_ranked
    ) or "| — | — | — |"

    return f"""# M2 — Failure bank + clustering

**Status:** auto-generated by `scripts/m2_cluster_report.py`  
**Data:** `{result.data_source}` · n={result.n_episodes} · seed={result.seed}  
**Split:** development={result.n_development} · eval={result.n_eval} (hidden_eval **never** loaded)  
**Bank artifact:** `{result.bank_path}` ({result.n_bank_cases} cases)  
**Cluster source:** `{result.cluster_source}`

## Honesty (read first)

- **Cluster purity = label consistency only** vs `failure_type`. It is **not** engineering usefulness.
- **Review-priority ranking ≠ business importance** (proxy rules A–D; E/F bonuses only).
- No Streamlit / dashboard in this milestone.

## Compression sketch

| Metric | Value |
|--------|------:|
| Fail cases clustered | {result.n_clustered} |
| Typed fails (for purity) | {result.n_typed} |
| Clusters | {result.n_clusters} |
| Compression (fails / clusters) | {result.compression:.2f} |
| Auto purity vs `failure_type` (labeled-only) | {result.purity:.3f} |
| Purity including untyped (denom=all) | {result.purity_all:.3f} |

Size distribution: {size_lines or "—"}

## Top-15 review-priority (eval fails)

| case_id | score | rule_hits |
|---------|------:|-----------|
{ranked_rows}

## Notes

""" + "\n".join(f"- {n}" for n in result.notes) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--force-synthetic", action="store_true")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    cfg = load_m2_config(args.config)
    result = run_m2(
        cfg,
        force_synthetic=args.force_synthetic,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    text = render_report(result)
    out = args.out or ROOT / cfg.get("report_out", "reports/m2_clusters.md")
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    print(
        f"n_fails={result.n_clustered} n_typed={result.n_typed} "
        f"n_clusters={result.n_clusters} purity_labeled={result.purity:.3f} "
        f"purity_all={result.purity_all:.3f} compression={result.compression:.2f} "
        f"source={result.data_source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
