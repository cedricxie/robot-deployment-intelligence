# Robot Deployment Intelligence MVP

Evaluation / Failure Intelligence for real robot deployment data.

Prove that **deployment-only** robot data can form a small closed loop:
detect → classify → cluster → rank → recommend → bounded evaluator improvement —
without training robot policies.

## Status

- [x] Implementation plan (`docs/mvp-plan.md`) — rev 6.1 (proxy-important / review-priority + baseline ladder + impl PR plan + PR review gate)
- **Implementation PR gate:** after every update, obtain ChatGPT Career **Approve** (not **Need changes**), then ask the user for final merge confirmation.
- [ ] M0 Dataset pipeline
- [ ] M0.5 End-to-end thin slice
- [ ] M1 Failure intelligence (JEV stub ladder report landed; full M1 ship bar still open)
- [x] M2 Failure bank + clustering (PR-impl-7: bank + agglomerative + purity report)
- [x] M3 Cascade economics (PR-impl-8: cheap→fast→deep + cost metrics)
- [ ] M4 Evaluator improvement loop
- [ ] M5 Demo + Streamlit + final report

## Quick links

- [MVP Implementation Plan](docs/mvp-plan.md) — **read Goals + Validation before coding**
- [LeRobot visualization preview](docs/lerobot-visualization.md) — RoboFAC→Hub (`cedricxie/robofac-lerobot-preview`)
- [Dataset inventory](data/DATA.md)

## Principles

Goals-first · Evaluator-first · Data-first · API-first · Modular · Measurable · Reproducible

API keys (when used) come from environment variables only. The pipeline must run with mock providers when keys are absent.

## Decision ladder (PR-impl-4 + PR-impl-6)

Runnable rungs: **Random** → **Frequency-only** → **BaselineFastDecisionEngine** → **JEV** (stub via env `JEV_MODE`).

| Rung | Behavior |
|------|----------|
| Random | Seeded Bernoulli success/fail + uniform type |
| Frequency-only | Majority success/fail + most-common fail type from **development** reference empirical freqs (constant predictor) |
| BaselineFast | GT-free path/id keyword cues (+ optional prior / type keywords). Default ship path |
| JEV (stub / llm_proxy) | GT-free path + instruction cues with cost proxy. `llm_proxy` = deterministic systemone role-play (NOT live Jev). Modes: stub (default), llm_proxy, off, real (raises until adapter exists) |

Eval harness (`evaluation/harness.py`) emits side-by-side columns: detection P/R/F1, failure-type agreement, **proxy retention**, **review reduction** (Top-K), **cost proxy**. Proxy labels come from `proxy_importance` (A∧(B∨C∨D)); improvement-loop helpers still must not load `hidden_eval`.

**Baseline miss modes:** renamed paths without `fail`/`success` tokens; no type keywords in path/instruction → falls back to development prior / majority type (may tie Frequency). On RoboFAC-shaped fixtures with path cues, Baseline beats Random on detection F1.

### M1 failure intelligence (PR-impl-6)

Full ladder + cost–quality report (`reports/m1_failure_intelligence.md`):

```bash
JEV_MODE=stub python scripts/m1_failure_intelligence.py --config configs/m1_failure_intelligence.json
```


### Path-label leakage ablation

RoboFAC folder names (`dataset_success_cleaned`, `stack_ok`, `fail`, …) leak GT into `path_text`. Strip them and compare:

```bash
python scripts/m1_leak_ablation.py
# or: LEAK_ABLATE=1 on feature extract / DecisionConfig.ablate_path_leak=true
```

Report: `reports/m1_leak_ablation.md`. **Ablated metrics are the fairer detection ceiling**; leaked F1≈1 was path leakage. Career review pending before merge.

### M1 JEV llm_proxy (role-play)

Baseline vs deterministic systemone-shaped judge (`reports/m1_jev_llm_proxy.md`). **Not** live Jev API.

```bash
JEV_MODE=llm_proxy python scripts/m1_jev_llm_proxy.py
```

## M0.5 thin slice (PR-impl-5)

End-to-end: Episode → features → Random/Frequency/Baseline → proxy tagging → Top-K → `reports/m0_5_thin_slice.md`.

```bash
python scripts/m0_5_thin_slice.py --config configs/m0_5_thin_slice.json
# optional: --top-k 20 --force-synthetic
```

Prefers labeled RoboFAC `test_qa_sim` when present; else fixture/synthetic with path cues. No API keys. **Review priority ≠ business importance.**

## M2 failure bank + clustering (PR-impl-7)

JSONL `FailureCase` store, GT-free bag-of-tokens clustering, auto purity vs `failure_type` (**label consistency only** — not eng usefulness). Ranking weights honor A–D set + E/F bonuses. No Streamlit.

```bash
python scripts/m2_cluster_report.py --config configs/m2_clusters.json
# optional: --force-synthetic
```

Report: `reports/m2_clusters.md`. Artifact: `data/artifacts/failure_bank.jsonl` (gitignored). **Review priority ≠ business importance.**


## M3 cascade economics (PR-impl-8)

Cheap → fast → deep routing with A (all-deep) vs B (cascade) comparison. Mock `VisionReasoningProvider` / `DeepReasoner` (no API keys). Configurable policy in `configs/cascade_routing.json`.

```bash
JEV_MODE=stub python scripts/m3_cascade_report.py --config configs/m3_cascade.json
# optional: --force-synthetic
```

Report: `reports/m3_cascade.md` — deep-call speedup, cost proxy per ladder backend, proxy-important recall delta. **Review priority ≠ business importance.**


