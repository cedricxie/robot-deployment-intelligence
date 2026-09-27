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
- [ ] M1 Failure intelligence
- [ ] M2 Failure bank + clustering
- [ ] M3 Cascade economics
- [ ] M4 Evaluator improvement loop
- [ ] M5 Demo + Streamlit + final report

## Quick links

- [MVP Implementation Plan](docs/mvp-plan.md) — **read Goals + Validation before coding**
- [LeRobot visualization preview](docs/lerobot-visualization.md) — RoboFAC→Hub (`cedricxie/robofac-lerobot-preview`)
- [Dataset inventory](data/DATA.md)

## Principles

Goals-first · Evaluator-first · Data-first · API-first · Modular · Measurable · Reproducible

API keys (when used) come from environment variables only. The pipeline must run with mock providers when keys are absent.

## Decision ladder (PR-impl-4)

Runnable rungs: **Random** → **Frequency-only** → **BaselineFastDecisionEngine** (JEV later).

| Rung | Behavior |
|------|----------|
| Random | Seeded Bernoulli success/fail + uniform type |
| Frequency-only | Majority success/fail + most-common fail type from **development** reference empirical freqs (constant predictor) |
| BaselineFast | GT-free path/id keyword cues (+ optional prior / type keywords). Default ship path |

Eval harness (`evaluation/harness.py`) emits side-by-side columns: detection P/R/F1, failure-type agreement, **proxy retention**, **review reduction** (Top-K). Proxy labels come from `proxy_importance` (A∧(B∨C∨D)); improvement-loop helpers still must not load `hidden_eval`.

**Baseline miss modes:** renamed paths without `fail`/`success` tokens; no type keywords in path/instruction → falls back to development prior / majority type (may tie Frequency). On RoboFAC-shaped fixtures with path cues, Baseline beats Random on detection F1.
