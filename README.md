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

## Principles

Goals-first · Evaluator-first · Data-first · API-first · Modular · Measurable · Reproducible

API keys (when used) come from environment variables only. The pipeline must run with mock providers when keys are absent.
