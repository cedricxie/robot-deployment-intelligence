# Robot Deployment Intelligence MVP

Evaluation / Failure Intelligence for real robot deployment data.

This repository starts as a **docs-first** MVP: prove a small closed loop from deployment episodes → failure detection/classification → clustering → ranking → bounded RSI, without training robot policies.

## Status

- [x] Implementation plan (`docs/mvp-plan.md`)
- [ ] M0 Dataset pipeline
- [ ] M1 Failure intelligence
- [ ] M2 Failure bank + clustering
- [ ] M3 Cascade economics
- [ ] M4 Bounded RSI
- [ ] M5 Demo + final report

## Quick links

- [MVP Implementation Plan](docs/mvp-plan.md)

## Principles

Evaluator-first · Data-first · API-first · Modular · Measurable · Reproducible

API keys (when used) are read from environment variables only. The pipeline must run with mock providers when keys are absent.
