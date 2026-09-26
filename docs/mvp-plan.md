# Robot Deployment Intelligence MVP — Implementation Plan

**Status:** Draft (docs-only)  
**Date:** 2026-09-26  
**Owner:** cedricxie  
**Repo goal:** Prove that deployment-only robot data can form a small, measurable closed loop: detect → classify → cluster → rank → recommend → improve via bounded RSI.

This document is the authoritative plan for implementation. It deliberately **scopes down** the original research brief into something a small team can ship as a runnable MVP without training policies, simulators, or a hard Jev dependency.

---

## 1. One-sentence goal

Build a modular pipeline that ingests ~500–2,000 RoboFAC episodes, compresses them into a small set of high-value failure cases/clusters, measures accuracy and review reduction, and runs at least one bounded RSI improvement loop with a human promote gate.

---

## 2. What “done” means for this MVP

The MVP is successful if we can run one command (or a short script sequence) that:

1. Loads a **fixed RoboFAC subset** into a unified `Episode` schema.
2. Runs **fast decision → optional deep reasoning → failure bank → clustering → ranking**.
3. Emits an **evaluation report** (metrics + cost/latency proxies).
4. Produces a **minimal Streamlit dashboard** for clusters / diagnoses / simulated human corrections.
5. Completes **≥3 RSI iterations** that only change configs/prompts/thresholds, with promote/reject on **hidden eval** (no auto-overwrite of production).

We are **not** aiming for paper SOTA. We are aiming for a reproducible product/research hypothesis test.

---

## 3. Scope decisions (vs original brief)

| Area | Original ask | MVP decision | Rationale |
|------|--------------|--------------|-----------|
| Dataset size | 1k–10k, then DROID | **500–2,000 RoboFAC episodes** first; DROID deferred | Avoid download/format blockers; keep iteration fast |
| Ground truth | Full diagnosis/correction | Use whatever RoboFAC labels exist; map into schema; mark missing fields | Adapter must tolerate incomplete GT |
| Fast decision | Baseline + real Jev | **Baseline required**; Jev adapter interface + mock | Do not block on TypeSafe Jev access |
| Vision / LLM | Commercial + local | Abstract providers; **mock + optional OpenAI-compatible env keys** | Runs offline by default |
| Deep reasoning | Strong VLM on hard cases | Triggered only when `needs_deep_review`; mockable | Prove cascade economics even with mocks |
| Clustering | HDBSCAN or agglomerative | Start with **embeddings + agglomerative**; HDBSCAN if deps easy | Prefer one working path |
| Dashboard | Streamlit or FastAPI+React | **Streamlit only** | Demo, not product UI |
| RSI | Prompt/threshold/routing/schema/features/clustering/ranking (+ optional code) | **Config + prompt + threshold + routing + ranking weights only** in v1 | Bounded, reviewable diffs |
| Human feedback | Simulated + UI | **Simulated from GT first**; Streamlit correction form wired to store | Real reviewers later |
| Learning curve | 100→2000 labelled ramp | **Optional stretch after M4**; not a ship blocker | Nice chart, not core loop |
| External datasets | DROID / Guardian / FailCoT | **Out of MVP critical path** | Portability proven by adapter interface + one real adapter |

---

## 4. Core questions the MVP must answer

| ID | Question | How we measure |
|----|----------|----------------|
| Q1 | Can we compress many episodes into few high-value reviews? | `review_reduction_ratio`, Top-N coverage of GT failures |
| Q2 | Is failure detection/classification usable vs GT? | Precision/recall/F1, macro-F1, confusion matrix |
| Q3 | Can failures cluster into ~15–50 issues? | Cluster count, silhouette/qualitative spot-check, size distribution |
| Q4 | Does cheap→fast→deep cascade save cost? | VLM/LLM calls per 1k episodes; recall delta ≤ ~5% vs all-deep (or mock-cost proxy) |
| Q5 | Does bounded RSI improve hidden eval? | ≥3 iterations; promote only if hidden score improves |

---

## 5. Non-goals (explicit)

Do **not** implement in MVP:

1. Training foundation models / VLA / RL policies  
2. Robot simulators or control stacks  
3. Distributed infra or multi-tenant productization  
4. Polished frontend beyond Streamlit demo  
5. Hard dependency on Jev API  
6. Auto-promote RSI candidates to production without human approval  
7. Full DROID / Guardian / FailCoT integration as a ship requirement  

---

## 6. Architecture (runnable slice)

```
RoboFAC subset
    → RoboFACAdapter
    → Episode (unified schema)
    → Feature / Summary Extraction (deterministic + optional VLM summary)
    → FastDecisionEngine (Baseline; Jev stub)
    → route: pass | failure/uncertain/high-value
    → DeepReasoner (selected only)
    → FailureBank
    → Clustering + Ranking
    → EvaluationHarness + Report
    → Streamlit Dashboard
    → HumanFeedbackStore (simulated or UI)
    → RSIHarness (candidates → replay → public eval → hidden eval → promote gate)
```

### 6.1 Unified Episode schema (minimum)

```yaml
episode_id: str
task: str | null
instruction: str | null
video_paths: list[str]
frames: list[str] | null
robot_state: object | null
actions: object | null
metadata: object
ground_truth:
  success: bool | null
  failure_type: str | null
  failure_timestamp: float | null
  diagnosis: str | null
  correction: str | null
model_output:
  failure_probability: float | null
  failure_type: str | null
  confidence: float | null
  severity: float | null
  novelty: float | null
  needs_deep_review: bool | null
  diagnosis: str | null
  evidence: list[str]
  recommended_actions: list[object]
```

Adapters must never leak RoboFAC-only field names into decision/reasoning/clustering code.

### 6.2 Interfaces (must exist even if one backend is mock)

- `DatasetAdapter.load() -> Iterable[Episode]`
- `FeatureExtractor.extract(episode) -> EpisodeFeatures`
- `FastDecisionEngine.evaluate(features, decision_schema) -> DecisionResult`
- `VisionReasoningProvider.summarize(frames, prompt) -> str`
- `DeepReasoner.reason(context) -> StructuredDiagnosis` (+ raw/version metadata)
- `FailureBank.upsert(FailureCase)`
- `Clusterer.fit_predict(cases) -> Clusters`
- `Ranker.score(case_or_cluster, weights) -> float`
- `EvaluationHarness.evaluate(predictions, labels, split) -> Metrics`
- `RSIHarness.propose_and_eval(production_config) -> CandidateResult` (with human gate)

---

## 7. Dataset plan (M0)

### 7.1 Primary: RoboFAC

- Sources: [MINT-SJTU/RoboFAC](https://github.com/MINT-SJTU/RoboFAC), [Hugging Face dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)
- **First download:** smallest usable sample that includes success + multiple failure types (target **500–2,000** episodes, not full dump)
- Persist under `data/raw/robofac/` (gitignored) and a checked-in `data/samples/` tiny fixture for CI

### 7.2 Splits (fixed seed)

| Split | Ratio | Who may see labels |
|-------|-------|--------------------|
| development | 60% | pipeline + RSI agent |
| public_eval | 20% | evaluation harness + RSI selection |
| hidden_eval | 20% | **evaluation harness only** |

`seed = 42` (configurable). RSI agent code paths must not load hidden labels.

### 7.3 Fallback if RoboFAC download/format fails

1. Use published metadata / video subset from the same project if available.  
2. Else synthesize a **tiny fixture dataset** matching the Episode schema (clearly marked synthetic) so pipeline tests still run.  
3. Record assumption in milestone report; do not block the repo.

DROID is **post-MVP validation**, not a dependency of M0–M5 ship.

---

## 8. Milestone plan

### M0 — Dataset pipeline (ship first)

**Deliverables**

- Repo layout + `README.md` (run instructions)
- `Episode` pydantic/dataclass schema
- `RoboFACAdapter` + unit tests on fixture
- Script: `scripts/m0_dataset_stats.py` → episode counts, success/failure rates, missing-field report
- Gitignored data paths; sample fixture committed

**Exit criteria**

- `pytest` green without API keys  
- Stats report written to `reports/m0_dataset_stats.md`

### M1 — Failure intelligence

**Deliverables**

- Deterministic feature extractor (duration, retries heuristics, success flags from metadata when present)
- `BaselineFastDecisionEngine` (rules + optional embedding/classifier + optional LLM structured output)
- `JevDecisionEngine` stub/mock implementing same interface
- Evaluation: detection P/R/F1; classification accuracy + macro-F1; confusion matrix
- Report: `reports/m1_failure_intelligence.md`

**Exit criteria (soft targets, analyze if miss)**

- Detection F1 ≥ **0.70** on public_eval **or** documented failure modes if not  
- High-severity recall tracked separately when severity proxy exists

### M2 — Failure bank + clustering + ranking

**Deliverables**

- `FailureCase` store (JSONL or SQLite)
- Embedding + agglomerative clustering (configurable `n_clusters` / distance threshold)
- High-value score with configurable weights:
  - `value = w_s*severity + w_f*frequency + w_n*novelty + w_u*uncertainty`
- Streamlit pages: overview, top clusters, case detail, GT vs prediction
- Report: `reports/m2_clusters.md`

**Exit criteria**

- Example: ≥100 labelled failures → ≤ **50** clusters (or ≤ max(20, 0.1 * N))  
- Manual spot-check notes for 5–10 representative clusters

### M3 — Cascade economics

**Deliverables**

- Deep reasoner behind provider abstraction (mock + optional real API)
- Routing policy: only `needs_deep_review` / failure / uncertain / novel
- Comparison A vs B:
  - A: all episodes → deep  
  - B: fast → selected deep  
- Metrics: calls/1k, estimated cost, latency proxy, recall delta
- Report: `reports/m3_cascade.md`

**Exit criteria**

- Show ≥ **3×** reduction in deep calls (target 5× if features allow) with failure recall drop ≤ **5%** (or honest miss + analysis)

### M4 — Bounded RSI

**Deliverables**

- RSI agent (strong coding/reasoning model via env, else heuristic proposer)
- Allowed mutation surface: prompts, thresholds, routing rules, ranking weights, feature flags
- Historical replay on development + public_eval
- Hidden eval only in harness
- Artifacts per iteration: hypothesis, diff, metrics, cost, promote/reject decision
- Human approval gate (CLI flag or Streamlit button); **no auto-promote**
- ≥3 iterations recorded under `reports/rsi/`

**Success function (configurable weights)**

```
score =
  0.40 * failure_recall
+ 0.20 * failure_precision
+ 0.15 * classification_f1
+ 0.15 * review_reduction
+ 0.10 * cost_reduction
```

Constraint: high-severity recall must not drop beyond a configured guardrail.

**Exit criteria**

- 3 iterations logged  
- At least one candidate with hidden-eval improvement **or** written postmortem why not

### M5 — Demo pack

**Deliverables**

- Single entry: `scripts/run_mvp_demo.py` (or documented make target)
- Dashboard shows: totals, success rate, failures, high-value set, clusters, diagnoses, actions, metrics, RSI history
- Final report answering the 10 product questions (see §12)
- Optional: customer learning-curve experiment (100→250→500…) if time remains

---

## 9. Success criteria (MVP bar)

| ID | Criterion | MVP bar |
|----|-----------|---------|
| A | Failure intelligence works | Detection F1 ≥ 0.70 on held-out **or** high-severity recall ≥ 0.80 with analysis if miss |
| B | Review reduction | ≤30% of episodes enter deep/human path while retaining ≥80% important failures |
| C | Failure compression | Failures → ≤50 clusters (or ≤10% of failure count) with semantic sanity check |
| D | Cascade value | Deep-call reduction ≥3× with recall drop ≤5% |
| E | RSI real improvement | ≥3 iterations; ≥1 hidden-eval improve **or** documented negative result |
| F | Portability | New dataset = new `DatasetAdapter` only; core pipeline untouched |

Numbers are slightly softened from the original brief so the first ship is achievable; we keep the original aggressive targets as **stretch** in reports.

---

## 10. Engineering principles

1. **Evaluator-first** — metrics harness lands in M1, not after the demo.  
2. **Data-first** — tiny fixtures always run; real RoboFAC is additive.  
3. **API-first** — providers behind interfaces; keys only via env vars.  
4. **Modular** — dataset specifics stay in adapters.  
5. **Measurable** — every milestone writes a report under `reports/`.  
6. **Reproducible** — fixed seeds, versioned prompts/configs, logged model/prompt versions.

### Env vars (documented in README)

```bash
OPENAI_API_KEY=          # optional
OPENAI_BASE_URL=         # optional OpenAI-compatible
VLM_PROVIDER=mock|openai|local
LLM_PROVIDER=mock|openai|local
JEV_API_KEY=             # optional; unused if unset
RSI_MODEL=               # optional
```

Missing keys → mock providers; never halt the pipeline.

---

## 11. Proposed repo layout

```
robot-deployment-intelligence/
  README.md
  pyproject.toml
  configs/
    default.yaml
    ranking_weights.yaml
    splits.yaml
  data/
    adapters/
    schemas/
    samples/           # tiny committed fixtures
    raw/               # gitignored
  features/
  decision/
    base.py
    baseline.py
    jev.py             # stub/mock + optional real client
  reasoning/
    base.py
    providers/
  failure_bank/
  clustering/
  ranking/
  evaluation/
  rsi/
    agent.py
    experiment.py
    replay.py
    promotion.py
  dashboard/
    app.py
  tests/
  scripts/
  reports/
  docs/
    mvp-plan.md        # this document
```

Python 3.11+, `pytest`, `pydantic`, `numpy`, `scikit-learn`, `streamlit`, `pyyaml`. Optional: `hdbscan`, HTTP client for providers.

---

## 12. Final report questions (answered at M5)

1. Does the pipeline meaningfully reduce human review?  
2. What is the best fast-decision backend we actually ran?  
3. If Jev is available, what advantage/disadvantage vs baseline? If not, what would be needed to test it?  
4. Detection/classification levels on public + hidden eval?  
5. Does clustering compress failures into few issues?  
6. How much cascade reduces API cost (or call count)?  
7. Did RSI improve the evaluator on hidden eval?  
8. Which improvements came from prompt / routing / features / model selection?  
9. Minimum data a real robotics partner must provide?  
10. Next step: optimize evaluator further, or find a design partner?

---

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| RoboFAC download/schema mismatch | Fixture-first; adapter isolation; document assumptions |
| No API keys | Mock providers + cost proxies from call counts |
| Labels incomplete | Nullable GT fields; metrics skip undefined labels |
| RSI overfits public eval | Hidden split sealed; promote gate |
| Scope creep (DROID, React, code-mutating RSI) | Explicit non-goals; PR checklist against this plan |
| Clustering without good embeddings | Fall back to TF-IDF on diagnoses/summaries |

---

## 14. Implementation order (do not skip)

1. Scaffold repo + CI-less `pytest` on fixtures  
2. M0 adapter + stats  
3. M1 baseline detector/classifier + harness  
4. M2 bank/cluster/rank + Streamlit  
5. M3 cascade comparison  
6. M4 RSI loop ×3  
7. M5 demo script + final report  

If a step is blocked, choose the simplest runnable substitute, log it in the milestone report, and continue.

---

## 15. Open assumptions (to confirm during M0)

1. RoboFAC provides enough labelled success/failure diversity in a ≤2k subset.  
2. Video may be unavailable for some episodes; pipeline must work on metadata-only features.  
3. “Severity” may be proxied (e.g., failure type mapping) if not labelled.  
4. Jev remains optional for the entire MVP unless credentials appear.  
5. English labels/prompts are fine for RoboFAC; localization is out of scope.

---

## 16. PR intent

This PR adds **documentation only**: the implementation plan above. No application code yet. Implementation should follow this plan starting at **M0**.
