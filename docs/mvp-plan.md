# Robot Deployment Intelligence MVP — Implementation Plan

**Status:** Draft (docs-only, revision 4)  
**Date:** 2026-09-26  
**Owner:** cedricxie  
**Source of truth for implementation.** Refined from the original research brief plus ChatGPT reviews: goals/validation (Approve with comments), then Career GT-workflow review (Conditional approve) folded into §5.5.

---

## 1. Project Goals

### 1.1 What problem are we trying to prove?

Robot companies collect large amounts of deployment data, but converting that data into actionable engineering improvements remains hard.

**Core hypothesis:**

> Deployment data itself contains enough signal to identify high-value failures, prioritize engineering attention, and recommend actionable next steps — even before we have access to the robot training pipeline.

This MVP validates whether an AI-native evaluation system can create a measurable feedback loop:

```text
Deployment Data → Failure Intelligence → Engineering Action → Feedback → Better Evaluation System
```

### 1.2 Who are we proving value for?

Primary users:

1. Robotics companies operating deployed robots  
2. Robotics data collection / training platforms  
3. Industrial partners with real-world deployment scenarios  

The MVP does **not** replace robot training systems. It targets the operational bottleneck **before** training:

- Which failures actually matter?  
- Which episodes deserve human investigation?  
- What are the recurring failure patterns?  
- What data or engineering action should happen next?  

### 1.3 What does “value” mean?

| Value | Meaning |
|-------|---------|
| **Attention efficiency** | Fewer episodes that engineers must manually inspect, while retaining important failures |
| **Failure discovery** | Surface high-severity, repeated, novel, or otherwise actionable failures |
| **Actionable recommendations** | Not only “what happened,” but “what to do next” — and whether an engineer would act |
| **Bounded evaluator improvement** | Human feedback → error analysis → experiments → replay → better evaluator (with approval gate) |

Success is **not** processing more data. Success is improving the **signal-to-attention ratio**.

### 1.4 One-sentence engineering goal

Build a modular pipeline that ingests ~500–2,000 RoboFAC episodes, compresses them into a small set of high-value failure cases/clusters, measures detection (separate from diagnosis), attention efficiency, and actionability, and runs ≥3 bounded evaluator-improvement experiments with a human promote gate.

### 1.5 What this MVP does NOT attempt to prove

| Non-goal | Clarification |
|----------|---------------|
| No robot training loop | Not proving Deployment → Collect → Train → Redeploy |
| No real-time robot control | Offline/batch analysis only; not safety monitoring or online intervention |
| No universal robot failure ontology | Taxonomy is task/dataset-specific and iteratively refined |
| No foundation model / VLA / RL training | May call existing APIs; do not train policies |
| No fully autonomous RSI | Bounded evaluator improvement only; production changes need human approval |
| No Jev hard dependency | Baseline fast decision required; Jev is optional/stub |
| No polished product UI | Streamlit demo at M5 only |
| No DROID / Guardian / FailCoT as ship blockers | Portability via adapter interface; one real adapter (RoboFAC) |

---

## 2. What “done” means (ship bar)

The MVP is successful if we can run a short script sequence that:

1. Loads a fixed RoboFAC subset into a unified `Episode` schema.  
2. Completes an **M0.5 thin slice** (100–200 episodes → failure ranking + report) early.  
3. Runs fast decision → optional deep reasoning → failure bank → clustering → ranking.  
4. Emits evaluation reports covering detection, attention efficiency, cascade cost, and (spot-checked) diagnosis/actionability.  
5. Completes ≥3 **evaluator improvement** iterations (config/prompt/threshold/routing/weights only), promote/reject on **hidden eval**, no auto-overwrite.  
6. Ships a minimal Streamlit demo at **M5** (not earlier).  

We are not aiming for paper SOTA. We are aiming for a reproducible product/research hypothesis test.

**Final acceptance question:**

> If this system were connected to a robotics engineering team, would they spend less time searching for problems and more time solving important ones?

Answer must be supported by quantitative metrics **and** human evaluation.

---

## 3. Scope decisions (vs original brief)

| Area | Original ask | MVP decision | Rationale |
|------|--------------|--------------|-----------|
| Dataset size | 1k–10k, then DROID | **500–2,000 RoboFAC** first; DROID deferred | Unblock iteration |
| Ground truth | Full diagnosis/correction | Map available labels; nullable fields; **detection ≠ diagnosis** | Diagnosis often weak/missing |
| Fast decision | Baseline + real Jev | **Baseline required**; Jev interface + mock | Do not block on Jev |
| Vision / LLM | Commercial + local | Abstract providers; **mock by default** | Runs without API keys |
| Deep reasoning | Strong VLM on hard cases | Only when routed; mockable | Prove cascade economics |
| Clustering | HDBSCAN or agglomerative | Embeddings + agglomerative first | One working path |
| Dashboard | Streamlit or React | **Streamlit at M5 only** | Avoid UI-driven mid-milestones |
| RSI | Broad agent / optional code | **Evaluator Improvement Loop**; config/prompt/threshold/routing/weights | Bounded, reviewable |
| Human feedback | Simulated + UI | GT-simulated first; UI corrections at M5 | Real reviewers later |
| Learning curve | 100→2000 ramp | Optional stretch after M4 | Not ship blocker |

### Freeze before coding M0

Must freeze: Project Goals, success definition, dataset choice, Episode schema, validation protocol, hidden-split policy, non-goals, **and §5.5 P0 GT freezes (label lifecycle + split/hidden isolation, important-failure definition, severity proxy, spot-check/dispute/confirmer, synthetic/weak-label use)**.

May defer: which VLM/LLM, Jev availability, dashboard stack details, RSI agent framework, DB choice, cloud deploy.

---

## 4. Core questions the MVP must answer

| ID | Question | How we measure |
|----|----------|----------------|
| Q1 | Can we reduce human review while keeping important failures? | Attention efficiency; review reduction ≥70% with ≥80% important-failure retention |
| Q2 | Is failure **detection** usable vs GT? | Precision / recall / F1; high-severity recall (separate from diagnosis) |
| Q3 | Can failures compress into coherent engineering issues? | Cluster usefulness spot-check ≥80% coherent |
| Q4 | Does cheap→fast→deep cascade save cost? | Deep calls / cost proxy; recall drop ≤5% |
| Q5 | Would engineers act on outputs? | Majority “would act” on sampled recommendations |
| Q6 | Does bounded improvement help on hidden eval? | ≥3 experiments; ≥1 hidden-eval improve **or** documented negative result |

---

## 5. Validation & Acceptance Protocol

### 5.1 Philosophy

Model accuracy alone is insufficient. Evaluation must answer:

1. Can we reduce human review effort?  
2. Can we find important failures?  
3. Can we provide useful diagnosis (spot-checked)?  
4. Would engineers act on recommendations?  
5. Can the evaluator improve over iterations without overfitting?  

### 5.2 Dataset splits (fixed seed, default `42`)

| Split | Ratio | Who may see labels |
|-------|-------|--------------------|
| development | 60% | pipeline + improvement loop |
| public_eval | 20% | harness + milestone reports + candidate selection |
| hidden_eval | 20% | **evaluation harness only** |

The improvement loop **must not** load hidden labels. Promote only if hidden eval improves (or human consciously accepts a tradeoff documented in the experiment log).

### 5.3 Evaluation dimensions

#### Failure detection (objective)

- Precision, recall, F1  
- High-severity failure recall (when severity proxy exists)  
- Detection is **separate** from diagnosis  

#### Failure diagnosis (initially weaker)

- Category accuracy where labels exist  
- Root-cause agreement / evidence quality via **expert spot check**  
- MVP does **not** require perfect automatic diagnosis scoring  

#### Attention efficiency (primary product metric)

```text
Attention Efficiency ≈ Important Failures Found / Human Review Effort
```

Example target shape: 10k episodes → ~300 prioritized reviews while retaining ~80–90% of important failures.

#### Clustering usefulness

- Cluster count and size distribution (informative, not the goal)  
- Spot check: ≥80% of sampled clusters = coherent engineering issue (what / why it matters / what to investigate)  

#### Actionability

For sampled recommendations, humans answer:

1. Is the diagnosis understandable?  
2. Is the recommended action reasonable?  
3. Would you investigate / change / collect data based on this?  

Pass when a **majority** of sampled items get “yes, I would act.”

#### Cascade economics

Compare all-deep vs fast→selected-deep: deep-call / cost reduction vs important-failure recall delta.

#### Evaluator improvement

Logged experiments under `experiments/`; ≥3 iterations; hidden-eval comparison; human promote gate.

### 5.4 Milestone acceptance gates

| Milestone | Pass if… | Fail if… |
|-----------|----------|----------|
| **M0** Dataset pipeline | RoboFAC → Episode; stats report; fixtures in CI | Dataset-specific fields leak into core pipeline |
| **M0.5** Thin slice | 100–200 episodes → ranking + structured report + basic metrics in 1–2 days of work | Cannot produce useful end-to-end insight before larger infra |
| **M1** Failure intelligence | Detection metrics + attention efficiency reported | No measurable detection or prioritization |
| **M2** Failure bank + clustering | Cases stored; clusters + usefulness spot-check notes | No compression or incoherent clusters only |
| **M3** Cascade | ≥3× (stretch 5×) fewer deep calls; recall drop ≤5% | Cascade saves nothing or tanks recall |
| **M4** Evaluator improvement loop | ≥3 experiments; ≥1 hidden improve **or** written postmortem; no auto-promote | Only overfits public/dev |
| **M5** Demo | Full demo outputs + metrics + human “would act” sample | Pretty UI without evidence |


### 5.5 Ground-truth gaps, weak labels, and human-adjudicated reference labels

RoboFAC gives a strong **detection / diagnosis / correction** substrate, but several MVP acceptance metrics need labels the raw dataset does not ship as first-class fields. This section freezes how we fill those gaps **without** treating LLM-vs-LLM agreement as final engineer-actionability GT.

**Framing (ChatGPT Career, conditional approve):** prefer the term **human-adjudicated reference / benchmark labels** for acceptance GT. The machine field may remain `label_status: human_confirmed` for schema continuity; in docs and reports it means *human-adjudicated reference/benchmark label*, not “a human casually agreed with an LLM.”

**Doc principles carried from review:**
- Keep **detection** metrics separate from **diagnosis** (already §5.3 / Q2).
- **Important failure** is independent of failure-type taxonomy (do not require a taxonomy label to mark importance).
- **Cluster usefulness** and **actionability** require human adjudication for acceptance — never LLM-only.

#### What RoboFAC already provides vs what we must add

| Need for MVP gates | RoboFAC typically provides | Must we add? | Notes |
|--------------------|----------------------------|--------------|-------|
| Failure **detection** (`success` / fail) | Yes — episode `success` + failure trajectories; QA includes failure detection | No (map into `ground_truth.success`) | Primary objective metric; separate from diagnosis |
| Failure **type** / identification | Partially — paper taxonomy + QA “failure identification”; not always a single structured enum | **Optional enum** — map when present; otherwise nullable / weak-label | Important ≠ taxonomy; do not block M0 on perfect taxonomy |
| Diagnosis / explanation / correction text | Yes — via QA dimensions | Map when usable; nullable otherwise | Spot-check quality; not automatic scoring blocker |
| **Important / high-value failure** (attention-efficiency numerator) | No first-class field | **Yes — weak label + human adjudication** | Required for Q1 / criterion B; independent of taxonomy |
| **Attention must-keep** (should this episode stay in the review budget?) | No | **Yes — weak label + human adjudication** | Overlaps with “important” but may include rare/novel/safety-adjacent cases |
| **Cluster coherence** (cluster = one engineering issue?) | No | **Yes — human adjudication** (rules/LLM may draft notes) | Gate for M2 / criterion C |
| **Actionability** (“would an engineer act?”) | No | **Yes — human adjudication only** | Criterion E; LLM draft is advisory |
| Severity (high-severity recall) | No dedicated severity score | Dataset/human when available; else heuristic proxy | Document proxy; do not invent precision |

Schema extension (additive; keep existing nullable GT fields):

```yaml
ground_truth:
  # …existing fields…
  important_failure: bool | null
  attention_must_keep: bool | null
  cluster_coherence: bool | null          # usually filled at cluster-sample level
  actionability_would_act: bool | null    # human only for acceptance
  failure_type: str | null                # optional enum; map RoboFAC when present
  severity: int | null                    # 3=high, 2=mid, 1=low when proxied
  label_status: draft | chatgpt_reviewed | human_confirmed
  # human_confirmed == human-adjudicated reference/benchmark label (naming kept for schema)
  label_source: rules | llm | hybrid | human
  label_notes: str | null
  label_disputed: bool | null
```

#### Weak-label / synthetic-label generation

1. **Rules first** where deterministic signals exist (e.g. `success=false` → candidate failure; type map from RoboFAC identification answers; severity proxy when dataset lacks severity).  
2. **LLM draft** for product-facing fields RoboFAC lacks: `important_failure`, `attention_must_keep`, draft actionability rationale, optional cluster-coherence notes.  
3. Every generated label starts as `label_status: draft` with `label_source: rules | llm | hybrid`.  
4. **P0 use constraint:** synthetic / weak labels are for **development augmentation only** (ranking experiments, UI prototypes, harness dry-runs). They **must not** count toward hidden-eval acceptance or milestone gates until promoted to human-adjudicated reference labels.

#### Workflow: generate → ChatGPT review → human adjudicate → promote

```text
Assistant generates candidate GT (rules + LLM)
        ↓
ChatGPT Career / review pass
  - consistency vs RoboFAC success/type
  - flag disputes / overconfident importance
  - advisory only (does not auto-promote)
        ↓  label_status → chatgpt_reviewed
Human adjudication (non-hidden splits only — see isolation)
  - all disputed items
  - random sample of non-disputed
  - actionability + cluster usefulness samples always human
        ↓  only then
label_status → human_confirmed
  (= human-adjudicated reference/benchmark label)
        ↓
Acceptance gates (Q1–Q5, criteria A–E) may use these labels as GT
```

**Promotion rule:** only `human_confirmed` rows (human-adjudicated reference/benchmark labels) count as ground truth for milestone acceptance and for attention / importance / actionability / cluster usefulness scoring. `draft` and `chatgpt_reviewed` stay in the label store for audit and iteration.

**Split / hidden isolation (P0):** freeze splits **first** (§5.2). Human adjudication runs on **development** and **public_eval** only. Do **not** human-review or promote weak labels on `hidden_eval` episodes for acceptance — hidden isolation stays sealed; the harness alone may score against whatever sealed reference labels are provisioned without leaking them into the improvement loop.

#### Explicit limits

- **LLM-vs-LLM agreement is not engineer-actionability final GT.** ChatGPT (or any other model) reviewing another model’s draft cannot close criterion E.  
- ChatGPT review is for **consistency, dispute surfacing, and checklist completeness** — not a substitute for human “would act” judgment.  
- Detection metrics may use RoboFAC `success` directly; product metrics that RoboFAC does not define must follow the promotion rule above.  
- Cluster usefulness and actionability **require** human adjudication for acceptance.

#### GT-specific risks (from ChatGPT Career review)

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Synthetic / weak-label **distribution shift** | Dev metrics look good; acceptance fails on human-adjudicated labels | Weak/synthetic for development only; gates use human-adjudicated labels |
| **LLM-review bias** | ChatGPT rubber-stamps another model’s drafts | Advisory only; never auto-promote; disputes + spot-checks |
| **Hidden-eval contamination** | Human/LLM labeling on hidden leaks into training the evaluator | Split first; human adjudication only on non-hidden; improvement loop must not load hidden labels |
| **Early taxonomy freeze** | Wrong/partial `failure_type` enum blocks importance or diagnosis work | Keep taxonomy optional/`other`/`unknown`; important independent of taxonomy |

#### P0 freezes (resolved defaults — ChatGPT Career conditional approve)

Resolve before coding M0/M1 acceptance harnesses that depend on them. Defaults below are **frozen for MVP** unless an ADR overrides:

| # | Decision | Frozen default |
|---|----------|----------------|
| 1 | **Label lifecycle + split/hidden isolation** | Lifecycle: `draft` → `chatgpt_reviewed` → `human_confirmed` (human-adjudicated). Freeze splits first (§5.2 seed `42`, 60/20/20). Human review / adjudication **only on non-hidden**. Synthetic/weak labels **development augmentation only** — not for hidden acceptance. |
| 2 | **Important failure** definition | Mark important if **ANY** of: high severity, high frequency (repeated pattern), high engineering cost, novelty, or actionable. Independent of taxonomy. |
| 3 | **Severity proxy** | Prefer dataset field or human label when present; else heuristic **3 / 2 / 1** (high / mid / low) from type→impact table; document mapping in reports. |
| 4 | **Spot-check size + dispute process** | ~**20 clusters × 5 episodes (~100)** for cluster/importance coherence sampling; plus **20–30 actionability** items. All `label_disputed=true` items reviewed. Dispute → human adjudication (or second reviewer) before promote. |
| 5 | **Who confirms (adjudicator)** | Prefer a **robotics expert**; else **two independent reviewers** with disagreement escalation recorded in `label_notes`. |
| 6 | **`failure_type` taxonomy** | Prefer RoboFAC six-class map; extend only with `other` / `unknown`; do not freeze a universal ontology; do not couple importance to taxonomy. |
| 7 | Is ChatGPT review **advisory only**? | **Yes** — advisory; never auto-`human_confirmed`. |
| 8 | Split policy | **Already frozen (§5.2):** seed `42`; 60/20/20; improvement loop must not load hidden labels. |

Items 1–5 are the **P0 freeze set** from the Career review (lifecycle/isolation, important definition, severity proxy, spot-check/dispute, confirmer). Items 6–8 remain aligned with prior plan freezes.


---

## 6. Architecture (runnable slice)

```text
RoboFAC subset
    → RoboFACAdapter
    → Episode (unified schema)
    → Feature / Summary Extraction
    → FastDecisionEngine (Baseline; Jev stub)
    → route: pass | failure / uncertain / high-value
    → DeepReasoner (selected only)
    → FailureBank
    → Clustering + Ranking
    → EvaluationHarness + Report
    → (M5) Streamlit Dashboard + human correction UI
    → HumanFeedbackStore
    → Evaluator Improvement Loop
         → experiments/ → replay → public eval → hidden eval → promote gate
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

Adapters must not leak RoboFAC-only field names into decision / reasoning / clustering code.

### 6.2 Required interfaces

- `DatasetAdapter.load() -> Iterable[Episode]`
- `FeatureExtractor.extract(episode) -> EpisodeFeatures`
- `FastDecisionEngine.evaluate(features, decision_schema) -> DecisionResult`
- `VisionReasoningProvider.summarize(frames, prompt) -> str`
- `DeepReasoner.reason(context) -> StructuredDiagnosis` (+ raw / model / prompt versions)
- `FailureBank.upsert(FailureCase)`
- `Clusterer.fit_predict(cases) -> Clusters`
- `Ranker.score(case_or_cluster, weights) -> float`
- `EvaluationHarness.evaluate(...) -> Metrics`
- `ImprovementLoop.propose_and_eval(production_config) -> CandidateResult` (human gate)

---

## 7. Dataset plan (M0)

### 7.1 Primary: RoboFAC

- [MINT-SJTU/RoboFAC](https://github.com/MINT-SJTU/RoboFAC), [HF dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)  
- First download: smallest usable subset with success + multiple failure types (**500–2,000** episodes)  
- `data/raw/robofac/` gitignored; tiny `data/samples/` committed for CI  

### 7.2 Fallback

1. Public metadata / video subset from the same project if full download fails  
2. Else synthetic fixture matching Episode schema (clearly marked) so tests still run  
3. Log assumptions in the milestone report; do not block the repo  

DROID is post-MVP validation.

---

## 8. Milestone plan

### M0 — Dataset pipeline

**Deliverables:** repo layout, `Episode` schema, `RoboFACAdapter`, fixture tests, `scripts/m0_dataset_stats.py` → `reports/m0_dataset_stats.md`  

**Exit:** `pytest` green without API keys; no dataset leakage into core modules  

### M0.5 — End-to-end thin slice (NEW)

**Goal:** Prove dataset → insight before more infrastructure.

**Pipeline:** Episode → features → baseline fast decision → failure ranking → markdown/HTML report  

**Input:** 100–200 episodes  

**Deliverables:** `reports/m0_5_thin_slice.md` with ranking, examples, basic detection metrics  

**Exit:** A human can skim the report and say whether the direction looks useful — within ~1–2 days of work after M0  

### M1 — Failure intelligence

**Deliverables:** deterministic features; `BaselineFastDecisionEngine`; `JevDecisionEngine` stub; detection metrics + attention efficiency; `reports/m1_failure_intelligence.md`  

**Exit:** Detection F1 ≥ 0.70 **or** high-severity recall ≥ 0.80 on public_eval, **or** documented failure modes if missed; attention efficiency reported  

### M2 — Failure bank + clustering (no dashboard)

**Deliverables:** `FailureCase` store (JSONL/SQLite); embedding + agglomerative clustering; configurable ranking weights; `failure_bank` artifact + `reports/m2_clusters.md` with usefulness spot-check notes  

**Exit:** Failures compressed; ≥80% of sampled clusters coherent; **no Streamlit required here**  

### M3 — Fast / deep cascade

**Deliverables:** Deep reasoner behind providers; routing policy; A (all-deep) vs B (cascade) comparison; `reports/m3_cascade.md`  

**Exit:** ≥3× fewer deep calls (stretch 5×); important-failure recall drop ≤5%  

### M4 — Evaluator improvement loop (formerly “RSI Agent”)

**Deliverables:** experiment generator + replay harness + comparator; allowed mutations: prompts, thresholds, routing, ranking weights, feature flags; artifacts under `experiments/` and `reports/rsi/`; human approve/reject only  

**Success function (configurable):**

```text
score =
  0.40 * failure_recall
+ 0.20 * failure_precision
+ 0.15 * classification_f1
+ 0.15 * review_reduction
+ 0.10 * cost_reduction
```

Guardrail: high-severity recall must not drop beyond configured limit.

**Exit:** ≥3 logged experiments; ≥1 hidden-eval improvement **or** postmortem  

### M5 — Final MVP demo (+ Streamlit)

**Deliverables:** `scripts/run_mvp_demo.py`; Streamlit for overview / clusters / case detail / GT vs pred / correction; final report answering §12  

**Demo outputs:** deployment summary, failure ranking, clusters, representative cases, AI diagnosis, recommended actions, metrics, attention efficiency, improvement history, actionability sample  

---

## 9. MVP success criteria

| ID | Criterion | Target |
|----|-----------|--------|
| A | Failure detection | F1 ≥ 0.70 **or** high-severity recall ≥ 0.80 |
| B | Attention efficiency | ≥70% review volume reduction while retaining ≥80% important failures |
| C | Failure compression | ≥80% of spot-checked clusters useful/coherent |
| D | Cascade efficiency | ≥5× cost/call reduction stretch (≥3× minimum) with ≤5% important-recall drop |
| E | Actionability | Majority of sampled recommendations: “would act” |
| F | Evaluator improvement | ≥3 experiments; ≥1 hidden-eval improve **or** documented negative result |
| G | Portability | New dataset = new `DatasetAdapter` only |

---

## 10. Engineering principles

1. **Goals-first** — freeze success definition before features  
2. **Evaluator-first** — harness lands by M1; thin slice by M0.5  
3. **Data-first** — fixtures always run; RoboFAC is additive  
4. **API-first** — providers behind interfaces; keys via env only  
5. **Modular** — dataset specifics stay in adapters; taxonomy in `ontology/`  
6. **Measurable & reproducible** — fixed seeds; versioned prompts/configs; logged experiments  

### Env vars

```bash
OPENAI_API_KEY=          # optional
OPENAI_BASE_URL=         # optional
VLM_PROVIDER=mock|openai|local
LLM_PROVIDER=mock|openai|local
JEV_API_KEY=             # optional
IMPROVEMENT_MODEL=       # optional
```

Missing keys → mock providers; never halt the pipeline.

---

## 11. Proposed repo layout

```text
robot-deployment-intelligence/
  README.md
  pyproject.toml
  configs/
  data/
    adapters/
    schemas/
    samples/
    raw/                 # gitignored
  ontology/              # task/dataset failure taxonomy mappings
  features/
  decision/
    base.py
    baseline.py
    jev.py
  reasoning/
    base.py
    providers/
  failure_bank/
  clustering/
  ranking/
  evaluation/
  experiments/          # exp001_*.yaml + results
  rsi/                   # improvement loop (name kept for continuity)
    agent.py
    experiment.py
    replay.py
    promotion.py
  dashboard/             # Streamlit — used at M5
  tests/
  scripts/
  reports/
  docs/
    mvp-plan.md
```

Python 3.11+, pytest, pydantic, numpy, scikit-learn, streamlit, pyyaml.

---

## 12. Final report questions (M5)

1. Does the pipeline meaningfully reduce human review (attention efficiency)?  
2. Best fast-decision backend we actually ran?  
3. If Jev available: advantage/disadvantage vs baseline; if not: what is needed to test it?  
4. Detection levels on public + hidden? Diagnosis only via spot check if labels weak?  
5. Does clustering produce coherent engineering issues?  
6. Cascade cost/call reduction?  
7. Did the improvement loop help on hidden eval?  
8. Which wins came from prompt / routing / features / weights?  
9. Minimum data a real robotics partner must provide?  
10. Next: optimize evaluator further, or find a design partner?  
11. Would sampled engineers act on the recommendations?  

---

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| RoboFAC download/schema mismatch | Fixture-first; adapter isolation |
| No API keys | Mock providers; call-count cost proxies |
| Incomplete labels / product GT gaps | Nullable GT; §5.5 weak labels + ChatGPT review + human-adjudicated (`human_confirmed`) only for gates; spot-check diagnosis |
| Weak/synthetic label distribution shift | Dev-only weak labels; acceptance on human-adjudicated reference labels |
| LLM-review bias | ChatGPT advisory only; never auto-promote; dispute + spot-check |
| Hidden-eval label contamination | Split first; human adjudication on non-hidden only |
| Early taxonomy freeze | Taxonomy optional; important independent of failure_type |
| Diagnosis harder than detection | Separate metrics; do not block MVP on diagnosis accuracy |
| Improvement loop overfits | Hidden split sealed; promote gate |
| Scope creep | Non-goals + freeze list; thin slice before M2–M4 |
| Cluster count gaming | Usefulness spot-check, not cluster count alone |
| UI distraction | Dashboard only at M5 |

---

## 14. Implementation order

1. Scaffold + fixture tests  
2. M0 adapter + stats  
3. **M0.5 thin slice report**  
4. M1 detection + attention efficiency  
5. M2 bank/cluster (reports only)  
6. M3 cascade comparison  
7. M4 improvement loop ×3  
8. M5 demo + Streamlit + final report  

If blocked, choose the simplest runnable substitute, log it, continue.

---

## 15. Open assumptions

1. RoboFAC ≤2k subset has enough success/failure diversity.  
2. Pipeline works metadata-only when video missing.  
3. Severity may be proxied from failure type if unlabelled.  
4. Jev optional for entire MVP unless credentials appear.  
5. English labels/prompts OK for RoboFAC.  

---

## 16. Revision history

| Rev | Change |
|-----|--------|
| 1 | Initial scoped plan (docs-only PR) |
| 2 | Added Project Goals + Validation protocol; M0.5 thin slice; M4 renamed to Evaluator Improvement Loop; Streamlit deferred to M5; success criteria emphasize attention efficiency, diagnosis separation, actionability; added `experiments/` + `ontology/`; expanded non-goals (no realtime, no universal ontology) |
| 3 | Added §5.5 Ground-truth gaps / weak labels / ChatGPT review + human confirm workflow; promotion rule (`human_confirmed` only for acceptance gates); pre-MVP open-questions checklist; freeze list + incomplete-labels risk updated |
| 4 | ChatGPT Career GT-workflow review (conditional approve): rename framing to human-adjudicated reference/benchmark labels (keep `human_confirmed` field); P0 freezes + defaults (lifecycle/isolation, important=ANY severity/freq/cost/novelty/actionable, severity 3/2/1 heuristic, spot-check ~20×5 + 20–30 actionability, confirmer=robotics expert else two reviewers); weak/synthetic for dev only; risks (distribution shift, LLM-review bias, hidden contamination, early taxonomy freeze); detection≠diagnosis, important⊥taxonomy, cluster/actionability need human adjudication |

---

## 17. PR intent

Docs-only plan freeze before M0 coding. Implementation follows this document starting at **M0**, then **M0.5** before larger infrastructure.
