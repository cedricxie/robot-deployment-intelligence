# Robot Deployment Intelligence MVP — Implementation Plan

**Status:** Draft (docs-only, revision 6.1)
**Date:** 2026-09-26  
**Owner:** cedricxie  
**Source of truth for implementation.** Refined from the original research brief plus ChatGPT reviews; **rev 5 pivot** (automatable core, rule-based proxy importance, Baseline↔JEV); **rev 6** folds Career rev5 clarifications (terminology, baseline ladder, cluster-purity honesty, diagnosis metrics, hidden isolation) and adds a sequenced **implementation-PR plan**. **Rev 6.1** adds the standing ChatGPT Career review/merge gate for every implementation PR.

---

## 1. Project Goals

### 1.1 What problem are we trying to prove?

Robot companies collect large amounts of deployment data, but converting that data into actionable engineering improvements remains hard.

**Core hypothesis:**

> Deployment data itself contains enough signal to identify **proxy-important / review-priority** failures, prioritize engineering attention, and recommend actionable next steps — even before we have access to the robot training pipeline.

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

- Which episodes failed, and with what type?  
- Which failures are common vs rare / novel?  
- Which episodes / clusters deserve prioritized review (via **proxy** importance)?  
- What diagnosis draft and next-step sketch can we propose (weak output)?  

**Honesty note:** MVP “importance” is a **rule-based proxy** for **review priority**, not business criticality or confirmed model-iteration impact. **Review priority ≠ business importance.** Prefer terms **`proxy-important`** / **`review-priority`** over casual “high-value.” The user cannot yet judge downstream criticality; we do **not** require human importance GT.

### 1.3 What does “value” mean?

| Value | Meaning |
|-------|---------|
| **Automatable failure intelligence** | Success/fail, failure type, common vs rare / novelty — measurable without human importance labels |
| **Attention efficiency (proxy / review-priority)** | Fewer episodes to inspect while retaining **proxy-important** (review-priority) failures (rules A∧(B∨C∨D)); **not** business importance |
| **Compression & ranking** | Clustering / dedup + Top-K ranking over proxy-important cases |
| **Cascade economics** | Cheap→fast→deep routing with measurable cost vs retention tradeoff |
| **Comparable decision paths** | Ladder: Random → Frequency-only → BaselineFastDecisionEngine → JEV (stub→real) on the **same** splits/metrics; JEV measures cost–quality tradeoff, not assumed superiority |
| **Bounded evaluator improvement** | Config/prompt/threshold/routing/weights only; human promote gate |
| **Diagnosis draft (weak)** | Limited validation only; **not** a hard ship gate |
| **Actionability / true criticality** | **Post-MVP** — not required for MVP pass |

Success is **not** processing more data. Success is proving an **automatable** signal-to-attention loop with honest proxy metrics.

### 1.4 One-sentence engineering goal

Build a modular pipeline that ingests ~500–2,000 RoboFAC episodes, proves automatable core labels (success/fail, failure type, common/rare/novelty), compresses via clustering/dedup + Top-K **review-priority** ranking, runs cascade routing, emits auto metrics reports, and **compares the baseline ladder vs JEV** on the same harness — using **rule-based proxy importance / review-priority** only (no business-critical human GT).

### 1.5 What this MVP does NOT attempt to prove

| Non-goal | Clarification |
|----------|---------------|
| No robot training loop | Not proving Deployment → Collect → Train → Redeploy |
| No real-time robot control | Offline/batch analysis only; not safety monitoring or online intervention |
| No universal robot failure ontology | Taxonomy is task/dataset-specific and iteratively refined |
| No foundation model / VLA / RL training | May call existing APIs; do not train policies |
| No fully autonomous RSI | Bounded evaluator improvement only; production changes need human approval |
| No JEV hard dependency for shipping | Baseline fast decision required; JEV optional for ship — **but Baseline↔JEV comparison experiment is in-scope** |
| No polished product UI | Streamlit demo at M5 only |
| No DROID / Guardian / FailCoT as ship blockers | Portability via adapter interface; one real adapter (RoboFAC) |
| **No business-critical / downstream-impact GT** | User cannot judge criticality or model-iteration impact yet; **no human importance GT** |
| **No actionability / true criticality as MVP gate** | Post-MVP; diagnosis draft is weak output only |
| **No severity referee / human importance adjudication** | Human role shrunk to optional cluster sanity spot-check |

---

## 2. What “done” means (ship bar)

The MVP is successful if we can run a short script sequence that:

1. Loads a fixed RoboFAC subset into a unified `Episode` schema.  
2. Completes an **M0.5 thin slice** (100–200 episodes → failure ranking + report) early.  
3. Runs the **baseline ladder + JEV** (Random → Frequency-only → BaselineFastDecisionEngine → JEV stub/real) → optional deep reasoning → failure bank → clustering → Top-K **review-priority** ranking.  
4. Emits **auto** evaluation reports: detection P/R/F1, failure-type agreement, proxy-important / review-priority retention, review reduction, cascade cost (if applicable) — **side-by-side ladder columns**.  
5. Completes ≥3 **evaluator improvement** iterations (config/prompt/threshold/routing/weights only), promote/reject using **hidden eval for harness final scoring only** (never for tuning), no auto-overwrite.  
6. Ships a minimal Streamlit demo at **M5** (not earlier).  

We are not aiming for paper SOTA. We are aiming for a reproducible product/research hypothesis test of the **automatable core**.

**Final acceptance question:**

> If this system were connected to a robotics engineering team, would the automatable pipeline (detect → type → novelty/frequency → cluster → Top-K) reduce search time for **review-priority** failures **under our published proxy rules** (not claiming business importance) — and do we know how Random / Frequency-only / Baseline / JEV compare on the same metrics (JEV as cost–quality experiment)?

Answer must be supported by **quantitative auto metrics** (and optional cluster spot-check notes). Human “would act” / true criticality are **out of MVP scope**.

---

## 3. Scope decisions (vs original brief)

| Area | Original ask | MVP decision | Rationale |
|------|--------------|--------------|-----------|
| Dataset size | 1k–10k, then DROID | **500–2,000 RoboFAC** first; DROID deferred | Unblock iteration |
| Ground truth | Full diagnosis/correction + human importance | Map RoboFAC success + type; **rule-based proxy importance only**; diagnosis nullable/weak | User cannot judge business criticality |
| Fast decision | Baseline + real Jev | **Ladder required:** Random → Frequency-only → Baseline → JEV; Baseline = default ship; JEV = cost–quality experiment (stub OK) — **same eval harness** | Comparison in-scope; JEV not ship-blocker; not assumed superior |
| Vision / LLM | Commercial + local | Abstract providers; **mock by default** | Runs without API keys |
| Deep reasoning | Strong VLM on hard cases | Only when routed; mockable | Prove cascade economics |
| Clustering | HDBSCAN or agglomerative | Embeddings + agglomerative first; **auto purity vs `failure_type` = label consistency only** | Human cluster spot-check optional; purity ≠ eng usefulness |
| Dashboard | Streamlit or React | **Streamlit at M5 only** | Avoid UI-driven mid-milestones |
| RSI | Broad agent / optional code | **Evaluator Improvement Loop**; config/prompt/threshold/routing/weights | Bounded, reviewable |
| Human feedback | Simulated + UI + importance GT | **Optional cluster sanity only**; no severity referee; ChatGPT path = taxonomy drafts advisory | Shrink human role |
| Learning curve | 100→2000 ramp | Optional stretch after M4 | Not ship blocker |
| Actionability | “Would act” majority gate | **Post-MVP** | Not automatable / not judgment-ready |

### Freeze before coding M0

Must freeze: Project Goals, success definition, dataset choice, Episode schema, validation protocol, **hidden-split policy (harness final scoring only)**, non-goals, **automatable core definition**, **proxy-important / review-priority rule defaults (A–F / acceptance set)**, **baseline ladder + JEV comparison harness contract**, diagnosis metric priority (type agreement primary), cluster-purity honesty, and §5.5 label policy (proxy-only review-priority; ChatGPT advisory for taxonomy drafts only).

May defer: which VLM/LLM, JEV real credentials, dashboard stack details, RSI agent framework, DB choice, cloud deploy, actionability / true-criticality labeling.

---

## 4. Core questions the MVP must answer

| ID | Question | How we measure |
|----|----------|----------------|
| Q1 | Can we reduce review volume while retaining **proxy-important / review-priority** failures? | Review reduction ≥70% with ≥80% **proxy-important** retention (rules A∧(B∨C∨D)); review priority ≠ business importance |
| Q2 | Is failure **detection** usable vs RoboFAC success GT? | Precision / recall / F1 (separate from diagnosis) |
| Q2b | Is **failure type** usable vs RoboFAC labels/QA mapping? | Type agreement / accuracy where mapped labels exist |
| Q2c | Can we separate **common vs rare / novelty**? | Type frequency stats + Failure Bank novelty score; reported in auto metrics |
| Q3 | Do clusters show **label-consistent** compression? | **Auto cluster purity vs `failure_type`** = label consistency only (not eng usefulness); human spot-check **optional**, not required for pass |
| Q4 | Does cheap→fast→deep cascade save cost? | Deep calls / cost proxy; proxy-importance recall drop ≤5% |
| Q5 | How do **Random / Frequency-only / Baseline / JEV** compare on the same splits? | Side-by-side ladder: detection P/R/F1, proxy-important retention, review reduction, cascade cost (if applicable), failure-type agreement; JEV = cost–quality hypothesis |
| Q6 | Does bounded improvement help on hidden eval? | ≥3 experiments; ≥1 hidden-eval improve **or** documented negative result; hidden labels = harness final scoring only (never for tuning) |
| — | Diagnosis draft quality? | **Weak output** — **type agreement primary**; text similarity de-emphasized (informative only); optional spot-check; **not a hard gate** |
| — | Actionability / true criticality? | **Post-MVP** |

---

## 5. Validation & Acceptance Protocol

### 5.1 Philosophy

Model accuracy alone is insufficient. Evaluation must answer:

1. Can we **automatically** detect success vs failure?  
2. Can we map **failure type** from RoboFAC labels/QA?  
3. Can we score **common vs rare / novelty** (frequency + Failure Bank)?  
4. Can we reduce review volume while retaining **proxy-important / review-priority** failures (honest: review priority ≠ business importance)?  
5. Do clustering/dedup + Top-K + cascade produce measurable **label-consistent** compression and cost savings?  
6. How do **Random / Frequency-only / Baseline / JEV** compare on the same harness (JEV as cost–quality experiment)?  
7. Can the evaluator improve over iterations without overfitting (hidden = final scoring only)?  

We explicitly **do not** gate MVP on human importance judgment, severity referee, or “would act” actionability.

### 5.2 Dataset splits (fixed seed, default `42`)

| Split | Ratio | Who may see labels |
|-------|-------|--------------------|
| development | 60% | pipeline + improvement loop (tuning allowed) |
| public_eval | 20% | harness + milestone reports + candidate selection |
| hidden_eval | 20% | **evaluation harness final scoring only** |

**Hidden-label isolation (hard rule):**

- Hidden labels may be read **only** by the evaluation harness for **final scoring** of a frozen candidate.  
- **Never** use hidden labels for prompt tuning, threshold search, RSI / improvement-loop mutations, routing policy search, ranking-weight search, or any other candidate design.  
- The improvement loop **must not** load hidden labels when proposing or selecting candidates.  
- Promote only if hidden-eval score improves (or human consciously accepts a tradeoff documented in the experiment log).

RoboFAC-provided fields (`success`, mapped `failure_type`) are available to the harness on all splits. **Proxy importance / review-priority** is computed deterministically from rules + bank state — no human promotion step.

### 5.3 Evaluation dimensions

#### Automatable core (MVP must prove)

| Signal | Source | Metric |
|--------|--------|--------|
| Success vs failure | RoboFAC `success` / failure trajectories | Detection P / R / F1 |
| Failure type | RoboFAC labels / QA identification mapping | Type agreement / accuracy (nullable when unmapped) |
| Common vs rare | Empirical type frequency on the split / bank | Frequency rank; rare-type flag |
| Novelty | Distance / non-match vs Failure Bank | Novelty score; novel flag |

Plus pipeline capabilities that must ship with auto reports:

- Clustering / dedup  
- Top-K ranking (proxy-importance + ranking bonuses E/F)  
- Cascade routing  
- Auto metrics reports (including Baseline↔JEV side-by-side)

#### Failure detection (objective)

- Precision, recall, F1  
- Detection is **separate** from diagnosis  
- High-severity recall is **replaced** by **proxy-importance recall** for MVP gates (see §5.3 proxy rules)

#### Failure diagnosis (weak output — not a hard gate)

- **Primary:** failure-type agreement where mapped labels exist  
- **De-emphasized:** free-text similarity vs RoboFAC diagnosis/correction (informative only; not a gate signal)  
- Optional human spot-check (advisory)  
- MVP does **not** require diagnosis quality as a pass/fail criterion  

#### Proxy importance / review-priority & attention efficiency (primary product metric for MVP)

**Honesty:** `proxy-important` / `review-priority` ≠ business importance ≠ business criticality ≠ confirmed model-iteration impact. Prefer these terms over “high-value.”

```text
Attention Efficiency ≈ Proxy-Important Failures Found / Human Review Effort
```

Example target shape: large episode set → small Top-K review set while retaining ≥80% of **proxy-important** failures under frozen rules.

#### Clustering (auto-primary)

- **Primary:** auto cluster purity / homogeneity vs `failure_type` — measures **label consistency only**, **not** engineering usefulness  
- Cluster count and size distribution (informative)  
- Human cluster sanity spot-check: **optional**, not required for M2 / criterion C pass  
- Do not claim “engineer-useful clusters” from purity alone  

#### Cascade economics

Compare all-deep vs fast→selected-deep: deep-call / cost reduction vs **proxy-importance** recall delta.

#### Baseline ladder vs JEV comparison (required experiment)

**Hierarchy (weakest → strongest candidate; do not skip weaker rungs):**

```text
Random ranking
  → Frequency-only ranking
  → BaselineFastDecisionEngine
  → JEV (stub → real)
```

**JEV hypothesis:** measure **cost–quality tradeoff** vs Baseline (and weaker baselines). Do **not** assume JEV superiority a priori.

Same splits, same metrics, side-by-side report columns for **all** ladder rungs:

- Detection P / R / F1  
- Proxy-important / review-priority retention  
- Review reduction  
- Cascade cost (if applicable)  
- Failure-type agreement  

JEV may be stub/mock until a real adapter exists; the harness still runs. **Shipping** may use Baseline-only; the **ladder comparison experiment remains in-scope for MVP**.

#### Evaluator improvement

Logged experiments under `experiments/`; ≥3 iterations; hidden-eval comparison; human promote gate.

#### Out of MVP gate (post-MVP)

- Actionability (“would an engineer act?”)  
- True business criticality / downstream model-iteration impact  
- Human severity referee  

### 5.3.1 Proxy importance rules (frozen defaults)

Rule letters below are **frozen for MVP** unless an ADR overrides.

| Rule | Definition | Role |
|------|------------|------|
| **A** | Episode is a **fail** (`success=false` / failure trajectory) | **Required** for proxy-important |
| **B** | **Rare** failure type (low empirical frequency on the reference split/bank) | Acceptance disjunct |
| **C** | **High-frequency** failure type (repeated / common pattern) | Acceptance disjunct |
| **D** | **Novelty** vs Failure Bank (new pattern / high novelty score) | Acceptance disjunct |
| **E** | Actionable-phrase heuristic on diagnosis/correction text | **Ranking bonus only** — not hard GT |
| **F** | Cascade-uncertainty / low-confidence routing signal | **Ranking bonus only** — not hard GT |

**Default acceptance set (proxy-important):**

```text
proxy_important ⇔ A ∧ (B ∨ C ∨ D)
```

E and F may boost Top-K rank order but **do not** admit an episode into the proxy-important set by themselves.

Document in every metrics report: *“proxy importance is rule-based; not business-critical GT.”*

### 5.4 Milestone acceptance gates

| Milestone | Pass if… | Fail if… |
|-----------|----------|----------|
| **M0** Dataset pipeline | RoboFAC → Episode; stats report; fixtures in CI | Dataset-specific fields leak into core pipeline |
| **M0.5** Thin slice | 100–200 episodes → ranking + structured report + basic detection metrics in 1–2 days of work | Cannot produce useful end-to-end insight before larger infra |
| **M1** Failure intelligence | Detection P/R/F1 + type agreement (where mapped) + proxy-important retention + review reduction reported; **Random / Frequency / Baseline / JEV** ladder columns present (JEV may be stub) | No measurable detection/prioritization; ladder harness missing |
| **M2** Failure bank + clustering | Cases stored; clusters + **auto purity vs failure_type** (label consistency only); ranking uses A–F rules | No compression; purity not reported |
| **M3** Cascade | ≥3× (stretch 5×) fewer deep calls; **proxy-importance** recall drop ≤5% | Cascade saves nothing or tanks proxy recall |
| **M4** Evaluator improvement loop | ≥3 experiments; ≥1 hidden improve **or** written postmortem; no auto-promote; hidden labels never used for tuning | Only overfits public/dev; hidden leakage |
| **M5** Demo | Full demo outputs + auto metrics + full ladder (Random→…→JEV) comparison summary | Pretty UI without evidence |

Human cluster spot-check and diagnosis spot-check may appear as **optional notes** in reports; they are **not** pass blockers.

### 5.5 Labels, proxy importance, and the shrunk human role

RoboFAC provides a strong **detection / type / diagnosis text** substrate. MVP acceptance uses that substrate plus **deterministic proxy rules**. We do **not** build human importance / criticality GT for MVP.

**Pivot from rev 4:** rev 4 required human-adjudicated `important_failure` / actionability for gates. **Rev 5 drops that requirement.** User cannot judge business criticality or model-iteration impact yet; requiring human importance GT would invent false precision.

#### What RoboFAC provides vs what MVP adds

| Need for MVP gates | RoboFAC typically provides | Must we add? | Notes |
|--------------------|----------------------------|--------------|-------|
| Failure **detection** (`success` / fail) | Yes | No — map into `ground_truth.success` | Automatable core |
| Failure **type** | Partially — taxonomy + QA identification | Map when present; nullable / rules+LLM draft for taxonomy only | Automatable core when mapped |
| Diagnosis / correction text | Yes via QA | Map when usable; nullable | Weak output; limited validation |
| **Proxy importance** | No | **Yes — rules A–D only** (E/F ranking bonuses) | **Not** human GT; document honesty |
| Common / rare / novelty | No first-class field | **Yes — frequency stats + Failure Bank novelty** | Automatable core |
| Cluster quality | No | **Auto purity vs failure_type** | Human spot-check optional |
| Actionability / true criticality | No | **Post-MVP** | Not an MVP gate |
| Severity referee | No | **Do not require** | No human severity adjudication for MVP |

Schema extension (additive; keep existing nullable GT fields):

```yaml
ground_truth:
  # …existing fields…
  success: bool | null
  failure_type: str | null                # map RoboFAC when present
  diagnosis: str | null
  correction: str | null
  # Proxy fields — computed, not human-adjudicated for MVP:
  proxy_important: bool | null            # A ∧ (B ∨ C ∨ D)
  proxy_rule_hits: list[str] | null       # e.g. ["A","C"]
  frequency_bucket: rare | mid | common | null
  novelty_score: float | null
  # Post-MVP / advisory only (nullable; not acceptance GT):
  important_failure: bool | null          # DEPRECATED for MVP gates — do not require human fill
  attention_must_keep: bool | null        # post-MVP
  actionability_would_act: bool | null    # post-MVP
  cluster_coherence: bool | null          # optional spot-check only
  severity: int | null                    # optional heuristic; not a referee GT
  label_status: draft | chatgpt_reviewed | human_optional
  label_source: rules | llm | hybrid | human
  label_notes: str | null
```

#### Weak-label / ChatGPT path (narrowed)

1. **Rules first** for proxy importance (A–D) and frequency/novelty.  
2. **LLM draft** may help **taxonomy / failure_type mapping drafts** and diagnosis text when RoboFAC mapping is incomplete.  
3. **ChatGPT / weak-label review** stays **advisory for taxonomy drafts only** — **not** for importance GT.  
4. Synthetic importance labels must **not** be invented as fake human GT; if used at all for UI dry-runs, mark clearly and exclude from acceptance numerators (acceptance uses rule-computed `proxy_important`).

#### Human role (shrunk)

| Role | MVP policy |
|------|------------|
| Importance / criticality adjudicator | **Removed** — rules only |
| Severity referee | **Removed** |
| Actionability “would act” | **Post-MVP** |
| Cluster sanity spot-check | **Optional** — not required for pass |
| Taxonomy draft review (ChatGPT/human) | Advisory only |
| Improvement-loop promote gate | Still human (config promote/reject) |

```text
RoboFAC success + type map
        ↓
Rule engine: A, B, C, D → proxy_important
        ↓
E, F → ranking bonuses only
        ↓
Auto metrics (detection, type, novelty, proxy retention, cascade, Baseline↔JEV)
        ↓
Optional: human cluster sanity notes (non-blocking)
```

#### Explicit limits

- **Proxy-important / review-priority is not business importance** and not business-critical GT. Say so in reports.  
- **LLM-vs-LLM agreement is not importance GT** and not actionability GT.  
- Detection and type metrics use RoboFAC fields directly when mapped.  
- Diagnosis: **type agreement primary**; text similarity de-emphasized; draft quality is **not** a hard MVP gate.  
- Cluster purity = **label consistency only**, not engineering usefulness.  
- Hidden labels = **harness final scoring only** — never for prompt/threshold/RSI/routing/weight tuning.  
- Actionability / true criticality = **post-MVP**.

#### Label / proxy risks

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Proxy≠business critical | Stakeholders may over-read “important” | Naming + report disclaimer; freeze rule text |
| Rare∪common (B∨C) feels broad | Almost all fails could be proxy-important | Tune rare/common thresholds on public_eval; report prevalence; keep A required |
| Novelty cold-start | Empty bank → everything novel | Seed bank from dev split; document burn-in |
| Taxonomy draft bias | LLM invents types | Prefer RoboFAC six-class map; `other`/`unknown`; ChatGPT advisory only |
| Hidden-eval contamination | Improvement loop peeks | Split first; loop must not load hidden labels |
| Treating diagnosis as gate | Blocks ship on weak text metrics | Diagnosis = weak output; not pass/fail |

#### P0 freezes (rev 5 + rev 6 clarifications)

| # | Decision | Frozen default |
|---|----------|----------------|
| 1 | **Split / hidden isolation** | Seed `42`; 60/20/20; hidden labels = **harness final scoring only**; never for prompt / threshold / RSI / routing / weight tuning; improvement loop must not load hidden labels when designing candidates |
| 2 | **Automatable core** | success/fail; failure type (RoboFAC map); common/rare + novelty (freq + bank); plus cluster/dedup, Top-K, cascade, auto reports |
| 3 | **Proxy-important / review-priority** | Prefer these terms over “high-value”; A required; acceptance = **A ∧ (B ∨ C ∨ D)**; E/F ranking bonuses only; **review priority ≠ business importance** |
| 4 | **No human importance GT** | Do not require human criticality / severity referee for MVP |
| 5 | **Cluster gate** | Auto purity vs `failure_type` = **label consistency only** (not eng usefulness); human spot-check optional |
| 6 | **Baseline ladder ↔ JEV** | Random → Frequency-only → BaselineFastDecisionEngine → JEV; same splits/metrics; side-by-side; JEV stub OK; **JEV = cost–quality hypothesis**, not assumed superiority; comparison in-scope even if ship is Baseline-only |
| 7 | **Diagnosis** | Weak output; **type agreement primary**; text similarity de-emphasized; optional spot-check; **not** hard gate |
| 8 | **ChatGPT / weak-label path** | Advisory for **taxonomy drafts only**; never importance / review-priority GT |
| 9 | **Actionability / true criticality** | Post-MVP |
| 10 | **`failure_type` taxonomy** | Prefer RoboFAC six-class map; extend only with `other` / `unknown` |

---

## 6. Architecture (runnable slice)

```text
RoboFAC subset
    → RoboFACAdapter
    → Episode (unified schema)
    → Feature / Summary Extraction
    → Decision / ranking ladder (same harness)
         ├─ Random ranking
         ├─ Frequency-only ranking
         ├─ BaselineFastDecisionEngine  (default ship path)
         └─ JevDecisionEngine           (stub→real; cost–quality experiment)
    → route: pass | failure / uncertain / review-priority
    → DeepReasoner (selected only)
    → FailureBank  (frequency + novelty signals)
    → Clustering + Ranking  (proxy rules A–D; bonuses E/F)
    → EvaluationHarness + Report
         (side-by-side ladder columns on same splits/metrics)
    → (M5) Streamlit Dashboard
    → Evaluator Improvement Loop
         → experiments/ → replay → public eval
         → hidden eval (final scoring only) → promote gate
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
  proxy_important: bool | null
  proxy_rule_hits: list[str] | null
  frequency_bucket: rare | mid | common | null
  novelty_score: float | null
model_output:
  failure_probability: float | null
  failure_type: str | null
  confidence: float | null
  severity: float | null          # optional score; not human referee GT
  novelty: float | null
  needs_deep_review: bool | null
  diagnosis: str | null
  evidence: list[str]
  recommended_actions: list[object]
  decision_backend: baseline | jev | null
```

Adapters must not leak RoboFAC-only field names into decision / reasoning / clustering code.

### 6.2 Required interfaces

- `DatasetAdapter.load() -> Iterable[Episode]`
- `FeatureExtractor.extract(episode) -> EpisodeFeatures`
- `FastDecisionEngine.evaluate(features, decision_schema) -> DecisionResult`
  - `RandomRanker` / `FrequencyOnlyRanker` (required weak baselines for the ladder)
  - `BaselineFastDecisionEngine` (required default ship path)
  - `JevDecisionEngine` (real or stub/mock; always runnable in harness; cost–quality experiment)
- `VisionReasoningProvider.summarize(frames, prompt) -> str`
- `DeepReasoner.reason(context) -> StructuredDiagnosis` (+ raw / model / prompt versions)
- `FailureBank.upsert(FailureCase)` (+ frequency / novelty queries)
- `Clusterer.fit_predict(cases) -> Clusters`
- `Ranker.score(case_or_cluster, weights) -> float` (honors A–D set + E/F bonuses)
- `ProxyImportance.tag(episode, bank_stats) -> ProxyLabel` (deterministic rules)
- `EvaluationHarness.evaluate(...) -> Metrics` (emits Baseline↔JEV side-by-side)
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

**Pipeline:** Episode → features → BaselineFastDecisionEngine (+ optional Random/Frequency columns) → proxy-important / review-priority tagging → failure ranking → markdown/HTML report  

**Input:** 100–200 episodes  

**Deliverables:** `reports/m0_5_thin_slice.md` with ranking, examples, basic detection metrics  

**Exit:** A human can skim the report and say whether the automatable direction looks useful — within ~1–2 days of work after M0  

### M1 — Failure intelligence

**Deliverables:** deterministic features; Random + Frequency-only baselines; `BaselineFastDecisionEngine`; `JevDecisionEngine` stub (or real); proxy-important / review-priority rules A–F; detection + type + novelty metrics; attention efficiency on proxy set; **full ladder side-by-side** in `reports/m1_failure_intelligence.md`  

**Exit:** Detection F1 ≥ 0.70 **or** documented failure modes if missed; proxy-important retention + review reduction reported; ladder comparison table present (stub JEV OK); JEV framed as cost–quality experiment  

### M2 — Failure bank + clustering (no dashboard)

**Deliverables:** `FailureCase` store (JSONL/SQLite); embedding + agglomerative clustering; configurable ranking weights (A–D set, E/F bonuses); `failure_bank` artifact + `reports/m2_clusters.md` with **auto purity vs failure_type** (label consistency only; optional human sanity notes)  

**Exit:** Failures compressed; auto purity reported as label consistency; **no Streamlit required here**; human spot-check not required for pass  

### M3 — Fast / deep cascade

**Deliverables:** Deep reasoner behind providers; routing policy; A (all-deep) vs B (cascade) comparison; `reports/m3_cascade.md` (include per-backend columns when both paths run)  

**Exit:** ≥3× fewer deep calls (stretch 5×); **proxy-importance** recall drop ≤5%  

### M4 — Evaluator improvement loop (formerly “RSI Agent”)

**Deliverables:** experiment generator + replay harness + comparator; allowed mutations: prompts, thresholds, routing, ranking weights, feature flags; artifacts under `experiments/` and `reports/rsi/`; human approve/reject only  

**Success function (configurable):**

```text
score =
  0.40 * failure_recall
+ 0.20 * failure_precision
+ 0.15 * failure_type_agreement
+ 0.15 * review_reduction
+ 0.10 * cost_reduction
```

Guardrail: **proxy-importance** recall must not drop beyond configured limit.

**Exit:** ≥3 logged experiments; ≥1 hidden-eval improvement **or** postmortem; hidden labels never used for candidate tuning  

### M5 — Final MVP demo (+ Streamlit)

**Deliverables:** `scripts/run_mvp_demo.py`; Streamlit for overview / clusters / case detail / GT vs pred; final report answering §12  

**Demo outputs:** deployment summary, review-priority failure ranking, clusters, representative cases, AI diagnosis **draft**, recommended-action sketches, auto metrics, proxy attention efficiency, full ladder (Random→…→JEV) comparison, improvement history  

**Not required for demo pass:** human “would act” majority, business-critical GT, severity referee.

---

## 9. MVP success criteria

| ID | Criterion | Target |
|----|-----------|--------|
| A | Failure detection | F1 ≥ 0.70 **or** documented miss modes + remediation plan |
| B | Attention efficiency (proxy / review-priority) | ≥70% review volume reduction while retaining ≥80% **proxy-important** failures (A∧(B∨C∨D)); review priority ≠ business importance |
| C | Failure compression | Auto cluster purity vs `failure_type` (= label consistency only) reported and above configured floor; human spot-check optional |
| D | Cascade efficiency | ≥5× cost/call reduction stretch (≥3× minimum) with ≤5% **proxy-importance** recall drop |
| E | Baseline ladder ↔ JEV | Side-by-side Random / Frequency-only / Baseline / JEV on same splits for detection P/R/F1, proxy retention, review reduction, cascade cost (if applicable), type agreement; JEV = cost–quality hypothesis; stub JEV acceptable |
| F | Evaluator improvement | ≥3 experiments; ≥1 hidden-eval improve **or** documented negative result; hidden = final scoring only |
| G | Portability | New dataset = new `DatasetAdapter` only |
| — | Diagnosis draft | Weak — **type agreement primary**; text similarity de-emphasized (+ optional spot-check); **not a hard gate** |
| — | Actionability / true criticality | **Post-MVP** |

---

## 10. Engineering principles

1. **Goals-first** — freeze success definition before features  
2. **Evaluator-first** — harness lands by M1; thin slice by M0.5; full ladder columns (Random→…→JEV) from day one of harness  
3. **Data-first** — fixtures always run; RoboFAC is additive  
4. **API-first** — providers behind interfaces; keys via env only  
5. **Modular** — dataset specifics stay in adapters; taxonomy in `ontology/`  
6. **Measurable & reproducible** — fixed seeds; versioned prompts/configs; logged experiments  
7. **Honest proxies** — never present rule-based review-priority as business importance; cluster purity ≠ eng usefulness; JEV not assumed superior  

### Env vars

```bash
OPENAI_API_KEY=          # optional
OPENAI_BASE_URL=         # optional
VLM_PROVIDER=mock|openai|local
LLM_PROVIDER=mock|openai|local
JEV_API_KEY=             # optional
JEV_MODE=stub|real       # default stub until credentials/adapter ready
IMPROVEMENT_MODEL=       # optional
```

Missing keys → mock providers; never halt the pipeline. Missing JEV → stub path still participates in comparison harness.

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
    jev.py               # real or stub
  reasoning/
    base.py
    providers/
  failure_bank/
  clustering/
  ranking/
  proxy_importance/      # rules A–F
  evaluation/            # side-by-side Baseline vs JEV
  experiments/           # exp001_*.yaml + results
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

1. Does the pipeline reduce review volume while retaining **proxy-important / review-priority** failures (rules disclosed; review priority ≠ business importance)?  
2. Detection P/R/F1 on public + hidden (hidden = final scoring only)?  
3. Failure-type agreement where labels exist?  
4. Do frequency + Failure Bank novelty separate common vs rare / novel?  
5. Ladder results — **Random / Frequency-only / Baseline / JEV** side-by-side? Did JEV justify its cost?  
6. If JEV was stub-only: what is needed to re-run with a real adapter?  
7. Auto cluster purity vs `failure_type` (label consistency only)? (Optional: human sanity notes — not eng-usefulness claims?)  
8. Cascade cost/call reduction vs proxy-important recall delta?  
9. Did the improvement loop help on hidden eval without hidden leakage into tuning?  
10. Which wins came from prompt / routing / features / weights?  
11. Minimum data a real robotics partner must provide?  
12. Next: optimize evaluator further, find a design partner, or start **post-MVP** actionability / true-criticality labeling?  
13. Diagnosis drafts: **type agreement primary**; text similarity de-emphasized — not treated as a ship gate?

---

## 13. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| RoboFAC download/schema mismatch | Fixture-first; adapter isolation |
| No API keys | Mock providers; call-count cost proxies |
| Proxy / review-priority ≠ business importance over-read | Report disclaimer; naming (`proxy_important` / `review-priority`); avoid “high-value”; freeze rule text |
| B∨C too inclusive / too exclusive | Publish prevalence; tune thresholds on public_eval; keep A required |
| Novelty cold-start | Seed bank from dev; document burn-in window |
| Incomplete type labels | Nullable type; agreement only on mapped rows; taxonomy drafts advisory |
| ChatGPT path misused as importance GT | Explicit: taxonomy drafts only; never importance |
| Diagnosis treated as hard gate | Weak output policy; limited validation only |
| JEV unavailable or over-assumed superior | Stub/mock still runs ladder; ship may be Baseline-only; frame JEV as cost–quality experiment |
| Improvement loop overfits | Hidden split sealed; promote gate |
| Scope creep (actionability, criticality GT) | Non-goals + freeze list; post-MVP explicit |
| Cluster count gaming / purity over-claim | Auto purity = label consistency only; do not claim eng usefulness; optional human notes only |
| UI distraction | Dashboard only at M5 |
| Hidden-eval contamination | Split first; hidden = harness final scoring only; never for prompt/threshold/RSI/routing/weight tuning |

---

## 14. MVP development plan (implementation PRs)

Docs PR **#1** (`docs/mvp-implementation-plan`) freezes the plan. **Application code lands in separate, focused PRs** numbered `PR-impl-N` below (distinct from docs PR #1). Prefer ~8–12 small PRs over one giant merge.

**Principles for impl PRs:** one milestone capability per PR when practical; fixtures + tests in the same PR as the code they protect; mock providers by default; no hidden-label use for tuning.

### 14.0 Standing implementation-PR gate

**For every implementation PR:** after each update, send the updated PR to **ChatGPT Career** for review. Iterate on the PR and resubmit after each update until the verdict is exactly **Approve** (not **Need changes**). Only after an **Approve** verdict, ask the user for final confirmation on whether to merge.**

### 14.1 Milestone → PR map

| Milestone | Primary impl PRs | Exit evidence |
|-----------|------------------|---------------|
| **M0** Dataset pipeline | PR-impl-1, PR-impl-2 | Adapter + splits + CI smoke |
| **M0 / M1 prep** Proxy rules | PR-impl-3 | Deterministic `proxy_important` + report |
| **M1** Failure intelligence | PR-impl-4, PR-impl-6 | Ladder harness + metrics |
| **M0.5** Thin slice | PR-impl-5 | End-to-end report on 100–200 eps |
| **M2** Failure bank + clustering | PR-impl-7 | Bank + purity (label consistency) |
| **M3** Cascade economics | PR-impl-8 | Cheap→fast→deep + cost metrics |
| **M4** Evaluator improvement | PR-impl-9 | ≥3 experiments; hidden final-score only |
| **M5** Demo + final pack | PR-impl-10 | Streamlit + metrics pack answering §12 |

### 14.2 Sequenced implementation PRs

#### PR-impl-1 — Episode schema + RoboFACAdapter + fixtures / CI smoke

| | |
|--|--|
| **Milestone** | M0 |
| **Scope / files** | `data/schemas/episode.py` (or YAML schema + pydantic); `data/adapters/robofac.py`; `data/samples/` tiny fixtures; `tests/test_adapter_robofac.py`; `tests/test_episode_schema.py`; `pyproject.toml` / pytest config; `scripts/m0_dataset_stats.py` (stub OK) |
| **Goal** | Unified `Episode` loads from RoboFAC sample without leaking RoboFAC field names into core modules |
| **Acceptance / verify** | `pytest` green offline; fixture episodes validate schema; adapter unit test covers success + ≥1 failure type; CI smoke job runs without API keys |
| **Depends-on** | Docs freeze (this plan); none code |

#### PR-impl-2 — GT mapping (success / type) + split writer (60/20/20)

| | |
|--|--|
| **Milestone** | M0 |
| **Scope / files** | `ontology/robofac_type_map.py`; GT mapping helpers; `scripts/write_splits.py`; `data/splits/{development,public_eval,hidden_eval}.jsonl` (or path manifests); `tests/test_splits.py`; `tests/test_gt_mapping.py` |
| **Goal** | Deterministic seed-`42` 60/20/20 splits; map RoboFAC `success` + `failure_type` into `ground_truth`; hidden split sealed for harness-only scoring |
| **Acceptance / verify** | Split sizes within tolerance; no episode ID overlap; remapping idempotent; test asserts improvement-loop helpers **cannot** load hidden labels; mapped-type agreement fixture check |
| **Depends-on** | PR-impl-1 |

#### PR-impl-3 — Proxy review-priority rules A–D + report

| | |
|--|--|
| **Milestone** | M0 → M1 prep |
| **Scope / files** | `proxy_importance/` (rules A–F); config for rare/common thresholds; `scripts/m0_proxy_importance_report.py` → `reports/m0_proxy_importance.md`; `tests/test_proxy_rules.py` |
| **Goal** | Deterministic `proxy_important ⇔ A ∧ (B ∨ C ∨ D)`; E/F as ranking bonuses only; terminology = proxy-important / review-priority |
| **Acceptance / verify** | Unit tests for each rule and conjunction; report states *review priority ≠ business importance*; prevalence of proxy-important on public_eval logged; no hidden labels used to tune thresholds (tune on public_eval / dev only) |
| **Depends-on** | PR-impl-2 |

#### PR-impl-4 — BaselineFastDecisionEngine + Random / Frequency baselines + eval harness skeleton

| | |
|--|--|
| **Milestone** | M1 |
| **Scope / files** | `features/`; `decision/base.py`, `decision/baseline.py`, `decision/random.py`, `decision/frequency.py`; `evaluation/harness.py` (ladder columns); `tests/test_decision_*.py`; `tests/test_harness_skeleton.py` |
| **Goal** | Runnable ladder rungs Random → Frequency-only → BaselineFastDecisionEngine; harness emits detection P/R/F1 + proxy retention + review reduction columns |
| **Acceptance / verify** | Offline pytest with mocks; harness runs on fixtures; Random and Frequency-only produce finite metrics; Baseline beats Random on fixture smoke **or** miss modes documented; no JEV required yet |
| **Depends-on** | PR-impl-3 |

#### PR-impl-5 — Thin-slice M0.5 end-to-end report (100–200 episodes)

| | |
|--|--|
| **Milestone** | M0.5 |
| **Scope / files** | `scripts/m0_5_thin_slice.py` → `reports/m0_5_thin_slice.md`; thin-slice config; optional ranking CLI |
| **Goal** | Prove dataset → insight early: Episode → features → Baseline (+ optional weak baselines) → proxy tagging → Top-K → markdown report |
| **Acceptance / verify** | Report generates on 100–200 eps (fixture subset OK if full download blocked); includes ranking examples + basic detection metrics + proxy-important retention sketch; runs without API keys |
| **Depends-on** | PR-impl-4 (Baseline + harness skeleton); PR-impl-3 |

#### PR-impl-6 — JEV stub adapter + side-by-side comparison report

| | |
|--|--|
| **Milestone** | M1 |
| **Scope / files** | `decision/jev.py` (stub/real switch via `JEV_MODE`); harness ladder column for JEV; `reports/m1_failure_intelligence.md` generator; `tests/test_jev_stub.py` |
| **Goal** | Full ladder Random → Frequency-only → Baseline → JEV on same splits/metrics; frame JEV as **cost–quality** experiment, not assumed winner |
| **Acceptance / verify** | Stub path always runnable; side-by-side table present; cost proxy column (calls / tokens / stub cost units); detection F1 ≥ 0.70 **or** documented miss modes; proxy retention + review reduction reported for all rungs |
| **Depends-on** | PR-impl-4; ideally after or parallel with PR-impl-5 |

#### PR-impl-7 — Failure bank + clustering + purity metric

| | |
|--|--|
| **Milestone** | M2 |
| **Scope / files** | `failure_bank/`; `clustering/`; ranking weights config; `scripts/m2_cluster_report.py` → `reports/m2_clusters.md`; `tests/test_failure_bank.py`, `tests/test_clustering_purity.py` |
| **Goal** | Upsert failure cases; novelty/frequency signals; agglomerative clustering; **auto purity vs `failure_type` = label consistency only** |
| **Acceptance / verify** | Bank round-trip test; purity metric unit-tested; report states purity ≠ engineering usefulness; ranking honors A–D set + E/F bonuses; no Streamlit |
| **Depends-on** | PR-impl-6 (or PR-impl-4 minimum) |

#### PR-impl-8 — Cascade cheap → fast → deep + cost metrics

| | |
|--|--|
| **Milestone** | M3 |
| **Scope / files** | `reasoning/` providers (mock default); routing policy; A (all-deep) vs B (cascade) runner; `reports/m3_cascade.md`; `tests/test_cascade_routing.py` |
| **Goal** | Prove cascade economics with proxy-important recall guardrail |
| **Acceptance / verify** | ≥3× fewer deep calls vs all-deep (stretch 5×); proxy-important recall drop ≤5%; cost proxy logged per ladder backend when both run; mock providers keep CI green |
| **Depends-on** | PR-impl-7; PR-impl-6 |

#### PR-impl-9 — Evaluator improvement loop (config experiments, human promote, hidden isolation)

| | |
|--|--|
| **Milestone** | M4 |
| **Scope / files** | `rsi/` (experiment, replay, promotion); `experiments/` sample YAMLs; `reports/rsi/`; `tests/test_improvement_loop_isolation.py` |
| **Goal** | Bounded mutations (prompts, thresholds, routing, ranking weights, feature flags); human promote/reject; **hidden labels for harness final scoring only** |
| **Acceptance / verify** | ≥3 logged experiments; test proves loop cannot read hidden labels during propose/tune; promote path scores hidden **only** via harness API; ≥1 hidden improve **or** written postmortem; no auto-overwrite of production config |
| **Depends-on** | PR-impl-8 (or PR-impl-6 + harness if cascade deferred) |

#### PR-impl-10 — M5 demo Streamlit + final metrics pack

| | |
|--|--|
| **Milestone** | M5 |
| **Scope / files** | `dashboard/` Streamlit app; `scripts/run_mvp_demo.py`; `reports/m5_final.md` (answers §12); metrics pack artifact |
| **Goal** | Demo the automatable loop + ladder comparison + honesty disclaimers |
| **Acceptance / verify** | Demo runs with mocks; shows overview / clusters / case detail / GT vs pred; final report covers Q1–Q6 + criteria A–G; UI copy uses proxy-important / review-priority (not “high-value” as business claim) |
| **Depends-on** | PR-impl-9 (soft: can demo earlier slices, but M5 exit needs loop history + ladder metrics) |

### 14.3 Dependency sketch

```text
PR-impl-1 → PR-impl-2 → PR-impl-3 → PR-impl-4 ┬→ PR-impl-5 (M0.5 thin slice)
                                              └→ PR-impl-6 (JEV + ladder report)
                                                     ↓
                                               PR-impl-7 (bank + clusters)
                                                     ↓
                                               PR-impl-8 (cascade)
                                                     ↓
                                               PR-impl-9 (improvement loop)
                                                     ↓
                                               PR-impl-10 (M5 demo)
```

If blocked on RoboFAC download, substitute fixtures, log assumptions, continue. If blocked on real JEV, keep stub and still ship ladder columns.

### 14.4 Final MVP done checklist

We hit MVP when **all** of the following are true (tie to §4 Qs and §9 criteria):

| Check | Maps to | Evidence |
|-------|---------|----------|
| ☐ Detection usable | Q2, criterion A | P/R/F1 on public + hidden (harness); F1 ≥ 0.70 **or** documented miss modes |
| ☐ Failure-type usable | Q2b | Type agreement on mapped rows; diagnosis text similarity **not** required to pass |
| ☐ Common / rare / novelty reported | Q2c | Frequency + Failure Bank novelty in auto metrics |
| ☐ Review volume ↓ with proxy retention | Q1, criterion B | ≥70% review reduction; ≥80% proxy-important retention; disclaimer: review priority ≠ business importance |
| ☐ Clusters label-consistent | Q3, criterion C | Auto purity vs `failure_type` reported; no eng-usefulness claim required |
| ☐ Cascade saves cost | Q4, criterion D | ≥3× deep-call reduction; ≤5% proxy-important recall drop |
| ☐ Ladder comparison complete | Q5, criterion E | Random / Frequency-only / Baseline / JEV side-by-side; JEV framed as cost–quality |
| ☐ Bounded improvement honest | Q6, criterion F | ≥3 experiments; ≥1 hidden improve **or** postmortem; **no hidden tuning** |
| ☐ Portability | criterion G | New dataset = new `DatasetAdapter` only |
| ☐ Thin slice early | M0.5 | `reports/m0_5_thin_slice.md` exists |
| ☐ Demo + final report | M5, §12 | Streamlit + `reports/m5_final.md` answering final report questions |
| ☐ Non-goals respected | §1.5 | No business-critical GT, no actionability gate, no training loop, no realtime control |

**Diagnosis draft** and **actionability / true criticality** remain non-blocking / post-MVP respectively.

---

## 15. Open assumptions

1. RoboFAC ≤2k subset has enough success/failure diversity.  
2. Pipeline works metadata-only when video missing.  
3. Proxy rules A∧(B∨C∨D) are a good-enough **review-priority** stand-in until business-importance GT exists (post-MVP).  
4. Rare vs common frequency thresholds can be set from public_eval without hidden leakage.  
5. JEV optional for shipping; stub is enough to exercise the ladder harness; JEV not assumed superior.  
6. English labels/prompts OK for RoboFAC.  
7. Auto cluster purity vs `failure_type` (label consistency) is a sufficient M2 gate without mandatory human review; eng usefulness is out of scope.

---

## 16. Revision history

| Rev | Change |
|-----|--------|
| 1 | Initial scoped plan (docs-only PR) |
| 2 | Added Project Goals + Validation protocol; M0.5 thin slice; M4 renamed to Evaluator Improvement Loop; Streamlit deferred to M5; success criteria emphasize attention efficiency, diagnosis separation, actionability; added `experiments/` + `ontology/`; expanded non-goals (no realtime, no universal ontology) |
| 3 | Added §5.5 Ground-truth gaps / weak labels / ChatGPT review + human confirm workflow; promotion rule (`human_confirmed` only for acceptance gates); pre-MVP open-questions checklist; freeze list + incomplete-labels risk updated |
| 4 | ChatGPT Career GT-workflow review (conditional approve): rename framing to human-adjudicated reference/benchmark labels (keep `human_confirmed` field); P0 freezes + defaults (lifecycle/isolation, important=ANY severity/freq/cost/novelty/actionable, severity 3/2/1 heuristic, spot-check ~20×5 + 20–30 actionability, confirmer=robotics expert else two reviewers); weak/synthetic for dev only; risks (distribution shift, LLM-review bias, hidden contamination, early taxonomy freeze); detection≠diagnosis, important⊥taxonomy, cluster/actionability need human adjudication |
| 5 | **Pivot — no business-critical / human importance GT for MVP.** Automatable core = success/fail + failure type + common/rare/novelty (+ cluster/dedup, Top-K, cascade, auto reports). Proxy importance frozen: A fail required; B rare; C high-freq; D novelty; acceptance **A∧(B∨C∨D)**; E actionable-phrase + F cascade-uncertainty = ranking bonuses only. Diagnosis = weak output (not hard gate). Actionability / true criticality = post-MVP. Cluster gate = auto purity vs `failure_type`; human spot-check optional. Human role shrunk (no severity referee). ChatGPT/weak-label path = taxonomy drafts advisory only. **Required Baseline↔JEV comparison** on same splits/metrics (JEV stub OK; comparison in-scope even if ship is Baseline-only). Updated goals, freezes, Qs, §5, criteria, milestones, risks, assumptions. |
| 6 | **ChatGPT Career rev5 clarifications + development PR plan.** Terminology: prefer **proxy-important / review-priority** over “high-value”; review priority ≠ business importance. Baseline hierarchy: Random → Frequency-only → BaselineFastDecisionEngine → JEV; JEV = cost–quality hypothesis (not assumed superiority). Cluster purity = label consistency only (not eng usefulness). Diagnosis: type agreement primary; text similarity de-emphasized. Hidden labels: harness final scoring only — never for prompt/threshold/RSI/routing/weight tuning. New **§14 MVP development plan** with PR-impl-1…10 (scope, acceptance, depends-on, M0→M5 map, final done checklist). Review note: `docs/reviews/chatgpt-rev5-review.md`. Freeze list + README → rev 6. |
| 6.1 | **Standing implementation-PR review gate.** After every PR update, obtain ChatGPT Career review and iterate until the verdict is **Approve** (not **Need changes**); only then request the user’s final merge confirmation. |

---

## 17. PR intent

Docs-only plan freeze before M0 coding. Implementation follows this document via **PR-impl-1…10** (§14), starting at **M0**, then **M0.5** before larger infrastructure. Rev 5 locked automatable-core + proxy-importance + Baseline↔JEV; **rev 6** locks terminology, baseline ladder, cluster-purity honesty, diagnosis metric priority, hidden-isolation, and the sequenced impl-PR plan; **rev 6.1** adds the standing implementation-PR review/merge gate.
