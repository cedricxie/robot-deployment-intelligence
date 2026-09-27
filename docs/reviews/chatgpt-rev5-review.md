# ChatGPT Career review — rev 5 plan critique

**Date:** 2026-09-26 (PT)  
**Target doc:** `docs/mvp-plan.md` (rev 5 → rev 6)  
**Outcome:** **Fold clarifications** — terminology, baseline ladder, cluster purity honesty, diagnosis metrics, hidden-label isolation; plus a sequenced implementation-PR plan. Do not invent beyond this review.

## Verdict

Approve the rev 5 automatable-core / proxy-importance / Baseline↔JEV pivot **with five clarifications** that must land before M0 coding, plus an explicit multi-PR development sequence so implementation does not land as one giant PR.

## Feedback folded into rev 6

### 1. Terminology: proxy-important / review-priority (not “high-value”)

- Prefer **`proxy-important`** and **`review-priority`** over casual “high-value”.
- **Review priority ≠ business importance.** Ranking for human attention under frozen proxy rules is not a claim of downstream criticality or model-iteration impact.
- Reports and UI copy must keep this honesty line visible.

### 2. Baseline hierarchy (do not assume JEV wins)

Required comparison ladder (same splits, same metrics):

```text
Random ranking
  → Frequency-only ranking
  → BaselineFastDecisionEngine
  → JEV (stub → real)
```

- **JEV hypothesis = measure cost–quality tradeoff**, not assume superiority.
- Side-by-side reports must include the weaker baselines so gains are interpretable.
- Shipping may still default to Baseline; the ladder experiment remains in-scope.

### 3. Cluster purity = label consistency only

- Auto cluster purity / homogeneity vs `failure_type` measures **label consistency**, not engineering usefulness.
- Do not claim “useful clusters for engineers” from purity alone.
- Human cluster sanity remains optional and non-blocking; true usefulness is post-MVP / qualitative.

### 4. Diagnosis eval: type agreement primary

- **Primary:** failure-type agreement where mapped labels exist.
- **De-emphasize** free-text similarity vs RoboFAC diagnosis/correction strings (informative, weak, not a gate).
- Diagnosis remains weak output; not a hard ship criterion.

### 5. Hidden labels: harness final scoring only

- `hidden_eval` labels are for **evaluation harness final scoring only**.
- **Never** use hidden labels for prompt tuning, threshold tuning, RSI / improvement-loop mutations, routing policy search, or ranking-weight search.
- Improvement loop may see structure/IDs on hidden only if needed for replay scoring through the harness API — never raw labels for candidate design.

## Plan follow-through

Incorporated into `docs/mvp-plan.md` revision 6:

- Terminology sweep + honesty lines
- Baseline ladder + JEV cost–quality hypothesis
- Cluster purity honesty
- Diagnosis metric priority
- Strengthened hidden-isolation freeze
- New **§14 MVP development plan (implementation PRs)** — PR-impl-1…10 mapped to M0→M5, with acceptance / depends-on / final MVP done checklist
- README status → rev 6; freeze list + revision history updated
