# ChatGPT Career review — GT / weak-label workflow (§5.5)

**Date:** 2026-09-26 (PT)  
**Target doc:** `docs/mvp-plan.md` (rev 3 → rev 4)  
**Outcome:** **Conditional approve** — fold framing, P0 freezes, defaults, and risks into the plan; do not invent beyond this review.

## Verdict

Conditional approve of the weak-label → ChatGPT review → human confirm workflow, with rename/framing and hard freezes before M0/M1 acceptance harnesses.

## Framing

- Prefer the term **human-adjudicated reference / benchmark labels** for acceptance GT.
- Machine field may stay `label_status: human_confirmed` if useful; document that it means human-adjudicated reference/benchmark label, not casual LLM agreement.

## Risks called out

1. **Synthetic / weak-label distribution shift** — dev metrics diverge from human-adjudicated acceptance.
2. **LLM-review bias** — model reviewing another model’s drafts.
3. **Hidden-eval contamination** — labeling or promoting on hidden leaks into the evaluator loop.
4. **Early taxonomy freeze** — locking `failure_type` too early couples importance/diagnosis poorly.

## P0 freezes (must land before M0/M1 coding that depends on them)

1. **Label lifecycle + split/hidden isolation** — freeze splits first; human review / adjudication only on non-hidden; synthetic/weak only for development augmentation, not hidden acceptance.
2. **Important-failure definition.**
3. **Severity proxy.**
4. **Spot-check size + dispute process + confirmer.**
5. (Implicit with 1) Synthetic/weak labels are development-only until human adjudication.

## Frozen defaults from review

| Topic | Default |
|-------|---------|
| Important failure | **ANY** of: high severity / high frequency / high eng cost / novelty / actionable |
| Severity | Dataset or human when present; else heuristic **3 / 2 / 1** (high / mid / low) |
| Spot-check | ~**20 clusters × 5 episodes (~100)** + **20–30 actionability** items; disputes always reviewed |
| Confirmer | Robotics expert; else **two independent reviewers** |

## Doc principles

- Separate **detection** vs **diagnosis**.
- **Important** is independent of taxonomy.
- **Cluster usefulness** and **actionability** require human adjudication for acceptance.

## Plan follow-through

Incorporated into `docs/mvp-plan.md` revision 4 (§5.5, freeze list, §13 risks, revision history).
