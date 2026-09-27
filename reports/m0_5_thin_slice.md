# M0.5 thin-slice end-to-end report

> **Disclaimer:** **review priority ≠ business importance.** `proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business criticality or confirmed model-iteration impact. BaselineFast is GT-free path/id heuristics — miss modes: renamed paths without `fail`/`success` tokens; missing type keywords → development prior / majority type (may tie Frequency).

## Run metadata

| Field | Value |
|-------|-------|
| Data source | real RoboFAC (test_qa_sim labeled subset) |
| Dataset size (thin corpus) | 150 |
| Seed | 42 |
| Development (reference) | 105 |
| Eval split | `public_eval` (holdout) n=45 |
| Top-K | 15 |
| API keys | none required |

### Notes

- Loaded 1512 labeled RoboFAC test_qa episodes; sampled n=150 (seed=42).
- Reference = development (Frequency/Baseline fit + proxy bank). Eval = public_eval-style holdout. hidden_eval never loaded.

## Ladder metrics (eval holdout)

| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention | Review reduction | n_proxy_imp |
|---------|-------|-------|--------|------------|-----------------|------------------|-------------|
| random | 0.850 | 0.447 | 0.586 | 0.250 | 0.316 | 0.667 | 38 |
| frequency | 0.844 | 1.000 | 0.916 | 1.000 | 0.342 | 0.667 | 38 |
| baseline | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 | 0.667 | 38 |

## Proxy-important / attention sketch (Baseline Top-K)

- Proxy-important count on eval: **38**
- Retention of proxy-important in Top-15: **39.5%**
- Review reduction vs reviewing all eval episodes: **66.7%** (review 15/45)
- Review reduction vs reviewing all GT fails on eval: **60.5%** (review 15/38 fails)

## Ranking examples (Baseline Top-K)

| Rank | episode_id | p_fail | proxy | rule hits | GT success | GT type |
|------|------------|--------|-------|-----------|------------|---------|
| 1 | `04119f95-3bdb-4ade-b05d-9382de1afdca` | 0.78 | Y | ['A', 'D'] | False | None |
| 2 | `09297761-39f1-4754-a884-62b5ba06d484` | 0.78 | Y | ['A', 'D'] | False | None |
| 3 | `0ef1f71a-cfb8-4041-a252-53d77b70bf53` | 0.78 | Y | ['A', 'D'] | False | None |
| 4 | `225a5bc9-9039-4810-8f15-8e73052b093c` | 0.78 | Y | ['A', 'C'] | False | position_deviation |
| 5 | `24989eb1-8c4a-4425-a39d-052d8bef2633` | 0.78 | Y | ['A', 'D'] | False | None |
| 6 | `38f76492-8083-47b9-b71b-0cacf393c287` | 0.78 | Y | ['A', 'D'] | False | None |
| 7 | `3b79bc38-59ff-47e8-a93c-9bd6e9cb6e59` | 0.78 | Y | ['A', 'D'] | False | None |
| 8 | `3d296edb-e4be-4605-9f39-18c5a25c40e9` | 0.78 | Y | ['A', 'D'] | False | None |
| 9 | `47502d68-dcb7-41f4-a9e9-b324a0d34825` | 0.78 | Y | ['A', 'C'] | False | position_deviation |
| 10 | `51452c1a-d655-4690-94e4-e1794e584e25` | 0.78 | Y | ['A', 'D'] | False | None |
| 11 | `52c9c5ef-f751-41b2-812e-67e4a3b0ea72` | 0.78 | Y | ['A', 'D'] | False | None |
| 12 | `54a3b661-3194-43ec-9fea-1ccbd8ba0afa` | 0.78 | Y | ['A', 'D'] | False | None |
| 13 | `58fcef5f-bd22-46cc-961c-3f320e2fb842` | 0.78 | Y | ['A', 'D'] | False | None |
| 14 | `5a052db3-8239-48ad-938c-e1659d9cedb2` | 0.78 | Y | ['A', 'D'] | False | None |
| 15 | `659414ad-a497-42ab-b393-7d762f320123` | 0.78 | Y | ['A', 'D'] | False | None |

## What this tells us about rule-based effect

On this slice (source=robofac_real), Baseline detection F1=1.000 vs Frequency=0.916 vs Random=0.586.
RoboFAC success trajectories often live under `dataset_success_cleaned` (strong success path cue); many fails lack `fail`/`failure` path tokens, so Baseline leans on the development fail prior for cue-less fails. That works when the prior is fail-heavy (typical here) but is still a **heuristic**, not perception. Type agreement stays limited: many RoboFAC fails have null `failure_type` in QA, and keyword→type mapping only fires when path/instruction carries type tokens.
Proxy retention 39.5% at Top-15 is a **review-priority** sketch only — ranking by predicted fail probability does not encode business importance. Rule-based miss modes remain: renamed paths, cue collisions, and types absent from keyword tables.

## Reproduce

```bash
python scripts/m0_5_thin_slice.py --config configs/m0_5_thin_slice.json
```
