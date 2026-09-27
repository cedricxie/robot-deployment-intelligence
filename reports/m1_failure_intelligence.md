# M1 failure intelligence — ladder + JEV cost–quality

> **Disclaimer:** **review priority ≠ business importance.** `proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business criticality. **JEV is a cost–quality experiment**, not an assumed winner over BaselineFast. Stub is GT-free (path + instruction cues) with stub cost units.

## Run metadata

| Field | Value |
|-------|-------|
| Data source | real RoboFAC (test_qa_sim labeled subset) |
| Dataset size (corpus) | 150 |
| Seed | 42 |
| Development (reference) | 105 |
| Eval split | `public_eval` (holdout) n=45 |
| Top-K | 15 |
| JEV mode | `stub` |
| API keys | none required (stub) |

### Notes

- Loaded 1512 labeled RoboFAC test_qa episodes; sampled n=150 (seed=42).
- Reference = development (Frequency/Baseline/JEV fit + proxy bank). Eval = public_eval-style holdout. hidden_eval never loaded.
- JEV_MODE=stub. JEV is a cost–quality experiment — not assumed better than Baseline.

## Side-by-side ladder (eval holdout)

| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention | Review reduction | Cost calls | Cost tokens | Stub cost units | n_proxy_imp |
|---------|-------|-------|--------|------------|-----------------|------------------|------------|-------------|-----------------|-------------|
| random | 0.850 | 0.447 | 0.586 | 0.250 | 0.316 | 0.667 | 0 | 0 | 0.0 | 38 |
| frequency | 0.844 | 1.000 | 0.916 | 1.000 | 0.342 | 0.667 | 0 | 0 | 0.0 | 38 |
| baseline | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 | 0.667 | 0 | 0 | 0.0 | 38 |
| jev | 1.000 | 0.895 | 0.944 | 0.857 | 0.395 | 0.667 | 45 | 842 | 450.0 | 38 |

## Detection gate

Baseline detection F1=1.000 **≥ 0.70** (M1 gate met on this slice).
JEV stub detection F1=0.944 ≥ 0.70 on this slice (cost=450.0 stub units / 45 calls / 842 tokens).

## Proxy retention + review reduction (all rungs)

Proxy-important count on eval: **38** (shared rule labels A∧(B∨C∨D)).

| Backend | Proxy retention | Review reduction | vs all GT fails |
|---------|-----------------|------------------|-----------------|
| random | 31.6% | 66.7% | 60.5% |
| frequency | 34.2% | 66.7% | 60.5% |
| baseline | 39.5% | 66.7% | 60.5% |
| jev | 39.5% | 66.7% | 60.5% |

## Cost–quality framing (JEV vs Baseline)

- Quality delta (JEV − Baseline): detection F1 **-0.056**, proxy retention **+0.000**.
- Cost (JEV stub): **45** calls, **842** tokens, **450.0** stub cost units; Baseline/Random/Frequency local heuristics → **0** stub units.
- Stub does not beat Baseline here; cost–quality hypothesis **not** supported by this stub run (honest negative / tie leaning Baseline).

## What this tells us

Ladder on source=robofac_real: Random F1=0.586, Frequency F1=0.916, Baseline F1=1.000, JEV stub F1=0.944.
RoboFAC success paths often include `dataset_success_cleaned` / `stack_ok` (strong success cue). Many fails lack `fail` path tokens, so Baseline/JEV lean on the development fail prior for cue-less fails. Frequency (majority fail) also scores high when the slice is fail-heavy. Type agreement is limited when QA `failure_type` is null or keywords miss.
Proxy retention at Top-K is a **review-priority** sketch only. Shipping may stay Baseline-only; the ladder comparison remains in-scope.

## Reproduce

```bash
JEV_MODE=stub python scripts/m1_failure_intelligence.py --config configs/m1_failure_intelligence.json
```
