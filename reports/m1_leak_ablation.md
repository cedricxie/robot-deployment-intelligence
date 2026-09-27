# M1 path-label leakage ablation

> **Honesty:** High Baseline / llm_proxy detection F1 on RoboFAC was driven by **success/fail path directory cues** (`dataset_success_cleaned`, `stack_ok`, `fail`, …) embedded in `path_text`. The **ablated** column strips those cues (`features.sanitize` + emptied `fail_path_tokens` / `success_path_tokens`). **Ablated is the fairer detection ceiling**; leaked F1=1 was path leakage, not perception. **review priority ≠ business importance.** llm_proxy / stub ≠ live Jev API (no credits).

## Run metadata

| Field | Value |
|-------|-------|
| Data source | real RoboFAC (test_qa_sim labeled subset) |
| Corpus / seed | n=150, seed=42 |
| Development / eval | 105 / holdout n=45 |
| Top-K | 15 |
| GT fails on eval | 38 |
| Hidden eval | never loaded |
| Jev API | not called |

### Notes

- Loaded 1512 labeled RoboFAC test_qa episodes; sampled n=150 (seed=42).
- Reference = development (Frequency/Baseline/JEV fit + proxy bank). Eval = public_eval-style holdout. hidden_eval never loaded.
- JEV stub + llm_proxy are local heuristics — **not** live Jev API; no credits spent.
- Ablated run is the fairer detection ceiling; leaked F1≈1 was path leakage.

## Side-by-side: leaked vs ablated

| Backend | Leaked F1 | Ablated F1 | Leaked type | Ablated type | Leaked proxy ret | Ablated proxy ret |
|---------|-----------|------------|-------------|--------------|------------------|-------------------|
| random | 0.586 | 0.586 | 0.250 | 0.250 | 0.316 | 0.316 |
| frequency | 0.916 | 0.916 | 1.000 | 1.000 | 0.342 | 0.342 |
| baseline | 1.000 | 0.916 | 0.857 | 0.857 | 0.395 | 0.342 |
| jev_stub | 0.944 | 0.861 | 0.857 | 0.857 | 0.395 | 0.316 |
| llm_proxy | 1.000 | 0.916 | 0.857 | 0.714 | 0.395 | 0.289 |

## Full leaked ladder

| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention |
|---------|-------|-------|--------|------------|-----------------|
| random | 0.850 | 0.447 | 0.586 | 0.250 | 0.316 |
| frequency | 0.844 | 1.000 | 0.916 | 1.000 | 0.342 |
| baseline | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 |
| jev_stub | 1.000 | 0.895 | 0.944 | 0.857 | 0.395 |
| llm_proxy | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 |

## Full ablated ladder

| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention |
|---------|-------|-------|--------|------------|-----------------|
| random | 0.850 | 0.447 | 0.586 | 0.250 | 0.316 |
| frequency | 0.844 | 1.000 | 0.916 | 1.000 | 0.342 |
| baseline | 0.844 | 1.000 | 0.916 | 0.857 | 0.342 |
| jev_stub | 0.829 | 0.895 | 0.861 | 0.857 | 0.316 |
| llm_proxy | 0.844 | 1.000 | 0.916 | 0.714 | 0.289 |

## What this tells us

- **Baseline** detection F1 1.000 → 0.916 after stripping path-label cues (Δ -0.084). Leaked F1≈1 was directory leakage.
- **llm_proxy** detection F1 1.000 → 0.916 (Δ -0.084); proxy retention 0.395 → 0.289.
- **jev_stub** detection F1 0.944 → 0.861.
- Random / Frequency ignore path tokens, so leaked≈ablated (sanity).
- Residual ablated signal may remain from type-semantic folders (`position_offset`, `gripper_error`, …) or instruction text — that is content, not success/fail directory labels. `stack_error` under success trees can still confuse llm_proxy's softer `error` cue.
- Prefer **ablated** metrics when quoting a detection ceiling.

## Reproduce

```bash
python scripts/m1_leak_ablation.py
# or: LEAK_ABLATE=1 on any M1 runner after this change
# optional: --force-synthetic --out reports/m1_leak_ablation.md
```
