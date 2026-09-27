# M1 micro — Baseline vs real JEV (n≤4)

> ## HUGE DISCLAIMER
>
> **n is tiny (≤4).** Free-tier credits are scarce (~single-digit remaining). This is a **plumbing / smoke** check that the real adapter maps `noul`/`choice`/`usage` correctly — **not** a ship gate, **not** evidence that JEV beats Baseline, **not** a statistically meaningful F1 comparison.
>
> Do not cite these F1 numbers as product claims. Re-run on a paid quota with a proper eval split before any cost–quality decision.

## Run metadata

| Field | Value |
|-------|-------|
| Data source | fixtures + synthetic (path cues) |
| Seed | 42 |
| Eval n | 4 (max_real_calls=4) |
| JEV mode | `real` |
| Endpoint | `https://jev-agent.com/api/v1/systemone` |
| Cost calls | 4 |
| Cost tokens (input) | 2135 |
| Stub cost units | 0 (real) |
| Quota (API) | remaining=0, used=5, limit=5, month=2026-09 |

### Notes

- Forced fixture/synthetic corpus.
- Eval holdout slice n=4 (max_real_calls=4); development reference n=28 for Baseline fit only.
- JEV real calls are GT-free (path/instruction/meta state text only). Free-tier credits are tiny — this micro run is not a ship gate.

## Metrics (failure detection)

| Backend | Det P | Det R | Det F1 | Type agree | Calls | Tokens |
|---------|-------|-------|--------|------------|-------|--------|
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| jev (real) | 1.000 | 1.000 | 1.000 | 1.000 | 4 | 2135 |

## Per-episode rows

| episode_id | GT success | GT type | Baseline pred | Base p_fail | Base type | JEV pred | JEV p_fail (noul) | JEV type | JEV tokens |
|------------|------------|---------|---------------|-------------|-----------|----------|-------------------|----------|------------|
| `synth-fail-0088` | False | timing_error | fail | 0.850 | timing_error | fail | 0.950 | timing_error | 534 |
| `synth-fail-0039` | False | step_omission | fail | 0.850 | step_omission | fail | 0.960 | step_omission | 534 |
| `synth-fail-0082` | False | position_deviation | fail | 0.850 | position_deviation | fail | 0.930 | position_deviation | 533 |
| `synth-fail-0023` | False | orientation_deviation | fail | 0.850 | orientation_deviation | fail | 0.950 | orientation_deviation | 534 |

## Reproduce

```bash
# Key must already be in the environment — never echo it.
export JEV_MODE=real
python scripts/m1_jev_real_micro.py --max-real-calls 4
```
