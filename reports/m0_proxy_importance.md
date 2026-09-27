# M0 proxy review-priority report

> **Disclaimer:** proxy importance is rule-based; **review priority ≠ business importance**; not business-critical GT.

## Formula (frozen)

```text
proxy_important ⇔ A ∧ (B ∨ C ∨ D)
A = fail (success=false)
B = rare failure_type (count ≤ rare_freq_quantile of per-type counts)
C = high-frequency recurrence (count ≥ high_freq_min_count)
D = novelty vs Failure Bank (unseen type → novelty_score=1.0)
E/F = ranking bonuses only (stubs; not admission)
```

## Config

```json
{
  "rare_freq_quantile": 0.25,
  "high_freq_min_count": 3,
  "novelty_min_score": 1.0,
  "top_k": 10
}
```

Reference frequency (development failures): counts={'step_omission': 3, 'position_deviation': 2, 'timing_error': 1}, rare_count_cutoff=1

## Split summaries

| Split | n | proxy_important | prevalence | A | B | C | D |
|-------|---|-----------------|------------|---|---|---|---|
| development | 8 | 4 | 50.00% | 6 | 1 | 3 | 0 |
| public_eval | 5 | 1 | 20.00% | 4 | 0 | 0 | 1 |

## Top-K proxy-important

### development

- `fixture-sim-fail-omit-001` hits=['A', 'C'] bucket=common novelty=0.0
- `synthetic-proxy-0004` hits=['A', 'C'] bucket=common novelty=0.0
- `synthetic-proxy-0005` hits=['A', 'C'] bucket=common novelty=0.0
- `synthetic-proxy-0007` hits=['A', 'B'] bucket=rare novelty=0.0

### public_eval

- `synthetic-proxy-0006` hits=['A', 'D'] bucket=None novelty=1.0

## Notes

- Frequency + Failure Bank seeded from **development**-style reference only.
- `hidden_eval` is never loaded (harness final scoring only).
- Cold-start: types absent from the bank are novel (D).
