# Split manifests

Deterministic seed-`42` 60/20/20 episode-id lists:

| File | Role |
|------|------|
| `development.jsonl` | Pipeline + improvement-loop tuning |
| `public_eval.jsonl` | Harness / milestone reports / candidate selection |
| `hidden_eval.jsonl` | **Harness final scoring only** — improvement-loop helpers must not load these labels |

Regenerate:

```bash
python scripts/write_splits.py
```
