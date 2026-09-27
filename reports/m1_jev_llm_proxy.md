# M1 JEV llm_proxy — Baseline vs role-play systemone

> **Disclaimer:** **review priority ≠ business importance.** `proxy_important` / review-priority is rule-based (A∧(B∨C∨D)), not business criticality. **LLM-proxy ≠ real JEV** — this is a deterministic assistant role-play of systemone `{answers: is_fail.noul, fail_type.choice}` on GT-free `features_to_state` text. **Not** live Jev API, **not** real probs/billing, **not** a substitute for credentialed systemone. Baseline may still win when path cues leak.

## Run metadata

| Field | Value |
|-------|-------|
| Data source | real RoboFAC (test_qa_sim labeled subset) |
| Dataset size (corpus) | 150 |
| Seed | 42 |
| Development (reference) | 105 |
| Eval split | `public_eval` (holdout) n=45 |
| Top-K | 15 |
| JEV mode | `llm_proxy` |
| API keys / JEV credits | none (local role-play) |

### Notes

- Loaded 1512 labeled RoboFAC test_qa episodes; sampled n=150 (seed=42).
- Reference = development (Frequency/Baseline/JEV fit + proxy bank). Eval = public_eval-style holdout. hidden_eval never loaded.
- JEV_MODE=llm_proxy — deterministic role-play of systemone answers. **Not** live Jev API; no credits; not a substitute for real Jev probs.

## Side-by-side (eval holdout)

| Backend | Det P | Det R | Det F1 | Type agree | Proxy retention | Review reduction | Cost calls | Cost tokens | Stub cost units | n_proxy_imp |
|---------|-------|-------|--------|------------|-----------------|------------------|------------|-------------|-----------------|-------------|
| random | 0.850 | 0.447 | 0.586 | 0.250 | 0.316 | 0.667 | 0 | 0 | 0.0 | 38 |
| frequency | 0.844 | 1.000 | 0.916 | 1.000 | 0.342 | 0.667 | 0 | 0 | 0.0 | 38 |
| baseline | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 | 0.667 | 0 | 0 | 0.0 | 38 |
| jev | 1.000 | 1.000 | 1.000 | 0.857 | 0.395 | 0.667 | 45 | 3078 | 0.0 | 38 |

## Stub context (prior M1 report)

From `reports/m1_failure_intelligence.md` on the same RoboFAC holdout-45 (seed=42): stub Det F1=0.944, type agree=0.857, proxy retention=0.395, cost=(45, 842, 450.0 stub units).

## Cost–quality framing (llm_proxy vs Baseline)

- Quality delta (llm_proxy − Baseline): detection F1 **+0.000**, type agree **+0.000**, proxy retention **+0.000**.
- Cost (llm_proxy): **45** calls, **3078** approx tokens (≈ state length/4), **0** stub units / **0** JEV credits; Baseline local heuristic → 0 calls.
- Quality ≈ Baseline with extra call/token accounting — does **not** justify spend; real Jev must beat Baseline on quality or unlock cascade savings.

## Proxy retention + review reduction

Proxy-important count on eval: **38** (shared rule labels A∧(B∨C∨D)).

| Backend | Proxy retention | Review reduction |
|---------|-----------------|------------------|
| frequency | 34.2% | 66.7% |
| baseline | 39.5% | 66.7% |
| jev | 39.5% | 66.7% |

## What this tells us

Holdout n=45 source=robofac_real: Baseline F1=1.000, llm_proxy F1=1.000, Frequency F1=0.916; type agree Baseline=0.857 / llm_proxy=0.857.
llm_proxy parses semantic path folders (`position_offset`, `gripper_error`, …) with soft noul and avoids SafeTask→`safe` false friends that hurt the stub. When success/fail path tokens already leak, Baseline remains hard to beat on detection.
**LLM-proxy ≠ real JEV.** Treat this as a reproducible ceiling sketch for text-only cues, not billing-backed systemone quality.

## Reproduce

```bash
JEV_MODE=llm_proxy python scripts/m1_jev_llm_proxy.py
# optional: --force-synthetic --out reports/m1_jev_llm_proxy.md
```
