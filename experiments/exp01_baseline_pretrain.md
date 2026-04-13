# Exp 01 — Baseline Pretraining

**Date:** 2026-04-08
**Status:** Complete
**Plot:** `out/results/exp01_overview.png`

---

## Config

| | |
|---|---|
| Model | GPT-2 Small — 12L / 12H / 768D |
| Params | 85.7M (weight-tied) |
| Data | FineWeb-Edu, 1B tokens |
| Steps | 1907 |
| Effective batch | 524K tokens (8 × 64 × 1024) |
| LR | 6e-4 → 6e-5 cosine, warmup 50 steps |
| Optimizer | AdamW β1=0.9 β2=0.95 wd=0.1 clip=1.0 |
| Precision | BF16 autocast |
| Throughput | ~30,500 tok/s → ~9h |

---

## What to watch

- Loss shape: steep drop in warmup → steady decay → near-flat end
- Train/val gap: should stay < 0.1 (can't overfit 124M on 1B tokens)
- grad_norm: starts high, settles to ~1–2 by step ~200
- grad_clipped: expected in warmup, should be rare after step 50
- param_norm: slow monotonic rise
- update_ratio: target ~1e-3 at peak LR, drifts down with cosine decay
- activation/mean: stay within ±0.1
- activation/std: stay within 0.8–1.2

---

## Plot

![Exp 01 Overview](../out/results/exp01_overview.png)

---

## Findings

train and val loss both decreased in a healthy manner — started near 11 which makes sense given random init over vocab 50257 (ln(50257) ≈ 10.82), came down to around 4 by end of 1B tokens. the two track closely throughout, no sign of overfitting. that said, ~4 is a bit above the ~3.0–3.2 target, worth keeping in mind when comparing against other variants.

lr warmed up steeply from ~0 to 6e-4 in 50 steps then gradually underwent cosine decay all the way to 6e-5 at the end — shape looks exactly as expected.

grad norm showed spikes during warmup. random init puts the model in a high-curvature region of the loss landscape — gradient directions are large and incoherent early on, initialization state, erratic first few updates etc. warmup keeps the step size small while this is happening, so it's the mitigation not the cause. post-warmup the norm stabilises, implying stable training and healthy gradient flow through the rest of the run.

---

## Notes / Brainstorm
