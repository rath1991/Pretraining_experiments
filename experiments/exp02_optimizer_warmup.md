# Exp 02 — Optimizer + Warmup Stability

**Date:** 2026-04-12
**Status:** Complete — all variants run
**Plot:** `out/results/exp02_overview.png`

---

## Variants

| Variant | Optimizer | Warmup    |
|---------|-----------|-----------|
| A       | Adam      | None      |
| B       | Adam      | 50 steps  |
| C       | AdamW     | None      |
| D       | AdamW     | 50 steps  ← baseline |

Everything else identical to Exp 01 config.

---

## Framework

```bash
uv run python run.py --variant A   # etc.
```

- Each variant writes to `out/exp02_varX/` (checkpoints + tb_logs)
- `EXP02_VARIANTS` in `run.py` applies overrides via `setattr` on `TrainConfig`
- `plot_exp02.py` overlays all 4 variants on a single 4×2 subplot figure

---

## Hypotheses

- A/C (no warmup): grad_norm spike at step 0, steeper initial drop
- A/B (Adam): param_norm grows faster — no weight decay to damp it
- D: should reproduce Exp 01 val loss closely
- update_ratio: Adam variants likely run higher due to unbounded weight growth

---

## Plot

![Exp 02 Overview](../out/results/exp02_overview.png)

---

## Findings

**Train and val loss:** both no-warmup cases (A and C) track together strikingly higher than their warmup counterparts throughout — warmup is the dominant variable here, not the choice of optimizer. the mechanism is the cold-start problem in moment estimates: `v_t` initializes near zero, so the effective step size `lr / sqrt(v_t)` is enormous in the first ~20 steps. with `β2=0.95` the memory is ~20 steps, so 50 warmup steps is well-calibrated — it buys time for `v_t` to converge before full LR is applied. weight decay produces no distinguishable difference in the loss curves.

**Learning rate:** all four variants follow the same cosine schedule shape. no meaningful divergence — expected since warmup only affects the first 50 of 1907 steps.

**Gradient norm:** the clearest optimizer-level signal. AdamW (C, D) shows notably better damping of grad_norm fluctuations than Adam (A, B), and this effect becomes pronounced after ~750 steps. the timing aligns with weight decay accumulating enough regularization pressure to visibly constrain parameter growth — early in training the decay term is small relative to the gradient signal, but by ~750 steps the separation is clear. Adam no-warmup (A) is the most erratic throughout.

**Parameter norm:** no-warmup variants (A, C) end up with smaller L2 parameter norms than their warmup counterparts as steps elapse — counterintuitive given that no-warmup applies larger steps early. this doesn't have a clean first-principles explanation from the current data. worth flagging and revisiting with per-layer norm tracking in a later experiment rather than rationalising it now.

**Update-to-weight ratio:** all four converge to the healthy ~1e-3 zone. Adam no-warmup (A) arrives there fastest, followed by AdamW no-warmup (C). this is a scale artifact — at early steps weight norms are small (near init), so the ratio is naturally high. the faster "convergence" reflects large step magnitudes against a small denominator, not genuine optimizer health.

**Activation mean (ln_final):** Adam with warmup (B) is the most stable — mean stays near zero throughout. AdamW no-warmup (C) drifts upward after ~1000 steps, a signal of residual stream bias accumulating across blocks. the magnitude is small but the trend is concerning — this kind of drift computes across blocks and can make training increasingly unstable in longer runs or deeper models.

**Takeaway:** warmup is load-bearing. optimizer choice is secondary at this scale — AdamW provides better grad_norm stability but this doesn't translate to better loss on 1B tokens with a data-rich corpus. weight decay is a slow-burn effect: irrelevant to loss dynamics, visible only in gradient stability and only after several hundred steps.

---

## Notes / Brainstorm

### Activation mean drift — AdamW no-warmup (C)

the `ln_final` activation mean in variant C starts drifting upward after ~1000 steps while all other variants stay near zero. this is worth unpacking in detail.

**what is being measured.** `ln_final` sits after the last transformer block and before `lm_head`. its activation mean reflects the mean of the residual stream at the end of the forward pass — the net aggregate of the token embedding, positional embedding, and 12 blocks of residual additions. LayerNorm normalises activations per-position across the embedding dimension before each block, so it does suppress local drift — but it cannot prevent the *accumulated* mean from drifting if each block consistently adds a small positive bias to the stream.

**why AdamW no-warmup specifically.** the combination matters. without warmup, the first ~20 steps take enormous effective steps (cold `v_t` → large `lr / sqrt(v_t)`). these early steps push weight matrices into asymmetric configurations before the optimizer has reliable curvature estimates. weight decay in AdamW then shrinks all weights proportionally toward zero — but if certain embedding directions have already been pushed systematically positive by those early noisy steps, decay shrinks them slower than new gradient signal pushes them further. Adam no-warmup (A) doesn't have decoupled weight decay, so this reinforcement mechanism doesn't operate the same way.

**why the drift appears late (~1000 steps) not early.** two reasons. first, LayerNorm suppresses early-stage drift effectively — the bias per block is small and the normalisation is strong. second, the effect is cumulative: each block contributes a small mean bias to the residual stream. that bias only becomes visible at `ln_final` once enough steps have elapsed for the per-block biases to structurally solidify and accumulate across 12 layers.

**why this is a problem at scale.** in a 12-layer model the observed magnitude is small. but the concern is the *trend*, not the magnitude. in a 24L or 48L model the accumulation is proportionally larger. in longer runs, a drifting residual stream mean shifts the effective operating point of every LayerNorm downstream — each normalisation is now re-centering a distribution that is itself drifting, which can cause gradient flow through the residual stream to become progressively less well-conditioned.

**what would confirm this.** per-block activation mean tracking (not just `ln_final`) would show whether the drift is uniform across blocks or concentrated in a few layers. if it is front-loaded (early blocks drifting more), that points to the early chaotic steps as the origin. if it is back-loaded, it suggests compounding through the residual additions is the primary driver.

**open question.** variant A (Adam no-warmup) does not show the same drift despite also having chaotic early steps. the decoupled weight decay in AdamW appears to be the distinguishing factor — but the exact mechanism by which it reinforces rather than corrects the asymmetry is not fully clear. worth revisiting in Exp 04 (Post-LN vs Pre-LN) where the normalisation placement changes the residual stream dynamics substantially.
