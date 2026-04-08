# GPT-2 Small Pretraining — Experiment List

**Phase:** Phase 1 (Apr 8–16)
**Model:** GPT-2 Small (85.7M params, 12L/12H/768D)
**Baseline:** Exp 01 — FineWeb-Edu, 1B tokens, bf16, ~9h on RTX 5060 Ti
**Token budget per experiment:** 1B (keeps each run ~9h, comparable across experiments)

---

## Exp 01 — Baseline Pretraining
**Status:** Ready to run
**File:** `exp01_baseline_pretrain.md`

Standard config. FineWeb-Edu, AdamW, warmup=50, cosine LR, Pre-LN, 524K effective batch.
This is the reference — every other experiment compares val loss curve against this.

---

## Exp 02 — Optimizer + Warmup Stability
**From original #1 — kept as-is**

| Variant | Optimizer | Warmup |
|---|---|---|
| A | Adam | None |
| B | Adam | 50 steps |
| C | AdamW | None |
| D (baseline) | AdamW | 50 steps |

**Goal:** Isolate optimizer stability during cold start. Adam has no weight decay regularization —
does it destabilize early training? Does skipping warmup cause loss spikes at step 0?

**Track:** Early loss spikes (steps 0–100), gradient norm trajectory, final val loss.

---

## Exp 03 — Initialization + Depth Stability
**From original #2 — kept as-is**

| Variant | Init scheme |
|---|---|
| A | Standard (std=0.02 everywhere) |
| B | GPT-2 scaled (residual projections scaled by 1/√(2 × n_layer)) |

GPT-2 scaled init: at initialization, each residual block contributes equal variance to
the residual stream. Without it, deeper layers see growing signal variance — gradients
in early layers become small (under-training) or large (instability).

**Goal:** Does scaled init produce more stable gradient magnitudes across all 12 layers?

**Track:** Per-block activation std, per-block gradient norm (hook into each Block.forward),
loss curve smoothness.

---

## Exp 04 — Normalization Strategy
**From original #3 — kept as-is**

| Variant | Norm placement |
|---|---|
| A | Pre-LN (baseline: norm before attn/MLP) |
| B | Post-LN (original transformer: norm after residual add) |

Both variants run with identical LR sweep (3e-4, 6e-4, 1e-3) to find max stable LR.

**Goal:** Pre-LN is empirically more stable — does that hold here? What is the divergence
point for Post-LN, and does Pre-LN allow a higher max LR?

**Track:** Divergence point per LR, loss smoothness, gradient norm variance.

---

## Exp 05 — Batch Size vs LR Coupling
**From original #4 — kept as-is**

| Variant | Effective batch | LR (linear scaling rule) |
|---|---|---|
| A | 131K tokens (grad_accum=16) | 1.5e-4 |
| B | 262K tokens (grad_accum=32) | 3e-4 |
| C (baseline) | 524K tokens (grad_accum=64) | 6e-4 |
| D | 1M tokens (grad_accum=128) | 1.2e-3 |

Linear scaling rule: LR scales proportionally with batch size (Goyal et al 2017).

**Goal:** Does the linear scaling rule hold at this token budget and model size?
Small batch = noisy gradients but more updates. Large batch = clean gradient but
fewer steps. Where is the sweet spot on this loss landscape?

**Track:** Loss variance across steps, convergence speed (loss vs tokens seen), final val loss.

---

## Exp 06 — Context Length vs Optimization
**From original #5 — kept as-is**

| Variant | block_size | Steps (same 1B token budget) |
|---|---|---|
| A | 128 | 15,258 |
| B | 256 | 7,629 |
| C | 512 | 3,815 |
| D (baseline) | 1024 | 1,907 |

Same total tokens — shorter context = more optimizer steps, more gradient updates,
but each sample carries less long-range context signal.

**Goal:** Does shorter context with more steps learn faster early, or does the reduced
long-range signal hurt? How does perplexity scale with context length at fixed compute?

**Track:** Perplexity vs tokens seen, loss curve shape, stability across context lengths.

---

## Exp 07 — Data Quality Sensitivity
**From original #7 — kept as-is**

| Variant | Dataset | Character |
|---|---|---|
| A (baseline) | FineWeb-Edu | Curated educational web |
| B | Pile: Wikipedia | Factual, encyclopedic |
| C | Pile: Books1 | Long-form narrative |
| D | Pile: HackerNews | Noisy, short-form web |

Data already downloaded and tokenized (see `download_data.py`).

**Goal:** How does corpus entropy and quality affect convergence speed and stability?
FineWeb-Edu → HackerNews is a controlled clean→noisy gradient.

**Track:** Convergence speed, final val loss, loss curve smoothness, generation coherence.

---

## Dropped from original list

| Original | Reason dropped |
|---|---|
| #6 Positional encoding (learned vs sinusoidal) | Sinusoidal is obsolete — RoPE is the relevant direction (Phase 2). Learned vs sinusoidal is well-studied with no new stability insight to extract. |
| #8 Checkpoint + decoding | Not a training experiment — this is an evaluation protocol. Should be applied to all experiments, not run as a standalone. |

---

## Evaluation Protocol (applied to all experiments)

From original #8 — folded into standard eval practice:
- Always save checkpoints at early / mid / late (steps ~20%, 60%, 100% of run)
- At each checkpoint, run greedy + temperature (0.7, 1.0) + top-k (40) decoding
- Track: output coherence, repetition rate, diversity
- Goal: separate model capability (checkpoint quality) from decoding behavior

---

## Priority Order

1. Exp 02 — Optimizer + Warmup (most fundamental, run first)
2. Exp 03 — Init scaling (biggest architecture impact on stability)
3. Exp 04 — Pre-LN vs Post-LN (directly relevant to stability analysis thesis)
4. Exp 05 — Batch × LR (optimization theory)
5. Exp 06 — Context length (compute tradeoff)
6. Exp 07 — Data quality (corpus sensitivity)
