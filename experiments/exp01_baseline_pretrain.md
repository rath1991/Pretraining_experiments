# Experiment 01 — Baseline Pretraining Run

**Date:** 2026-04-08
**Status:** Planned
**Goal:** Establish a baseline pretraining run on FineWeb-Edu with GPT-2 Small architecture.
         This is the reference point all future experiments will compare against.

---

## Hardware

**GPU:** NVIDIA GeForce RTX 5060 Ti (Blackwell GB206)

### Why this matters for this run

| Feature | Detail | Impact |
|---|---|---|
| Architecture | Blackwell (5th gen tensor cores) | Native BF16 throughput ~2-3x faster than Ampere |
| VRAM | 16 GB GDDR7 | Fits 124M model + batch comfortably without offloading |
| Memory bandwidth | ~672 GB/s (GDDR7) | Reduces memory-bound bottleneck vs GDDR6 |
| BF16 tensor cores | 5th gen | Key reason bf16 gives ~2.4x speedup here (13k → 30k tok/s) |

Blackwell's BF16 advantage over older GPUs: on Ampere (A100), BF16 gave ~2x over FP32.
On Blackwell, the tensor core pipeline is more tightly integrated — BF16 ops
are effectively the native precision, making the speedup more pronounced.

---

## Dataset

| Setting | Value |
|---|---|
| Source | FineWeb-Edu (HuggingFace `sample-10BT`) |
| Format | Flat uint16 binary (`train.bin`, `val.bin`) |
| Tokenizer | GPT-2 BPE (tiktoken), vocab size 50,257 |
| Total available | ~10B tokens (~20 GB on disk at 2 bytes/token) |
| Tokens used (this run) | 1B (10% of available) |
| Val split | 0.5% (same as GPT-2 paper) |

**Why FineWeb-Edu:** Filtered subset of Common Crawl scored for educational quality.
Higher signal-to-noise than raw web text. Good baseline corpus before experimenting
with data mixing or quality filtering.

**Why 1B tokens:** At ~30k tok/s (bf16), 1B tokens ≈ 9 hours on the 5060 Ti.
Enough for the loss curve to show meaningful structure and for val loss to stabilize —
without committing to a multi-day run for every experiment.

---

## Model Configuration

```
n_layer    = 12       # transformer depth
n_head     = 12       # attention heads
n_embd     = 768      # embedding / hidden dimension
vocab_size = 50257    # GPT-2 tokenizer vocabulary
block_size = 1024     # context length (tokens per sample)
bias       = False    # no bias in Linear layers or LayerNorm
dropout    = 0.0      # no dropout during pretraining
```

**Effective parameter count:** 85.7M (124M gross minus 38.6M shared embedding weight)

### Rationale

- **12/12/768** — GPT-2 Small architecture. Well-studied, reproducible reference numbers
  exist (GPT-2 paper, nanoGPT). Loss curves have known shape, making anomalies easy to spot.

- **bias=False** — Modern best practice (PaLM, LLaMA). Biases add parameters without
  meaningful capacity gain; removing them slightly simplifies the loss landscape.

- **dropout=0.0** — Pretraining on large data does not need dropout. Regularization
  comes from the data distribution itself. Dropout would slow convergence.

- **Weight tying** — The token embedding matrix (50257×768) is shared with the output
  projection. Rationale: a token's representation in input space should be geometrically
  similar to its representation in output space. Also saves ~38M parameters.

---

## Training Configuration

```
batch_size   = 8       # micro-batch: sequences per single forward pass
grad_accum   = 64      # gradient accumulation steps before one optimizer update
# effective batch = 8 × 64 × 1024 = 524,288 tokens per optimizer step

max_steps    = 1907    # 1,000,000,000 / 524,288 ≈ 1907
warmup_steps = 50      # ~26M tokens — linear LR warmup phase

max_lr       = 6e-4    # peak learning rate
min_lr       = 6e-5    # cosine decay floor (10% of max_lr)

beta1        = 0.9     # Adam momentum — smoothed gradient direction
beta2        = 0.95    # Adam velocity — faster magnitude adaptation (vs default 0.999)
weight_decay = 0.1     # applied only to weight matrices, not layernorm/biases
grad_clip    = 1.0     # hard cap on gradient norm
```

### Rationale

**Effective batch size (~524K tokens)**
Each optimizer step sees ~524K tokens before updating weights. This is the
Chinchilla/nanoGPT standard. Large enough that the gradient estimate is low-noise;
small enough to take ~1907 meaningful update steps over 1B tokens.
Too small a batch → noisy gradients, unstable training. Too large → few updates, slow convergence.

**Gradient accumulation (64 micro-steps)**
Running batch_size=512 directly would require ~2GB of activations simultaneously.
Instead, 64 micro-steps of batch_size=8 accumulate gradients before one optimizer.step().
Mathematically identical to a batch of 512. `loss / grad_accum` before `.backward()`
ensures gradients average (not sum) across micro-steps.

**Warmup (50 steps / ~26M tokens)**
At step 0, weights are random and gradients are large and incoherent.
Jumping straight to lr=6e-4 causes the optimizer to overshoot badly.
Warmup linearly ramps lr from ~0 → 6e-4 over 50 steps, letting gradient
directions stabilize before taking large steps. ~2.6% of total steps.

**Cosine decay (max_lr → min_lr)**
Early training: model is far from optimum, needs large steps to traverse the loss landscape.
Late training: model is near a good basin, large steps would overshoot.
Cosine schedule reduces lr smoothly. Decaying to min_lr=6e-5 (not 0) keeps
fine-grained learning active at the end without fully stopping.

**beta2=0.95 (not default 0.999)**
Adam's velocity term tracks squared gradient magnitudes per parameter.
beta2=0.999 has a memory of ~1000 steps — adapts very slowly to gradient scale changes.
beta2=0.95 has a memory of ~20 steps — reacts quickly as gradient statistics shift
during training. Language model gradients are non-stationary; 0.95 works better.

**grad_clip=1.0**
Prevents any single bad batch from causing a catastrophic weight update.
If grad norm > 1.0, all gradients are scaled down proportionally.
Critical safety net especially early in training when gradients are large.

---

## Precision: BF16 Autocast

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    _, loss = model(x, y)
```

**What autocast does:**
PyTorch automatically selects bf16 for ops that benefit from it (matmuls, attention)
and keeps fp32 for ops that need precision (softmax, layernorm, loss computation).
The model parameters stay in fp32 — only the forward pass arithmetic is in bf16.
No manual casting needed; autocast handles op-level decisions.

**Why BF16 and not FP16:**
| | FP16 | BF16 |
|---|---|---|
| Exponent bits | 5 | 8 (same as FP32) |
| Mantissa bits | 10 | 7 |
| Dynamic range | Limited — needs GradScaler | Same as FP32 — no GradScaler needed |
| Risk | Overflow/underflow in loss | Essentially none |

BF16 trades mantissa precision for dynamic range. For neural network weights and
gradients, dynamic range matters more than mantissa precision — BF16 is numerically
safe without a GradScaler, whereas FP16 frequently overflows during LM pretraining.

**Measured speedup on RTX 5060 Ti:**
- FP32 baseline: ~13,000 tok/s
- BF16 autocast: ~30,500 tok/s
- Speedup: **2.4×**

**Estimated training time:** 1B tokens / 30,500 tok/s ≈ **9.1 hours**

---

## Monitoring

| Metric | Logged to TensorBoard | Frequency |
|---|---|---|
| `loss/train` | Yes | Every step |
| `loss/val` | Yes | Every 100 steps (~19 evals) |
| `lr` | Yes | Every step |
| `grad_norm` | Yes | Every step |
| `tok_per_sec` | Yes | Every step |

**Viewing remotely via SSH:**
```bash
# On desktop — start TensorBoard
uv run tensorboard --logdir out/tb_logs --host 0.0.0.0 --port 6006

# On SSH client — tunnel
ssh -L 6006:localhost:6006 username@desktop-ip

# Then open in browser
http://localhost:6006
```

**Expected loss curve shape:**
```
Steps   0–50   (warmup):  loss drops steeply  ~10.8 → ~7.0
Steps  50–500  (early):   loss drops           ~7.0 → ~4.0
Steps 500–1500 (mid):     loss slows           ~4.0 → ~3.2
Steps 1500–1907 (end):    near flat            ~3.2 → ~3.0
```
Val loss should track within 0.05–0.1 of train loss throughout.
A gap > 0.2 or val loss rising while train loss falls = investigate.

---

## Checkpoints

Saved to `out/ckpt_NNNNN.pt` every 100 steps. Each checkpoint contains:
- `model` — state dict
- `optimizer` — state dict (for resuming)
- `step`, `val_loss`, `mcfg`, `tcfg`

---

## What to look for in this run

1. **Loss curve shape** — should follow the expected trajectory above
2. **Train/val gap** — should stay small (< 0.1) throughout; 1B tokens is not enough to overfit a 124M model
3. **grad_norm** — should start high (~5), drop to ~1–2 by step 200, stay stable
4. **tok/s stability** — should stay near 30k throughout; a drop signals memory pressure
5. **Final val loss** — target ~3.0–3.2 for 1B tokens on FineWeb-Edu with GPT-2 Small
