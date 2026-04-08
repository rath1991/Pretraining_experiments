# CLAUDE.md — LLM From Scratch Research Project

## Who I am working with

Researcher/architect. Has completed Vizuara LLM from scratch (30 lectures).
Understands transformer internals well — does not need basics explained.
Designs experiments himself. Claude's role: critique, implement, debug.
Hardware: RTX 5060 Ti 16GB, 64GB RAM, AMD Ryzen 7700.

---

## Collaboration rules

- **Discuss before writing.** When a new topic starts, enter discussion mode first.
  Ask clarifying questions. Only write plan/code once direction is confirmed.
- **Experiments are user-designed.** Do not originate experiments.
  Critique honestly — what's strong, what's weak, what's missing.
- **Plan before code.** Get explicit approval on approach before implementing.
- **Every line is reviewed.** Write tight, purposeful code. No over-engineering,
  no speculative helpers, no unnecessary comments.

---

## Project structure

```
llm-env/
  pretrain/          # training codebase
    config.py        # ModelConfig + TrainConfig dataclasses
    data.py          # BinDataset — memmapped uint16 binary reader
    model.py         # GPT-2 Small: CausalSelfAttention, MLP, Block, GPT
    train.py         # training loop, val eval, TensorBoard logging
    run.py           # entry point: python run.py
  data/
    fineweb_edu/     # train.bin + val.bin (~20GB, 10B tokens, uint16)
    pile/            # wikipedia / books / hackernews splits
  experiments/
    experiment_list.md          # all planned experiments
    exp01_baseline_pretrain.md  # full notes for Exp 01
  out/               # checkpoints + TensorBoard logs
    tb_logs/         # tensorboard --logdir out/tb_logs
```

---

## Current config (pretrain/config.py)

```
Model:   GPT-2 Small — 12L / 12H / 768D / block_size=1024 / no bias / no dropout
Params:  85.7M (124M gross minus tied embedding weight)

Tokens:  1B (FineWeb-Edu)
Steps:   1907  (1B / 524K tokens per step)
Warmup:  50 steps (~26M tokens)
LR:      6e-4 → 6e-5 cosine decay
Batch:   8 micro × 64 accum × 1024 ctx = 524,288 tokens effective
AdamW:   β1=0.9  β2=0.95  wd=0.1  clip=1.0
BF16:    True
```

**Throughput:** ~30,500 tok/s on RTX 5060 Ti with bf16 autocast → ~9h per 1B token run.

---

## Key technical decisions and rationale

**BF16 autocast**
`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass.
Parameters stay float32. BF16 selected automatically for matmuls/attention.
Chose BF16 over FP16: same dynamic range as FP32, no GradScaler needed.
Measured 2.4× speedup (13k → 30.5k tok/s) on Blackwell tensor cores.

**Gradient accumulation**
64 micro-steps of batch=8 before one optimizer.step() = 524K token effective batch.
`loss / grad_accum` before `.backward()` ensures gradients average not sum.
Mathematically identical to batch=512 but uses 1/64th the VRAM.

**β2=0.95 not 0.999**
Memory of ~20 steps vs ~1000 steps. Language model gradient statistics shift
rapidly during training — 0.95 adapts faster. Slightly better final loss.

**Weight tying**
`wte.weight = lm_head.weight` — token embedding matrix shared with output projection.
num_params() subtracts the tied tensor to avoid double-counting (38.6M params).

**Pre-LN (norm before attention/MLP)**
More stable than Post-LN. Allows higher learning rates without divergence.
Direct relevance to stability analysis research goal.

**No bias, no dropout**
Bias: no capacity gain, slightly cleaner loss landscape.
Dropout: pretraining on 1B tokens from a large corpus provides sufficient regularization.

---

## Monitoring setup

TensorBoard scalars per step: `loss/train`, `loss/val`, `lr`, `grad_norm`, `tok_per_sec`
Val eval every 100 steps (20 batches averaged) → 19 evaluations across 1907-step run.
Checkpoints saved at every val eval to `out/ckpt_NNNNN.pt`.

**Remote viewing via SSH port forward:**
```bash
# Desktop: tensorboard --logdir out/tb_logs --host 0.0.0.0 --port 6006
# SSH client: ssh -L 6006:localhost:6006 user@desktop-ip
# Browser: http://localhost:6006
```

---

## Research phases

| Phase | Period | Focus |
|---|---|---|
| Phase 1 | Apr 8–16 | GPT-2 Small pretraining experiments (Exp 01–07) |
| Travel | Apr 17–28 | Reading: GPT-2, LLaMA, DeepSeek-V2/R1, DPO papers |
| Phase 2 | Apr 29 – May 9 | DeepSeek components: RMSNorm, SwiGLU, RoPE, GQA |
| Phase 3 | May 12–23 | SFT + DPO |
| Phase 4 | May 26 – Jun 15 | GRPO/RLVR from scratch — core research |

**Research thesis:** Stability analysis of Small Language Models with and without RLVR/RL tuning.
Output: running research journal on GitHub.io.

---

## Experiments — Phase 1 summary

See `experiments/experiment_list.md` for full details.

| Exp | Topic | Key variable |
|---|---|---|
| 01 | Baseline | Reference run |
| 02 | Optimizer + Warmup | Adam vs AdamW, warmup on/off |
| 03 | Init + Depth stability | Standard vs GPT-2 scaled init |
| 04 | Normalization | Pre-LN vs Post-LN |
| 05 | Batch × LR coupling | Effective batch 131K→1M + linear LR scaling |
| 06 | Context length | block_size 128→1024, same token budget |
| 07 | Data quality | FineWeb-Edu → Wikipedia → Books → HackerNews |

Dropped: sinusoidal positional encoding (obsolete, RoPE is Phase 2 direction).
Evaluation protocol (checkpoints + decoding) applied across all experiments.
