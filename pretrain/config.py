from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layer:    int   = 12
    n_head:     int   = 12
    n_embd:     int   = 768
    vocab_size: int   = 50257    # GPT-2 tokenizer
    block_size: int   = 1024     # context length
    bias:       bool  = False    # no bias in linears or layernorms
    dropout:    float = 0.0      # pretraining: no dropout

@dataclass
class TrainConfig:
    # batch
    batch_size: int = 8          # sequences per forward pass (micro-batch)
    grad_accum: int = 64         # effective batch = 8 × 64 × 1024 ≈ 524K tokens

    # learning rate
    max_lr:       float = 6e-4
    min_lr:       float = 6e-5   # 10% of max_lr — cosine decay floor
    warmup_steps: int   = 50     # ~26M tokens; ~2.6% of 1B — enough to stabilize gradients

    # training duration — 1B tokens / 524K tokens per step
    max_steps: int = 1907        # 1,000,000,000 / 524,288 ≈ 1907

    # optimizer
    beta1:        float = 0.9
    beta2:        float = 0.95   # faster magnitude adaptation than default 0.999
    weight_decay: float = 0.1
    grad_clip:    float = 1.0

    # precision — bf16 uses tensor cores, ~3-5x faster than float32 on Blackwell
    bf16: bool = True

    # optimizer
    optimizer: str = "adamw"     # "adamw" or "adam"

    # logging & eval
    log_every:  int = 10         # print train loss every N steps
    eval_every: int = 100        # val loss every N steps (~19 evals across 1907 steps)
    eval_steps: int = 20         # val batches to average over

    # paths (relative to pretrain/ directory)
    data_dir: str = "../data/fineweb_edu"
    out_dir:  str = "../out"

    # device
    device: str = "cuda"
