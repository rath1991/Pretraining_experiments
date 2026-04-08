import os
import math
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from config import ModelConfig, TrainConfig
from data import get_datasets
from model import GPT


def get_lr(step: int, cfg: TrainConfig) -> float:
    # 1. linear warmup
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    # 2. cosine decay from max_lr → min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + cosine * (cfg.max_lr - cfg.min_lr)


@torch.no_grad()
def eval_val_loss(model: GPT, val_set, tcfg: TrainConfig) -> float:
    model.eval()
    dtype = torch.bfloat16 if tcfg.bf16 else torch.float32
    losses = []
    for _ in range(tcfg.eval_steps):
        x, y = val_set.get_batch(tcfg.batch_size, tcfg.device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def build_optimizer(model: GPT, tcfg: TrainConfig):
    # weight decay only on matrices (weights), not on vectors (layernorm scales)
    decay   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() <  2]
    groups  = [
        {"params": decay,   "weight_decay": tcfg.weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=tcfg.max_lr, betas=(tcfg.beta1, tcfg.beta2))


def train(mcfg: ModelConfig, tcfg: TrainConfig):
    os.makedirs(tcfg.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(tcfg.out_dir, "tb_logs"))

    train_set, val_set = get_datasets(tcfg.data_dir, mcfg.block_size)

    model     = GPT(mcfg).to(tcfg.device)
    optimizer = build_optimizer(model, tcfg)

    dtype = torch.bfloat16 if tcfg.bf16 else torch.float32
    # GradScaler only needed for fp16; bf16 is numerically stable enough without it
    scaler = torch.cuda.amp.GradScaler(enabled=not tcfg.bf16)

    print(f"Parameters : {model.num_params()/1e6:.1f}M")
    print(f"Device     : {torch.cuda.get_device_name(0)}")
    print(f"Precision  : {'bfloat16' if tcfg.bf16 else 'float32'}")
    print(f"Steps      : {tcfg.max_steps} ({tcfg.max_steps * tcfg.batch_size * mcfg.block_size * tcfg.grad_accum / 1e9:.1f}B tokens)")

    model.train()
    for step in range(tcfg.max_steps):
        t0 = time.time()

        lr = get_lr(step, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # --- gradient accumulation ---
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_set.get_batch(tcfg.batch_size, tcfg.device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(x, y)
            (loss / tcfg.grad_accum).backward()
            accum_loss += loss.item() / tcfg.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()

        torch.cuda.synchronize()
        dt          = time.time() - t0
        tokens_seen = tcfg.batch_size * mcfg.block_size * tcfg.grad_accum
        tok_per_sec = tokens_seen / dt

        # --- console log ---
        if step % tcfg.log_every == 0:
            eta_hours = (tcfg.max_steps - step) * dt / 3600
            print(
                f"step {step:5d}/{tcfg.max_steps} | loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | ETA {eta_hours:.1f}h"
            )

        # --- tensorboard ---
        writer.add_scalar("loss/train",  accum_loss,  step)
        writer.add_scalar("lr",          lr,          step)
        writer.add_scalar("grad_norm",   grad_norm,   step)
        writer.add_scalar("tok_per_sec", tok_per_sec, step)

        # --- val eval + checkpoint ---
        if step % tcfg.eval_every == 0:
            val_loss = eval_val_loss(model, val_set, tcfg)
            print(f"  >>> val_loss {val_loss:.4f} @ step {step}")
            writer.add_scalar("loss/val", val_loss, step)
            ckpt = {
                "step":      step,
                "val_loss":  val_loss,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "mcfg":      mcfg,
                "tcfg":      tcfg,
            }
            torch.save(ckpt, os.path.join(tcfg.out_dir, f"ckpt_{step:05d}.pt"))

    writer.close()
