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
def eval_val_loss(model: GPT, val_set, tcfg: TrainConfig):
    model.eval()
    dtype = torch.bfloat16 if tcfg.bf16 else torch.float32

    # online activation stats via hook on ln_f (final LayerNorm output)
    stats = {"sum": 0.0, "sq_sum": 0.0, "count": 0}
    def _hook(module, input, output):
        o = output.detach().float()
        stats["sum"]    += o.sum().item()
        stats["sq_sum"] += (o ** 2).sum().item()
        stats["count"]  += o.numel()
    hook = model.transformer.ln_f.register_forward_hook(_hook)

    losses = []
    for _ in range(tcfg.eval_steps):
        x, y = val_set.get_batch(tcfg.batch_size, tcfg.device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    hook.remove()

    act_mean = stats["sum"] / stats["count"]
    act_var  = stats["sq_sum"] / stats["count"] - act_mean ** 2
    act_std  = act_var ** 0.5

    model.train()
    return sum(losses) / len(losses), act_mean, act_std


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

        # snapshot params before step for update-to-weight ratio (eval steps only)
        at_eval = (step % tcfg.eval_every == 0)
        if at_eval:
            with torch.no_grad():
                param_snap = [p.data.clone() for p in model.parameters()]

        grad_norm   = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        grad_clipped = int(grad_norm.item() > tcfg.grad_clip)
        optimizer.step()

        torch.cuda.synchronize()
        dt          = time.time() - t0
        tokens_seen = tcfg.batch_size * mcfg.block_size * tcfg.grad_accum
        tok_per_sec = tokens_seen / dt

        # param norm — every step
        with torch.no_grad():
            param_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

        # update-to-weight ratio — eval steps only (param clone is expensive)
        if at_eval:
            with torch.no_grad():
                upd_norm  = sum((p.data - s).norm().item() ** 2
                                for p, s in zip(model.parameters(), param_snap)) ** 0.5
                upd_ratio = upd_norm / (param_norm + 1e-8)
            del param_snap

        # --- console log ---
        if step % tcfg.log_every == 0:
            eta_hours = (tcfg.max_steps - step) * dt / 3600
            print(
                f"step {step:5d}/{tcfg.max_steps} | loss {accum_loss:.4f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | ETA {eta_hours:.1f}h"
            )

        # --- tensorboard (every step) ---
        writer.add_scalar("loss/train",  accum_loss,   step)
        writer.add_scalar("lr",          lr,           step)
        writer.add_scalar("grad_norm",   grad_norm,    step)
        writer.add_scalar("tok_per_sec", tok_per_sec,  step)
        writer.add_scalar("param_norm",  param_norm,   step)
        writer.add_scalar("grad_clipped", grad_clipped, step)

        # --- val eval + checkpoint (every eval_every steps) ---
        if at_eval:
            val_loss, act_mean, act_std = eval_val_loss(model, val_set, tcfg)
            print(f"  >>> val_loss {val_loss:.4f} @ step {step}")
            writer.add_scalar("loss/val",              val_loss,  step)
            writer.add_scalar("update_to_weight_ratio", upd_ratio, step)
            writer.add_scalar("activation/mean",        act_mean,  step)
            writer.add_scalar("activation/std",         act_std,   step)
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
