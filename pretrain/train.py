import os
import math
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from config import ModelConfig, TrainConfig
from data import get_datasets
from model import GPT


def get_lr(step: int, train_cfg: TrainConfig) -> float:
    """
    Two-phase LR schedule:
      1. Linear warmup from 0 → max_lr over warmup_steps.
         Prevents large gradient updates before the optimiser has reliable statistics.
      2. Cosine decay from max_lr → min_lr for the remainder of training.
         Smooth reduction avoids abrupt loss spikes late in the run.
    """
    if step < train_cfg.warmup_steps:
        return train_cfg.max_lr * (step + 1) / train_cfg.warmup_steps

    progress     = (step - train_cfg.warmup_steps) / (train_cfg.max_steps - train_cfg.warmup_steps)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_lr + cosine_scale * (train_cfg.max_lr - train_cfg.min_lr)


@torch.no_grad()
def _collect_activation_stats(model: GPT) -> tuple:
    """
    Attaches a forward hook to ln_final (the last LayerNorm before lm_head).
    The hook accumulates online statistics (sum, sum-of-squares, count) across
    all batches processed while it is active.

    Returns (hook_handle, stats_dict).
    Caller is responsible for calling hook_handle.remove() when done.

    Why ln_final? Its output is the direct input to the unembedding projection —
    a good proxy for the health of learned representations.
    """
    stats = {"sum": 0.0, "sq_sum": 0.0, "count": 0}

    def _hook(module, input, output):
        out = output.detach().float()
        stats["sum"]    += out.sum().item()
        stats["sq_sum"] += (out ** 2).sum().item()
        stats["count"]  += out.numel()

    handle = model.ln_final.register_forward_hook(_hook)
    return handle, stats


@torch.no_grad()
def eval_val_loss(model: GPT, val_set, train_cfg: TrainConfig) -> tuple:
    """
    Evaluates the model on the validation set.
    Averages loss over eval_steps batches to reduce variance.
    Simultaneously collects activation statistics on ln_final outputs.

    Returns (val_loss, act_mean, act_std).
    """
    model.eval()
    dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32

    # Attach activation hook before running any batches
    hook_handle, act_stats = _collect_activation_stats(model)

    losses = []
    for _ in range(train_cfg.eval_steps):
        input_tokens, targets = val_set.get_batch(train_cfg.batch_size, train_cfg.device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(input_tokens, targets)
        losses.append(loss.item())

    # Detach hook before returning — must not leak into training steps
    hook_handle.remove()

    # Compute mean and std from accumulated online statistics
    act_mean = act_stats["sum"] / act_stats["count"]
    act_var  = act_stats["sq_sum"] / act_stats["count"] - act_mean ** 2
    act_std  = act_var ** 0.5

    model.train()
    return sum(losses) / len(losses), act_mean, act_std


@torch.no_grad()
def _compute_update_ratio(model: GPT, pre_step_params: list) -> float:
    """
    Measures the size of the parameter update relative to the current weights.
    Requires a snapshot taken *before* optimizer.step().

    update_ratio = ‖Δθ‖ / ‖θ‖

    Healthy range: ~1e-3.
    Consistently >1e-2 may indicate the learning rate is too high.
    Consistently <1e-4 may indicate the learning rate is too low or the model has stalled.
    """
    param_norm  = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
    update_norm = sum(
        (p.data - snap).norm().item() ** 2
        for p, snap in zip(model.parameters(), pre_step_params)
    ) ** 0.5
    return update_norm / (param_norm + 1e-8)


def build_optimizer(model: GPT, train_cfg: TrainConfig):
    """
    AdamW with selective weight decay.
    Weight decay is applied only to matrices (nn.Linear weights, embedding weights)
    because they have sufficient capacity to benefit from regularisation.
    Vectors (LayerNorm scale/bias, Linear bias) are excluded — decaying them
    shrinks learned scale parameters and harms training.
    """
    decay_params   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() <  2]
    param_groups = [
        {"params": decay_params,   "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=train_cfg.max_lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
    )


def train(model_cfg: ModelConfig, train_cfg: TrainConfig):
    os.makedirs(train_cfg.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(train_cfg.out_dir, "tb_logs"))

    train_set, val_set = get_datasets(train_cfg.data_dir, model_cfg.block_size)

    model     = GPT(model_cfg).to(train_cfg.device)
    optimizer = build_optimizer(model, train_cfg)

    # BF16 autocast wraps the forward pass only; parameters remain float32.
    # BF16 is preferred over FP16 because it matches float32's dynamic range,
    # so no GradScaler is needed.
    dtype = torch.bfloat16 if train_cfg.bf16 else torch.float32

    print(f"Parameters : {model.num_params()/1e6:.1f}M")
    print(f"Device     : {torch.cuda.get_device_name(0)}")
    print(f"Precision  : {'bfloat16' if train_cfg.bf16 else 'float32'}")
    print(f"Steps      : {train_cfg.max_steps} "
          f"({train_cfg.max_steps * train_cfg.batch_size * model_cfg.block_size * train_cfg.grad_accum / 1e9:.1f}B tokens)")

    model.train()
    for step in range(train_cfg.max_steps):
        step_start   = time.time()
        is_eval_step = (step % train_cfg.eval_every == 0)

        # ── 1. Learning rate update ───────────────────────────────────────────
        # Recompute every step; manually assign to each param group because
        # PyTorch schedulers don't compose cleanly with custom schedules.
        current_lr = get_lr(step, train_cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # ── 2. Pre-step parameter snapshot (eval steps only) ─────────────────
        # The update ratio measures how much this step changed the weights.
        # The snapshot must be taken *before* optimizer.step(); afterwards the
        # old values are gone. We only do this on eval steps — cloning all
        # parameters is expensive (~85M floats).
        if is_eval_step:
            pre_step_params = [p.data.clone() for p in model.parameters()]

        # ── 3. Forward + backward (gradient accumulation) ────────────────────
        # Running grad_accum micro-batches before one optimizer step is
        # mathematically equivalent to a single batch of size
        # (batch_size × grad_accum) but uses only 1/grad_accum the VRAM.
        # Dividing loss by grad_accum before .backward() makes the gradients
        # average (not sum) across micro-batches.
        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(train_cfg.grad_accum):
            input_tokens, targets = train_set.get_batch(train_cfg.batch_size, train_cfg.device)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, loss = model(input_tokens, targets)
            (loss / train_cfg.grad_accum).backward()
            train_loss += loss.item() / train_cfg.grad_accum

        # ── 4. Gradient clipping + optimizer step ────────────────────────────
        # Clipping the global gradient norm to grad_clip prevents a single
        # large-gradient step from destabilising the weight space.
        # was_grad_clipped is logged to TensorBoard to track clipping frequency.
        grad_norm        = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        was_grad_clipped = int(grad_norm.item() > train_cfg.grad_clip)
        optimizer.step()

        # ── 5. Timing + throughput ────────────────────────────────────────────
        # synchronize() ensures the CUDA kernel is done before we stop the clock.
        torch.cuda.synchronize()
        step_time       = time.time() - step_start
        tokens_per_step = train_cfg.batch_size * model_cfg.block_size * train_cfg.grad_accum
        tok_per_sec     = tokens_per_step / step_time

        # ── 6. Param norm (every step) ────────────────────────────────────────
        # Global L2 norm of all parameters. A steadily growing param norm with
        # a flat or rising loss is a sign of weight explosion.
        with torch.no_grad():
            param_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

        # ── 7. Console log ────────────────────────────────────────────────────
        if step % train_cfg.log_every == 0:
            eta_hours = (train_cfg.max_steps - step) * step_time / 3600
            print(
                f"step {step:5d}/{train_cfg.max_steps} | loss {train_loss:.4f} | "
                f"lr {current_lr:.2e} | grad_norm {grad_norm:.3f} | "
                f"tok/s {tok_per_sec:,.0f} | ETA {eta_hours:.1f}h"
            )

        # ── 8. TensorBoard scalars (every step) ───────────────────────────────
        writer.add_scalar("loss/train",   train_loss,       step)
        writer.add_scalar("lr",           current_lr,       step)
        writer.add_scalar("grad_norm",    grad_norm,        step)
        writer.add_scalar("tok_per_sec",  tok_per_sec,      step)
        writer.add_scalar("param_norm",   param_norm,       step)
        writer.add_scalar("grad_clipped", was_grad_clipped, step)

        # ── 9. Val eval + checkpoint (every eval_every steps) ─────────────────
        if is_eval_step:
            # Val loss averaged over eval_steps batches; act_mean/std from ln_final
            val_loss, act_mean, act_std = eval_val_loss(model, val_set, train_cfg)

            # Update ratio: how large was this step's Δθ relative to ‖θ‖?
            update_weight_ratio = _compute_update_ratio(model, pre_step_params)
            del pre_step_params   # free the clone — no longer needed

            print(f"  >>> val_loss {val_loss:.4f} @ step {step}")

            writer.add_scalar("loss/val",               val_loss,            step)
            writer.add_scalar("update_to_weight_ratio", update_weight_ratio, step)
            writer.add_scalar("activation/mean",        act_mean,            step)
            writer.add_scalar("activation/std",         act_std,             step)

            ckpt = {
                "step":       step,
                "val_loss":   val_loss,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "model_cfg":  model_cfg,
                "train_cfg":  train_cfg,
            }
            torch.save(ckpt, os.path.join(train_cfg.out_dir, f"ckpt_{step:05d}.pt"))

    writer.close()
