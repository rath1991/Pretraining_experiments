import argparse
import torch
from config import ModelConfig, TrainConfig
from train import train

# Exp 02 — Optimizer + Warmup Stability
# Each variant changes exactly one or two knobs against the baseline (D).
EXP02_VARIANTS = {
    "A": dict(optimizer="adam",  warmup_steps=0,  out_dir="../out/exp02_varA"),
    "B": dict(optimizer="adam",  warmup_steps=50, out_dir="../out/exp02_varB"),
    "C": dict(optimizer="adamw", warmup_steps=0,  out_dir="../out/exp02_varC"),
    "D": dict(optimizer="adamw", warmup_steps=50, out_dir="../out/exp02_varD"),
}

torch.set_float32_matmul_precision("high")   # use TF32 on Ampere+ GPUs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=list(EXP02_VARIANTS.keys()),
        help="Exp 02 variant (A/B/C/D). Omit to run baseline config."
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.variant is not None:
        overrides = EXP02_VARIANTS[args.variant]
        for k, v in overrides.items():
            setattr(train_cfg, k, v)
        print(f"Exp 02 variant {args.variant}: optimizer={train_cfg.optimizer}, "
              f"warmup_steps={train_cfg.warmup_steps}, out_dir={train_cfg.out_dir}")

    train(model_cfg, train_cfg)
