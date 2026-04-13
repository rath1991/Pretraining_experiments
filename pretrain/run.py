import argparse
import torch
from config import ModelConfig, TrainConfig
from train import train

# Exp 03 — Init + Depth Stability
# Factorial: 3 depths × 2 init schemes. 300 steps each (diagnostic run).
# Primary signal: grad_norm trajectory and early loss dynamics.
EXP03_VARIANTS = {
    "A": dict(model=dict(n_layer=3,  scaled_init=False), train=dict(max_steps=300, out_dir="../out/exp03_varA")),
    "B": dict(model=dict(n_layer=3,  scaled_init=True),  train=dict(max_steps=300, out_dir="../out/exp03_varB")),
    "C": dict(model=dict(n_layer=6,  scaled_init=False), train=dict(max_steps=300, out_dir="../out/exp03_varC")),
    "D": dict(model=dict(n_layer=6,  scaled_init=True),  train=dict(max_steps=300, out_dir="../out/exp03_varD")),
    "E": dict(model=dict(n_layer=12, scaled_init=False), train=dict(max_steps=300, out_dir="../out/exp03_varE")),
    "F": dict(model=dict(n_layer=12, scaled_init=True),  train=dict(max_steps=300, out_dir="../out/exp03_varF")),
}

torch.set_float32_matmul_precision("high")   # use TF32 on Ampere+ GPUs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=list(EXP03_VARIANTS.keys()),
        help="Exp 03 variant (A–F). Omit to run baseline config."
    )
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.variant is not None:
        overrides = EXP03_VARIANTS[args.variant]
        for k, v in overrides["model"].items():
            setattr(model_cfg, k, v)
        for k, v in overrides["train"].items():
            setattr(train_cfg, k, v)
        print(f"Exp 03 variant {args.variant}: n_layer={model_cfg.n_layer}, "
              f"scaled_init={model_cfg.scaled_init}, out_dir={train_cfg.out_dir}")

    train(model_cfg, train_cfg)
