import torch
from config import ModelConfig, TrainConfig
from train import train

torch.set_float32_matmul_precision("high")   # use TF32 on Ampere+ GPUs

if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    train(model_cfg, train_cfg)
