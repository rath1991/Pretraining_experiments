import torch
from config import ModelConfig, TrainConfig
from train import train

torch.set_float32_matmul_precision("high")   # use TF32 on Ampere+ GPUs

if __name__ == "__main__":
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    train(mcfg, tcfg)
