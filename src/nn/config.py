"""
Note: A lot of this code has been adapted from Linköpings NLP course ete387
"""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TextNeuralNetworkConfig:
    # Name
    version: str
    name: str
    # Neural Network
    activate: nn.Module
    loss_function: nn.Module
    vocab_size: int
    trainable: bool = False
    n_embd: int = 768
    n_classes: int = 20
    n_hidden_layers: int = 2
    use_kaiming_initialization: bool = True
    drop_out: float = 0.3

    # Training
    epochs: int = 500
    batch_size: int = 32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Early stopping
    min_delta: float = 1e-2
    patience: int = 3
    # Optimisation and learning rate scheduling
    shuffle: bool = True
    learning_rate: float = 1e-2
    weight_decay: float = 1e-3
    max_lr: float = 6e-5
    min_lr: float = 6e-7
    n_steps: int = 15076
    n_warmup_steps: int = 3000
    n_decay_steps: int = 4000
    betas: tuple[float, float] = (0.9, 0.95)
    clip_norm: float = 1.0


def configure_optimizer(
    model: nn.Module, config: TextNeuralNetworkConfig
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    no_decay_params = [p for p in params if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=config.max_lr,
        betas=config.betas,
        fused=(config.device.type == "cuda"),
    )


def get_lr_factor(step: int, config: TextNeuralNetworkConfig) -> float:
    if step < config.n_warmup_steps:
        return step / config.n_warmup_steps
    x = (step - config.n_warmup_steps) / (config.n_decay_steps - config.n_warmup_steps)
    x = min(x, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * x))
    return config.min_lr + (1.0 - config.min_lr) * cosine_decay
