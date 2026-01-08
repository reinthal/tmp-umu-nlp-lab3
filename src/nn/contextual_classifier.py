import torch
from torch import nn

from .config import TextNeuralNetworkConfig
from .modules import Block, GPT2Embedder


class TextContextualNeuralNetwork(nn.Module):
    def __init__(
        self,
        config: TextNeuralNetworkConfig,
    ):
        super().__init__()
        self.config: TextNeuralNetworkConfig = config
        self.transformer_layer: GPT2Embedder = GPT2Embedder(trainable=config.trainable)
        self.activate: nn.Module = config.activate
        self.hidden_stack: nn.Sequential = nn.Sequential(
            *[Block(config)] * config.n_hidden_layers
        )
        self.output_layer: nn.Linear = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x: torch.Tensor | dict):
        # Handle dictionary input from dataset
        if isinstance(x, dict):
            input_ids = x["input_ids"]
            attention_mask = x.get("attention_mask", None)  # ty:ignore[no-matching-overload]
            x = self.transformer_layer(input_ids, attention_mask)
        else:
            x = self.transformer_layer(x)
        if self.config.n_hidden_layers > 0:
            hidden_logits = self.hidden_stack(x)
        logits = self.output_layer(hidden_logits)
        pred_probab = nn.Softmax(dim=-1)(logits)
        return pred_probab.argmax(dim=-1)
