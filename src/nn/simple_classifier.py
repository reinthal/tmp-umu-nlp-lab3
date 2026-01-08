"""
Note: A lot of this code has been adapted from Linköpings NLP course ete387
"""

from torch import nn

from .config import TextNeuralNetworkConfig


class SimpleClassifier(nn.Module):
    def __init__(self, config: TextNeuralNetworkConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        hidden_layers = [
            nn.Linear(config.n_embd, config.n_embd),
            config.activate,
        ] * config.n_hidden_layers
        if config.use_kaiming_initialization:
            k = 1.0 / self.embedding.embedding_dim
            nn.init.uniform_(self.embedding.weight, -(k**0.5), k**0.5)
        self.embedding_dropout = nn.Dropout(config.drop_out)
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        # Create mask for non-padding tokens (padding_idx=0)
        mask = (x != 0).unsqueeze(-1).float()  # Shape: (batch_size, seq_len, 1)

        embedded = self.embedding_dropout(
            self.embedding(x)
        )  # Shape: (batch_size, seq_len, n_embd)

        # Apply mask and compute mean only over non-padding tokens
        masked_embedded = embedded * mask  # Zero out padding embeddings
        sum_embedded = masked_embedded.sum(dim=-2)  # Sum over sequence length
        seq_lengths = mask.sum(dim=-2)  # Count non-padding tokens

        # Avoid division by zero for empty sequences (though shouldn't happen)
        seq_lengths = seq_lengths.clamp(min=1)

        x = sum_embedded / seq_lengths

        x = self.hidden_layers(x)
        return self.output(x)

    def predict(self, x):
        logits = self.forward(x)
        pred_probab = nn.Softmax(dim=-1)(logits)
        return pred_probab.argmax(dim=-1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return nn.Softmax(dim=-1)(logits)
