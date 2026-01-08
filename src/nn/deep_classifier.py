from torch import nn

from .config import TextNeuralNetworkConfig
from .modules import Block, GloVeEmbedding, RandomWordEmbedding, ContextualEmbedding


class DeepClassifier(nn.Module):
    def __init__(self, config: TextNeuralNetworkConfig):
        super().__init__()
        self.embedding = ContextualEmbedding(trainable=config.trainable)
        self.config: TextNeuralNetworkConfig = config
        self.config.n_embd = self.embedding.output_dim
        hidden_layers = [
            nn.Linear(config.n_embd, config.n_embd),
            config.activate,
        ] * config.n_hidden_layers
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x):
        x = self.embedding(x)  # ContextualEmbedding returns (batch_size, embedding_dim)
        x = self.hidden_layers(x)
        return self.output_layer(x)

    def predict(self, x):
        logits = self.forward(x)
        pred_probab = nn.Softmax(dim=-1)(logits)
        return pred_probab.argmax(dim=-1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return nn.Softmax(dim=-1)(logits)


class OldDeepClassifier(nn.Module):
    def __init__(
        self,
        config: TextNeuralNetworkConfig,
    ):
        super().__init__()
        self.embedding_layer: RandomWordEmbedding = RandomWordEmbedding(config)
        self.n_classes: int = config.n_classes
        self.activate: nn.Module = config.activate
        self.hidden_stack: nn.Sequential = nn.Sequential(
            *[Block(config)] * config.n_hidden_layers
        )
        self.output_layer: nn.Linear = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, x, return_logits=False):
        # Handle dictionary input from dataset
        x = self.embedding_layer(x)
        hidden_logits = self.hidden_stack(x)
        return self.output_layer(hidden_logits)

    def predict(self, x):
        logits = self.forward(x)
        pred_probab = nn.Softmax(dim=-1)(logits)
        return pred_probab.argmax(dim=-1)

    def predict_proba(self, x):
        logits = self.forward(x)
        return nn.Softmax(dim=-1)(logits)


class GloVeClassifier(nn.Module):
    """Classifier using GloVe static word embeddings."""

    def __init__(
        self,
        config: TextNeuralNetworkConfig,
        glove_model_name="glove-wiki-gigaword-100",
    ):
        super().__init__()
        self.embedding = GloVeEmbedding(config, glove_model_name)
        self.config: TextNeuralNetworkConfig = config
        # Update n_embd to match GloVe dimension
        self.config.n_embd = self.embedding.output_dim
        self.hidden_layers = nn.Sequential(
            *[Block(self.config)] * self.config.n_hidden_layers
        )
        self.output_layer = nn.Linear(self.config.n_embd, config.n_classes)

    def forward(self, input_ids, attention_mask):
        # GloVe embedding takes input_ids and attention_mask
        x = self.embedding(input_ids, attention_mask)  # (batch_size, embedding_dim)
        x = self.hidden_layers(x)
        return self.output_layer(x)

    def predict(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        pred_probab = nn.Softmax(dim=-1)(logits)
        return pred_probab.argmax(dim=-1)

    def predict_proba(self, input_ids, attention_mask):
        logits = self.forward(input_ids, attention_mask)
        return nn.Softmax(dim=-1)(logits)
