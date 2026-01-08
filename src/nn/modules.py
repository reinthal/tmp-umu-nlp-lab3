"""
Note: A lot of this code has been adapted from Linköpings NLP course ete387
"""

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import GPT2Model
import gensim.downloader as api
import numpy as np
from typing import Optional, List

from .config import TextNeuralNetworkConfig


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha: int = alpha
        self.gamma: int = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x) -> torch.Tensor:
        batch_size, n_embd = x.shape
        x = self.c_fc(x)  # shape: [batch_size, n_embd * 4]
        x = self.config.activate(x)  # shape: [batch_size, n_embd * 4]
        x = self.c_proj(x)  # shape: [batch_size, n_embd]
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp: MLP = MLP(config)
        self.ln_2: LayerNorm = LayerNorm(config)
        self.dropout = nn.Dropout(config.drop_out)

    def forward(self, x):
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class LayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.g = nn.Parameter(torch.ones(config.n_embd))
        self.b = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(unbiased=False, dim=-1, keepdim=True)
        return self.g * (x - mean) / torch.sqrt(variance + 1e-05) + self.b


class GPT2Embedder(nn.Module):
    def __init__(self, trainable: bool = False):
        super().__init__()

        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        self.gpt2.requires_grad_(trainable)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, -1, :]


class ContextualEmbedding(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()

        if trainable:
            raise ValueError(
                "ContextualEmbedding (all-MiniLM-L6-v2) cannot be trained. "
                "The SentenceTransformer.encode() method uses inference_mode internally. "
                "Set trainable=False to use frozen embeddings."
            )

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.trainable = trainable
        self.model.eval()
        self.output_dim: int = 384

    def forward(self, texts):
        # ContextualEmbedding is always frozen (trainable=False)
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False
            )
            # Clone to convert inference tensors to regular tensors for autograd
            return embeddings.clone()


class RandomWordEmbedding(nn.Module):
    def __init__(self, config: TextNeuralNetworkConfig):
        super().__init__()
        self.config: TextNeuralNetworkConfig = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Kaiming Initialization https://docs.pytorch.org/docs/stable/nn.init.html
        if config.use_kaiming_initialization:
            nn.init.kaiming_uniform_(
                self.embedding.weight, mode="fan_in", nonlinearity="relu"
            )
        self.output_dim: int = config.n_embd

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, seq_len) or (seq_len,)
            attention_mask: (B, seq_len) or (seq_len,)
        Returns:
            embeddings: (B, embedding_dim) or (embedding_dim,) - averaged
        """
        emb = self.embedding(input_ids)  # (B, seq_len, dim) or (seq_len, dim)

        if attention_mask is not None:
            # Average non-padding tokens
            emb = emb * attention_mask.unsqueeze(-1)
            return emb.sum(-2) / attention_mask.sum(-1, keepdim=True)
        else:
            return emb.mean(-2)  # (B, 1, dim) or (1, dim)


class GloVeEmbedding(nn.Module):
    """
    Static GloVe word embeddings module using pretrained vectors from Gensim.

    This module loads pretrained GloVe embeddings and creates a PyTorch embedding layer.
    The embeddings can be frozen or fine-tuned based on the config.trainable parameter.

    Args:
        config: TextNeuralNetworkConfig containing:
            - trainable: Whether to allow fine-tuning of embeddings
            - n_embd: Expected embedding dimension (must match GloVe dimension)
            - vocab_size: Vocabulary size (used for reference)
        glove_model_name: Name of pretrained GloVe model from Gensim
            Options: 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-100',
                    'glove-wiki-gigaword-200', 'glove-wiki-gigaword-300',
                    'glove-twitter-25', 'glove-twitter-50', 'glove-twitter-100', 'glove-twitter-200'

    Example:
        config = TextNeuralNetworkConfig(
            version="v1",
            name="glove_classifier",
            activate=nn.GELU(),
            loss_function=nn.CrossEntropyLoss(),
            vocab_size=10000,
            trainable=False,  # Freeze embeddings
            n_embd=100,
        )
        embedding_layer = GloVeEmbedding(config, 'glove-wiki-gigaword-100')
    """

    def __init__(
        self,
        config: TextNeuralNetworkConfig,
        glove_model_name: str = 'glove-wiki-gigaword-100'
    ):
        super().__init__()
        self.config = config
        self.glove_model_name = glove_model_name

        # Load pretrained GloVe vectors using Gensim
        print(f"Loading GloVe model: {glove_model_name}...")
        self.glove_vectors = api.load(glove_model_name)
        print(f"Loaded {len(self.glove_vectors)} word vectors")

        # Get embedding dimension from loaded vectors
        self.embedding_dim = self.glove_vectors.vector_size
        self.output_dim = self.embedding_dim

        # Verify config matches GloVe dimension
        if config.n_embd != self.embedding_dim:
            print(f"Warning: config.n_embd ({config.n_embd}) != GloVe dimension ({self.embedding_dim})")
            print(f"Using GloVe dimension: {self.embedding_dim}")

        # Create word-to-index mapping
        self.word2idx = {word: idx for idx, word in enumerate(self.glove_vectors.index_to_key)}
        self.vocab_size = len(self.word2idx)

        # Special tokens
        self.PAD_IDX = self.vocab_size
        self.UNK_IDX = self.vocab_size + 1

        # Create embedding matrix
        embedding_matrix = self._create_embedding_matrix()

        # Create PyTorch embedding layer
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=not config.trainable,
            padding_idx=self.PAD_IDX
        )

        print(f"GloVe embedding layer created:")
        print(f"  - Vocabulary size: {self.vocab_size + 2} (+ PAD and UNK)")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Trainable: {config.trainable}")

    def _create_embedding_matrix(self) -> torch.Tensor:
        """Create embedding matrix from GloVe vectors with special tokens."""
        # Initialize matrix with zeros
        embedding_matrix = np.zeros((self.vocab_size + 2, self.embedding_dim))

        # Fill in GloVe vectors
        for word, idx in self.word2idx.items():
            embedding_matrix[idx] = self.glove_vectors[word]

        # PAD token (zeros)
        embedding_matrix[self.PAD_IDX] = np.zeros(self.embedding_dim)

        # UNK token (mean of all word vectors)
        embedding_matrix[self.UNK_IDX] = np.mean(embedding_matrix[:self.vocab_size], axis=0)

        return torch.FloatTensor(embedding_matrix)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the embedding layer.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len) or (seq_len,)
            attention_mask: Optional attention mask of shape (batch_size, seq_len) or (seq_len,)
                          1 for real tokens, 0 for padding

        Returns:
            Averaged embeddings of shape (batch_size, embedding_dim) or (embedding_dim,)
        """
        # Get embeddings: (batch_size, seq_len, embedding_dim)
        emb = self.embedding(input_ids)

        if attention_mask is not None:
            # Mask out padding tokens and average over sequence length
            # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            emb = emb * mask_expanded

            # Sum over sequence length and divide by number of non-padding tokens
            sum_emb = emb.sum(dim=-2)  # (batch_size, embedding_dim)
            sum_mask = mask_expanded.sum(dim=-2)  # (batch_size, 1)
            return sum_emb / sum_mask.clamp(min=1e-9)  # Avoid division by zero
        else:
            # Average over all tokens
            return emb.mean(dim=-2)  # (batch_size, embedding_dim) or (embedding_dim,)
