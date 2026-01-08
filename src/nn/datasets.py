from collections import Counter

import nltk
import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset

nltk.download("stopwords")
nltk.download("punkt_tab")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def clean_and_stem(
    doc: str,
    clean_stop_words: bool = True,
    include_stemming: bool = True,
    eliminate_non_letters: bool = True,
    convert_to_lowercase: bool = True,
) -> str:
    """
    Extended clean_and_stem function with configurable preprocessing options.

    Args:
        doc: Input document text
        clean_stop_words: Whether to remove stop words
        include_stemming: Whether to apply stemming
        eliminate_non_letters: Whether to remove tokens containing non-letter characters
        convert_to_lowercase: Whether to convert all words to lowercase

    Returns:
        Preprocessed document as a string
    """

    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    tokens = word_tokenize(doc)
    new_tokens = []
    for token in tokens:
        # Apply lowercase conversion if enabled
        if convert_to_lowercase:
            token = token.lower()

        # Apply filtering based on options
        should_keep = True

        # Check if token contains only letters (if enabled)
        if eliminate_non_letters and not token.isalpha():
            should_keep = False

        # Check if token is a stop word (if enabled)
        if clean_stop_words and token.lower() in stop_words:
            should_keep = False

        if should_keep:
            # Apply stemming if enabled
            if include_stemming:
                stemmed = ps.stem(token)
                new_tokens.append(stemmed)
            else:
                new_tokens.append(token)

    return " ".join(new_tokens)


class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        texts = [clean_and_stem(doc) for doc in texts]
        # Filter out empty texts to avoid tokenization issues

        valid_indices = [
            i for i, text in enumerate(texts) if len(str(text).strip()) > 0
        ]  # noqa: E501
        self.texts = [texts[i] for i in valid_indices]
        self.labels = [labels[i] for i in valid_indices]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        # Ensure text is a string and not empty
        text = str(self.texts[idx]).strip()
        if not text:
            text = " "  # Fallback to single space if empty

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


Item = tuple[list[str], str]


class NewsGroupDatasetSimple(Dataset):
    def __init__(self, subset: str, contextual: bool = False) -> None:
        assert subset in ["train", "test", "all"]
        newsgroups = fetch_20newsgroups(
            subset=subset, remove=("headers", "footers", "quotes")
        )
        if contextual:
            texts = [doc for doc in newsgroups.data]
            self.items = [
                (text[:min(512, len(ext))], label) for text, label in zip(texts, newsgroups.target)
            ]
        else:
            texts = [clean_and_stem(doc) for doc in newsgroups.data]
            self.items = [
                (text.split(), label) for text, label in zip(texts, newsgroups.target)
            ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[list[str], np.int64]:  # ty:ignore[unresolved-reference, invalid-method-override]
        return self.items[idx]


class NewsGroupsVectorizerSimple:
    """
    Note: this code is gratuitously borrowed from a lab in Linköpings University in NLP
    """

    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(
        self, dataset: NewsGroupDatasetSimple, n_vocab: int = 2048, min_freq=3
    ) -> None:
        # Unzip the dataset into reviews and labels
        reviews, labels = zip(*dataset)  # noqa: B905

        # Count the tokens and get the most common ones
        counter = Counter(t for r in reviews for t in r)
        most_common = [t for t, _ in counter.most_common(n_vocab - 2)]
        # Create the token-to-index and label-to-index mappings
        self.t2i = {t: i for i, t in enumerate([self.PAD, self.UNK] + most_common)}
        self.l2i = {l: i for i, l in enumerate(sorted(set(labels)))}

    def __call__(self, items: list[Item]) -> tuple[torch.Tensor, torch.Tensor]:
        XS = []
        YS = []
        max_tokens_per_sentence = max(len(item) for item, _ in items)
        for item in items:
            sentiment, label = item
            n_pads = max_tokens_per_sentence - len(sentiment)
            sentiment = sentiment + [self.PAD for _ in range(n_pads)]
            x = []
            for word in sentiment:
                if word in self.t2i:
                    x.append(self.t2i[word])
                else:
                    x.append(self.t2i[self.UNK])
            y = self.l2i[label]
            XS.append(x)
            YS.append(y)

        return torch.tensor(XS, dtype=torch.long), torch.tensor(YS, dtype=torch.long)


class NewsGroupsTextCollator:
    """
    Collate function for models that use text embeddings (like SentenceTransformer).
    Returns raw text strings instead of tokenized tensors.
    """

    def __init__(self, dataset: NewsGroupDatasetSimple) -> None:
        # Get labels from dataset to create label mapping
        _, labels = zip(*dataset)
        self.l2i = {l: i for i, l in enumerate(sorted(set(labels)))}

    def __call__(self, items: list[Item]) -> tuple[list[str], torch.Tensor]:
        texts = []
        labels = []
        for item in items:
            words, label = item
            # Join words back into text
            text = " ".join(words)
            texts.append(text)
            labels.append(self.l2i[label])
        return texts, torch.tensor(labels, dtype=torch.long)


def glove_tokenize(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
) -> list[str]:
    """
    Simple tokenizer for GloVe embeddings.

    Args:
        text: Input text to tokenize
        lowercase: Convert to lowercase (recommended for GloVe)
        remove_stopwords: Whether to remove English stopwords
        min_token_length: Minimum token length to keep

    Returns:
        List of tokens
    """
    # Tokenize using NLTK
    tokens = word_tokenize(text)

    # Apply lowercase
    if lowercase:
        tokens = [t.lower() for t in tokens]

    # Remove stopwords if requested
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    # Filter by length and keep only alphabetic tokens
    tokens = [t for t in tokens if t.isalpha() and len(t) >= min_token_length]

    return tokens


class GloVeVectorizer:
    """
    Vectorizer for GloVe embeddings that converts text to token indices.

    This vectorizer uses a GloVe embedding layer's vocabulary to convert
    tokens to indices. It handles padding and unknown tokens automatically.

    Args:
        embedding_layer: GloVeEmbedding instance with word2idx mapping
        max_length: Maximum sequence length (longer sequences are truncated)

    Example:
        from src.nn.modules import GloVeEmbedding
        from src.nn.config import TextNeuralNetworkConfig
        import torch.nn as nn

        config = TextNeuralNetworkConfig(
            version="v1",
            name="glove_test",
            activate=nn.GELU(),
            loss_function=nn.CrossEntropyLoss(),
            vocab_size=10000,
            trainable=False,
            n_embd=100
        )
        embedding = GloVeEmbedding(config, 'glove-wiki-gigaword-100')
        vectorizer = GloVeVectorizer(embedding, max_length=128)

        # Tokenize and vectorize
        tokens = glove_tokenize("This is a test sentence")
        input_ids, attention_mask = vectorizer.vectorize(tokens)
    """

    def __init__(self, embedding_layer, max_length: int = 128):
        """
        Initialize the vectorizer.

        Args:
            embedding_layer: GloVeEmbedding instance
            max_length: Maximum sequence length
        """
        self.word2idx = embedding_layer.word2idx
        self.PAD_IDX = embedding_layer.PAD_IDX
        self.UNK_IDX = embedding_layer.UNK_IDX
        self.max_length = max_length

    def vectorize(self, tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert tokens to indices with padding and attention mask.

        Args:
            tokens: List of string tokens

        Returns:
            Tuple of (input_ids, attention_mask)
            - input_ids: Tensor of shape (max_length,) with token indices
            - attention_mask: Tensor of shape (max_length,) with 1 for real tokens, 0 for padding
        """
        # Truncate if necessary
        tokens = tokens[: self.max_length]

        # Convert tokens to indices
        input_ids = []
        for token in tokens:
            if token in self.word2idx:
                input_ids.append(self.word2idx[token])
            else:
                input_ids.append(self.UNK_IDX)

        # Create attention mask (1 for real tokens)
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        input_ids.extend([self.PAD_IDX] * padding_length)
        attention_mask.extend([0] * padding_length)

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(
            attention_mask, dtype=torch.long
        )

    def __call__(
        self, text: str, tokenize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize and vectorize text in one call.

        Args:
            text: Input text string
            tokenize: Whether to tokenize the text (if False, expects text to be a list of tokens)

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        if tokenize:
            tokens = glove_tokenize(text)
        else:
            tokens = text if isinstance(text, list) else text.split()

        return self.vectorize(tokens)


class GloVeCollator:
    """
    Collate function for GloVe-based models.
    Tokenizes and vectorizes text on-the-fly during data loading.

    Args:
        dataset: NewsGroupDatasetSimple instance to get label mappings
        vectorizer: GloVeVectorizer instance for converting text to indices

    Example:
        train_dataset = NewsGroupDatasetSimple(subset="train")
        embedding = GloVeEmbedding(config, 'glove-wiki-gigaword-100')
        vectorizer = GloVeVectorizer(embedding, max_length=128)
        collator = GloVeCollator(train_dataset, vectorizer)

        dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collator)
    """

    def __init__(self, dataset: NewsGroupDatasetSimple, vectorizer):
        # Get labels from dataset to create label mapping
        _, labels = zip(*dataset)
        self.l2i = {l: i for i, l in enumerate(sorted(set(labels)))}
        self.vectorizer = vectorizer

    def __call__(
        self, items: list[Item]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate a batch of items.

        Args:
            items: List of (tokens, label) tuples from NewsGroupDatasetSimple

        Returns:
            Tuple of (input_ids, attention_masks, labels)
            - input_ids: Tensor of shape (batch_size, max_length)
            - attention_masks: Tensor of shape (batch_size, max_length)
            - labels: Tensor of shape (batch_size,)
        """
        batch_input_ids = []
        batch_attention_masks = []
        labels = []

        for item in items:
            words, label = item
            # Join words back into text
            text = " ".join(words)

            # Vectorize using GloVe vectorizer
            input_ids, attention_mask = self.vectorizer(text)

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            labels.append(self.l2i[label])

        return (
            torch.stack(batch_input_ids),
            torch.stack(batch_attention_masks),
            torch.tensor(labels, dtype=torch.long),
        )
