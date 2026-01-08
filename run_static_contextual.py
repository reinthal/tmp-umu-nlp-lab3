import argparse
import csv
import glob
import os
import pickle
from typing import Dict, List
from src.nn.modules import FocalLoss

import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from tqdm import tqdm

from src.nn.config import TextNeuralNetworkConfig
from src.nn.datasets import (
    GloVeCollator,
    GloVeVectorizer,
    NewsGroupDatasetSimple,
    NewsGroupsTextCollator,
)
from src.nn.deep_classifier import DeepClassifier, GloVeClassifier
from src.utils import get_version_from_pyproject, save_model


def train_text_model(
    model: nn.Module,
    config: TextNeuralNetworkConfig,
):
    """
    Training function for models that use text embeddings (e.g., SentenceTransformer).
    The collate function returns raw text strings instead of tokenized tensors.
    """
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")

    # Use text collator instead of vectorizer
    text_collator = NewsGroupsTextCollator(train_dataset)

    model.to(config.device)

    # Use differential learning rates: smaller for pre-trained embeddings, larger for classifier
    if config.trainable:
        # Fine-tuning: use different learning rates for different parts
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.embedding.parameters(),
                    "lr": config.learning_rate * 0.02,  # 2e-5 if base lr is 1e-3
                    "weight_decay": 0.0,  # Don't apply weight decay to embeddings
                },
                {
                    "params": list(model.hidden_layers.parameters())
                    + list(model.output_layer.parameters()),
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
            ]
        )
    else:
        # Frozen embeddings: standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=text_collator,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=text_collator,
    )

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_model_state = model.state_dict().copy()
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        running_test_loss = 0

        # Train
        train_loss = 0
        tqdm_data_loader = tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]"
        )
        for bx, by in tqdm_data_loader:
            optimizer.zero_grad()
            # bx is now a list of text strings, not a tensor
            by = by.to(config.device)
            output = model(bx)  # Pass text strings directly to model
            loss = config.loss_function(output, by)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()
            train_loss = loss.item()
            running_loss += train_loss

        # Validate
        test_loss = 0
        tqdm_test_data_loader = tqdm(
            test_data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Test]"
        )
        for bx, by in tqdm_test_data_loader:
            model.eval()
            with torch.no_grad():
                # bx is a list of text strings
                by = by.to(config.device)
                output = model(bx)
                loss = config.loss_function(output, by)
                test_loss = loss.item()
                running_test_loss += test_loss

        train_losses.append(running_loss / len(data_loader))
        test_losses.append(running_test_loss / len(test_data_loader))

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Running training loss : {running_loss / len(data_loader):.4f}, "
            f"Running test loss: {running_test_loss / len(test_data_loader):.4f}"
        )
        if test_losses[-1] < best_test_loss - config.min_delta:
            best_test_loss = test_losses[-1]
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"  → Test loss improved! New best: {best_test_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement, {epochs_without_improvement}/{config.patience}")
            if epochs_without_improvement >= config.patience:
                model.load_state_dict(best_model_state)
                break

    return text_collator, model


def train_glove_model(
    model: nn.Module,
    config: TextNeuralNetworkConfig,
    glove_collator: GloVeCollator,
):
    """
    Training function for GloVe-based models.
    The collate function returns tokenized tensors (input_ids, attention_mask, labels).
    """
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")

    model.to(config.device)

    # Use differential learning rates: smaller for pre-trained embeddings, larger for classifier
    if config.trainable:
        # Fine-tuning: use different learning rates for different parts
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.embedding.parameters(),
                    "lr": config.learning_rate * 0.02,  # 2e-5 if base lr is 1e-3
                    "weight_decay": 0.0,  # Don't apply weight decay to embeddings
                },
                {
                    "params": list(model.hidden_layers.parameters())
                    + list(model.output_layer.parameters()),
                    "lr": config.learning_rate,
                    "weight_decay": config.weight_decay,
                },
            ]
        )
    else:
        # Frozen embeddings: standard optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=glove_collator,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=glove_collator,
    )

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_model_state = model.state_dict().copy()
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        running_test_loss = 0

        # Train
        train_loss = 0
        tqdm_data_loader = tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]"
        )
        for input_ids, attention_mask, labels in tqdm_data_loader:
            optimizer.zero_grad()
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)

            output = model(input_ids, attention_mask)
            loss = config.loss_function(output, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()
            train_loss = loss.item()
            running_loss += train_loss

        # Validate
        test_loss = 0
        tqdm_test_data_loader = tqdm(
            test_data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Test]"
        )
        for input_ids, attention_mask, labels in tqdm_test_data_loader:
            model.eval()
            with torch.no_grad():
                input_ids = input_ids.to(config.device)
                attention_mask = attention_mask.to(config.device)
                labels = labels.to(config.device)

                output = model(input_ids, attention_mask)
                loss = config.loss_function(output, labels)
                test_loss = loss.item()
                running_test_loss += test_loss

        train_losses.append(running_loss / len(data_loader))
        test_losses.append(running_test_loss / len(test_data_loader))

        print(
            f"Epoch {epoch + 1}/{config.epochs} - "
            f"Running training loss : {running_loss / len(data_loader):.4f}, "
            f"Running test loss: {running_test_loss / len(test_data_loader):.4f}"
        )
        if test_losses[-1] < best_test_loss - config.min_delta:
            best_test_loss = test_losses[-1]
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"  → Test loss improved! New best: {best_test_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement, {epochs_without_improvement}/{config.patience}")
            if epochs_without_improvement >= config.patience:
                model.load_state_dict(best_model_state)
                break

    return glove_collator, model


def evaluate(
    model: nn.Module,
    text_collator: NewsGroupsTextCollator,
    config: TextNeuralNetworkConfig,
) -> Dict[str, str | float]:
    """Evaluate a text model and return metrics."""
    # Load datasets
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")

    # Create data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=text_collator,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=text_collator,
    )

    model.to(config.device)
    model.eval()

    # Collect predictions and labels
    train_predictions = []
    train_labels = []
    test_predictions = []
    test_labels = []

    print(f"\nEvaluating model: {config.name}-{config.version}")

    # Get train predictions
    with torch.no_grad():
        for bx, by in tqdm(train_data_loader, desc="Evaluating on train set"):
            preds = model.predict(bx)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(by.numpy())

    # Get test predictions
    with torch.no_grad():
        for bx, by in tqdm(test_data_loader, desc="Evaluating on test set"):
            preds = model.predict(bx)
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(by.numpy())

    # Calculate accuracies
    train_accuracy = sum(p == l for p, l in zip(train_predictions, train_labels)) / len(
        train_labels
    )
    test_accuracy = sum(p == l for p, l in zip(test_predictions, test_labels)) / len(
        test_labels
    )

    print(f"\n{config.name}-{config.version} Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate metrics with macro averaging
    precision = precision_score(test_labels, test_predictions, average="macro")
    recall = recall_score(test_labels, test_predictions, average="macro")
    f1_macro = f1_score(test_labels, test_predictions, average="macro")

    print("\nMetrics (Macro Averaged):")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1_macro:.4f}")

    # Get detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions))

    return {
        "model_name": config.name,
        "model_version": config.version,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
    }


def evaluate_glove(
    model: nn.Module,
    glove_collator: GloVeCollator,
    config: TextNeuralNetworkConfig,
) -> Dict[str, str | float]:
    """Evaluate a GloVe-based model and return metrics."""
    # Load datasets
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")

    # Create data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=glove_collator,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=glove_collator,
    )

    model.to(config.device)
    model.eval()

    # Collect predictions and labels
    train_predictions = []
    train_labels = []
    test_predictions = []
    test_labels = []

    print(f"\nEvaluating model: {config.name}-{config.version}")

    # Get train predictions
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(
            train_data_loader, desc="Evaluating on train set"
        ):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            preds = model.predict(input_ids, attention_mask)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.numpy())

    # Get test predictions
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(
            test_data_loader, desc="Evaluating on test set"
        ):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            preds = model.predict(input_ids, attention_mask)
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    # Calculate accuracies
    train_accuracy = sum(p == l for p, l in zip(train_predictions, train_labels)) / len(
        train_labels
    )
    test_accuracy = sum(p == l for p, l in zip(test_predictions, test_labels)) / len(
        test_labels
    )

    print(f"\n{config.name}-{config.version} Results:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate metrics with macro averaging
    precision = precision_score(test_labels, test_predictions, average="macro")
    recall = recall_score(test_labels, test_predictions, average="macro")
    f1_macro = f1_score(test_labels, test_predictions, average="macro")

    print("\nMetrics (Macro Averaged):")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1 Score:     {f1_macro:.4f}")

    # Get detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions))

    return {
        "model_name": config.name,
        "model_version": config.version,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1_macro,
    }


def find_model_files(models_dir: str = "models/text") -> List[str]:
    """Find all model config files in the models directory."""
    config_files = glob.glob(os.path.join(models_dir, "*-config.pkl"))
    # Extract base paths (without -config.pkl suffix)
    base_paths = [f.replace("-config.pkl", "") for f in config_files]
    return base_paths


def evaluate_all_models(
    models_dir: str = "models/text", output_csv: str = "text_evaluation_results.csv"
):
    """Load and evaluate all text models, saving results to CSV."""
    model_paths = find_model_files(models_dir)

    if not model_paths:
        print(f"No models found in {models_dir}")
        return

    print(f"Found {len(model_paths)} model(s) to evaluate")

    all_results = []

    for model_path in model_paths:
        try:
            # Check if model file exists
            model_file = f"{model_path}-model.pt"
            if not os.path.exists(model_file):
                print(f"Skipping {model_path}: model file not found")
                continue

            # Load config
            with open(f"{model_path}-config.pkl", "rb") as f:
                config = pickle.load(f)

            # Load collator/vectorizer
            with open(f"{model_path}-vectorizer.pkl", "rb") as f:
                collator = pickle.load(f)

            # Detect model type based on config name
            is_glove_model = "glove" in config.name.lower()

            # Create appropriate model class
            if is_glove_model:
                # For GloVe models, we need to recreate with the glove model name
                model = GloVeClassifier(config, glove_model_name="glove-wiki-gigaword-100")
            else:
                # For DeepClassifier (SentenceTransformer-based)
                model = DeepClassifier(config)

            # Load model state dict
            model.load_state_dict(
                torch.load(
                    model_file,
                    map_location=config.device,
                    weights_only=True,
                )
            )

            # Evaluate using appropriate function
            if is_glove_model:
                results = evaluate_glove(model, collator, config)
            else:
                results = evaluate(model, collator, config)

            all_results.append(results)

        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
            continue

    # Save results to CSV
    if all_results:
        csv_filename = output_csv
        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = [
                "model_name",
                "model_version",
                "train_accuracy",
                "test_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

        print(f"\nResults saved to {csv_filename}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate text classifier")
    parser.add_argument(
        "mode",
        choices=["train", "train-glove", "evaluate"],
        help="Mode: train (SentenceTransformer), train-glove (GloVe), or evaluate existing models",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/text",
        help="Directory for model storage (default: models/text)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="text_evaluation_results.csv",
        help="Output CSV file for evaluation results (only for evaluate mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        gelu = nn.GELU()
        focal = FocalLoss()
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        deep_config = TextNeuralNetworkConfig(
            name="static-pretrain-frozen",
            version=get_version_from_pyproject(),
            activate=gelu,
            loss_function=FocalLoss(),
            vocab_size=-1,
            trainable=False,
            epochs=100,
            patience=5,
            learning_rate=1e-5,
            n_embd=256,
            batch_size=8,
            n_hidden_layers=1,
            drop_out=0.1,  
            weight_decay=1e-2,
        )
        deep_model = DeepClassifier(deep_config)
        print("Training deep classifier with text embeddings")
        text_collator, deep_model = train_text_model(deep_model, deep_config)

        # Save the model to models/text/
        save_model(deep_model, text_collator, deep_config, models_dir=args.models_dir)
        print(f"Model saved to {args.models_dir}")
        print("Done")

    elif args.mode == "train-glove":
        gelu = nn.GELU()
        ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Configuration for GloVe
        glove_config = TextNeuralNetworkConfig(
            name="glove-static-frozen",
            version=get_version_from_pyproject(),
            activate=gelu,
            loss_function=FocalLoss(),
            vocab_size=-1,  # Not used for GloVe
            trainable=False, 
            epochs=50,
            patience=5,
            learning_rate=1e-5,
            n_embd=100,  # Must match GloVe dimension
            batch_size=8,
            n_hidden_layers=1,
            drop_out=0.3,
            weight_decay=1e-3,
        )

        print("Initializing GloVe classifier...")
        # Initialize GloVe model
        glove_model = GloVeClassifier(
            glove_config, glove_model_name="glove-wiki-gigaword-100"
        )

        # Create vectorizer and collator
        print("Creating vectorizer and collator...")
        train_dataset = NewsGroupDatasetSimple(subset="train")
        vectorizer = GloVeVectorizer(glove_model.embedding, max_length=128)
        glove_collator = GloVeCollator(train_dataset, vectorizer)

        print("Training GloVe classifier...")
        glove_collator, glove_model = train_glove_model(
            glove_model, glove_config, glove_collator
        )

        # Save the model
        save_model(
            glove_model, glove_collator, glove_config, models_dir=args.models_dir
        )
        print(f"Model saved to {args.models_dir}")
        print("Done")

    elif args.mode == "evaluate":
        evaluate_all_models(models_dir=args.models_dir, output_csv=args.output)


if __name__ == "__main__":
    main()
