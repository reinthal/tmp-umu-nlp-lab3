import argparse
import csv
import glob
import os
import pickle
from typing import Dict, List

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
    NewsGroupDatasetSimple,
    NewsGroupsVectorizerSimple,
)
from src.nn.modules import FocalLoss
from src.nn.simple_classifier import SimpleClassifier
from src.utils import get_version_from_pyproject, save_model


def train(
    model: nn.Module,
    config: TextNeuralNetworkConfig,
):
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")
    # Define our model and vectorizer
    # Uses a vocab size of 1024 tokens
    processor = NewsGroupsVectorizerSimple(train_dataset, n_vocab=config.vocab_size)

    model.to(config.device)
    # Uses Adam optimizer which averages gradients over each batch
    # for smoother learning, sets learning rate to a constant 0.001
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Defines how batches are produced
    # using our custom vectorizer that normalizes reviews
    # to a uniform length using pads
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=processor,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=processor,
    )
    train_losses = []
    test_losses = []
    # Early stopping variables
    best_test_loss = float("inf")
    best_model_state = model.state_dict().copy()
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        # Keep track of the running loss
        # for progress reporting
        running_loss = 0
        running_test_loss = 0
        # iterate over the data using our
        # batched data loader

        # Train
        train_loss = 0
        tqdm_data_loader = tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]"
        )
        for bx, by in tqdm_data_loader:
            optimizer.zero_grad()  # 1. Clear the gradients
            bx = bx.to(config.device)
            by = by.to(config.device)
            output = model(bx)  # 2. Forward pass, compute the output of the model
            loss = config.loss_function(
                output, by
            )  # 3. Compute the loss using cross entropy
            loss.backward()  # 4. Backward pass, compute the gradients
            optimizer.step()  # 5. Update the weights
            train_loss = loss.item()
            running_loss += train_loss
            tqdm_data_loader.set_postfix_str(f"Train Loss: {train_loss:.4f}")

        # Validate
        test_loss = 0
        tqdm_test_data_loader = tqdm(
            test_data_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Test]"
        )
        for bx, by in tqdm_test_data_loader:
            model.eval()
            with torch.no_grad():
                bx = bx.to(config.device)
                by = by.to(config.device)
                output = model(bx)
                loss = config.loss_function(output, by)
                test_loss = loss.item()
                running_test_loss += test_loss
                tqdm_test_data_loader.set_postfix_str(f"Test Loss: {test_loss:.4f}")

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

    return processor, model  # Return the vectorizer and model


def evaluate(
    model: nn.Module,
    vectorizer: NewsGroupsVectorizerSimple,
    config: TextNeuralNetworkConfig,
) -> Dict[str, str | float]:
    """Evaluate a model and return metrics."""
    # Load datasets
    train_dataset = NewsGroupDatasetSimple(subset="train")
    test_dataset = NewsGroupDatasetSimple(subset="test")

    # Create data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=vectorizer,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=vectorizer,
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
            bx = bx.to(config.device)
            preds = model.predict(bx)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(by.numpy())

    # Get test predictions
    with torch.no_grad():
        for bx, by in tqdm(test_data_loader, desc="Evaluating on test set"):
            bx = bx.to(config.device)
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


def find_model_files(models_dir: str = "models") -> List[str]:
    """Find all model config files in the models directory."""
    config_files = glob.glob(os.path.join(models_dir, "*-config.pkl"))
    # Extract base paths (without -config.pkl suffix)
    base_paths = [f.replace("-config.pkl", "") for f in config_files]
    return base_paths


def evaluate_all_models(
    models_dir: str = "models", output_csv: str = "evaluation_results.csv"
):
    """Load and evaluate all models, saving results to CSV."""
    model_paths = find_model_files(models_dir)

    if not model_paths:
        print(f"No models found in {models_dir}")
        return

    print(f"Found {len(model_paths)} model(s) to evaluate")

    all_results = []

    for model_path in model_paths:
        try:
            # Load config
            with open(f"{model_path}-config.pkl", "rb") as f:
                config = pickle.load(f)

            # Load vectorizer
            with open(f"{model_path}-vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)

            # Load model state dict and reconstruct model
            model = SimpleClassifier(config)
            model.load_state_dict(
                torch.load(f"{model_path}-model.pt", map_location=config.device)
            )

            # Evaluate
            results = evaluate(model, vectorizer, config)
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
    parser = argparse.ArgumentParser(description="Train or evaluate simple classifier")
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "test"],
        help="Mode: train a new model or evaluate existing models",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the model after training (only for train mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file for evaluation results (only for evaluate mode)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # Define activation functions and loss functions
        relu = nn.ReLU()
        gelu = nn.GELU()
        ce = nn.CrossEntropyLoss(label_smoothing=0.3)
        focal = FocalLoss()

        version = get_version_from_pyproject()
        vocab_experiments = [
            {
                "name": "simple-vocab-6k",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 6000,
                "drop_out": 0.01
            },
            {
                "name": "simple-vocab-10k",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 10000,
                "drop_out": 0.01
            },
            {
                "name": "simple-vocab-1k",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 2000,
                "drop_out": 0.01
            }
        ]
        layer_variants = [
            {
                "name": "simple-baseline-n-hidden-1",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 8000,
                "drop_out": 0.01
            },
            {
                "name": "simple-baseline-n-hidden-2",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 5e-4,
                "batch_size": 16,
                "drop_out": 0.1,
                "vocab_size": 8000,

                "n_hidden_layers": 2,
            },
            {
                "name": "simple-baseline-n-hidden-3",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "drop_out": 0.3,
                "vocab_size": 8000,

                "n_hidden_layers": 3,
            },
        ]
        # Define configurations for each variation
        configs = layer_variants + [
            # Baseline: GeLU + CrossEntropy + lr=1e-4 16 batch 1 hl
            # RELU variation
            {
                "name": "simple-relu",
                "activate": relu,
                "loss_function": ce,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 8000,

                "drop_out": 0.01
            },
            # FocalLoss variation
            {
                "name": "simple-focal",
                "activate": gelu,
                "loss_function": focal,
                "learning_rate": 1e-4,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 8000,

                "drop_out": 0.01
            },
            # Low learning rate variation
            {
                "name": "simple-eta-1e-5",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-5,
                "batch_size": 16,
                "n_hidden_layers": 1,
                "vocab_size": 8000,

                "drop_out": 0.01
            },
            # High learning rate variation
            {
                "name": "simple-eta-1e-2",
                "activate": gelu,
                "loss_function": ce,
                "learning_rate": 1e-2,
                "batch_size": 48,
                "n_hidden_layers": 1,
                "vocab_size": 8000,
                "drop_out": 0.01
            },
        ]

        # Train each configuration
        for config_params in vocab_experiments:
            simple_config = TextNeuralNetworkConfig(
                version=version,
                use_kaiming_initialization=True,
                epochs=20,
                n_embd=256,
                **config_params,
            )
            simple_model = SimpleClassifier(simple_config)
            print(
                f"\nTraining {config_params['name']} with lr={config_params['learning_rate']}"
            )
            simple_vectorizer, simple_model = train(simple_model, simple_config)

            # Save each model after training
            if args.save:
                save_model(simple_model, simple_vectorizer, simple_config)
                print(f"Saved model: {config_params['name']}-{version}")

        print("\nDone training all models!")
    elif args.mode == "test":
        config = TextNeuralNetworkConfig(
            name="simple-test",
            version=get_version_from_pyproject(),
            activate=nn.GELU(),
            use_kaiming_initialization=True,
            loss_function=FocalLoss(),
            epochs=160,
            patience=10,
            learning_rate=1e-5,
            n_embd=256,
            n_hidden_layers=1,
            vocab_size=2000,
            batch_size=14,
            drop_out=0.01,
        )
        model = SimpleClassifier(config)
        vectorizer, model = train(model, config)
        evaluate(model, vectorizer, config)

    elif args.mode == "evaluate":
        evaluate_all_models(output_csv=args.output)


if __name__ == "__main__":
    main()
