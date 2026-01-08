import pickle
import tomllib
from pathlib import Path
from typing import Any, Tuple

import requests as rq
import torch
from torch import nn


def get_ip() -> str:
    return rq.get("https://api.ipify.org").text


def get_version_from_pyproject() -> str:
    """Read the version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]


def save_model(
    model: nn.Module, vectorizer, config, models_dir: str = "models"
) -> None:
    """Save model, vectorizer, and config to disk.

    Args:
        model: The neural network model to save
        vectorizer: The vectorizer/tokenizer to save
        config: The configuration object
        models_dir: Directory to save models to (default: "models")
    """
    # Create directory if it doesn't exist
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(), f"{models_dir}/{config.name}-{config.version}-model.pt"
    )
    pickle.dump(
        vectorizer,
        open(f"{models_dir}/{config.name}-{config.version}-vectorizer.pkl", "wb"),
    )
    pickle.dump(
        config, open(f"{models_dir}/{config.name}-{config.version}-config.pkl", "wb")
    )


def load_model(path: str) -> Tuple[nn.Module, Any, Any]:
    model = torch.load(path + "-model.pt")
    vectorizer = pickle.load(open(path + "-vectorizer.pkl", "rb"))
    config = pickle.load(open(path + "-config.pkl", "rb"))
    return model, vectorizer, config
