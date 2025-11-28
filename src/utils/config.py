"""
Configuration management utilities.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_preference: str = "auto") -> torch.device:
    """
    Get PyTorch device (CPU or CUDA).

    Args:
        device_preference: "auto", "cuda", or "cpu"

    Returns:
        torch.device object
    """
    if device_preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_preference)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist.

    Args:
        directory: Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
