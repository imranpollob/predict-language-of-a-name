"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional
import torch


def plot_training_curve(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot training loss curve.

    Args:
        losses: List of loss values
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (12, 10),
):
    """
    Plot confusion matrix as heatmap.

    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Normalize confusion matrix
    cm_norm = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )

    # Create heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Accuracy"},
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Predicted Language", fontsize=12)
    plt.ylabel("True Language", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    input_words: List[str],
    output_words: List[str],
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (10, 8),
):
    """
    Plot attention weights as heatmap.

    Args:
        attention_weights: Attention weight tensor [output_len, input_len]
        input_words: List of input tokens
        output_words: List of output tokens
        title: Plot title
        save_path: Path to save figure (optional)
        show: Whether to display plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Convert tensor to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=input_words,
        yticklabels=output_words,
        cmap="viridis",
        cbar_kws={"label": "Attention Weight"},
        square=True,
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Input Sequence", fontsize=12)
    plt.ylabel("Output Sequence", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple_curves(
    data_dict: dict,
    title: str = "Comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Value",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot multiple curves on the same plot.

    Args:
        data_dict: Dictionary with {label: values_list}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(10, 6))

    for label, values in data_dict.items():
        plt.plot(values, label=label, linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
