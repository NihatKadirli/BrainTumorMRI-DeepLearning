"""
visualizer.py
-------------
Egitim ve degerlendirme sonuclari icin grafik uretim modulu.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from . import config


# Tutarli, profesyonel gorunum
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "DejaVu Sans"


def plot_training_curves(history: Dict, model_name: str, save_path: Path) -> None:
    """Egitim/validation loss ve accuracy egrilerini cizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["loss"]) + 1)

    axes[0].plot(epochs, history["loss"], "b-", label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Kayip Egrisi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], "b-", label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Dogruluk Egrisi")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: Path) -> None:
    """Confusion matrix'i heatmap olarak cizer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={"label": "Ornek Sayisi"},
        ax=ax,
    )
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gercek Sinif")
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(comparison_df: pd.DataFrame, save_path: Path) -> None:
    """Tum modellerin metrik karsilastirmasini bar chart ile gosterir."""
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    df_plot = comparison_df[["Model"] + metrics].copy()

    df_melted = df_plot.melt(id_vars="Model", var_name="Metrik", value_name="Deger")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Model", y="Deger", hue="Metrik",
                palette="viridis", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Karsilastirma - Temel Metrikler")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(distribution: Dict, save_path: Path) -> None:
    """Sinif dagilimini train/test icin bar chart ile gosterir."""
    classes = config.CLASS_NAMES
    train_counts = [distribution["train"].get(c, 0) for c in classes]
    test_counts = [distribution["test"].get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, train_counts, width, label="Training", color="#2E86AB")
    b2 = ax.bar(x + width/2, test_counts, width, label="Testing", color="#E63946")

    ax.set_xlabel("Sinif")
    ax.set_ylabel("Ornek Sayisi")
    ax.set_title("Veri Seti Sinif Dagilimi")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for container in [b1, b2]:
        ax.bar_label(container, fontsize=10, padding=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_training_curves(all_histories: Dict[str, Dict], save_path: Path) -> None:
    """Tum modellerin validation accuracy egrilerini tek grafikte gosterir."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]

    for (name, h), c in zip(all_histories.items(), colors):
        epochs = range(1, len(h["val_accuracy"]) + 1)
        axes[0].plot(epochs, h["val_accuracy"], label=name, linewidth=2.5, color=c)
        axes[1].plot(epochs, h["val_loss"], label=name, linewidth=2.5, color=c)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Tum Modellerin Validation Accuracy'si")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Tum Modellerin Validation Loss'u")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
