"""
visualization.py — Plotting Utilities for EDA & Model Evaluation
=================================================================

PURPOSE:
    Optional visualization functions for exploring the dataset and
    evaluating model performance. These were partially present in the
    notebook as commented-out code (category distribution plot, confusion
    matrix heatmap). Now they live here as reusable functions.

WHAT EACH FUNCTION DOES:
    - plot_category_distribution(): Bar chart showing how many resumes per category
    - plot_confusion_matrix():      Heatmap showing where the model gets confused
    - plot_model_comparison():      Bar chart comparing accuracy of all models

USAGE:
    from src.visualization import plot_category_distribution, plot_confusion_matrix
    
    plot_category_distribution(df)
    plot_confusion_matrix(y_test, y_pred, label_encoder)
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import CATEGORY_COLUMN


def plot_category_distribution(df, save_path=None):
    """
    Plot a bar chart showing the number of resumes per category.

    This was Cell 5 in the notebook (commented out).

    Args:
        df: DataFrame with a 'Category' column.
        save_path: Optional file path to save the plot (e.g., "plots/categories.png").
    """
    plt.figure(figsize=(15, 6))
    category_counts = df[CATEGORY_COLUMN].value_counts()

    sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
    plt.title("Resume Count per Category", fontsize=16, fontweight="bold")
    plt.xlabel("Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_encoder=None, save_path=None):
    """
    Plot a confusion matrix heatmap.

    This was Cell 14 in the notebook (commented out). A confusion matrix
    shows where the model gets predictions right and wrong:
    - Diagonal = correct predictions (ideally all numbers are here)
    - Off-diagonal = mistakes (predicted one category but it was actually another)

    Args:
        y_true: True labels (from test set).
        y_pred: Predicted labels (from model).
        label_encoder: Optional LabelEncoder to show category names on axes.
        save_path: Optional file path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))

    labels = None
    if label_encoder is not None:
        labels = label_encoder.classes_

    sns.heatmap(
        cm,
        annot=True,          # Show numbers inside each cell
        fmt="d",             # Format as integers
        cmap="Blues",         # Blue color gradient
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved to: {save_path}")

    plt.show()


def plot_model_comparison(results: dict, save_path=None):
    """
    Bar chart comparing accuracy of all trained models.

    Args:
        results: Dict from train_and_select_best_model() — {name: {accuracy: ...}}
        save_path: Optional file path to save the plot.
    """
    names = list(results.keys())
    accuracies = [results[name]["accuracy"] * 100 for name in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"][:len(names)])

    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{acc:.2f}%", ha="center", fontsize=12, fontweight="bold")

    plt.title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(95, 101)  # Zoom in since all models are >95%
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved to: {save_path}")

    plt.show()
