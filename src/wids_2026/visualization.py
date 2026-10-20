"""Visualization helpers for EDA and model interpretation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(target_series, title: str = "Class distribution") -> None:
    """Plot the distribution of the target variable."""
    sns.countplot(x=target_series)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, title: str = "Feature correlation heatmap") -> None:
    """Plot a correlation heatmap for the dataframe."""
    corr = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_names, importances, title: str = "Feature importance") -> None:
    """Plot feature importance values."""
    ranked = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    names = [item[0] for item in ranked[:15]]
    values = [item[1] for item in ranked[:15]]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=names, orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.show()
