"""Reusable plotting functions, shared by the notebook and the asset-export script."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_boiling_point_histogram(df: pd.DataFrame, col: str = "boiling_point_kelvin", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure
    sns.histplot(df[col], ax=ax)
    ax.set_title("Boiling point histogram")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="crest", ax=ax)
    ax.set_title("Heatmap of the dataset")
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(y_true, y_pred, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.scatter(y_true, y_pred, color="blue", label="Predicted vs Actual")
    lims = [min(y_true), max(y_true)]
    ax.plot(lims, lims, color="red", linestyle="--", label="Ideal Line")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title("True vs Predicted Values on Test Dataset")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, residuals, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    sns.scatterplot(x=y_true, y=residuals, color="blue", label="Residuals", ax=ax)
    ax.axhline(0, color="red", linestyle="--", label="Ideal Zero Residuals")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    ax.set_ylim(-400, 400)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    series.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    return fig
