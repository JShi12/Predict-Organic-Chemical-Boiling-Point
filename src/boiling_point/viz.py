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


# Active-learning strategies: model-driven selectors take the first categorical
# slots; random is the reference, drawn as a neutral dashed baseline.
STRATEGY_STYLES = {
    "gp_uncertainty": {"label": "GP uncertainty", "color": "#2a78d6", "linestyle": "-"},
    "bootstrap_xgb": {"label": "Bootstrap XGBoost", "color": "#eb6834", "linestyle": "-"},
    "heuristic": {"label": "Heuristic rule", "color": "#1baf7a", "linestyle": "-"},
    "random": {"label": "Random", "color": "#7a7a75", "linestyle": "--"},
}


def plot_active_learning_curves(results: pd.DataFrame, metric: str = "rmse", ax=None,
                                 title: str = None):
    """Seed-averaged test RMSE of the production ensemble vs number of
    labelled compounds, one line per selection strategy, with a ±1 std band
    across seeds. Legend rather than end labels: every line converges at the
    full pool by design, so end labels would collide."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    summary = results.groupby(["strategy", "n_labels"])[metric].agg(["mean", "std"]).reset_index()
    for strategy, style in STRATEGY_STYLES.items():
        curve = summary[summary["strategy"] == strategy]
        if curve.empty:
            continue
        ax.plot(curve["n_labels"], curve["mean"], color=style["color"], linestyle=style["linestyle"],
                linewidth=2, marker="o", markersize=4, label=style["label"])
        ax.fill_between(curve["n_labels"], curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                        color=style["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Labelled compounds")
    ax.set_ylabel("Test RMSE (K)")
    ax.set_title(title or "Production ensemble test RMSE vs labels")
    ax.grid(True, color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
