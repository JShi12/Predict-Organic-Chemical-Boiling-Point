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


# NIST lookup replay: model-driven orders take categorical slots in fixed
# order; the rule's own order is the fourth slot; random is the neutral baseline.
LOOKUP_ORDER_STYLES = {
    "feasibility_weighted": {"label": "Uncertainty × feasibility", "color": "#2a78d6", "linestyle": "-"},
    "feasibility": {"label": "Feasibility only", "color": "#eb6834", "linestyle": "-"},
    "uncertainty": {"label": "Uncertainty only", "color": "#1baf7a", "linestyle": "-"},
    "rule_order": {"label": "Heuristic run (actual order)", "color": "#eda100", "linestyle": "-"},
    "random": {"label": "Random order", "color": "#7a7a75", "linestyle": "--"},
}


def plot_lookup_replay(results: pd.DataFrame, ax=None):
    """Cumulative NIST hits vs lookups for each ordering of the same pool,
    mean across seeds with a ±1 std band."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.2))
    else:
        fig = ax.figure
    summary = results.groupby(["strategy", "lookups"])["hits"].agg(["mean", "std"]).reset_index()
    for strategy, style in LOOKUP_ORDER_STYLES.items():
        c = summary[summary["strategy"] == strategy]
        if c.empty:
            continue
        ax.plot(c["lookups"], c["mean"], color=style["color"], linestyle=style["linestyle"],
                linewidth=2, label=style["label"])
        ax.fill_between(c["lookups"], c["mean"] - c["std"].fillna(0), c["mean"] + c["std"].fillna(0),
                        color=style["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("NIST lookups")
    ax.set_ylabel("Boiling points found")
    ax.set_title("Same 5,400 candidates, different lookup orders")
    ax.grid(True, color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))  # curves fill the plot area
    fig.tight_layout()
    return fig


# Bayesian-optimisation strategies: the two BO acquisitions take the first
# categorical slots, then the non-exploring baselines; random is neutral.
BO_STYLES = {
    "prob_in_spec": {"label": "BO: probability in spec", "color": "#2a78d6", "linestyle": "-"},
    "qlogei_target": {"label": "BO: qLogEI toward target", "color": "#eb6834", "linestyle": "-"},
    "greedy_gp": {"label": "Greedy GP (no exploration)", "color": "#1baf7a", "linestyle": "-"},
    "mw_heuristic": {"label": "Molecular-weight rule", "color": "#eda100", "linestyle": "-"},
    "random": {"label": "Random", "color": "#7a7a75", "linestyle": "--"},
}


def plot_bo_campaign(results: pd.DataFrame, n_in_spec: int, ax=None, title: str = None):
    """Cumulative in-spec compounds found vs experiments, mean ± 1 std
    across seeds, with the best possible (every experiment in spec) as a
    dotted reference."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.2))
    else:
        fig = ax.figure
    summary = results.groupby(["strategy", "experiment"])["hits"].agg(["mean", "std"]).reset_index()
    n_max = int(results["experiment"].max())
    ax.plot([0, n_max], [0, min(n_max, n_in_spec)], color="#c9c8c2", linestyle=":", linewidth=1.5,
            label="Every experiment in spec")
    for strategy, style in BO_STYLES.items():
        c = summary[summary["strategy"] == strategy]
        if c.empty:
            continue
        ax.plot(c["experiment"], c["mean"], color=style["color"], linestyle=style["linestyle"],
                linewidth=2, label=style["label"])
        ax.fill_between(c["experiment"], c["mean"] - c["std"], c["mean"] + c["std"],
                        color=style["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Experiments (boiling points measured)")
    ax.set_ylabel("In-spec compounds found")
    ax.set_title(title or "Finding compounds that meet a boiling-point spec")
    ax.grid(True, color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig
