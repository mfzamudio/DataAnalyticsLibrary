"""
datavisualization
==================

STAGE 4 of the analytics lifecycle: **Data Visualization**.

Goal of this stage
------------------
Turn data into charts for human understanding. Split into two tiers:

* **Basic plots**     : histogram, bar, line, scatter, boxplot, count plot.
* **Advanced plots**  : correlation heatmap, pairplot, distribution+KDE,
                        violin plot, missing-value matrix, time-series
                        decomposition, feature-importance bar.

Design contract
---------------
* Built on ``matplotlib`` (+ ``seaborn`` for the statistical charts).
* Every function ACCEPTS an optional ``ax`` (matplotlib Axes). If none is
  given, one is created. This lets callers compose subplots and lets the
  notebook control figure size.
* Every function RETURNS the Axes (or Figure) so it can be further customized
  or saved with ``fig.savefig(...)``.
* Functions draw but do not call ``plt.show()``; the caller (or Jupyter's
  inline backend) decides when to render. Use :func:`save_figure` to persist.

These functions raise ``ImportError`` with a clear message if matplotlib /
seaborn are not installed, since plotting is an optional concern.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

# Plotting libraries are imported lazily inside a helper so that importing this
# module never fails on a headless / minimal install. We resolve them once.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _PLOTTING_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    _PLOTTING_AVAILABLE = False


def _require_plotting() -> None:
    """Raise a helpful error if matplotlib/seaborn are missing."""
    if not _PLOTTING_AVAILABLE:
        raise ImportError(
            "Plotting requires matplotlib and seaborn. Install them with "
            "`pip install matplotlib seaborn`."
        )


def _get_ax(ax, figsize):
    """Return ``ax`` if given, otherwise create a new figure+axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def set_style(style: str = "whitegrid", context: str = "notebook") -> None:
    """
    Apply a consistent seaborn look across all charts.

    Parameters
    ----------
    style : str, default "whitegrid"
        Seaborn style (``"white"``, ``"darkgrid"``, ``"ticks"``, ...).
    context : str, default "notebook"
        Scales fonts/lines for the medium (``"paper"``, ``"talk"``,
        ``"poster"``).
    """
    _require_plotting()
    sns.set_theme(style=style, context=context)


# ===========================================================================
# BASIC PLOTS
# ===========================================================================

def plot_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 30,
    ax=None,
    figsize=(8, 5),
):
    """
    Histogram of a numeric column (distribution of values).

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Numeric column to plot.
    bins : int, default 30
        Number of histogram bins.
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    ax.hist(df[column].dropna(), bins=bins, edgecolor="white")
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return ax


def plot_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    ax=None,
    figsize=(8, 5),
):
    """
    Bar chart of an aggregated/categorical relationship (``y`` per ``x``).

    Parameters
    ----------
    df : pandas.DataFrame
        Typically an already-aggregated frame (e.g. sales by region).
    x : str
        Category column (bars).
    y : str
        Numeric column (bar height).
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(f"{y} by {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # Rotate labels so long category names stay readable.
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_line(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    ax=None,
    figsize=(10, 5),
):
    """
    Line chart, typically for a metric over time.

    Parameters
    ----------
    df : pandas.DataFrame
    x : str
        X axis column (often a datetime).
    y : str
        Y axis numeric column.
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    ax.plot(df[x], df[y], marker="o", markersize=3, linewidth=1.2)
    ax.set_title(f"{y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def plot_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ax=None,
    figsize=(8, 6),
):
    """
    Scatter plot of two numeric columns to inspect their relationship.

    Parameters
    ----------
    df : pandas.DataFrame
    x, y : str
        Numeric columns for the two axes.
    hue : str, optional
        Categorical column used to color points (adds a third dimension).
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 6)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    # seaborn handles the optional hue + legend cleanly.
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f"{y} vs {x}")
    return ax


def plot_box(
    df: pd.DataFrame,
    *,
    y: str,
    x: Optional[str] = None,
    ax=None,
    figsize=(8, 5),
):
    """
    Boxplot to visualize spread and outliers of a numeric column, optionally
    split by a category.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
        Numeric column.
    x : str, optional
        Category column to split the boxes by.
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    sns.boxplot(data=df, x=x, y=y, ax=ax)
    title = f"Boxplot of {y}" + (f" by {x}" if x else "")
    ax.set_title(title)
    return ax


def plot_count(
    df: pd.DataFrame,
    column: str,
    *,
    top_n: Optional[int] = None,
    ax=None,
    figsize=(8, 5),
):
    """
    Count plot (bar chart of category frequencies).

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Categorical column to count.
    top_n : int, optional
        Show only the ``top_n`` most frequent categories (useful for
        high-cardinality columns).
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    # Determine display order (most frequent first), optionally truncated.
    order = df[column].value_counts().index
    if top_n is not None:
        order = order[:top_n]
    sns.countplot(data=df, x=column, order=order, ax=ax)
    ax.set_title(f"Counts of {column}")
    ax.tick_params(axis="x", rotation=45)
    return ax


# ===========================================================================
# ADVANCED PLOTS
# ===========================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    annot: bool = True,
    ax=None,
    figsize=(10, 8),
):
    """
    Heatmap of the numeric correlation matrix.

    The fastest way to spot multicollinearity and feature relationships.

    Parameters
    ----------
    df : pandas.DataFrame
    method : {"pearson", "spearman", "kendall"}, default "pearson"
    annot : bool, default True
        Write the correlation value inside each cell.
    ax : matplotlib Axes, optional
    figsize : tuple, default (10, 8)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    corr = df.select_dtypes(include="number").corr(method=method)
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap="coolwarm",
        center=0,  # so 0 correlation maps to the neutral color
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(f"Correlation heatmap ({method})")
    return ax


def plot_pairplot(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    hue: Optional[str] = None,
):
    """
    Pairplot: scatter matrix of numeric columns with histograms on the diagonal.

    Note: this creates its OWN figure (seaborn ``PairGrid``) and therefore does
    not accept an ``ax``. Best used on a handful of columns; it scales O(n^2).

    Parameters
    ----------
    df : pandas.DataFrame
    columns : sequence of str, optional
        Restrict to these columns to keep the grid readable.
    hue : str, optional
        Categorical column to color the points by.

    Returns
    -------
    seaborn.axisgrid.PairGrid
    """
    _require_plotting()
    data = df[list(columns)] if columns is not None else df
    # Re-attach hue column if it was excluded by the column selection.
    if hue is not None and hue not in data.columns:
        data = data.join(df[hue])
    return sns.pairplot(data, hue=hue)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    kde: bool = True,
    ax=None,
    figsize=(8, 5),
):
    """
    Distribution plot: histogram overlaid with a KDE (smoothed density) curve.

    Richer than a plain histogram for judging shape, modality and skew.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Numeric column.
    kde : bool, default True
        Overlay the kernel density estimate.
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    sns.histplot(data=df, x=column, kde=kde, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return ax


def plot_violin(
    df: pd.DataFrame,
    *,
    y: str,
    x: Optional[str] = None,
    ax=None,
    figsize=(8, 5),
):
    """
    Violin plot: a boxplot fused with a KDE, showing full distribution shape
    per category.

    Parameters
    ----------
    df : pandas.DataFrame
    y : str
        Numeric column.
    x : str, optional
        Category column to split by.
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 5)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    sns.violinplot(data=df, x=x, y=y, ax=ax)
    title = f"Violin plot of {y}" + (f" by {x}" if x else "")
    ax.set_title(title)
    return ax


def plot_missing_matrix(
    df: pd.DataFrame,
    *,
    ax=None,
    figsize=(12, 6),
):
    """
    Visualize the pattern of missing values across the dataset.

    Renders a heatmap where missing cells are highlighted, making it easy to
    see whether nulls are random or clustered in particular rows/columns.

    Parameters
    ----------
    df : pandas.DataFrame
    ax : matplotlib Axes, optional
    figsize : tuple, default (12, 6)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    # ``isna()`` -> boolean matrix; True (missing) shows as the highlighted color.
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing-value matrix (bright = missing)")
    return ax


def plot_time_series_decomposition(
    df: pd.DataFrame,
    *,
    value_column: str,
    date_column: Optional[str] = None,
    period: int = 12,
    model: str = "additive",
    figsize=(12, 8),
):
    """
    Decompose a time series into trend, seasonal, and residual components.

    Wraps ``statsmodels.tsa.seasonal.seasonal_decompose``. Requires
    ``statsmodels`` (in requirements.txt).

    Parameters
    ----------
    df : pandas.DataFrame
    value_column : str
        Numeric column containing the series values.
    date_column : str, optional
        Datetime column to use as the index. If ``None``, the existing index
        is assumed to be the time axis.
    period : int, default 12
        Number of observations per seasonal cycle (12 = monthly-in-year,
        7 = daily-in-week, ...).
    model : {"additive", "multiplicative"}, default "additive"
        Decomposition model.
    figsize : tuple, default (12, 8)

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If statsmodels is not installed.
    """
    _require_plotting()
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Time-series decomposition requires statsmodels. Install it with "
            "`pip install statsmodels`."
        ) from exc

    # Build a clean, time-indexed, sorted series for decomposition.
    series = df.copy()
    if date_column is not None:
        series = series.set_index(date_column)
    series = series[value_column].sort_index()

    result = seasonal_decompose(series, model=model, period=period)
    # statsmodels returns a result object with its own ``.plot()``.
    fig = result.plot()
    fig.set_size_inches(*figsize)
    fig.suptitle(f"Time-series decomposition of {value_column} ({model})")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importances,
    feature_names: Sequence[str],
    *,
    top_n: int = 20,
    ax=None,
    figsize=(8, 6),
):
    """
    Horizontal bar chart of model feature importances.

    Pairs naturally with :func:`analytics.machinelearning` /
    :func:`analytics.predictiveanalysis` outputs (e.g. a fitted tree model's
    ``.feature_importances_`` or a linear model's coefficients).

    Parameters
    ----------
    importances : array-like
        Importance score per feature (same order as ``feature_names``).
    feature_names : sequence of str
        Names aligned with ``importances``.
    top_n : int, default 20
        Show only the ``top_n`` most important features.
    ax : matplotlib Axes, optional
    figsize : tuple, default (8, 6)

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_plotting()
    ax = _get_ax(ax, figsize)
    # Sort features by importance and keep the top ``top_n``.
    imp = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        # Reverse so the largest bar appears at the TOP of a horizontal chart.
        .iloc[::-1]
    )
    ax.barh(imp.index.astype(str), imp.values)
    ax.set_title(f"Top {min(top_n, len(imp))} feature importances")
    ax.set_xlabel("Importance")
    return ax


def save_figure(fig, path: str, *, dpi: int = 150) -> str:
    """
    Save a matplotlib Figure to disk with sensible defaults.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save. If you only have an Axes ``ax``, pass ``ax.figure``.
    path : str
        Output path (extension decides the format: ``.png``, ``.pdf``, ...).
    dpi : int, default 150
        Resolution in dots per inch.

    Returns
    -------
    str
        The path the figure was written to.
    """
    _require_plotting()
    # ``bbox_inches="tight"`` trims surrounding whitespace.
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
