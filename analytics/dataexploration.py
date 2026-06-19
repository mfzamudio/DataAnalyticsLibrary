"""
dataexploration
===============

STAGE 3 of the analytics lifecycle: **Exploratory Data Analysis (EDA)**.

Goal of this stage
------------------
Understand the data BEFORE modeling: shape, types, distributions,
relationships, and cardinality. EDA answers "what does this dataset look
like?" and surfaces issues (skew, imbalance, leakage candidates) early.

Note on separation of concerns
-------------------------------
This module computes NUMERIC summaries only and returns DataFrames/Series.
Charts live in :mod:`analytics.datavisualization`. Keeping numbers and plots
separate makes both easier to test and reuse.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    """
    High-level snapshot of the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Keys: ``n_rows``, ``n_cols``, ``memory_mb`` (deep memory usage in MB),
        ``n_duplicated_rows``, ``numeric_cols``, ``categorical_cols``,
        ``datetime_cols``.
    """
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        # ``deep=True`` accounts for the actual memory used by object columns.
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        "n_duplicated_rows": int(df.duplicated().sum()),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "datetime_cols": datetime,
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for numeric columns.

    Goes beyond ``df.describe()`` by adding skewness and kurtosis, which
    quantify asymmetry and tail heaviness of each distribution.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Statistics as rows, numeric columns as columns. Empty DataFrame if
        there are no numeric columns.
    """
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()

    # ``describe`` gives count/mean/std/min/quartiles/max.
    summary = numeric.describe().T
    # Add shape statistics for distribution insight.
    summary["skew"] = numeric.skew()
    summary["kurtosis"] = numeric.kurtosis()
    return summary.round(4)


def categorical_summary(
    df: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Summarize categorical / text columns.

    Parameters
    ----------
    df : pandas.DataFrame
    top_n : int, default 5
        How many top values to list per column in ``top_values``.

    Returns
    -------
    pandas.DataFrame
        One row per categorical column with: ``n_unique``, ``top_value``,
        ``top_freq``, ``top_values`` (a dict of the ``top_n`` most frequent
        values and their counts).
    """
    cat = df.select_dtypes(include=["object", "category", "string"])
    rows = []
    for col in cat.columns:
        counts = cat[col].value_counts(dropna=True)
        rows.append(
            {
                "column": col,
                "n_unique": cat[col].nunique(dropna=True),
                "top_value": counts.index[0] if not counts.empty else None,
                "top_freq": int(counts.iloc[0]) if not counts.empty else 0,
                "top_values": counts.head(top_n).to_dict(),
            }
        )
    # ``set_index`` makes the result easy to read and slice by column name.
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute the correlation matrix of numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
    method : {"pearson", "spearman", "kendall"}, default "pearson"
        * ``pearson``  : linear correlation (assumes roughly linear relations).
        * ``spearman`` : rank correlation (monotonic, robust to outliers).
        * ``kendall``  : rank correlation for small samples / many ties.

    Returns
    -------
    pandas.DataFrame
        Square correlation matrix in ``[-1, 1]``.
    """
    return df.select_dtypes(include="number").corr(method=method).round(4)


def top_correlations(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    List the strongest pairwise correlations above a threshold.

    Flattens the correlation matrix, removes self-correlations and duplicate
    mirror pairs, and keeps pairs whose absolute correlation is >= ``threshold``.
    Useful for spotting redundant features (multicollinearity).

    Parameters
    ----------
    df : pandas.DataFrame
    method : str, default "pearson"
        Correlation method (see :func:`correlation_matrix`).
    threshold : float, default 0.5
        Minimum absolute correlation to report.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature_1``, ``feature_2``, ``correlation``; sorted by
        absolute strength, descending.
    """
    corr = correlation_matrix(df, method=method)
    # Keep only the upper triangle to avoid duplicate (a,b)/(b,a) pairs and the diagonal.
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # ``stack`` turns the matrix into a (pair -> value) Series, dropping NaNs.
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    # Keep pairs above the threshold, then sort by absolute strength (the
    # ``key`` keeps the sign while ordering on magnitude).
    strong = pairs[pairs["correlation"].abs() >= threshold]
    strong = strong.sort_values(
        "correlation", key=lambda s: s.abs(), ascending=False
    )
    return strong.reset_index(drop=True)


def value_counts(
    df: pd.DataFrame,
    column: str,
    *,
    normalize: bool = False,
    dropna: bool = False,
) -> pd.DataFrame:
    """
    Frequency table for a single column.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Column to tabulate.
    normalize : bool, default False
        If ``True`` report proportions (0-1) instead of raw counts.
    dropna : bool, default False
        If ``False`` (default) missing values get their own row, which is
        usually what you want during exploration.

    Returns
    -------
    pandas.DataFrame
        Columns: the value and either ``count`` or ``proportion``.
    """
    counts = df[column].value_counts(normalize=normalize, dropna=dropna)
    name = "proportion" if normalize else "count"
    # Name the index (the category) then the value column for a clean 2-col frame.
    return counts.rename_axis(column).reset_index(name=name)


def summary_report(df: pd.DataFrame) -> dict:
    """
    One-call EDA bundle combining the building blocks above.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Keys: ``overview``, ``numeric_summary``, ``categorical_summary``,
        ``correlations`` (top correlations DataFrame). Designed to be printed
        section by section in a notebook.
    """
    return {
        "overview": overview(df),
        "numeric_summary": numeric_summary(df),
        "categorical_summary": categorical_summary(df),
        "correlations": top_correlations(df, threshold=0.5),
    }
