"""
diagnosticanalysis
==================

STAGE 6 of the analytics lifecycle: **Diagnostic Analytics** -> "Why did it
happen?"

Goal of this stage
------------------
Explain the patterns that descriptive analytics surfaced: which factors move
with the outcome, whether group differences are statistically real, and how
the metric changed between two periods (drill-down / variance analysis).

IMPORTANT caveat on causality
-----------------------------
Diagnostic analytics finds ASSOCIATIONS, not proof of causation. Correlation
and a significant test tell you variables move together; they do not prove one
caused the other. Treat results as hypotheses to validate, not verdicts.

Methodology / steps
-------------------
    1. Correlate candidate drivers with the outcome.
    2. Segment / compare groups and test whether differences are significant.
    3. Decompose change between two periods (contribution analysis).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def correlation_with_target(
    df: pd.DataFrame,
    target: str,
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Rank numeric features by their correlation with a target variable.

    A first, fast screen for "what moves with the outcome?".

    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        Numeric outcome column.
    method : {"pearson", "spearman", "kendall"}, default "pearson"

    Returns
    -------
    pandas.DataFrame
        Columns ``feature`` and ``correlation``, sorted by absolute strength
        (the target itself is excluded).
    """
    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        raise ValueError(f"Target '{target}' must be a numeric column.")
    corr = numeric.corr(method=method)[target].drop(labels=[target])
    # Series.reset_index(name=...) names the value column; rename_axis names the index.
    result = corr.rename_axis("feature").reset_index(name="correlation")
    return result.reindex(
        result["correlation"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def group_comparison(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    """
    Compare a numeric metric across the levels of a categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
    group_column : str
        Categorical column defining the groups.
    value_column : str
        Numeric metric to compare.

    Returns
    -------
    pandas.DataFrame
        Per-group count/mean/median/std/min/max, sorted by mean descending.
    """
    summary = (
        df.groupby(group_column, dropna=False)[value_column]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
    )
    return summary.round(4)


def ttest_two_groups(
    df: pd.DataFrame,
    *,
    group_column: str,
    value_column: str,
    group_a,
    group_b,
    equal_var: bool = False,
) -> dict:
    """
    Independent two-sample t-test: is the mean of ``value_column`` different
    between two groups?

    Wraps ``scipy.stats.ttest_ind``. By default uses Welch's t-test
    (``equal_var=False``), which does not assume equal variances and is the
    safer default.

    Parameters
    ----------
    df : pandas.DataFrame
    group_column : str
        Column identifying the groups.
    value_column : str
        Numeric column being compared.
    group_a, group_b : Any
        The two group labels to compare.
    equal_var : bool, default False
        If ``True`` performs Student's t-test (assumes equal variance).

    Returns
    -------
    dict
        ``t_statistic``, ``p_value``, ``mean_a``, ``mean_b``, ``n_a``, ``n_b``,
        and ``significant_at_0.05`` (bool). Interpretation: p < 0.05 suggests
        the difference is unlikely to be due to chance alone.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise ImportError("ttest_two_groups requires scipy.") from exc

    a = df.loc[df[group_column] == group_a, value_column].dropna()
    b = df.loc[df[group_column] == group_b, value_column].dropna()
    # Coerce to plain floats up front (scipy returns a typed result tuple).
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
    t_stat, p_value = float(t_stat), float(p_value)
    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def chi_square_independence(
    df: pd.DataFrame,
    *,
    column_a: str,
    column_b: str,
) -> dict:
    """
    Chi-square test of independence between two categorical columns.

    Tests whether the two categories are associated (dependent) or independent.

    Parameters
    ----------
    df : pandas.DataFrame
    column_a, column_b : str
        The two categorical columns.

    Returns
    -------
    dict
        ``chi2``, ``p_value``, ``degrees_of_freedom``, and
        ``significant_at_0.05`` (bool). p < 0.05 suggests the two variables are
        NOT independent (i.e. they are associated).

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise ImportError("chi_square_independence requires scipy.") from exc

    # Build the contingency table of joint frequencies.
    contingency = pd.crosstab(df[column_a], df[column_b])
    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency)
    # Coerce to plain scalars (scipy returns typed/array-like values).
    chi2, p_value, dof = float(chi2), float(p_value), int(dof)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant_at_0.05": bool(p_value < 0.05),
    }


def contribution_to_change(
    df: pd.DataFrame,
    *,
    dimension: str,
    value: str,
    period_column: str,
    period_from,
    period_to,
) -> pd.DataFrame:
    """
    Decompose the change in a metric between two periods by dimension.

    Classic variance / drill-down analysis: "total sales fell 8% — which
    regions drove that?". For each dimension level it reports the value in each
    period, the absolute delta, and that level's share of the total change.

    Parameters
    ----------
    df : pandas.DataFrame
    dimension : str
        Breakdown dimension (e.g. region, product).
    value : str
        Numeric metric.
    period_column : str
        Column identifying the period (e.g. year, month).
    period_from, period_to : Any
        The baseline and comparison period labels.

    Returns
    -------
    pandas.DataFrame
        Columns: ``dimension``, ``value_from``, ``value_to``, ``delta``,
        ``pct_contribution_to_change`` (share of total delta, 0-100), sorted by
        absolute contribution descending.
    """
    # Sum the metric per dimension level within each of the two periods.
    sub = df[df[period_column].isin([period_from, period_to])]
    pivot = (
        sub.groupby([dimension, period_column], dropna=False)[value]
        .sum()
        .unstack(period_column)
        .fillna(0)
    )
    # Defensive: ensure both period columns exist even if one had no rows.
    for p in (period_from, period_to):
        if p not in pivot.columns:
            pivot[p] = 0.0

    result = pd.DataFrame(
        {
            "value_from": pivot[period_from],
            "value_to": pivot[period_to],
        }
    )
    result["delta"] = result["value_to"] - result["value_from"]
    total_delta = result["delta"].sum()
    result["pct_contribution_to_change"] = (
        (result["delta"] / total_delta * 100).round(2) if total_delta else 0.0
    )
    result = result.sort_values(
        "delta", key=lambda s: s.abs(), ascending=False
    )
    return result.reset_index()
