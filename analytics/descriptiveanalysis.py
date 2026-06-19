"""
descriptiveanalysis
===================

STAGE 5 of the analytics lifecycle: **Descriptive Analytics** -> "What happened?"

Goal of this stage
------------------
Summarize the past: aggregate metrics, KPIs, group comparisons, trends over
time. Descriptive analytics is the reporting layer that most dashboards live
on. It does NOT explain causes (that is diagnostic) or forecast (predictive).

Methodology / steps
-------------------
    1. Define the metric(s) of interest (sum, mean, count, ...).
    2. Aggregate at the right grain (overall, by group, by time bucket).
    3. Rank and compare (top-N, growth, share of total).
    4. Track trends over time (resampling, moving averages).

All functions take a DataFrame and return a DataFrame/Series of results.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import pandas as pd


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard descriptive statistics (count, mean, std, min, quartiles, max)
    for every numeric column.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Statistics transposed so each column is a row.
    """
    return df.describe().T


def aggregate(
    df: pd.DataFrame,
    *,
    group_by: Union[str, Sequence[str]],
    metrics: dict[str, Union[str, Sequence[str]]],
) -> pd.DataFrame:
    """
    Group-and-aggregate: the core descriptive operation.

    Parameters
    ----------
    df : pandas.DataFrame
    group_by : str or sequence of str
        Column(s) defining the groups (the "by" dimension).
    metrics : dict
        Mapping of ``column -> aggregation(s)``, e.g.
        ``{"sales": "sum", "units": ["mean", "max"]}``.

    Returns
    -------
    pandas.DataFrame
        Aggregated results with a flat column index and the group keys as
        regular columns (index reset for easy downstream use).
    """
    grouped = df.groupby(group_by, dropna=False).agg(metrics)
    # ``.agg`` with list values produces a MultiIndex on columns; flatten it
    # to "column_aggfunc" so the result is easy to read and plot.
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = ["_".join(map(str, c)).strip("_") for c in grouped.columns]
    return grouped.reset_index()


def top_n(
    df: pd.DataFrame,
    *,
    by: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return the top (or bottom) ``n`` rows ranked by a column.

    Parameters
    ----------
    df : pandas.DataFrame
    by : str
        Column to rank on.
    n : int, default 10
        How many rows to return.
    ascending : bool, default False
        ``False`` = largest first (top). ``True`` = smallest first (bottom).

    Returns
    -------
    pandas.DataFrame
    """
    return df.sort_values(by=by, ascending=ascending).head(n).reset_index(drop=True)


def share_of_total(
    df: pd.DataFrame,
    *,
    group_by: Union[str, Sequence[str]],
    value: str,
) -> pd.DataFrame:
    """
    Each group's contribution as a percentage of the grand total.

    Parameters
    ----------
    df : pandas.DataFrame
    group_by : str or sequence of str
        Grouping dimension(s).
    value : str
        Numeric column to sum and convert to a share.

    Returns
    -------
    pandas.DataFrame
        Columns: the group key(s), ``<value>`` (group sum), and ``pct_of_total``
        (0-100), sorted by share descending.
    """
    grouped = df.groupby(group_by, dropna=False)[value].sum().reset_index()
    total = grouped[value].sum()
    # Guard against divide-by-zero on an all-zero metric.
    grouped["pct_of_total"] = (
        (grouped[value] / total * 100).round(2) if total else 0.0
    )
    return grouped.sort_values("pct_of_total", ascending=False).reset_index(drop=True)


def trend_over_time(
    df: pd.DataFrame,
    *,
    date_column: str,
    value: str,
    freq: str = "M",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Resample a metric onto a regular time grid to reveal its trend.

    Parameters
    ----------
    df : pandas.DataFrame
    date_column : str
        Datetime column (will be coerced to datetime if needed).
    value : str
        Numeric column to aggregate.
    freq : str, default "M"
        Pandas offset alias for the bucket size: ``"D"`` daily, ``"W"`` weekly,
        ``"M"`` month-end, ``"Q"`` quarterly, ``"Y"`` yearly.
    agg : str, default "sum"
        Aggregation applied within each bucket (``"sum"``, ``"mean"``, ...).

    Returns
    -------
    pandas.DataFrame
        Columns: ``date_column`` (period start) and ``value``.
    """
    out = df.copy()
    # Ensure the date column is a real datetime so resampling works.
    out[date_column] = pd.to_datetime(out[date_column], errors="coerce")
    # ``resample`` requires a datetime index.
    series = out.set_index(date_column)[value].resample(freq).agg(agg)
    return series.reset_index()


def moving_average(
    df: pd.DataFrame,
    *,
    value: str,
    window: int = 3,
) -> pd.DataFrame:
    """
    Add a simple moving average (rolling mean) to smooth a series.

    Parameters
    ----------
    df : pandas.DataFrame
        Assumed already sorted by time.
    value : str
        Numeric column to smooth.
    window : int, default 3
        Number of periods in the rolling window.

    Returns
    -------
    pandas.DataFrame
        Copy with a new column ``<value>_ma<window>``.
    """
    out = df.copy()
    out[f"{value}_ma{window}"] = out[value].rolling(window=window).mean()
    return out


def pivot_summary(
    df: pd.DataFrame,
    *,
    index: Union[str, Sequence[str]],
    columns: Union[str, Sequence[str]],
    values: str,
    aggfunc: str = "sum",
) -> pd.DataFrame:
    """
    Cross-tabulate a metric across two dimensions (a pivot table).

    Parameters
    ----------
    df : pandas.DataFrame
    index : str or sequence of str
        Row dimension(s).
    columns : str or sequence of str
        Column dimension(s).
    values : str
        Numeric column to aggregate in the cells.
    aggfunc : str, default "sum"
        Cell aggregation.

    Returns
    -------
    pandas.DataFrame
        The pivot table (missing combinations filled with 0).
    """
    return pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=0,
    )
