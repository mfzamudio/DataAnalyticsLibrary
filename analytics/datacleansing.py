"""
datacleansing
=============

STAGE 2 of the analytics lifecycle: **Data Cleansing / Data Quality**.

Goal of this stage
------------------
Take a raw ``DataFrame`` and progressively improve its quality so it is safe
to analyze and model. Each function does ONE thing, takes a DataFrame, and
returns a NEW DataFrame (functions are non-mutating by default).

The typical cleansing checklist, in order
-----------------------------------------
    1. Standardize column names (snake_case, no spaces).
    2. Strip whitespace from text values.
    3. Diagnose missing values (where and how many).
    4. Handle missing values (drop or impute).
    5. Remove duplicate rows.
    6. Fix data types (numbers stored as text, dates as text, ...).
    7. Detect and treat outliers.
    8. (Optional) Produce a before/after quality report.

Non-mutating contract
---------------------
Every transformer copies its input with ``df.copy()`` so the caller's
original DataFrame is never altered. This makes pipelines reproducible and
easy to debug.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Diagnostics (read-only): understand the quality problem before fixing it.
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values per column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    Returns
    -------
    pandas.DataFrame
        One row per column with ``n_missing`` and ``pct_missing`` (0-100),
        sorted by the worst columns first. Columns with no missing values are
        omitted to keep the report focused.
    """
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    # Keep only problematic columns and show the worst first.
    return report[report["n_missing"] > 0].sort_values(
        "n_missing", ascending=False
    )


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact per-column data-quality scorecard.

    For every column reports: dtype, count of non-null values, count and
    percentage of missing values, and number of unique values. This is the
    single most useful "first look" at a dataset's health.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Indexed by column name.
    """
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().sum() / len(df) * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    )


# ---------------------------------------------------------------------------
# 2. Structural cleanup: names, whitespace, types.
# ---------------------------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lower snake_case.

    Steps applied to each name: strip surrounding whitespace -> lowercase ->
    replace any run of non-alphanumeric characters with a single underscore ->
    trim leading/trailing underscores.

    Example: ``" Total Sales ($) "`` -> ``"total_sales"``.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Copy with cleaned column names.
    """
    out = df.copy()

    def _clean(name: str) -> str:
        name = str(name).strip().lower()
        # Collapse any non-alphanumeric sequence into one underscore.
        name = re.sub(r"[^0-9a-z]+", "_", name)
        # Remove leading/trailing underscores left by the substitution.
        return name.strip("_")

    out.columns = [_clean(c) for c in out.columns]
    return out


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from all text (object/string) columns.

    Trailing spaces are a classic cause of "why don't these two values match?"
    bugs in grouping and joins.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Copy with trimmed string values.
    """
    out = df.copy()
    # Select only text-like columns; numeric columns have no whitespace.
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        # ``.str.strip()`` leaves NaN untouched.
        out[col] = out[col].str.strip()
    return out


def convert_dtypes(
    df: pd.DataFrame,
    *,
    to_numeric: Optional[Sequence[str]] = None,
    to_datetime: Optional[Sequence[str]] = None,
    to_category: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Coerce columns to explicit data types.

    Real-world data often stores numbers and dates as text. Fixing dtypes early
    unlocks correct sorting, math, and date arithmetic downstream.

    Parameters
    ----------
    df : pandas.DataFrame
    to_numeric : sequence of str, optional
        Columns to convert with :func:`pandas.to_numeric`. Unparseable values
        become ``NaN`` (``errors="coerce"``).
    to_datetime : sequence of str, optional
        Columns to convert with :func:`pandas.to_datetime`. Unparseable values
        become ``NaT``.
    to_category : sequence of str, optional
        Columns to convert to the memory-efficient ``category`` dtype (ideal
        for low-cardinality text like country or status).

    Returns
    -------
    pandas.DataFrame
        Copy with converted columns.
    """
    out = df.copy()
    for col in to_numeric or []:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in to_datetime or []:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in to_category or []:
        out[col] = out[col].astype("category")
    return out


# ---------------------------------------------------------------------------
# 3. Row-level cleanup: duplicates and missing values.
# ---------------------------------------------------------------------------

def drop_duplicates(
    df: pd.DataFrame,
    *,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
    subset : sequence of str, optional
        Consider only these columns when identifying duplicates. ``None`` uses
        all columns (an exact-row duplicate).
    keep : {"first", "last", False}, default "first"
        Which duplicate to keep. ``False`` drops every duplicated row.

    Returns
    -------
    pandas.DataFrame
        Copy with duplicates removed and the index reset.
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def handle_missing(
    df: pd.DataFrame,
    *,
    strategy: str = "drop_rows",
    columns: Optional[Sequence[str]] = None,
    fill_value=None,
) -> pd.DataFrame:
    """
    Handle missing values with a chosen strategy.

    Parameters
    ----------
    df : pandas.DataFrame
    strategy : str, default "drop_rows"
        One of:
            * ``"drop_rows"``  : drop rows containing any NaN (in ``columns``).
            * ``"drop_cols"``  : drop columns containing any NaN.
            * ``"mean"``       : impute numeric columns with their mean.
            * ``"median"``     : impute numeric columns with their median.
            * ``"mode"``       : impute with the most frequent value (works for
              text too).
            * ``"constant"``   : impute with ``fill_value``.
    columns : sequence of str, optional
        Restrict the operation to these columns. ``None`` = all columns.
    fill_value : Any, optional
        Required when ``strategy="constant"``.

    Returns
    -------
    pandas.DataFrame
        Copy with missing values handled.

    Raises
    ------
    ValueError
        If an unknown strategy is passed, or ``constant`` is used without
        ``fill_value``.
    """
    out = df.copy()
    # Default target = every column.
    cols = list(columns) if columns is not None else list(out.columns)

    if strategy == "drop_rows":
        return out.dropna(subset=cols).reset_index(drop=True)

    if strategy == "drop_cols":
        # Drop only the targeted columns that actually contain NaNs.
        # bool(...) collapses the per-column NaN check to a plain boolean.
        bad = [c for c in cols if bool(out[c].isna().any())]
        return out.drop(columns=bad)

    if strategy in ("mean", "median"):
        # Statistical imputation only makes sense for numeric columns.
        numeric = out[cols].select_dtypes(include="number").columns
        for c in numeric:
            value = out[c].mean() if strategy == "mean" else out[c].median()
            out[c] = out[c].fillna(value)
        return out

    if strategy == "mode":
        for c in cols:
            mode = out[c].mode(dropna=True)
            # ``mode`` can be empty if the column is all-NaN; guard against it.
            if not mode.empty:
                out[c] = out[c].fillna(mode.iloc[0])
        return out

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("strategy='constant' requires a fill_value.")
        out[cols] = out[cols].fillna(fill_value)
        return out

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# 4. Outliers: detect and treat.
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    *,
    factor: float = 1.5,
) -> pd.Series:
    """
    Flag outliers in a numeric column using the IQR (Tukey) rule.

    A value is an outlier if it falls below ``Q1 - factor*IQR`` or above
    ``Q3 + factor*IQR``, where ``IQR = Q3 - Q1``. ``factor=1.5`` is the
    classic boxplot whisker; ``3.0`` flags only extreme outliers.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Numeric column to inspect.
    factor : float, default 1.5
        IQR multiplier controlling sensitivity.

    Returns
    -------
    pandas.Series
        Boolean mask, ``True`` where the row is an outlier.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    # Element-wise comparison yields the boolean mask.
    return (df[column] < lower) | (df[column] > upper)


def cap_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    *,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Treat outliers by WINSORIZING (capping) them to the IQR bounds.

    Instead of deleting rows, extreme values are clipped to the nearest bound.
    This keeps the row count intact while limiting the influence of extremes,
    which is often preferable for downstream modeling.

    Parameters
    ----------
    df : pandas.DataFrame
    column : str
        Numeric column to winsorize.
    factor : float, default 1.5
        IQR multiplier.

    Returns
    -------
    pandas.DataFrame
        Copy with the column's values clipped to ``[lower, upper]``.
    """
    out = df.copy()
    q1 = out[column].quantile(0.25)
    q3 = out[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    # ``clip`` replaces values outside the range with the boundary values.
    out[column] = out[column].clip(lower=lower, upper=upper)
    return out


# ---------------------------------------------------------------------------
# 5. Orchestration: a sensible default pipeline.
# ---------------------------------------------------------------------------

def clean_pipeline(
    df: pd.DataFrame,
    *,
    missing_strategy: str = "drop_rows",
    drop_dupes: bool = True,
) -> pd.DataFrame:
    """
    Run a reasonable default cleansing pipeline end to end.

    Order of operations:
        1. Standardize column names.
        2. Strip whitespace from text columns.
        3. Drop duplicate rows (optional).
        4. Handle missing values with the chosen strategy.

    This is a convenience entry point; for fine control call the individual
    functions in the order your data needs.

    Parameters
    ----------
    df : pandas.DataFrame
    missing_strategy : str, default "drop_rows"
        Passed straight to :func:`handle_missing`.
    drop_dupes : bool, default True
        Whether to remove duplicate rows.

    Returns
    -------
    pandas.DataFrame
        The cleaned copy.
    """
    out = standardize_column_names(df)
    out = strip_whitespace(out)
    if drop_dupes:
        out = drop_duplicates(out)
    out = handle_missing(out, strategy=missing_strategy)
    return out
