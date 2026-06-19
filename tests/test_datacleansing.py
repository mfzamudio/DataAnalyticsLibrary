"""Tests for analytics.datacleansing."""

import numpy as np
import pandas as pd
import pytest

from analytics import datacleansing


def test_standardize_column_names(messy_df):
    out = datacleansing.standardize_column_names(messy_df)
    assert list(out.columns) == ["order_id", "region", "sales", "units_sold"]


def test_standardize_does_not_mutate_input(messy_df):
    original = list(messy_df.columns)
    datacleansing.standardize_column_names(messy_df)
    # The original frame must be untouched (non-mutating contract).
    assert list(messy_df.columns) == original


def test_strip_whitespace(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    out = datacleansing.strip_whitespace(df)
    assert out["region"].tolist()[0] == "North"  # was " North "
    assert "West" in out["region"].tolist()


def test_missing_value_report(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    report = datacleansing.missing_value_report(df)
    # Only 'sales' has missing values.
    assert "sales" in report.index
    assert report.loc["sales", "n_missing"] == 1


def test_quality_report_shape(clean_df):
    report = datacleansing.quality_report(clean_df)
    assert set(["dtype", "non_null", "n_missing", "pct_missing", "n_unique"]).issubset(
        report.columns
    )
    assert len(report) == clean_df.shape[1]


def test_drop_duplicates(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    out = datacleansing.drop_duplicates(df)
    # messy_df has 6 rows, one being a duplicate of the first.
    assert len(out) == 5


def test_handle_missing_drop_rows(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    out = datacleansing.handle_missing(df, strategy="drop_rows")
    assert out["sales"].isna().sum() == 0
    assert len(out) < len(df)


def test_handle_missing_median(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    out = datacleansing.handle_missing(df, strategy="median")
    assert out["sales"].isna().sum() == 0


def test_handle_missing_constant_requires_value(clean_df):
    with pytest.raises(ValueError):
        datacleansing.handle_missing(clean_df, strategy="constant")


def test_handle_missing_unknown_strategy(clean_df):
    with pytest.raises(ValueError):
        datacleansing.handle_missing(clean_df, strategy="nonsense")


def test_convert_dtypes():
    df = pd.DataFrame({"n": ["1", "2", "x"], "d": ["2023-01-01", "bad", "2023-03-01"]})
    out = datacleansing.convert_dtypes(df, to_numeric=["n"], to_datetime=["d"])
    assert pd.api.types.is_numeric_dtype(out["n"])
    # The unparseable "x" becomes NaN.
    assert out["n"].isna().sum() == 1
    assert pd.api.types.is_datetime64_any_dtype(out["d"])


def test_detect_outliers_iqr(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    mask = datacleansing.detect_outliers_iqr(df, "sales")
    # The 999999 value must be flagged as an outlier.
    assert mask.sum() >= 1
    assert df.loc[mask, "sales"].max() == 999999.0


def test_cap_outliers_iqr_reduces_max(messy_df):
    df = datacleansing.standardize_column_names(messy_df)
    out = datacleansing.cap_outliers_iqr(df, "sales")
    # Capping must bring the extreme value down.
    assert out["sales"].max() < 999999.0


def test_clean_pipeline(messy_df):
    out = datacleansing.clean_pipeline(messy_df, missing_strategy="median")
    assert list(out.columns) == ["order_id", "region", "sales", "units_sold"]
    assert out.isna().sum().sum() == 0
    # Duplicate removed.
    assert len(out) == 5
