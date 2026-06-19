"""Tests for analytics.dataexploration."""

import pandas as pd

from analytics import dataexploration


def test_overview_keys(clean_df):
    ov = dataexploration.overview(clean_df)
    assert ov["n_rows"] == 6
    assert ov["n_cols"] == 5
    assert "sales" in ov["numeric_cols"]
    assert "region" in ov["categorical_cols"]


def test_numeric_summary_includes_shape_stats(clean_df):
    summary = dataexploration.numeric_summary(clean_df)
    assert "skew" in summary.columns
    assert "kurtosis" in summary.columns
    # Numeric columns appear as rows.
    assert "sales" in summary.index


def test_numeric_summary_empty_when_no_numeric():
    df = pd.DataFrame({"a": ["x", "y"]})
    assert dataexploration.numeric_summary(df).empty


def test_categorical_summary(clean_df):
    summary = dataexploration.categorical_summary(clean_df)
    assert "region" in summary.index
    assert summary.loc["region", "n_unique"] == 3


def test_correlation_matrix_is_square(clean_df):
    corr = dataexploration.correlation_matrix(clean_df)
    assert corr.shape[0] == corr.shape[1]
    # A column correlates perfectly with itself.
    assert corr.loc["sales", "sales"] == 1.0


def test_top_correlations_threshold(clean_df):
    pairs = dataexploration.top_correlations(clean_df, threshold=0.9)
    # sales/cost/profit are near-perfectly correlated in the fixture.
    assert (pairs["correlation"].abs() >= 0.9).all()
    assert {"feature_1", "feature_2", "correlation"}.issubset(pairs.columns)


def test_value_counts(clean_df):
    vc = dataexploration.value_counts(clean_df, "region")
    assert list(vc.columns) == ["region", "count"]
    assert vc["count"].sum() == len(clean_df)


def test_summary_report_bundle(clean_df):
    report = dataexploration.summary_report(clean_df)
    assert set(report.keys()) == {
        "overview",
        "numeric_summary",
        "categorical_summary",
        "correlations",
    }
