"""Tests for analytics.descriptiveanalysis."""

import pandas as pd

from analytics import descriptiveanalysis


def test_aggregate_single_metric(clean_df):
    out = descriptiveanalysis.aggregate(
        clean_df, group_by="region", metrics={"sales": "sum"}
    )
    assert "region" in out.columns
    # North = 100 + 150 = 250.
    north = out.loc[out["region"] == "North", "sales"].iloc[0]
    assert north == 250.0


def test_aggregate_flattens_multiindex(clean_df):
    out = descriptiveanalysis.aggregate(
        clean_df, group_by="region", metrics={"sales": ["sum", "mean"]}
    )
    # Multi-agg columns are flattened to 'sales_sum', 'sales_mean'.
    assert "sales_sum" in out.columns
    assert "sales_mean" in out.columns


def test_top_n(clean_df):
    out = descriptiveanalysis.top_n(clean_df, by="sales", n=2)
    assert len(out) == 2
    # Largest first.
    assert out["sales"].iloc[0] == 300.0


def test_share_of_total_sums_to_100(clean_df):
    out = descriptiveanalysis.share_of_total(
        clean_df, group_by="region", value="sales"
    )
    assert round(out["pct_of_total"].sum(), 2) == 100.0


def test_trend_over_time(timeseries_df):
    out = descriptiveanalysis.trend_over_time(
        timeseries_df, date_column="order_date", value="sales", freq="MS", agg="sum"
    )
    assert len(out) == 12
    assert "sales" in out.columns


def test_moving_average(timeseries_df):
    out = descriptiveanalysis.moving_average(timeseries_df, value="sales", window=3)
    assert "sales_ma3" in out.columns
    # First two values are NaN (window not full yet).
    assert out["sales_ma3"].isna().sum() == 2


def test_pivot_summary(clean_df):
    pivot = descriptiveanalysis.pivot_summary(
        clean_df, index="region", columns="channel", values="sales", aggfunc="sum"
    )
    assert "Online" in pivot.columns
    assert "Retail" in pivot.columns
