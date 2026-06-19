"""Tests for analytics.diagnosticanalysis."""

import numpy as np
import pandas as pd
import pytest

from analytics import diagnosticanalysis


def test_correlation_with_target(clean_df):
    out = diagnosticanalysis.correlation_with_target(clean_df, "profit")
    assert list(out.columns) == ["feature", "correlation"]
    # Target itself is excluded.
    assert "profit" not in out["feature"].tolist()
    # Sorted by absolute strength, descending.
    abs_vals = out["correlation"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_correlation_with_target_non_numeric_raises(clean_df):
    with pytest.raises(ValueError):
        diagnosticanalysis.correlation_with_target(clean_df, "region")


def test_group_comparison(clean_df):
    out = diagnosticanalysis.group_comparison(
        clean_df, group_column="channel", value_column="sales"
    )
    assert "mean" in out.columns
    assert set(out.index) == {"Online", "Retail"}


def test_ttest_two_groups():
    pytest.importorskip("scipy")
    # Two clearly different groups => significant difference.
    df = pd.DataFrame(
        {
            "grp": ["A"] * 20 + ["B"] * 20,
            "val": list(np.arange(20)) + list(np.arange(100, 120)),
        }
    )
    res = diagnosticanalysis.ttest_two_groups(
        df, group_column="grp", value_column="val", group_a="A", group_b="B"
    )
    assert res["n_a"] == 20 and res["n_b"] == 20
    assert res["significant_at_0.05"] is True


def test_chi_square_independence():
    pytest.importorskip("scipy")
    # Perfectly associated categories => not independent.
    df = pd.DataFrame(
        {
            "a": ["x", "x", "y", "y"] * 10,
            "b": ["p", "p", "q", "q"] * 10,
        }
    )
    res = diagnosticanalysis.chi_square_independence(df, column_a="a", column_b="b")
    assert res["significant_at_0.05"] is True


def test_contribution_to_change():
    df = pd.DataFrame(
        {
            "region": ["N", "S", "N", "S"],
            "year": [2022, 2022, 2023, 2023],
            "sales": [100, 100, 150, 80],
        }
    )
    out = diagnosticanalysis.contribution_to_change(
        df, dimension="region", value="sales",
        period_column="year", period_from=2022, period_to=2023,
    )
    # N grew +50, S fell -20; total delta = +30.
    n_row = out.loc[out["region"] == "N"].iloc[0]
    assert n_row["delta"] == 50
    assert round(out["pct_contribution_to_change"].sum(), 2) == 100.0
