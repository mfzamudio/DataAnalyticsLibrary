"""Tests for analytics.prescriptiveanalysis."""

import pandas as pd
import pytest

from analytics import prescriptiveanalysis


def test_recommend_by_rules_first_match_wins():
    df = pd.DataFrame({"risk": [0.9, 0.6, 0.1]})
    rules = [
        (lambda d: d["risk"] > 0.8, "call"),
        (lambda d: d["risk"] > 0.5, "discount"),
    ]
    out = prescriptiveanalysis.recommend_by_rules(df, rules, default_action="none")
    assert out["recommended_action"].tolist() == ["call", "discount", "none"]


def test_recommend_by_rules_does_not_mutate():
    df = pd.DataFrame({"risk": [0.9]})
    prescriptiveanalysis.recommend_by_rules(df, [(lambda d: d["risk"] > 0.5, "x")])
    assert "recommended_action" not in df.columns


def test_scenario_analysis():
    # A trivial fake model: prediction == value of 'x'.
    class FakeModel:
        def predict(self, frame):
            return frame["x"].to_numpy()

    base = pd.DataFrame({"x": [1], "other": [10]})
    out = prescriptiveanalysis.scenario_analysis(
        FakeModel(), base, variable="x", values=[1, 2, 3]
    )
    assert out["prediction"].tolist() == [1, 2, 3]


def test_scenario_analysis_requires_single_row():
    class FakeModel:
        def predict(self, frame):
            return frame["x"].to_numpy()

    base = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError):
        prescriptiveanalysis.scenario_analysis(
            FakeModel(), base, variable="x", values=[1]
        )


def test_optimize_allocation_picks_best_return():
    pytest.importorskip("scipy")
    out = prescriptiveanalysis.optimize_allocation(
        returns=[0.1, 0.5, 0.2],
        total_budget=1000,
        labels=["A", "B", "C"],
    )
    assert out.attrs["success"] is True
    # All budget should flow to the highest-return option (B).
    b_alloc = out.loc[out["option"] == "B", "allocation"].iloc[0]
    assert b_alloc == pytest.approx(1000.0)


def test_optimize_allocation_respects_caps():
    pytest.importorskip("scipy")
    out = prescriptiveanalysis.optimize_allocation(
        returns=[0.1, 0.5, 0.2],
        total_budget=1000,
        max_alloc=[1000, 400, 1000],  # cap the best option at 400
        labels=["A", "B", "C"],
    )
    b_alloc = out.loc[out["option"] == "B", "allocation"].iloc[0]
    assert b_alloc <= 400 + 1e-6
