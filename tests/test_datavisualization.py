"""
Tests for analytics.datavisualization.

These tests use the non-interactive 'Agg' matplotlib backend so they run
headless (no display). We assert that each function returns a valid Axes/Figure
and does not raise; we do NOT assert on pixel content.
"""

import matplotlib
import pytest

# Force a headless backend BEFORE importing pyplot anywhere.
matplotlib.use("Agg")

# Skip the whole module cleanly if plotting libs are unavailable.
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

from matplotlib.axes import Axes  # noqa: E402

from analytics import datavisualization as viz  # noqa: E402


def test_plot_histogram_returns_axes(clean_df):
    ax = viz.plot_histogram(clean_df, "sales")
    assert isinstance(ax, Axes)


def test_plot_bar(clean_df):
    ax = viz.plot_bar(clean_df, x="region", y="sales")
    assert isinstance(ax, Axes)


def test_plot_scatter(clean_df):
    ax = viz.plot_scatter(clean_df, x="sales", y="profit", hue="region")
    assert isinstance(ax, Axes)


def test_plot_box(clean_df):
    ax = viz.plot_box(clean_df, y="sales", x="region")
    assert isinstance(ax, Axes)


def test_plot_count(clean_df):
    ax = viz.plot_count(clean_df, "region")
    assert isinstance(ax, Axes)


def test_plot_correlation_heatmap(clean_df):
    ax = viz.plot_correlation_heatmap(clean_df)
    assert isinstance(ax, Axes)


def test_plot_distribution(clean_df):
    ax = viz.plot_distribution(clean_df, "profit")
    assert isinstance(ax, Axes)


def test_plot_missing_matrix(clean_df):
    ax = viz.plot_missing_matrix(clean_df)
    assert isinstance(ax, Axes)


def test_plot_feature_importance():
    ax = viz.plot_feature_importance([0.5, 0.3, 0.2], ["a", "b", "c"], top_n=3)
    assert isinstance(ax, Axes)


def test_plot_time_series_decomposition(timeseries_df):
    pytest.importorskip("statsmodels")
    fig = viz.plot_time_series_decomposition(
        timeseries_df, value_column="sales", date_column="order_date", period=4
    )
    # Returns a Figure, not an Axes.
    assert fig is not None


def test_save_figure(clean_df, tmp_path):
    ax = viz.plot_histogram(clean_df, "sales")
    out = viz.save_figure(ax.figure, str(tmp_path / "fig.png"))
    assert out.endswith(".png")
