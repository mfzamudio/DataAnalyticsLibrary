"""
Shared pytest configuration and fixtures.

This file is auto-discovered by pytest. It does two jobs:

1. Put the PROJECT ROOT on ``sys.path`` so ``import analytics`` works without
   installing the package (the repo is run directly from source).
2. Provide reusable in-memory DataFrame fixtures so individual tests stay
   small and focused. Fixtures use deterministic data (no randomness) so
   assertions are exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- 1. Make the package importable ----------------------------------------
# conftest.py lives in tests/, so the project root is its parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- 2. Fixtures ------------------------------------------------------------

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small, already-clean DataFrame for analysis tests."""
    return pd.DataFrame(
        {
            "region": ["North", "South", "North", "South", "East", "East"],
            "channel": ["Online", "Retail", "Online", "Retail", "Online", "Retail"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0, 50.0],
            "cost": [60.0, 120.0, 90.0, 150.0, 180.0, 30.0],
            "profit": [40.0, 80.0, 60.0, 100.0, 120.0, 20.0],
        }
    )


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """
    A DataFrame with deliberate quality problems for cleansing tests:
    messy column names, trailing whitespace, missing values, duplicates,
    and an extreme outlier.
    """
    df = pd.DataFrame(
        {
            "Order ID": [1, 2, 3, 4, 5],
            "Region ": [" North ", "South", "North", " East", "West "],
            "Sales": [100.0, 200.0, np.nan, 150.0, 999999.0],  # NaN + outlier
            "Units Sold": [1, 2, 3, 4, 5],
        }
    )
    # Append an exact duplicate of the first row.
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def timeseries_df() -> pd.DataFrame:
    """A monthly time series for trend/resampling tests."""
    dates = pd.date_range("2023-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "order_date": dates,
            "sales": np.arange(100, 100 + 12 * 10, 10, dtype=float),
        }
    )


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """
    A larger DataFrame with a (near) linear relationship so a model can learn
    something measurable. ``y`` depends strongly on ``x1`` plus a categorical
    effect.
    """
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(0, 1, n)
    group = rng.choice(["A", "B"], size=n)
    # Deterministic-ish target: strong x1 signal, small group offset, low noise.
    y = 3 * x1 + np.where(group == "A", 5, -5) + rng.normal(0, 2, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "group": group, "y": y})


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a tiny CSV to a temp dir and return its path (for loader tests)."""
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "d": ["2023-01-01", "2023-02-01", "2023-03-01"]}
    ).to_csv(path, index=False)
    return path
