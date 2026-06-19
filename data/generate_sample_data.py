"""
generate_sample_data
=====================

Create a SYNTHETIC sales dataset so the demo notebook is fully reproducible
without any external download. The data is fabricated with NumPy and is not
real — it only exists to exercise the library's loading/cleansing/analysis
functions.

Run from the project root:
    python data/generate_sample_data.py

Output:
    data/sample_sales.csv

The generated dataset intentionally includes realistic data-quality issues
(missing values, duplicate rows, messy column names, whitespace, a few
outliers) so the cleansing module has something meaningful to fix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed seed => identical dataset on every run (reproducibility).
RNG = np.random.default_rng(42)

# Number of transaction rows to generate.
N_ROWS = 2_000


def generate() -> pd.DataFrame:
    """Build and return the synthetic sales DataFrame."""
    # --- Dimensions (categorical) -----------------------------------------
    regions = RNG.choice(
        ["North", "South", "East", "West"], size=N_ROWS, p=[0.3, 0.25, 0.25, 0.2]
    )
    categories = RNG.choice(
        ["Electronics", "Furniture", "Office", "Software"], size=N_ROWS
    )
    channels = RNG.choice(["Online", "Retail"], size=N_ROWS, p=[0.6, 0.4])

    # --- Dates spread across two years (for trend & period analysis) ------
    start = pd.Timestamp("2023-01-01")
    offsets = RNG.integers(0, 730, size=N_ROWS)  # up to ~2 years of days
    order_dates = start + pd.to_timedelta(offsets, unit="D")

    # --- Measures (numeric) -----------------------------------------------
    # Units sold: small positive counts.
    units = RNG.integers(1, 20, size=N_ROWS)
    # Unit price varies by category to create real correlations.
    base_price = {
        "Electronics": 450,
        "Furniture": 300,
        "Office": 40,
        "Software": 120,
    }
    unit_price = np.array([base_price[c] for c in categories]) * RNG.normal(
        1.0, 0.15, size=N_ROWS
    )
    unit_price = unit_price.round(2)

    # Sales = units * price, with a little noise.
    sales = (units * unit_price * RNG.normal(1.0, 0.05, size=N_ROWS)).round(2)
    # Cost is ~60-80% of sales; profit derives from it (useful for diagnostics).
    cost = (sales * RNG.uniform(0.6, 0.8, size=N_ROWS)).round(2)
    profit = (sales - cost).round(2)

    # Customer satisfaction score 1-5 (drives a classification demo later).
    satisfaction = RNG.integers(1, 6, size=N_ROWS)

    df = pd.DataFrame(
        {
            # Deliberately messy column names for the cleansing demo.
            "Order ID": np.arange(1, N_ROWS + 1),
            "Order Date": order_dates,
            "Region ": regions,            # trailing space on purpose
            "Product Category": categories,
            "Channel": channels,
            "Units Sold": units,
            "Unit Price ($)": unit_price,
            "Sales": sales,
            "Cost": cost,
            "Profit": profit,
            "Satisfaction": satisfaction,
        }
    )

    # --- Inject realistic data-quality problems ---------------------------
    # 1. Whitespace in some text values.
    messy_idx = RNG.choice(N_ROWS, size=50, replace=False)
    df.loc[messy_idx, "Region "] = " " + df.loc[messy_idx, "Region "].astype(str) + " "

    # 2. Missing values in a few columns.
    for col, frac in [("Cost", 0.04), ("Satisfaction", 0.06), ("Unit Price ($)", 0.02)]:
        nan_idx = RNG.choice(N_ROWS, size=int(N_ROWS * frac), replace=False)
        df.loc[nan_idx, col] = np.nan

    # 3. A handful of duplicate rows.
    dupes = df.sample(n=20, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    # 4. A few extreme outliers in Sales.
    out_idx = RNG.choice(len(df), size=5, replace=False)
    df.loc[out_idx, "Sales"] = df.loc[out_idx, "Sales"] * 50

    return df


def main() -> None:
    """Generate the dataset and write it next to this script."""
    out_path = Path(__file__).parent / "sample_sales.csv"
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
