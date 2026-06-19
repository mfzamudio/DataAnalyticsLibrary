"""Tests for analytics.dataloading."""

from pathlib import Path

import pandas as pd
import pytest

from analytics import dataloading


def test_load_csv_returns_dataframe(sample_csv):
    df = dataloading.load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "d"]
    assert len(df) == 3


def test_load_csv_parse_dates(sample_csv):
    df = dataloading.load_csv(sample_csv, parse_dates=["d"])
    # The parsed column should be a datetime dtype.
    assert pd.api.types.is_datetime64_any_dtype(df["d"])


def test_load_csv_usecols(sample_csv):
    df = dataloading.load_csv(sample_csv, usecols=["a"])
    assert list(df.columns) == ["a"]


def test_load_csv_nrows(sample_csv):
    df = dataloading.load_csv(sample_csv, nrows=2)
    assert len(df) == 2


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        dataloading.load_csv("/no/such/file.csv")


def test_load_auto_dispatches_csv(sample_csv):
    df = dataloading.load_auto(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_auto_unsupported_extension(tmp_path):
    bad = tmp_path / "data.unknown"
    bad.write_text("whatever")
    with pytest.raises(ValueError):
        dataloading.load_auto(bad)


def test_load_database_sqlite(tmp_path):
    """End-to-end SQL path using a throwaway SQLite file (no driver needed)."""
    pytest.importorskip("sqlalchemy")
    db_path = tmp_path / "test.db"
    conn_str = f"sqlite:///{db_path}"

    # Seed a table via SQLAlchemy/pandas.
    from sqlalchemy import create_engine

    engine = create_engine(conn_str)
    pd.DataFrame({"id": [1, 2], "name": ["a", "b"]}).to_sql(
        "people", engine, index=False
    )

    df = dataloading.load_database("SELECT * FROM people ORDER BY id", conn_str)
    assert list(df.columns) == ["id", "name"]
    assert df.loc[0, "name"] == "a"
