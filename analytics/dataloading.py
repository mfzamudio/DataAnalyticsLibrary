"""
dataloading
===========

STAGE 1 of the analytics lifecycle: **Data Loading / Ingestion**.

Goal of this stage
------------------
Read raw data from heterogeneous sources (delimited text, CSV, Excel,
relational databases, JSON, Parquet) and return ONE standard, predictable
output: a pandas :class:`~pandas.DataFrame`.

Design contract
---------------
* Every public ``load_*`` function returns a ``pandas.DataFrame``.
* Functions never mutate global state; they only read.
* Errors are raised as clear exceptions (``FileNotFoundError``,
  ``ValueError``) instead of returning ``None`` silently, so failures are
  loud and easy to debug.

Why standardize on a DataFrame?
-------------------------------
The rest of the library (cleansing, exploration, modeling) is built on top of
pandas. By forcing every source to converge to a DataFrame here, downstream
code stays source-agnostic.
"""

from __future__ import annotations

# ``pathlib`` gives us safe, OS-independent path handling.
from pathlib import Path
# Typing imports are documentation that the IDE and reader can both use.
from typing import Optional, Sequence, Union

import pandas as pd

# A small type alias: anything that can point at a file on disk.
PathLike = Union[str, Path]


def _validate_file_exists(path: PathLike) -> Path:
    """
    Internal helper. Convert ``path`` to a :class:`Path` and assert the file
    exists on disk.

    Parameters
    ----------
    path : str or Path
        Location of the file to read.

    Returns
    -------
    Path
        The validated, resolved path object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a regular file.
    """
    # Resolve to an absolute path so error messages are unambiguous.
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")
    return p


def load_csv(
    path: PathLike,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    header: Optional[int] = 0,
    usecols: Optional[Sequence[str]] = None,
    parse_dates: Optional[Sequence[str]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.
    sep : str, default ","
        Field delimiter. Use ``";"`` for European-style CSVs, ``"\\t"`` for TSV.
    encoding : str, default "utf-8"
        File encoding. Try ``"latin-1"`` if you hit ``UnicodeDecodeError``.
    header : int or None, default 0
        Row number to use as column names. ``None`` means "no header row".
    usecols : sequence of str, optional
        Subset of columns to read. Reading fewer columns is faster and uses
        less memory on wide files.
    parse_dates : sequence of str, optional
        Column names to parse as datetimes at load time.
    nrows : int, optional
        Read only the first ``nrows`` rows. Handy for sampling huge files.

    Returns
    -------
    pandas.DataFrame
    """
    p = _validate_file_exists(path)
    # pandas does the heavy lifting; we simply expose the most useful knobs.
    return pd.read_csv(
        p,
        sep=sep,
        encoding=encoding,
        header=header,
        usecols=usecols,
        parse_dates=parse_dates,
        nrows=nrows,
    )


def load_text(
    path: PathLike,
    *,
    sep: Optional[str] = None,
    encoding: str = "utf-8",
    # pandas accepts an int row index, the literal "infer", or None.
    header: Union[int, str, None] = "infer",
    column_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a delimited TEXT file (e.g. ``.txt``, ``.tsv``, fixed-delimiter logs).

    This is a thin convenience wrapper over :func:`pandas.read_csv` with
    ``sep=None``, which triggers pandas' automatic delimiter sniffing via the
    Python engine. Use :func:`load_csv` when you already know the delimiter.

    Parameters
    ----------
    path : str or Path
        Path to the text file.
    sep : str, optional
        Delimiter. If ``None`` (default) pandas tries to auto-detect it.
    encoding : str, default "utf-8"
        File encoding.
    header : int, "infer", or None, default "infer"
        Row to use as the header. ``None`` if the file has no header.
    column_names : sequence of str, optional
        Explicit column names. Pass this together with ``header=None`` for
        headerless files.

    Returns
    -------
    pandas.DataFrame
    """
    p = _validate_file_exists(path)
    return pd.read_csv(
        p,
        sep=sep,
        encoding=encoding,
        header=header,
        names=column_names,
        # The "python" engine is required for delimiter auto-detection.
        engine="python",
    )


def load_excel(
    path: PathLike,
    *,
    sheet_name: Union[str, int] = 0,
    header: Optional[int] = 0,
    usecols: Optional[Union[str, Sequence[str]]] = None,
    parse_dates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a single sheet from an Excel workbook (``.xlsx`` / ``.xls``).

    Requires the optional dependency ``openpyxl`` (already in requirements.txt)
    for ``.xlsx`` files.

    Parameters
    ----------
    path : str or Path
        Path to the workbook.
    sheet_name : str or int, default 0
        Sheet to read, by name or zero-based index. To read every sheet at
        once, use :func:`load_excel_all_sheets` instead.
    header : int or None, default 0
        Row to use as column names.
    usecols : str or sequence of str, optional
        Columns to read. Accepts Excel ranges like ``"A:D"`` or a list of names.
    parse_dates : sequence of str, optional
        Columns to parse as datetimes.

    Returns
    -------
    pandas.DataFrame
    """
    p = _validate_file_exists(path)
    return pd.read_excel(
        p,
        sheet_name=sheet_name,
        header=header,
        usecols=usecols,
        parse_dates=parse_dates,
    )


def load_excel_all_sheets(path: PathLike) -> dict[str, pd.DataFrame]:
    """
    Load EVERY sheet of an Excel workbook.

    Unlike the other loaders this returns a ``dict`` mapping sheet name ->
    DataFrame, because a workbook is inherently a collection of tables. Callers
    can then pick or concatenate sheets as needed.

    Parameters
    ----------
    path : str or Path
        Path to the workbook.

    Returns
    -------
    dict of {str: pandas.DataFrame}
    """
    p = _validate_file_exists(path)
    # ``sheet_name=None`` is the pandas idiom for "give me all sheets".
    return pd.read_excel(p, sheet_name=None)


def load_json(
    path: PathLike,
    *,
    orient: Optional[str] = None,
    lines: bool = False,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a JSON or JSON-Lines file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the ``.json`` / ``.jsonl`` file.
    orient : str, optional
        Expected JSON layout (e.g. ``"records"``, ``"columns"``, ``"split"``).
        Leave as ``None`` to let pandas infer it.
    lines : bool, default False
        Set ``True`` for newline-delimited JSON (one JSON object per line).
    encoding : str, default "utf-8"
        File encoding.

    Returns
    -------
    pandas.DataFrame
    """
    p = _validate_file_exists(path)
    return pd.read_json(p, orient=orient, lines=lines, encoding=encoding)


def load_parquet(
    path: PathLike,
    *,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.

    Parquet is a columnar format that is far faster and smaller than CSV for
    large datasets. Requires ``pyarrow`` (in requirements.txt).

    Parameters
    ----------
    path : str or Path
        Path to the ``.parquet`` file.
    columns : sequence of str, optional
        Read only these columns (column pruning, very efficient in Parquet).

    Returns
    -------
    pandas.DataFrame
    """
    p = _validate_file_exists(path)
    return pd.read_parquet(p, columns=columns)


def load_database(
    query: str,
    connection_string: str,
    *,
    params: Optional[dict] = None,
    parse_dates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Run a SQL query against a relational database and return the result set.

    Uses SQLAlchemy under the hood, so it works with any backend that has a
    SQLAlchemy driver (PostgreSQL, MySQL, SQLite, SQL Server, etc.).

    Parameters
    ----------
    query : str
        A SQL ``SELECT`` statement. Use bound parameters (``:name``) plus the
        ``params`` argument instead of f-strings to avoid SQL injection.
    connection_string : str
        SQLAlchemy URL. Examples:
            * SQLite   : ``"sqlite:///local.db"``
            * Postgres : ``"postgresql+psycopg2://user:pass@host:5432/dbname"``
            * MySQL    : ``"mysql+pymysql://user:pass@host:3306/dbname"``
        NOTE: install the matching DB driver separately
        (``psycopg2-binary``, ``pymysql``, ...). SQLite needs no extra driver.
    params : dict, optional
        Values for the bound parameters in ``query``.
    parse_dates : sequence of str, optional
        Columns to parse as datetimes.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ImportError
        If SQLAlchemy is not installed.
    """
    try:
        # Imported lazily so the library still works for file-only users who
        # have not installed SQLAlchemy.
        from sqlalchemy import create_engine, text
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "load_database requires SQLAlchemy. Install it with "
            "`pip install sqlalchemy` (plus your DB driver)."
        ) from exc

    # The engine manages the underlying connection pool.
    engine = create_engine(connection_string)
    # ``with`` guarantees the connection is returned/closed even on error.
    with engine.connect() as conn:
        return pd.read_sql(
            sql=text(query),
            con=conn,
            params=params,
            parse_dates=parse_dates,
        )


def load_auto(path: PathLike, **kwargs) -> pd.DataFrame:
    """
    Convenience dispatcher: pick the right loader based on the file extension.

    Supported extensions: ``.csv``, ``.txt``, ``.tsv``, ``.xlsx``, ``.xls``,
    ``.json``, ``.jsonl``, ``.parquet``.

    Parameters
    ----------
    path : str or Path
        File to load.
    **kwargs
        Forwarded to the underlying ``load_*`` function.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the file extension is not recognized.
    """
    p = _validate_file_exists(path)
    # Normalize the suffix to lower case for case-insensitive matching.
    suffix = p.suffix.lower()

    # Map each known extension to its loader. Keeping this as a dict makes the
    # supported set obvious and easy to extend.
    if suffix == ".csv":
        return load_csv(p, **kwargs)
    if suffix in (".txt", ".tsv"):
        return load_text(p, **kwargs)
    if suffix in (".xlsx", ".xls"):
        return load_excel(p, **kwargs)
    if suffix in (".json", ".jsonl"):
        # JSON Lines files conventionally use the .jsonl extension.
        return load_json(p, lines=(suffix == ".jsonl"), **kwargs)
    if suffix == ".parquet":
        return load_parquet(p, **kwargs)

    raise ValueError(
        f"Unsupported file extension '{suffix}'. "
        "Use a specific load_* function for non-standard formats."
    )
