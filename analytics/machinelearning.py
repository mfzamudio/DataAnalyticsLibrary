"""
machinelearning
===============

STAGE 9 of the analytics lifecycle: **Machine Learning** — a reusable,
end-to-end modeling toolkit built on scikit-learn.

Goal of this module
-------------------
Provide the plumbing that ALL supervised modeling shares, so the
:mod:`predictiveanalysis` module (and your own notebooks) can stay short:

    1. Split data into train/test sets.
    2. Build a preprocessing pipeline (impute + scale numeric, one-hot encode
       categorical) so raw DataFrames can feed any estimator.
    3. Wrap preprocessing + estimator into a single ``Pipeline``.
    4. Cross-validate, fit, predict.
    5. Persist / load fitted models.

Why ``Pipeline``?
-----------------
Bundling preprocessing with the estimator prevents data leakage (the scaler /
encoder is fit only on training folds) and makes the model a single,
deployable object.

Requires scikit-learn (in requirements.txt). ``joblib`` (a scikit-learn
dependency) is used for model persistence.
"""

from __future__ import annotations

from typing import Optional, Sequence, cast

import pandas as pd

# scikit-learn is the backbone here; import the pieces we reuse everywhere.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target: str,
    *,
    drop: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate a DataFrame into a feature matrix ``X`` and target vector ``y``.

    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        Name of the target/label column.
    drop : sequence of str, optional
        Extra columns to exclude from ``X`` (IDs, leakage columns, etc.).

    Returns
    -------
    (X, y) : (pandas.DataFrame, pandas.Series)
    """
    to_drop = [target] + list(drop or [])
    X = df.drop(columns=to_drop)
    # Selecting a single column label yields a Series; cast makes that explicit
    # for the type checker (df[str] is otherwise inferred as Series | DataFrame).
    y = cast(pd.Series, df[target])
    return X, y


def train_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    """
    Split features/target into train and test partitions.

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series
    test_size : float, default 0.2
        Fraction held out for testing.
    random_state : int, default 42
        Seed for reproducibility.
    stratify : bool, default False
        If ``True`` preserve the class distribution of ``y`` in both splits
        (recommended for classification, especially when classes are imbalanced).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        # Passing the labels to ``stratify`` enables stratified sampling.
        stratify=y if stratify else None,
    )


def build_preprocessor(
    X: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    scale: bool = True,
) -> ColumnTransformer:
    """
    Build a ``ColumnTransformer`` that prepares raw columns for modeling.

    Numeric columns       -> impute missing values (+ optional standard scaling).
    Categorical columns   -> impute missing values + one-hot encode.

    Column types are detected automatically from ``X``'s dtypes.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix used to infer column types.
    numeric_strategy : {"median", "mean", "most_frequent"}, default "median"
        Imputation strategy for numeric columns.
    scale : bool, default True
        Standardize numeric features (zero mean, unit variance). Important for
        distance- and gradient-based models; harmless for tree models.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted; fit happens inside the pipeline on training data only.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    # Numeric branch: impute then (optionally) scale.
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    # Categorical branch: impute with the most frequent value, then one-hot.
    # ``handle_unknown="ignore"`` keeps prediction working if unseen
    # categories appear at inference time.
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        # Drop any columns not explicitly handled (e.g. datetimes).
        remainder="drop",
    )


def build_model_pipeline(estimator, preprocessor: ColumnTransformer) -> Pipeline:
    """
    Compose a preprocessing transformer and an estimator into one pipeline.

    Parameters
    ----------
    estimator : sklearn estimator
        Any classifier or regressor implementing ``fit``/``predict``.
    preprocessor : ColumnTransformer
        Usually from :func:`build_preprocessor`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        ``preprocess -> model``. Call ``.fit(X_train, y_train)`` on it directly.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv: int = 5,
    scoring: Optional[str] = None,
) -> dict:
    """
    K-fold cross-validation score for a pipeline.

    Cross-validation gives a more reliable performance estimate than a single
    train/test split by averaging over ``cv`` folds.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X : pandas.DataFrame
    y : pandas.Series
    cv : int, default 5
        Number of folds.
    scoring : str, optional
        scikit-learn scoring name (e.g. ``"r2"``, ``"accuracy"``, ``"f1"``).
        ``None`` uses the estimator's default scorer.

    Returns
    -------
    dict
        ``scores`` (per-fold list), ``mean``, ``std``.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
    return {
        "scores": scores.tolist(),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
    }


def fit(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Fit a pipeline on the training data and return it (fitted in place).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X_train : pandas.DataFrame
    y_train : pandas.Series

    Returns
    -------
    sklearn.pipeline.Pipeline
        The same, now-fitted pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, path: str) -> str:
    """
    Persist a fitted pipeline to disk with joblib.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline.
    path : str
        Destination path, conventionally ``.joblib``.

    Returns
    -------
    str
        The path written to.
    """
    import joblib  # bundled with scikit-learn

    joblib.dump(pipeline, path)
    return path


def load_model(path: str) -> Pipeline:
    """
    Load a pipeline previously saved with :func:`save_model`.

    Parameters
    ----------
    path : str
        Path to the ``.joblib`` file.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    import joblib

    return joblib.load(path)
