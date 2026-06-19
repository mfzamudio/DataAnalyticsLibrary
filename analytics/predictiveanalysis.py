"""
predictiveanalysis
==================

STAGE 7 of the analytics lifecycle: **Predictive Analytics** -> "What will
happen?"

Goal of this stage
------------------
Use historical data to forecast future / unseen outcomes. This module offers
ready-to-use, functional helpers (built on :mod:`analytics.machinelearning`)
for the three most common supervised tasks plus unsupervised clustering:

    * Regression      : predict a numeric value (sales, price, demand).
    * Classification  : predict a category (churn yes/no, segment).
    * Clustering      : discover groups when there is no label (segmentation).

Methodology / steps
-------------------
    1. Frame the problem (regression vs classification vs clustering).
    2. Split data (train/test) — handled via ``machinelearning``.
    3. Build preprocessing + model pipeline.
    4. Train.
    5. Evaluate with task-appropriate metrics.
    6. Predict on new data.

Requires scikit-learn.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import machinelearning as ml


# ---------------------------------------------------------------------------
# REGRESSION
# ---------------------------------------------------------------------------

def train_regressor(
    df: pd.DataFrame,
    target: str,
    *,
    model: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    drop: Optional[list] = None,
) -> dict:
    """
    Train and evaluate a regression model end to end.

    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        Numeric column to predict.
    model : {"linear", "random_forest", "gradient_boosting"}, default "random_forest"
        Which estimator to use.
    test_size : float, default 0.2
    random_state : int, default 42
    drop : list, optional
        Extra columns to exclude from features.

    Returns
    -------
    dict
        ``pipeline`` (fitted), ``metrics`` (dict of R2/MAE/RMSE on the test
        set), ``X_test``, ``y_test``, ``y_pred``. The fitted pipeline can be
        saved with :func:`analytics.machinelearning.save_model`.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    # Map friendly names to estimator instances.
    estimators = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=random_state),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    if model not in estimators:
        raise ValueError(f"Unknown regression model: {model!r}")

    X, y = ml.split_features_target(df, target, drop=drop)
    X_train, X_test, y_train, y_test = ml.train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = ml.build_preprocessor(X)
    pipeline = ml.build_model_pipeline(estimators[model], preprocessor)
    pipeline = ml.fit(pipeline, X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # RMSE computed as sqrt(MSE) for compatibility across sklearn versions.
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    metrics = {
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "rmse": round(rmse, 4),
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def train_classifier(
    df: pd.DataFrame,
    target: str,
    *,
    model: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    drop: Optional[list] = None,
) -> dict:
    """
    Train and evaluate a classification model end to end.

    Uses stratified splitting to preserve class balance.

    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        Categorical column to predict.
    model : {"logistic", "random_forest", "gradient_boosting"}, default "random_forest"
    test_size : float, default 0.2
    random_state : int, default 42
    drop : list, optional
        Extra columns to exclude from features.

    Returns
    -------
    dict
        ``pipeline`` (fitted), ``metrics`` (accuracy + weighted
        precision/recall/F1), ``confusion_matrix``, ``X_test``, ``y_test``,
        ``y_pred``.
    """
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    estimators = {
        # ``max_iter`` raised so logistic regression reliably converges.
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }
    if model not in estimators:
        raise ValueError(f"Unknown classification model: {model!r}")

    X, y = ml.split_features_target(df, target, drop=drop)
    X_train, X_test, y_train, y_test = ml.train_test(
        X, y, test_size=test_size, random_state=random_state, stratify=True
    )

    preprocessor = ml.build_preprocessor(X)
    pipeline = ml.build_model_pipeline(estimators[model], preprocessor)
    pipeline = ml.fit(pipeline, X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # ``zero_division=0`` avoids warnings when a class has no predictions.
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(
            float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        ),
        "recall_weighted": round(
            float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        ),
        "f1_weighted": round(
            float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
        ),
    }
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# CLUSTERING (unsupervised)
# ---------------------------------------------------------------------------

def cluster_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int = 3,
    features: Optional[list] = None,
    random_state: int = 42,
) -> dict:
    """
    Segment rows into ``n_clusters`` groups with K-Means.

    Numeric features are standardized first (K-Means is distance-based, so
    scale matters). Use when there is no label and you want to discover natural
    groupings (e.g. customer segmentation).

    Parameters
    ----------
    df : pandas.DataFrame
    n_clusters : int, default 3
        Number of clusters (K).
    features : list, optional
        Numeric columns to cluster on. ``None`` uses all numeric columns.
    random_state : int, default 42

    Returns
    -------
    dict
        ``labels`` (cluster id per row), ``model`` (fitted KMeans),
        ``inertia`` (within-cluster sum of squares; lower = tighter clusters),
        ``silhouette`` (cohesion/separation score in [-1, 1]; higher = better).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    numeric = df.select_dtypes(include="number")
    if features is not None:
        numeric = numeric[features]
    # Drop rows with NaNs so the distance computation is well-defined.
    numeric = numeric.dropna()

    X_scaled = StandardScaler().fit_transform(numeric)
    # ``n_init="auto"`` is the modern scikit-learn default for stability.
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)

    # Silhouette is only defined for 2..n-1 clusters.
    sil = (
        round(float(silhouette_score(X_scaled, labels)), 4)
        if 1 < n_clusters < len(numeric)
        else None
    )
    return {
        "labels": labels,
        "model": kmeans,
        "inertia": round(float(kmeans.inertia_), 4),
        "silhouette": sil,
    }


def get_feature_importance(result: dict) -> Optional[pd.Series]:
    """
    Extract feature importances / coefficients from a trained result dict.

    Works with the dicts returned by :func:`train_regressor` /
    :func:`train_classifier`. Maps the importances back onto the
    post-preprocessing feature names so they are interpretable.

    Parameters
    ----------
    result : dict
        Output of ``train_regressor`` or ``train_classifier``.

    Returns
    -------
    pandas.Series or None
        Importance per feature, sorted descending. ``None`` if the underlying
        estimator exposes neither ``feature_importances_`` nor ``coef_``.
    """
    pipeline = result["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]

    # Recover the feature names produced by the ColumnTransformer (post one-hot).
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:  # pragma: no cover - very old sklearn
        names = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For multi-class, coef_ is 2D; collapse to magnitude per feature.
        coef = np.asarray(model.coef_)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    index = names if names is not None else range(len(values))
    return pd.Series(values, index=index).sort_values(ascending=False)
