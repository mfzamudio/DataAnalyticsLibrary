"""Tests for analytics.predictiveanalysis."""

import numpy as np
import pandas as pd
import pytest

from analytics import predictiveanalysis

pytest.importorskip("sklearn")


def test_train_regressor_metrics(regression_df):
    res = predictiveanalysis.train_regressor(
        regression_df, target="y", model="linear"
    )
    assert {"r2", "mae", "rmse"}.issubset(res["metrics"])
    # Strong linear signal => high R2 on the held-out set.
    assert res["metrics"]["r2"] > 0.8
    assert len(res["y_pred"]) == len(res["y_test"])


def test_train_regressor_unknown_model(regression_df):
    with pytest.raises(ValueError):
        predictiveanalysis.train_regressor(regression_df, target="y", model="bogus")


def test_get_feature_importance_linear(regression_df):
    res = predictiveanalysis.train_regressor(
        regression_df, target="y", model="linear"
    )
    imp = predictiveanalysis.get_feature_importance(res)
    assert imp is not None
    # Sorted descending by importance.
    assert list(imp.values) == sorted(imp.values, reverse=True)


def test_train_classifier_metrics():
    # A separable 2-class problem.
    rng = np.random.default_rng(1)
    n = 200
    x = np.concatenate([rng.normal(0, 1, n), rng.normal(6, 1, n)])
    label = np.array([0] * n + [1] * n)
    df = pd.DataFrame({"x": x, "label": label})

    res = predictiveanalysis.train_classifier(df, target="label", model="logistic")
    assert "accuracy" in res["metrics"]
    # Well-separated classes => high accuracy.
    assert res["metrics"]["accuracy"] > 0.9
    assert res["confusion_matrix"].shape == (2, 2)


def test_cluster_kmeans():
    # Two well-separated blobs.
    rng = np.random.default_rng(2)
    a = rng.normal(0, 0.5, (50, 2))
    b = rng.normal(10, 0.5, (50, 2))
    df = pd.DataFrame(np.vstack([a, b]), columns=["f1", "f2"])

    res = predictiveanalysis.cluster_kmeans(df, n_clusters=2)
    assert len(res["labels"]) == len(df)
    # Clear separation => strong silhouette score.
    assert res["silhouette"] > 0.7
