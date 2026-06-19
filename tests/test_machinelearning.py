"""Tests for analytics.machinelearning."""

from typing import Any, cast

import pandas as pd
import pytest

from analytics import machinelearning as ml

# scikit-learn underpins this whole module.
pytest.importorskip("sklearn")


def test_split_features_target(regression_df):
    X, y = ml.split_features_target(regression_df, "y")
    assert "y" not in X.columns
    assert y.name == "y"
    assert len(X) == len(y)


def test_split_features_target_with_drop(regression_df):
    X, y = ml.split_features_target(regression_df, "y", drop=["x2"])
    assert "x2" not in X.columns


def test_train_test_proportions(regression_df):
    X, y = ml.split_features_target(regression_df, "y")
    X_train, X_test, y_train, y_test = ml.train_test(X, y, test_size=0.25)
    assert len(X_test) == 50  # 25% of 200
    assert len(X_train) == 150


def test_build_preprocessor_handles_mixed_types(regression_df):
    X, _ = ml.split_features_target(regression_df, "y")
    pre = ml.build_preprocessor(X)
    # Transformer is unfitted but should fit+transform without error.
    # cast to Any: sklearn types fit_transform as Optional, but it returns a
    # (possibly sparse) matrix here; we only need its row count.
    transformed = cast(Any, pre.fit_transform(X))
    assert transformed.shape[0] == len(X)


def test_full_pipeline_fit_predict_and_persist(regression_df, tmp_path):
    from sklearn.linear_model import LinearRegression

    X, y = ml.split_features_target(regression_df, "y")
    X_train, X_test, y_train, y_test = ml.train_test(X, y)
    pre = ml.build_preprocessor(X)
    pipe = ml.build_model_pipeline(LinearRegression(), pre)
    pipe = ml.fit(pipe, X_train, y_train)

    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test)

    # Round-trip save/load yields identical predictions.
    path = tmp_path / "model.joblib"
    ml.save_model(pipe, str(path))
    loaded = ml.load_model(str(path))
    assert list(loaded.predict(X_test)) == list(preds)


def test_cross_validate(regression_df):
    from sklearn.linear_model import LinearRegression

    X, y = ml.split_features_target(regression_df, "y")
    pre = ml.build_preprocessor(X)
    pipe = ml.build_model_pipeline(LinearRegression(), pre)
    res = ml.cross_validate(pipe, X, y, cv=3, scoring="r2")
    assert len(res["scores"]) == 3
    # With a strong linear signal, mean R2 should be high.
    assert res["mean"] > 0.8
