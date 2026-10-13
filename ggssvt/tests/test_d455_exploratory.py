import numpy as np

from ggssvt.eval.d455_exploratory import (
    _candidate_models,
    _metrics,
    _nested_loocv,
    _row_features,
)


def test_row_features_are_finite_for_empty_depth():
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.zeros((720, 1280), dtype=np.float32)
    features = _row_features(rgb, depth)
    assert features.ndim == 1
    assert np.isfinite(features).all()


def test_nested_loocv_is_specimen_level_and_returns_baseline():
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    target = np.linspace(150.0, 240.0, 10)
    predictions, baseline, alphas = _nested_loocv(features, target)
    assert predictions.shape == target.shape
    assert baseline.shape == target.shape
    assert len(alphas) == target.size
    assert np.isfinite(predictions).all()
    assert np.isfinite(_metrics(predictions, target)["rmse_g"])
    assert baseline[0] == np.mean(target[1:])


def test_model_comparison_includes_distinct_regressor_families():
    names = {name for name, _ in _candidate_models(n_train=9)}
    assert names == {
        "ridge",
        "pls",
        "svr",
        "knn",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
    }
