"""The backbone comparison harness, and the leakage it must not have."""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.dino_probe import _fit_pca, loocv_probe
from ggssvt.eval.experiment import ComparisonReport, ConditionResult
from ggssvt.eval.metrics import paired_bootstrap_difference, regression_metrics


def test_pca_is_fitted_on_the_training_split_only():
    """The held-out row must not influence the projection used to predict it."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(12, 5))

    with_all = _fit_pca(features, 3)
    without_first = _fit_pca(features[1:], 3)

    assert not np.allclose(with_all[0], without_first[0])


def test_probe_cannot_predict_pure_noise():
    """The single most important guard: no signal in, no signal out.

    If the fold-internal PCA and standardisation ever leak the held-out
    specimen, random features will score well above zero here. At 28 samples
    and 768 features that leak would be enormous, so this test is the thing
    standing between an honest result and a fabricated one.
    """
    rng = np.random.default_rng(0)
    n_samples = 28
    features = rng.normal(size=(n_samples, 768))
    targets = rng.uniform(0.4, 2.4, size=n_samples)

    predictions = loocv_probe(features, targets, n_components=8, alpha=1.0)
    r2 = regression_metrics(predictions, targets).r2

    assert r2 < 0.25, f"probe found structure in noise (R2={r2:.3f}); check for leakage"


def test_probe_recovers_a_signal_on_a_high_variance_direction():
    rng = np.random.default_rng(1)
    n_samples = 28
    features = rng.normal(size=(n_samples, 40))
    features[:, 0] *= 8.0        # make the informative axis a leading component
    targets = 1.2 + 0.1 * features[:, 0] + 0.05 * rng.normal(size=n_samples)

    predictions = loocv_probe(features, targets, n_components=8, alpha=0.1)
    assert regression_metrics(predictions, targets).r2 > 0.6


def test_pca_reduction_can_discard_a_low_variance_predictive_direction():
    """A real limitation of the probe, pinned down rather than left implicit.

    PCA is unsupervised, so it keeps the directions with the most variance, not
    the ones that predict the target. When the informative axis carries no more
    variance than the noise axes, truncating to eight components throws it away
    and the probe reports nothing.

    This matters for interpreting a *negative* probe result: it bounds what the
    representation contributes through the leading components, not what it
    contains. On the real DINOv2 descriptors the finding survives PCA-4, PCA-8
    and un-truncated ridge alike, so it is not an artefact of this choice -- but
    that had to be checked, not assumed.
    """
    rng = np.random.default_rng(1)
    n_samples = 28
    features = rng.normal(size=(n_samples, 40))
    targets = 1.2 + 0.8 * features[:, 0] + 0.05 * rng.normal(size=n_samples)

    truncated = loocv_probe(features, targets, n_components=8, alpha=0.1)
    assert regression_metrics(truncated, targets).r2 < 0.4

    # Keeping every available component recovers it.
    full = loocv_probe(features, targets, n_components=40, alpha=0.1)
    assert regression_metrics(full, targets).r2 > regression_metrics(
        truncated, targets
    ).r2


def test_paired_bootstrap_finds_no_difference_between_identical_methods():
    rng = np.random.default_rng(0)
    target = rng.uniform(0.4, 2.4, 28)
    predicted = target + rng.normal(0, 0.2, 28)

    d = paired_bootstrap_difference(predicted, predicted, target, n_resamples=400)
    assert d["difference"] == pytest.approx(0.0)
    assert d["low"] <= 0 <= d["high"]


def test_paired_bootstrap_detects_a_large_real_difference():
    rng = np.random.default_rng(0)
    target = rng.uniform(0.4, 2.4, 28)
    good = target + rng.normal(0, 0.05, 28)
    bad = target + rng.normal(0, 0.60, 28)

    d = paired_bootstrap_difference(good, bad, target, n_resamples=800)
    assert d["difference"] < 0          # good has the lower RMSE
    assert d["high"] < 0                # and the interval excludes zero


def test_comparison_report_renders_skipped_conditions_without_crashing():
    targets = np.array([1.0, 2.0, 3.0])
    report = ComparisonReport(
        conditions={
            "cnn (no DINO)": ConditionResult(
                label="cnn (no DINO)",
                metrics=regression_metrics(np.array([1.1, 2.1, 2.9]), targets),
                predictions=np.array([1.1, 2.1, 2.9]),
                extras={"n_features": 7},
            ),
            "dinov3": ConditionResult(
                label="dinov3", metrics=None, skipped_reason="gated on HuggingFace"
            ),
        },
        targets=targets,
        n_specimens=3,
    )

    table = report.to_table()
    assert "skipped" in table
    assert "gated" in table
    assert "dinov3" not in report.paired_against_control()

    import json

    payload = json.loads(report.to_json())
    assert payload["conditions"]["dinov3"]["skipped"] == "gated on HuggingFace"


def test_comparison_report_pairs_every_ran_condition_against_the_control():
    rng = np.random.default_rng(0)
    targets = rng.uniform(0.4, 2.4, 20)
    control = targets + rng.normal(0, 0.3, 20)
    treatment = targets + rng.normal(0, 0.15, 20)

    report = ComparisonReport(
        conditions={
            "cnn (no DINO)": ConditionResult(
                "cnn (no DINO)", regression_metrics(control, targets), control
            ),
            "dinov2-base": ConditionResult(
                "dinov2-base", regression_metrics(treatment, targets), treatment
            ),
        },
        targets=targets,
        n_specimens=20,
    )

    paired = report.paired_against_control()
    assert set(paired) == {"dinov2-base"}
    assert "p_direction" in paired["dinov2-base"]
