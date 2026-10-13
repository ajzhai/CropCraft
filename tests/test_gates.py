"""Every gate must fire on the failure it exists for.

A check that cannot fail is decoration. These construct the failure each gate
was written to catch and assert it is caught, then construct a healthy case and
assert it is not, because a gate that fires on everything is equally useless.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.gates import (
    check_reconstruction,
    check_regression,
    check_segmentation,
    check_training,
    summarise,
)


class FakeCached:
    """The minimum surface the reconstruction and segmentation gates touch."""

    def __init__(self, mask=None, occupancy=None, rim=0.28, confident=True):
        self.plant_id = "T001"
        self.voxel_size_m = 0.012
        self.pot_height_m = rim
        self.mask = mask if mask is not None else np.ones((12, 40, 40), bool)
        if occupancy is None:
            occupancy = np.zeros((128, 128, 128), bool)
            occupancy[60:68, 60:68, 5:60] = True
        self.occupancy = occupancy
        self.pot = type("Pot", (), {"confident": confident})()


def _named(report, name):
    return next(c for c in report.checks if c.name == name)


# --- segmentation --------------------------------------------------------

def test_an_empty_view_blocks():
    mask = np.zeros((12, 40, 40), bool)
    mask[:, 10:30, 10:30] = True
    mask[7] = False                       # one view lost the subject

    report = check_segmentation(FakeCached(mask=mask))

    assert report.blocked
    assert _named(report, "mask not empty in any view").failed_blocking


def test_a_mask_that_swallowed_the_background_blocks():
    mask = np.ones((12, 40, 40), bool)    # the whole frame

    report = check_segmentation(FakeCached(mask=mask))

    assert report.blocked
    assert _named(report, "mask has not taken the background").failed_blocking


def test_a_collapsed_view_is_advisory_not_blocking():
    """It still carves; it just carves wrongly, and that needs a human."""
    mask = np.zeros((12, 40, 40), bool)
    mask[:, 10:30, 10:30] = True
    mask[3] = False
    mask[3, 19:21, 19:21] = True          # tiny but not empty

    report = check_segmentation(FakeCached(mask=mask))

    collapsed = _named(report, "no view collapsed relative to the rest")
    assert not collapsed.passed
    assert not collapsed.blocking


def test_a_reasonable_mask_passes_everything():
    mask = np.zeros((12, 40, 40), bool)
    mask[:, 12:28, 12:28] = True          # 16 per cent of the frame

    report = check_segmentation(FakeCached(mask=mask))

    assert not report.blocked
    assert all(c.passed for c in report.checks)


# --- reconstruction ------------------------------------------------------

def test_an_empty_occupancy_blocks_and_stops_there():
    report = check_reconstruction(FakeCached(occupancy=np.zeros((128,) * 3, bool)))

    assert report.blocked
    assert len(report.checks) == 1, "no point checking a volume that does not exist"


def test_nothing_above_the_rim_blocks():
    occupancy = np.zeros((128, 128, 128), bool)
    occupancy[60:68, 60:68, 2:10] = True          # all of it below 0.28 m

    report = check_reconstruction(FakeCached(occupancy=occupancy))

    assert report.blocked
    assert _named(report, "something sits above the pot rim").failed_blocking


def test_an_absurd_density_blocks_but_a_merely_poor_one_does_not():
    cached = FakeCached()

    absurd = check_reconstruction(cached, mass_kg=1e6)
    assert absurd.blocked

    # An envelope: far below plant tissue, but a legitimate input to a
    # comparison of methods, so it must not be dropped.
    poor = check_reconstruction(cached, mass_kg=0.02)
    assert not poor.blocked
    assert not _named(poor, "implied density is physically plausible").passed


def test_a_fallback_rim_is_reported_without_blocking():
    report = check_reconstruction(FakeCached(confident=False))

    rim = _named(report, "pot rim was measured, not assumed")
    assert not rim.passed
    assert not rim.blocking
    assert not report.blocked


# --- training ------------------------------------------------------------

def test_a_model_that_collapsed_onto_the_mean_blocks():
    """The failure this whole module exists for: respectable RMSE, nothing learned."""
    targets = [0.5, 1.0, 1.5, 2.0]
    metrics = {
        "rmse_kg": 0.55, "occupancy_ap": 0.6,
        "predictions": [1.25, 1.25, 1.25, 1.25], "targets": targets,
    }

    report = check_training(metrics, mean_rmse=0.57)

    assert report.blocked
    assert _named(report, "predictions vary").failed_blocking


def test_a_run_worse_than_the_mean_predictor_blocks():
    metrics = {"rmse_kg": 0.80, "predictions": [0.2, 1.0, 1.8], "targets": [0.5, 1.0, 1.5]}

    report = check_training(metrics, mean_rmse=0.57)

    assert _named(report, "beats the mean predictor").failed_blocking


def test_a_flat_loss_curve_blocks():
    metrics = {"rmse_kg": 0.4, "predictions": [0.2, 1.0, 1.8], "targets": [0.5, 1.0, 1.5]}

    report = check_training(metrics, mean_rmse=0.57, losses=[2.0, 1.99, 1.98])

    assert _named(report, "loss actually fell").failed_blocking


def test_non_finite_metrics_block():
    report = check_training({"rmse_kg": float("nan")})

    assert _named(report, "metrics are finite").failed_blocking


def test_occupancy_at_chance_blocks():
    metrics = {"occupancy_ap": 0.03, "predictions": [0.2, 1.0, 1.8], "targets": [0.5, 1.0, 1.5]}

    report = check_training(metrics)

    assert _named(report, "occupancy is better than chance").failed_blocking


def test_a_healthy_run_passes():
    metrics = {
        "rmse_kg": 0.335, "occupancy_ap": 0.62,
        "predictions": [0.4, 0.9, 1.6, 2.1], "targets": [0.5, 1.0, 1.5, 2.0],
    }

    report = check_training(metrics, mean_rmse=0.568, losses=[3.1, 1.4, 0.9])

    assert not report.blocked
    assert all(c.passed for c in report.checks)


# --- regression ----------------------------------------------------------

def test_a_moved_number_is_reported():
    report = check_regression({"rmse": 0.60}, {"rmse": 0.544})

    moved = _named(report, "rmse unchanged")
    assert not moved.passed
    assert "0.5440" in moved.message and "0.6000" in moved.message


def test_movement_inside_tolerance_passes():
    report = check_regression({"rmse": 0.5445}, {"rmse": 0.544})

    assert _named(report, "rmse unchanged").passed


def test_a_vanished_number_is_reported():
    report = check_regression({}, {"rmse": 0.544})

    assert not _named(report, "rmse still reported").passed


def test_regression_is_never_blocking():
    """Drift is for a human to judge; it must not stop a pipeline on its own."""
    report = check_regression({"a": 9.0}, {"a": 1.0})

    assert not report.blocked


# --- summary -------------------------------------------------------------

def test_summary_counts_blocked_subjects_once_each():
    bad = check_segmentation(FakeCached(mask=np.ones((12, 40, 40), bool)))
    good = check_segmentation(FakeCached(
        mask=np.pad(np.ones((12, 16, 16), bool), ((0, 0), (12, 12), (12, 12)))))

    summary = summarise([bad, good])

    assert summary["subjects"] == 2
    assert summary["blocked"] == 1
    assert summary["checks_run"] == len(bad.checks) + len(good.checks)


def test_reports_are_json_serialisable():
    import json

    report = check_reconstruction(FakeCached(), mass_kg=0.74)

    assert json.loads(json.dumps(report.as_dict())) == report.as_dict()


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_a_nonsense_mass_does_not_crash_the_gate(mass):
    report = check_reconstruction(FakeCached(), mass_kg=mass)

    assert isinstance(report.blocked, bool)
