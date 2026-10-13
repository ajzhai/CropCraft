"""H2's held-out-view consistency.

The point of this experiment is that it measures something `reproject` does not:
agreement with a view the reconstruction never saw. So the properties worth
pinning are that the held-out view really is held out, and that the gap is
computed in the direction that makes a large gap mean poor generalisation.
"""

from __future__ import annotations

import numpy as np

from ggssvt.eval.viewpoint import ViewScore, _subset, summarise


class _Fake:
    """Enough of a CachedSpecimen for the subsetting to be exercised."""

    def __init__(self, n_views=12):
        self.plant_id = "E001"
        self.position_ids = [f"p{i}" for i in range(n_views)]
        self.rotation = np.arange(n_views * 9).reshape(n_views, 3, 3).astype(float)
        self.centre = np.arange(n_views * 3).reshape(n_views, 3).astype(float)
        self.depth_m = np.arange(n_views * 4).reshape(n_views, 2, 2).astype(np.float32)
        self.mask = np.ones((n_views, 2, 2), dtype=bool)
        self.occupancy = np.ones((4, 4, 4), dtype=bool)
        self.voxel_size_m = 0.012
        self.crop_top = 4
        self.target_kg = 1.0
        self.n_views = n_views


def test_the_held_out_view_is_absent_from_the_subset():
    """If the withheld view leaked in, the experiment would measure nothing."""
    cached = _Fake()
    keep = [v for v in range(12) if v != 5]
    reduced = _subset(cached, keep)

    assert reduced.n_views == 11
    assert "p5" not in reduced.position_ids
    # and the arrays line up with the ids rather than being off by one
    assert np.array_equal(reduced.centre[5], cached.centre[6])


def test_subsetting_does_not_mutate_the_cache():
    cached = _Fake()
    before = cached.mask.copy()
    reduced = _subset(cached, [0, 1, 2])
    reduced.mask[:] = False
    assert np.array_equal(cached.mask, before)


def test_the_gap_is_in_sample_minus_held_out():
    """Signed so that a large positive gap means poor generalisation."""
    score = ViewScore("E001", 0, in_sample_iou=0.50, held_out_iou=0.20,
                      in_sample_depth_mae_m=0.07, held_out_depth_mae_m=0.09)
    assert score.iou_gap == 0.30

    generalises = ViewScore("E002", 0, 0.50, 0.49, 0.07, 0.07)
    assert generalises.iou_gap < score.iou_gap


def test_summarise_reports_the_relative_drop():
    rows = [
        {"plant_id": "A", "in_sample_iou": 0.5, "held_out_iou": 0.4,
         "iou_gap": 0.1, "in_sample_depth_mae_m": 0.07,
         "held_out_depth_mae_m": 0.08},
        {"plant_id": "B", "in_sample_iou": 0.5, "held_out_iou": 0.4,
         "iou_gap": 0.1, "in_sample_depth_mae_m": 0.07,
         "held_out_depth_mae_m": 0.08},
    ]
    out = summarise(rows)
    assert out["n_specimens"] == 2
    assert out["iou_gap"] == 0.1
    assert out["relative_drop"] == 0.2       # 0.1 of 0.5


def test_summarise_of_nothing_is_empty_rather_than_a_crash():
    assert summarise([]) == {}
