"""H1's label-efficiency half.

The experiment's whole value is the *shape* of the curve, so the properties
worth pinning are the ones that decide shape: that a subsample leaks nothing,
that a numerically failed fit is not mistaken for a poor one, and that
`labels_to_reach` refuses to read a bar off an unstable point.
"""

from __future__ import annotations

import numpy as np

from ggssvt.eval.label_efficiency import Curve, CurvePoint, compare, curve


def _point(fraction, n_labels, rmse, sd=0.01, floor=0.6):
    return CurvePoint("c", fraction, n_labels, rmse, sd, 0.5, floor)


def test_a_wildly_scattered_point_is_marked_unstable():
    """The seven geometric features at eight labels gave 128.8 +- 293.6."""
    assert _point(0.25, 8, 128.844, sd=293.634).unstable is True


def test_a_point_far_above_the_mean_predictor_is_marked_unstable():
    """A fit worse than five times the floor has failed, not merely done badly."""
    assert _point(0.25, 8, 4.0, sd=0.1, floor=0.6).unstable is True


def test_an_ordinary_point_is_not_marked():
    assert _point(0.5, 16, 0.45, sd=0.03).unstable is False


def test_labels_to_reach_ignores_unstable_points():
    """An unstable point can undercut the bar by accident; it must not count."""
    unstable = _point(0.25, 8, 0.1, sd=5.0)      # tiny mean, huge spread
    good = _point(0.75, 24, 0.40)
    assert Curve("c", [unstable, good]).labels_to_reach(0.5) == 24


def test_labels_to_reach_is_none_when_the_bar_is_never_met():
    assert Curve("c", [_point(1.0, 32, 0.9)]).labels_to_reach(0.5) is None


def test_the_comparison_reads_the_bar_off_the_reference_at_full_labels():
    reference = Curve("ref", [_point(0.5, 16, 0.9), _point(1.0, 32, 0.58)])
    reference.points[0].condition = reference.points[1].condition = "ref"
    other = Curve("other", [_point(0.25, 8, 0.46), _point(1.0, 32, 0.39)])
    for p in other.points:
        p.condition = "other"

    out = compare([reference, other], reference="ref")
    assert out["bar_rmse"] == 0.58
    assert out["labels_to_reach"] == {"ref": 32, "other": 8}


def test_a_subsample_never_sees_the_held_out_specimen():
    """Leakage here would manufacture the label-efficiency result.

    A feature that is a perfect copy of the target makes leakage visible: if the
    held-out row were ever in the fit, RMSE would be zero.
    """
    rng = np.random.default_rng(0)
    targets = rng.normal(1.0, 0.3, 24)
    features = np.stack([targets, rng.normal(0, 1, 24)], axis=1)

    built = curve(features, targets, condition="leaky", repeats=2, components=None)
    assert all(p.rmse > 1e-9 for p in built.points), "a fold saw its own target"


def test_more_labels_do_not_make_it_worse_on_a_clean_signal():
    rng = np.random.default_rng(1)
    targets = rng.normal(1.0, 0.3, 30)
    features = np.stack([targets + rng.normal(0, 0.05, 30),
                         rng.normal(0, 1, 30)], axis=1)
    built = curve(features, targets, condition="clean", repeats=3, components=None)
    quarter = built.at(0.25).rmse
    full = built.at(1.0).rmse
    assert full <= quarter
