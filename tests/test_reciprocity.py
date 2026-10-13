"""Closing the reconstruction-to-segmentation loop.

The property that matters here is not that the rules work. It is that the
comparison is honest: a re-carve of the *unchanged* masks has to reproduce the
cached volume, or every number measures the difference between two carvers
rather than the effect under test. That is not hypothetical. Skipping the
largest-connected-component step made E001 look like all three rules rescued it,
when the control alone moved the volume further than any rule did.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.reciprocity import RULES, combine


def test_the_four_rules_do_what_their_names_say():
    original = np.array([[True, True, False, False]])
    projected = np.array([[True, False, True, False]])

    assert combine(original, projected, "original").tolist() == original.tolist()
    assert combine(original, projected, "union").tolist() == [[True, True, True, False]]
    assert combine(original, projected, "intersection").tolist() == [
        [True, False, False, False]
    ]
    assert combine(original, projected, "reconstruction_only").tolist() == (
        projected.tolist()
    )


def test_an_unknown_rule_is_refused_rather_than_guessed():
    a = np.array([[True]])
    with pytest.raises(ValueError, match="unknown rule"):
        combine(a, a, "average")


def test_union_never_shrinks_and_intersection_never_grows():
    """The direction of each rule is the whole argument, so pin it.

    Union grows the masks and therefore the hull, which is already too large;
    intersection shrinks it. Measured, union took the plausible count from 8/36
    down to 4/36 and intersection took it up to 19/36, exactly as the direction
    predicts.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        original = rng.random((8, 8)) > 0.5
        projected = rng.random((8, 8)) > 0.5
        assert combine(original, projected, "union").sum() >= original.sum()
        assert combine(original, projected, "intersection").sum() <= original.sum()


def test_every_named_rule_is_implemented():
    """RULES drives the report's columns; an unimplemented name would raise."""
    a = np.array([[True, False]])
    for rule in RULES:
        assert combine(a, a, rule).shape == a.shape


def test_uncrop_puts_the_image_back_where_the_carve_expects_it():
    """carve() indexes at the sensor's native height; the cache is cropped.

    Without this the row index runs off the end for any voxel projecting into
    the strip that was cropped away, which is an IndexError rather than a wrong
    answer, but only for specimens tall enough to reach it.
    """
    from ggssvt.config import KINECT_V2
    from ggssvt.eval.reciprocity import _uncrop

    cropped = np.ones((416, 512), dtype=bool)
    full = _uncrop(cropped, 4, bool)

    assert full.shape == (KINECT_V2.height, KINECT_V2.width)
    assert not full[:4].any(), "the cropped strip must stay empty"
    assert full[4:420].all(), "the image must land at its original offset"


def test_the_summary_reports_a_count_per_rule():
    from ggssvt.eval.reciprocity import summarise

    rows = [
        {"rule": "original", "plausible_after": True, "density_after": 300.0,
         "volume_after_l": 2.0},
        {"rule": "original", "plausible_after": False, "density_after": 50.0,
         "volume_after_l": 9.0},
        {"rule": "union", "plausible_after": False, "density_after": 40.0,
         "volume_after_l": 12.0},
    ]
    out = summarise(rows)
    assert out["original"] == {"n": 2, "plausible": 1, "median_density": 175.0,
                               "mean_volume_l": 5.5}
    assert out["union"]["plausible"] == 0


def test_an_emptied_reconstruction_serialises_as_valid_json():
    """A rule can empty a volume, making the density infinite.

    json.dump writes that as bare `Infinity`, which Python reads back happily
    and no other parser accepts. It is not valid JSON, and it broke the document
    generator that consumes these reports.
    """
    import json

    from ggssvt.eval.reciprocity import RefinementResult

    emptied = RefinementResult(
        plant_id="X", rule="intersection",
        mask_px_before=100, mask_px_after=0,
        volume_before_l=4.0, volume_after_l=0.0,
        density_before=130.0, density_after=float("inf"),
        plausible_before=False, plausible_after=False,
    )
    text = json.dumps(emptied.as_dict(), allow_nan=False)   # raises on Infinity
    assert json.loads(text)["density_after"] is None


def test_summarise_ignores_the_nulls_rather_than_crashing():
    from ggssvt.eval.reciprocity import summarise

    rows = [
        {"rule": "union", "plausible_after": False, "density_after": None,
         "volume_after_l": None},
        {"rule": "union", "plausible_after": True, "density_after": 300.0,
         "volume_after_l": 2.0},
    ]
    out = summarise(rows)
    assert out["union"]["median_density"] == 300.0
    assert out["union"]["mean_volume_l"] == 2.0
