"""H4's noise and occlusion arms.

The property worth pinning is not that degradation degrades. It is that a
destroyed reconstruction cannot be reported as a success: a mid-height occlusion
severs the plant, the largest-connected-component step keeps whichever side is
bigger, and the surviving piece can land inside the plausible band by
coincidence. Measured on E001 at 50% occlusion: 5% of the volume survives, and
the fragment's implied density is 484 kg/m3, comfortably "plausible".
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.robustness import Degraded, add_depth_noise, occlude, summarise


def test_zero_noise_returns_the_depth_untouched():
    """Level 0 is the control; it must be the identity, not almost the identity."""
    depth = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
    assert np.array_equal(add_depth_noise(depth, 0.0, np.random.default_rng(0)), depth)


def test_missing_returns_stay_missing():
    """Zero depth means no return, not zero metres; perturbing it invents data."""
    depth = np.zeros((4, 4), dtype=np.float32)
    noisy = add_depth_noise(depth, 4.0, np.random.default_rng(0))
    assert np.array_equal(noisy, depth)


def test_noise_grows_with_range_as_the_sensor_does():
    """Kinect v2 error grows as z^2, which is why the carve margin has that term."""
    near = np.full((200, 200), 1.0, dtype=np.float32)
    far = np.full((200, 200), 3.0, dtype=np.float32)
    def spread(depth):
        noisy = add_depth_noise(depth, 2.0, np.random.default_rng(1))
        return float(np.std(noisy - depth))

    assert spread(far) > 5 * spread(near)


def test_occlusion_removes_a_band_and_zero_removes_nothing():
    mask = np.zeros((100, 10), dtype=bool)
    mask[20:80] = True

    assert np.array_equal(occlude(mask, 0.0), mask)
    quarter = occlude(mask, 0.25)
    assert quarter.sum() < mask.sum()
    # The band sits inside the subject, so rows survive both above and below it.
    kept = np.flatnonzero(quarter.any(axis=1))
    assert kept.min() == 20 and kept.max() == 79


def test_occluding_an_empty_mask_is_a_no_op():
    empty = np.zeros((10, 10), dtype=bool)
    assert not occlude(empty, 0.5).any()


def test_a_surviving_sliver_is_flagged_as_a_fragment():
    """This is the check that stops a destroyed reconstruction scoring as a pass."""
    sliver = Degraded("E001", "occlusion", 0.5, 1.14, 483.7, True,
                      voxels=658, surviving_fraction=0.05)
    assert sliver.fragment is True

    intact = Degraded("E001", "noise", 1.0, 4.11, 133.9, False,
                      voxels=12400, surviving_fraction=0.99)
    assert intact.fragment is False


def test_summarise_does_not_count_a_fragment_as_plausible():
    rows = [
        {"kind": "occlusion", "level": 0.5, "plausible": True, "fragment": True,
         "density": 483.7, "volume_l": 1.14},
        {"kind": "occlusion", "level": 0.5, "plausible": True, "fragment": False,
         "density": 400.0, "volume_l": 3.0},
    ]
    out = summarise(rows)["occlusion@0.5"]
    assert out["plausible"] == 1, "the fragment must not be counted as a success"
    assert out["fragments"] == 1


def test_an_unknown_degradation_is_refused():
    from ggssvt.eval.robustness import degrade

    with pytest.raises(ValueError, match="unknown degradation"):
        degrade(object(), "blur", 1.0)
