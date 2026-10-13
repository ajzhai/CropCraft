"""The encoded clouds have to carry a scale the browser can trust.

A voxel's height was recovered in the page as ``resolution * 0.024 / 255``,
which is within half a percent when the caller downsamples by two and exactly
twice the truth when it does not. The stage panels do not downsample, so every
height they reported was doubled, and the pot-and-plant split then sat above the
whole plant: pot, stem and leaves were painted one colour and the split looked
broken rather than misscaled.

Nothing caught it because the error is silent. It changes no shape, throws no
exception, and only shows up when a height is compared against a physical
threshold. These tests pin the scale to the geometry it came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.config import VOXEL_SIZE_M
from ggssvt.eval.dashboard_data import _quantise


def _decode_heights(payload: dict) -> np.ndarray:
    """The z byte of each point in metres, the way the browser reads it."""
    import base64
    import zlib

    raw = np.frombuffer(
        zlib.decompress(base64.b64decode(payload["data"])), dtype=np.uint8)
    return raw.reshape(-1, 3)[:, 2] * payload["metres_per_byte"]


@pytest.mark.parametrize("downsample", [1, 2, 4])
def test_encoded_height_matches_the_voxel_it_came_from(downsample):
    """A voxel at index k decodes to k * pitch metres, whatever the downsample."""
    resolution = 128
    grid = np.zeros((resolution,) * 3, dtype=bool)
    indices = [0, 8, 40, 96, resolution - 1]
    for k in indices:
        grid[64, 64, k] = True

    heights = np.sort(_decode_heights(_quantise(grid, downsample=downsample)))
    expected = np.sort(np.unique(
        (np.array(indices) // downsample) * VOXEL_SIZE_M * downsample))

    assert heights.shape == expected.shape
    # Within one voxel of the downsampled grid, which is the quantisation and
    # not an error in the scale.
    assert np.abs(heights - expected).max() <= VOXEL_SIZE_M * downsample


def test_the_old_inferred_scale_would_fail_this():
    """The regression itself, so the guess cannot quietly come back.

    At downsample 1 the browser's old formula doubles every height. Asserting
    the size of that error keeps the test honest about what it is protecting.
    """
    resolution = 128
    grid = np.zeros((resolution,) * 3, dtype=bool)
    grid[64, 64, 96] = True

    payload = _quantise(grid, downsample=1)
    guessed = payload["resolution"] * 0.024 / 255

    assert payload["metres_per_byte"] == pytest.approx(VOXEL_SIZE_M / 2)
    assert guessed / payload["metres_per_byte"] == pytest.approx(2.0, rel=0.01)


def test_top_of_the_grid_decodes_below_the_working_volume():
    """The highest voxel cannot decode above the volume the grid represents."""
    resolution = 128
    grid = np.zeros((resolution,) * 3, dtype=bool)
    grid[64, 64, resolution - 1] = True

    payload = _quantise(grid, downsample=1)
    assert _decode_heights(payload).max() <= resolution * VOXEL_SIZE_M
