"""TSDF fusion, and the property that makes it worth having.

The point of these is not that fusion produces a nice number. It is that fusion
can represent things a visual hull provably cannot, and that it says "unknown"
where a carve would say "solid".
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.config import Intrinsics, voxel_grid_centres
from ggssvt.geometry.fusion import fuse

INTRINSICS = Intrinsics(fx=200.0, fy=200.0, cx=32.0, cy=32.0, width=64, height=64)


def _looking_at_origin(distance: float, azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """One camera on a circle in the z = 0.4 m plane, pointing at the axis.

    Returns camera-to-world rotation and centre in the convention the rest of
    the pipeline uses: world = cam @ R.T + centre.
    """
    a = np.deg2rad(azimuth_deg)
    centre = np.array([distance * np.cos(a), distance * np.sin(a), 0.4])
    forward = -np.array([np.cos(a), np.sin(a), 0.0])          # +z of the camera
    down = np.array([0.0, 0.0, -1.0])                         # +y of the camera
    right = np.cross(down, forward)
    rotation = np.stack([right, down, forward], axis=1)
    return rotation, centre


def _rig(n_views: int, distance: float = 1.2):
    rots, cents = [], []
    for i in range(n_views):
        r, c = _looking_at_origin(distance, i * 360.0 / n_views)
        rots.append(r)
        cents.append(c)
    return np.stack(rots), np.stack(cents)


def _constant_depth(n_views: int, value: float) -> np.ndarray:
    return np.full((n_views, INTRINSICS.height, INTRINSICS.width), value, np.float32)


def test_unobserved_space_stays_unknown_rather_than_solid():
    """The property a carve cannot have: absence of evidence recorded as absence."""
    rotation, centre = _rig(4)
    depth = _constant_depth(4, 1.0)

    result = fuse(depth, rotation, centre, intrinsics=INTRINSICS,
                  resolution=32, voxel_size_m=0.048)

    assert result.coverage() < 1.0
    # Whatever is unobserved must not count as interior.
    assert not (result.interior & ~result.observed).any()


def test_interior_lies_behind_the_measured_surface():
    """The invariant the whole thing rests on.

    Not a size comparison: the interior is a band one truncation width deep
    behind each observed surface, so its voxel count follows the geometry of
    that band rather than the distance to it. What must hold is the sign. Every
    interior voxel is farther from the camera than the depth it measured, and
    none is farther than the truncation allows.
    """
    rotation, centre = _rig(1)
    measured = 1.0
    voxel = 0.048
    truncation = 3.0 * voxel

    result = fuse(_constant_depth(1, measured), rotation, centre,
                  intrinsics=INTRINSICS, resolution=32, voxel_size_m=voxel)

    grid = voxel_grid_centres(32, voxel)
    cam = (grid.reshape(-1, 3) - centre[0]) @ rotation[0]
    depth_of_voxel = cam[:, 2].reshape(32, 32, 32)

    behind = depth_of_voxel[result.interior]
    assert behind.size > 0
    assert (behind > measured).all()
    assert (behind < measured + truncation).all()


def test_a_mask_keeps_the_background_out():
    rotation, centre = _rig(4)
    depth = _constant_depth(4, 1.0)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[:, 20:44, 20:44] = True

    masked = fuse(depth, rotation, centre, mask=mask, intrinsics=INTRINSICS,
                  resolution=32, voxel_size_m=0.048)
    unmasked = fuse(depth, rotation, centre, intrinsics=INTRINSICS,
                    resolution=32, voxel_size_m=0.048)

    assert masked.observed.sum() < unmasked.observed.sum()


def test_depth_outside_the_sensor_range_is_ignored():
    """Zero means 'no return' on a Kinect, not 'a surface at zero metres'."""
    rotation, centre = _rig(4)

    result = fuse(_constant_depth(4, 0.0), rotation, centre, intrinsics=INTRINSICS,
                  resolution=32, voxel_size_m=0.048)

    assert result.coverage() == 0.0
    assert not result.interior.any()


def test_the_field_starts_empty_and_stays_in_range():
    rotation, centre = _rig(4)

    result = fuse(_constant_depth(4, 1.0), rotation, centre, intrinsics=INTRINSICS,
                  resolution=32, voxel_size_m=0.048)

    assert result.tsdf.min() >= -1.0
    assert result.tsdf.max() <= 1.0
    assert result.weight.min() >= 0.0


def test_more_views_observe_more_of_the_volume():
    depth4 = _constant_depth(4, 1.0)
    depth12 = _constant_depth(12, 1.0)

    four = fuse(depth4, *_rig(4), intrinsics=INTRINSICS,
                resolution=32, voxel_size_m=0.048)
    twelve = fuse(depth12, *_rig(12), intrinsics=INTRINSICS,
                  resolution=32, voxel_size_m=0.048)

    assert twelve.coverage() > four.coverage()


def test_shape_and_metadata_round_trip():
    rotation, centre = _rig(3)

    result = fuse(_constant_depth(3, 1.0), rotation, centre, intrinsics=INTRINSICS,
                  resolution=16, voxel_size_m=0.096)

    assert result.tsdf.shape == (16, 16, 16)
    assert result.weight.shape == (16, 16, 16)
    assert result.n_views == 3
    assert result.voxel_size_m == pytest.approx(0.096)
