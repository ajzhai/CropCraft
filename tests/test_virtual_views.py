"""The virtual-view renderer, on shapes whose reconstruction is known in advance.

Pheno4D is 12 GB and not in the repository, so none of these need it. They test
the parts that would silently produce a wrong answer: the camera convention, the
z-buffer, and the claim that a visual hull scores well by reprojection while
being wrong about the shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.config import KINECT_V2, VOXEL_RESOLUTION, VOXEL_SIZE_M
from ggssvt.data.pheno4d import voxelise
from ggssvt.eval.virtual_views import (
    camera_poses,
    render,
    silhouette_iou,
    voxel_iou,
)


def sphere(radius_m=0.15, centre_z=0.35, n=40000, seed=0):
    """Points on a sphere: convex, so a visual hull should recover it exactly."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = direction * radius_m
    points[:, 2] += centre_z
    return points


def dumbbell(radius_m=0.09, gap_m=0.22, centre_z=0.35, n=40000, seed=0):
    """Two spheres with empty space between them.

    Strongly non-convex: the visual hull has to fill the gap, because from every
    azimuth some ray through the middle is blocked by one lobe or the other.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    points = direction * radius_m
    points[: n // 2, 0] -= gap_m / 2.0
    points[n // 2 :, 0] += gap_m / 2.0
    points[:, 2] += centre_z
    return points


def test_cameras_ring_the_subject_at_the_requested_radius_and_height():
    rotations, centres, azimuths = camera_poses(n_views=12, distance_m=1.4,
                                                height_m=1.0)
    assert rotations.shape == (12, 3, 3)
    assert np.allclose(np.linalg.norm(centres[:, :2], axis=1), 1.4)
    assert np.allclose(centres[:, 2], 1.0)
    assert azimuths[1] - azimuths[0] == pytest.approx(30.0)


def test_the_rotations_are_proper_and_match_the_pipelines_convention():
    rotations, centres, _ = camera_poses(n_views=6)
    for rotation, centre in zip(rotations, centres):
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        # x_world = R @ x_cam + c, and the camera looks along its own +z, so a
        # point one metre ahead must move toward the subject axis.
        ahead = rotation @ np.array([0.0, 0.0, 1.0]) + centre
        assert np.linalg.norm(ahead[:2]) < np.linalg.norm(centre[:2])


def test_the_depth_buffer_keeps_the_nearer_of_two_surfaces():
    # Two planes of points, one directly behind the other from every camera.
    near = np.array([[0.0, 0.0, 0.35]])
    far = np.array([[0.0, 0.0, 0.35], [0.0, 0.0, 0.30]])

    one = render(near, target_z_m=0.35, splat_radius=0)
    two = render(np.vstack([near, far]), target_z_m=0.35, splat_radius=0)

    hit = one.mask[0]
    assert hit.any()
    # Adding a point behind must never push the recorded range further away.
    assert np.all(two.depth_m[0][hit] <= one.depth_m[0][hit] + 1e-6)


def test_a_sphere_renders_a_disc_of_about_the_right_angular_size():
    points = sphere(radius_m=0.15, centre_z=0.35)
    rendered = render(points, target_z_m=0.35)

    assert rendered.mask.shape == (12, KINECT_V2.height, KINECT_V2.width)
    for view in range(12):
        assert rendered.mask[view].sum() > 500
        # 0.30 m across at about 1.4 m, through fx: the silhouette's width.
        cols = np.nonzero(rendered.mask[view].any(axis=0))[0]
        width_px = cols.max() - cols.min()
        expected = 0.30 * KINECT_V2.fx / np.linalg.norm(
            rendered.centre[view] - np.array([0.0, 0.0, 0.35]))
        assert width_px == pytest.approx(expected, rel=0.2)


def test_silhouette_iou_rewards_a_hull_that_is_wrong_about_the_shape():
    """The claim behind FINDINGS 7f, on a shape where the truth is not in doubt.

    A dumbbell's visual hull fills the gap between the lobes: from every azimuth
    a lobe blocks the view through the middle, so no silhouette ever votes the
    centre empty. The filled hull is therefore *wrong* -- and it reprojects into
    every input silhouette perfectly, scoring better than the true shape does.
    """
    points = dumbbell()
    rendered = render(points, target_z_m=0.35)

    truth = voxelise(points, resolution=VOXEL_RESOLUTION,
                     voxel_size_m=VOXEL_SIZE_M)

    # The hull, approximated by filling the truth's bounding region along x --
    # exactly the concavity a carve cannot see.
    filled = truth.copy()
    for iy in range(filled.shape[1]):
        for iz in range(filled.shape[2]):
            occupied = np.nonzero(truth[:, iy, iz])[0]
            if occupied.size >= 2:
                filled[occupied.min():occupied.max() + 1, iy, iz] = True

    assert filled.sum() > truth.sum()          # the gap really did get filled
    assert voxel_iou(filled, truth) < 1.0      # so it is wrong about the shape

    # ... and yet it scores at least as well by reprojection.
    assert silhouette_iou(filled, rendered) >= silhouette_iou(truth, rendered)


def test_voxel_iou_is_zero_for_disjoint_and_one_for_identical():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    a[0, 0, 0] = True
    b[3, 3, 3] = True
    assert voxel_iou(a, b) == 0.0
    assert voxel_iou(a, a) == 1.0
    assert voxel_iou(np.zeros((2, 2, 2), bool), np.zeros((2, 2, 2), bool)) == 0.0


def test_voxelise_puts_the_soil_plane_at_the_bottom_of_the_grid():
    points = np.array([[0.0, 0.0, 0.001], [0.0, 0.0, 0.5]])
    grid = voxelise(points, resolution=64, voxel_size_m=0.012)
    occupied_z = np.nonzero(grid.any(axis=(0, 1)))[0]
    assert occupied_z.min() == 0
    assert occupied_z.max() == int(0.5 / 0.012)
