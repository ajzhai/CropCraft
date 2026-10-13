"""Geometry: plane fitting, pose algebra, carving, and the metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ggssvt.config import KINECT_V2
from ggssvt.data.io import backproject, project
from ggssvt.eval.metrics import reconstruction_metrics, regression_metrics, voxel_iou
from ggssvt.geometry.carving import dilate, largest_connected_component
from ggssvt.geometry.plane import PlaneFitError, fit_plane_ransac
from ggssvt.geometry.rig import CAMERA_UP_PRIOR, levelling_rotation, nominal_view_pose


def test_plane_fit_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    # A floor 1.2 m below a camera looking level: y = 1.2 in camera coordinates.
    points = np.stack(
        [
            rng.uniform(-2, 2, 4000),
            np.full(4000, 1.2) + rng.normal(0, 0.003, 4000),
            rng.uniform(0.5, 4.0, 4000),
        ],
        axis=1,
    )
    plane = fit_plane_ransac(points, normal_prior=CAMERA_UP_PRIOR, rng=rng)

    assert plane.camera_height_m == pytest.approx(1.2, abs=0.02)
    assert abs(plane.normal @ CAMERA_UP_PRIOR) == pytest.approx(1.0, abs=1e-3)
    assert plane.inlier_fraction > 0.9


def test_plane_fit_rejects_a_point_set_with_no_plane():
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, size=(2000, 3))
    with pytest.raises(PlaneFitError):
        fit_plane_ransac(points, min_inlier_fraction=0.9, iterations=50, rng=rng)


def test_backprojection_and_projection_are_inverse():
    depth = np.zeros((KINECT_V2.height, KINECT_V2.width), dtype=np.float32)
    depth[100:140, 200:240] = 1.35

    points = backproject(depth, KINECT_V2)
    valid = depth > 0
    uv, z = project(points[valid], KINECT_V2)

    rows, cols = np.nonzero(valid)
    assert np.allclose(uv[:, 0], cols, atol=1e-3)
    assert np.allclose(uv[:, 1], rows, atol=1e-3)
    assert np.allclose(z, 1.35, atol=1e-5)


def test_levelling_rotation_is_orthonormal_and_puts_up_on_z():
    from ggssvt.geometry.plane import Plane

    normal = np.array([0.1, -0.98, 0.17])
    normal = normal / np.linalg.norm(normal)
    plane = Plane(normal=normal, offset=0.9, inlier_fraction=1.0)

    rotation = levelling_rotation(plane)

    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(rotation @ normal, [0, 0, 1], atol=1e-9)


@pytest.mark.parametrize("azimuth", [0, 30, 90, 180, 270, 330])
def test_nominal_pose_places_the_camera_on_its_azimuth_looking_inward(azimuth):
    pose = nominal_view_pose(azimuth)

    radius = np.linalg.norm(pose.centre[:2])
    bearing = math.degrees(math.atan2(pose.centre[1], pose.centre[0])) % 360
    assert (bearing - azimuth + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-6)
    assert radius > 0
    assert pose.centre[2] > 0        # above the floor

    # The optical axis points inward. Compare in the horizontal plane only: the
    # nominal camera is level, so it looks at the plant's mid-height rather than
    # down at the world origin on the floor.
    forward = pose.rotation @ np.array([0.0, 0.0, 1.0])
    forward_xy = forward[:2] / np.linalg.norm(forward[:2])
    inward_xy = -pose.centre[:2] / np.linalg.norm(pose.centre[:2])
    assert forward_xy @ inward_xy == pytest.approx(1.0, abs=1e-6)


def test_pose_round_trips_world_and_camera_coordinates():
    pose = nominal_view_pose(60)
    points = np.array([[0.0, 0.0, 0.4], [0.2, -0.1, 1.0], [-0.3, 0.25, 0.05]])
    assert np.allclose(pose.to_world(pose.to_camera(points)), points, atol=1e-9)


def test_dilate_grows_by_one_pixel_per_iteration():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    assert dilate(mask, 0).sum() == 1
    assert dilate(mask, 1).sum() == 9      # 3x3
    assert dilate(mask, 2).sum() == 25     # 5x5


def test_largest_component_keeps_a_diagonal_stem():
    """26-connectivity must not sever a stem that climbs diagonally."""
    grid = np.zeros((12, 12, 12), dtype=bool)
    for step in range(10):
        grid[step, step, step] = True
    grid[11, 11, 0] = True   # an isolated speck

    kept = largest_connected_component(grid)
    assert kept.sum() == 10
    assert not kept[11, 11, 0]


def test_voxel_iou_bounds():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    assert voxel_iou(a, b) == 1.0          # both empty

    a[0, 0, 0] = a[0, 0, 1] = True
    b[0, 0, 1] = b[0, 0, 2] = True
    assert voxel_iou(a, b) == pytest.approx(1 / 3)
    assert voxel_iou(a, a) == 1.0


def test_regression_metrics_against_hand_computed_values():
    predicted = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.0, 2.5])

    metrics = regression_metrics(predicted, target)
    assert metrics.mae_kg == pytest.approx(1 / 3)
    assert metrics.rmse_kg == pytest.approx(math.sqrt(0.5 / 3))
    assert metrics.bias_kg == pytest.approx(0.0)
    assert metrics.n == 3


def test_r_squared_is_zero_for_the_mean_predictor():
    target = np.array([0.4, 1.0, 2.0, 1.6])
    predicted = np.full_like(target, target.mean())
    assert regression_metrics(predicted, target).r2 == pytest.approx(0.0)


def test_perfect_reconstruction_scores_perfectly():
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.3, 0.3, size=(200, 3))

    metrics = reconstruction_metrics(points, points, threshold_m=0.01)
    assert metrics.chamfer_m == pytest.approx(0.0)
    assert metrics.f_score == pytest.approx(1.0)


def test_f_score_can_stay_high_while_volume_is_wrong():
    """The surface/volume disagreement this project is meant to measure.

    A hollow shell reproduces the surface exactly, so at any tolerance that
    spans its thickness the F-score is perfect -- while it holds barely half the
    true volume. An evaluation reporting only a surface metric would call this a
    flawless reconstruction; volumetric IoU does not.
    """
    grid = np.zeros((16, 16, 16), dtype=bool)
    grid[4:12, 4:12, 4:12] = True

    shell = grid.copy()
    shell[5:11, 5:11, 5:11] = False

    coords = np.stack(np.meshgrid(*[np.arange(16)] * 3, indexing="ij"), axis=-1)
    metrics = reconstruction_metrics(
        coords[shell].astype(float),
        coords[grid].astype(float),
        predicted_grid=shell,
        truth_grid=grid,
        threshold_m=3.5,
    )

    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f_score == pytest.approx(1.0)
    assert metrics.voxel_iou < 0.6


def test_hd95_ignores_a_single_outlier_that_hausdorff_reports():
    """Why HD95 and not raw Hausdorff on a carved hull."""
    from ggssvt.eval.metrics import hausdorff

    rng = np.random.default_rng(0)
    truth = rng.uniform(0, 0.2, size=(500, 3))
    predicted = np.vstack([truth, np.array([[9.0, 9.0, 9.0]])])   # one stray speck

    result = hausdorff(predicted, truth)
    assert result["hausdorff"] > 5.0        # the speck sets the worst case
    assert result["hd95"] < 0.1             # the percentile shrugs it off


def test_hd95_recall_detects_a_missing_branch():
    """The direction that measures missing canopy."""
    from ggssvt.eval.metrics import hausdorff

    body = np.random.default_rng(1).uniform(0, 0.1, size=(400, 3))
    branch = np.column_stack(
        [np.full(60, 0.05), np.full(60, 0.05), np.linspace(0.3, 0.8, 60)]
    )
    truth = np.vstack([body, branch])

    complete = hausdorff(truth, truth)
    missing = hausdorff(body, truth)        # reconstruction dropped the branch

    assert complete["hd95_recall"] == pytest.approx(0.0, abs=1e-9)
    assert missing["hd95_recall"] > 0.3
    # Precision is unaffected: everything predicted is still correct.
    assert missing["hd95_precision"] == pytest.approx(0.0, abs=1e-9)


def test_psnr_is_infinite_for_identical_images_and_finite_otherwise():
    from ggssvt.eval.metrics import psnr

    image = np.random.default_rng(0).random((16, 16, 3))
    assert psnr(image, image) == float("inf")

    noisy = np.clip(image + 0.1, 0, 1)
    assert 15.0 < psnr(noisy, image) < 40.0


def test_carve_thresholds_scale_with_the_view_count():
    """A four-view carve must return a poor volume, not an empty one.

    The tuned thresholds (6 informative views, 3 carve votes) were chosen against
    the full 12-view sweep. Held fixed, they make any carve below six views
    return nothing at all -- no voxel can have six informative views when only
    four exist -- so a view-count ablation would measure the leftover constant
    rather than the reconstruction.
    """
    from ggssvt.config import KINECT_V2
    from ggssvt.geometry.carving import carve
    from ggssvt.geometry.rig import RigSolution, nominal_view_pose
    from ggssvt.geometry.segment import ViewSegmentation

    def stub(position_id, pose):
        # A subject filling the frame at a constant range: every voxel near the
        # origin is supported and nothing is carved.
        depth = np.full((KINECT_V2.height, KINECT_V2.width), 1.4, dtype=np.float32)
        mask = np.ones_like(depth, dtype=bool)
        return ViewSegmentation(
            position_id=position_id, mask=mask, depth_m=depth,
            points_world=np.zeros((0, 3), dtype=np.float32), colours=None,
        )

    for n_views in (3, 4, 6, 12):
        step = 360 // n_views
        ids = [f"camA_{a:03d}" for a in range(0, 360, step)]
        poses = {p: nominal_view_pose(a, position_id=p)
                 for p, a in zip(ids, range(0, 360, step))}
        rig = RigSolution(plant_id="stub", poses=poses, warnings=[])
        segs = {p: stub(p, poses[p]) for p in ids}

        volume = carve(rig, segs, plant_id="stub")
        assert volume.occupancy.any(), (
            f"{n_views} views produced an empty volume; the thresholds did not scale"
        )


def test_carve_thresholds_can_still_be_overridden():
    """The scaling is a default, not a policy."""
    import inspect

    from ggssvt.geometry.carving import carve

    signature = inspect.signature(carve)
    assert signature.parameters["min_informative_views"].default is None
    assert signature.parameters["max_carve_votes"].default is None
