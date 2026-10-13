import json

import cv2
import numpy as np
import pytest

from rig_calibration.collect_specimen_stationary_kinect import (
    _capture_sweep,
    checkerboard_pose,
    checkerboard_object_points,
    reject_pose_discontinuities,
    reference_camera_transforms,
    validate_actual_angle,
    validate_step_degrees,
)


@pytest.mark.parametrize("step", [5, 10, 15, 30, 45, 60, 90, 120, 180])
def test_step_must_divide_full_rotation(step):
    assert validate_step_degrees(step) == step


@pytest.mark.parametrize("step", [0, -15, 7, 361])
def test_invalid_step_is_rejected(step):
    with pytest.raises(ValueError, match="divisor"):
        validate_step_degrees(step)


def test_measured_angle_must_be_close_to_expected_step():
    assert validate_actual_angle(17.0, frame_index=1, step_degrees=15) == 17.0
    with pytest.raises(ValueError, match="too far"):
        validate_actual_angle(28.0, frame_index=1, step_degrees=15)
    with pytest.raises(ValueError, match="range"):
        validate_actual_angle(360, frame_index=0, step_degrees=15)


def test_checkerboard_object_points_use_measured_square_size():
    points = checkerboard_object_points((3, 2), 0.025)
    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[0], [0, 0, 0])
    np.testing.assert_allclose(points[-1], [0.05, 0.025, 0])


def test_checkerboard_pose_detects_a_known_size_board():
    image = np.full((424, 512, 3), 210, dtype=np.uint8)
    square_px = 32
    start_x, start_y = 100, 100
    for row in range(6):
        for column in range(8):
            if (row + column) % 2 == 0:
                x0 = start_x + column * square_px
                y0 = start_y + row * square_px
                cv2.rectangle(
                    image,
                    (x0, y0),
                    (x0 + square_px, y0 + square_px),
                    (0, 0, 0),
                    thickness=-1,
                )
    cv2.circle(image, (89, 89), 7, (255, 0, 0), thickness=-1)

    pose = checkerboard_pose(
        image,
        inner_corners=(7, 5),
        square_size_m=0.025,
    )
    assert pose is not None
    assert pose["detected_inner_corners"] == 35
    assert pose["orientation_marker_detected"] is True
    assert pose["reprojection_rmse_px"] < 1.0
    assert np.asarray(pose["camera_from_turntable_board"]).shape == (4, 4)


def test_reference_transforms_map_each_pose_into_reference_board_frame():
    reference_pose = np.eye(4)
    current_pose = np.eye(4)
    current_pose[:3, 3] = [1, 0, 0]
    index, transforms = reference_camera_transforms(
        [
            {"camera_from_turntable_board": reference_pose.tolist()},
            {"camera_from_turntable_board": current_pose.tolist()},
        ]
    )
    assert index == 0
    np.testing.assert_allclose(transforms[0], np.eye(4))
    np.testing.assert_allclose(
        transforms[1],
        np.array(
            [
                [1, 0, 0, -1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        ),
    )


def test_discontinuous_checkerboard_pose_is_rejected():
    identity = np.eye(4)
    rotated = np.eye(4)
    rotated[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    poses, statuses = reject_pose_discontinuities(
        [
            {
                "camera_from_turntable_board": identity.tolist(),
                "orientation_marker_detected": True,
            },
            {
                "camera_from_turntable_board": rotated.tolist(),
                "orientation_marker_detected": True,
            },
            {
                "camera_from_turntable_board": identity.tolist(),
                "orientation_marker_detected": True,
            },
        ],
        step_degrees=15,
    )
    assert poses[0] is not None
    assert poses[1] is None
    assert poses[2] is not None
    assert statuses[1].startswith("rejected_pose_jump_")


class FakeKinect:
    serial = "fake-kinect"
    camera_info = {}

    def __init__(self):
        self.captures = 0

    def capture(self):
        self.captures += 1
        rgb = np.full((424, 512, 3), self.captures, dtype=np.uint8)
        depth = np.full((424, 512), 1500, dtype=np.uint16)
        return rgb, depth


def test_sweep_saves_registered_rgbd_and_explicit_missing_pose(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    camera = FakeKinect()
    path = _capture_sweep(
        output_dir=tmp_path / "plants" / "P001",
        sweep_id="P001",
        camera=camera,
        step_degrees=90,
        checkerboard_inner_corners=None,
        checkerboard_square_m=0.025,
    )

    manifest = json.loads(
        (path / "capture_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["frames"]) == 4
    assert [frame["target_angle_deg"] for frame in manifest["frames"]] == [
        0,
        90,
        180,
        270,
    ]
    assert all(frame["reference_camera_from_camera"] is None for frame in manifest["frames"])
    assert camera.captures == 4
    assert cv2.imread(
        str(path / "depth" / "view_000.png"), cv2.IMREAD_UNCHANGED
    )[0, 0] == 1500
