"""Nerfstudio export, and the coordinate convention that is easy to get wrong."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ggssvt.config import PLANTS_DIR
from ggssvt.eval.nerfstudio_export import (
    OPENCV_TO_OPENGL,
    build_rig_positions,
    build_transforms,
    intrinsics_dict,
    nerfstudio_pose_to_opencv,
    opencv_pose_to_nerfstudio,
)
from ggssvt.geometry.rig import nominal_view_pose

requires_dataset = pytest.mark.skipif(
    not PLANTS_DIR.exists(), reason="dataset/plants is not present"
)


@pytest.mark.parametrize("azimuth", [0, 30, 90, 180, 270, 330])
def test_exported_camera_looks_at_the_plant_with_y_up(azimuth):
    """The whole point of the conversion, stated as an assertion.

    Nerfstudio cameras look down -z with +y up. Export without the flip and the
    scene trains upside down and back to front -- which reads as a failed
    reconstruction rather than a failed export, so it is worth pinning here.
    """
    pose = nominal_view_pose(azimuth)
    matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)

    centre = matrix[:3, 3]
    forward = -matrix[:3, 2]          # OpenGL convention
    up = matrix[:3, 1]

    inward = -centre[:2] / np.linalg.norm(centre[:2])
    assert forward[:2] @ inward / np.linalg.norm(forward[:2]) == pytest.approx(1.0, abs=1e-6)
    assert up @ np.array([0.0, 0.0, 1.0]) == pytest.approx(1.0, abs=1e-6)


def test_conversion_stays_a_proper_rotation():
    pose = nominal_view_pose(60)
    matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)

    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_conversion_preserves_the_camera_centre():
    """Only the axes flip; the camera does not move."""
    pose = nominal_view_pose(120)
    matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)
    assert np.allclose(matrix[:3, 3], pose.centre, atol=1e-12)


@pytest.mark.parametrize("azimuth", [0, 45, 210])
def test_conversion_round_trips(azimuth):
    pose = nominal_view_pose(azimuth)
    matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)
    rotation, centre = nerfstudio_pose_to_opencv(matrix)

    assert np.allclose(rotation, pose.rotation, atol=1e-12)
    assert np.allclose(centre, pose.centre, atol=1e-12)


def test_the_flip_is_actually_applied():
    """Guards against someone 'simplifying' the conversion to an identity."""
    assert not np.allclose(OPENCV_TO_OPENGL, np.eye(4))
    pose = nominal_view_pose(0)
    naive = np.eye(4)
    naive[:3, :3] = pose.rotation
    naive[:3, 3] = pose.centre
    assert not np.allclose(opencv_pose_to_nerfstudio(pose.rotation, pose.centre), naive)


def test_intrinsics_match_the_kinect_frame():
    block = intrinsics_dict()
    assert block["w"] == 512
    assert block["h"] == 424
    assert block["fl_x"] == pytest.approx(365.456)
    assert block["k1"] == 0.0


@requires_dataset
def test_transforms_payload_is_well_formed():
    from ggssvt.data.dataset import load_specimen
    from ggssvt.geometry.rig import estimate_rig

    specimen = load_specimen("M001")
    rig = estimate_rig(specimen)
    payload = build_transforms(specimen, rig)

    assert payload["camera_model"] == "OPENCV"
    assert len(payload["frames"]) == 12
    assert payload["depth_unit_scale_factor"] == pytest.approx(0.001)

    for frame in payload["frames"]:
        assert (specimen.root / frame["file_path"]).exists()
        assert (specimen.root / frame["depth_file_path"]).exists()
        matrix = np.array(frame["transform_matrix"])
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[3], [0, 0, 0, 1])

    # It must survive a JSON round trip -- numpy floats would not.
    json.loads(json.dumps(payload))


@requires_dataset
def test_cameras_sit_on_a_ring_above_the_floor():
    from ggssvt.data.dataset import load_specimen
    from ggssvt.geometry.rig import estimate_rig

    specimen = load_specimen("M001")
    payload = build_transforms(specimen, estimate_rig(specimen))
    centres = np.array([f["transform_matrix"] for f in payload["frames"]])[:, :3, 3]

    radii = np.linalg.norm(centres[:, :2], axis=1)
    assert radii.std() < 0.08, "cameras should sit at a near-constant radius"
    assert (centres[:, 2] > 0.3).all(), "cameras should be above the floor"


@requires_dataset
def test_rig_positions_cover_every_position_id():
    """make_transforms.py looks positions up by these exact strings."""
    from ggssvt.data.dataset import load_specimen
    from ggssvt.geometry.rig import estimate_rig

    specimen = load_specimen("M001")
    positions = build_rig_positions(specimen, estimate_rig(specimen))

    assert set(positions) == {v.position_id for v in specimen.views}
    for entry in positions.values():
        assert entry["camera"] in {"camA", "camB"}
        assert np.array(entry["transform_matrix"]).shape == (4, 4)
