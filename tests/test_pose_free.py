"""Pose-free comparison: the alignment maths, and the guards around it.

The backends themselves cannot be exercised without a GPU and three cloned
repositories, so what is pinned here is everything between them and a number:
the similarity alignment, the scale recovery, the convention sanity check, and
the failure paths that decide whether a bad reconstruction is caught or quietly
reported as a result.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ggssvt.geometry.pose_free import (
    PoseFreeError,
    PoseFreeResult,
    available_backends,
    backend_is_available,
    compare_poses,
    recover_scale_from_depth,
    rotation_angle_deg,
    umeyama,
)
from ggssvt.geometry.pose_free_backends import BACKENDS, build_backend, sanity_check_result
from ggssvt.geometry.rig import RigSolution, nominal_view_pose

AZIMUTHS = list(range(0, 360, 30))
IDS = [f"camA_{a:03d}" for a in AZIMUTHS]


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _reference_rig() -> RigSolution:
    poses = {
        pid: nominal_view_pose(a, position_id=pid) for pid, a in zip(IDS, AZIMUTHS)
    }
    return RigSolution(plant_id="synthetic", poses=poses, warnings=[])


def _result_from(azimuths, *, frame_rotation=0.0, scale=1.0, offset=(0.0, 0.0, 0.0)):
    """Poses at the given azimuths, expressed in an arbitrary frame and scale."""
    poses = [nominal_view_pose(a) for a in azimuths]
    gauge = _rotation_z(frame_rotation)
    return PoseFreeResult(
        method="synthetic",
        position_ids=list(IDS),
        rotations=np.stack([gauge @ p.rotation for p in poses]),
        centres=np.stack([scale * (gauge @ p.centre) + np.array(offset) for p in poses]),
        points=np.zeros((0, 3)),
    )


# --- similarity alignment -------------------------------------------------


def test_umeyama_recovers_a_known_similarity_exactly():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(20, 3))
    rotation = _rotation_z(0.7)
    scale, translation = 2.35, np.array([1.0, -2.0, 0.5])
    target = scale * source @ rotation.T + translation

    fitted_r, fitted_t, fitted_s = umeyama(source, target)
    assert fitted_s == pytest.approx(scale, rel=1e-9)
    assert np.allclose(fitted_r, rotation, atol=1e-9)
    assert np.allclose(fitted_t, translation, atol=1e-9)


def test_umeyama_never_returns_a_reflection():
    """Without the guard a mirrored fit can beat the correct one.

    A reflection would report a flipped rig as well aligned, which is worse than
    reporting a large error -- it looks like agreement.
    """
    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 3))
    mirrored = source.copy()
    mirrored[:, 2] *= -1

    rotation, _, _ = umeyama(source, mirrored)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)


def test_umeyama_refuses_too_few_correspondences():
    with pytest.raises(ValueError):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


def test_rotation_angle_is_zero_for_identical_rotations():
    rotation = _rotation_z(0.4)
    assert rotation_angle_deg(rotation, rotation) == pytest.approx(0.0, abs=1e-9)
    assert rotation_angle_deg(np.eye(3), _rotation_z(math.radians(30))) == pytest.approx(
        30.0, abs=1e-6
    )


# --- pose comparison ------------------------------------------------------


def test_identical_poses_leave_no_residual():
    agreement = compare_poses(_result_from(AZIMUTHS), _reference_rig())
    assert agreement.centre_rmse_m == pytest.approx(0.0, abs=1e-9)
    assert agreement.azimuth_rmse_deg == pytest.approx(0.0, abs=1e-9)
    # Looser bound on the rotation term on purpose: it goes through acos, which
    # is ill-conditioned at zero angle, so identical rotations land a few
    # microdegrees from zero rather than at it. That is float noise, not error --
    # 1e-4 degrees is still eleven orders of magnitude below anything physical.
    assert agreement.rotation_rmse_deg == pytest.approx(0.0, abs=1e-4)


def test_gauge_freedom_is_removed_not_reported_as_error():
    """A different frame and scale is not disagreement, and must not read as it."""
    result = _result_from(
        AZIMUTHS, frame_rotation=1.1, scale=0.4, offset=(5.0, 3.0, -1.0)
    )
    agreement = compare_poses(result, _reference_rig())

    assert agreement.centre_rmse_m == pytest.approx(0.0, abs=1e-6)
    assert agreement.azimuth_rmse_deg == pytest.approx(0.0, abs=1e-6)
    assert agreement.scale == pytest.approx(1 / 0.4, rel=1e-6)


def test_an_injected_azimuth_error_is_recovered():
    """The measurement this whole comparison exists to make."""
    injected = np.array([3.0, -5.0, 8.0, -2.0, 0.0, 6.0, -7.0, 1.0, 4.0, -3.0, 2.0, -6.0])
    result = _result_from(
        [a + e for a, e in zip(AZIMUTHS, injected)],
        frame_rotation=1.1,
        scale=0.4,
        offset=(5.0, 3.0, -1.0),
    )
    agreement = compare_poses(result, _reference_rig())

    expected = float(np.sqrt((injected ** 2).mean()))
    # The similarity fit absorbs the mean rotation, so the recovered spread is
    # slightly below the injected one; it must still be close.
    assert agreement.azimuth_rmse_deg == pytest.approx(expected, rel=0.15)
    assert agreement.centre_rmse_m > 0.05


def test_comparison_refuses_when_too_few_views_are_shared():
    result = _result_from(AZIMUTHS)
    result.position_ids = ["camA_000", "nope_1", "nope_2"] + result.position_ids[3:]
    result.position_ids = result.position_ids[:3]
    result.rotations = result.rotations[:3]
    result.centres = result.centres[:3]

    with pytest.raises(PoseFreeError):
        compare_poses(result, RigSolution("x", {"camA_000": nominal_view_pose(0)}, []))


def test_per_view_errors_are_reported_not_just_the_aggregate():
    injected = [0.0] * 11 + [12.0]
    result = _result_from([a + e for a, e in zip(AZIMUTHS, injected)])
    agreement = compare_poses(result, _reference_rig())

    worst = max(agreement.per_view.items(), key=lambda kv: abs(kv[1]["azimuth_error_deg"]))
    assert worst[0] == IDS[-1]
    assert abs(worst[1]["azimuth_error_deg"]) > 5.0


# --- scale recovery -------------------------------------------------------


def test_scale_recovery_is_exact_on_a_clean_ratio():
    rng = np.random.default_rng(0)
    predicted = rng.uniform(0.5, 3.0, (100, 100))
    measured = predicted * 0.37
    measured[rng.random((100, 100)) < 0.3] = 0.0     # Kinect dropouts

    scale, diagnostics = recover_scale_from_depth(predicted, measured)
    assert scale == pytest.approx(0.37, rel=1e-9)
    assert diagnostics["n_pixels"] > 5000


def test_scale_recovery_survives_wild_outliers():
    """Ratio of medians, not least squares -- background junk must not set scale."""
    rng = np.random.default_rng(2)
    predicted = rng.uniform(0.8, 1.2, (80, 80))
    measured = predicted * 0.5
    corrupt = rng.random((80, 80)) < 0.1
    measured[corrupt] = 40.0                          # absurd far readings

    scale, _ = recover_scale_from_depth(predicted, measured)
    assert scale == pytest.approx(0.5, rel=0.05)


def test_scale_recovery_refuses_when_there_is_nothing_to_match():
    predicted = np.ones((50, 50))
    measured = np.zeros((50, 50))
    with pytest.raises(PoseFreeError):
        recover_scale_from_depth(predicted, measured)


def test_scale_recovery_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        recover_scale_from_depth(np.ones((10, 10)), np.ones((10, 11)))


# --- convention guards ----------------------------------------------------


def test_sanity_check_passes_a_well_formed_result():
    poses = [nominal_view_pose(a) for a in AZIMUTHS]
    result = PoseFreeResult(
        method="synthetic",
        position_ids=list(IDS),
        rotations=np.stack([p.rotation for p in poses]),
        centres=np.stack([p.centre for p in poses]),
        points=np.random.default_rng(0).normal(scale=0.1, size=(500, 3)),
    )
    assert sanity_check_result(result) == []


def test_sanity_check_catches_an_inside_out_pose_convention():
    """If an upstream release flips convention, this must say so."""
    poses = [nominal_view_pose(a) for a in AZIMUTHS]
    flip = np.diag([1.0, -1.0, -1.0])
    result = PoseFreeResult(
        method="synthetic",
        position_ids=list(IDS),
        rotations=np.stack([p.rotation @ flip for p in poses]),
        centres=np.stack([p.centre for p in poses]),
        points=np.random.default_rng(0).normal(scale=0.1, size=(500, 3)),
    )
    warnings = sanity_check_result(result)
    assert any("convention" in w for w in warnings)


def test_sanity_check_catches_a_non_rotation():
    result = PoseFreeResult(
        method="synthetic",
        position_ids=["camA_000"],
        rotations=np.array([np.diag([1.0, 1.0, 2.0])]),
        centres=np.zeros((1, 3)),
        points=np.zeros((0, 3)),
    )
    assert any("determinant" in w for w in sanity_check_result(result))


# --- backend plumbing -----------------------------------------------------


def test_every_backend_reports_install_instructions_when_missing():
    for name in BACKENDS:
        available, reason = backend_is_available(name)
        if not available:
            assert "git clone" in reason, f"{name} must say how to install it"


def test_unknown_backend_is_rejected():
    from ggssvt.geometry.pose_free import PoseFreeError as Error

    with pytest.raises(Error):
        build_backend("colmap")


def test_available_backends_covers_all_three():
    assert set(available_backends()) == {"dust3r", "mast3r", "fast3r"}


def test_mast3r_is_declared_metric_and_the_others_are_not():
    """Scale handling differs per backend and the code must know which is which."""
    assert BACKENDS["mast3r"].returns_metric_scale is True
    assert BACKENDS["dust3r"].returns_metric_scale is False
    assert BACKENDS["fast3r"].returns_metric_scale is False
