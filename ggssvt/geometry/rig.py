"""Calibration-free rig registration.

``dataset/calib`` is empty: no ChArUco intrinsics or per-day ``rig_positions``
were ever captured, so there are no measured extrinsics to load. This module
recovers them from the depth data itself, which is possible because the capture
protocol constrains the rig heavily:

1. **Floor plane.** RANSAC on each depth frame gives the camera's tilt, roll and
   height above the floor -- four of six degrees of freedom, per view, with no
   target. See :mod:`ggssvt.geometry.plane`.
2. **Subject axis.** The plant stands on the floor near the centre of frame.
   The densest above-floor column fixes the remaining two translation degrees of
   freedom by putting the world origin on the plant axis at floor level.
3. **Azimuth.** The filename gives the nominal azimuth (0, 30, ... 330). Because
   the cameras are hand-repositioned rather than mounted on an indexed turntable,
   :func:`refine_azimuths` optionally corrects each one by minimising the volume
   of the resulting visual hull -- misaligned silhouettes inflate the hull, so
   the tightest hull is the best-registered one.

World frame convention: ``+z`` up from the floor, origin on the plant axis, and
a camera at azimuth ``theta`` sits at ``(r cos theta, r sin theta, h)`` looking
inward.

Everything here is NumPy; no PyTorch and no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import (
    KINECT_V2,
    NOMINAL_CAM_HEIGHT_M,
    NOMINAL_CAM_RADIUS_M,
    ROI_RADIUS_M,
    ROI_Z_MAX_M,
    Intrinsics,
)
from ..data.dataset import Specimen
from ..data.io import backproject, depth_validity
from .plane import Plane, PlaneFitError, fit_plane_ransac
from .refine import RefinementResult, refine_registration

# The camera looks along +z with +y down, so "up" in the camera frame is -y.
CAMERA_UP_PRIOR = np.array([0.0, -1.0, 0.0])

# Subject search band: the plant stands between these distances from the camera.
SUBJECT_MIN_DIST_M = 0.6
SUBJECT_MAX_DIST_M = 2.6
SUBJECT_MIN_HEIGHT_M = 0.10
SUBJECT_HIST_BIN_M = 0.05
SUBJECT_LAYER_M = 0.10          # height-layer thickness used to score columns
SUBJECT_MAX_OFFAXIS_M = 0.35    # the operator frames the plant on the optical axis

MAX_SUBJECT_HYPOTHESES = 6      # distinct subject distances scored per specimen
AGREEMENT_Z_MIN_M = 0.05        # clear of the floor when scoring agreement
MIN_ACCEPTABLE_AGREEMENT = 0.25 # below this the registration is flagged
LARGE_AZIMUTH_CORRECTION_DEG = 6.0  # corrections beyond this are worth reporting


@dataclass(frozen=True)
class ViewPose:
    """Extrinsics for one view, plus the diagnostics that produced them.

    ``rotation`` and ``centre`` map camera-frame points into the world frame::

        x_world = rotation @ x_cam + centre
    """

    position_id: str
    azimuth_deg: float
    rotation: np.ndarray      # (3, 3) world_from_cam
    centre: np.ndarray        # (3,) camera centre in world coordinates
    camera_height_m: float
    tilt_deg: float
    subject_distance_m: float
    floor_inlier_fraction: float

    @property
    def world_from_cam(self) -> np.ndarray:
        """The 4x4 homogeneous transform."""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.centre
        return transform

    @property
    def cam_from_world(self) -> np.ndarray:
        """Inverse of :attr:`world_from_cam`."""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation.T
        transform[:3, 3] = -self.rotation.T @ self.centre
        return transform

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` world points into this camera's frame."""
        points_world = np.asarray(points_world, dtype=np.float64)
        return (points_world - self.centre) @ self.rotation

    def to_world(self, points_cam: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` camera-frame points into the world frame."""
        points_cam = np.asarray(points_cam, dtype=np.float64)
        return points_cam @ self.rotation.T + self.centre


class RigEstimationError(RuntimeError):
    """Raised when a view's pose cannot be recovered from its depth frame."""


def levelling_rotation(plane: Plane, camera_forward: np.ndarray | None = None) -> np.ndarray:
    """Build the rotation taking camera coordinates to a floor-levelled frame.

    The levelled frame has ``+x`` right, ``+y`` forward along the camera's
    horizontal viewing direction, and ``+z`` up along the floor normal.

    Args:
        plane: the fitted floor, normal already oriented upward.
        camera_forward: the camera's optical axis in camera coordinates.
            Defaults to ``+z``.

    Returns:
        ``(3, 3)`` rotation ``M`` such that ``x_level = M @ x_cam``.
    """
    up = np.asarray(plane.normal, dtype=np.float64)
    up = up / np.linalg.norm(up)

    forward = np.array([0.0, 0.0, 1.0]) if camera_forward is None else np.asarray(
        camera_forward, dtype=np.float64
    )
    forward = forward - (forward @ up) * up
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        raise RigEstimationError(
            "camera optical axis is parallel to the floor normal; cannot level"
        )
    forward = forward / norm

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    return np.stack([right, forward, up])


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def fit_floor(
    depth_m: np.ndarray,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    rng: np.random.Generator | None = None,
) -> Plane:
    """Fit the floor plane in one depth frame.

    Candidates are restricted to the lower two thirds of the image, where the
    floor is, and to hypotheses whose normal is within 45 degrees of the
    camera's up axis.
    """
    points = backproject(depth_m, intrinsics)
    valid = depth_validity(depth_m)

    # The floor occupies the lower part of the frame in every capture.
    row_gate = np.zeros_like(valid)
    row_gate[int(0.35 * depth_m.shape[0]) :, :] = True

    candidates = points[valid & row_gate].astype(np.float64)
    if candidates.shape[0] < 500:
        raise RigEstimationError(
            f"only {candidates.shape[0]} candidate floor points; depth frame is too sparse"
        )

    try:
        return fit_plane_ransac(
            candidates,
            normal_prior=CAMERA_UP_PRIOR,
            prior_tolerance=0.7,
            rng=rng,
        )
    except PlaneFitError as exc:
        raise RigEstimationError(f"floor fit failed: {exc}") from exc


@dataclass(frozen=True)
class SubjectCandidate:
    """A column of above-floor points that might be the plant."""

    axis_xy: np.ndarray     # (2,) horizontal position in the levelled frame
    layers: int             # distinct 10 cm height layers the column touches
    distance_m: float       # horizontal range from the camera


def subject_candidates(
    points_level: np.ndarray,
    *,
    min_dist_m: float = SUBJECT_MIN_DIST_M,
    max_dist_m: float = SUBJECT_MAX_DIST_M,
    bin_m: float = SUBJECT_HIST_BIN_M,
    max_offaxis_m: float = SUBJECT_MAX_OFFAXIS_M,
    top_k: int = 8,
) -> list[SubjectCandidate]:
    """Rank the columns of above-floor points that could be the plant.

    Columns are scored by how many distinct 10 cm height layers they touch,
    rather than by raw point count, so a tall thin plant outranks the wide flat
    speckle of the floor. The rig's own structural poles score well on that
    metric too, which is why the caller must disambiguate -- see
    :func:`estimate_rig`, which takes the candidate consistent with the rest of
    the sweep.

    Args:
        points_level: ``(N, 3)`` levelled-frame points, ``z`` = height above floor.
        max_offaxis_m: reject columns further than this from the camera's
            optical axis. The operator frames the plant centrally, so the
            subject is always near ``x_level = 0``.
        top_k: how many candidates to return, best first.

    Returns:
        Candidates ordered by descending layer count. Never empty.

    Raises:
        RigEstimationError: if too few above-floor points survive the gates.
    """
    points_level = np.asarray(points_level, dtype=np.float64)
    height = points_level[:, 2]
    distance = np.linalg.norm(points_level[:, :2], axis=1)

    gate = (
        (height > SUBJECT_MIN_HEIGHT_M)
        & (height < ROI_Z_MAX_M)
        & (distance > min_dist_m)
        & (distance < max_dist_m)
        & (np.abs(points_level[:, 0]) < max_offaxis_m)
    )
    subject = points_level[gate]
    if subject.shape[0] < 200:
        raise RigEstimationError(
            f"only {subject.shape[0]} above-floor points near the optical axis; "
            "the subject could not be located"
        )

    xs, ys, zs = subject[:, 0], subject[:, 1], subject[:, 2]
    ix = np.floor((xs - xs.min()) / bin_m).astype(np.int64)
    iy = np.floor((ys - ys.min()) / bin_m).astype(np.int64)
    iz = np.floor(zs / SUBJECT_LAYER_M).astype(np.int64)

    n_y = int(iy.max()) + 1
    n_z = int(iz.max()) + 1
    column = ix * n_y + iy

    # One entry per (column, height layer) pair, then count layers per column.
    unique_columns, layer_counts = np.unique(
        np.unique(column * n_z + iz) // n_z, return_counts=True
    )

    order = np.argsort(layer_counts)[::-1][:top_k]
    candidates: list[SubjectCandidate] = []
    for index in order:
        bx, by = divmod(int(unique_columns[index]), n_y)
        near = (np.abs(ix - bx) <= 1) & (np.abs(iy - by) <= 1)
        if near.sum() >= 10:
            axis_xy = np.median(subject[near][:, :2], axis=0)
        else:
            axis_xy = np.array(
                [xs.min() + (bx + 0.5) * bin_m, ys.min() + (by + 0.5) * bin_m]
            )
        candidates.append(
            SubjectCandidate(
                axis_xy=axis_xy.astype(np.float64),
                layers=int(layer_counts[index]),
                distance_m=float(np.linalg.norm(axis_xy)),
            )
        )
    return candidates


def estimate_subject_axis(
    points_level: np.ndarray, **kwargs
) -> tuple[np.ndarray, float]:
    """Best single subject-axis estimate for one view.

    Convenience wrapper over :func:`subject_candidates` for standalone use.
    :func:`estimate_rig` uses the full candidate list instead, because the
    across-view consensus disambiguates the plant from the rig structure far
    more reliably than any single frame can.

    Returns:
        ``(axis_xy, support)``.
    """
    best = subject_candidates(points_level, **kwargs)[0]
    return best.axis_xy, float(best.layers)


@dataclass(frozen=True)
class ViewGeometry:
    """Everything recoverable from one depth frame before the azimuth is applied."""

    plane: Plane
    level_from_cam: np.ndarray
    points_level: np.ndarray
    candidates: list[SubjectCandidate]


def analyse_view(
    depth_m: np.ndarray,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    rng: np.random.Generator | None = None,
) -> ViewGeometry:
    """Fit the floor, level the point cloud, and rank subject candidates.

    Split out from :func:`estimate_view_pose` so :func:`estimate_rig` can do the
    expensive per-frame work once and still revisit the candidate choice after
    seeing the whole sweep.
    """
    plane = fit_floor(depth_m, intrinsics=intrinsics, rng=rng)
    level_from_cam = levelling_rotation(plane)

    points_cam = backproject(depth_m, intrinsics)[depth_validity(depth_m)]
    points_level = points_cam.astype(np.float64) @ level_from_cam.T
    points_level[:, 2] += plane.camera_height_m   # z now measures height above floor

    return ViewGeometry(
        plane=plane,
        level_from_cam=level_from_cam,
        points_level=points_level,
        candidates=subject_candidates(points_level),
    )


def pose_from_geometry(
    geometry: ViewGeometry,
    azimuth_deg: float,
    axis_xy: np.ndarray,
    *,
    position_id: str = "",
) -> ViewPose:
    """Assemble a :class:`ViewPose` from levelled geometry and a chosen axis."""
    plane = geometry.plane
    level_from_cam = geometry.level_from_cam

    # Translate so the world origin sits on the plant axis at floor level.
    translation = np.array([axis_xy[0], axis_xy[1], -plane.camera_height_m])

    # In the levelled frame the camera looks along +y, so it sits at azimuth
    # 270 degrees relative to the subject. Rotate it onto the nominal azimuth.
    yaw = math.radians(azimuth_deg) + math.pi / 2.0
    yaw_rotation = _rotation_z(yaw)

    rotation = yaw_rotation @ level_from_cam
    centre = -yaw_rotation @ translation

    tilt_deg = math.degrees(
        math.acos(float(np.clip(plane.normal @ CAMERA_UP_PRIOR, -1.0, 1.0)))
    )

    return ViewPose(
        position_id=position_id,
        azimuth_deg=float(azimuth_deg),
        rotation=rotation,
        centre=centre,
        camera_height_m=plane.camera_height_m,
        tilt_deg=tilt_deg,
        subject_distance_m=float(np.linalg.norm(axis_xy)),
        floor_inlier_fraction=plane.inlier_fraction,
    )


def estimate_view_pose(
    depth_m: np.ndarray,
    azimuth_deg: float,
    *,
    position_id: str = "",
    intrinsics: Intrinsics = KINECT_V2,
    rng: np.random.Generator | None = None,
) -> ViewPose:
    """Recover one view's extrinsics from its depth frame and nominal azimuth.

    Single-frame convenience path. It takes the top-ranked subject candidate,
    which can be fooled by the rig's structural poles; prefer
    :func:`estimate_rig` whenever the whole sweep is available.

    Raises:
        RigEstimationError: if the floor or the subject cannot be found.
    """
    geometry = analyse_view(depth_m, intrinsics=intrinsics, rng=rng)
    return pose_from_geometry(
        geometry,
        azimuth_deg,
        geometry.candidates[0].axis_xy,
        position_id=position_id,
    )


def nominal_view_pose(azimuth_deg: float, *, position_id: str = "") -> ViewPose:
    """The pose the protocol implies, ignoring the data.

    Used as a fallback when depth-based estimation fails on a view, and as the
    control condition in the rig ablation.
    """
    yaw = math.radians(azimuth_deg) + math.pi / 2.0
    level_from_cam = levelling_rotation(
        Plane(normal=CAMERA_UP_PRIOR.copy(), offset=NOMINAL_CAM_HEIGHT_M, inlier_fraction=1.0)
    )
    translation = np.array([0.0, NOMINAL_CAM_RADIUS_M, -NOMINAL_CAM_HEIGHT_M])
    yaw_rotation = _rotation_z(yaw)
    return ViewPose(
        position_id=position_id,
        azimuth_deg=float(azimuth_deg),
        rotation=yaw_rotation @ level_from_cam,
        centre=-yaw_rotation @ translation,
        camera_height_m=NOMINAL_CAM_HEIGHT_M,
        tilt_deg=0.0,
        subject_distance_m=NOMINAL_CAM_RADIUS_M,
        floor_inlier_fraction=float("nan"),
    )


@dataclass
class RigSolution:
    """Estimated extrinsics for every view of one specimen."""

    plant_id: str
    poses: dict[str, ViewPose]
    warnings: list[str]
    subject_distance_m: float = float("nan")
    agreement: float = float("nan")
    refinement: RefinementResult | None = None

    @property
    def n_views(self) -> int:
        return len(self.poses)

    def pose(self, position_id: str) -> ViewPose:
        return self.poses[position_id]

    def summary(self) -> dict:
        heights = np.array([p.camera_height_m for p in self.poses.values()])
        distances = np.array([p.subject_distance_m for p in self.poses.values()])
        tilts = np.array([p.tilt_deg for p in self.poses.values()])
        inliers = np.array([p.floor_inlier_fraction for p in self.poses.values()])
        return {
            "plant_id": self.plant_id,
            "n_views": self.n_views,
            "camera_height_m": {"mean": float(heights.mean()), "std": float(heights.std())},
            "subject_distance_m": {
                "mean": float(distances.mean()),
                "std": float(distances.std()),
            },
            "tilt_deg": {"mean": float(tilts.mean()), "std": float(tilts.std())},
            "floor_inlier_fraction": {"mean": float(np.nanmean(inliers))},
            "agreement": self.agreement,
            "refined_agreement": (
                None if self.refinement is None else self.refinement.score_after
            ),
            "n_warnings": len(self.warnings),
        }


def _world_points(
    geometry: ViewGeometry, azimuth_deg: float, axis_xy: np.ndarray
) -> np.ndarray:
    """Map a view's levelled points into the world frame for a given axis choice.

    Mirrors the transform :func:`pose_from_geometry` builds, but works straight
    off the cached levelled cloud so hypotheses can be scored without touching
    the depth images again.
    """
    points = geometry.points_level.copy()
    points[:, 0] -= axis_xy[0]
    points[:, 1] -= axis_xy[1]
    yaw = math.radians(azimuth_deg) + math.pi / 2.0
    return points @ _rotation_z(yaw).T


def _hypothesis_score(
    geometries: dict[str, ViewGeometry],
    azimuths: dict[str, float],
    choices: dict[str, SubjectCandidate],
    *,
    voxel_m: float = 0.02,
    min_points_per_view: int = 120,
) -> float:
    """Score a subject hypothesis by how well the views' clouds coincide.

    This is the criterion that actually distinguishes the plant from the
    background: put the world origin on the real subject and all twelve clouds
    land on the same surfaces, so most occupied voxels are seen by several
    cameras. Put it on a wall or a rig pole and each view contributes its own
    disjoint shell, because that structure is at a different world location in
    every frame.

    Returns:
        Fraction of occupied voxels supported by three or more views, or 0.0 if
        the hypothesis leaves too little of the subject visible to judge.
    """
    counts: dict[tuple[int, int, int], int] = {}
    per_view_totals: list[int] = []

    for position_id, geometry in geometries.items():
        world = _world_points(geometry, azimuths[position_id], choices[position_id].axis_xy)

        radial = np.linalg.norm(world[:, :2], axis=1)
        keep = (
            (radial < ROI_RADIUS_M)
            & (world[:, 2] > AGREEMENT_Z_MIN_M)
            & (world[:, 2] < ROI_Z_MAX_M)
        )
        subject = world[keep]
        per_view_totals.append(int(subject.shape[0]))
        if subject.shape[0] == 0:
            continue

        keys = np.floor(subject / voxel_m).astype(np.int32)
        for key in {tuple(k) for k in keys}:
            counts[key] = counts.get(key, 0) + 1

    if not counts or not per_view_totals:
        return 0.0
    if float(np.median(per_view_totals)) < min_points_per_view:
        return 0.0

    shared = sum(1 for c in counts.values() if c >= 3)
    return shared / len(counts)


def _candidate_targets(
    geometries: dict[str, ViewGeometry], *, bandwidth_m: float = 0.12
) -> list[float]:
    """Distinct subject-distance hypotheses worth testing, strongest first."""
    distances = np.array(
        [c.distance_m for g in geometries.values() for c in g.candidates]
    )
    weights = np.array(
        [float(c.layers) for g in geometries.values() for c in g.candidates]
    )
    if distances.size == 0:
        return [NOMINAL_CAM_RADIUS_M]

    density = (
        np.exp(-0.5 * ((distances[:, None] - distances[None, :]) / bandwidth_m) ** 2)
        * weights[None, :]
    ).sum(axis=1)

    targets: list[float] = []
    for index in np.argsort(density)[::-1]:
        value = float(distances[index])
        if all(abs(value - t) > bandwidth_m for t in targets):
            targets.append(value)
        if len(targets) >= MAX_SUBJECT_HYPOTHESES:
            break
    return targets


def _select_subject_hypothesis(
    geometries: dict[str, ViewGeometry], azimuths: dict[str, float]
) -> tuple[float, float]:
    """Pick the subject distance whose registration the views actually agree on.

    Returns:
        ``(distance_m, agreement)`` for the winning hypothesis.
    """
    best_target = NOMINAL_CAM_RADIUS_M
    best_score = -1.0

    for target in _candidate_targets(geometries):
        choices = {
            position_id: min(
                geometry.candidates, key=lambda c: abs(c.distance_m - target)
            )
            for position_id, geometry in geometries.items()
        }
        score = _hypothesis_score(geometries, azimuths, choices)
        if score > best_score:
            best_target, best_score = target, score

    return best_target, best_score


def estimate_rig(
    specimen: Specimen,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    seed: int = 0,
    fallback_to_nominal: bool = True,
    use_consensus: bool = True,
    refine: bool = True,
) -> RigSolution:
    """Estimate extrinsics for every view of a specimen.

    Runs in two passes. The first fits each view's floor and ranks its subject
    candidates; the second picks, per view, the candidate closest to the rig
    radius the whole sweep agrees on. Views whose geometry cannot be recovered
    fall back to the nominal protocol pose (when ``fallback_to_nominal``) and
    are recorded in ``warnings``.
    """
    warnings: list[str] = []
    rng = np.random.default_rng(seed)

    geometries: dict[str, ViewGeometry] = {}
    azimuths: dict[str, float] = {}
    failed: list[str] = []

    for view in specimen.views:
        azimuths[view.position_id] = float(view.azimuth_deg)
        try:
            geometries[view.position_id] = analyse_view(
                view.load_depth(), intrinsics=intrinsics, rng=rng
            )
        except RigEstimationError as exc:
            warnings.append(f"{view.position_id}: {exc}")
            if not fallback_to_nominal:
                raise
            failed.append(view.position_id)

    if use_consensus and geometries:
        target, agreement = _select_subject_hypothesis(geometries, azimuths)
        if agreement < MIN_ACCEPTABLE_AGREEMENT:
            warnings.append(
                f"best subject hypothesis at {target:.2f} m scores only "
                f"{agreement:.2f} multi-view agreement; registration is "
                f"unreliable for this specimen"
            )
    else:
        target, agreement = NOMINAL_CAM_RADIUS_M, float("nan")

    choices = {
        position_id: min(geometry.candidates, key=lambda c: abs(c.distance_m - target))
        for position_id, geometry in geometries.items()
    }

    refinement: RefinementResult | None = None
    if refine and len(geometries) >= 3:
        refinement = refine_registration(
            {pid: g.points_level for pid, g in geometries.items()},
            {pid: azimuths[pid] for pid in geometries},
            {pid: choices[pid].axis_xy for pid in geometries},
            seed=seed,
        )
        if refinement.improvement < 0:
            warnings.append(
                "registration refinement did not improve agreement "
                f"({refinement.score_before:.3f} -> {refinement.score_after:.3f}); "
                "keeping the nominal azimuths"
            )
            refinement = None

    poses: dict[str, ViewPose] = {}
    for position_id, geometry in geometries.items():
        chosen = choices[position_id]
        axis_xy = chosen.axis_xy
        azimuth = azimuths[position_id]

        if refinement is not None:
            correction = refinement.corrections[position_id]
            axis_xy = axis_xy + np.array([correction.dx_m, correction.dy_m])
            azimuth = azimuth + correction.d_azimuth_deg
            if abs(correction.d_azimuth_deg) >= LARGE_AZIMUTH_CORRECTION_DEG:
                warnings.append(
                    f"{position_id}: azimuth corrected by "
                    f"{correction.d_azimuth_deg:+.1f} degrees from its nominal "
                    f"{azimuths[position_id]:.0f}; check the rig placement for this step"
                )

        poses[position_id] = pose_from_geometry(
            geometry, azimuth, axis_xy, position_id=position_id
        )

    for position_id in failed:
        poses[position_id] = nominal_view_pose(
            azimuths[position_id], position_id=position_id
        )

    solution = RigSolution(
        plant_id=specimen.plant_id,
        poses=poses,
        warnings=warnings,
        subject_distance_m=float(target),
        agreement=float(agreement),
        refinement=refinement,
    )
    _flag_rig_outliers(solution)
    return solution


def _flag_rig_outliers(solution: RigSolution, *, z_threshold: float = 2.5) -> None:
    """Append warnings for views whose height or distance disagrees with the rest.

    The rig is carried by hand between positions, so a view that sits far from
    the group is more likely a repositioning error than a real measurement.
    """
    if solution.n_views < 4:
        return

    for label, getter in (
        ("camera height", lambda p: p.camera_height_m),
        ("subject distance", lambda p: p.subject_distance_m),
    ):
        values = np.array([getter(p) for p in solution.poses.values()])
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad < 1e-6:
            continue
        scores = 0.6745 * (values - median) / mad
        for position_id, score, value in zip(solution.poses, scores, values):
            if abs(score) > z_threshold:
                solution.warnings.append(
                    f"{position_id}: {label} {value:.3f} m deviates from the "
                    f"specimen median {median:.3f} m (robust z={score:+.1f})"
                )


__all__ = [
    "CAMERA_UP_PRIOR",
    "RigEstimationError",
    "RigSolution",
    "ViewPose",
    "estimate_rig",
    "estimate_subject_axis",
    "estimate_view_pose",
    "fit_floor",
    "levelling_rotation",
    "nominal_view_pose",
]
