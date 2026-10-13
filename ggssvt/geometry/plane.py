"""RANSAC plane fitting, used to recover the floor in each camera frame.

The floor is the one structure visible from every rig position, and it fixes
four of the six extrinsic degrees of freedom (two tilt angles, the roll, and
the camera height) without any calibration target. That is what makes the
capture usable despite ``dataset/calib`` being empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FLOOR_INLIER_TOL_M, FLOOR_MIN_INLIER_FRAC, FLOOR_RANSAC_ITERS


@dataclass(frozen=True)
class Plane:
    """A plane ``n . x + d = 0`` with unit normal ``n``.

    The normal is oriented so that :meth:`signed_distance` is positive on the
    side the camera sits on, i.e. positive means "above the floor".
    """

    normal: np.ndarray   # (3,) unit
    offset: float
    inlier_fraction: float

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of ``(N, 3)`` points from the plane, in metres."""
        return points @ self.normal + self.offset

    @property
    def camera_height_m(self) -> float:
        """Distance from the camera centre (the origin) to the plane."""
        return float(self.offset)


class PlaneFitError(RuntimeError):
    """Raised when no plane explains enough of the point set."""


def fit_plane_ransac(
    points: np.ndarray,
    *,
    tolerance_m: float = FLOOR_INLIER_TOL_M,
    iterations: int = FLOOR_RANSAC_ITERS,
    min_inlier_fraction: float = FLOOR_MIN_INLIER_FRAC,
    normal_prior: np.ndarray | None = None,
    prior_tolerance: float = 0.7,
    rng: np.random.Generator | None = None,
) -> Plane:
    """Fit a plane to ``(N, 3)`` points by RANSAC, then refine on the inliers.

    Args:
        points: candidate points, camera frame, metres.
        tolerance_m: inlier band half-width.
        iterations: RANSAC trials.
        min_inlier_fraction: fail below this share of inliers.
        normal_prior: if given, reject hypotheses whose unit normal has
            ``|n . prior| < prior_tolerance``. Used to insist the floor is
            roughly perpendicular to the camera's up axis.
        prior_tolerance: cosine threshold for ``normal_prior``.
        rng: source of randomness, for reproducibility.

    Returns:
        The refined plane, normal oriented so the camera origin is at positive
        signed distance.

    Raises:
        PlaneFitError: if fewer than 3 points are supplied, or no hypothesis
            reaches ``min_inlier_fraction``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got {points.shape}")
    n_points = points.shape[0]
    if n_points < 3:
        raise PlaneFitError(f"need at least 3 points to fit a plane, got {n_points}")

    rng = np.random.default_rng(0) if rng is None else rng
    if normal_prior is not None:
        normal_prior = np.asarray(normal_prior, dtype=np.float64)
        normal_prior = normal_prior / np.linalg.norm(normal_prior)

    best_inliers: np.ndarray | None = None
    best_count = 0

    # Vectorised hypothesis generation: draw all triplets up front.
    triplets = rng.integers(0, n_points, size=(iterations, 3))
    for triplet in triplets:
        a, b, c = points[triplet]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm

        if normal_prior is not None and abs(normal @ normal_prior) < prior_tolerance:
            continue

        offset = -(normal @ a)
        residual = np.abs(points @ normal + offset)
        inliers = residual < tolerance_m
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    fraction = best_count / n_points
    if best_inliers is None or fraction < min_inlier_fraction:
        raise PlaneFitError(
            f"no plane found: best inlier fraction {fraction:.3f} < "
            f"{min_inlier_fraction:.3f} over {n_points} points"
        )

    normal, offset = _refine_plane(points[best_inliers])

    # Orient so the camera centre (origin) sits at positive signed distance.
    if offset < 0:
        normal, offset = -normal, -offset

    return Plane(normal=normal, offset=float(offset), inlier_fraction=float(fraction))


def _refine_plane(inliers: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares plane through the inliers, via the smallest SVD axis."""
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, float(-(normal @ centroid))


__all__ = ["Plane", "PlaneFitError", "fit_plane_ransac"]
