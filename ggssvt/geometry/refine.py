"""Refining the hand-placed rig positions.

The azimuth of each view comes from its filename, which records the step the
operator was asked to move to -- not where the camera actually ended up. The
cameras are carried by hand with no floor marks or indexed turntable, and
``dataset/README.md`` says as much: it warns that without a physical position
guide a single per-day calibration should not be trusted across a day's plants.

At a working radius of about one metre, a five degree placement error displaces
a view by nearly nine centimetres. That is far larger than the plant's own
structures, and it is what makes a naive carve delete most of the canopy: each
view's surfaces land where its neighbours see empty space.

This module recovers the residual pose error directly from the point clouds. For
each view it searches a small range of azimuth corrections and lateral offsets,
scoring each candidate by how well that view's points fall onto the surfaces the
*other* views already agree on. Coordinate descent over the views converges in a
handful of passes.

The search is deliberately local. It corrects hand-placement error of a few
degrees and a few centimetres; it cannot rescue a view whose nominal azimuth is
wrong by a whole step, and it does not try to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import ROI_RADIUS_M, ROI_Z_MAX_M

REFINE_VOXEL_M = 0.02
REFINE_Z_MIN_M = 0.05
REFINE_MAX_POINTS = 6000       # per view, subsampled for the search
DEFAULT_AZIMUTH_RANGE_DEG = 8.0
DEFAULT_AZIMUTH_STEP_DEG = 1.0
DEFAULT_OFFSET_RANGE_M = 0.06
DEFAULT_OFFSET_STEP_M = 0.02
DEFAULT_PASSES = 3


@dataclass(frozen=True)
class ViewCorrection:
    """Residual pose correction for one view, in the levelled frame."""

    d_azimuth_deg: float = 0.0
    dx_m: float = 0.0
    dy_m: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.d_azimuth_deg == 0.0 and self.dx_m == 0.0 and self.dy_m == 0.0


@dataclass
class RefinementResult:
    """Outcome of the coordinate-descent refinement."""

    corrections: dict[str, ViewCorrection]
    score_before: float
    score_after: float
    passes_run: int
    per_pass_scores: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        return self.score_after - self.score_before


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _transform(
    points_level: np.ndarray,
    axis_xy: np.ndarray,
    azimuth_deg: float,
    correction: ViewCorrection,
) -> np.ndarray:
    """Apply a corrected axis and azimuth to one view's levelled points."""
    shifted = points_level.copy()
    shifted[:, 0] -= axis_xy[0] + correction.dx_m
    shifted[:, 1] -= axis_xy[1] + correction.dy_m
    yaw = math.radians(azimuth_deg + correction.d_azimuth_deg) + math.pi / 2.0
    return shifted @ _rotation_z(yaw).T


class _ReferenceGrid:
    """Boolean occupancy over the working cylinder, for fast membership tests."""

    def __init__(self, voxel_m: float = REFINE_VOXEL_M):
        self.voxel_m = voxel_m
        self.half = ROI_RADIUS_M
        self.nx = int(np.ceil(2 * self.half / voxel_m)) + 1
        self.nz = int(np.ceil(ROI_Z_MAX_M / voxel_m)) + 1
        self.grid = np.zeros((self.nx, self.nx, self.nz), dtype=bool)

    def _indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((points[:, 0] + self.half) / self.voxel_m).astype(np.int64)
        iy = np.floor((points[:, 1] + self.half) / self.voxel_m).astype(np.int64)
        iz = np.floor(points[:, 2] / self.voxel_m).astype(np.int64)
        inside = (
            (ix >= 0)
            & (ix < self.nx)
            & (iy >= 0)
            & (iy < self.nx)
            & (iz >= 0)
            & (iz < self.nz)
        )
        flat = np.stack([ix, iy, iz], axis=-1)
        return flat, inside

    def add(self, points: np.ndarray) -> None:
        flat, inside = self._indices(points)
        kept = flat[inside]
        if kept.size:
            self.grid[kept[:, 0], kept[:, 1], kept[:, 2]] = True

    def hit_fraction(self, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        flat, inside = self._indices(points)
        hits = np.zeros(points.shape[0], dtype=bool)
        kept = flat[inside]
        if kept.size:
            hits[inside] = self.grid[kept[:, 0], kept[:, 1], kept[:, 2]]
        return float(hits.mean())

    def dilated(self, iterations: int = 1) -> _ReferenceGrid:
        """A dilated copy, so a near miss still counts as a hit."""
        grown = self.grid.copy()
        for _ in range(max(0, iterations)):
            step = grown.copy()
            for axis in (0, 1, 2):
                step |= np.roll(grown, 1, axis=axis)
                step |= np.roll(grown, -1, axis=axis)
            grown = step
        clone = _ReferenceGrid(self.voxel_m)
        clone.grid = grown
        return clone


class _MultiScaleProbe:
    """Scores alignment at several tolerances at once.

    A single dilated occupancy test is degenerate as an objective: a view can
    raise its hit fraction simply by rotating its points into the densest part
    of the scene, whether or not they belong there. Combining an exact test with
    progressively looser ones removes that shortcut -- gross misalignment is
    still penalised by the loose terms, but only genuine alignment scores on the
    exact one, and the exact term carries the most weight.
    """

    WEIGHTS = (0.5, 0.3, 0.2)

    def __init__(self, reference: _ReferenceGrid):
        self.levels = [
            reference,
            reference.dilated(1),
            reference.dilated(2),
        ]

    def score(self, points: np.ndarray) -> float:
        return float(
            sum(
                weight * level.hit_fraction(points)
                for weight, level in zip(self.WEIGHTS, self.levels)
            )
        )


def _prepare(points_level: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Trim a view's levelled cloud to the working cylinder and subsample it."""
    radial = np.linalg.norm(points_level[:, :2], axis=1)
    # Generous radius here: the axis correction can move points in or out, and
    # the cylinder is applied about the uncorrected axis at this stage.
    keep = (radial < ROI_RADIUS_M * 2.0) & (points_level[:, 2] > REFINE_Z_MIN_M)
    subset = points_level[keep]
    if subset.shape[0] > REFINE_MAX_POINTS:
        index = rng.choice(subset.shape[0], REFINE_MAX_POINTS, replace=False)
        subset = subset[index]
    return subset


def refine_registration(
    points_level: dict[str, np.ndarray],
    azimuths: dict[str, float],
    axes: dict[str, np.ndarray],
    *,
    azimuth_range_deg: float = DEFAULT_AZIMUTH_RANGE_DEG,
    azimuth_step_deg: float = DEFAULT_AZIMUTH_STEP_DEG,
    offset_range_m: float = DEFAULT_OFFSET_RANGE_M,
    offset_step_m: float = DEFAULT_OFFSET_STEP_M,
    passes: int = DEFAULT_PASSES,
    seed: int = 0,
) -> RefinementResult:
    """Correct each view's azimuth and lateral offset by coordinate descent.

    Args:
        points_level: per position id, the ``(N, 3)`` levelled point cloud with
            ``z`` measured from the floor.
        azimuths: per position id, the nominal azimuth in degrees.
        axes: per position id, the ``(2,)`` estimated subject axis.
        azimuth_range_deg: half-width of the azimuth search.
        offset_range_m: half-width of the lateral offset search, both axes.
        passes: coordinate-descent sweeps over the views.

    Returns:
        The per-view corrections and the agreement score before and after.
    """
    rng = np.random.default_rng(seed)
    clouds = {pid: _prepare(points, rng) for pid, points in points_level.items()}
    positions = [pid for pid, cloud in clouds.items() if cloud.shape[0] > 0]

    corrections: dict[str, ViewCorrection] = {
        pid: ViewCorrection() for pid in points_level
    }
    if len(positions) < 3:
        return RefinementResult(corrections, 0.0, 0.0, 0, [])

    azimuth_deltas = np.arange(
        -azimuth_range_deg, azimuth_range_deg + 1e-9, azimuth_step_deg
    )
    offsets = np.arange(-offset_range_m, offset_range_m + 1e-9, offset_step_m)

    def total_score(current: dict[str, ViewCorrection]) -> float:
        scores = []
        for target in positions:
            reference = _ReferenceGrid()
            for other in positions:
                if other == target:
                    continue
                reference.add(
                    _transform(clouds[other], axes[other], azimuths[other], current[other])
                )
            scores.append(
                _MultiScaleProbe(reference).score(
                    _transform(clouds[target], axes[target], azimuths[target], current[target])
                )
            )
        return float(np.mean(scores))

    score_before = total_score(corrections)
    per_pass: list[float] = []

    for pass_index in range(passes):
        for target in positions:
            reference = _ReferenceGrid()
            for other in positions:
                if other == target:
                    continue
                reference.add(
                    _transform(clouds[other], axes[other], azimuths[other], corrections[other])
                )
            probe = _MultiScaleProbe(reference)

            best = corrections[target]
            best_score = probe.score(
                _transform(clouds[target], axes[target], azimuths[target], best)
            )

            # Azimuth first: it dominates the displacement at this radius.
            for delta in azimuth_deltas:
                candidate = ViewCorrection(float(delta), best.dx_m, best.dy_m)
                score = probe.score(
                    _transform(clouds[target], axes[target], azimuths[target], candidate)
                )
                if score > best_score:
                    best, best_score = candidate, score

            # Then the lateral offset, on the winning azimuth.
            if pass_index > 0:
                for dx in offsets:
                    for dy in offsets:
                        candidate = ViewCorrection(best.d_azimuth_deg, float(dx), float(dy))
                        score = probe.score(
                            _transform(
                                clouds[target], axes[target], azimuths[target], candidate
                            )
                        )
                        if score > best_score:
                            best, best_score = candidate, score

            corrections[target] = best

        per_pass.append(total_score(corrections))

    return RefinementResult(
        corrections=corrections,
        score_before=score_before,
        score_after=per_pass[-1] if per_pass else score_before,
        passes_run=passes,
        per_pass_scores=per_pass,
    )


__all__ = [
    "RefinementResult",
    "ViewCorrection",
    "refine_registration",
]
