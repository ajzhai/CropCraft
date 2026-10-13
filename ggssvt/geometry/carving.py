"""Space carving: depth and silhouettes to a voxel occupancy field.

This produces the self-supervision target for GG-SSVT. No occupancy labels
exist for these specimens and none can be obtained without destructive
sectioning, so the supervision has to come from the geometry of the capture
itself -- which is exactly what space carving provides.

The carve is stronger than a pure visual hull because the Kinect gives metric
depth as well as a silhouette. Each view can rule a voxel out two ways:

* **Silhouette carving.** The voxel projects outside the subject mask, so the
  ray missed the plant entirely.
* **Depth carving.** The voxel projects onto the subject but sits measurably in
  front of the observed surface, so it lies in free space.

A voxel that no view rules out, and that enough views actively support, is
occupied. Voxels behind the observed surface are unobservable rather than free,
so they are counted as "no information" and never used to carve -- the standard
visual-hull caveat (Laurentini 1994) still bounds what this can recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    CARVE_DEPTH_MARGIN_M,
    CARVE_DEPTH_MARGIN_SLOPE,
    CARVE_MASK_DILATION,
    KINECT_V2,
    POT_HEIGHT_M,
    VOXEL_RESOLUTION,
    VOXEL_SIZE_M,
    Intrinsics,
    voxel_grid_centres,
)
from .rig import RigSolution
from .segment import ViewSegmentation


@dataclass
class OccupancyVolume:
    """A carved occupancy field on the world-frame voxel grid."""

    plant_id: str
    occupancy: np.ndarray       # (R, R, R) bool
    support: np.ndarray         # (R, R, R) float32, fraction of views supporting
    n_informative: np.ndarray   # (R, R, R) int16, views that had an opinion
    resolution: int
    voxel_size_m: float

    @property
    def voxel_volume_m3(self) -> float:
        return self.voxel_size_m ** 3

    @property
    def volume_m3(self) -> float:
        """Total occupied volume."""
        return float(self.occupancy.sum()) * self.voxel_volume_m3

    def above_ground_volume_m3(self, pot_height_m: float = POT_HEIGHT_M) -> float:
        """Occupied volume above the pot rim -- the above-ground plant material."""
        z_index = int(np.ceil(pot_height_m / self.voxel_size_m))
        z_index = min(max(z_index, 0), self.resolution)
        return float(self.occupancy[:, :, z_index:].sum()) * self.voxel_volume_m3

    @property
    def height_m(self) -> float:
        """Height of the highest occupied voxel above the floor."""
        occupied_z = np.nonzero(self.occupancy.any(axis=(0, 1)))[0]
        if occupied_z.size == 0:
            return 0.0
        return float((occupied_z.max() + 1) * self.voxel_size_m)

    def centres(self) -> np.ndarray:
        """World-frame centres of every voxel, ``(R, R, R, 3)``."""
        return voxel_grid_centres()

    def occupied_points(self) -> np.ndarray:
        """World-frame centres of the occupied voxels, ``(N, 3)``."""
        return self.centres()[self.occupancy]

    def summary(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "resolution": self.resolution,
            "voxel_size_m": self.voxel_size_m,
            "n_occupied": int(self.occupancy.sum()),
            "volume_m3": self.volume_m3,
            "above_ground_volume_m3": self.above_ground_volume_m3(),
            "height_m": self.height_m,
            "mean_informative_views": float(self.n_informative.mean()),
        }


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Binary dilation with a 3x3 structuring element, without SciPy.

    Thin stems are one to three pixels wide at these ranges, so a voxel on a
    branch can project just off the silhouette and be carved away by rounding
    alone. Widening the mask by a pixel costs a little hull tightness and saves
    the structures the reconstruction exists to recover.
    """
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                grown |= padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = grown
    return out


def carve(
    rig: RigSolution,
    segmentations: dict[str, ViewSegmentation],
    *,
    plant_id: str = "",
    intrinsics: Intrinsics = KINECT_V2,
    resolution: int = VOXEL_RESOLUTION,
    voxel_size_m: float = VOXEL_SIZE_M,
    depth_margin_m: float = CARVE_DEPTH_MARGIN_M,
    depth_margin_slope: float = CARVE_DEPTH_MARGIN_SLOPE,
    max_carve_votes: int | None = None,
    min_informative_views: int | None = None,
    mask_dilation: int = CARVE_MASK_DILATION,
    chunk: int = 1 << 20,
) -> OccupancyVolume:
    """Carve an occupancy volume from registered, segmented views.

    A voxel survives when few enough views actively rule it out and enough views
    could see it at all::

        occupied = (carve_votes <= max_carve_votes) and
                   (informative_views >= min_informative_views)

    Counting dissent rather than requiring near-unanimity matters here: a leaf
    or a stem is one or two pixels wide, so a single view whose depth return
    slipped between two leaves would otherwise delete a structure that eleven
    other views agree on. The ``min_informative_views`` floor does the opposite
    job -- voxels above the top of frame are seen by nobody and would otherwise
    survive unopposed.

    Args:
        rig: estimated extrinsics for the specimen.
        segmentations: per-position subject masks and depth.
        depth_margin_m: constant part of the free-space tolerance.
        depth_margin_slope: quadratic term, ``margin = base + slope * z^2``.
            Kinect v2 range noise grows with the square of distance.
        max_carve_votes: how many views may rule a voxel out before it goes.
            Defaults to a quarter of the view count.
        min_informative_views: views that must have an opinion at all. Defaults
            to half the view count.
        mask_dilation: pixels to widen each subject mask by before carving.
        chunk: voxels processed per batch, to bound peak memory.

    Returns:
        The carved :class:`OccupancyVolume`.
    """
    centres = voxel_grid_centres().reshape(-1, 3)
    n_voxels = centres.shape[0]

    carve_votes = np.zeros(n_voxels, dtype=np.int16)
    informative = np.zeros(n_voxels, dtype=np.int16)

    positions = [p for p in segmentations if p in rig.poses]

    # Both thresholds scale with the number of views. The tuned values (6 and 3)
    # were chosen against the full 12-view sweep, and holding them fixed makes a
    # four-view carve return an empty volume rather than a poor one: no voxel can
    # possibly have six informative views when only four exist. Deriving them
    # keeps a view-count ablation measuring the reconstruction rather than a
    # constant left over from a different protocol.
    n_views = max(1, len(positions))
    if min_informative_views is None:
        min_informative_views = max(2, round(n_views / 2))
    if max_carve_votes is None:
        max_carve_votes = max(1, round(n_views / 4))
    masks = {
        position_id: dilate(segmentations[position_id].mask, mask_dilation)
        for position_id in positions
    }

    for start in range(0, n_voxels, chunk):
        stop = min(start + chunk, n_voxels)
        block = centres[start:stop]

        for position_id in positions:
            pose = rig.pose(position_id)
            segmentation = segmentations[position_id]

            cam = (block - pose.centre) @ pose.rotation
            z = cam[:, 2]
            in_front = z > 1e-3

            u = np.full(z.shape, -1.0)
            v = np.full(z.shape, -1.0)
            safe_z = np.where(in_front, z, 1.0)
            u[in_front] = (
                cam[in_front, 0] * intrinsics.fx / safe_z[in_front] + intrinsics.cx
            )
            v[in_front] = (
                cam[in_front, 1] * intrinsics.fy / safe_z[in_front] + intrinsics.cy
            )

            ui = np.round(u).astype(np.int32)
            vi = np.round(v).astype(np.int32)
            on_image = (
                in_front
                & (ui >= 0)
                & (ui < intrinsics.width)
                & (vi >= 0)
                & (vi < intrinsics.height)
            )
            if not on_image.any():
                continue

            rows = vi[on_image]
            cols = ui[on_image]
            measured = segmentation.depth_m[rows, cols]
            subject = masks[position_id][rows, cols]
            has_depth = measured > 0.0
            voxel_z = z[on_image]

            # A view is informative about a voxel when the ray either missed the
            # subject (background, so free space) or returned a range we can
            # compare against. A subject pixel with no depth return tells us
            # nothing.
            background = ~subject
            informative_here = background | has_depth

            # Free space: in front of the observed surface by more than the noise
            # margin. Behind the surface is occluded, not empty, so it is never
            # evidence of emptiness.
            margin = depth_margin_m + depth_margin_slope * voxel_z * voxel_z
            free = background | (has_depth & (voxel_z < measured - margin))

            indices = np.nonzero(on_image)[0] + start
            np.add.at(informative, indices, informative_here.astype(np.int16))
            np.add.at(carve_votes, indices, (informative_here & free).astype(np.int16))

    occupancy = (informative >= min_informative_views) & (carve_votes <= max_carve_votes)

    with np.errstate(invalid="ignore", divide="ignore"):
        support = np.where(
            informative > 0, 1.0 - carve_votes / np.maximum(informative, 1), 0.0
        )

    shape = (resolution, resolution, resolution)
    return OccupancyVolume(
        plant_id=plant_id or rig.plant_id,
        occupancy=occupancy.reshape(shape),
        support=support.astype(np.float32).reshape(shape),
        n_informative=informative.reshape(shape),
        resolution=resolution,
        voxel_size_m=voxel_size_m,
    )


def surface_coverage(
    volume: OccupancyVolume,
    segmentations: dict[str, ViewSegmentation],
) -> float:
    """Fraction of observed subject points that land inside an occupied voxel.

    A ground-truth-free check on the carve: every point the Kinect measured on
    the plant is, by construction, on the plant, so a carve that drops them has
    over-carved. Paired with the hull volume it gives the tightness/coverage
    trade-off to tune against -- maximise coverage, then minimise volume.
    """
    from .segment import fused_point_cloud

    points, _ = fused_point_cloud(segmentations)
    if points.shape[0] == 0:
        return 0.0

    half = volume.resolution * volume.voxel_size_m / 2.0
    ix = np.floor((points[:, 0] + half) / volume.voxel_size_m).astype(np.int64)
    iy = np.floor((points[:, 1] + half) / volume.voxel_size_m).astype(np.int64)
    iz = np.floor(points[:, 2] / volume.voxel_size_m).astype(np.int64)

    inside = (
        (ix >= 0)
        & (ix < volume.resolution)
        & (iy >= 0)
        & (iy < volume.resolution)
        & (iz >= 0)
        & (iz < volume.resolution)
    )
    if not inside.any():
        return 0.0

    hit = volume.occupancy[ix[inside], iy[inside], iz[inside]]
    return float(hit.mean())


def carve_specimen(
    specimen,
    rig: RigSolution,
    *,
    segmentations: dict[str, ViewSegmentation] | None = None,
    **kwargs,
) -> OccupancyVolume:
    """Segment (if needed) and carve one specimen."""
    if segmentations is None:
        from .segment import segment_specimen

        segmentations = segment_specimen(specimen, rig)
    return carve(rig, segmentations, plant_id=specimen.plant_id, **kwargs)


_NEIGHBOURHOOD_26 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def largest_connected_component(occupancy: np.ndarray) -> np.ndarray:
    """Keep only the largest 26-connected blob.

    Carving leaves isolated specks where a few views disagree. The plant is one
    connected object, so everything not attached to the main body is noise.

    Connectivity is 26-neighbour rather than 6-neighbour on purpose. A stem
    climbing diagonally through the grid touches its successor only at a corner,
    so face-connectivity severs it and amputates the whole canopy above the
    break -- on E002 that alone cost 0.3 m of plant height. Implemented as an
    iterative flood fill to avoid a SciPy dependency.
    """
    occupancy = np.ascontiguousarray(occupancy, dtype=bool)
    if not occupancy.any():
        return occupancy

    labels = np.zeros(occupancy.shape, dtype=np.int32)
    current = 0
    best_label, best_size = 0, 0
    shape = occupancy.shape

    for seed in zip(*np.nonzero(occupancy)):
        if labels[seed]:
            continue
        current += 1
        size = 0
        stack = [seed]
        labels[seed] = current
        while stack:
            x, y, z = stack.pop()
            size += 1
            for dx, dy, dz in _NEIGHBOURHOOD_26:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                    continue
                if occupancy[nx, ny, nz] and labels[nx, ny, nz] == 0:
                    labels[nx, ny, nz] = current
                    stack.append((nx, ny, nz))
        if size > best_size:
            best_label, best_size = current, size

    return labels == best_label


__all__ = [
    "OccupancyVolume",
    "carve",
    "carve_specimen",
    "dilate",
    "largest_connected_component",
    "surface_coverage",
]
