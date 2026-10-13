"""TSDF fusion: a reconstruction that is allowed to have holes in it.

Space carving intersects silhouette cones, and Laurentini's result says the
visual hull it produces is the *maximal* solid consistent with those silhouettes.
That is a hard ceiling, not a resolution problem. A pot's rim casts no silhouette
from anywhere on a circle around it, so the carve fills it solid; the gap between
two leaves casts no silhouette either, so the carve fills that too. Finer voxels
give a smoother envelope, never a gap. This is why 25 of 36 carved specimens
imply a bulk density one to two orders of magnitude below plant tissue: the
quantity being measured is the canopy envelope rather than the plant.

Depth maps are a different kind of evidence. A depth pixel does not say "the
subject is somewhere along this ray", it says "there is a surface at exactly this
distance". Integrating those measurements as a truncated signed distance field
(Curless and Levoy, 1996) puts the surface at the zero crossing, so concavities
survive, and space that no camera observed simply stays unknown rather than being
filled in.

**Holes are the feature.** A carve answers everywhere, confidently and often
wrongly. A TSDF answers where it was looked at, and the unobserved remainder is
visible as absence. For biomass that is the more honest input: an enclosed volume
computed from observed surfaces is a claim about the plant, whereas one computed
from an envelope is a claim about the room around it.

The sensor sets the useful resolution. Kinect v2 intrinsics put one depth pixel
at ``z / fx``, which is 3.0 mm at the 1.1 m working distance, with axial noise of
roughly 2 to 4 mm. A 6 mm voxel is therefore justified by the measurement, where
the 12 mm carving grid was chosen to keep silhouette carving tractable.

What this does not do is resolve individual leaves. Twelve views leave most leaf
undersides unobserved, and no fusion invents what was never seen. It escapes the
hull's ceiling; it does not replace a dense photogrammetric capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    KINECT_V2,
    Intrinsics,
    voxel_grid_centres,
)

# Six millimetres, from the sensor rather than from convenience: one depth pixel
# spans 3.0 mm at the working distance, so this is two samples per voxel.
FUSION_VOXEL_M = 0.006
FUSION_RESOLUTION = 256

# The truncation band, in multiples of the voxel size. Curless and Levoy's rule
# of thumb is a few voxels; wider closes holes at the cost of rounding off
# exactly the thin structure this is meant to preserve.
TRUNCATION_VOXELS = 3.0


@dataclass(frozen=True)
class FusionResult:
    """A fused field and what it was built from."""

    tsdf: np.ndarray            # (R, R, R) float32 in [-1, 1], 1 where unseen
    weight: np.ndarray          # (R, R, R) float32, views that observed each voxel
    voxel_size_m: float
    n_views: int

    @property
    def observed(self) -> np.ndarray:
        """Voxels any camera actually measured."""
        return self.weight > 0

    @property
    def interior(self) -> np.ndarray:
        """Voxels behind an observed surface.

        The occupancy the rest of the pipeline expects. Negative TSDF means past
        the surface along the viewing ray, which for a closed object is inside
        it. Unobserved voxels are excluded rather than assumed solid, which is
        the whole point of using this instead of a carve.
        """
        return self.observed & (self.tsdf < 0.0)

    def coverage(self) -> float:
        """Fraction of the working volume any camera saw."""
        return float(self.observed.mean())


def fuse(
    depth_m: np.ndarray,
    rotation: np.ndarray,
    centre: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    intrinsics: Intrinsics = KINECT_V2,
    resolution: int = FUSION_RESOLUTION,
    voxel_size_m: float = FUSION_VOXEL_M,
    crop_top: int = 0,
    truncation_voxels: float = TRUNCATION_VOXELS,
) -> FusionResult:
    """Fuse registered depth maps into a truncated signed distance field.

    Args:
        depth_m: ``(V, H, W)`` metres, zero where the sensor returned nothing.
        rotation: ``(V, 3, 3)`` camera-to-world rotations.
        centre: ``(V, 3)`` camera centres in world coordinates.
        mask: ``(V, H, W)`` subject mask. Supplied, only subject pixels
            contribute, which keeps the floor and the background out of the
            field without needing them carved away afterwards.
        crop_top: rows trimmed from the top of the original frame, so principal
            point arithmetic matches the stored images.

    Returns:
        A :class:`FusionResult`.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    n_views, height, width = depth_m.shape
    truncation = truncation_voxels * voxel_size_m

    centres = voxel_grid_centres(resolution=resolution, voxel_size_m=voxel_size_m)
    flat = centres.reshape(-1, 3).astype(np.float32)

    tsdf = np.ones(flat.shape[0], dtype=np.float32)
    weight = np.zeros(flat.shape[0], dtype=np.float32)

    for view in range(n_views):
        # World to camera. points_world does cam @ R.T + centre, so the inverse
        # is (world - centre) @ R.
        cam = (flat - centre[view].astype(np.float32)) @ rotation[view].astype(np.float32)
        z = cam[:, 2]
        in_front = z > 1e-6
        if not in_front.any():
            continue

        u = np.full(z.shape, -1.0, dtype=np.float32)
        v = np.full(z.shape, -1.0, dtype=np.float32)
        u[in_front] = cam[in_front, 0] * intrinsics.fx / z[in_front] + intrinsics.cx
        v[in_front] = cam[in_front, 1] * intrinsics.fy / z[in_front] + intrinsics.cy
        # The stored frames are cropped, so the row index shifts with them.
        v = v - crop_top

        col = np.round(u).astype(np.int32)
        row = np.round(v).astype(np.int32)
        inside = (
            in_front
            & (col >= 0) & (col < width)
            & (row >= 0) & (row < height)
        )
        if not inside.any():
            continue

        sampled = np.zeros(z.shape, dtype=np.float32)
        sampled[inside] = depth_m[view][row[inside], col[inside]]

        valid = inside & (sampled > DEPTH_MIN_M) & (sampled < DEPTH_MAX_M)
        if mask is not None:
            hit = np.zeros(z.shape, dtype=bool)
            hit[inside] = mask[view][row[inside], col[inside]].astype(bool)
            valid &= hit
        if not valid.any():
            continue

        # Signed distance along the ray: positive in front of the surface,
        # negative behind it.
        distance = sampled - z
        # Everything beyond the truncation band behind the surface is out of
        # range for this view and must not be integrated, or the field would
        # claim the whole occluded volume is solid.
        valid &= distance > -truncation

        clipped = np.clip(distance / truncation, -1.0, 1.0).astype(np.float32)

        w = weight[valid]
        tsdf[valid] = (tsdf[valid] * w + clipped[valid]) / (w + 1.0)
        weight[valid] = w + 1.0

    shape = (resolution, resolution, resolution)
    return FusionResult(
        tsdf=tsdf.reshape(shape),
        weight=weight.reshape(shape),
        voxel_size_m=voxel_size_m,
        n_views=n_views,
    )


def fuse_cached(cached, **kwargs) -> FusionResult:
    """Fuse one preprocessed specimen straight from the cache."""
    return fuse(
        cached.depth_m,
        cached.rotation,
        cached.centre,
        mask=cached.mask,
        crop_top=cached.crop_top,
        **kwargs,
    )


__all__ = [
    "FUSION_RESOLUTION",
    "FUSION_VOXEL_M",
    "FusionResult",
    "fuse",
    "fuse_cached",
]
