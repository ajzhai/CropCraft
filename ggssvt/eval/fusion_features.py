"""Shape descriptors read from a TSDF fusion rather than from a carve.

The carved features summarise a visual hull, which for these species is a canopy
envelope: 25 of 36 specimens imply a bulk density one to two orders of magnitude
below plant tissue. These summarise the fused field instead, and the point of
comparison is whether a reconstruction that is allowed to have holes in it says
anything more useful about mass.

Two of the descriptors have no counterpart on the carve side and are worth
naming. ``tsdf_coverage`` is the fraction of the working volume any camera
measured, so it records how much of the answer is observation rather than
absence. ``tsdf_surface_m2`` counts the zero-crossing band, which is a surface
area that follows the leaves rather than wrapping them, and is the quantity the
canopy-area hypothesis wanted and the hull could not supply.

The interior here is a band behind the observed surfaces, not a filled solid, so
its volume is not the plant's volume in any strict sense. Whether it is
nonetheless a better predictor is an empirical question, and the answer is what
this module exists to supply.
"""

from __future__ import annotations

import numpy as np

from ..config import voxel_grid_centres
from ..geometry.fusion import FUSION_RESOLUTION, FUSION_VOXEL_M, fuse_cached

FUSION_KEYS = (
    "tsdf_above_rim_m3",
    "tsdf_surface_m2",
    "tsdf_height_m",
    "tsdf_mean_spread_m",
    "tsdf_max_spread_m",
    "tsdf_coverage",
    "tsdf_observed_m3",
)


def fusion_features(
    cached,
    *,
    resolution: int = FUSION_RESOLUTION,
    voxel_size_m: float = FUSION_VOXEL_M,
) -> dict:
    """Fuse one specimen and summarise the field.

    Returns a plain dict so it can be cached to JSON; fusion costs about fifteen
    seconds a specimen and should not be repeated for every experiment.
    """
    result = fuse_cached(cached, resolution=resolution, voxel_size_m=voxel_size_m)
    centres = voxel_grid_centres(resolution, voxel_size_m)
    heights = centres[..., 2]

    voxel_volume = voxel_size_m ** 3
    interior = result.interior
    above = interior & (heights > cached.pot_height_m)

    # The zero crossing, as the band of observed voxels whose distance to the
    # surface is under half a voxel. Counting them and multiplying by the face
    # area is a coarse surface estimate, but it needs no meshing and it does not
    # depend on the field being closed, which it is not.
    surface = result.observed & (np.abs(result.tsdf) < (0.5 / 3.0))
    surface_area = float(surface.sum()) * voxel_size_m ** 2

    points = centres[above]
    if points.size:
        radial = np.linalg.norm(points[:, :2], axis=1)
        height = float(points[:, 2].max())
        mean_spread = float(radial.mean())
        max_spread = float(radial.max())
    else:
        height = mean_spread = max_spread = 0.0

    return {
        "tsdf_above_rim_m3": float(above.sum()) * voxel_volume,
        "tsdf_surface_m2": surface_area,
        "tsdf_height_m": height,
        "tsdf_mean_spread_m": mean_spread,
        "tsdf_max_spread_m": max_spread,
        "tsdf_coverage": result.coverage(),
        "tsdf_observed_m3": float(result.observed.sum()) * voxel_volume,
    }


def fusion_vector(features: dict) -> np.ndarray:
    """The descriptors as a feature vector, in a fixed order."""
    return np.array([features[k] for k in FUSION_KEYS], dtype=np.float64)


def write_fused_cache(
    plant_ids: list[str],
    source_cache,
    target_cache,
    *,
    verbose: bool = True,
) -> int:
    """Write a cache whose occupancy came from fusion instead of carving.

    Deliberately at the carve's own 128^3 and 12 mm rather than the 6 mm the
    sensor supports. Two reasons. It drops into every existing tool without a
    special case, and more importantly it isolates the variable: at identical
    resolution the only thing that differs between this cache and the geometric
    one is silhouette intersection versus depth integration. That control is what
    turns "fusion is better" into "the method is what did it", and it holds:
    8 of 36 plausible carved against 21 of 36 fused, both at 12 mm, rising to 31
    only when the finer grid is added on top.

    Everything except the occupancy is copied from the source, so the two caches
    describe the same views, poses and masks.
    """
    import shutil

    import numpy as np

    from ..config import VOXEL_RESOLUTION, VOXEL_SIZE_M
    from ..data.preprocess import cache_path, load_cached
    from ..geometry.fusion import fuse_cached

    target_cache.mkdir(parents=True, exist_ok=True)
    written = 0

    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, source_cache)
        fused = fuse_cached(
            cached, resolution=VOXEL_RESOLUTION, voxel_size_m=VOXEL_SIZE_M
        )
        occupancy = fused.interior

        source = cache_path(plant_id, source_cache)
        with np.load(source, allow_pickle=False) as data:
            fields = dict(data)
        fields["occupancy"] = np.packbits(occupancy, axis=None)
        fields["occupancy_shape"] = np.array(occupancy.shape)
        # How many views observed each voxel, which is the fused counterpart of
        # the carve's informative-view count and keeps the field meaningful.
        fields["n_informative"] = np.clip(fused.weight, 0, 127).astype(np.int8)
        fields["segmenter"] = "tsdf"

        np.savez_compressed(cache_path(plant_id, target_cache), **fields)
        written += 1
        if verbose:
            print(
                f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
                f"{occupancy.sum() / 1e3:6.1f}k voxels  "
                f"coverage {fused.coverage():.3f}"
            )

    quality = source_cache / "quality.json"
    if quality.exists():
        # The gate measured segmentation and registration, neither of which the
        # fusion changed, so the same report applies and copying it keeps
        # usable_plant_ids working against this cache.
        shutil.copyfile(quality, target_cache / "quality.json")

    return written


__all__ = [
    "FUSION_KEYS",
    "fusion_features",
    "fusion_vector",
    "write_fused_cache",
]
