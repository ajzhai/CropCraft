"""Subject segmentation in the registered world frame.

Once a view is registered (:mod:`ggssvt.geometry.rig`), separating the specimen
from the greenhouse is mostly a geometric question: the plant is what stands
inside a cylinder centred on the world origin. That alone removes the floor,
the far wall, the rig structure and the neighbouring clutter, without any
learned segmentation model.

The excess-green refinement is available on top for the foliage-only mask, but
it is off by default -- the eucalyptus specimens have thin woody stems that
score poorly on any greenness index, and dropping them would bias the carved
volume against exactly the structures the reconstruction is meant to recover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import (
    EXCESS_GREEN_THRESHOLD,
    KINECT_V2,
    POT_HEIGHT_M,
    ROI_RADIUS_M,
    ROI_Z_MAX_M,
    ROI_Z_MIN_M,
    Intrinsics,
)
from ..data.io import backproject, depth_validity, excess_green
from .rig import ViewPose


@dataclass(frozen=True)
class ViewSegmentation:
    """The segmented subject in one view."""

    position_id: str
    mask: np.ndarray            # (H, W) bool, True on the subject
    depth_m: np.ndarray         # (H, W) float32, 0 where invalid
    points_world: np.ndarray    # (M, 3) float32, subject points only
    colours: np.ndarray | None  # (M, 3) float32 in [0, 1], or None

    @property
    def n_points(self) -> int:
        return int(self.points_world.shape[0])

    @property
    def height_m(self) -> float:
        """Highest subject point above the floor."""
        if self.n_points == 0:
            return 0.0
        return float(self.points_world[:, 2].max())


def segment_view(
    depth_m: np.ndarray,
    pose: ViewPose,
    *,
    rgb: np.ndarray | None = None,
    intrinsics: Intrinsics = KINECT_V2,
    radius_m: float = ROI_RADIUS_M,
    z_min_m: float = ROI_Z_MIN_M,
    z_max_m: float = ROI_Z_MAX_M,
    use_excess_green: bool = False,
    exg_threshold: float = EXCESS_GREEN_THRESHOLD,
) -> ViewSegmentation:
    """Segment the subject from one registered view.

    Args:
        depth_m: ``(H, W)`` metres, 0 for invalid.
        pose: the view's estimated extrinsics.
        rgb: ``(H, W, 3)`` in [0, 1]; required if ``use_excess_green``.
        radius_m: cylinder radius about the world z axis.
        z_min_m, z_max_m: height band above the floor to keep.
        use_excess_green: additionally require a positive vegetation index.
            Off by default; see the module docstring.

    Returns:
        The segmentation, with subject points already in world coordinates.
    """
    points_cam = backproject(depth_m, intrinsics)
    valid = depth_validity(depth_m)

    flat_cam = points_cam.reshape(-1, 3).astype(np.float64)
    flat_world = flat_cam @ pose.rotation.T + pose.centre

    radial = np.linalg.norm(flat_world[:, :2], axis=1)
    height = flat_world[:, 2]

    inside = (radial < radius_m) & (height > z_min_m) & (height < z_max_m)
    mask = inside.reshape(depth_m.shape) & valid

    if use_excess_green:
        if rgb is None:
            raise ValueError("use_excess_green requires the rgb frame")
        foliage = excess_green(rgb) > exg_threshold
        # Keep anything below the pot rim regardless, so the stem base and pot
        # stay in the silhouette and the carved hull remains closed at the bottom.
        low = (flat_world[:, 2] < POT_HEIGHT_M).reshape(depth_m.shape)
        mask = mask & (foliage | low)

    points_world = flat_world.reshape(*depth_m.shape, 3)[mask].astype(np.float32)
    colours = rgb[mask].astype(np.float32) if rgb is not None else None

    return ViewSegmentation(
        position_id=pose.position_id,
        mask=mask,
        depth_m=depth_m,
        points_world=points_world,
        colours=colours,
    )


def segment_specimen(
    specimen,
    rig,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    load_colour: bool = False,
    **kwargs,
) -> dict[str, ViewSegmentation]:
    """Segment every view of a specimen. Keys are position ids."""
    segmentations: dict[str, ViewSegmentation] = {}
    for view in specimen.views:
        pose = rig.pose(view.position_id)
        segmentations[view.position_id] = segment_view(
            view.load_depth(),
            pose,
            rgb=view.load_rgb() if load_colour or kwargs.get("use_excess_green") else None,
            intrinsics=intrinsics,
            **kwargs,
        )
    return segmentations


def fused_point_cloud(
    segmentations: dict[str, ViewSegmentation],
    *,
    with_colour: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate every view's subject points into one world-frame cloud.

    This is the multi-view fusion the classical baseline reconstructs from, and
    the reference the rig refinement scores itself against.
    """
    clouds = [s.points_world for s in segmentations.values() if s.n_points]
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32), None
    points = np.concatenate(clouds, axis=0)

    colours = None
    if with_colour:
        colour_chunks = [
            s.colours for s in segmentations.values() if s.n_points and s.colours is not None
        ]
        if len(colour_chunks) == len(clouds):
            colours = np.concatenate(colour_chunks, axis=0)

    return points, colours


def multiview_agreement(
    segmentations: dict[str, ViewSegmentation],
    *,
    voxel_m: float = 0.02,
) -> float:
    """Fraction of occupied voxels that more than one view agrees on.

    A registration quality score that needs no ground truth: when the views are
    correctly aligned their point clouds land on the same surfaces, so most
    occupied voxels are seen by several cameras. Misalignment smears each view
    into its own shell and the score collapses.
    """
    per_view_keys: list[set[tuple[int, int, int]]] = []
    for segmentation in segmentations.values():
        if segmentation.n_points == 0:
            continue
        keys = np.floor(segmentation.points_world / voxel_m).astype(np.int32)
        per_view_keys.append({tuple(k) for k in keys})

    if len(per_view_keys) < 2:
        return 0.0

    counts: dict[tuple[int, int, int], int] = {}
    for keys in per_view_keys:
        for key in keys:
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        return 0.0
    shared = sum(1 for c in counts.values() if c > 1)
    return shared / len(counts)


__all__ = [
    "ViewSegmentation",
    "fused_point_cloud",
    "multiview_agreement",
    "segment_specimen",
    "segment_view",
]
