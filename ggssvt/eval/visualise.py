"""Qualitative overlays.

Two figures per specimen, both of which are diagnostics first and dissertation
figures second:

``rig``
    The world axis and the ROI circle projected into every view. If the red axis
    does not sit on the plant stem in all twelve frames, the registration failed
    and every number downstream of it is meaningless. This is the check that
    caught the original subject-detection failure on E011, where the estimator
    had locked onto background structure a metre behind the plant.

``mask``
    The subject segmentation over the colour frame.

Both are written as PNG montages using Pillow, which is already required for
reading the captures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import KINECT_V2, ROI_RADIUS_M, ROI_Z_MAX_M, WORK_DIR
from ..data.dataset import load_specimen
from ..data.io import project
from ..geometry.rig import estimate_rig
from ..geometry.segment import segment_view

AXIS_COLOUR = (230, 40, 40)
CIRCLE_COLOUR = (0, 160, 255)
MASK_COLOUR = (255, 90, 0)


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required to write overlays") from exc
    return Image


def _draw_points(canvas: np.ndarray, uv: np.ndarray, depth: np.ndarray, colour) -> None:
    height, width = canvas.shape[:2]
    visible = (
        (depth > 0.2)
        & (uv[:, 0] >= 1)
        & (uv[:, 0] < width - 1)
        & (uv[:, 1] >= 1)
        & (uv[:, 1] < height - 1)
    )
    for u, v in uv[visible].astype(int):
        canvas[v - 1 : v + 2, u - 1 : u + 2] = colour


def rig_overlay(plant_id: str, *, every: int = 2) -> np.ndarray:
    """Montage of the world axis and ROI circle projected into the views."""
    specimen = load_specimen(plant_id)
    rig = estimate_rig(specimen)

    axis = np.stack(
        [np.zeros(60), np.zeros(60), np.linspace(0, ROI_Z_MAX_M, 60)], axis=-1
    )
    angles = np.linspace(0, 2 * np.pi, 80)
    circle = np.stack(
        [
            ROI_RADIUS_M * 0.6 * np.cos(angles),
            ROI_RADIUS_M * 0.6 * np.sin(angles),
            np.zeros(80),
        ],
        axis=-1,
    )

    tiles = []
    for view in sorted(specimen.views, key=lambda v: v.azimuth_deg)[::every]:
        pose = rig.pose(view.position_id)
        canvas = (view.load_rgb() * 255).astype(np.uint8).copy()
        for points, colour in ((circle, CIRCLE_COLOUR), (axis, AXIS_COLOUR)):
            uv, depth = project(pose.to_camera(points), KINECT_V2)
            _draw_points(canvas, uv, depth, colour)
        tiles.append(canvas)

    return np.concatenate(tiles, axis=1)


def mask_overlay(plant_id: str, *, every: int = 2) -> np.ndarray:
    """Montage of the subject segmentation over the colour frames."""
    specimen = load_specimen(plant_id)
    rig = estimate_rig(specimen)

    tiles = []
    for view in sorted(specimen.views, key=lambda v: v.azimuth_deg)[::every]:
        pose = rig.pose(view.position_id)
        canvas = (view.load_rgb() * 255).astype(np.uint8).copy()
        segmentation = segment_view(view.load_depth(), pose)
        canvas[segmentation.mask] = MASK_COLOUR
        tiles.append(canvas)

    return np.concatenate(tiles, axis=1)


def write_overlays(
    plant_id: str, *, out_dir: Path = WORK_DIR / "reports" / "overlays", every: int = 2
) -> list[Path]:
    """Write both overlays for one specimen."""
    Image = _require_pil()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, builder in (("rig", rig_overlay), ("mask", mask_overlay)):
        path = out_dir / f"{plant_id}_{name}.png"
        Image.fromarray(builder(plant_id, every=every)).save(path)
        written.append(path)
    return written


__all__ = ["mask_overlay", "rig_overlay", "write_overlays"]
