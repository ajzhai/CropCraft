"""Image loading for the dual-Kinect captures.

Depth is stored as 16-bit single-channel PNG in millimetres, registered to the
colour frame, so RGB and depth share the 512x424 grid and the same intrinsics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import DEPTH_MAX_M, DEPTH_MIN_M, DEPTH_SCALE_M, KINECT_V2, Intrinsics

try:  # Pillow is the only image dependency and is optional at import time.
    from PIL import Image
except ImportError as exc:  # pragma: no cover - exercised only without Pillow
    Image = None
    _PIL_ERROR = exc
else:
    _PIL_ERROR = None


def _require_pil() -> None:
    if Image is None:  # pragma: no cover
        raise ImportError(
            "Pillow is required to read the capture PNGs. Install it with "
            "`pip install pillow`."
        ) from _PIL_ERROR


def load_rgb(path: str | Path) -> np.ndarray:
    """Load a registered colour frame as float32 RGB in [0, 1].

    Returns:
        ``(H, W, 3)`` float32.
    """
    _require_pil()
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return arr / 255.0


def load_depth(
    path: str | Path,
    *,
    min_m: float = DEPTH_MIN_M,
    max_m: float = DEPTH_MAX_M,
) -> np.ndarray:
    """Load a registered depth frame as float32 metres, invalid pixels 0.

    Pixels outside ``[min_m, max_m]`` are zeroed. The Kinect v2 writes 0 for
    "no return" already; this additionally discards the sub-half-metre noise
    floor and the room background beyond the working volume.

    Returns:
        ``(H, W)`` float32 in metres.
    """
    _require_pil()
    with Image.open(path) as img:
        raw = np.asarray(img)

    if raw.ndim != 2:
        raise ValueError(f"{path}: expected single-channel depth, got shape {raw.shape}")

    depth = raw.astype(np.float32) * DEPTH_SCALE_M
    invalid = (depth < min_m) | (depth > max_m)
    depth[invalid] = 0.0
    return depth


def depth_validity(depth_m: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels carrying a usable range measurement."""
    return depth_m > 0.0


def backproject(
    depth_m: np.ndarray,
    intrinsics: Intrinsics = KINECT_V2,
) -> np.ndarray:
    """Back-project a depth image to camera-frame 3D points.

    Uses the standard pinhole model with +x right, +y down, +z forward, which
    is the convention libfreenect2 reports depth in.

    Args:
        depth_m: ``(H, W)`` metres, 0 for invalid.
        intrinsics: camera model to use.

    Returns:
        ``(H, W, 3)`` float32 camera-frame points. Invalid pixels are (0, 0, 0).
    """
    height, width = depth_m.shape
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError(
            f"depth is {width}x{height} but intrinsics describe "
            f"{intrinsics.width}x{intrinsics.height}"
        )

    us = np.arange(width, dtype=np.float32)
    vs = np.arange(height, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(us, vs)

    z = depth_m
    x = (grid_u - intrinsics.cx) * z / intrinsics.fx
    y = (grid_v - intrinsics.cy) * z / intrinsics.fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    points[~depth_validity(depth_m)] = 0.0
    return points


def project(
    points_cam: np.ndarray,
    intrinsics: Intrinsics = KINECT_V2,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to pixel coordinates.

    Args:
        points_cam: ``(N, 3)`` camera-frame points.

    Returns:
        ``(uv, depth)`` where ``uv`` is ``(N, 2)`` float32 pixel coordinates and
        ``depth`` is ``(N,)`` float32 metres along +z. Points at or behind the
        image plane get depth <= 0 and meaningless uv; callers must mask them.
    """
    points_cam = np.asarray(points_cam, dtype=np.float32)
    z = points_cam[:, 2]
    safe_z = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = points_cam[:, 0] * intrinsics.fx / safe_z + intrinsics.cx
    v = points_cam[:, 1] * intrinsics.fy / safe_z + intrinsics.cy
    return np.stack([u, v], axis=-1).astype(np.float32), z.astype(np.float32)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess-green vegetation index, ``2G - R - B`` on [0, 1] channels.

    A cheap, illumination-tolerant foliage cue. Positive on leaves, near zero
    or negative on the concrete floor, black pots and grey rig structure.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (2.0 * g - r - b).astype(np.float32)


__all__ = [
    "backproject",
    "depth_validity",
    "excess_green",
    "load_depth",
    "load_rgb",
    "project",
]
