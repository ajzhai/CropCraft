"""Rendering carved reconstructions for inspection.

Three outputs, for three different jobs:

``render_volume`` / ``contact_sheet``
    Orthographic projections of the occupancy grid as PNGs. Fast, dependency-free
    beyond Pillow, and the right thing for spotting a failed reconstruction at a
    glance across thirty specimens.

``export_ply``
    Occupied voxel centres as an ASCII PLY point cloud, for MeshLab, CloudCompare
    or Open3D when you need to measure something rather than look at it.

``volume_payload``
    A compact quantised form for the interactive HTML gallery.

The projections are deliberately *not* shaded renders. A depth-cued orthographic
silhouette from three axes shows what a carved hull actually is -- where it is
solid, where it is hollow, and where the cameras never saw -- which a prettier
render would hide.
"""

from __future__ import annotations

import base64
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VOXEL_SIZE_M, WORK_DIR

AXES = {"front": (0, 2, 1), "side": (1, 2, 0), "top": (0, 1, 2)}


def _require_pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required to render volumes") from exc
    return Image


def occupied_points(occupancy: np.ndarray, voxel_size_m: float = VOXEL_SIZE_M) -> np.ndarray:
    """World-frame centres of the occupied voxels, ``(N, 3)`` metres."""
    resolution = occupancy.shape[0]
    half = resolution * voxel_size_m / 2.0
    index = np.array(np.nonzero(occupancy)).T.astype(np.float64)
    return np.stack(
        [
            index[:, 0] * voxel_size_m - half + voxel_size_m / 2,
            index[:, 1] * voxel_size_m - half + voxel_size_m / 2,
            index[:, 2] * voxel_size_m + voxel_size_m / 2,
        ],
        axis=-1,
    )


# Viridis, as ten anchors interpolated on demand. The same ramp the project page
# uses, so a specimen looks the same in the gallery as in the interactive viewer.
# Perceptually uniform and monotonic in lightness, which the green tint this
# replaced was not: mid-depth voxels there were indistinguishable from near ones
# in greyscale or to a red-green colour-blind reader.
VIRIDIS = np.array(
    [
        (68, 1, 84), (72, 40, 120), (62, 73, 137), (49, 104, 142),
        (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
        (181, 222, 43), (253, 231, 37),
    ],
    dtype=np.float64,
)


def viridis(t: np.ndarray) -> np.ndarray:
    """Sample the ramp at ``t`` in [0, 1]. Returns ``(n, 3)`` uint8."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    x = t * (len(VIRIDIS) - 1)
    lo = np.clip(np.floor(x).astype(int), 0, len(VIRIDIS) - 2)
    frac = (x - lo)[:, None]
    return np.round(
        VIRIDIS[lo] * (1.0 - frac) + VIRIDIS[lo + 1] * frac
    ).astype(np.uint8)


def _box_downscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Average each ``factor`` by ``factor`` block down to one pixel.

    This is the whole antialiasing strategy: render at a multiple of the target
    size, then average down. A voxel projection is all hard edges, so at 1x every
    boundary is a staircase; averaging turns each staircase into a gradient at no
    cost beyond the larger intermediate buffer.
    """
    if factor <= 1:
        return image
    height, width = image.shape[0] // factor, image.shape[1] // factor
    trimmed = image[: height * factor, : width * factor].astype(np.float64)
    blocks = trimmed.reshape(height, factor, width, factor, 3)
    return blocks.mean(axis=(1, 3)).round().astype(np.uint8)


def render_volume(
    occupancy: np.ndarray,
    *,
    view: str = "front",
    size: int = 220,
    background: tuple[int, int, int] = (255, 255, 255),
    point_radius: int = 1,
    supersample: int = 3,
) -> np.ndarray:
    """Depth-cued orthographic projection of an occupancy grid.

    Nearer voxels take the bright end of the viridis ramp, so the projection
    reads as a solid shape rather than a flat silhouette while still showing the
    whole extent.

    Args:
        size: side of the returned image in pixels.
        point_radius: half-width of the square drawn per voxel, in *output*
            pixels. 1 gives a 2 by 2 dot. Raise it for a sparse cloud that reads
            as speckle, lower it for a dense one that reads as a solid blob.
        supersample: render at this multiple and average down. 1 disables it and
            restores the original aliased output; 3 is a good default and costs
            about nine times the intermediate memory, which at these sizes is
            still under a megabyte.

    Returns:
        ``(size, size, 3)`` uint8.
    """
    if view not in AXES:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(AXES)}")

    scale = max(1, int(supersample))
    inner = size * scale
    horizontal, vertical, depth_axis = AXES[view]
    canvas = np.full((inner, inner, 3), background, dtype=np.uint8)

    if not occupancy.any():
        return _box_downscale(canvas, scale)

    index = np.array(np.nonzero(occupancy)).T
    resolution = occupancy.shape[0]

    u = (index[:, horizontal] / resolution * (inner - 1)).astype(np.int32)
    v = index[:, vertical] / resolution * (inner - 1)
    # Image rows grow downward; world z grows upward.
    v = ((inner - 1) - v).astype(np.int32)
    depth = index[:, depth_axis].astype(np.float64)

    # Painter's algorithm: draw far voxels first so near ones overwrite them.
    order = np.argsort(-depth) if view != "top" else np.argsort(depth)
    u, v, depth = u[order], v[order], depth[order]

    span = max(1.0, depth.max() - depth.min())
    # Near voxels take the bright end of the ramp. The top view reverses, because
    # there "near" is the top of the plant rather than the side facing the camera.
    position = (depth - depth.min()) / span
    if view != "top":
        position = 1.0 - position
    colours = viridis(position)

    radius = max(1, int(point_radius)) * scale
    for x, y, colour in zip(u, v, colours):
        canvas[max(0, y - radius) : y + radius, max(0, x - radius) : x + radius] = colour

    return _box_downscale(canvas, scale)


def _label_strip(text: str, width: int, height: int = 18) -> np.ndarray:
    """A small caption bar. Uses Pillow's default bitmap font."""
    Image = _require_pil()
    from PIL import ImageDraw

    strip = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    draw.text((3, 3), text[: max(1, width // 6)], fill=(40, 40, 40))
    return np.asarray(strip)


def specimen_card(
    occupancy: np.ndarray,
    label: str,
    *,
    size: int = 200,
    views: tuple[str, ...] = ("front", "side", "top"),
    point_radius: int = 1,
    supersample: int = 3,
) -> np.ndarray:
    """One specimen rendered from several axes, with a caption."""
    panels = [
        render_volume(occupancy, view=view, size=size,
                      point_radius=point_radius, supersample=supersample)
        for view in views
    ]
    row = np.concatenate(panels, axis=1)
    divider = np.full((2, row.shape[1], 3), (220, 220, 220), dtype=np.uint8)
    return np.concatenate([_label_strip(label, row.shape[1]), row, divider], axis=0)


def contact_sheet(
    cards: list[np.ndarray], *, columns: int = 4, gap: int = 6
) -> np.ndarray:
    """Tile specimen cards into a single sheet."""
    if not cards:
        raise ValueError("no cards to tile")

    height = max(c.shape[0] for c in cards)
    width = max(c.shape[1] for c in cards)

    padded = []
    for card in cards:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        canvas[: card.shape[0], : card.shape[1]] = card
        padded.append(canvas)

    rows = []
    for start in range(0, len(padded), columns):
        chunk = padded[start : start + columns]
        while len(chunk) < columns:
            chunk.append(np.full((height, width, 3), 255, dtype=np.uint8))
        rows.append(np.concatenate(chunk, axis=1))

    spacer = np.full((gap, rows[0].shape[1], 3), 255, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for row in rows:
        stacked.extend([row, spacer])
    return np.concatenate(stacked[:-1], axis=0)


def export_ply(occupancy: np.ndarray, path: Path, *, voxel_size_m: float = VOXEL_SIZE_M) -> Path:
    """Write occupied voxel centres as an ASCII PLY point cloud."""
    points = occupied_points(occupancy, voxel_size_m)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines += [f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in points]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass
class VolumePayload:
    """A carved volume compressed for the interactive gallery."""

    plant_id: str
    species: str
    segmenter: str
    target_kg: float
    n_voxels: int
    volume_l: float
    height_m: float
    resolution: int
    data: str          # base64 of zlib-compressed uint8 xyz triples

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def volume_payload(
    cached, *, downsample: int = 2, max_points: int = 20000
) -> VolumePayload:
    """Quantise an occupancy grid to uint8 coordinates for the web viewer.

    A 128^3 boolean grid is 2 MB per specimen and there are sixty of them across
    both segmenters. Downsampling to 64^3 and storing occupied cells as three
    bytes each, zlib-compressed, brings the whole gallery under a megabyte.
    """
    occupancy = cached.occupancy
    resolution = occupancy.shape[0]

    if downsample > 1:
        trimmed = resolution - (resolution % downsample)
        blocks = occupancy[:trimmed, :trimmed, :trimmed].reshape(
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
        )
        occupancy = blocks.any(axis=(1, 3, 5))
        resolution = occupancy.shape[0]

    index = np.array(np.nonzero(occupancy)).T.astype(np.int64)
    if index.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(index.shape[0], max_points, replace=False)
        index = index[np.sort(keep)]

    scaled = np.clip(index * (255 // max(1, resolution - 1)), 0, 255).astype(np.uint8)
    blob = zlib.compress(scaled.tobytes(), 9)

    voxel = cached.voxel_size_m
    return VolumePayload(
        plant_id=cached.plant_id,
        species=cached.species,
        segmenter=cached.segmenter,
        target_kg=round(float(cached.target_kg), 3),
        n_voxels=int(cached.occupancy.sum()),
        volume_l=round(float(cached.occupancy.sum()) * voxel ** 3 * 1000, 2),
        height_m=round(
            float(
                (np.nonzero(cached.occupancy.any(axis=(0, 1)))[0].max() + 1) * voxel
            )
            if cached.occupancy.any()
            else 0.0,
            3,
        ),
        resolution=resolution,
        data=base64.b64encode(blob).decode("ascii"),
    )


def build_gallery(
    plant_ids: list[str],
    *,
    cache_dirs: dict[str, Path] | None = None,
    out_dir: Path = WORK_DIR / "reports" / "gallery",
    write_ply: bool = True,
    write_sheets: bool = True,
    columns: int = 4,
    card_size: int = 200,
    point_radius: int = 1,
    supersample: int = 3,
    max_points: int = 20000,
    downsample: int = 2,
    verbose: bool = True,
) -> dict:
    """Render every reconstruction under every available segmenter.

    Returns:
        A manifest dict with the payloads for the interactive viewer and the
        paths of everything written.
    """
    from ..data.preprocess import load_cached
    from .factorial import CACHE_DIRS

    Image = _require_pil()
    cache_dirs = cache_dirs or {
        name: path for name, path in CACHE_DIRS.items() if (path / "quality.json").exists()
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"segmenters": sorted(cache_dirs), "volumes": [], "files": []}

    for segmenter, cache_dir in sorted(cache_dirs.items()):
        cards = []
        for plant_id in plant_ids:
            try:
                cached = load_cached(plant_id, cache_dir)
            except FileNotFoundError:
                continue

            manifest["volumes"].append(
                volume_payload(cached, downsample=downsample,
                               max_points=max_points).as_dict()
            )

            if write_sheets:
                label = (
                    f"{plant_id}  {cached.species[:10]}  "
                    f"{cached.target_kg:.2f}kg  "
                    f"{cached.occupancy.sum() * cached.voxel_size_m ** 3 * 1000:.1f}L"
                )
                cards.append(specimen_card(
                    cached.occupancy, label, size=card_size,
                    point_radius=point_radius, supersample=supersample,
                ))

            if write_ply:
                path = export_ply(
                    cached.occupancy, out_dir / segmenter / f"{plant_id}.ply"
                )
                manifest["files"].append(str(path))

        if write_sheets and cards:
            sheet = out_dir / f"contact_sheet_{segmenter}.png"
            Image.fromarray(contact_sheet(cards, columns=columns)).save(sheet)
            manifest["files"].append(str(sheet))
            if verbose:
                print(f"  wrote {sheet} ({len(cards)} specimens)")

    return manifest


__all__ = [
    "AXES",
    "VolumePayload",
    "build_gallery",
    "contact_sheet",
    "export_ply",
    "occupied_points",
    "render_volume",
    "specimen_card",
    "volume_payload",
]
