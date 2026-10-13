"""Render pot-excluded RGB-D fusion as presentation-ready turntable videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from ..config import KINECT_V2
from ..data.preprocess import load_cached


WIDTH, HEIGHT = 1280, 720
FPS = 24
DURATION_S = 10
CANVAS = (22, 25, 32)
PANEL = (35, 39, 48)
TEXT = (239, 242, 247)
GREEN = (96, 224, 154)
RED = (86, 97, 244)
MARGIN = 34


def _voxel_fuse(points: np.ndarray, colors: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points).all(axis=1)
    points, colors = points[valid], colors[valid]
    if not len(points):
        return points.reshape(0, 3), colors.reshape(0, 3)
    cells = np.floor(points / voxel_m).astype(np.int32)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    fused_points = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    fused_colors = np.column_stack(
        [np.bincount(inverse, weights=colors[:, axis]) / counts for axis in range(3)]
    )
    return fused_points, fused_colors.clip(0, 255).astype(np.uint8)


def _project_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    yaw: float,
    *,
    width: int = 680,
    height: int = 520,
) -> np.ndarray:
    image = np.full((height, width, 3), (245, 247, 249), dtype=np.uint8)
    if not len(points):
        return image

    centred = points - points.mean(axis=0, keepdims=True)
    c, s = math.cos(yaw), math.sin(yaw)
    rotated = centred @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    pitch = math.radians(12.0)
    cp, sp = math.cos(pitch), math.sin(pitch)
    pitched = rotated @ np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])

    span = max(float(np.ptp(pitched[:, :2], axis=0).max()), 1e-5)
    scale = 0.84 * min(width, height) / span
    px = np.rint(width / 2 + pitched[:, 0] * scale).astype(np.int32)
    py = np.rint(height / 2 - pitched[:, 1] * scale).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px, py = px[inside], py[inside]
    depth = pitched[inside, 2]
    rgb = colors[inside]
    height_t = np.clip((points[inside, 2] - points[:, 2].min()) /
                       max(float(np.ptp(points[:, 2])), 1e-6), 0.0, 1.0)
    low = np.array([44.0, 132.0, 73.0])
    high = np.array([190.0, 220.0, 112.0])
    height_color = low[None, :] * (1.0 - height_t[:, None]) + high[None, :] * height_t[:, None]
    rgb = (0.30 * rgb.astype(np.float32) + 0.70 * height_color).clip(0, 255).astype(np.uint8)

    order = np.argsort(depth)
    image[py[order], px[order]] = rgb[order]
    return cv2.GaussianBlur(image, (3, 3), 0.45)


def _resize_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    factor = min(width / w, height / h)
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (18, 21, 27), dtype=np.uint8)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    canvas[y : y + size[1], x : x + size[0]] = resized
    return canvas


def _masked_view(cached, points_world: np.ndarray, index: int) -> np.ndarray:
    image = cached.rgb[index].copy()
    depth_valid = cached.depth_m[index] > 0
    subject = cached.mask[index] & depth_valid
    retained = subject & (points_world[index, :, :, 2] > cached.pot_height_m)
    pot_removed = subject & ~retained

    image[~retained] = (image[~retained].astype(np.float32) * 0.34).astype(np.uint8)
    image[pot_removed] = (
        image[pot_removed].astype(np.float32) * 0.22
        + np.array([244, 88, 80], dtype=np.float32) * 0.78
    ).astype(np.uint8)

    # Mark the kept canopy boundary; pot pixels below the detected rim stay red.
    edge = cv2.morphologyEx(retained.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    image[edge > 0] = (96, 255, 177)
    return image


def _draw_label(frame: np.ndarray, text: str, position: tuple[int, int], *,
                scale: float = 0.55, color: tuple[int, int, int] = TEXT,
                thickness: int = 1) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _frame(cached, points_world, fused_points, fused_colors, frame_idx: int, metadata: dict) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), CANVAS, dtype=np.uint8)
    cv2.rectangle(frame, (24, 18), (WIDTH - 24, 92), PANEL, -1)
    _draw_label(frame, f"{cached.plant_id}  |  {cached.species}", (46, 52), scale=0.9, thickness=2)
    _draw_label(frame, "AB-SSViT INPUT PIPELINE  |  pot excluded before RGB-D fusion",
                (46, 78), scale=0.52, color=(180, 190, 205))

    left = (24, 108, 490, 680)
    right = (530, 108, 1256, 680)
    cv2.rectangle(frame, (left[0], left[1]), (left[2], left[3]), PANEL, -1)
    cv2.rectangle(frame, (right[0], right[1]), (right[2], right[3]), PANEL, -1)
    _draw_label(frame, "PER-VIEW SEGMENTATION", (42, 140), scale=0.58, color=GREEN, thickness=2)
    _draw_label(frame, "POT REMOVED BEFORE FUSION", (548, 140), scale=0.58, color=GREEN, thickness=2)

    view_idx = (frame_idx // max(1, FPS // 2)) % cached.n_views
    view = _masked_view(cached, points_world, view_idx)
    view_canvas = _resize_fit(view, 446, 430)
    frame[156:586, 34:480] = view_canvas
    _draw_label(frame, f"View {view_idx + 1}/{cached.n_views}  |  position {cached.position_ids[view_idx]}",
                (42, 612), scale=0.48)
    _draw_label(frame, f"Kept {metadata['kept_points_per_view'][view_idx]:,} pts  |  "
                       f"pot removed {metadata['excluded_pot_points_per_view'][view_idx]:,} pts",
                (42, 638), scale=0.43)
    _draw_label(frame, "Green outline = retained   Red = rim-filtered pixels",
                (42, 661), scale=0.40, color=(205, 210, 220))

    yaw = 2.0 * math.pi * frame_idx / (FPS * DURATION_S)
    cloud = _project_cloud(fused_points, fused_colors, yaw)
    frame[150:670, 548:1228] = cloud
    _draw_label(frame, f"{len(fused_points):,} fused plant points  |  voxel 8 mm",
                (548, 697), scale=0.49)
    _draw_label(frame, "Height-tinted point render; measured RGB is retained in the PLY",
                (548, 676), scale=0.42, color=(106, 112, 124))

    rim_label = "detected" if cached.pot.confident else "fallback estimate"
    cv2.rectangle(frame, (810, 36), (1235, 84), (28, 32, 40), -1)
    _draw_label(frame, f"Rim: {cached.pot_height_m:.3f} m ({rim_label})",
                (830, 67), scale=0.55)
    return frame


def render_one(plant_id: str, cache_dir: Path, output_dir: Path) -> dict:
    cached = load_cached(plant_id, cache_dir)
    if not cached.pot.confident:
        raise ValueError(
            f"{plant_id} has no confident geometric rim; refusing to label a "
            "fallback cut as pot segmentation."
        )

    points_world = cached.points_world()
    point_parts, color_parts = [], []
    mask = cached.mask
    valid = cached.depth_m > 0
    retained_counts, removed_counts = [], []

    for view in range(cached.n_views):
        subject = mask[view] & valid[view]
        above_rim = points_world[view, :, :, 2] > cached.pot_height_m
        keep = subject & above_rim
        removed = subject & ~above_rim
        retained_counts.append(int(keep.sum()))
        removed_counts.append(int(removed.sum()))
        if keep.any():
            point_parts.append(points_world[view][keep])
            color_parts.append(cached.rgb[view][keep])

    if not point_parts:
        raise ValueError(f"{plant_id} has no valid above-rim RGB-D points.")
    points, colors = _voxel_fuse(
        np.concatenate(point_parts), np.concatenate(color_parts), 0.008
    )
    if not len(points):
        raise ValueError(f"{plant_id} produced an empty fused plant cloud.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / f"{plant_id}_pot_excluded.ply"
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]
    ply_path.write_text(
        "\n".join(
            header
            + [
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}"
                for p, c in zip(points, colors)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    video_path = output_dir / f"{plant_id}_pot_excluded_reconstruction.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer for {video_path}")
    try:
        metadata = {
            "plant_id": plant_id,
            "species": cached.species,
            "rim_height_m": cached.pot_height_m,
            "rim_confident": cached.pot.confident,
            "rim_reason": cached.pot.reason,
            "views": cached.n_views,
            "kept_points_per_view": retained_counts,
            "excluded_pot_points_per_view": removed_counts,
            "fused_point_count": int(len(points)),
            "voxel_size_m": 0.008,
            "filter_order": "subject mask -> measured depth -> per-view world transform -> pot-rim exclusion -> point-cloud fusion",
            "video": video_path.name,
            "point_cloud": ply_path.name,
        }
        preview = _frame(cached, points_world, points, colors, 0, metadata)
        cv2.imwrite(
            str(output_dir / f"{plant_id}_preview.png"),
            cv2.cvtColor(preview, cv2.COLOR_RGB2BGR),
        )
        for frame_idx in range(FPS * DURATION_S):
            rgb_frame = _frame(cached, points_world, points, colors, frame_idx, metadata)
            writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    metadata_path = output_dir / f"{plant_id}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-ids", nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    results = [render_one(pid, args.cache_dir, args.output_dir) for pid in args.plant_ids]
    index = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Pot-excluded CropCraft reconstructions</title>"
        "<style>body{font:16px system-ui;background:#171a21;color:#eef;margin:2rem}article{margin:2rem auto;max-width:1100px}"
        "video{width:100%;border-radius:10px;background:#222}code{color:#8fe2ad}</style></head><body>"
        "<h1>AB-SSViT: pot-excluded RGB-D reconstructions</h1>"
        "<p>These are deterministic RGB-D input-pipeline reconstructions, not novel geometry predicted by DINOv2 or a learned decoder. "
        "Each view is segmented by the cached subject mask and specimen-specific rim in world coordinates before fusion. "
        "Green marks retained plant pixels; red marks subject pixels excluded below the rim. Only confidently detected rims are included.</p>"
    ]
    for item in results:
        index.append(
            f"<article><h2>{item['plant_id']} — {item['species']}</h2>"
            f"<p>Rim height: <code>{item['rim_height_m']:.3f} m</code> · "
            f"views: {item['views']} · fused points: {item['fused_point_count']:,}</p>"
            f"<video controls preload='metadata' src='{item['video']}'></video>"
            f"<p><a href='{item['point_cloud']}'>Pot-excluded colored PLY</a> · "
            f"<a href='{item['plant_id']}_metadata.json'>Processing metadata</a></p></article>"
        )
    index.append("</body></html>")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.html").write_text("\n".join(index), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
