"""Create exploratory stem-inclusive pseudo-turntable previews from D455 sweeps.

The capture protocol records intrinsics and turntable angles but no calibrated
camera-to-axis transform. These previews therefore use a fixed central plant
axis, a heuristic RGB/depth mask, and nominal turntable angles. They are useful
for inspecting capture coverage and stem visibility only, not as calibrated 3D
reconstructions or segmentation results.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from pathlib import Path

import numpy as np

from ..config import Intrinsics
from ..geometry.rig import fit_floor, levelling_rotation

EXPECTED_ANGLES = tuple(range(0, 360, 15))
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
MAX_AXIS_RADIUS_M = 0.70
PLANT_HEIGHT_RANGE_M = (-2.5, 0.12)


def _mask_view(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    background_m: np.ndarray,
    *,
    axis_x: int,
    pot_rim_y: int,
    foreground_margin_m: float = 0.12,
) -> np.ndarray:
    """Keep green foliage and near-background wood around the plant axis."""
    import cv2

    height, width = depth_m.shape
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    green = (
        (hsv[:, :, 0] >= 20)
        & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 32)
        & (hsv[:, :, 2] >= 24)
    )
    crop = np.zeros_like(green)
    crop[: max(0, pot_rim_y), int(0.30 * width) : int(0.70 * width)] = True
    foreground = (
        np.isfinite(background_m)
        & (depth_m > 0)
        & (depth_m < background_m - foreground_margin_m)
    )
    foliage = green & foreground & crop

    y_grid, x_grid = np.ogrid[:height, :width]
    stem_corridor = (
        (np.abs(x_grid - axis_x) <= 42)
        & (y_grid >= int(0.10 * height))
        & (y_grid < pot_rim_y)
    )
    stem_candidate = (depth_m > 0) & crop & stem_corridor & ~green
    stem_support = stem_candidate

    mask = foliage | stem_support
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    return mask.astype(bool)


def _rotation_about_vertical(points: np.ndarray, angle_deg: float, pivot: np.ndarray) -> np.ndarray:
    """Rotate points around the levelled-frame z axis."""
    angle = math.radians(-angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (points - pivot) @ rotation.T + pivot


def _project_cloud(points: np.ndarray, colours: np.ndarray, view_angle: float, canvas: np.ndarray) -> None:
    """Render a cloud in an orthographic pseudo-turntable view."""
    if points.size == 0:
        return
    theta = math.radians(view_angle)
    right = np.array([math.cos(theta), math.sin(theta), 0.0])
    horizontal = points @ right
    vertical = points[:, 2]
    span_x = max(float(np.ptp(horizontal)), 0.05)
    span_y = max(float(np.ptp(vertical)), 0.05)
    scale = 0.86 * min((PANEL_WIDTH - 36) / span_x, (PANEL_HEIGHT - 64) / span_y)
    center_x = (float(horizontal.min()) + float(horizontal.max())) * 0.5
    center_y = (float(vertical.min()) + float(vertical.max())) * 0.5
    px = np.rint((horizontal - center_x) * scale + PANEL_WIDTH * 0.5).astype(np.int32)
    py = np.rint(PANEL_HEIGHT * 0.5 - (vertical - center_y) * scale).astype(np.int32)
    visible = (px >= 0) & (px < PANEL_WIDTH) & (py >= 0) & (py < PANEL_HEIGHT)
    rgb = np.clip(colours[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    canvas[py[visible], px[visible]] = rgb[visible]


def _render_frame(
    rgb: np.ndarray,
    mask: np.ndarray,
    cloud: np.ndarray,
    colours: np.ndarray,
    angle: int,
    plant_id: str,
) -> np.ndarray:
    import cv2

    left = cv2.resize(rgb, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    overlay = left.copy()
    overlay[resized_mask] = (40, 230, 80)
    left = cv2.addWeighted(left, 0.58, overlay, 0.42, 0)

    right = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 248, dtype=np.uint8)
    _project_cloud(cloud, colours, angle * 2.0, right)
    frame = np.zeros((PANEL_HEIGHT + 96, PANEL_WIDTH * 2, 3), dtype=np.uint8)
    frame[:] = (30, 34, 42)
    frame[68 : 68 + PANEL_HEIGHT, :PANEL_WIDTH] = left
    frame[68 : 68 + PANEL_HEIGHT, PANEL_WIDTH:] = right
    cv2.putText(
        frame,
        f"{plant_id} | D455 exploratory reconstruction | source turntable {angle:03d} deg",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (240, 245, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Green overlay: heuristic foliage/stem candidates (not ground-truth masks)",
        (18, PANEL_HEIGHT + 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 225, 235),
        1,
        cv2.LINE_AA,
    )
    return frame


def _casefold_path(parent: Path, name: str) -> Path:
    matches = [path for path in parent.iterdir() if path.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one case-insensitive match for {name!r} in {parent}, "
            f"found {len(matches)}"
        )
    return matches[0]


def create_preview(
    dataset_root: Path,
    plant_id: str,
    output_dir: Path,
    *,
    pot_rim_y: int = 590,
    foreground_margin_m: float = 0.12,
) -> dict:
    """Write a provisional video, preview frame, and point-cloud artifact."""
    import cv2

    plant_dir = _casefold_path(dataset_root / "plants", plant_id)
    manifest = json.loads(
        (plant_dir / "capture_manifest.json").read_text(encoding="utf-8")
    )
    camera = manifest.get("camera", {})
    depth_scale = float(camera.get("depth_scale_m_per_unit", 0))
    intrinsics = camera.get("color_intrinsics", {})
    fx, fy = float(intrinsics.get("fx", 0)), float(intrinsics.get("fy", 0))
    ppx, ppy = float(intrinsics.get("ppx", 0)), float(intrinsics.get("ppy", 0))
    if depth_scale <= 0 or fx <= 0 or fy <= 0:
        raise ValueError(f"{plant_id} is missing valid depth scale or color intrinsics")

    rgb_views, depth_views, masks = [], [], []
    for angle in EXPECTED_ANGLES:
        stem = f"{angle:03d}.png"
        rgb = cv2.imread(str(plant_dir / "rgb" / stem), cv2.IMREAD_COLOR)
        raw_depth = cv2.imread(
            str(plant_dir / "depth" / stem), cv2.IMREAD_UNCHANGED
        )
        if rgb is None or raw_depth is None or raw_depth.ndim != 2:
            raise ValueError(f"missing or invalid RGB-D frame {plant_id}/{stem}")
        rgb_views.append(rgb)
        depth_views.append(raw_depth.astype(np.float32) * depth_scale)

    depth_stack = np.stack(depth_views)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        background = np.nanmedian(
            np.where(depth_stack > 0, depth_stack, np.nan), axis=0
        )
    background[~np.isfinite(background)] = 0.0

    height, width = depth_stack.shape[1:]
    axis_x = width // 2
    intrinsic = Intrinsics(
        width=int(intrinsics["width"]),
        height=int(intrinsics["height"]),
        fx=fx,
        fy=fy,
        cx=ppx,
        cy=ppy,
    )
    plane = fit_floor(depth_views[0], intrinsics=intrinsic)
    level_rotation = levelling_rotation(plane)

    pivot_points = []
    for depth_m in depth_views:
        y0, y1 = pot_rim_y + 25, min(height, pot_rim_y + 55)
        x0, x1 = max(0, axis_x - 18), min(width, axis_x + 18)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        depth_patch = depth_m[y0:y1, x0:x1]
        valid = depth_patch > 0
        z = depth_patch[valid]
        if z.size:
            camera_points = np.column_stack(
                [
                    (xs[valid] - ppx) * z / fx,
                    (ys[valid] - ppy) * z / fy,
                    z,
                ]
            )
            pivot_points.append(camera_points @ level_rotation.T)
    if not pivot_points:
        raise ValueError(f"cannot estimate an approximate turntable pivot for {plant_id}")
    pivot_surface = np.concatenate(pivot_points, axis=0)
    floor_height = -plane.offset
    pivot = np.array(
        [
            float(np.median(pivot_surface[:, 0])),
            float(np.median(pivot_surface[:, 1])),
            floor_height,
        ],
        dtype=np.float64,
    )
    rim_points = []
    y0, y1 = max(0, pot_rim_y - 12), min(height, pot_rim_y + 2)
    x0, x1 = max(0, axis_x - 16), min(width, axis_x + 16)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    for depth_m in depth_views:
        depth_patch = depth_m[y0:y1, x0:x1]
        valid = depth_patch > 0
        z = depth_patch[valid]
        if z.size:
            camera_points = np.column_stack(
                [
                    (xs[valid] - ppx) * z / fx,
                    (ys[valid] - ppy) * z / fy,
                    z,
                ]
            )
            rim_points.append(camera_points @ level_rotation.T)
    if not rim_points:
        raise ValueError(f"cannot estimate approximate pot-rim height for {plant_id}")
    rim_height = float(np.median(np.concatenate(rim_points, axis=0)[:, 2]))

    cloud_chunks, colour_chunks = [], []
    point_counts = []
    for angle, rgb, depth_m, mask in zip(
        EXPECTED_ANGLES,
        rgb_views,
        depth_views,
        [
            _mask_view(
                rgb,
                depth_m,
                background,
                axis_x=axis_x,
                pot_rim_y=pot_rim_y,
                foreground_margin_m=foreground_margin_m,
            )
            for rgb, depth_m in zip(rgb_views, depth_views)
        ],
    ):
        sampled_mask = mask[::2, ::2]
        ys, xs = np.nonzero(sampled_mask)
        ys, xs = ys * 2, xs * 2
        z = depth_m[ys, xs].astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        ys, xs, z = ys[valid], xs[valid], z[valid]
        points_camera = np.column_stack(
            [
                (xs - ppx) * z / fx,
                (ys - ppy) * z / fy,
                z,
            ]
        )
        points_level = points_camera @ level_rotation.T
        relative = points_level - pivot
        axis_radius = np.hypot(relative[:, 0], relative[:, 1])
        plant_height = points_level[:, 2] - rim_height
        accepted = (
            (axis_radius <= MAX_AXIS_RADIUS_M)
            & (plant_height > 0.0)
            & (plant_height <= abs(PLANT_HEIGHT_RANGE_M[0]))
        )
        ys, xs = ys[accepted], xs[accepted]
        points_camera = points_camera[accepted]
        points_level = points_level[accepted]
        accepted_mask = np.zeros_like(mask)
        accepted_mask[ys, xs] = True
        masks.append(accepted_mask)
        cloud_chunks.append(_rotation_about_vertical(points_level, angle, pivot))
        colour_chunks.append(rgb[ys, xs, ::-1].astype(np.float32) / 255.0)
        point_counts.append(int(points_camera.shape[0]))

    cloud = np.concatenate(cloud_chunks, axis=0)
    colours = np.concatenate(colour_chunks, axis=0)
    keep = np.isfinite(cloud).all(axis=1)
    cloud, colours = cloud[keep], colours[keep]
    if cloud.shape[0] > 150_000:
        indices = np.linspace(0, cloud.shape[0] - 1, 150_000, dtype=np.int64)
        cloud, colours = cloud[indices], colours[indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{plant_id}_d455_exploratory.mp4"
    preview_path = output_dir / f"{plant_id}_d455_preview.png"
    ply_path = output_dir / f"{plant_id}_d455_exploratory.ply"
    metadata_path = output_dir / f"{plant_id}_d455_metadata.json"

    first_frame = _render_frame(
        rgb_views[0], masks[0], cloud, colours, EXPECTED_ANGLES[0], plant_id
    )
    height_out, width_out = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width_out, height_out),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {video_path}")
    try:
        for angle, rgb, mask in zip(EXPECTED_ANGLES, rgb_views, masks):
            writer.write(_render_frame(rgb, mask, cloud, colours, angle, plant_id))
    finally:
        writer.release()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"video writer produced no output at {video_path}")
    if not cv2.imwrite(str(preview_path), first_frame):
        raise RuntimeError(f"could not write preview image at {preview_path}")

    with ply_path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(cloud)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, colour in zip(cloud, colours):
            red, green, blue = np.clip(colour * 255.0, 0, 255).astype(np.uint8)
            stream.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{red} {green} {blue}\n"
            )

    report = {
        "plant_id": plant_id,
        "capture_folder": plant_dir.name,
        "camera_serial_number": str(camera.get("serial_number", "unknown")),
        "views": len(EXPECTED_ANGLES),
        "per_view_mask_pixels": [int(mask.sum()) for mask in masks],
        "per_view_sampled_points": point_counts,
        "fused_point_count": int(len(cloud)),
        "heuristic": {
            "background": "per-pixel median depth over all 24 turntable views",
            "foreground_margin_m": foreground_margin_m,
            "foliage": "HSV threshold intersected with foreground and central crop",
            "stem": "non-green foreground pixels in a 42-pixel corridor around approximate plant axis, linked to foliage support",
            "pot_exclusion": f"discard image rows at and below approximate rim row {pot_rim_y}",
            "turntable_pose": "nominal angle applied as a vertical rotation around an approximate central pivot",
            "axis_radius_limit_m": MAX_AXIS_RADIUS_M,
            "height_relative_to_rim_limit_m": abs(PLANT_HEIGHT_RANGE_M[0]),
            "floor_plane_normal_camera": plane.normal.tolist(),
            "floor_plane_camera_height_m": plane.offset,
            "floor_inlier_fraction": plane.inlier_fraction,
            "estimated_pot_rim_height_above_floor_m": rim_height - floor_height,
        },
        "limitations": [
            "No calibrated camera-to-turntable extrinsics or axis direction.",
            "RGB/depth mask is a heuristic, not validated segmentation.",
            "Background depth variation leaves greenhouse clutter and can remove plant pixels.",
            "The approximate pixel-row pot cut is not a measured 3D pot-rim exclusion.",
            "Floor leveling uses an uncalibrated per-specimen RANSAC plane; camera distortion is not explicitly undistorted.",
            "The cylindrical proximity gate is a heuristic and may remove outer leaves or retain nearby background objects.",
            "This is a pseudo-turntable point cloud for visual inspection, not an accuracy-validated reconstruction.",
        ],
        "video": video_path.name,
        "preview": preview_path.name,
        "point_cloud": ply_path.name,
    }
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def create_gallery(reports: list[dict], output_dir: Path) -> Path:
    cards = []
    for report in reports:
        plant_id = html.escape(report["plant_id"])
        cards.append(
            f"<article><h2>{plant_id}</h2>"
            f"<video controls preload='metadata' src='{html.escape(report['video'])}'></video>"
            f"<p><a href='{html.escape(report['point_cloud'])}'>Exploratory colored PLY</a> · "
            f"<a href='{html.escape(report['metadata'])}'>Metadata and limitations</a></p></article>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>D455 exploratory reconstructions EU01-EU10</title>"
        "<style>body{font:16px system-ui;background:#171a21;color:#eef;margin:2rem}"
        "article{margin:2rem auto;max-width:1100px}video{width:100%;background:#222}"
        "a{color:#8fe2ad}</style></head><body>"
        "<h1>D455 EU01-EU10: exploratory pseudo-turntable views</h1>"
        "<p>Stem and foliage masks, pot cuts, axis, and turntable pose are heuristic. "
        "These visualizations are not calibrated or accuracy-validated reconstructions.</p>"
        + "\n".join(cards)
        + "</body></html>\n"
    )
    path = output_dir / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--plant-ids",
        nargs="+",
        default=[f"EU{i:02}" for i in range(1, 11)],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("work_dirs/ab_ssvit/d455-exploratory/reconstructions"),
    )
    parser.add_argument("--pot-rim-y", type=int, default=590)
    parser.add_argument("--foreground-margin-m", type=float, default=0.12)
    args = parser.parse_args()
    reports = [
        create_preview(
            args.dataset_root,
            plant_id,
            args.out_dir,
            pot_rim_y=args.pot_rim_y,
            foreground_margin_m=args.foreground_margin_m,
        )
        for plant_id in args.plant_ids
    ]
    for report in reports:
        report["metadata"] = f"{report['plant_id']}_d455_metadata.json"
    gallery = create_gallery(reports, args.out_dir)
    print(f"Wrote {len(reports)} exploratory D455 reconstruction previews to {args.out_dir}")
    print(f"Gallery: {gallery}")


if __name__ == "__main__":
    main()
