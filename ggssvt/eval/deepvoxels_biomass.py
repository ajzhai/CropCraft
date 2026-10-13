"""Backproject DeepVoxels metric depth and apply the CropCraft biomass contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _features(path: Path, pot_height_m: float) -> np.ndarray:
    values = []
    for line in path.read_text().splitlines():
        if line == "end_header":
            break
    in_data = False
    for line in path.read_text().splitlines():
        if in_data and line.strip():
            fields = line.split()
            if len(fields) >= 3:
                values.append([float(fields[0]), float(fields[1]), float(fields[2])])
        elif line == "end_header":
            in_data = True
    points = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    above = points[points[:, 2] > pot_height_m] if points.size else points
    if not above.size:
        return np.zeros(5)
    keys = np.floor(above / 0.012).astype(np.int64)
    occupied = np.unique(keys, axis=0).shape[0]
    height = float(above[:, 2].max() - pot_height_m)
    footprint = float(np.unique(keys[:, :2], axis=0).shape[0]) * 0.012**2
    volume = occupied * 0.012**3
    return np.array([volume, height, footprint, np.linalg.norm(above[:, :2], axis=1).max(), volume / max(footprint * height, 1e-9)])


def depth_to_ply(depth, pose, intrinsics, output, pot_height_m):
    depth = np.asarray(depth).squeeze()
    focal, cx, cy = np.loadtxt(intrinsics, max_rows=1)
    height, width = depth.shape
    scale = height / 424.0
    focal, cx, cy = focal * scale, cx * scale, cy * scale
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    camera = np.stack(((xx - cx) * depth / focal, (yy - cy) * depth / focal, depth), -1)
    points = camera[valid] @ pose[:3, :3].T + pose[:3, 3]
    points = points[points[:, 2] > pot_height_m]
    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z", "end_header",
    ]
    output.write_text(
        "\n".join(header + [f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]) + "\n"
    )
    return _features(output, pot_height_m)


def run(render_depth: Path, pose_dir: Path, intrinsics: Path, output_dir: Path, pot_height_m: float):
    depth = np.load(render_depth)
    rows = []
    for index, pose_path in enumerate(sorted(pose_dir.glob("*.txt"))[: len(depth)]):
        path = output_dir / f"{index:06d}.ply"
        rows.append(depth_to_ply(depth[index], np.loadtxt(pose_path), intrinsics, path, pot_height_m))
    return {"views": len(rows), "pot_height_m": pot_height_m, "features_above_rim": np.nanmean(rows, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-depth", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pot-height-m", type=float, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
