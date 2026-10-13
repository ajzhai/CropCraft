"""Prepare CropCraft specimens for the legacy DeepVoxels runner."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _poses(positions: dict) -> list[tuple[str, np.ndarray]]:
    poses = []
    for name in sorted(positions):
        pose = np.asarray(positions[name]["transform_matrix"], dtype=np.float64)
        pose[:3, 1:3] *= -1.0
        poses.append((name, pose))
    return poses


def _center(poses: list[tuple[str, np.ndarray]]) -> np.ndarray:
    system, target = [], []
    for _, pose in poses:
        direction = pose[:3, 2]
        projector = np.eye(3) - np.outer(direction, direction)
        system.append(projector)
        target.append(projector @ pose[:3, 3])
    result, *_ = np.linalg.lstsq(np.concatenate(system), np.concatenate(target), rcond=None)
    return result


def prepare(
    plant_dir: Path,
    out_dir: Path,
    *,
    scale: float = 2.0,
    focal_scale: float = 0.5,
    view_indices: list[int] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir, pose_dir = out_dir / "rgb", out_dir / "pose"
    rgb_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)
    positions = json.loads((plant_dir / "rig_positions.json").read_text())
    poses = _poses(positions)
    if view_indices is not None:
        poses = [pose for index, pose in enumerate(poses) if index in view_indices]
    for index, (name, pose) in enumerate(poses):
        shutil.copy2(plant_dir / "images" / f"{name}.png", rgb_dir / f"{index:06d}.png")
        (pose_dir / f"{index:06d}.txt").write_text(
            "\n".join(" ".join(f"{value:.9f}" for value in row) for row in pose) + "\n"
        )
    intrinsics = json.loads((plant_dir / "camA_intrinsics.json").read_text())
    width, height = int(intrinsics["w"]), int(intrinsics["h"])
    crop = min(width, height)
    cx = float(intrinsics["cx"]) - (width - crop) / 2.0
    cy = float(intrinsics["cy"])
    centre = _center(poses)
    (out_dir / "intrinsics.txt").write_text(
        f"{float(intrinsics['fl_x']) * focal_scale:.9f} {cx:.9f} {cy:.9f}\n"
        f"{centre[0]:.9f} {centre[1]:.9f} {centre[2]:.9f}\n"
        f"0.01\n{scale}\n{height} {width}\n"
    )
    (out_dir / "cropcraft_metadata.json").write_text(
        json.dumps(
            {
                "coordinate_conversion": "OpenGL camera-to-world to OpenCV camera-to-world",
                "grid_barycenter_m": centre.tolist(),
                "grid_scale_m": scale,
                "focal_scale": focal_scale,
                "view_count": len(poses),
                "pot_exclusion": "specimen-specific pot_height_m in biomass evaluation",
            },
            indent=2,
        )
        + "\n"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--focal-scale", type=float, default=0.5)
    parser.add_argument("--view-indices", type=int, nargs="+")
    args = parser.parse_args()
    print(prepare(**vars(args)))


if __name__ == "__main__":
    main()
