"""Aggregate DeepVoxels pot-excluded view clouds into plant-level biomass metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..data.preprocess import load_cached


def _features(path: Path, pot_height_m: float) -> np.ndarray:
    points = []
    data = False
    for line in path.read_text().splitlines():
        if line == "end_header":
            data = True
        elif data and line.strip():
            fields = line.split()
            points.append([float(fields[0]), float(fields[1]), float(fields[2])])
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not points.size:
        return np.zeros(5)
    points = points[points[:, 2] > pot_height_m]
    keys = np.floor(points / 0.012).astype(np.int64)
    occupied = np.unique(keys, axis=0).shape[0]
    height = float(points[:, 2].max() - pot_height_m)
    footprint = float(np.unique(keys[:, :2], axis=0).shape[0]) * 0.012**2
    volume = occupied * 0.012**3
    return np.array([volume, height, footprint, np.linalg.norm(points[:, :2], axis=1).max(), volume / max(footprint * height, 1e-9)])


def _loocv_ridge(x, y, alpha=1.0):
    predictions = np.zeros_like(y)
    for holdout in range(len(y)):
        train = np.arange(len(y)) != holdout
        mean, scale = x[train].mean(0), x[train].std(0)
        scale[scale < 1e-9] = 1
        z = (x[train] - mean) / scale
        design = np.c_[np.ones(len(z)), z]
        weights = np.linalg.solve(design.T @ design + alpha * np.diag([0] + [1] * z.shape[1]), design.T @ y[train])
        predictions[holdout] = np.r_[1, (x[holdout] - mean) / scale] @ weights
    return predictions


def _metrics(predicted, target):
    error = predicted - target
    return {"n": len(target), "rmse_kg": float(np.sqrt(np.mean(error**2))), "mae_kg": float(np.mean(np.abs(error))), "mare": float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1e-9))), "bias_kg": float(error.mean()), "r2": float(1 - np.sum(error**2) / max(np.sum((target - target.mean())**2), 1e-12))}


def run(cloud_root: Path, cache_dir: Path, output: Path) -> dict:
    rows = []
    for plant_dir in sorted(path for path in cloud_root.iterdir() if path.is_dir()):
        try:
            cached = load_cached(plant_dir.name, cache_dir)
        except FileNotFoundError:
            continue
        clouds = sorted(plant_dir.glob("*.ply"))
        if not clouds:
            continue
        features = np.stack([_features(path, cached.pot_height_m) for path in clouds])
        if not np.isfinite(features).all() or np.all(features == 0):
            continue
        rows.append((plant_dir.name, float(cached.target_kg), np.nanmean(features, axis=0)))
    if len(rows) < 4:
        result = {"status": "insufficient plants", "n": len(rows)}
    else:
        ids = [row[0] for row in rows]
        y = np.array([row[1] for row in rows])
        x = np.stack([row[2] for row in rows])
        result = {
            "status": "complete",
            "n": len(rows),
            "plant_ids": ids,
            "ridge_geometry": _metrics(_loocv_ridge(x, y), y),
            "features_above_rim": {plant_id: feature.tolist() for plant_id, _, feature in rows},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
