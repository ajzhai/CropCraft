"""Small-sample AB-SSViT biomass probe.

The DINOv2 backbone is frozen. A fold-local PCA and ridge head combine its
appearance descriptor with TSDF fused geometry, avoiding a high-variance
trainable attention stack for this small labelled dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from .baselines import load_features
from .dino_probe import descriptor_cache_path
from .metrics import regression_metrics


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _reduce(train: np.ndarray, test: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    _, _, vh = np.linalg.svd(train - mean, full_matrices=False)
    k = min(components, train.shape[0] - 1, train.shape[1])
    basis = vh[:k]
    return (train - mean) @ basis.T, (test - mean) @ basis.T


def _fit_fold(
    train_dino: np.ndarray,
    test_dino: np.ndarray,
    train_metadata: np.ndarray,
    test_metadata: np.ndarray,
    targets: np.ndarray,
    *,
    components: int,
    alphas: tuple[float, ...],
) -> float:
    dino, test_dino = _reduce(
        train_dino, test_dino, components=min(components, train_dino.shape[0] - 1)
    )
    dino, test_dino = _standardise(dino, test_dino)
    metadata, test_metadata = _standardise(train_metadata, test_metadata)
    train = np.concatenate((dino, metadata), axis=1)
    test = np.concatenate((test_dino, test_metadata), axis=1)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train = (train - mean) / scale
    test = (test - mean) / scale
    centred_targets = targets - targets.mean()
    predictions = []
    for alpha in alphas:
        gram = train.T @ train + alpha * np.eye(train.shape[1])
        weights = np.linalg.solve(gram, train.T @ centred_targets)
        predictions.append(float((test[0] @ weights) + targets.mean()))
    return float(np.mean(predictions))


def run(
    *,
    cache_dir: Path = WORK_DIR / "cache",
    variant: str = "base",
    components: int = 8,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> dict:
    """Run strict specimen-level LOOCV over cached DINOv2 and fused geometry."""
    descriptor_path = descriptor_cache_path("dinov2", variant, cache_dir)
    if not descriptor_path.exists():
        raise FileNotFoundError(
            f"Missing {descriptor_path}; run the DINOv2 probe first to build descriptors."
        )
    with np.load(descriptor_path, allow_pickle=False) as data:
        plant_ids = [str(value) for value in data["plant_ids"]]
        dino = np.asarray(data["features"], dtype=np.float64)
    features = load_features(plant_ids, cache_dir)
    if any(feature.fused is None for feature in features):
        raise FileNotFoundError(
            "TSDF fused descriptors are missing; run `ggssvt.cli fuse` first."
        )
    enriched_metadata = np.stack(
        [
            np.concatenate(
                [
                    feature.fused_vector(),
                    feature.geometric_vector(),
                    feature.profile_vector(),
                    feature.image_vector(),
                ]
            )
            for feature in features
        ]
    )
    fused_metadata = enriched_metadata[:, :7]
    targets = np.asarray([feature.target_kg for feature in features], dtype=np.float64)
    def evaluate(metadata: np.ndarray) -> np.ndarray:
        predictions = np.zeros(len(plant_ids), dtype=np.float64)
        for index in range(len(plant_ids)):
            train = np.arange(len(plant_ids)) != index
            predictions[index] = _fit_fold(
                dino[train], dino[index : index + 1],
                metadata[train], metadata[index : index + 1], targets[train],
                components=components, alphas=alphas,
            )
        return predictions

    predictions = evaluate(fused_metadata)
    enriched_predictions = evaluate(enriched_metadata)
    metrics = regression_metrics(predictions, targets).as_dict()
    return {
        "name": "ab_ssvit_dinov2_geometry",
        "status": "done",
        "n": len(plant_ids),
        "plant_ids": plant_ids,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "metrics": metrics,
        "enriched_metrics": regression_metrics(enriched_predictions, targets).as_dict(),
        "enriched_predictions": enriched_predictions.tolist(),
        "config": {
            "backbone": "dinov2",
            "variant": variant,
            "pca_components": components,
            "ridge_alphas": list(alphas),
            "metadata_features": int(enriched_metadata.shape[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=WORK_DIR / "cache")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        cache_dir=args.cache_dir, variant=args.variant, alphas=tuple(args.alphas)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"AB-SSViT RMSE={result['metrics']['rmse_kg']:.3f} kg R2={result['metrics']['r2']:.3f}")


if __name__ == "__main__":
    main()
