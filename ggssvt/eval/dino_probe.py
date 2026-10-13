"""Frozen-feature probe: does a DINO backbone add anything on this dataset?

Training the full GG-SSVT under three backbones costs three GPU runs. A linear
probe on frozen features answers the narrower question -- *how much biomass
information is in the representation itself* -- for the cost of one forward pass
per view, on a CPU, and it answers it before any GPU time is committed.

The probe is a fair test only if it is set up carefully at n=28:

* **The descriptor is pooled over subject patches only.** Pooling the whole frame
  would mostly measure the greenhouse, and would differ between specimens for
  reasons unrelated to the plant.
* **Raw backbone features, not the projected ones.** The learned projection
  inside the model is randomly initialised until trained, so probing through it
  would measure noise.
* **PCA and standardisation are fitted inside each fold.** DINOv2-small gives a
  768-dimensional descriptor for 28 specimens. Fitting the projection on all of
  them and then cross-validating leaks the held-out specimen into its own
  prediction, and at this ratio that leak is large enough to manufacture a
  result out of nothing.

A probe that fails does not prove the backbone is useless in the full model --
GG-SSVT uses DINO features per patch with 3D anchors, not pooled per specimen.
It does bound what the pretrained representation contributes on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from ..data.preprocess import CachedSpecimen, load_cached
from .baselines import SpecimenFeatures, load_features
from .metrics import RegressionMetrics, regression_metrics


@dataclass
class ProbeDescriptors:
    """Pooled frozen features for a set of specimens."""

    backbone: str
    plant_ids: list[str]
    features: np.ndarray        # (n_specimens, n_features)
    targets: np.ndarray         # (n_specimens,)
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])


def _pool_subject_patches(
    tokens: np.ndarray, coverage: np.ndarray, min_coverage: float = 0.10
) -> np.ndarray:
    """Mean and standard deviation over the patches covering the plant.

    Args:
        tokens: ``(P, C)`` patch features for one view.
        coverage: ``(P,)`` subject coverage per patch.

    Returns:
        ``(2C,)`` descriptor. Falls back to the best-covered patches when the
        mask is thin, so a view is never dropped entirely.
    """
    selected = coverage >= min_coverage
    if selected.sum() < 4:
        selected = np.zeros_like(coverage, dtype=bool)
        selected[np.argsort(coverage)[::-1][:4]] = True

    subset = tokens[selected]
    return np.concatenate([subset.mean(axis=0), subset.std(axis=0)])


def extract_descriptor(
    cached: CachedSpecimen, backbone, *, min_coverage: float = 0.10
) -> np.ndarray:
    """Pool one specimen's frozen backbone features into a single vector.

    Views are pooled independently and then averaged, so the descriptor does not
    depend on view ordering.
    """
    import torch
    import torch.nn.functional as F

    per_view: list[np.ndarray] = []

    for index in range(cached.n_views):
        rgb = torch.from_numpy(cached.rgb[index]).float().permute(2, 0, 1) / 255.0
        rgb = rgb.unsqueeze(0)

        with torch.no_grad():
            tokens, grid_h, grid_w = backbone.patch_tokens(rgb)

        mask = torch.from_numpy(cached.mask[index]).float().reshape(1, 1, *cached.mask.shape[1:])
        coverage = F.adaptive_avg_pool2d(mask, (grid_h, grid_w)).reshape(-1).numpy()

        per_view.append(
            _pool_subject_patches(tokens[0].numpy(), coverage, min_coverage)
        )

    return np.mean(per_view, axis=0)


def descriptor_cache_path(
    backbone_kind: str, variant: str, cache_dir: Path
) -> Path:
    """Where pooled descriptors for one backbone and one segmenter are stored."""
    return cache_dir / f"descriptors_{backbone_kind}_{variant}.npz"


def build_descriptors(
    plant_ids: list[str],
    backbone_kind: str,
    *,
    variant: str = "small",
    cache_dir: Path = WORK_DIR / "cache",
    verbose: bool = True,
    use_cache: bool = True,
) -> ProbeDescriptors:
    """Extract pooled frozen features for every specimen under one backbone.

    Descriptors are cached beside the specimen archives, keyed by backbone and
    segmenter. Extraction is the expensive part of the factorial -- one DINO
    forward pass per view per specimen per cell -- and it is deterministic given
    frozen weights, so recomputing it on every rerun is pure waste.
    """
    from ..models.backbones import build_backbone

    started = time.time()
    path = descriptor_cache_path(backbone_kind, variant, cache_dir)

    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as data:
            cached_ids = [str(p) for p in data["plant_ids"]]
            if cached_ids == list(plant_ids):
                if verbose:
                    print(f"    {backbone_kind}-{variant}: reusing {path.name}")
                return ProbeDescriptors(
                    backbone=f"{backbone_kind}-{variant}",
                    plant_ids=cached_ids,
                    features=data["features"],
                    targets=data["targets"],
                    notes=["loaded from cache"],
                )

    backbone = build_backbone(backbone_kind, variant=variant)
    backbone.eval()

    rows, targets = [], []
    for position, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        rows.append(extract_descriptor(cached, backbone))
        targets.append(cached.target_kg)
        if verbose and position % 5 == 0:
            print(f"    {backbone_kind}: {position}/{len(plant_ids)} specimens")

    descriptors = ProbeDescriptors(
        backbone=f"{backbone_kind}-{variant}",
        plant_ids=list(plant_ids),
        features=np.stack(rows),
        targets=np.array(targets),
        seconds=time.time() - started,
    )

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            plant_ids=np.array(descriptors.plant_ids),
            features=descriptors.features,
            targets=descriptors.targets,
        )

    return descriptors


def _fit_pca(train: np.ndarray, n_components: int):
    """PCA fitted on the training split only.

    Returns:
        ``(mean, components)`` for transforming any split.
    """
    mean = train.mean(axis=0)
    centred = train - mean
    n_components = int(min(n_components, centred.shape[0] - 1, centred.shape[1]))
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    return mean, vh[:n_components]


def _ridge(train: np.ndarray, targets: np.ndarray, alpha: float):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (train - mean) / scale

    gram = standardised.T @ standardised + alpha * np.eye(standardised.shape[1])
    weights = np.linalg.solve(gram, standardised.T @ (targets - targets.mean()))
    return weights, float(targets.mean()), mean, scale


def loocv_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_components: int = 8,
    alpha: float = 1.0,
) -> np.ndarray:
    """Leave-one-out predictions from a frozen-feature descriptor.

    PCA and standardisation are refitted inside every fold, on the training
    specimens only.
    """
    predictions = np.zeros(features.shape[0])

    for index in range(features.shape[0]):
        train_mask = np.ones(features.shape[0], dtype=bool)
        train_mask[index] = False

        train = features[train_mask]
        pca_mean, components = _fit_pca(train, n_components)

        train_reduced = (train - pca_mean) @ components.T
        test_reduced = (features[index] - pca_mean) @ components.T

        weights, intercept, mean, scale = _ridge(
            train_reduced, targets[train_mask], alpha
        )
        predictions[index] = float(
            ((test_reduced - mean) / scale) @ weights + intercept
        )

    return predictions


@dataclass
class ProbeResult:
    """One probe condition."""

    name: str
    metrics: RegressionMetrics
    predictions: np.ndarray
    n_features: int
    n_components: int


def run_probe_experiment(
    plant_ids: list[str],
    *,
    backbones: tuple[str, ...] = ("dinov2",),
    variant: str = "small",
    cache_dir: Path = WORK_DIR / "cache",
    n_components: int = 8,
    alpha: float = 1.0,
    verbose: bool = True,
) -> tuple[dict[str, ProbeResult], list[str]]:
    """Compare no-DINO geometry features against each available DINO backbone.

    Returns:
        ``(results, skipped)`` where ``skipped`` records backbones that could
        not be loaded, with the reason.
    """
    from ..models.backbones import backbone_is_available

    geometric: list[SpecimenFeatures] = load_features(plant_ids, cache_dir)
    targets = np.array([f.target_kg for f in geometric])
    geometry_matrix = np.stack([f.geometric_vector() for f in geometric])

    results: dict[str, ProbeResult] = {}
    skipped: list[str] = []

    # The no-DINO control, through the identical probe machinery so the
    # comparison differs only in the features.
    control = loocv_probe(
        geometry_matrix, targets, n_components=min(n_components, geometry_matrix.shape[1]), alpha=alpha
    )
    results["geometry only (no DINO)"] = ProbeResult(
        name="geometry only (no DINO)",
        metrics=regression_metrics(control, targets),
        predictions=control,
        n_features=geometry_matrix.shape[1],
        n_components=min(n_components, geometry_matrix.shape[1]),
    )

    for kind in backbones:
        available, reason = backbone_is_available(kind, variant)
        if not available:
            skipped.append(f"{kind}: {reason}")
            if verbose:
                print(f"  skipping {kind} -- {reason.splitlines()[0]}")
            continue

        if verbose:
            print(f"  extracting {kind} features for {len(plant_ids)} specimens...")
        descriptors = build_descriptors(
            plant_ids, kind, variant=variant, cache_dir=cache_dir, verbose=verbose
        )

        predictions = loocv_probe(
            descriptors.features, targets, n_components=n_components, alpha=alpha
        )
        results[descriptors.backbone] = ProbeResult(
            name=descriptors.backbone,
            metrics=regression_metrics(predictions, targets),
            predictions=predictions,
            n_features=descriptors.n_features,
            n_components=n_components,
        )

        # DINO on top of geometry: the question that actually matters is whether
        # the appearance prior adds anything the reconstruction has not already
        # captured, not whether it beats geometry outright.
        combined = np.concatenate([descriptors.features, geometry_matrix], axis=1)
        combined_predictions = loocv_probe(
            combined, targets, n_components=n_components, alpha=alpha
        )
        label = f"{descriptors.backbone} + geometry"
        results[label] = ProbeResult(
            name=label,
            metrics=regression_metrics(combined_predictions, targets),
            predictions=combined_predictions,
            n_features=combined.shape[1],
            n_components=n_components,
        )

    return results, skipped


__all__ = [
    "ProbeDescriptors",
    "ProbeResult",
    "build_descriptors",
    "descriptor_cache_path",
    "extract_descriptor",
    "loocv_probe",
    "run_probe_experiment",
]
