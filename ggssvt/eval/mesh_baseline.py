"""Biomass from mesh geometry, as a fourth method in the comparison.

The existing feature sets describe *how much space* the plant occupies. This one
describes *how much surface* it has, which is a different physical quantity and,
for a leafy canopy, plausibly the more relevant one: a leaf's mass scales with
its area and it encloses almost no volume.

Features, all from :mod:`ggssvt.geometry.mesh`:

``canopy_area_m2``
    Surface area above the pot rim. The headline term, and the one carrying the
    hypothesis.
``surface_area_m2``
    Total area including the pot, kept so the regression can separate the two.
``enclosed_volume_m3``
    Volume by the divergence theorem, the mesh counterpart of the voxel count.
``solidity``
    Mesh volume over convex hull volume. Separates a sparse canopy from a
    compact one at equal volume -- on this dataset it cleanly separates the
    mostly-pot E001-E010 (around 0.64) from the real canopies (around 0.14).
``area_to_volume``
    High for thin leafy structure, low for a blob.
``height_m``

Mesh extraction takes a few seconds per specimen, so the metrics are cached
beside the specimen archives exactly as the DINO descriptors are.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from ..geometry.mesh import mesh_metrics
from .baselines import Baseline, SpecimenFeatures, _ridge_fit
from .metrics import RegressionMetrics, regression_metrics

MESH_CACHE_NAME = "mesh_metrics.json"


def mesh_cache_path(cache_dir: Path) -> Path:
    return cache_dir / MESH_CACHE_NAME


def compute_mesh_table(
    plant_ids: list[str],
    *,
    cache_dir: Path = WORK_DIR / "cache",
    smoothing: int = 0,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, dict]:
    """Mesh metrics for every specimen, cached to disk.

    Returns:
        Per plant id, the dict form of its :class:`~ggssvt.geometry.mesh.MeshMetrics`.
    """
    from ..data.preprocess import load_cached

    path = mesh_cache_path(cache_dir)
    table: dict[str, dict] = {}

    if use_cache and path.exists():
        table = json.loads(path.read_text(encoding="utf-8"))
        if all(pid in table for pid in plant_ids):
            if verbose:
                print(f"  reusing {path.name}")
            return {pid: table[pid] for pid in plant_ids}

    started = time.time()
    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        _, metrics = mesh_metrics(
            cached.occupancy,
            voxel_size_m=cached.voxel_size_m,
            pot_height_m=cached.pot_height_m,
            smoothing=smoothing,
        )
        table[plant_id] = metrics.as_dict()
        if verbose and index % 5 == 0:
            print(f"    meshed {index}/{len(plant_ids)}")

    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    if verbose:
        print(f"  meshed {len(plant_ids)} specimens in {time.time() - started:.0f}s")

    return {pid: table[pid] for pid in plant_ids}


def mesh_vector(metrics: dict) -> np.ndarray:
    """Feature vector for one specimen's mesh.

    ``solidity`` can be NaN when the convex hull is degenerate, which ridge
    regression cannot absorb, so it is substituted with a neutral value rather
    than propagating and poisoning the whole fit.
    """
    solidity = metrics.get("solidity", float("nan"))
    if not np.isfinite(solidity):
        solidity = 0.5

    return np.array(
        [
            metrics["canopy_area_m2"],
            metrics["surface_area_m2"],
            metrics["enclosed_volume_m3"],
            metrics["enclosed_volume_m3"] ** (2.0 / 3.0),
            solidity,
            min(metrics.get("area_to_volume", 0.0), 1e4),
            metrics["height_m"],
        ],
        dtype=np.float64,
    )


class MeshDerived(Baseline):
    """Ridge regression on mesh shape descriptors."""

    name = "mesh geometry"

    def __init__(self, table: dict[str, dict], alpha: float = 1.0):
        self.table = table
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.intercept = 0.0
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _design(self, features: list[SpecimenFeatures]) -> np.ndarray:
        return np.stack([mesh_vector(self.table[f.plant_id]) for f in features])

    def fit(self, features: list[SpecimenFeatures]) -> MeshDerived:
        targets = np.array([f.target_kg for f in features])
        self.weights, self.intercept, self.mean, self.scale = _ridge_fit(
            self._design(features), targets, self.alpha
        )
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("MeshDerived was not fitted")
        standardised = (self._design(features) - self.mean) / self.scale
        return standardised @ self.weights + self.intercept


class CanopyAreaAllometric(Baseline):
    """The single-term area law: ``log m = a log A + b``.

    The area counterpart of ``VolumeAllometric``, and the cleanest test of the
    hypothesis. If canopy area alone beats canopy volume alone, that is the
    result -- no learned model required, and much harder to argue with than a
    seven-feature ridge fit at n=28.
    """

    name = "canopy area allometric"

    def __init__(self, table: dict[str, dict], eps: float = 1e-9):
        self.table = table
        self.eps = eps
        self.slope = 1.0
        self.intercept = 0.0

    def _area(self, features: list[SpecimenFeatures]) -> np.ndarray:
        return np.array(
            [max(self.table[f.plant_id]["canopy_area_m2"], self.eps) for f in features]
        )

    def fit(self, features: list[SpecimenFeatures]) -> CanopyAreaAllometric:
        area = self._area(features)
        mass = np.array([max(f.target_kg, self.eps) for f in features])
        design = np.stack([np.log(area), np.ones_like(area)], axis=1)
        (self.slope, self.intercept), *_ = np.linalg.lstsq(
            design, np.log(mass), rcond=None
        )
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        return np.exp(self.slope * np.log(self._area(features)) + self.intercept)


def evaluate_with_mesh(
    plant_ids: list[str],
    *,
    cache_dir: Path = WORK_DIR / "cache",
    alpha: float = 1.0,
    verbose: bool = True,
) -> tuple[dict[str, tuple[RegressionMetrics, np.ndarray]], dict[str, dict]]:
    """Run every existing baseline plus the two mesh ones, under one LOOCV.

    Returns:
        ``(results, mesh_table)``.
    """
    from .baselines import evaluate_baselines, load_features, loocv_baseline

    features = load_features(plant_ids, cache_dir)
    table = compute_mesh_table(plant_ids, cache_dir=cache_dir, verbose=verbose)

    results = evaluate_baselines(features)

    for factory, label in (
        (lambda: CanopyAreaAllometric(table), CanopyAreaAllometric.name),
        (lambda: MeshDerived(table, alpha=alpha), MeshDerived.name),
    ):
        predicted, target = loocv_baseline(factory, features)
        results[label] = (regression_metrics(predicted, target), predicted)

    return results, table


__all__ = [
    "CanopyAreaAllometric",
    "MeshDerived",
    "compute_mesh_table",
    "evaluate_with_mesh",
    "mesh_cache_path",
    "mesh_vector",
]
