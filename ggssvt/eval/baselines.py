"""Baselines GG-SSVT has to beat.

Three of them, chosen to isolate what the method claims:

``VolumeAllometric``
    Mass regressed on carved above-ground volume alone, in log-log space --
    the classical allometric form. If GG-SSVT cannot beat this, the learned
    parts are contributing nothing over the geometry that produced them.

``GeometricFeatures``
    Ridge regression on hand-crafted shape descriptors from the same carved
    volume: volume, height, spread, compactness and so on. This is the
    reconstruct-then-regress pipeline used in the predecessor work on this rig,
    and it is the honest comparator for "does a learned volumetric
    representation beat hand-designed features on the same reconstruction?".

``Direct2D``
    Mass from image statistics only -- silhouette area, depth spread, apparent
    height -- with no 3D reconstruction at all. This is the comparator research
    question 3 needs: does reconstructing geometry actually improve biomass
    estimation over predicting straight from pixels?

All three are fitted with the same leave-one-out protocol as the model, so the
numbers are directly comparable. Every model here is closed-form, so LOOCV over
thirty specimens costs milliseconds and there is no excuse for reporting a
baseline fitted on the full set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres
from ..data.preprocess import CachedSpecimen, load_cached
from .metrics import RegressionMetrics, regression_metrics


def _ridge_fit(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit ridge regression on standardised features.

    Returns:
        ``(weights, intercept, mean, scale)`` so the caller can apply the same
        standardisation at prediction time.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardised = (features - mean) / scale

    n_features = standardised.shape[1]
    gram = standardised.T @ standardised + alpha * np.eye(n_features)
    centred = targets - targets.mean()
    weights = np.linalg.solve(gram, standardised.T @ centred)

    return weights, float(targets.mean()), mean, scale


@dataclass
class SpecimenFeatures:
    """Everything the baselines need from one cached specimen."""

    plant_id: str
    target_kg: float
    above_ground_volume_m3: float
    total_volume_m3: float
    height_m: float
    mean_spread_m: float
    max_spread_m: float
    compactness: float
    silhouette_area_m2: float
    mean_subject_pixels: float
    mean_subject_depth_m: float
    subject_pixel_height: float
    # Optional, because they come from work the plain carve does not do.
    profile: np.ndarray | None = None      # vertical cross-section shape
    fused: np.ndarray | None = None        # TSDF descriptors, from fusion.json

    def geometric_vector(self) -> np.ndarray:
        """Descriptors derived from the 3D reconstruction."""
        return np.array(
            [
                self.above_ground_volume_m3,
                self.above_ground_volume_m3 ** (2.0 / 3.0),
                self.height_m,
                self.mean_spread_m,
                self.max_spread_m,
                self.compactness,
                self.silhouette_area_m2,
            ],
            dtype=np.float64,
        )

    def profile_vector(self) -> np.ndarray:
        """How cross-section is distributed with height, normalised to shape.

        Eight bands above the pot rim plus a taper term, each divided by the
        total so this describes architecture rather than size. Size is already
        carried by the volume and height features, and leaving it in here would
        just duplicate them.

        The reason to have it at all: if the hull's *volume* is an envelope, its
        *shape* may still be real, and a tall thin sapling differs from a short
        bushy one in a way that tracks mass. It helps Eucalyptus and hurts Mango,
        which is what a stem-versus-canopy split predicts.
        """
        if self.profile is None:
            return np.zeros(9, dtype=np.float64)
        return self.profile

    def fused_vector(self) -> np.ndarray:
        """Descriptors from the TSDF fusion rather than the carve."""
        if self.fused is None:
            return np.zeros(7, dtype=np.float64)
        return self.fused

    def image_vector(self) -> np.ndarray:
        """Descriptors available without any 3D reconstruction."""
        return np.array(
            [
                self.mean_subject_pixels,
                self.mean_subject_pixels ** 1.5,
                self.mean_subject_depth_m,
                self.subject_pixel_height,
                self.mean_subject_pixels * self.mean_subject_depth_m ** 2,
            ],
            dtype=np.float64,
        )


def _vertical_profile_shape(cached, pot_height: float, n_bins: int = 8) -> np.ndarray:
    """Occupied cross-section per height band above the rim, normalised.

    Returns ``n_bins`` shares summing to one, plus a taper term that is positive
    when the plant is bottom-heavy. Zeros when nothing sits above the rim, which
    is a real state for the E001-E010 specimens rather than an error.
    """
    from ..geometry.pot import vertical_profile

    counts = vertical_profile(cached.occupancy).astype(np.float64)
    rim = round(pot_height / cached.voxel_size_m)
    above = counts[rim:]
    occupied = np.nonzero(above)[0]
    if occupied.size == 0:
        return np.zeros(n_bins + 1, dtype=np.float64)

    above = above[: occupied.max() + 1]
    shares = np.array([band.sum() for band in np.array_split(above, n_bins)])
    total = shares.sum()
    shares = shares / total if total > 0 else shares
    return np.concatenate([shares, [float(shares[0] - shares[-1])]])


def extract_features(
    cached: CachedSpecimen, fused: np.ndarray | None = None
) -> SpecimenFeatures:
    """Compute baseline features for one preprocessed specimen."""
    centres = voxel_grid_centres()
    occupancy = cached.occupancy
    voxel_volume = cached.voxel_size_m ** 3

    # Per specimen, not the global constant: pot mass spans 0.7-32 kg across the
    # three batches, so one cut height cannot be right for all of them.
    pot_height = cached.pot_height_m
    above = occupancy & (centres[..., 2] > pot_height)
    above_points = centres[above]

    if above_points.size:
        height = float(above_points[:, 2].max())
        radial = np.linalg.norm(above_points[:, :2], axis=1)
        mean_spread = float(radial.mean())
        max_spread = float(radial.max())
        above_volume = float(above.sum()) * voxel_volume
        # Occupied fraction of the cylinder the canopy sweeps out.
        envelope = np.pi * max(max_spread, 1e-3) ** 2 * max(height - pot_height, 1e-3)
        compactness = above_volume / envelope
        # Projected area of the canopy onto the floor.
        footprint = above.any(axis=2).sum() * cached.voxel_size_m ** 2
    else:
        height = mean_spread = max_spread = above_volume = compactness = 0.0
        footprint = 0.0

    profile = _vertical_profile_shape(cached, pot_height)

    subject_pixels = cached.mask.sum(axis=(1, 2)).astype(np.float64)
    depths = [
        cached.depth_m[i][cached.mask[i]] for i in range(cached.n_views)
    ]
    valid_depths = np.concatenate([d[d > 0] for d in depths]) if depths else np.zeros(1)

    rows = np.arange(cached.mask.shape[1])
    pixel_heights = []
    for index in range(cached.n_views):
        occupied_rows = rows[cached.mask[index].any(axis=1)]
        pixel_heights.append(
            float(occupied_rows.max() - occupied_rows.min()) if occupied_rows.size else 0.0
        )

    return SpecimenFeatures(
        plant_id=cached.plant_id,
        target_kg=cached.target_kg,
        above_ground_volume_m3=above_volume,
        total_volume_m3=float(occupancy.sum()) * voxel_volume,
        height_m=height,
        mean_spread_m=mean_spread,
        max_spread_m=max_spread,
        compactness=compactness,
        silhouette_area_m2=float(footprint),
        mean_subject_pixels=float(subject_pixels.mean()),
        mean_subject_depth_m=float(valid_depths.mean()) if valid_depths.size else 0.0,
        subject_pixel_height=float(np.mean(pixel_heights)) if pixel_heights else 0.0,
        profile=profile,
        fused=fused,
    )


FUSION_REPORT = WORK_DIR / "reports" / "fusion.json"


def _load_fusion(path: Path = FUSION_REPORT) -> dict[str, np.ndarray]:
    """Cached TSDF descriptors, keyed by plant id.

    Read from disk rather than recomputed: fusing 36 specimens takes ten
    minutes, and a baseline sweep that quietly did that would stop being a thing
    anyone runs. Missing file means the fused baseline is simply not offered.
    """
    if not path.exists():
        return {}

    import json

    from .fusion_features import FUSION_KEYS

    table = json.loads(path.read_text(encoding="utf-8"))
    return {
        plant_id: np.array([row[k] for k in FUSION_KEYS], dtype=np.float64)
        for plant_id, row in table.items()
        if all(k in row for k in FUSION_KEYS)
    }


def load_features(
    plant_ids: list[str],
    cache_dir: Path = WORK_DIR / "cache",
    *,
    fusion_report: Path = FUSION_REPORT,
) -> list[SpecimenFeatures]:
    """Extract baseline features for many specimens.

    Raises:
        ValueError: if any specimen has no weighed mass. Unlabelled specimens
            are legitimate stage-1 pretraining data and are cached with a NaN
            target, but a NaN reaching a least-squares fit produces NaN
            coefficients and no error at all, so every score downstream would be
            NaN with nothing to say why. Refusing here is the only cheap place
            to catch it.
    """
    import numpy as np

    fusion = _load_fusion(fusion_report)
    features = [
        extract_features(load_cached(pid, cache_dir), fused=fusion.get(pid))
        for pid in plant_ids
    ]

    unlabelled = [f.plant_id for f in features if not np.isfinite(f.target_kg)]
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} specimen(s) have no weighed mass and cannot be "
            f"regressed: {', '.join(unlabelled[:5])}"
            f"{' ...' if len(unlabelled) > 5 else ''}. "
            "usable_plant_ids(labelled=True) is the default and excludes them; "
            "only stage-1 pretraining should pass labelled=None."
        )
    return features


class Baseline(ABC):
    """A closed-form biomass regressor."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, features: list[SpecimenFeatures]) -> Baseline:
        ...

    @abstractmethod
    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        ...


class VolumeAllometric(Baseline):
    """Classical allometry: ``log m = a log V + b``.

    The one-parameter form every biomass paper starts from. Fitted in log space
    because plant mass and volume are related by a power law, not a line.
    """

    name = "volume allometric"

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.slope = 1.0
        self.intercept = 0.0

    def fit(self, features: list[SpecimenFeatures]) -> VolumeAllometric:
        volume = np.array([max(f.above_ground_volume_m3, self.eps) for f in features])
        mass = np.array([max(f.target_kg, self.eps) for f in features])
        design = np.stack([np.log(volume), np.ones_like(volume)], axis=1)
        (self.slope, self.intercept), *_ = np.linalg.lstsq(design, np.log(mass), rcond=None)
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        volume = np.array([max(f.above_ground_volume_m3, self.eps) for f in features])
        return np.exp(self.slope * np.log(volume) + self.intercept)


class _RidgeBaseline(Baseline):
    """Shared ridge machinery for the feature-vector baselines."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: np.ndarray | None = None
        self.intercept = 0.0
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        raise NotImplementedError

    def fit(self, features: list[SpecimenFeatures]) -> _RidgeBaseline:
        design = np.stack([self._vector(f) for f in features])
        targets = np.array([f.target_kg for f in features])
        self.weights, self.intercept, self.mean, self.scale = _ridge_fit(
            design, targets, self.alpha
        )
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(f"{self.name} was not fitted")
        design = np.stack([self._vector(f) for f in features])
        standardised = (design - self.mean) / self.scale
        return standardised @ self.weights + self.intercept


class GeometricFeatures(_RidgeBaseline):
    """Ridge on hand-crafted descriptors of the carved reconstruction."""

    name = "geometric features"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return feature.geometric_vector()


class Direct2D(_RidgeBaseline):
    """Ridge on image statistics only, with no 3D reconstruction."""

    name = "direct 2D"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return feature.image_vector()


class FusedGeometry(_RidgeBaseline):
    """Ridge on descriptors from the TSDF fusion instead of the carve.

    Present only when `cli fuse` has been run, because fusing all 36 specimens
    costs ten minutes and should not happen inside a baseline sweep.
    """

    name = "fused geometry"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return feature.fused_vector()


class ProfileAugmented(_RidgeBaseline):
    """Image statistics plus the vertical cross-section profile.

    The only combination this project has found that improves on its component
    parts with an interval clear of zero, and only within Eucalyptus: -0.112 kg
    with a 95% interval of [-0.219, -0.009] against direct 2D alone. On the
    pooled set it is not resolved, because it helps stems and hurts canopies.
    Read the species split before quoting it.
    """

    name = "2D + profile"

    @staticmethod
    def _vector(feature: SpecimenFeatures) -> np.ndarray:
        return np.concatenate([feature.image_vector(), feature.profile_vector()])


class MeanPredictor(Baseline):
    """Predicts the training mean. The floor any real method must clear."""

    name = "mean"

    def __init__(self):
        self.value = 0.0

    def fit(self, features: list[SpecimenFeatures]) -> MeanPredictor:
        self.value = float(np.mean([f.target_kg for f in features]))
        return self

    def predict(self, features: list[SpecimenFeatures]) -> np.ndarray:
        return np.full(len(features), self.value)


def loocv_baseline(
    baseline_factory, features: list[SpecimenFeatures]
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out predictions for a baseline.

    Args:
        baseline_factory: a zero-argument callable returning a fresh baseline.

    Returns:
        ``(predicted_kg, target_kg)``, aligned with ``features``.
    """
    predictions = np.zeros(len(features))
    targets = np.array([f.target_kg for f in features])

    for index in range(len(features)):
        train = [f for position, f in enumerate(features) if position != index]
        model = baseline_factory().fit(train)
        predictions[index] = float(model.predict([features[index]])[0])

    return predictions, targets


def evaluate_baselines(
    features: list[SpecimenFeatures],
) -> dict[str, tuple[RegressionMetrics, np.ndarray]]:
    """Run every baseline under LOOCV.

    Returns:
        Per baseline name: its metrics and its leave-one-out predictions.
    """
    factories = {
        MeanPredictor.name: MeanPredictor,
        VolumeAllometric.name: VolumeAllometric,
        GeometricFeatures.name: GeometricFeatures,
        Direct2D.name: Direct2D,
        ProfileAugmented.name: ProfileAugmented,
    }
    # Every specimen or none. A partially populated fusion.json -- which is what
    # `cli fuse --plants ...` leaves behind -- would otherwise put zero vectors in
    # for the specimens it skipped, and the method would appear in the table
    # scoring worse than the mean for a reason that has nothing to do with the
    # method. Silently wrong beats loudly absent only in the wrong direction.
    if features and all(f.fused is not None for f in features):
        factories[FusedGeometry.name] = FusedGeometry

    results: dict[str, tuple[RegressionMetrics, np.ndarray]] = {}
    for name, factory in factories.items():
        predicted, target = loocv_baseline(factory, features)
        results[name] = (regression_metrics(predicted, target), predicted)
    return results


__all__ = [
    "Baseline",
    "Direct2D",
    "GeometricFeatures",
    "MeanPredictor",
    "SpecimenFeatures",
    "VolumeAllometric",
    "evaluate_baselines",
    "extract_features",
    "load_features",
    "loocv_baseline",
]
