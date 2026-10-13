"""Precompute the geometry stage and cache it.

Registering, segmenting and carving one specimen takes several seconds, which is
fine once and unacceptable once per training epoch. This module runs the whole
NumPy pipeline ahead of time and writes a compressed archive per specimen:
registered poses, cropped RGB-D, subject masks, and the carved occupancy field
that serves as the self-supervision target.

The cache also records the quality diagnostics -- multi-view agreement, surface
coverage, the azimuth corrections the refinement applied -- so a specimen whose
registration failed can be identified and excluded before it contaminates
training, rather than after.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from ..config import (
    INPUT_CROP_TOP,
    INPUT_HEIGHT,
    KINECT_V2,
    VOXEL_SIZE_M,
    WORK_DIR,
)
from ..geometry.carving import carve, largest_connected_component, surface_coverage
from ..geometry.pot import PotEstimate, estimate_pot_height
from ..geometry.rig import estimate_rig
from ..geometry.segment import multiview_agreement, segment_specimen
from .dataset import Specimen, load_dataset, load_specimen, select_views

CACHE_VERSION = 2


@dataclass
class SpecimenQuality:
    """Diagnostics that decide whether a cached specimen is trustworthy."""

    plant_id: str
    n_views: int
    subject_distance_m: float
    hypothesis_agreement: float
    multiview_agreement: float
    surface_coverage: float
    refinement_gain: float
    max_azimuth_correction_deg: float
    volume_m3: float
    above_ground_volume_m3: float
    height_m: float
    connected_fraction: float
    n_warnings: int
    segmenter: str = "geometric"
    sam_acceptance_rate: float = float("nan")
    sam_pixel_change: float = float("nan")

    def is_usable(
        self,
        *,
        min_coverage: float = 0.35,
        min_agreement: float = 0.30,
        min_connected: float = 0.50,
    ) -> bool:
        """Whether this specimen should enter training.

        The thresholds are deliberately loose. They are there to catch a
        registration that failed outright, not to quietly drop the specimens
        that merely reconstruct poorly -- those are the interesting ones.
        """
        return (
            self.surface_coverage >= min_coverage
            and self.multiview_agreement >= min_agreement
            and self.connected_fraction >= min_connected
            and self.above_ground_volume_m3 > 0.0
        )


def _crop(array: np.ndarray) -> np.ndarray:
    """Crop the 424-row frame to a patch-divisible height."""
    top = INPUT_CROP_TOP
    return array[top : top + INPUT_HEIGHT]


def cache_path(plant_id: str, cache_dir: Path = WORK_DIR / "cache") -> Path:
    return cache_dir / f"{plant_id}.npz"


def preprocess_specimen(
    specimen: Specimen,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    keep_largest_component: bool = True,
    seed: int = 0,
    segmenter: str = "geometric",
    sam_segmenter=None,
    n_views: int | None = None,
) -> SpecimenQuality:
    """Run the geometry pipeline for one specimen and write its cache entry.

    Args:
        segmenter: ``"geometric"`` for the cylinder ROI, or ``"sam3d"`` for the
            SAM-refined mask. The choice propagates into the carved occupancy
            and therefore into the self-supervision targets, which is why each
            segmenter needs its own cache directory rather than a flag at
            training time.
        sam_segmenter: a loaded :class:`~ggssvt.geometry.sam3d.Sam3DSegmenter`,
            reused across specimens so the weights load once.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if n_views is not None:
        specimen = select_views(specimen, n_views)

    rig = estimate_rig(specimen, seed=seed)

    sam_stats = None
    if segmenter == "sam3d":
        if sam_segmenter is None:
            from ..geometry.sam3d import Sam3DSegmenter

            sam_segmenter = Sam3DSegmenter().load()
        segmentations, sam_stats = sam_segmenter.segment_specimen(specimen, rig)
    elif segmenter == "geometric":
        segmentations = segment_specimen(specimen, rig)
    else:
        raise ValueError(
            f"unknown segmenter {segmenter!r}; expected 'geometric' or 'sam3d'"
        )

    volume = carve(rig, segmentations, plant_id=specimen.plant_id)

    coverage = surface_coverage(volume, segmentations)
    connected = largest_connected_component(volume.occupancy)
    connected_fraction = (
        float(connected.sum()) / float(max(1, volume.occupancy.sum()))
    )
    if keep_largest_component:
        volume.occupancy = connected

    position_ids = [view.position_id for view in specimen.views]

    rgb = np.stack([_crop((view.load_rgb() * 255).astype(np.uint8)) for view in specimen.views])
    depth_mm = np.stack(
        [
            _crop((segmentations[pid].depth_m * 1000.0).astype(np.uint16))
            for pid in position_ids
        ]
    )
    masks = np.stack([_crop(segmentations[pid].mask) for pid in position_ids])

    rotations = np.stack([rig.pose(pid).rotation for pid in position_ids]).astype(np.float32)
    centres = np.stack([rig.pose(pid).centre for pid in position_ids]).astype(np.float32)
    azimuths = np.array([rig.pose(pid).azimuth_deg for pid in position_ids], dtype=np.float32)

    corrections = (
        rig.refinement.corrections if rig.refinement is not None else {}
    )
    azimuth_corrections = np.array(
        [
            corrections[pid].d_azimuth_deg if pid in corrections else 0.0
            for pid in position_ids
        ],
        dtype=np.float32,
    )

    np.savez_compressed(
        cache_path(specimen.plant_id, cache_dir),
        version=CACHE_VERSION,
        plant_id=specimen.plant_id,
        position_ids=np.array(position_ids),
        rgb=rgb,
        depth_mm=depth_mm,
        mask=np.packbits(masks, axis=None),
        mask_shape=np.array(masks.shape),
        rotation=rotations,
        centre=centres,
        azimuth_deg=azimuths,
        azimuth_correction_deg=azimuth_corrections,
        occupancy=np.packbits(volume.occupancy, axis=None),
        occupancy_shape=np.array(volume.occupancy.shape),
        n_informative=volume.n_informative.astype(np.int8),
        crop_top=INPUT_CROP_TOP,
        voxel_size_m=VOXEL_SIZE_M,
        segmenter=str(segmenter),
        target_kg=np.float32(specimen.target_kg if specimen.target_kg is not None else np.nan),
        species=str(specimen.species or "unknown"),
    )

    return SpecimenQuality(
        plant_id=specimen.plant_id,
        n_views=specimen.n_views,
        subject_distance_m=rig.subject_distance_m,
        hypothesis_agreement=rig.agreement,
        multiview_agreement=multiview_agreement(segmentations),
        surface_coverage=coverage,
        refinement_gain=(
            0.0 if rig.refinement is None else rig.refinement.improvement
        ),
        max_azimuth_correction_deg=float(np.abs(azimuth_corrections).max(initial=0.0)),
        volume_m3=volume.volume_m3,
        above_ground_volume_m3=volume.above_ground_volume_m3(),
        height_m=volume.height_m,
        connected_fraction=connected_fraction,
        n_warnings=len(rig.warnings) + len(specimen.warnings),
        segmenter=segmenter,
        sam_acceptance_rate=(
            float("nan") if sam_stats is None else sam_stats.acceptance_rate
        ),
        sam_pixel_change=(
            float("nan") if sam_stats is None else sam_stats.pixel_change
        ),
    )


def preprocess_dataset(
    *,
    cache_dir: Path = WORK_DIR / "cache",
    plant_ids: list[str] | None = None,
    seed: int = 0,
    verbose: bool = True,
    segmenter: str = "geometric",
    sam_model: str = "base",
    sam_device: str = "cpu",
    n_views: int | None = None,
    require_ground_truth: bool = True,
) -> list[SpecimenQuality]:
    """Preprocess every specimen and write a quality report beside the cache.

    Args:
        require_ground_truth: when False, specimens with no row in
            ``ground_truth.csv`` are carved too, and cached with a NaN target.
            Stage-1 pretraining fits occupancy against the carve and never reads
            the mass, so an unharvested plant is a perfectly good training
            example there. It costs twenty minutes of capture instead of a
            destroyed specimen, which is the only cheap axis this dataset has.
    """
    specimens = (
        [load_specimen(pid) for pid in plant_ids]
        if plant_ids
        else load_dataset(require_ground_truth=require_ground_truth)
    )

    sam_segmenter = None
    if segmenter == "sam3d":
        from ..geometry.sam3d import Sam3DSegmenter

        if verbose:
            print(f"Loading SAM ({sam_model}) on {sam_device}...")
        sam_segmenter = Sam3DSegmenter(model=sam_model, device=sam_device).load()

    report: list[SpecimenQuality] = []
    for index, specimen in enumerate(specimens, start=1):
        started = time.time()
        quality = preprocess_specimen(
            specimen,
            cache_dir=cache_dir,
            seed=seed,
            segmenter=segmenter,
            sam_segmenter=sam_segmenter,
            n_views=n_views,
        )
        report.append(quality)
        if verbose:
            flag = "" if quality.is_usable() else "  <-- EXCLUDED"
            print(
                f"[{index:2d}/{len(specimens)}] {specimen.plant_id}  "
                f"coverage={quality.surface_coverage:.3f}  "
                f"agreement={quality.multiview_agreement:.3f}  "
                f"above-ground={quality.above_ground_volume_m3 * 1000:6.2f} L  "
                f"({time.time() - started:.1f}s){flag}"
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "quality.json").write_text(
        json.dumps([asdict(q) for q in report], indent=2), encoding="utf-8"
    )
    return report


def load_quality(cache_dir: Path = WORK_DIR / "cache") -> dict[str, SpecimenQuality]:
    """Read the cached quality report."""
    path = cache_dir / "quality.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no quality report at {path}; run `python -m ggssvt.cli preprocess` first"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["plant_id"]: SpecimenQuality(**row) for row in rows}


@dataclass
class CachedSpecimen:
    """A preprocessed specimen, ready to become tensors."""

    plant_id: str
    species: str
    position_ids: list[str]
    rgb: np.ndarray            # (V, H, W, 3) uint8
    depth_m: np.ndarray        # (V, H, W) float32
    mask: np.ndarray           # (V, H, W) bool
    rotation: np.ndarray       # (V, 3, 3)
    centre: np.ndarray         # (V, 3)
    occupancy: np.ndarray      # (R, R, R) bool
    target_kg: float
    voxel_size_m: float
    crop_top: int
    segmenter: str = "geometric"

    @property
    def n_views(self) -> int:
        return len(self.position_ids)

    @cached_property
    def pot(self) -> PotEstimate:
        """Where this specimen's pot rim sits, measured from its own carve.

        Derived rather than stored: it is a function of the occupancy already in
        the cache, so keeping it here costs one pass over a 128^3 boolean and
        avoids a cache format that could disagree with the volume beside it.
        """
        return estimate_pot_height(self.occupancy, voxel_size_m=self.voxel_size_m)

    @property
    def pot_height_m(self) -> float:
        """Pot rim height, per specimen, falling back to the global constant.

        Prefer this over ``config.POT_HEIGHT_M``. The constant was fitted by eye
        to the E and M batches and is wrong by 0.14 m on V001-V008, whose pots
        weigh 17-32 kg against their 0.7-2.2 kg.
        """
        return self.pot.height_m

    def points_world(self) -> np.ndarray:
        """Per-pixel world coordinates, ``(V, H, W, 3)`` float32.

        Recomputed from depth and the cached pose rather than stored, which
        keeps the cache to roughly a third of the size at negligible cost.
        """
        height, width = self.depth_m.shape[1:]
        rows = np.arange(height, dtype=np.float32) + self.crop_top
        cols = np.arange(width, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(cols, rows)

        out = np.zeros((self.n_views, height, width, 3), dtype=np.float32)
        for index in range(self.n_views):
            z = self.depth_m[index]
            x = (grid_u - KINECT_V2.cx) * z / KINECT_V2.fx
            y = (grid_v - KINECT_V2.cy) * z / KINECT_V2.fy
            cam = np.stack([x, y, z], axis=-1)
            world = cam @ self.rotation[index].T + self.centre[index]
            out[index] = np.where((z > 0)[..., None], world, 0.0)
        return out


def load_cached(
    plant_id: str, cache_dir: Path = WORK_DIR / "cache"
) -> CachedSpecimen:
    """Load one preprocessed specimen."""
    path = cache_path(plant_id, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{plant_id} is not cached at {path}; "
            "run `python -m ggssvt.cli preprocess` first"
        )

    with np.load(path, allow_pickle=False) as data:
        version = int(data["version"])
        if version != CACHE_VERSION:
            raise ValueError(
                f"{path} was written by cache version {version}, this build "
                f"expects {CACHE_VERSION}; re-run preprocessing"
            )

        mask_shape = tuple(int(v) for v in data["mask_shape"])
        occupancy_shape = tuple(int(v) for v in data["occupancy_shape"])

        mask = np.unpackbits(data["mask"], count=int(np.prod(mask_shape)))
        occupancy = np.unpackbits(
            data["occupancy"], count=int(np.prod(occupancy_shape))
        )

        return CachedSpecimen(
            plant_id=str(data["plant_id"]),
            species=str(data["species"]),
            position_ids=[str(p) for p in data["position_ids"]],
            rgb=data["rgb"],
            depth_m=data["depth_mm"].astype(np.float32) / 1000.0,
            mask=mask.astype(bool).reshape(mask_shape),
            rotation=data["rotation"],
            centre=data["centre"],
            occupancy=occupancy.astype(bool).reshape(occupancy_shape),
            target_kg=float(data["target_kg"]),
            voxel_size_m=float(data["voxel_size_m"]),
            crop_top=int(data["crop_top"]),
            segmenter=str(data["segmenter"]) if "segmenter" in data else "geometric",
        )


def usable_plant_ids(
    cache_dir: Path = WORK_DIR / "cache",
    *,
    labelled: bool | None = True,
    **thresholds,
) -> list[str]:
    """Plant ids whose cached geometry passes the quality gate.

    Args:
        labelled: True keeps only specimens with a weighed mass, which is the
            default and what every regression must use -- a NaN target reaching
            a least-squares fit produces NaN coefficients and no error. False
            keeps only the unlabelled ones. None keeps both, which is what
            stage-1 pretraining wants, since it never reads the mass.
    """
    quality = load_quality(cache_dir)
    ids = sorted(pid for pid, q in quality.items() if q.is_usable(**thresholds))
    if labelled is None:
        return ids

    import numpy as np

    keep = []
    for pid in ids:
        path = cache_path(pid, cache_dir)
        if not path.exists():
            continue
        # np.load on an npz is lazy per member, so this reads one float rather
        # than the ~50 MB of views and occupancy that load_cached would.
        with np.load(path) as data:
            target = float(data["target_kg"])
        if np.isfinite(target) == labelled:
            keep.append(pid)
    return keep


__all__ = [
    "CACHE_VERSION",
    "CachedSpecimen",
    "SpecimenQuality",
    "cache_path",
    "load_cached",
    "load_quality",
    "preprocess_dataset",
    "preprocess_specimen",
    "usable_plant_ids",
]
