"""Torch dataset over the preprocessed cache.

One item is one specimen: twelve registered RGB-D views plus the carved
occupancy field. The occupancy field is not returned as a dense 128^3 grid --
that is 2.1 million voxels of which fewer than one percent are occupied, and
training on all of them wastes almost the whole batch on empty air far from the
plant.

Instead each item carries a balanced sample of query points: occupied voxels,
free voxels near the surface, and free voxels drawn uniformly. The near-surface
negatives are the ones that matter, because they are where the decision boundary
lives; uniform negatives alone let the network score well by predicting "empty"
everywhere.

Biomass fine-tuning needs the *unbiased* volume integral instead, so it uses a
uniform grid sample. Both sampling modes are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import (
    POT_HEIGHT_M,
    VOXEL_RESOLUTION,
    VOXEL_SIZE_M,
    WORK_DIR,
    voxel_grid_centres,
)
from ..data.preprocess import CachedSpecimen, load_cached, usable_plant_ids


@dataclass(frozen=True)
class SamplingConfig:
    """How query points are drawn for the occupancy loss."""

    n_queries: int = 8192
    occupied_fraction: float = 0.35
    near_surface_fraction: float = 0.40
    near_surface_radius_voxels: int = 3


def _dilate_3d(volume: np.ndarray, iterations: int) -> np.ndarray:
    """Binary dilation on a 3D grid, used to find the near-surface band."""
    out = volume.copy()
    for _ in range(max(0, iterations)):
        grown = out.copy()
        for axis in (0, 1, 2):
            grown |= np.roll(out, 1, axis=axis)
            grown |= np.roll(out, -1, axis=axis)
        out = grown
    return out


def sample_queries(
    occupancy: np.ndarray,
    centres: np.ndarray,
    config: SamplingConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced set of query points from a carved volume.

    Returns:
        ``(points, labels)`` of shape ``(n_queries, 3)`` and ``(n_queries,)``.
    """
    flat_occupancy = occupancy.reshape(-1)
    flat_centres = centres.reshape(-1, 3)

    occupied_index = np.nonzero(flat_occupancy)[0]
    if occupied_index.size == 0:
        # Nothing carved: fall back to uniform sampling so the item is still
        # usable as a negative-only example rather than crashing the epoch.
        index = rng.choice(flat_centres.shape[0], config.n_queries, replace=False)
        return flat_centres[index], np.zeros(config.n_queries, dtype=np.float32)

    band = _dilate_3d(occupancy, config.near_surface_radius_voxels).reshape(-1)
    near_index = np.nonzero(band & ~flat_occupancy)[0]
    far_index = np.nonzero(~band)[0]

    n_occupied = int(config.n_queries * config.occupied_fraction)
    n_near = int(config.n_queries * config.near_surface_fraction)
    n_far = config.n_queries - n_occupied - n_near

    def draw(pool: np.ndarray, count: int) -> np.ndarray:
        if pool.size == 0 or count <= 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(pool, count, replace=pool.size < count)

    chosen = np.concatenate(
        [draw(occupied_index, n_occupied), draw(near_index, n_near), draw(far_index, n_far)]
    )
    if chosen.size < config.n_queries:
        filler = draw(np.arange(flat_centres.shape[0]), config.n_queries - chosen.size)
        chosen = np.concatenate([chosen, filler])

    points = flat_centres[chosen].astype(np.float32)
    labels = flat_occupancy[chosen].astype(np.float32)

    # Jitter within the voxel so the decoder learns a continuous field rather
    # than memorising the grid it was carved on.
    jitter = rng.uniform(-0.5, 0.5, size=points.shape).astype(np.float32) * VOXEL_SIZE_M
    return points + jitter, labels


def uniform_grid_sample(
    centres: np.ndarray, stride: int
) -> tuple[np.ndarray, float]:
    """A strided, uniform subsample of the voxel grid, for volume integration.

    Returns:
        ``(points, volume_per_point_m3)``.
    """
    subsampled = centres[::stride, ::stride, ::stride]
    points = subsampled.reshape(-1, 3).astype(np.float32)
    return points, (VOXEL_SIZE_M * stride) ** 3


@dataclass
class SpecimenBatch:
    """One collated specimen, ready for :class:`~ggssvt.models.ggssvt.GGSSVT`."""

    plant_id: list[str]
    rgb: torch.Tensor              # (B, V, 3, H, W)
    depth: torch.Tensor            # (B, V, 1, H, W)
    points_world: torch.Tensor     # (B, V, 3, H, W)
    subject: torch.Tensor          # (B, V, 1, H, W)
    query_points: torch.Tensor     # (B, Q, 3)
    query_labels: torch.Tensor     # (B, Q)
    target_kg: torch.Tensor        # (B,)
    pot_height_m: torch.Tensor     # (B,) per-specimen rim, not a constant
    occupancy: torch.Tensor | None = None   # (B, R, R, R), eval only

    def to(self, device: torch.device | str) -> SpecimenBatch:
        moved = {}
        for key, value in self.__dict__.items():
            moved[key] = value.to(device) if torch.is_tensor(value) else value
        return SpecimenBatch(**moved)


class SpecimenDataset(Dataset):
    """Preprocessed specimens as tensors.

    Args:
        plant_ids: which specimens to serve. Defaults to every cached specimen
            that passes the quality gate.
        cache_dir: where :mod:`ggssvt.data.preprocess` wrote its archives.
        sampling: query sampling settings for the occupancy loss.
        uniform_stride: grid stride for the biomass integration sample. A stride
            of 2 gives 64^3 = 262k points, enough for a stable volume integral
            at a fraction of the memory of the full grid.
        mode: ``"occupancy"`` for self-supervised pretraining (balanced query
            sampling), ``"biomass"`` for fine-tuning (uniform grid sampling).
        return_volume: also return the dense carved grid, for evaluation.
    """

    def __init__(
        self,
        plant_ids: list[str] | None = None,
        *,
        cache_dir: Path = WORK_DIR / "cache",
        sampling: SamplingConfig = SamplingConfig(),
        uniform_stride: int = 2,
        mode: str = "occupancy",
        return_volume: bool = False,
        seed: int = 0,
    ):
        if mode not in {"occupancy", "biomass"}:
            raise ValueError(f"unknown mode {mode!r}")

        self.plant_ids = list(plant_ids) if plant_ids else usable_plant_ids(cache_dir)
        if not self.plant_ids:
            raise RuntimeError(
                f"no usable specimens in {cache_dir}; run preprocessing and check "
                "work_dirs/ggssvt/cache/quality.json"
            )

        self.cache_dir = cache_dir
        self.sampling = sampling
        self.uniform_stride = uniform_stride
        self.mode = mode
        self.return_volume = return_volume
        self.seed = seed

        self._centres = voxel_grid_centres()
        self._cache: dict[str, CachedSpecimen] = {}

    def __len__(self) -> int:
        return len(self.plant_ids)

    def specimen(self, plant_id: str) -> CachedSpecimen:
        """Load (and memoise) one cached specimen."""
        if plant_id not in self._cache:
            self._cache[plant_id] = load_cached(plant_id, self.cache_dir)
        return self._cache[plant_id]

    def __getitem__(self, index: int) -> dict:
        plant_id = self.plant_ids[index]
        cached = self.specimen(plant_id)
        rng = np.random.default_rng(self.seed + index)

        rgb = torch.from_numpy(cached.rgb).float().permute(0, 3, 1, 2) / 255.0
        depth = torch.from_numpy(cached.depth_m).float().unsqueeze(1)
        points = torch.from_numpy(cached.points_world()).float().permute(0, 3, 1, 2)
        subject = torch.from_numpy(cached.mask).float().unsqueeze(1)

        if self.mode == "occupancy":
            query_points, query_labels = sample_queries(
                cached.occupancy, self._centres, self.sampling, rng
            )
        else:
            grid_points, _ = uniform_grid_sample(self._centres, self.uniform_stride)
            query_points = grid_points
            query_labels = self._labels_at(cached.occupancy, grid_points)

        item = {
            "plant_id": plant_id,
            "rgb": rgb,
            "depth": depth,
            "points_world": points,
            "subject": subject,
            "query_points": torch.from_numpy(query_points),
            "query_labels": torch.from_numpy(query_labels),
            "target_kg": torch.tensor(cached.target_kg, dtype=torch.float32),
            "pot_height_m": torch.tensor(cached.pot_height_m, dtype=torch.float32),
        }
        if self.return_volume:
            item["occupancy"] = torch.from_numpy(cached.occupancy)
        return item

    def _labels_at(self, occupancy: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Carved occupancy sampled at arbitrary world points."""
        half = VOXEL_RESOLUTION * VOXEL_SIZE_M / 2.0
        ix = np.clip(
            np.floor((points[:, 0] + half) / VOXEL_SIZE_M).astype(np.int64),
            0,
            VOXEL_RESOLUTION - 1,
        )
        iy = np.clip(
            np.floor((points[:, 1] + half) / VOXEL_SIZE_M).astype(np.int64),
            0,
            VOXEL_RESOLUTION - 1,
        )
        iz = np.clip(
            np.floor(points[:, 2] / VOXEL_SIZE_M).astype(np.int64),
            0,
            VOXEL_RESOLUTION - 1,
        )
        return occupancy[ix, iy, iz].astype(np.float32)

    @property
    def volume_per_query_m3(self) -> float:
        """Volume each uniform query point stands for, in biomass mode."""
        return (VOXEL_SIZE_M * self.uniform_stride) ** 3

    def targets_kg(self) -> np.ndarray:
        """Ground-truth mass for every served specimen, in order."""
        return np.array([self.specimen(pid).target_kg for pid in self.plant_ids])

    def species(self) -> list[str]:
        return [self.specimen(pid).species for pid in self.plant_ids]


def collate(items: list[dict]) -> SpecimenBatch:
    """Stack specimen items into a :class:`SpecimenBatch`."""
    return SpecimenBatch(
        plant_id=[item["plant_id"] for item in items],
        rgb=torch.stack([item["rgb"] for item in items]),
        depth=torch.stack([item["depth"] for item in items]),
        points_world=torch.stack([item["points_world"] for item in items]),
        subject=torch.stack([item["subject"] for item in items]),
        query_points=torch.stack([item["query_points"] for item in items]),
        query_labels=torch.stack([item["query_labels"] for item in items]),
        target_kg=torch.stack([item["target_kg"] for item in items]),
        pot_height_m=torch.stack([item["pot_height_m"] for item in items]),
        occupancy=(
            torch.stack([item["occupancy"] for item in items])
            if "occupancy" in items[0]
            else None
        ),
    )


__all__ = [
    "POT_HEIGHT_M",
    "SamplingConfig",
    "SpecimenBatch",
    "SpecimenDataset",
    "collate",
    "sample_queries",
    "uniform_grid_sample",
]
