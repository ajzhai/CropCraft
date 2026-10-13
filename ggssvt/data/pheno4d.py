"""Pheno4D, as the reference geometry this project has never had.

Schunck et al., *Pheno4D: a spatio-temporal dataset of maize and tomato plant
point clouds for phenotyping and advanced plant analysis*, PLOS ONE 2021.
Seven maize and seven tomato plants, laser-scanned over two to three weeks, with
roughly 260 million points hand-labelled at organ level.

**Why it matters here.** Every reconstruction claim in this project rests on the
implied bulk density criterion, because no laser scan of our own specimens
exists and correctness therefore cannot be measured directly. §7f goes further
and argues that silhouette IoU ranks our reconstructions *backwards*, an
argument that currently rests on the density screen disagreeing with the metric,
which is an inference rather than a demonstration.

A Pheno4D cloud is a plant whose true shape is known. Render virtual views of it
at our azimuths and camera model, run our carve and our fusion on those views,
and the reconstruction can be scored against the truth directly. The density
criterion can then be checked rather than defended, and the metric inversion
shown rather than inferred.

**The format**, from the files themselves rather than from the paper: whitespace-
separated text, millimetres, one point per line. Files ending ``_a`` carry two
extra integer columns: a semantic label where 0 is soil, 1 is stem and 2 and
above are individual leaves, and an instance label. Files without the suffix are
the same scans unlabelled. The soil points sit at z about 0, so the scanner's
frame already has its origin on the ground plane, which is this dataset's
equivalent of our measured pot rim.

**One caveat that shapes the whole experiment.** A laser scan is a *surface*, not
a solid. At our 12 mm voxels that distinction mostly vanishes: a maize leaf is
well under a millimetre thick, so the voxels a leaf passes through are the plant
-- but it is why the truth here is an occupancy grid built from the points and
not a filled volume. A visual hull's excess over that grid is not a defect of the
comparison; it is the envelope error §7f is about, finally measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset_pheno4d"

MM_TO_M = 1e-3

# Column 4 of a labelled file. 0 is soil; everything above it is plant.
SOIL_LABEL = 0
STEM_LABEL = 1


@dataclass(frozen=True)
class Scan:
    """One labelled scan of one plant on one day, in metres."""

    plant_id: str                 # Maize01, Tomato03, ...
    scan_id: str                  # M01_0325_a
    species: str                  # maize | tomato
    points: np.ndarray            # (N, 3) plant points, metres, z above soil
    semantic: np.ndarray          # (N,) 1 = stem, 2+ = leaf instances
    soil_z_m: float               # where the soil plane sat before centring

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def height_m(self) -> float:
        return float(self.points[:, 2].max()) if self.n_points else 0.0

    @property
    def n_organs(self) -> int:
        return len({int(v) for v in self.semantic if v > STEM_LABEL})


def labelled_scans(root: Path = DATASET_DIR) -> list[Path]:
    """Every labelled scan, sorted. The unlabelled duplicates are skipped."""
    if not root.exists():
        raise FileNotFoundError(
            f"no Pheno4D at {root}. It is free from "
            f"https://www.ipb.uni-bonn.de/data/pheno4d/"
        )
    return sorted(root.glob("*/*_a.txt"))


def latest_per_plant(root: Path = DATASET_DIR) -> list[Path]:
    """The last labelled scan of each plant: the largest, most structured one.

    A growth series is fourteen plants times a dozen days, and reconstructing all
    of it is hours. The final scan of each plant is the hardest case, most
    leaves, most self-occlusion, which is the one worth testing a reconstruction
    against.
    """
    by_plant: dict[str, Path] = {}
    for path in labelled_scans(root):
        by_plant[path.parent.name] = path       # sorted, so the last one wins
    return [by_plant[name] for name in sorted(by_plant)]


def load_scan(path: Path, *, subsample: int | None = None) -> Scan:
    """Read one labelled scan, drop the soil, and put z on the soil plane.

    Args:
        subsample: keep one point in this many. A full scan is over a million
            points and the virtual renderer does not need them all; None keeps
            everything.
    """
    import pandas as pd

    table = pd.read_csv(path, sep=r"\s+", header=None).to_numpy()
    if table.shape[1] < 4:
        raise ValueError(f"{path.name} has no labels; expected a *_a.txt file")

    xyz_mm = table[:, :3].astype(np.float64)
    semantic = table[:, 3].astype(np.int32)

    soil = semantic == SOIL_LABEL
    # The soil plane, from the soil points themselves rather than assumed at
    # zero. The scanner's origin is close to it but not exactly on it.
    soil_z_mm = float(np.median(xyz_mm[soil, 2])) if soil.any() else 0.0

    plant = ~soil
    points = xyz_mm[plant] * MM_TO_M
    points[:, 2] -= soil_z_mm * MM_TO_M
    labels = semantic[plant]

    if subsample and subsample > 1:
        points = points[::subsample]
        labels = labels[::subsample]

    # Centre horizontally on the plant, since our voxel grid is centred on the
    # subject axis rather than on the scanner.
    points[:, 0] -= float(np.median(points[:, 0]))
    points[:, 1] -= float(np.median(points[:, 1]))

    name = path.parent.name
    return Scan(
        plant_id=name,
        scan_id=path.stem,
        species="maize" if name.lower().startswith("maize") else "tomato",
        points=points.astype(np.float64),
        semantic=labels,
        soil_z_m=soil_z_mm * MM_TO_M,
    )


def voxelise(
    points: np.ndarray, *, resolution: int, voxel_size_m: float
) -> np.ndarray:
    """Occupancy on the project's voxel grid: True where a point falls.

    This is the ground truth a reconstruction is scored against. At 12 mm a leaf
    is thinner than a voxel, so the voxels its points pass through are the plant
   : an occupancy grid rather than a filled solid, and the right target for a
    method whose output is also an occupancy grid.
    """
    extent = resolution * voxel_size_m
    origin = np.array([-extent / 2.0, -extent / 2.0, 0.0])
    index = np.floor((points - origin) / voxel_size_m).astype(np.int64)

    inside = np.all((index >= 0) & (index < resolution), axis=1)
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    if inside.any():
        kept = index[inside]
        grid[kept[:, 0], kept[:, 1], kept[:, 2]] = True
    return grid


__all__ = [
    "DATASET_DIR", "MM_TO_M", "SOIL_LABEL", "STEM_LABEL", "Scan",
    "labelled_scans", "latest_per_plant", "load_scan", "voxelise",
]
