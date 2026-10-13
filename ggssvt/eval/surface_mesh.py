"""A second operator: volume as occupied surface voxels, after Nombambela (2025).

Odwa Nombambela's EPR402 report, *Plant Mass Estimation Using 3D Modelling*
(University of Pretoria, November 2025, same study leader as this project),
takes a different route from either operator here. Four Kinect v2 views thirty
degrees apart are back-projected, filtered, smoothed and registered by ICP, and
the volume is then the count of occupied voxels in the *registered surface point
cloud* at 7 mm, `n_occupied * voxel^3`. There is no carving and no signed
distance field: the plant's volume is the space its measured surface passes
through.

That is worth having as a third arm, because it fails differently. A visual hull
fills the gaps between leaves and comes out several times too large (§7p); a
surface count cannot fill a gap it never observed.

**But the volume it reports is a property of the sampling, not of the plant.**
Run on our specimens at his four views it admits 13 of 36 against the carve's 8;
run on the same specimens at our twelve, it admits 7. The measured ratio between
the two is a median of 2.00, range 1.67 to 2.40: doubling the views roughly
doubles the reported volume, because every extra view lays down more surface
points and more points fall in more voxels. Nothing about the plant changed.

That is not fatal to his result, and it is worth saying why. His protocol fixes
the view count at four for every specimen, so the bias is a constant scale factor
across his whole set and a regressor fitted on those features absorbs it. It does
mean the number is not a volume in any transferable sense, and that two studies
using this operator at different view counts cannot be compared.

**This is a reimplementation, not his code.** The operator is one line of
arithmetic once stated, this repository is public, and his report is unpublished
coursework. Reproduced from the method as described, and verified against his own
output: his plant 1 reports 0.00349071 m³, which is exactly 10,177 voxels at
7 mm, so the arithmetic here matches his to the voxel.

**One difference has to be stated whenever the two are compared.** His ground
truth is the plant *and its pot together* ("place plant (including pot) on
scale", report p. 72), and no pot is subtracted anywhere in his pipeline. Ours is
plant mass net of pot. His 40 masses span 0.85 to 1.75 kg because the pot
dominates them. The operator transfers; the target does not, and his trained
regressor cannot be applied to our specimens for that reason alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres

# His grid, from integration.py: reconstruction_params['voxel_size'] = 0.007.
SURFACE_VOXEL_M = 0.007

# The screen this project applies to every operator, unchanged (§7b).
DENSITY_LO, DENSITY_HI = 200.0, 1000.0

# His capture protocol is four views at ninety degrees. Ours is twelve at thirty,
# so the same specimen can be measured both ways and the view count separated
# from the operator.
HIS_AZIMUTHS = (0, 90, 180, 270)


@dataclass
class SurfaceResult:
    """One specimen under the surface-voxel operator."""

    plant_id: str
    species: str
    mass_kg: float
    rim_m: float
    n_views: int
    occupied_voxels: int
    volume_l: float
    density: float
    plausible: bool
    top_m: float

    def as_dict(self) -> dict:
        return asdict(self)


def surface_volume(
    points: np.ndarray,
    *,
    voxel_size_m: float = SURFACE_VOXEL_M,
    above_m: float = 0.0,
) -> tuple[int, float]:
    """Occupied voxels, and their volume, for a set of surface points.

    The whole operator. Points above ``above_m`` are binned onto a grid of the
    given pitch and the occupied bins are counted, which is what his
    `calculate_volume` does when handed grid data.
    """
    kept = points[points[:, 2] > above_m] if points.size else points
    if kept.shape[0] == 0:
        return 0, 0.0

    index = np.floor(kept / voxel_size_m).astype(np.int64)
    occupied = int(np.unique(index, axis=0).shape[0])
    return occupied, occupied * voxel_size_m ** 3


def _view_subset(cached, azimuths: tuple[int, ...] | None) -> list[int]:
    """View indices closest to the requested azimuths, or all of them."""
    n_views = np.asarray(cached.mask).shape[0]
    if azimuths is None:
        return list(range(n_views))
    step = 360.0 / n_views
    return sorted({round(a / step) % n_views for a in azimuths})


def masked_points_from(cached, views: list[int]) -> np.ndarray:
    """Back-project the subject masks of the chosen views into the world."""
    from ..config import KINECT_V2

    mask = np.asarray(cached.mask)
    depth = np.asarray(cached.depth_m)
    rotation = np.asarray(cached.rotation)
    centre = np.asarray(cached.centre)

    out = []
    for view in views:
        rows, cols = np.nonzero(mask[view] & (depth[view] > 0))
        if rows.size == 0:
            continue
        z = depth[view][rows, cols]
        x = (cols - KINECT_V2.cx) * z / KINECT_V2.fx
        y = (rows + cached.crop_top - KINECT_V2.cy) * z / KINECT_V2.fy
        out.append(np.column_stack([x, y, z]) @ rotation[view].T + centre[view])
    return np.vstack(out) if out else np.zeros((0, 3))


def assess(cached, mass_kg: float, azimuths: tuple[int, ...] | None) -> SurfaceResult:
    """Run the operator on one specimen at one view count."""
    views = _view_subset(cached, azimuths)
    points = masked_points_from(cached, views)

    occupied, volume_m3 = surface_volume(points, above_m=cached.pot_height_m)
    volume_l = volume_m3 * 1000.0
    density = mass_kg / volume_m3 if volume_m3 > 0 else float("inf")

    above = points[points[:, 2] > cached.pot_height_m] if points.size else points
    return SurfaceResult(
        plant_id=cached.plant_id,
        species=cached.species,
        mass_kg=round(mass_kg, 3),
        rim_m=round(float(cached.pot_height_m), 3),
        n_views=len(views),
        occupied_voxels=occupied,
        volume_l=round(volume_l, 3),
        density=round(density, 1) if np.isfinite(density) else -1.0,
        plausible=bool(DENSITY_LO <= density <= DENSITY_HI),
        top_m=round(float(above[:, 2].max()), 3) if above.shape[0] else 0.0,
    )


def run(
    *,
    cache_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "surface_mesh.json",
    verbose: bool = True,
) -> dict:
    """Score the surface-voxel operator against carve and fusion on our data."""
    import csv

    from ..config import VOXEL_SIZE_M
    from ..data.preprocess import load_cached, usable_plant_ids

    cache_dir = cache_dir or WORK_DIR / "cache"
    ground_truth = {
        row["plant"]: float(row["mass_kg"])
        for row in csv.DictReader(
            (WORK_DIR / "reports" / "dataset.csv").open(encoding="utf-8"))
    }

    heights = voxel_grid_centres()[..., 2]
    litres = VOXEL_SIZE_M ** 3 * 1000.0

    rows, carve_rows = [], []
    for plant_id in sorted(usable_plant_ids(cache_dir)):
        cached = load_cached(plant_id, cache_dir)
        mass = ground_truth.get(plant_id, float("nan"))

        for label, azimuths in (("four views", HIS_AZIMUTHS), ("twelve views", None)):
            result = assess(cached, mass, azimuths)
            entry = result.as_dict()
            entry["protocol"] = label
            rows.append(entry)

        # The carve on the same specimen, for the comparison that matters.
        carve_volume = float(
            (cached.occupancy & (heights > cached.pot_height_m)).sum()) * litres
        carve_density = mass / (carve_volume / 1000.0) if carve_volume > 0 else float("inf")
        carve_rows.append({
            "plant_id": plant_id,
            "volume_l": round(carve_volume, 3),
            "density": round(carve_density, 1) if np.isfinite(carve_density) else -1.0,
            "plausible": bool(DENSITY_LO <= carve_density <= DENSITY_HI),
        })

        if verbose:
            four = next(r for r in rows[-2:] if r["protocol"] == "four views")
            twelve = next(r for r in rows[-2:] if r["protocol"] == "twelve views")
            print(f"  {plant_id:6s} carve {carve_volume:7.2f} L "
                  f"({carve_density:7.0f})   surface 4v {four['volume_l']:6.2f} L "
                  f"({four['density']:7.0f})   12v {twelve['volume_l']:6.2f} L "
                  f"({twelve['density']:7.0f})"
                  + ("   surface plausible" if twelve["plausible"] else ""))

    def passes(protocol: str) -> int:
        return sum(r["plausible"] for r in rows if r["protocol"] == protocol)

    report = {
        "method": "surface voxel count, after Nombambela (2025)",
        "note": "volume is the count of occupied voxels in the registered surface "
                "point cloud, not a carved hull or a signed distance field; "
                "reimplemented from the method as described, verified against his "
                "plant 1 at 10,177 voxels of 7 mm",
        "caveat": "his ground truth is plant and pot together and ours is plant "
                  "alone, so the operator transfers but the target does not, and "
                  "his trained regressor cannot be applied to our specimens",
        "voxel_size_m": SURFACE_VOXEL_M,
        "density_band": [DENSITY_LO, DENSITY_HI],
        "n_specimens": len(carve_rows),
        "note_view_dependence": "volume scales with the number of views: the "
                                "median ratio between twelve views and four is "
                                "2.00, so the figure is a property of the "
                                "sampling and not of the plant",
        "carve_passes": sum(r["plausible"] for r in carve_rows),
        "surface_passes_four_views": passes("four views"),
        "surface_passes_twelve_views": passes("twelve views"),
        "rows": rows,
        "carve": carve_rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        n = len(carve_rows)
        print(f"\n  inside the density band, of {n}:")
        print(f"    silhouette carving          {report['carve_passes']}")
        print(f"    surface voxels, four views  {report['surface_passes_four_views']}")
        print(f"    surface voxels, twelve      {report['surface_passes_twelve_views']}")
    return report


__all__ = [
    "DENSITY_HI", "DENSITY_LO", "HIS_AZIMUTHS", "SURFACE_VOXEL_M",
    "SurfaceResult", "assess", "masked_points_from", "run", "surface_volume",
]
