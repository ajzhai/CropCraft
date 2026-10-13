"""Every stage of one specimen's reconstruction, in one frame, for the page.

The pipeline runs segmentation, then carving, then fusion, then meshing, and
until now each stage could only be inspected on its own. That hides the thing
that matters most about this dataset: on 17 of 36 specimens the segmentation
reaches most of a metre higher than the carve does (§7r). Reading that as two
numbers in a table takes an act of imagination. Seeing the plant in one panel and
the stump beside it does not.

All five stages are encoded onto the same voxel grid and in the same byte format
the specimen viewer already decodes, so the page gains a panel rather than a
renderer, and the four are directly comparable because they share a frame.

**These are not five parallel methods, and the panels have to say so.** Reading
them left to right as a sequence of equals is the natural mistake and it is
wrong in two places at once.

The segmentation is the *shared input*: the subject masks from all twelve views
with their depth, back-projected into the world. Nothing to its right sees the
raw frames; they all consume this.

Carving and fusion are two genuinely *alternative operators* on that input, and
they differ in what they will assume about space no camera resolved. Carving
keeps a voxel unless enough views vote it away, so it fills the gaps between
leaves and overestimates. Fusion integrates a truncated signed distance field
from the same depth, so it only accepts surface the sensor returned.

The mesh is *not* a third operator. It is marching cubes over the carve's own
occupancy, so it is the same object drawn as a shell rather than a solid. It
can never contain anything the carve discarded, which is why the two panels
reach the same height on every specimen, and why treating a mesh result as
independent evidence would be double counting.

The surface count is the fifth, and the sharpest way to state what it is: it
back-projects the same masks with the same function the segmentation panel uses
and counts occupied voxels, differing only in view count and grid pitch. At
twelve views it would be the first panel counted differently. There is no
carving and no distance field, so it cannot fill a gap it never observed, and
its volume scales with how many views you give it (§7t).

**Point budgets are deliberate.** The segmentation of one Mango runs past 300,000
points and the page embeds every specimen, so each stage is capped. The cap costs
detail and not shape, and the alternative is a page too heavy to open.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import VOXEL_RESOLUTION, VOXEL_SIZE_M, WORK_DIR, voxel_grid_centres

# Points kept per stage before encoding. Four stages for all 36 specimens go into
# one page that has to open from a file:// URL, and the budget is what decides
# whether it does. Measured: 5000 points costs 2.0 MB, 2500 costs 1.0 MB and 1500
# costs 0.65 MB. At 2500 a stem is still a stem and a hull is still a hull, which
# is the whole job of these panels.
MAX_POINTS = 2500

STAGE_LABELS = {
    "segmentation": ("Segmentation", "what the twelve cameras saw"),
    "carve": ("Silhouette carving", "the visual hull of those masks"),
    "fusion": ("Depth fusion", "the same depth, integrated"),
    "mesh": ("Mesh", "marching cubes over the carve, as a surface"),
    "surface": ("Surface voxels", "after Nombambela (2025), his four views"),
}


@dataclass
class Stage:
    """One stage of one specimen, encoded for the browser."""

    key: str
    title: str
    detail: str
    n_points: int
    top_m: float
    volume_l: float
    # Only the surface-voxel arm reports one, because it is the only operator
    # here whose output is a volume without a fitted regressor behind it.
    density: float
    cloud: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _voxelise(points: np.ndarray) -> np.ndarray:
    """Points onto the project's grid, so every stage shares one frame."""
    from ..data.pheno4d import voxelise

    return voxelise(points, resolution=VOXEL_RESOLUTION, voxel_size_m=VOXEL_SIZE_M)


def _stage(key: str, grid: np.ndarray, *, from_points: bool,
           rim_m: float = 0.0) -> Stage:
    from .dashboard_data import _quantise

    litres = VOXEL_SIZE_M ** 3 * 1000.0
    occupied = np.flatnonzero(grid.any(axis=(0, 1)))
    title, detail = STAGE_LABELS[key]

    # Above the rim, matching the figure the rest of the project reports. The
    # whole-grid count includes the pot and contradicts the note under the panel.
    above = grid & (voxel_grid_centres()[..., 2] > rim_m)
    return Stage(
        key=key,
        title=title,
        detail=detail,
        n_points=int(grid.sum()),
        top_m=round(float(occupied.max() + 1) * VOXEL_SIZE_M, 3)
        if occupied.size else 0.0,
        # A surface is not a volume, so only the solid stages report one.
        volume_l=0.0 if from_points else round(float(above.sum()) * litres, 3),
        density=0.0,
        cloud=_quantise(grid, downsample=1, max_points=MAX_POINTS),
    )


def stages_for(
    plant_id: str,
    *,
    cache_dir: Path,
    fused_dir: Path | None = None,
) -> dict:
    """Encode every stage the caches can supply for one specimen."""
    from ..data.preprocess import load_cached
    from ..geometry.mesh import mesh_from_occupancy
    from .pedestal import masked_points

    cached = load_cached(plant_id, cache_dir)
    stages: list[Stage] = []

    points = masked_points(cached)
    if points.shape[0]:
        stages.append(_stage("segmentation", _voxelise(points), from_points=True))

    stages.append(_stage("carve", cached.occupancy, from_points=False,
                             rim_m=cached.pot_height_m))

    if fused_dir is not None and (fused_dir / "quality.json").exists():
        try:
            fused = load_cached(plant_id, fused_dir)
        except (FileNotFoundError, KeyError):
            fused = None
        if fused is not None:
            stages.append(_stage("fusion", fused.occupancy, from_points=False,
                                 rim_m=cached.pot_height_m))

    mesh = mesh_from_occupancy(cached.occupancy, voxel_size_m=VOXEL_SIZE_M)
    if mesh.n_vertices:
        stages.append(_stage("mesh", _voxelise(mesh.vertices), from_points=True))

    # The surface-voxel operator at his own four views. The panel draws the
    # points on the shared 12 mm display grid so it can be compared with the
    # others; the volume beneath it is his, counted on his own 7 mm grid.
    from .surface_mesh import HIS_AZIMUTHS, masked_points_from, surface_volume
    from .surface_mesh import _view_subset as his_views

    four = masked_points_from(cached, his_views(cached, HIS_AZIMUTHS))
    if four.shape[0]:
        stage = _stage("surface", _voxelise(four), from_points=True)
        _, volume_m3 = surface_volume(four, above_m=cached.pot_height_m)
        stage.volume_l = round(volume_m3 * 1000.0, 3)
        mass = float(cached.target_kg)
        stage.density = round(mass / volume_m3, 1) if volume_m3 > 0 else -1.0
        stages.append(stage)

    heights = voxel_grid_centres()[..., 2]
    litres = VOXEL_SIZE_M ** 3 * 1000.0

    # Where the display splits "pot" from "plant". Where the detector found a
    # rim, that rim. Where it did not, the 0.28 m fallback is useless here: it
    # lies inside the staging and paints 16 cm of plastic as plant.
    #
    # What the profile actually shows on E001, and what it does not. The
    # occupied cross-section tapers from 645 voxels at the floor to 258 by
    # 0.300 m, steps down to about 143 by 0.324 m, then holds flat between 138
    # and 147 until it ends at 0.444 m. The plateau is real and says there is a
    # distinct upper structure of constant width sitting on a tapering base.
    #
    # It does *not* establish where the pot ends and the plant begins, and this
    # module must not imply that it does. `geometry/pot.py` declines to place a
    # rim on these specimens deliberately: its test is a step, and the step here
    # has an above/below median ratio of 0.50 to 0.55 where a rim collapse is
    # 0.35 or less. Its docstring also records that an earlier fraction-of-the-
    # maximum rule, which is exactly the kind of rule that would read a rim off
    # this plateau, returned 0.432, 0.336 and 0.432 m for E001, E003 and E008:
    # three different answers to a question the geometry does not answer.
    #
    # So the constant below is used for what it can support. It is the top of
    # the rigid structure the carve kept, which is the correct place to divide
    # the colouring, because everything above it on these captures is plant that
    # the carve discarded. It is not offered as a pot rim, and no volume is
    # measured from it.
    from .recarve import STAND_TOP_M

    rim_measured = abs(cached.pot_height_m - 0.28) > 1e-6
    split_m = float(cached.pot_height_m) if rim_measured else STAND_TOP_M
    split_source = (
        "the detected pot rim" if rim_measured
        else "the top of the rigid structure the carve kept, because no rim "
             "step was found; not a measured pot rim")

    return {
        "plant_id": plant_id,
        "species": cached.species,
        "mass_kg": round(float(cached.target_kg), 3),
        "rim_m": round(float(cached.pot_height_m), 3),
        "rim_measured": rim_measured,
        "above_rim_l": round(
            float((cached.occupancy & (heights > cached.pot_height_m)).sum()) * litres,
            3),
        "segmented_top_m": round(float(points[:, 2].max()), 3) if points.shape[0] else 0.0,
        "split_m": round(split_m, 3),
        "split_source": split_source,
        "stages": [s.as_dict() for s in stages],
    }


def run(
    *,
    cache_dir: Path | None = None,
    fused_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "stages.json",
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Encode every usable specimen's stages for the project page."""
    from ..data.preprocess import usable_plant_ids

    cache_dir = cache_dir or WORK_DIR / "cache"
    fused_dir = fused_dir or WORK_DIR / "cache_tsdf"

    plant_ids = sorted(usable_plant_ids(cache_dir))
    if limit:
        plant_ids = plant_ids[:limit]

    specimens = []
    for plant_id in plant_ids:
        entry = stages_for(plant_id, cache_dir=cache_dir, fused_dir=fused_dir)
        specimens.append(entry)
        if verbose:
            names = "  ".join(
                f"{s['key'][:4]} {s['top_m']:.2f} m" for s in entry["stages"])
            print(f"  {plant_id:6s} {names}", flush=True)

    report = {
        "note": "every stage on one voxel grid and in the specimen viewer's own "
                "byte format, so the panels share a frame and the page needs "
                "no second renderer",
        "max_points": MAX_POINTS,
        "voxel_size_m": VOXEL_SIZE_M,
        "n_specimens": len(specimens),
        "specimens": specimens,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, separators=(",", ":")), encoding="utf-8")
    if verbose:
        print(f"\n  wrote {out} ({out.stat().st_size // 1024} KB, "
              f"{len(specimens)} specimens)")
    return report


__all__ = ["MAX_POINTS", "STAGE_LABELS", "Stage", "run", "stages_for"]
