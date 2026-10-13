"""Can the carve be made to keep a thin stem without keeping the background?

`eval/pedestal.py` shows that 17 of 36 reconstructions stop well below what the
segmentation found, and that on 14 of them the discarded points are a narrow
column about the plant axis rather than mask leak. That places the failure in the
carve rule:

    occupied = (carve_votes <= max_carve_votes) and
               (informative_views >= min_informative_views)

with the defaults deriving from the view count, six and three on twelve views. A
stem two centimetres across is thinner than a voxel, so most cameras look past it
and return the background behind, which votes the voxel free. Four such votes
delete it.

**The obvious fix does not work on its own.** Raising `max_carve_votes` alone
lets the background in: at six votes of twelve, E001 comes back at 336 L, which
is the working volume rather than a plant. The two thresholds have to move
together, because tolerating dissent is only safe when the voxel is one that
several cameras actually had an opinion about.

**The criterion is fixed before the sweep runs**, and it is the project's
existing one rather than a new one invented to suit the answer: a reconstruction
passes when its implied bulk density falls inside 200 to 1000 kg per cubic metre
(§7b). A setting is judged on how many specimens it brings inside that band
across the *whole* set, not on the flagged ones, because a setting that rescues
E001 by flooding a specimen that already passes has not helped. Ties break toward
the setting closest to the current default, so the change stays as small as the
evidence allows.

**Neither repair works alone.** A first run of this sweep scored every setting at
zero, because it measured volume above the rim the cache carries, and on exactly
the specimens under test that rim is the 0.28 m fallback sitting inside the
stand. The stand was counted as plant, so no carve setting could bring the
density inside the band. `STAND_TOP_M` overrides it, and the two defects have to
be repaired together or neither shows any benefit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, VOXEL_SIZE_M, WORK_DIR, voxel_grid_centres

# The screen, unchanged from section 7b. Fixed here before any setting is run.
DENSITY_LO, DENSITY_HI = 200.0, 1000.0

# The region worth searching. The current default is (3, 6); below it nothing
# survives that does not already, and far above it the working volume fills.
VOTE_GRID = (3, 4, 5, 6)
INFORMATIVE_GRID = (6, 7, 8, 9)

# Where the stand ends on the captures that were raised on one. Measured from the
# vertical profile of the reconstructions rather than assumed: the solid block
# runs to about here on every specimen staged this way.
STAND_TOP_M = 0.44

# A spread of specimens rather than only the broken ones, so a setting that
# rescues a seedling by flooding a mango is caught by the same sweep.
DEFAULT_SUBSET = (
    "E001", "E005", "E008",     # flagged, thin Eucalyptus on a stand
    "M008",                     # flagged, Mango on a stand
    "E002", "E014",             # already inside the band, as positive controls
    "V004", "V005",             # weighed pots, already inside the band
)


@dataclass
class Attempt:
    """One carve setting on one specimen."""

    plant_id: str
    max_carve_votes: int
    min_informative_views: int
    volume_l: float
    top_m: float
    density: float
    plausible: bool

    def as_dict(self) -> dict:
        return asdict(self)


def rebuild(cached):
    """Rig and segmentations from a cached specimen, ready for `carve`.

    The cache stores frames cropped by `crop_top` rows at each end while `carve`
    projects through the uncropped intrinsics, so the rows have to be put back
    before the two agree on which pixel a voxel lands in.
    """
    from ..geometry.rig import RigSolution, ViewPose
    from ..geometry.segment import ViewSegmentation

    mask = np.asarray(cached.mask)
    top = cached.crop_top
    bottom = KINECT_V2.height - mask.shape[1] - top
    pad = lambda a, fill: np.pad(a, ((0, 0), (top, bottom), (0, 0)),
                                 constant_values=fill)
    masks = pad(mask, False)
    depths = pad(np.asarray(cached.depth_m).astype(np.float32), 0.0)
    rotation = np.asarray(cached.rotation)
    centre = np.asarray(cached.centre)

    poses, segmentations = {}, {}
    for index, position in enumerate(cached.position_ids):
        poses[position] = ViewPose(
            position_id=position,
            azimuth_deg=float(index * 30),
            rotation=rotation[index],
            centre=centre[index],
            camera_height_m=float(centre[index][2]),
            tilt_deg=0.0,
            subject_distance_m=float(np.linalg.norm(centre[index][:2])),
            floor_inlier_fraction=1.0,
        )
        segmentations[position] = ViewSegmentation(
            position_id=position, mask=masks[index], depth_m=depths[index],
            points_world=np.zeros((0, 3), dtype=np.float32), colours=None,
        )
    return RigSolution(plant_id=cached.plant_id, poses=poses, warnings=[]), segmentations


def attempt(cached, mass_kg: float, votes: int, informative: int,
            rim_m: float | None = None) -> Attempt:
    """Carve one specimen at one setting and score it against the screen.

    ``rim_m`` overrides the rim the cache carries. It matters: the specimens this
    sweep is trying to rescue are the ones whose rim detection failed, so scoring
    them above the 0.28 m fallback counts the stand as plant and no carve setting
    can bring the density inside the band. Fixing the carve without fixing the rim
    changes nothing, and the sweep reports zero at every setting when run that
    way. The two defects have to be repaired together.
    """
    from ..geometry.carving import carve, largest_connected_component

    rig, segmentations = rebuild(cached)
    occupancy = largest_connected_component(carve(
        rig, segmentations, plant_id=cached.plant_id,
        max_carve_votes=votes, min_informative_views=informative).occupancy)

    heights = voxel_grid_centres()[..., 2]
    litres = VOXEL_SIZE_M ** 3 * 1000.0
    rim = cached.pot_height_m if rim_m is None else rim_m
    volume = float((occupancy & (heights > rim)).sum()) * litres
    occupied = np.flatnonzero(occupancy.any(axis=(0, 1)))
    density = mass_kg / (volume / 1000.0) if volume > 0 else float("inf")

    return Attempt(
        plant_id=cached.plant_id,
        max_carve_votes=votes,
        min_informative_views=informative,
        volume_l=round(volume, 3),
        top_m=round(float(occupied.max() + 1) * VOXEL_SIZE_M, 3)
        if occupied.size else 0.0,
        density=round(density, 1) if np.isfinite(density) else -1.0,
        plausible=bool(DENSITY_LO <= density <= DENSITY_HI),
    )


def run(
    *,
    subset: tuple[str, ...] = DEFAULT_SUBSET,
    cache_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "recarve.json",
    verbose: bool = True,
) -> dict:
    """Sweep the two thresholds and report which setting satisfies the screen."""
    import csv

    from ..data.preprocess import load_cached

    cache_dir = cache_dir or WORK_DIR / "cache"
    ground_truth = {
        row["plant"]: float(row["mass_kg"])
        for row in csv.DictReader(
            (WORK_DIR / "reports" / "dataset.csv").open(encoding="utf-8"))
    }
    cached = {pid: load_cached(pid, cache_dir) for pid in subset}

    # Where the rim detector fell back to the constant, score above the top of
    # the stand instead. Without this the stand is counted as plant and the
    # sweep cannot distinguish any setting from any other.
    rims = {pid: (STAND_TOP_M if abs(c.pot_height_m - 0.28) < 1e-6 else None)
            for pid, c in cached.items()}

    attempts: list[Attempt] = []
    for votes in VOTE_GRID:
        for informative in INFORMATIVE_GRID:
            passed = 0
            for pid in subset:
                result = attempt(cached[pid], ground_truth[pid], votes,
                                 informative, rims.get(pid))
                attempts.append(result)
                passed += result.plausible
            if verbose:
                print(f"  votes {votes}, informative {informative}: "
                      f"{passed}/{len(subset)} inside the density band", flush=True)

    # Best setting, with ties broken toward the current default so the change is
    # no larger than the evidence requires.
    scores: dict[tuple[int, int], int] = {}
    for a in attempts:
        key = (a.max_carve_votes, a.min_informative_views)
        scores[key] = scores.get(key, 0) + int(a.plausible)
    best = max(scores.items(),
               key=lambda kv: (kv[1], -abs(kv[0][0] - 3) - abs(kv[0][1] - 6)))

    report = {
        "note": "criterion fixed before the sweep: implied bulk density inside "
                "200 to 1000 kg/m3, scored over a spread of specimens rather "
                "than the flagged ones alone",
        "density_band": [DENSITY_LO, DENSITY_HI],
        "subset": list(subset),
        "current_setting": {"max_carve_votes": 3, "min_informative_views": 6,
                            "passed": scores.get((3, 6), 0)},
        "best_setting": {"max_carve_votes": best[0][0],
                         "min_informative_views": best[0][1], "passed": best[1]},
        "scores": [{"max_carve_votes": k[0], "min_informative_views": k[1],
                    "passed": v} for k, v in sorted(scores.items())],
        "attempts": [a.as_dict() for a in attempts],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        cur, bst = report["current_setting"], report["best_setting"]
        print(f"\n  current ({cur['max_carve_votes']}, "
              f"{cur['min_informative_views']}): {cur['passed']}/{len(subset)}")
        print(f"  best    ({bst['max_carve_votes']}, "
              f"{bst['min_informative_views']}): {bst['passed']}/{len(subset)}")
    return report


__all__ = [
    "DENSITY_HI", "DENSITY_LO", "INFORMATIVE_GRID", "STAND_TOP_M", "VOTE_GRID",
    "Attempt",
    "attempt", "rebuild", "run",
]
