"""Where the carve threw away plant the camera had already photographed.

Ten of the Eucalyptus captures and two of the Mango ones were staged on an
inverted pot used as a pedestal, with the plant in a plastic bag on top of it.
The rim detector missed the top of that stack on most of them and fell back to
the configured 0.28 m, so their reported volume is integrated from partway
inside the stand.

That is the smaller half of the problem. The larger half is that the carve stops
at the top of the stand and keeps almost none of the plant: on E001 it stops at
0.456 m while the segmentation reaches 1.257 m, and the ten specimens report
volumes of 3.72 to 4.26 L for masses spanning 0.40 to 0.70 kg, which is the
signature of measuring one object ten times. It is most of what §7l attributes
to the capture batch, because within that batch the reconstruction holds nothing
about the plant for a model to use.

**The camera saw the plant and the segmenter found it.** Back-projecting the
masks puts 24,020 points above the carve on E001, reaching 1.257 m, in a column
whose median radius about the plant axis is 5.8 cm. The carve deletes it: a stem
a couple of centimetres across is thinner than a voxel, most of the twelve
cameras look straight past it and return background, and a voxel survives only
when at most three of twelve dissent.

So the failure belongs to the operator, not to the capture and not to the
segmentation, which is a far more recoverable position than the numbers alone
suggest. This module measures it per specimen rather than asserting it, so the
page and the gallery can mark the affected reconstructions instead of presenting
a stand volume as though it were a plant.

An earlier version tried to locate the top of the stand and threshold against it.
That fights the data, because these pots taper and any fixed share of the base
area is reached partway down the pot rather than at its rim. Comparing where the
carve stops against where the segmentation reaches needs no such threshold and
answers the same question more directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, WORK_DIR, voxel_grid_centres

# How much taller the segmentation has to reach than the carve before the gap is
# worth reporting. Below this it is voxel quantisation at the tip.
HEIGHT_LOST_FLOOR = 0.15

# Masked points above the carve needed before the gap is more than stray returns.
SEEN_POINTS = 500


@dataclass
class Verdict:
    """One specimen: what was reported, and what is actually in it."""

    plant_id: str
    species: str
    mass_kg: float
    rim_m: float
    rim_measured: bool               # False means the 0.28 m fallback was used
    reported_volume_l: float
    carved_top_m: float              # highest voxel the carve kept
    segmented_top_m: float           # highest point the segmentation found
    height_lost_m: float             # the gap between them
    discarded_points: int            # masked points above the carve
    discarded_compactness: float     # share of them within 30 cm of the axis
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def masked_points(cached) -> np.ndarray:
    """The segmentation's own points, back-projected into the world frame.

    The cache keeps masks and depth rather than points, so this recovers what the
    segmenter actually found. It is the evidence that separates "the camera never
    saw the plant" from "the carve discarded it".
    """
    mask = np.asarray(cached.mask)
    depth = np.asarray(cached.depth_m)
    rotation = np.asarray(cached.rotation)
    centre = np.asarray(cached.centre)

    out = []
    for view in range(mask.shape[0]):
        rows, cols = np.nonzero(mask[view] & (depth[view] > 0))
        if rows.size == 0:
            continue
        z = depth[view][rows, cols]
        x = (cols - KINECT_V2.cx) * z / KINECT_V2.fx
        y = (rows + cached.crop_top - KINECT_V2.cy) * z / KINECT_V2.fy
        out.append(np.column_stack([x, y, z]) @ rotation[view].T + centre[view])
    return np.vstack(out) if out else np.zeros((0, 3))


def assess(cached, mass_kg: float) -> Verdict:
    """Measure what the segmentation found against what the carve kept.

    An earlier version of this tried to locate the top of the stand from the
    vertical profile and threshold against it. That fights the data: these pots
    taper, so the cross-section declines smoothly and any fixed share of the base
    is reached partway down the pot rather than at its rim. The comparison below
    needs no such threshold and settles the same question more directly.
    """
    litres = cached.voxel_size_m ** 3 * 1000.0
    heights = voxel_grid_centres()[..., 2]
    occupancy = cached.occupancy

    carved = np.flatnonzero(occupancy.any(axis=(0, 1)))
    carved_top = float(carved.max() + 1) * cached.voxel_size_m if carved.size else 0.0

    points = masked_points(cached)
    if points.shape[0] == 0:
        segmented_top = 0.0
        discarded = 0
    else:
        segmented_top = float(points[:, 2].max())
        discarded = int((points[:, 2] > carved_top).sum())

    reported = float((occupancy & (heights > cached.pot_height_m)).sum()) * litres
    lost = max(segmented_top - carved_top, 0.0)

    # A narrow column of discarded points is a stem the carve could not hold. A
    # wide scatter is background that leaked into the mask, which is a
    # segmentation problem and a different fix.
    if discarded:
        above = points[points[:, 2] > carved_top]
        radius = np.linalg.norm(above[:, :2], axis=1)
        compact = float((radius < 0.30).mean())
    else:
        compact = 0.0

    if lost < HEIGHT_LOST_FLOOR or discarded < SEEN_POINTS:
        verdict = "the carve kept what was segmented"
    elif compact >= 0.8:
        verdict = "a plant was segmented above the carve and discarded"
    else:
        verdict = "points above the carve, but too scattered to be plant"

    return Verdict(
        plant_id=cached.plant_id,
        species=cached.species,
        mass_kg=round(mass_kg, 3),
        rim_m=round(cached.pot_height_m, 3),
        rim_measured=abs(cached.pot_height_m - 0.28) > 1e-6,
        reported_volume_l=round(reported, 3),
        carved_top_m=round(carved_top, 3),
        segmented_top_m=round(segmented_top, 3),
        height_lost_m=round(lost, 3),
        discarded_points=discarded,
        discarded_compactness=round(compact, 3),
        verdict=verdict,
    )


def run(
    *,
    cache_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "pedestal.json",
    verbose: bool = True,
) -> dict:
    """Assess every usable specimen and record which ones report furniture."""
    import csv

    from ..data.preprocess import load_cached, usable_plant_ids

    cache_dir = cache_dir or WORK_DIR / "cache"
    ground_truth = {
        row["plant"]: float(row["mass_kg"])
        for row in csv.DictReader(
            (WORK_DIR / "reports" / "dataset.csv").open(encoding="utf-8"))
    }

    verdicts = []
    for plant_id in sorted(usable_plant_ids(cache_dir)):
        cached = load_cached(plant_id, cache_dir)
        verdicts.append(assess(cached, ground_truth.get(plant_id, float("nan"))))
        if verbose:
            v = verdicts[-1]
            flag = ("  <-- " + v.verdict
                    if v.verdict != "the carve kept what was segmented" else "")
            print(f"  {v.plant_id:6s} rim {v.rim_m:.3f}"
                  f"{'' if v.rim_measured else '*'}  carve stops {v.carved_top_m:5.3f} "
                  f"segmentation reaches {v.segmented_top_m:5.3f}  lost "
                  f"{v.height_lost_m:5.3f} m  {v.discarded_points:7d} pts{flag}")

    affected = [v for v in verdicts
                if v.height_lost_m >= HEIGHT_LOST_FLOOR
                and v.discarded_points >= SEEN_POINTS]
    recoverable = [v for v in affected if v.discarded_compactness >= 0.8]

    report = {
        "note": "a carve that stops well below the segmentation has discarded "
                "plant the camera photographed; where the discarded points form "
                "a narrow column about the axis the loss is the operator's and "
                "is recoverable",
        "height_lost_floor_m": HEIGHT_LOST_FLOOR,
        "n_specimens": len(verdicts),
        "n_flagged": len(affected),
        "n_recoverable": len(recoverable),
        "flagged": [v.plant_id for v in affected],
        "median_height_lost_m": round(
            float(np.median([v.height_lost_m for v in affected])), 3)
        if affected else 0.0,
        "rows": [v.as_dict() for v in verdicts],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print()
        print(f"  {len(affected)} of {len(verdicts)} specimens have plant above "
              f"the carve that the camera photographed")
        print(f"  {len(recoverable)} of those are a narrow column about the axis, "
              f"so the loss is the carve's and not the segmentation's")
    return report


__all__ = [
    "HEIGHT_LOST_FLOOR", "SEEN_POINTS", "Verdict", "assess", "masked_points",
    "run",
]
