"""Can the estimated pot masses be checked against the reconstruction?

Thirty-one of the forty-two captures have a pot mass that was estimated rather
than weighed, and the plant mass reported for them is the total minus that
estimate. If the estimates are wrong the regression targets are wrong, which no
amount of work on the reconstruction can repair.

Eleven pots *were* weighed, all in the V batch, so there is something to
calibrate against. The check is the same implied-density argument §7b uses on the
plant, turned on the pot: mass divided by the volume the reconstruction puts
below the rim. Pot plus medium is a real material with a real density, so a
figure far outside what wet potting medium can weigh means either the mass or the
volume is wrong.

**What it finds.** The weighed pots come out at 312 to 485 kg per cubic metre,
which is what a pot of damp medium should weigh. E011 to E020, estimated, come
out at 250 to 321, agreeing with the weighed ones closely enough that their
estimates need no correction. E001 to E010 and the whole Mango batch come out at
36 to 80, which no potting medium can be.

**But they cannot be corrected from this.** Those are precisely the specimens
staged on an inverted pot used as a pedestal, and the below-rim hull contains
that pedestal. The pedestal was never weighed, because it is not part of the
specimen, so the density is low for a reason that has nothing to do with the
estimate being wrong. Reverse-estimating a pot mass from a volume that includes
unweighed furniture would replace one error with a larger one.

So the answer is: the calibration works where the geometry is clean, it clears
E011 to E020, and it is not identifiable for the ones that raised the question.
Separating pedestal from pot is the same unsolved problem as finding the rim, and
until that is solved the honest position is that E001 to E010 and M001 to M010
have pot masses that cannot be verified either way.

**A separate hazard, worth stating on its own.** In the V batch the pot is 10.5
to 46.4 times the plant, so the plant mass is a small difference between two
large weighings. At 50 g of scale error on each, V002's 500 g plant carries 14%
uncertainty before any reconstruction is attempted. That is a property of how the
measurement was taken, not of any method, and it bounds what any model can score
on that batch.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import VOXEL_SIZE_M, WORK_DIR, voxel_grid_centres

# Damp potting medium in a plastic pot. Below this the number is not a pot.
POT_DENSITY_LO, POT_DENSITY_HI = 150.0, 900.0

# Scale errors to report the target's uncertainty against. The plant mass is a
# difference of two weighings, so each contributes.
SCALE_ERRORS_G = (20.0, 50.0, 100.0)


@dataclass
class PotCheck:
    """One specimen's pot, as weighed or estimated, against its reconstruction."""

    plant_id: str
    source: str                     # measured | estimated
    pot_g: float
    plant_g: float
    total_g: float
    below_rim_l: float
    implied_density: float
    plausible: bool
    pot_to_plant: float
    plant_uncertainty: dict         # scale error in grams -> relative error
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def check(cached, row: dict) -> PotCheck:
    """Score one pot against the volume the reconstruction puts below its rim."""
    heights = voxel_grid_centres()[..., 2]
    litres = VOXEL_SIZE_M ** 3 * 1000.0
    below = float((cached.occupancy & (heights <= cached.pot_height_m)).sum()) * litres

    pot = float(row["pot_weight_g"])
    plant = float(row["net_weight_g"])
    total = float(row["total_fresh_weight_with_pot_g"])
    # Grams over litres is already kilograms per cubic metre; dividing by the
    # litres-to-cubic-metres factor as well gave grams per cubic metre and put
    # every figure out by a thousand.
    density = pot / below if below > 0 else float("inf")
    plausible = bool(POT_DENSITY_LO <= density <= POT_DENSITY_HI)

    # The rim fell back to the configured constant on exactly the specimens that
    # were raised on a stand, and their below-rim hull holds that stand.
    on_a_stand = abs(cached.pot_height_m - 0.28) < 1e-6

    if plausible:
        verdict = "consistent with a pot of damp medium"
    elif on_a_stand or density < POT_DENSITY_LO:
        verdict = ("volume includes an unweighed stand, so the density says "
                   "nothing about the mass")
    else:
        verdict = "denser than a pot of medium can be"

    return PotCheck(
        plant_id=cached.plant_id,
        source=row["pot_weight_source"],
        pot_g=pot,
        plant_g=plant,
        total_g=total,
        below_rim_l=round(below, 2),
        implied_density=round(density, 1) if np.isfinite(density) else -1.0,
        plausible=plausible,
        pot_to_plant=round(pot / plant, 1) if plant > 0 else -1.0,
        plant_uncertainty={
            f"{int(s)}g": round(float(np.sqrt(2) * s / plant), 4) if plant > 0 else -1.0
            for s in SCALE_ERRORS_G
        },
        verdict=verdict,
    )


def calibration(checks: list[PotCheck]) -> dict:
    """How well the weighed pots' mass follows their reconstructed volume."""
    weighed = [c for c in checks if c.source == "measured" and c.below_rim_l > 0]
    if len(weighed) < 3:
        return {"n": len(weighed), "note": "too few weighed pots to calibrate"}

    volume = np.array([c.below_rim_l for c in weighed])
    mass = np.array([c.pot_g for c in weighed])
    slope, intercept = np.polyfit(volume, mass, 1)
    predicted = slope * volume + intercept

    return {
        "n": len(weighed),
        "pearson_r": round(float(np.corrcoef(volume, mass)[0, 1]), 3),
        "slope_g_per_l": round(float(slope), 1),
        "intercept_g": round(float(intercept), 1),
        "residual_rmse_g": round(float(np.sqrt(((mass - predicted) ** 2).mean())), 1),
        "note": "fitted on the weighed pots only; applying it to a specimen whose "
                "below-rim hull contains an unweighed stand would be wrong, which "
                "is every specimen that needs correcting",
    }


def run(
    *,
    cache_dir: Path | None = None,
    ground_truth: Path | None = None,
    out: Path = WORK_DIR / "reports" / "pot_mass.json",
    verbose: bool = True,
) -> dict:
    """Check every pot against its reconstruction and report what can be fixed."""
    from ..data.preprocess import load_cached, usable_plant_ids

    cache_dir = cache_dir or WORK_DIR / "cache"
    ground_truth = ground_truth or (
        Path(__file__).resolve().parents[2] / "dataset" / "ground_truth.csv")
    rows = {r["plant_id"]: r for r in csv.DictReader(
        ground_truth.open(newline="", encoding="utf-8")) if r["plant_id"]}

    checks = []
    for plant_id in sorted(usable_plant_ids(cache_dir)):
        if plant_id not in rows:
            continue
        checks.append(check(load_cached(plant_id, cache_dir), rows[plant_id]))

    implausible = [c for c in checks if not c.plausible]
    correctable = [c for c in implausible
                   if "unweighed stand" not in c.verdict]

    report = {
        "note": "the pot is checked the way section 7b checks the plant, by "
                "implied bulk density; a pot of damp medium cannot weigh less "
                "than about 150 kg per cubic metre",
        "density_band": [POT_DENSITY_LO, POT_DENSITY_HI],
        "n_specimens": len(checks),
        "n_measured": sum(c.source == "measured" for c in checks),
        "n_estimated": sum(c.source == "estimated" for c in checks),
        "n_implausible": len(implausible),
        "n_correctable": len(correctable),
        "calibration": calibration(checks),
        "rows": [c.as_dict() for c in checks],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        for c in checks:
            mark = "" if c.plausible else "   <-- " + c.verdict
            print(f"  {c.plant_id:6s} {c.source:9s} pot {c.pot_g:6.0f} g  "
                  f"below rim {c.below_rim_l:6.2f} L  "
                  f"{c.implied_density:7.1f} kg/m3{mark}")
        cal = report["calibration"]
        print(f"\n  weighed pots: mass follows volume at r = {cal.get('pearson_r')}, "
              f"{cal.get('slope_g_per_l')} g per litre, residual "
              f"{cal.get('residual_rmse_g')} g")
        print(f"  {len(implausible)} of {len(checks)} pots are outside the band, "
              f"{len(correctable)} of them for a reason the reconstruction can fix")
    return report


__all__ = [
    "POT_DENSITY_HI", "POT_DENSITY_LO", "SCALE_ERRORS_G", "PotCheck",
    "calibration", "check", "run",
]
