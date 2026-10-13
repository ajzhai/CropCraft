"""Is a reconstructed volume physically capable of weighing what the plant weighs?

Every biomass method here maps a reconstructed above-ground volume to a mass, and
every one of them has been mediocre. The usual way to report that is a regression
metric, which says the fit is poor without saying why. Dividing the measured mass
by the reconstructed volume says why, in units anyone can check.

Fresh above-ground plant tissue has a bulk density in the region of 300-900 kg
per cubic metre -- lighter than water because of air spaces, heavier than a
canopy's worth of gaps. A visual hull, though, encloses the convex envelope of a
plant and not the plant, and for a leafy canopy the envelope is mostly air. So the
implied density is a direct test of whether a reconstruction is even the right
kind of object:

  - **far below the plausible band**: the volume is an envelope, not a plant. The
    hull has swallowed the space between leaves and branches. Every Mango
    specimen lands here, at 26-77 kg/m^3, which is one to two orders of magnitude
    short and is not something a better regressor can rescue.
  - **far above it**: the reconstruction is missing material -- thin stems that
    were never carved. E019 implies 22,569 kg/m^3, which is to say almost nothing
    of it was reconstructed at all.
  - **inside it**: the volume is at least the right size for the mass, and a
    regression against it is a meaningful thing to fit.

The band is deliberately wide and its edges are conventions, not measurements, so
:func:`classify` reports which side a specimen falls on rather than pretending to
a precision it does not have. What survives that caveat is the direction and the
order of magnitude, and those are what the argument rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fresh (not oven-dry) above-ground tissue. Wide on purpose: the point is to
# separate "plausible" from "wrong by an order of magnitude", not to grade.
PLAUSIBLE_MIN_KG_M3 = 200.0
PLAUSIBLE_MAX_KG_M3 = 1000.0


@dataclass(frozen=True)
class Plausibility:
    """Implied bulk density of one specimen, and what it means."""

    plant_id: str
    mass_kg: float
    volume_m3: float
    density_kg_m3: float
    verdict: str            # plausible | envelope | missing | no volume

    @property
    def plausible(self) -> bool:
        return self.verdict == "plausible"

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "mass_kg": round(self.mass_kg, 4),
            "volume_m3": round(self.volume_m3, 6),
            "density_kg_m3": round(self.density_kg_m3, 1),
            "verdict": self.verdict,
        }


def classify(
    plant_id: str,
    mass_kg: float,
    volume_m3: float,
    *,
    low: float = PLAUSIBLE_MIN_KG_M3,
    high: float = PLAUSIBLE_MAX_KG_M3,
) -> Plausibility:
    """Implied density of one specimen and which side of the band it sits on."""
    if volume_m3 <= 0:
        return Plausibility(plant_id, mass_kg, volume_m3, float("inf"), "no volume")

    density = mass_kg / volume_m3
    if density < low:
        verdict = "envelope"
    elif density > high:
        verdict = "missing"
    else:
        verdict = "plausible"
    return Plausibility(plant_id, mass_kg, volume_m3, density, verdict)


def summarise(results: list[Plausibility]) -> dict:
    """Counts and the median density, for reporting across a batch.

    The median rather than the mean: one specimen implying 22,569 kg/m^3 would
    otherwise set the average for its whole group.
    """
    if not results:
        return {"n": 0}

    finite = np.array(
        [r.density_kg_m3 for r in results if np.isfinite(r.density_kg_m3)]
    )
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    return {
        "n": len(results),
        "n_plausible": counts.get("plausible", 0),
        "verdicts": counts,
        "median_density_kg_m3": round(float(np.median(finite)), 1) if finite.size else None,
        "band_kg_m3": [PLAUSIBLE_MIN_KG_M3, PLAUSIBLE_MAX_KG_M3],
    }


__all__ = [
    "PLAUSIBLE_MAX_KG_M3",
    "PLAUSIBLE_MIN_KG_M3",
    "Plausibility",
    "classify",
    "summarise",
]
