"""Where the pot ends and the plant begins, measured per specimen.

Every above-ground quantity in this project -- the volume feature, the mesh
biomass score, the occupancy head's density prior -- depends on knowing which
part of a carved volume is pot and soil rather than plant. Until now that was a
single constant, ``POT_HEIGHT_M = 0.28``, applied to every specimen.

The V001-V008 batch makes that untenable. Their pots weigh 17-32 kg against
0.7-2.2 kg for E001-E020, so they are physically much larger containers, and a
0.28 m cut leaves a slab of pot counted as plant. V001 is the clear case: its
above-ground volume under the constant is 15.4 L for a 1.0 kg shoot, an implied
density of 65 g/L when fresh plant tissue is 300-900 g/L. The number is not
wrong by a little.

A pot is a wide solid of revolution and the plant above it is thin, so the
occupied cross-section collapses at the rim -- for V001, from 307 voxels a slice
to 29 within four slices. That collapse is the measurement this module makes.

The estimate is geometric and needs no calibration, which matters because
``dataset/calib`` is empty. What it cannot do is find a rim that was never
carved: if a specimen's pot merges into the floor plane or the carve is too
sparse to show a collapse, :func:`estimate_pot_height` says so by returning the
fallback rather than inventing a height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import POT_HEIGHT_M, VOXEL_SIZE_M


@dataclass(frozen=True)
class PotEstimate:
    """A per-specimen rim height and the evidence behind it."""

    height_m: float
    confident: bool          # False means the fallback was used
    body_voxels: int         # widest slice of the pot body
    rim_voxels: int          # slice count just above the detected rim
    drop_ratio: float        # rim_voxels / body_voxels; small means a clean rim
    reason: str

    def as_dict(self) -> dict:
        return {
            "pot_height_m": round(self.height_m, 4),
            "confident": self.confident,
            "body_voxels": int(self.body_voxels),
            "rim_voxels": int(self.rim_voxels),
            "drop_ratio": round(self.drop_ratio, 4),
            "reason": self.reason,
        }


def vertical_profile(occupancy: np.ndarray) -> np.ndarray:
    """Occupied voxels per horizontal slice, floor to top."""
    return np.asarray(occupancy, dtype=bool).sum(axis=(0, 1))


def estimate_pot_height(
    occupancy: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    step_ratio: float = 0.35,
    below_slices: int = 3,
    above_slices: int = 5,
    search_max_m: float = 0.9,
    min_body_fraction: float = 0.15,
    min_above_voxels: int = 8,
    fallback_m: float = POT_HEIGHT_M,
) -> PotEstimate:
    """Height of the pot rim above the floor, from the occupancy profile alone.

    A rim is a *step*, not a slope. The test is therefore local: compare the
    median cross-section over the few slices below a candidate height against
    the median over the few slices above it, and call it a rim only when the
    ratio collapses. Two things follow from insisting on a step, and both matter
    on this dataset.

    A smooth taper is rejected rather than cut at an arbitrary point. E001,
    E003 and E008 carve as single cones -- 645 voxels at the floor decaying
    evenly to 140 and then ending -- with no pot/plant boundary visible
    anywhere. An earlier version of this function thresholded against the widest
    slice and duly returned a "rim" for each of them, at 0.432, 0.336 and 0.432
    m, three different answers to a question their geometry does not answer.
    Those specimens now come back ``confident=False``, which is the truthful
    result: the reconstruction does not separate plant from pot.

    Comparing locally also avoids being anchored to the flared base. V001 spills
    3547 voxels across its widest floor slice but its pot body above that is
    around 930, so any fraction-of-the-maximum rule fires far too low -- it put
    V001's rim at 0.336 m, on the shoulder, when the actual collapse is 307
    voxels to 35 at 0.40 m.

    Args:
        occupancy: ``(R, R, R)`` boolean grid, axis 2 up.
        step_ratio: the above/below median ratio at or below which a height
            counts as a rim.
        below_slices: how many slices the "below" median spans.
        above_slices: how many slices the "above" median spans. Wider than
            ``below_slices`` on purpose -- a pot that narrows briefly and widens
            again has a waist, not a rim, and only a median taken over enough
            slices to outlast the waist tells the two apart.
        search_max_m: rims are not looked for above this; a pot taller than
            this is not a pot, and searching further finds canopy gaps instead.
        min_body_fraction: the structure below a candidate must still be this
            fraction of the pot body, which keeps the search from finding a
            "rim" inside a sparse canopy well above the actual pot.
        min_above_voxels: a rim must have plant above it, or it is just the top
            of the object -- where every profile collapses to nothing.
        fallback_m: returned when no step is found.

    Returns:
        A :class:`PotEstimate`. Check ``confident`` before using the height: a
        False means this specimen showed no rim and the caller is getting the
        configured constant back.
    """
    profile = vertical_profile(occupancy).astype(np.float64)
    n = profile.size
    if not profile.any():
        return PotEstimate(fallback_m, False, 0, 0, 0.0, "empty volume")

    search_max = min(n - 1 - above_slices, round(search_max_m / voxel_size_m))
    if search_max <= below_slices:
        return PotEstimate(fallback_m, False, 0, 0, 0.0, "search window too small")

    # The pot body is the sustained width low down, taken as a median so the
    # flared base and any soil spill do not stand in for it.
    body_top = max(below_slices + 1, search_max // 2)
    occupied_low = profile[:body_top][profile[:body_top] > 0]
    body = float(np.median(occupied_low)) if occupied_low.size else 0.0
    if body <= 0:
        return PotEstimate(fallback_m, False, 0, 0, 0.0, "no occupancy near the floor")

    for i in range(below_slices, search_max + 1):
        below = float(np.median(profile[i - below_slices + 1 : i + 1]))
        above = float(np.median(profile[i + 1 : i + 1 + above_slices]))
        if below < min_body_fraction * body:
            continue                      # too thin to still be the pot
        ratio = above / below
        if ratio > step_ratio:
            continue
        if float(profile[i + 1 + above_slices :].sum()) < min_above_voxels:
            break                         # the top of the object, not a rim

        # The medians detect that a step happened nearby, but a window straddling
        # the boundary flips as soon as most of it is past -- which puts the
        # answer a slice or three low. Snap to the real edge: the first slice
        # that is no longer half the body width.
        limit = 0.5 * below
        rim = i
        while rim + 1 < n and profile[rim + 1] >= limit:
            rim += 1
        rim += 1

        return PotEstimate(
            height_m=float(rim * voxel_size_m),
            confident=True,
            body_voxels=round(body),
            rim_voxels=int(profile[rim]) if rim < n else 0,
            drop_ratio=ratio,
            reason=(
                f"cross-section stepped down to {ratio:.0%} of the pot body "
                f"over {below_slices} slices"
            ),
        )

    return PotEstimate(
        height_m=fallback_m,
        confident=False,
        body_voxels=round(body),
        rim_voxels=0,
        drop_ratio=1.0,
        reason="no step found; the profile tapers smoothly, so pot and plant "
               "are not separable in this carve",
    )


__all__ = ["PotEstimate", "estimate_pot_height", "vertical_profile"]
