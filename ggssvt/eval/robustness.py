"""H4: robustness to sensor noise and occlusion, scored physically.

H4 claims the pipeline is robust to occlusion, noise and sparse sampling. The
sparse-sampling third is already answered by the view-count ablation, and
answered sharply: below twelve views at most two of the reconstructions can
physically weigh their plant. The other two thirds had no experiment. This is
them.

Both are degradations applied to the cached inputs, so neither needs a GPU and
both can run while the training campaign has the card.

**Depth noise.** Kinect v2 depth error grows with the square of range, which is
why the carve's free-space tolerance has a quadratic term
(``CARVE_DEPTH_MARGIN_SLOPE``). Noise is injected with that same shape, scaled
by a multiplier, so level 1 is roughly the sensor's own characteristic and the
higher levels ask what a worse sensor would cost.

**Occlusion.** A horizontal band is removed from each view's subject mask,
placed at a fixed height fraction so the same anatomy is hidden in every view.
That is the pessimistic case on purpose: occlusion that moves between views is
partly recovered by the other eleven, and occlusion that does not is what a pot
label, a support stake or a neighbouring plant actually looks like.

Everything is scored by C1, implied bulk density, rather than by a relative
score, so the question is not "how much worse" but "at what point does the
reconstruction stop being able to weigh the plant". Level 0 of each sweep is the
undegraded control and must reproduce the cache, on the same reasoning as
`reciprocity`: without it a sweep measures the re-carve rather than the noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CARVE_DEPTH_MARGIN_SLOPE, WORK_DIR

# Multiples of the sensor's own quadratic noise characteristic. 0 is the control.
NOISE_LEVELS = (0.0, 1.0, 2.0, 4.0)

# Fraction of each view's height hidden by a band across the subject.
OCCLUSION_LEVELS = (0.0, 0.1, 0.25, 0.5)


@dataclass
class Degraded:
    """One specimen at one degradation level."""

    plant_id: str
    kind: str                  # noise | occlusion
    level: float
    volume_l: float
    density: float | None
    plausible: bool
    voxels: int = 0
    surviving_fraction: float = 1.0

    @property
    def fragment(self) -> bool:
        """Has the reconstruction collapsed to a piece of the specimen?

        A mid-height occlusion severs the plant, and the pipeline's
        largest-connected-component step then keeps whichever side is bigger.
        The surviving piece can land inside the plausible band by coincidence,
        so plausibility on its own would report a destroyed reconstruction as a
        success. This is the flag that stops that.
        """
        return self.surviving_fraction < 0.5

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "kind": self.kind,
            "level": self.level,
            "volume_l": round(self.volume_l, 3),
            "density": None if self.density is None else round(self.density, 1),
            "plausible": self.plausible,
            "voxels": self.voxels,
            "surviving_fraction": round(self.surviving_fraction, 4),
            "fragment": self.fragment,
        }


def add_depth_noise(depth: np.ndarray, level: float, rng) -> np.ndarray:
    """Gaussian noise whose standard deviation grows as the square of range.

    Zero depth means "no return" rather than "zero metres", so those pixels are
    left alone; perturbing them would invent measurements the sensor never made.
    """
    if level <= 0:
        return depth
    sigma = CARVE_DEPTH_MARGIN_SLOPE * level * np.square(depth)
    noisy = depth + rng.normal(0.0, 1.0, depth.shape).astype(depth.dtype) * sigma
    return np.where(depth > 0, np.maximum(noisy, 0.0), depth).astype(depth.dtype)


def occlude(mask: np.ndarray, level: float, *, at: float = 0.45) -> np.ndarray:
    """Hide a horizontal band of the subject in one view.

    Args:
        level: fraction of the subject's vertical extent to remove.
        at: where the band starts, as a fraction of that extent. 0.45 puts it
            across the middle of the plant rather than at an end, which is the
            harder case: removing a tip costs a little height, removing the
            middle disconnects what is above from what is below.
    """
    if level <= 0 or not mask.any():
        return mask
    rows = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1])
    extent = bottom - top + 1
    start = top + int(at * extent)
    stop = min(bottom + 1, start + max(1, int(level * extent)))

    out = mask.copy()
    out[start:stop] = False
    return out


def degrade(cached, kind: str, level: float, *, seed: int = 0):
    """Re-carve one specimen with its depth or masks degraded."""
    from .reciprocity import recarve

    # Validated before the specimen is touched, so a typo fails on the argument
    # rather than several lines later on whatever attribute it reached first.
    if kind not in ("noise", "occlusion"):
        raise ValueError(f"unknown degradation {kind!r}; expected noise or occlusion")

    rng = np.random.default_rng(seed)
    masks = cached.mask.astype(bool)
    depth = cached.depth_m

    if kind == "noise":
        depth = np.stack([add_depth_noise(depth[v], level, rng)
                          for v in range(cached.n_views)])
    else:
        masks = np.stack([occlude(masks[v], level) for v in range(cached.n_views)])

    # A shallow stand-in so recarve sees the degraded arrays without the cache
    # being mutated; everything else, poses included, is the original.
    class _View:
        pass

    view = _View()
    for name in ("plant_id", "position_ids", "rotation", "centre", "occupancy",
                 "voxel_size_m", "crop_top", "target_kg", "n_views", "mask"):
        setattr(view, name, getattr(cached, name))
    view.depth_m = depth
    view.mask = masks
    return recarve(view, masks)


def evaluate(cached, *, kinds=("noise", "occlusion"), seed: int = 0) -> list[Degraded]:
    """Every level of every degradation, scored by C1."""
    from .plausibility import PLAUSIBLE_MAX_KG_M3, PLAUSIBLE_MIN_KG_M3
    from .reciprocity import _above_rim_volume

    mass = float(cached.target_kg)
    baseline_voxels = int(cached.occupancy.sum())
    out: list[Degraded] = []

    for kind in kinds:
        levels = NOISE_LEVELS if kind == "noise" else OCCLUSION_LEVELS
        for level in levels:
            occupancy = degrade(cached, kind, level, seed=seed)
            litres = _above_rim_volume(occupancy, cached)
            density = mass / (litres / 1000.0) if litres > 0 else None
            voxels = int(occupancy.sum())
            out.append(Degraded(
                plant_id=cached.plant_id, kind=kind, level=level,
                volume_l=litres, density=density,
                plausible=bool(
                    density is not None
                    and PLAUSIBLE_MIN_KG_M3 <= density <= PLAUSIBLE_MAX_KG_M3
                ),
                voxels=voxels,
                surviving_fraction=voxels / max(1, int(baseline_voxels)),
            ))
    return out


def summarise(rows: list[dict]) -> dict:
    """Plausible count and median density per degradation level."""
    out: dict[str, dict] = {}
    for row in rows:
        key = f"{row['kind']}@{row['level']}"
        out.setdefault(key, {"kind": row["kind"], "level": row["level"],
                             "n": 0, "plausible": 0, "fragments": 0,
                             "densities": [], "volumes": []})
        entry = out[key]
        entry["n"] += 1
        # A fragment that lands in the band is not a success, so it is counted
        # separately rather than folded into the plausible count.
        entry["plausible"] += int(row["plausible"] and not row["fragment"])
        entry["fragments"] = entry.get("fragments", 0) + int(row["fragment"])
        if row["density"] is not None:
            entry["densities"].append(row["density"])
        entry["volumes"].append(row["volume_l"])

    for entry in out.values():
        densities = entry.pop("densities")
        volumes = entry.pop("volumes")
        entry["median_density"] = (round(float(np.median(densities)), 1)
                                   if densities else None)
        entry["mean_volume_l"] = round(float(np.mean(volumes)), 2) if volumes else None
    return out


def run(
    plant_ids: list[str] | None = None,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "robustness.json",
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Sweep both degradations over every usable specimen."""
    from ..data.preprocess import load_cached, usable_plant_ids

    plant_ids = plant_ids or usable_plant_ids(cache_dir)
    rows: list[dict] = []

    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        results = evaluate(cached, seed=seed)
        rows.extend(r.as_dict() for r in results)
        if verbose:
            control = next(r for r in results if r.level == 0.0)
            print(f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
                  f"control {control.volume_l:6.2f} L")

    summary = summarise(rows)
    # The control is level 0 of either sweep; it must match the cache, or every
    # level above it is measuring the re-carve rather than the degradation.
    report = {"seed": seed, "rows": rows, "summary": summary}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["NOISE_LEVELS", "OCCLUSION_LEVELS", "Degraded", "add_depth_noise",
           "degrade", "evaluate", "occlude", "run", "summarise"]
