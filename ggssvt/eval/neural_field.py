"""A neural field as a third reconstruction operator, scored physically.

The project compares two operators: silhouette carving, which recovers the
canopy envelope, and depth fusion, which does not. A neural radiance field is
the obvious third, `transforms.json` is already exported for every specimen, and
`eval/methods.py` lists it as the next candidate worth running.

There is one obstacle, and it is the interesting part rather than an
inconvenience. A NeRF gives a **density** field, not occupancy. Turning it into a
volume needs a threshold, and that threshold has no physical calibration: it is
exactly the free parameter the plausibility criterion exists to remove. Picking
one and reporting the volume it produces would be picking the answer.

So this does not pick one. It sweeps the threshold across orders of magnitude and
asks, for each specimen, **whether any threshold makes the reconstruction
physically able to weigh the plant**. Both outcomes are results:

* No threshold works, on any specimen. The envelope finding generalises from
  silhouette hulls to neural fields, which is a much broader claim than the
  project currently makes.
* Some threshold works, and consistently. That threshold is then a *measured*
  density calibration for this sensor and subject, which nobody publishes,
  because nobody has a mass to calibrate it against.

**The coordinate trap.** Nerfstudio re-centres and rescales the scene at load
time, so a trained field lives in its normalised space and not in metres. The map
back is in ``dataparser_transforms.json`` in the training output directory.
Sampling the metric voxel grid without applying it puts every query in the wrong
place, and the failure looks like an empty reconstruction rather than a
mis-registration. This is the same class of mistake as the OpenCV to OpenGL
convention in the export, which is documented there for the same reason.

The sweep and the scoring are pure NumPy and need nothing installed. Only
:func:`sample_density` imports nerfstudio, which lives in the other conda
environment, so everything here is testable on a machine that cannot train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import VOXEL_RESOLUTION, VOXEL_SIZE_M, WORK_DIR, voxel_grid_centres

# Densities span orders of magnitude, so the sweep is geometric. The range is
# deliberately wider than any published default: the point is to find out
# whether a working threshold exists, not to confirm one.
THRESHOLDS = tuple(float(10.0 ** e) for e in np.arange(-2.0, 3.01, 0.25))


@dataclass
class ThresholdScore:
    """One specimen at one density threshold."""

    plant_id: str
    threshold: float
    voxels: int
    volume_l: float
    density_kg_m3: float | None
    plausible: bool

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "threshold": self.threshold,
            "voxels": self.voxels,
            "volume_l": round(self.volume_l, 4),
            "density_kg_m3": (None if self.density_kg_m3 is None
                              else round(self.density_kg_m3, 1)),
            "plausible": self.plausible,
        }


def sweep(
    density: np.ndarray,
    *,
    plant_id: str,
    mass_kg: float,
    voxel_size_m: float = VOXEL_SIZE_M,
    pot_height_m: float | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> list[ThresholdScore]:
    """Score every threshold by implied bulk density.

    Pure. Give it any density grid on the working volume and it says which
    thresholds, if any, produce a reconstruction that could weigh ``mass_kg``.
    """
    from .plausibility import PLAUSIBLE_MAX_KG_M3, PLAUSIBLE_MIN_KG_M3

    resolution = density.shape[0]
    centres = voxel_grid_centres(resolution, voxel_size_m)
    above = (centres[..., 2] > pot_height_m) if pot_height_m is not None else True

    scores = []
    for threshold in thresholds:
        occupancy = (density > threshold) & above
        voxels = int(occupancy.sum())
        litres = voxels * voxel_size_m ** 3 * 1000.0
        implied = mass_kg / (litres / 1000.0) if litres > 0 else None
        scores.append(ThresholdScore(
            plant_id=plant_id,
            threshold=threshold,
            voxels=voxels,
            volume_l=litres,
            density_kg_m3=implied,
            plausible=bool(
                implied is not None
                and PLAUSIBLE_MIN_KG_M3 <= implied <= PLAUSIBLE_MAX_KG_M3
            ),
        ))
    return scores


def working_thresholds(scores: list[ThresholdScore]) -> list[float]:
    """Thresholds at which this specimen is physically plausible.

    An empty list is the informative case: no setting of the free parameter
    makes the field able to weigh the plant, which is a statement about the
    operator rather than about the threshold.
    """
    return [s.threshold for s in scores if s.plausible]


def consensus(per_specimen: dict[str, list[ThresholdScore]]) -> dict:
    """Is there a single threshold that works for every specimen?

    A calibration is only a calibration if one value serves the whole set. A
    different best threshold per specimen is a fitted parameter wearing a
    calibration's clothes, and would be worth exactly nothing on a new plant.
    """
    if not per_specimen:
        return {"shared": [], "n_specimens": 0}

    sets = [set(working_thresholds(scores)) for scores in per_specimen.values()]
    shared = sorted(set.intersection(*sets)) if all(sets) else []
    per_count = {
        threshold: sum(1 for s in sets if threshold in s) for threshold in THRESHOLDS
    }
    best = max(per_count.items(), key=lambda kv: kv[1], default=(None, 0))

    return {
        "n_specimens": len(per_specimen),
        "shared_thresholds": shared,
        "best_threshold": best[0],
        "best_threshold_covers": best[1],
        "specimens_with_no_working_threshold": sorted(
            pid for pid, scores in per_specimen.items()
            if not working_thresholds(scores)
        ),
        "note": (
            "shared_thresholds are values plausible for every specimen at once. "
            "Empty means no single density calibration serves the set, and a "
            "per-specimen threshold is a fitted parameter rather than a "
            "calibration"
        ),
    }


def density_cache_path(output_dir: Path) -> Path:
    """Where a sampled density grid is kept, beside the training output.

    Sampling needs nerfstudio and a GPU; looking at the result should need
    neither. Caching the grid is what lets `cli show --source neural` work in
    the main environment, on a laptop, after the fact.
    """
    return output_dir / "density_grid.npz"


def load_density(output_dir: Path) -> np.ndarray:
    """Read a previously sampled grid. Raises if it was never sampled."""
    path = density_cache_path(output_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no sampled density at {path}. Run `cli neural-field` in the "
            "cropcraft environment first; it caches the grid so that looking "
            "at it afterwards needs neither nerfstudio nor a GPU."
        )
    with np.load(path) as data:
        return np.asarray(data["density"], dtype=np.float32)


def sample_density(
    output_dir: Path,
    *,
    resolution: int = VOXEL_RESOLUTION,
    voxel_size_m: float = VOXEL_SIZE_M,
    device: str = "cuda",
    cache: bool = True,
) -> np.ndarray:
    """Query a trained Nerfstudio field on the metric voxel grid.

    Args:
        output_dir: a Nerfstudio training output directory, the one holding
            ``config.yml`` and ``dataparser_transforms.json``.

    Raises:
        ImportError: nerfstudio is not installed. It lives in the other conda
            environment, pinned to torch 2.0.1 and Python 3.8, which cannot host
            the transformers version the rest of this project needs.
        FileNotFoundError: the dataparser transform is missing, which would mean
            sampling in the wrong space.
    """
    try:
        import torch
        from nerfstudio.utils.eval_utils import eval_setup
    except ImportError as exc:  # pragma: no cover - the other environment
        raise ImportError(
            "nerfstudio is not installed here. It needs the `cropcraft` "
            "environment; see RUNBOOK.md 'Two environments, not one'."
        ) from exc

    transforms = output_dir / "dataparser_transforms.json"
    if not transforms.exists():
        raise FileNotFoundError(
            f"no dataparser_transforms.json in {output_dir}. Without it the "
            "metric grid cannot be mapped into the field's normalised space, "
            "and every query would land in the wrong place while looking like "
            "an empty reconstruction."
        )

    meta = json.loads(transforms.read_text(encoding="utf-8"))
    # (3, 4) world-to-nerfstudio, then an isotropic scale.
    transform = np.asarray(meta["transform"], dtype=np.float64)
    scale = float(meta["scale"])

    _, pipeline, _, _ = eval_setup(output_dir / "config.yml", test_mode="inference")
    field = pipeline.model.field

    centres = voxel_grid_centres(resolution, voxel_size_m).reshape(-1, 3)
    homogeneous = np.concatenate(
        [centres, np.ones((centres.shape[0], 1))], axis=1
    )
    queried = (homogeneous @ transform.T) * scale

    with torch.no_grad():
        points = torch.from_numpy(queried).float().to(device)
        densities = field.density_fn(points).squeeze(-1).cpu().numpy()

    grid = densities.reshape(resolution, resolution, resolution).astype(np.float32)
    if cache:
        np.savez_compressed(
            density_cache_path(output_dir), density=grid,
            voxel_size_m=voxel_size_m, resolution=resolution,
        )
    return grid


def run(
    outputs: Path,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "neural_field.json",
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """Sweep every trained specimen's field and report whether any threshold works.

    Args:
        outputs: the directory holding one Nerfstudio output per specimen, named
            by plant id.
    """
    from ..data.preprocess import load_cached

    per_specimen: dict[str, list[ThresholdScore]] = {}
    rows: list[dict] = []

    for directory in sorted(p for p in outputs.iterdir() if p.is_dir()):
        plant_id = directory.name
        try:
            cached = load_cached(plant_id, cache_dir)
        except FileNotFoundError:
            if verbose:
                print(f"  {plant_id}: no cache, skipping")
            continue

        try:
            density = load_density(directory)
        except FileNotFoundError:
            density = sample_density(
                directory, resolution=cached.occupancy.shape[0],
                voxel_size_m=cached.voxel_size_m, device=device,
            )
        scores = sweep(
            density, plant_id=plant_id, mass_kg=float(cached.target_kg),
            voxel_size_m=cached.voxel_size_m,
            pot_height_m=cached.pot_height_m,
        )
        per_specimen[plant_id] = scores
        rows.extend(s.as_dict() for s in scores)

        if verbose:
            works = working_thresholds(scores)
            print(f"  {plant_id}: {len(works)} of {len(scores)} thresholds "
                  f"plausible" + (f", {works[0]:.3g} to {works[-1]:.3g}"
                                  if works else " (none)"))

    report = {"rows": rows, "consensus": consensus(per_specimen)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["THRESHOLDS", "ThresholdScore", "consensus", "density_cache_path",
           "load_density", "run", "sample_density", "sweep",
           "working_thresholds"]
