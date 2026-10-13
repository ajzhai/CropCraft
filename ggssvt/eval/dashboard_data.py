"""Assemble everything the research dashboard needs into one JSON payload.

One pass over the caches produces the whole page: voxel clouds for the 3D
viewer, per-specimen quality and mesh diagnostics, the method comparison, and
the leave-one-out predictions behind the scatter plot.

The horizontal slices the page shows are *not* precomputed. The voxel cloud is
already embedded for the 3D view, and slicing it in the browser costs nothing,
so shipping slice images as well would double the payload to show the same
information twice.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import POT_HEIGHT_M, VOXEL_SIZE_M, WORK_DIR, voxel_grid_centres


def _batch_of(plant_id: str) -> str | None:
    """Which collection batch a Eucalyptus specimen belongs to.

    Batch, not species prefix. E001-E010 and E011-E020 were collected two weeks
    apart and sit at different sizes, which is the confound; V001-V008 are
    Eucalyptus as well and were collected specifically to span both, so counting
    them would be wrong and excluding them would overstate the problem.
    """
    if not plant_id[1:].isdigit():
        return None
    index = int(plant_id[1:])
    if plant_id.startswith("V"):
        return "V001-V008"
    if plant_id.startswith("E"):
        return "E001-E010" if index <= 10 else "E011-E020"
    return None


def _architecture() -> list:
    """The methodology diagrams, listed from the factories that draw them."""
    from .architecture import manifest

    return manifest()


def _references() -> dict:
    """The bibliography, generated rather than kept as markup."""
    from .references import payload

    return payload()


def _quantise(occupancy: np.ndarray, downsample: int = 2, max_points: int = 18000,
              voxel_size_m: float = VOXEL_SIZE_M) -> dict:
    """Occupied voxels as base64 zlib-compressed uint8 triples.

    The payload carries its own metres-per-byte. It used to be inferred in the
    browser as ``resolution * 0.024 / 255``, which happens to land within half a
    percent when the caller downsamples by two, and is exactly twice the truth
    when it does not. The stage panels do not downsample, so every height they
    reported was doubled and the pot-and-plant split fell above the whole plant,
    painting stem, leaves and pot alike as canopy.

    Both halves of the old guess were unsound and only cancelled by luck: the
    byte step is an *integer* division of 255 by the resolution, so the top of
    the byte range is never reached, and the 0.024 pitch is the downsampled
    grid's rather than this one's. Deriving a physical scale from a display
    resolution was the mistake. The encoder knows the answer exactly, so it
    states it and the browser stops guessing.
    """
    resolution = occupancy.shape[0]
    if downsample > 1:
        trimmed = resolution - (resolution % downsample)
        blocks = occupancy[:trimmed, :trimmed, :trimmed].reshape(
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
            trimmed // downsample, downsample,
        )
        occupancy = blocks.any(axis=(1, 3, 5))
        resolution = occupancy.shape[0]

    index = np.array(np.nonzero(occupancy)).T.astype(np.int64)
    if index.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(index.shape[0], max_points, replace=False)
        index = index[np.sort(keep)]

    step = 255 // max(1, resolution - 1)
    scaled = np.clip(index * step, 0, 255).astype(np.uint8)
    return {
        "resolution": int(resolution),
        # Height of one byte, after any downsampling and the integer byte step.
        "metres_per_byte": voxel_size_m * downsample / step,
        "data": base64.b64encode(zlib.compress(scaled.tobytes(), 9)).decode("ascii"),
    }


@dataclass
class DashboardPayload:
    """Everything the page embeds."""

    specimens: list[dict]
    methods: list[dict]
    summary: dict
    notes: dict
    # Results that arrive from their own reports rather than from the caches.
    # Each is absent on a machine that has not run it, and the page omits the
    # section rather than showing an empty frame.
    reconstruction: dict | None = None
    resolution: dict | None = None
    pedestal: dict | None = None
    pot_mass: dict | None = None
    stages: dict | None = None
    filmstrip: dict | None = None
    references: dict | None = None
    architecture: list | None = None
    backbone_viz: dict | None = None
    dino_probe: dict | None = None
    recon_metrics: dict | None = None
    batch_holdout: dict | None = None
    external: dict | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "specimens": self.specimens,
                "methods": self.methods,
                "summary": self.summary,
                "notes": self.notes,
                "reconstruction": self.reconstruction,
                "resolution": self.resolution,
                "pedestal": self.pedestal,
                "pot_mass": self.pot_mass,
                "stages": self.stages,
                "filmstrip": self.filmstrip,
                "references": self.references,
                "architecture": self.architecture,
                "backbone_viz": self.backbone_viz,
                "dino_probe": self.dino_probe,
                "recon_metrics": self.recon_metrics,
                "batch_holdout": self.batch_holdout,
                "external": self.external,
            },
            separators=(",", ":"),
        )


def build_payload(
    *,
    cache_dirs: dict[str, Path] | None = None,
    verbose: bool = True,
) -> DashboardPayload:
    """Collect specimens, reconstructions, mesh metrics and biomass results."""
    from ..data.preprocess import load_cached, load_quality, usable_plant_ids
    from .baselines import load_features
    from .mesh_baseline import evaluate_with_mesh
    from .methods import UNSUPPORTED, describe
    from .methods import cache_dirs as available_cache_dirs
    from .metrics import paired_bootstrap_difference
    from .plausibility import classify, summarise
    from .progress import summarise as summarise_progress
    from .progress import survey

    cache_dirs = cache_dirs or available_cache_dirs()
    if "geometric" not in cache_dirs:
        raise RuntimeError("the geometric cache is required; run preprocess first")

    primary = cache_dirs["geometric"]
    plant_ids = usable_plant_ids(primary)

    if verbose:
        print(f"Building payload for {len(plant_ids)} specimens")

    results, mesh_table = evaluate_with_mesh(plant_ids, cache_dir=primary, verbose=verbose)
    features = load_features(plant_ids, primary)
    targets = np.array([f.target_kg for f in features])

    quality = {name: load_quality(path) for name, path in cache_dirs.items()}

    specimens: list[dict] = []
    for position, plant_id in enumerate(plant_ids):
        entry: dict = {
            "id": plant_id,
            "clouds": {},
            "quality": {},
            "predictions": {},
        }
        for name, path in cache_dirs.items():
            try:
                cached = load_cached(plant_id, path)
            except FileNotFoundError:
                continue
            # The sparse-view hulls balloon to fifteen times the volume, so they
            # carry many more occupied voxels and compress worse. They are on the
            # page to show that ballooning, which reads fine at a third of the
            # point budget, and the full budget would quadruple the page weight.
            budget = 18000 if name in ("geometric", "sam3d") else 6000
            entry["clouds"][name] = _quantise(cached.occupancy, max_points=budget)
            entry.setdefault("species", cached.species)
            entry.setdefault("target_kg", round(float(cached.target_kg), 3))
            # Rim, volume and implied density all belong to the *method*, not to
            # the specimen: a fused reconstruction has a different profile, so a
            # different rim, so a different above-rim volume and density. Keeping
            # them per specimen showed the carve's numbers beside the fusion's
            # point cloud, which is the sort of quiet contradiction a reader
            # notices and cannot resolve.
            above = cached.occupancy & (
                voxel_grid_centres()[..., 2] > cached.pot_height_m
            )
            above_volume = float(above.sum()) * cached.voxel_size_m ** 3
            check = classify(plant_id, float(cached.target_kg), above_volume)
            per_method = {
                "pot_height_m": round(cached.pot_height_m, 3),
                "pot_confident": bool(cached.pot.confident),
                "above_rim_l": round(above_volume * 1000.0, 2),
                "density_kg_m3": (
                    round(check.density_kg_m3, 1)
                    if np.isfinite(check.density_kg_m3) else None
                ),
                "density_verdict": check.verdict,
            }
            # The primary method's values stay on the entry as well, because the
            # summary and the specimen list are not method-specific.
            if name == "geometric":
                entry.update(per_method)

            q = quality[name].get(plant_id)
            if q is not None:
                entry["quality"][name] = {
                    **per_method,
                    "coverage": round(q.surface_coverage, 3),
                    "agreement": round(q.multiview_agreement, 3),
                    "volume_l": round(q.volume_m3 * 1000, 2),
                    # Computed at preprocess time against the global constant.
                    # `above_rim_l` on the entry is the per-specimen figure, and
                    # is the one the density is derived from, do not show both.
                    "above_ground_l": round(q.above_ground_volume_m3 * 1000, 2),
                    "height_m": round(q.height_m, 3),
                    "usable": bool(q.is_usable()),
                }

        mesh = mesh_table.get(plant_id, {})
        entry["mesh"] = {
            k: (round(v, 5) if isinstance(v, float) and np.isfinite(v) else None)
            for k, v in mesh.items()
        }
        for method, (_, predictions) in results.items():
            entry["predictions"][method] = round(float(predictions[position]), 3)

        specimens.append(entry)

    reference = "geometric features"
    methods: list[dict] = []
    for name, (metrics, predictions) in sorted(
        results.items(), key=lambda kv: kv[1][0].rmse_kg
    ):
        row = {
            "name": name,
            "rmse_kg": round(metrics.rmse_kg, 3),
            "mae_kg": round(metrics.mae_kg, 3),
            "mare_pct": round(metrics.mare * 100, 1),
            "r2": round(metrics.r2, 3),
        }
        if name != reference and reference in results:
            d = paired_bootstrap_difference(
                predictions, results[reference][1], targets, n_resamples=2000
            )
            row["vs_reference"] = {
                "difference": round(d["difference"], 3),
                "low": round(d["low"], 3),
                "high": round(d["high"], 3),
                "resolved": bool(d["high"] < 0 or d["low"] > 0),
            }
        methods.append(row)

    # The batch confound, computed rather than asserted. Grouped by collection
    # batch rather than by the "E" prefix: V001-V008 are Eucalyptus too, and
    # they are the batch that weakens this, so filtering them out would report
    # the confound as worse than it now is.
    eucalyptus = [
        (p, t)
        for p, t in zip(plant_ids, targets)
        if _batch_of(p) is not None
    ]
    batches: dict[str, list[float]] = {}
    for plant_id, target in eucalyptus:
        batches.setdefault(_batch_of(plant_id), []).append(float(target))

    masses = np.array([t for _, t in eucalyptus], dtype=float)
    means = np.array([np.mean(batches[_batch_of(p)]) for p, _ in eucalyptus])
    spread = ((masses - masses.mean()) ** 2).sum()
    batch_r2 = (
        float(1.0 - ((masses - means) ** 2).sum() / spread) if spread > 0 else 0.0
    )

    summary = {
        "n_specimens": len(plant_ids),
        "n_views": 12,
        "methods_available": [k for k in describe() if k in cache_dirs],
        "method_info": {k: v for k, v in describe().items() if k in cache_dirs},
        "methods_excluded": UNSUPPORTED,
        "segmenters": sorted(cache_dirs),   # kept: older readers of this payload
        "species": sorted({s["species"] for s in specimens}),
        "mass_range_kg": [round(float(targets.min()), 2), round(float(targets.max()), 2)],
        "pot_height_m": POT_HEIGHT_M,   # fallback only; specimens carry their own
        "plausibility": summarise(
            [
                classify(
                    s["id"],
                    s.get("target_kg", 0.0),
                    (s.get("target_kg", 0.0) / s["density_kg_m3"])
                    if s.get("density_kg_m3")
                    else 0.0,
                )
                for s in specimens
            ]
        ),
        "batch_confound_r2": round(batch_r2, 3),
        "batch_means_kg": {
            k: round(float(np.mean(v)), 3) for k, v in batches.items() if v
        },
        "reference_method": reference,
        "progress": [s.as_dict() for s in survey()],
        "progress_summary": summarise_progress(survey()),
    }

    notes = {
        "calibration": (
            "dataset/calib is empty. No ChArUco calibration was captured, so all "
            "camera poses are estimated from the depth data itself."
        ),
        "target": (
            "Ground truth is as-collected fresh mass, not oven-dry above-ground "
            "biomass. Pot mass is measured for V001-V008 and estimated for the "
            "rest; against the measured eight the estimates run 10.9% light "
            "(sd 1.8), so E and M net masses are overstated by roughly that "
            "much and more where the pot dominates the total."
        ),
        "plausibility": (
            "Measured mass over reconstructed above-ground volume gives an "
            "implied bulk density. Fresh tissue is 300-900 kg/m3, and only 8 of "
            "36 specimens land inside a generous 200-1000 band. Most fall far "
            "below it: the visual hull encloses the air between leaves, so what "
            "is being measured is the canopy envelope and not the plant."
        ),
        "confound": (
            "Batch membership alone explains more of the Eucalyptus mass variance "
            "than any method achieves. The comparison measures size-class "
            "separation, not mass estimation among comparable plants."
        ),
    }

    def report(name: str):
        path = WORK_DIR / "reports" / name
        return (json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else None)

    return DashboardPayload(
        specimens=specimens, methods=methods, summary=summary, notes=notes,
        resolution=report("resolution.json"),
        pedestal=report("pedestal.json"),
        pot_mass=report("pot_mass.json"),
        stages=report("stages.json"),
        filmstrip=report("filmstrip.json"),
        references=_references(),
        architecture=_architecture(),
        backbone_viz=report("backbone_viz.json"),
        dino_probe=report("dino_probe.json"),
        recon_metrics=report("recon_metrics.json"),
        reconstruction=report("reconstruction_clouds.json"),
        batch_holdout=report("batch_holdout.json"),
        external=report("external_lettuce.json"),
    )


def write_payload(
    out_path: Path = WORK_DIR / "reports" / "dashboard_payload.json", **kwargs
) -> Path:
    payload = build_payload(**kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload.to_json(), encoding="utf-8")
    return out_path


__all__ = ["DashboardPayload", "build_payload", "write_payload"]
