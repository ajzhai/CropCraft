"""Closing Malik's loop: let the reconstruction refine the segmentation.

The pipeline runs one way. A 2D segmenter decides which pixels are plant, and
those masks are carved into a volume. Nothing ever travels back. Malik et al.
argue that this is the wrong shape: grouping generates candidates, and what comes
out of the later stage should refine the earlier one.

There is a specific reason to expect that to help here. Each mask is decided from
one view in isolation, by a colour threshold or by SAM. A reconstruction is
decided from all twelve at once. So re-projecting the reconstruction into a view
gives that view a second opinion informed by evidence it never saw, and measured
on this data the two opinions genuinely differ in both directions: on E001 the
carve claims 1541 pixels the excess-green mask missed, and rejects 3326 it
included.

**Which direction should win is the interesting question, and it is not obvious.**

Union grows the masks and therefore grows the hull, and the hull is already too
large: that is the project's central finding, 9 of 38 physically plausible.
Intersection shrinks it. Refining against the *fused* reconstruction rather than
the carve is the sharper version of the same idea, because fusion only claims
surfaces a camera measured, so its projection is a tighter set than the hull's.

Nothing here decides that by argument. Each rule is run and scored by implied
bulk density, which is the criterion that does not need a reference geometry, and
whichever wins is reported with the count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, WORK_DIR, Intrinsics, voxel_grid_centres

# How masks from the two sources are combined. Named rather than passed as a
# lambda so the report can say which one produced a number.
RULES = ("original", "union", "intersection", "reconstruction_only")


@dataclass
class RefinementResult:
    """One specimen, one rule, and what it did to the reconstruction."""

    plant_id: str
    rule: str
    mask_px_before: int
    mask_px_after: int
    volume_before_l: float
    volume_after_l: float
    density_before: float
    density_after: float
    plausible_before: bool
    plausible_after: bool

    def as_dict(self) -> dict:
        # A rule can empty a reconstruction, making the implied density
        # infinite. json.dump writes that as bare `Infinity`, which Python
        # reads back and no other parser accepts: it is not valid JSON, and it
        # broke the document generator that reads these reports. null is the
        # honest encoding of "no volume, so no density".
        def finite(value: float, places: int) -> float | None:
            return round(value, places) if np.isfinite(value) else None

        return {
            "plant_id": self.plant_id,
            "rule": self.rule,
            "mask_px_before": self.mask_px_before,
            "mask_px_after": self.mask_px_after,
            "volume_before_l": finite(self.volume_before_l, 3),
            "volume_after_l": finite(self.volume_after_l, 3),
            "density_before": finite(self.density_before, 1),
            "density_after": finite(self.density_after, 1),
            "plausible_before": self.plausible_before,
            "plausible_after": self.plausible_after,
        }


def project_occupancy(
    cached, view: int, *, intrinsics: Intrinsics = KINECT_V2
) -> np.ndarray:
    """Where an occupancy grid lands in one camera, as a boolean image.

    This is the reconstruction's opinion about which pixels are subject, formed
    from every view at once rather than from this one.
    """
    height, width = cached.mask[view].shape
    points = voxel_grid_centres(
        cached.occupancy.shape[0], cached.voxel_size_m
    )[cached.occupancy]

    camera = (points - cached.centre[view]) @ cached.rotation[view]
    depth = camera[:, 2]
    front = depth > 1e-6
    out = np.zeros((height, width), dtype=bool)
    if not front.any():
        return out

    u = camera[front, 0] * intrinsics.fx / depth[front] + intrinsics.cx
    v = camera[front, 1] * intrinsics.fy / depth[front] + intrinsics.cy - cached.crop_top
    col = np.round(u).astype(np.int32)
    row = np.round(v).astype(np.int32)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    out[row[inside], col[inside]] = True
    return out


def combine(original: np.ndarray, projected: np.ndarray, rule: str) -> np.ndarray:
    """Apply one combination rule to a view's two opinions."""
    if rule == "original":
        return original
    if rule == "union":
        return original | projected
    if rule == "intersection":
        return original & projected
    if rule == "reconstruction_only":
        return projected
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def refined_masks(cached, rule: str, *, source=None) -> np.ndarray:
    """Every view's mask, refined against a reconstruction.

    Args:
        cached: the specimen whose masks are refined.
        rule: one of :data:`RULES`.
        source: the specimen supplying the reconstruction. Defaults to ``cached``
            itself; pass the fused cache to refine a carve against the tighter
            depth-based volume, which is the comparison worth making.
    """
    source = source or cached
    return np.stack([
        combine(cached.mask[view].astype(bool), project_occupancy(source, view), rule)
        for view in range(cached.n_views)
    ])


def _uncrop(image: np.ndarray, crop_top: int, dtype) -> np.ndarray:
    """Put a cached, cropped image back at its position in the full frame."""
    from ..config import KINECT_V2

    full = np.zeros((KINECT_V2.height, KINECT_V2.width), dtype=dtype)
    height = min(image.shape[0], KINECT_V2.height - crop_top)
    full[crop_top : crop_top + height, : image.shape[1]] = image[:height]
    return full


def recarve(cached, masks: np.ndarray) -> np.ndarray:
    """Re-run the carve with different masks, everything else identical.

    The cache stores the arrays the carve needs but not the objects it takes, so
    the rig and segmentations are rebuilt around them. Poses come straight from
    the cache rather than being re-estimated, which is the point: the only thing
    that differs between this carve and the original is the masks.
    """
    from ..geometry.carving import carve, largest_connected_component
    from ..geometry.rig import RigSolution, ViewPose
    from ..geometry.segment import ViewSegmentation

    poses, segmentations = {}, {}
    for view, position_id in enumerate(cached.position_ids):
        # The diagnostics are not read by the carve; only rotation and centre
        # are, and both come straight from the cache so the pose is identical
        # to the one that produced the original volume.
        poses[position_id] = ViewPose(
            position_id=position_id,
            azimuth_deg=float("nan"),
            rotation=cached.rotation[view],
            centre=cached.centre[view],
            camera_height_m=float("nan"),
            tilt_deg=float("nan"),
            subject_distance_m=float("nan"),
            floor_inlier_fraction=float("nan"),
        )
        # carve() indexes images at the sensor's native height, and the cache
        # stores them cropped, so they are padded back before being handed over.
        # Without this the row index runs past the end for any voxel projecting
        # into the strip that was cropped away.
        segmentations[position_id] = ViewSegmentation(
            position_id=position_id,
            mask=_uncrop(masks[view], cached.crop_top, bool),
            depth_m=_uncrop(cached.depth_m[view], cached.crop_top, np.float32),
            points_world=np.zeros((0, 3), dtype=np.float32),
            colours=None,
        )

    volume = carve(
        RigSolution(plant_id=cached.plant_id, poses=poses, warnings=[]),
        segmentations,
        plant_id=cached.plant_id,
        resolution=cached.occupancy.shape[0],
        voxel_size_m=cached.voxel_size_m,
    )
    # preprocess keeps only the largest connected component, and skipping it
    # here made a re-carve of the *identical* masks disagree with the cache by
    # more than any refinement rule moved it. Without this step every number
    # below would have measured the difference between two carvers rather than
    # the effect being tested.
    return largest_connected_component(volume.occupancy)


def _above_rim_volume(occupancy: np.ndarray, cached) -> float:
    """Litres above the pot rim, with the rim re-estimated on this occupancy.

    Re-estimated rather than taken from the cache, because a refinement that
    changes the volume can change where the rim sits, and holding the old rim
    fixed would credit or blame the rule for a cut it did not make.
    """
    from ..geometry.pot import estimate_pot_height

    rim = estimate_pot_height(occupancy, voxel_size_m=cached.voxel_size_m)
    centres = voxel_grid_centres(occupancy.shape[0], cached.voxel_size_m)
    above = occupancy & (centres[..., 2] > rim.height_m)
    return float(above.sum() * cached.voxel_size_m ** 3 * 1000.0)


def evaluate(cached, *, source=None, rules: tuple[str, ...] = RULES
             ) -> list[RefinementResult]:
    """Score every rule on one specimen against implied bulk density."""
    from .plausibility import PLAUSIBLE_MAX_KG_M3, PLAUSIBLE_MIN_KG_M3

    mass = float(cached.target_kg)
    before_l = _above_rim_volume(cached.occupancy, cached)
    before_density = mass / (before_l / 1000.0) if before_l > 0 else float("inf")
    plausible = PLAUSIBLE_MIN_KG_M3 <= before_density <= PLAUSIBLE_MAX_KG_M3

    results = []
    for rule in rules:
        masks = refined_masks(cached, rule, source=source)
        occupancy = cached.occupancy if rule == "original" else recarve(cached, masks)
        after_l = _above_rim_volume(occupancy, cached)
        after_density = mass / (after_l / 1000.0) if after_l > 0 else float("inf")

        results.append(RefinementResult(
            plant_id=cached.plant_id,
            rule=rule,
            mask_px_before=int(cached.mask.astype(bool).sum()),
            mask_px_after=int(masks.sum()),
            volume_before_l=before_l,
            volume_after_l=after_l,
            density_before=before_density,
            density_after=after_density,
            plausible_before=plausible,
            plausible_after=bool(
                PLAUSIBLE_MIN_KG_M3 <= after_density <= PLAUSIBLE_MAX_KG_M3
            ),
        ))
    return results


def run(
    plant_ids: list[str] | None = None,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    fused_dir: Path = WORK_DIR / "cache_tsdf",
    refine_from: str = "fused",
    out: Path = WORK_DIR / "reports" / "reciprocity.json",
    verbose: bool = True,
) -> dict:
    """Run the whole comparison and write the report."""
    from ..data.preprocess import load_cached, usable_plant_ids

    plant_ids = plant_ids or usable_plant_ids(cache_dir)
    rows: list[dict] = []

    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        source = cached
        if refine_from == "fused" and (fused_dir / "quality.json").exists():
            try:
                source = load_cached(plant_id, fused_dir)
            except FileNotFoundError:
                source = cached

        # The control, computed per specimen and reported with the result. A
        # re-carve of the unchanged masks must reproduce the cache; when it does
        # not, every rule below is measuring the difference between two carvers
        # rather than the effect under test. That happened once during
        # development and produced a convincing false positive.
        control_l = _above_rim_volume(recarve(cached, cached.mask.astype(bool)), cached)
        original_l = _above_rim_volume(cached.occupancy, cached)
        drift = abs(control_l - original_l) / max(original_l, 1e-9)

        results = evaluate(cached, source=source)
        for r in results:
            row = r.as_dict()
            row["control_drift"] = round(float(drift), 4)
            rows.append(row)
        if verbose:
            summary = "  ".join(
                f"{r.rule[:5]} {r.density_after:7.1f}" for r in results
            )
            print(f"  [{index:2d}/{len(plant_ids)}] {plant_id}  {summary}")

    drifts = [r["control_drift"] for r in rows]
    report = {
        "refine_from": refine_from,
        "control": {
            "max_drift": round(max(drifts), 4) if drifts else None,
            "median_drift": round(float(np.median(drifts)), 4) if drifts else None,
            "note": "re-carving the unchanged masks against the cached volume; "
                    "a rule that moves the volume by less than this drift has "
                    "not been shown to do anything",
        },
        "rows": rows,
        "summary": summarise(rows),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarise(rows: list[dict]) -> dict:
    """Plausible count and median density per rule, which is what decides this."""
    out: dict[str, dict] = {}
    for rule in RULES:
        subset = [r for r in rows if r["rule"] == rule]
        if not subset:
            continue
        densities = [r["density_after"] for r in subset
                     if r["density_after"] is not None
                     and np.isfinite(r["density_after"])]
        out[rule] = {
            "n": len(subset),
            "plausible": sum(1 for r in subset if r["plausible_after"]),
            "median_density": round(float(np.median(densities)), 1) if densities else None,
            "mean_volume_l": round(float(np.mean(
                [r["volume_after_l"] for r in subset
                 if r["volume_after_l"] is not None]
            )), 2) if any(r["volume_after_l"] is not None for r in subset) else None,
        }
    return out


__all__ = ["RULES", "RefinementResult", "combine", "evaluate", "project_occupancy",
           "recarve", "refined_masks", "run", "summarise"]
