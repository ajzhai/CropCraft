"""H2: viewpoint consistency, measured on a view the reconstruction never saw.

H2 claims geometry-grounded models "show higher consistency across viewpoints".
The project has a re-projection score already, in `reconstruction_quality`, and
it does not measure that. It projects a reconstruction back into the twelve views
it was built from, which is self-consistency: a visual hull is consistent with
every silhouette it was carved from by construction, which is exactly why that
metric ranks the carve above the fusion while physical plausibility ranks it
below.

Generalisation needs a view held out. Reconstruct from eleven, predict the
twelfth, and compare against what the sensor measured at that azimuth. The
reconstruction has never seen it, so agreement there is earned rather than
guaranteed.

**The number H2 is actually about is the gap.** In-sample agreement is high for
any operator; what separates a reconstruction that has captured the subject from
one that has memorised its inputs is how far the score falls when the view is
withheld. A hull should fall a long way, because every silhouette it was given
constrained it and the missing one did not. A reconstruction carrying real
geometry should fall less. That difference, per specimen, is what this reports.

Twelve re-carves per specimen, so about twenty-five minutes for the set on one
CPU core, and no GPU at any point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, WORK_DIR, Intrinsics


@dataclass
class ViewScore:
    """One held-out view, scored against what the sensor measured there."""

    plant_id: str
    view: int
    in_sample_iou: float
    held_out_iou: float
    in_sample_depth_mae_m: float
    held_out_depth_mae_m: float

    @property
    def iou_gap(self) -> float:
        """How much agreement is lost when the view is withheld.

        This is the quantity H2 is about. A large gap means the reconstruction
        was fitted to its inputs rather than to the subject.
        """
        return self.in_sample_iou - self.held_out_iou

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "view": self.view,
            "in_sample_iou": round(self.in_sample_iou, 4),
            "held_out_iou": round(self.held_out_iou, 4),
            "iou_gap": round(self.iou_gap, 4),
            "in_sample_depth_mae_m": round(self.in_sample_depth_mae_m, 5),
            "held_out_depth_mae_m": round(self.held_out_depth_mae_m, 5),
        }


def _subset(cached, keep: list[int]):
    """A stand-in specimen carrying only the kept views.

    Shallow, so nothing in the cache is mutated, and the poses are the originals
    rather than being re-estimated from eleven views. Re-estimating them would
    change two things at once and the gap would no longer be attributable to the
    missing view.
    """

    class _Subset:
        pass

    out = _Subset()
    out.plant_id = cached.plant_id
    out.position_ids = [cached.position_ids[v] for v in keep]
    out.rotation = cached.rotation[keep]
    out.centre = cached.centre[keep]
    out.depth_m = cached.depth_m[keep]
    out.mask = cached.mask[keep]
    out.occupancy = cached.occupancy
    out.voxel_size_m = cached.voxel_size_m
    out.crop_top = cached.crop_top
    out.target_kg = cached.target_kg
    out.n_views = len(keep)
    return out


def _render(occupancy, cached, view: int, *, intrinsics: Intrinsics):
    """Nearest surface per pixel, as a mask and a depth image.

    The mask alone is not enough here: a reconstruction can put the right
    silhouette at the wrong distance, and on a held-out view that is exactly the
    failure worth catching.
    """
    from ..config import voxel_grid_centres

    height, width = cached.mask[view].shape
    points = voxel_grid_centres(occupancy.shape[0], cached.voxel_size_m)[occupancy]

    camera = (points - cached.centre[view]) @ cached.rotation[view]
    z = camera[:, 2]
    front = z > 1e-6
    rendered = np.full((height, width), np.inf)
    if not front.any():
        return np.zeros((height, width), bool), rendered

    u = camera[front, 0] * intrinsics.fx / z[front] + intrinsics.cx
    v = camera[front, 1] * intrinsics.fy / z[front] + intrinsics.cy - cached.crop_top
    col = np.round(u).astype(np.int32)
    row = np.round(v).astype(np.int32)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    col, row, depth_of = col[inside], row[inside], z[front][inside]

    # Painter's algorithm in reverse: far first, so near voxels overwrite.
    order = np.argsort(-depth_of)
    rendered[row[order], col[order]] = depth_of[order]
    return np.isfinite(rendered), rendered


def _score_against(occupancy, cached, view: int, *, intrinsics: Intrinsics) -> tuple:
    """Silhouette IoU and depth error of one occupancy grid in one camera."""
    predicted, rendered = _render(occupancy, cached, view, intrinsics=intrinsics)
    measured = cached.mask[view].astype(bool)

    union = predicted | measured
    iou = float((predicted & measured).sum() / union.sum()) if union.any() else 0.0

    depth = cached.depth_m[view]
    # Only where the reconstruction claims a surface and the sensor returned
    # one; elsewhere there is nothing to compare.
    both = predicted & measured & (depth > 0)
    mae = (float(np.abs(rendered[both] - depth[both]).mean())
           if both.any() else float("nan"))
    return iou, mae, int(both.sum())


def evaluate(cached, *, intrinsics: Intrinsics = KINECT_V2) -> list[ViewScore]:
    """Hold out each view in turn and score the reconstruction there."""
    from .reciprocity import recarve

    scores: list[ViewScore] = []
    for view in range(cached.n_views):
        keep = [v for v in range(cached.n_views) if v != view]
        reduced = _subset(cached, keep)
        held_out_occupancy = recarve(reduced, reduced.mask.astype(bool))

        in_iou, in_mae, _ = _score_against(
            cached.occupancy, cached, view, intrinsics=intrinsics
        )
        out_iou, out_mae, _ = _score_against(
            held_out_occupancy, cached, view, intrinsics=intrinsics
        )
        scores.append(ViewScore(
            plant_id=cached.plant_id, view=view,
            in_sample_iou=in_iou, held_out_iou=out_iou,
            in_sample_depth_mae_m=in_mae, held_out_depth_mae_m=out_mae,
        ))
    return scores


def summarise(rows: list[dict]) -> dict:
    """Means over every held-out view of every specimen."""
    if not rows:
        return {}

    def mean(key):
        return round(float(np.mean([r[key] for r in rows])), 4)

    return {
        "n_views_scored": len(rows),
        "n_specimens": len({r["plant_id"] for r in rows}),
        "in_sample_iou": mean("in_sample_iou"),
        "held_out_iou": mean("held_out_iou"),
        "iou_gap": mean("iou_gap"),
        "relative_drop": round(
            float(np.mean([r["iou_gap"] for r in rows]))
            / max(1e-9, float(np.mean([r["in_sample_iou"] for r in rows]))), 4
        ),
        "note": (
            "held_out_iou is agreement with a view the reconstruction never "
            "saw; in_sample_iou is agreement with the views it was built from. "
            "The gap is what H2 is about. A hull is consistent with its own "
            "silhouettes by construction, so its in-sample score is not "
            "evidence of anything"
        ),
    }


def run(
    plant_ids: list[str] | None = None,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "viewpoint.json",
    verbose: bool = True,
) -> dict:
    """Score every specimen's every view, held out in turn."""
    from ..data.preprocess import load_cached, usable_plant_ids

    plant_ids = plant_ids or usable_plant_ids(cache_dir)
    rows: list[dict] = []

    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        scores = evaluate(cached)
        rows.extend(s.as_dict() for s in scores)
        if verbose:
            gap = float(np.mean([s.iou_gap for s in scores]))
            held = float(np.mean([s.held_out_iou for s in scores]))
            print(f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
                  f"held-out IoU {held:.3f}  gap {gap:+.3f}")

    report = {"rows": rows, "summary": summarise(rows)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = ["ViewScore", "evaluate", "run", "summarise"]
