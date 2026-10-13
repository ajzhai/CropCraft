"""The reconstruction metrics, and an honest account of what they can measure.

`eval/metrics.py` has carried Chamfer distance, Hausdorff and HD95, F-score at a
distance threshold, voxel IoU and PSNR since early in the project, and until now
not one of them was ever called. That was not an oversight. Chamfer and HD95
measure distance to a reference, and there is no reference: destructive harvest
produced a mass, not a geometry, and no laser scan, CT or CAD model of any
specimen exists. Pointed at nothing, those metrics report nothing.

Two things they can legitimately do, and this module does both while keeping them
apart, because conflating them would overstate the result.

**Agreement between two reconstructions.** Chamfer, HD95, F-score and voxel IoU
between the carve and the fusion say how far apart the two operators are. That is
a real and useful number and it is *not* accuracy: two methods can agree closely
and both be wrong, which given the envelope result is the likely case.

**Explanatory power against the images.** Project a reconstruction back into each
camera and compare what it predicts against what was measured. Silhouette IoU
against the subject mask, and depth error and PSNR against the sensor's depth.
This does not need a reference because the captured views are the reference.

The second is the closer thing to accuracy, with one caveat that has to travel
with it: these views built the reconstruction, so this is self-consistency rather
than held-out generalisation. A volume that fails to explain the images it was
carved from is definitely wrong; one that explains them may still be an envelope,
because a hull is by construction consistent with every silhouette it was built
from. That asymmetry is the whole reason the plausibility check exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, WORK_DIR, Intrinsics, voxel_grid_centres
from .metrics import hausdorff, reconstruction_metrics, voxel_iou


@dataclass
class Agreement:
    """How far apart two reconstructions of the same specimen are."""

    plant_id: str
    voxel_iou: float
    chamfer_m: float
    f_score: float
    hausdorff_m: float
    hd95_m: float

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "voxel_iou": round(self.voxel_iou, 4),
            "chamfer_m": round(self.chamfer_m, 5),
            "f_score": round(self.f_score, 4),
            "hausdorff_m": round(self.hausdorff_m, 4),
            "hd95_m": round(self.hd95_m, 4),
        }


@dataclass
class Reprojection:
    """How well a reconstruction explains the views it was built from."""

    plant_id: str
    silhouette_iou: float          # against the subject mask, per view, averaged
    depth_mae_m: float             # where both predict a surface
    depth_psnr_db: float
    coverage: float                # share of measured subject pixels explained
    per_view_iou: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "silhouette_iou": round(self.silhouette_iou, 4),
            "depth_mae_m": round(self.depth_mae_m, 5),
            "depth_psnr_db": round(self.depth_psnr_db, 2),
            "coverage": round(self.coverage, 4),
            "per_view_iou": [round(v, 4) for v in self.per_view_iou],
        }


def _points(occupancy: np.ndarray, voxel_size_m: float, resolution: int) -> np.ndarray:
    return voxel_grid_centres(resolution, voxel_size_m)[occupancy]


def agreement(cached_a, cached_b, *, max_points: int = 6000, seed: int = 0) -> Agreement:
    """Chamfer, HD95, F-score and IoU between two reconstructions.

    Subsampled, because the nearest-neighbour terms are quadratic and a carved
    volume can hold 70,000 occupied voxels. Six thousand points either side is
    ample for a distance distribution and keeps this seconds rather than minutes.
    """
    rng = np.random.default_rng(seed)
    a = _points(cached_a.occupancy, cached_a.voxel_size_m, cached_a.occupancy.shape[0])
    b = _points(cached_b.occupancy, cached_b.voxel_size_m, cached_b.occupancy.shape[0])

    def thin(points):
        if points.shape[0] <= max_points:
            return points
        return points[rng.choice(points.shape[0], max_points, replace=False)]

    a_thin, b_thin = thin(a), thin(b)
    metrics = reconstruction_metrics(
        a_thin, b_thin,
        predicted_grid=cached_a.occupancy, truth_grid=cached_b.occupancy,
    )
    distances = hausdorff(a_thin, b_thin)

    return Agreement(
        plant_id=cached_a.plant_id,
        voxel_iou=float(voxel_iou(cached_a.occupancy, cached_b.occupancy)),
        chamfer_m=float(metrics.chamfer_m),
        f_score=float(metrics.f_score),
        hausdorff_m=float(distances["hausdorff"]),
        hd95_m=float(distances["hd95"]),
    )


def reproject(
    cached, *, intrinsics: Intrinsics = KINECT_V2, depth_tolerance_m: float = 0.05
) -> Reprojection:
    """Render the reconstruction into every camera and score it against the data.

    Silhouette IoU is the reconstruction's 2D footprint against the subject mask.
    Depth error compares the nearest occupied voxel along each ray against the
    measured depth, over pixels where both have a surface, so a hole in the
    reconstruction lowers coverage rather than inflating the error.
    """
    occupancy = cached.occupancy
    resolution = occupancy.shape[0]
    points = _points(occupancy, cached.voxel_size_m, resolution)
    height, width = cached.depth_m.shape[1:]

    ious, abs_errors, explained, measured_total = [], [], 0, 0

    for view in range(cached.n_views):
        cam = (points - cached.centre[view]) @ cached.rotation[view]
        z = cam[:, 2]
        front = z > 1e-6
        if not front.any():
            ious.append(0.0)
            continue

        u = cam[front, 0] * intrinsics.fx / z[front] + intrinsics.cx
        v = cam[front, 1] * intrinsics.fy / z[front] + intrinsics.cy - cached.crop_top
        col = np.round(u).astype(np.int32)
        row = np.round(v).astype(np.int32)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        if not inside.any():
            ious.append(0.0)
            continue

        col, row, depth_of = col[inside], row[inside], z[front][inside]

        # Nearest surface per pixel: the painter's algorithm, in reverse.
        rendered_depth = np.full((height, width), np.inf, dtype=np.float64)
        order = np.argsort(-depth_of)
        rendered_depth[row[order], col[order]] = depth_of[order]
        rendered = np.isfinite(rendered_depth)

        mask = cached.mask[view].astype(bool)
        union = rendered | mask
        ious.append(float((rendered & mask).sum() / union.sum()) if union.any() else 0.0)

        measured = cached.depth_m[view]
        both = rendered & mask & (measured > 0)
        measured_total += int((mask & (measured > 0)).sum())
        explained += int(both.sum())
        if both.any():
            error = np.abs(rendered_depth[both] - measured[both])
            abs_errors.append(error[error < depth_tolerance_m * 10])

    errors = np.concatenate(abs_errors) if abs_errors else np.zeros(1)
    # PSNR over the depth range the sensor works in, so the decibel figure is
    # comparable between specimens rather than scaled by each one's own extent.
    span = 4.0
    mse = float(np.mean(errors ** 2))
    db = float(10.0 * np.log10(span ** 2 / mse)) if mse > 0 else float("inf")

    return Reprojection(
        plant_id=cached.plant_id,
        silhouette_iou=float(np.mean(ious)) if ious else 0.0,
        depth_mae_m=float(errors.mean()),
        depth_psnr_db=db,
        coverage=float(explained / max(measured_total, 1)),
        per_view_iou=[float(v) for v in ious],
    )


def run(
    plant_ids: list[str] | None = None,
    *,
    carve_dir: Path = WORK_DIR / "cache",
    fused_dir: Path = WORK_DIR / "cache_tsdf",
    out: Path = WORK_DIR / "reports" / "reconstruction_quality.json",
    verbose: bool = True,
) -> dict:
    """Score both operators, and how far apart they are."""
    from ..data.preprocess import load_cached, usable_plant_ids

    plant_ids = plant_ids or usable_plant_ids(carve_dir)
    have_fused = (fused_dir / "quality.json").exists()

    rows = {"reprojection": {"carve": [], "fused": []}, "agreement": []}
    for index, plant_id in enumerate(plant_ids, start=1):
        carved = load_cached(plant_id, carve_dir)
        rows["reprojection"]["carve"].append(reproject(carved).as_dict())

        line = f"  [{index:2d}/{len(plant_ids)}] {plant_id}"
        line += f"  carve IoU {rows['reprojection']['carve'][-1]['silhouette_iou']:.3f}"

        if have_fused:
            fused = load_cached(plant_id, fused_dir)
            rows["reprojection"]["fused"].append(reproject(fused).as_dict())
            rows["agreement"].append(agreement(carved, fused).as_dict())
            line += f"  fused IoU {rows['reprojection']['fused'][-1]['silhouette_iou']:.3f}"
            line += f"  IoU(a,b) {rows['agreement'][-1]['voxel_iou']:.3f}"
        if verbose:
            print(line)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def summarise(rows: dict) -> dict:
    """Means over specimens, for the tables."""
    def mean(items, key):
        values = [r[key] for r in items if np.isfinite(r[key])]
        return round(float(np.mean(values)), 4) if values else None

    out = {}
    for operator, items in rows["reprojection"].items():
        if items:
            out[operator] = {
                "n": len(items),
                "silhouette_iou": mean(items, "silhouette_iou"),
                "depth_mae_m": mean(items, "depth_mae_m"),
                "depth_psnr_db": mean(items, "depth_psnr_db"),
                "coverage": mean(items, "coverage"),
            }
    if rows["agreement"]:
        out["carve_vs_fused"] = {
            "n": len(rows["agreement"]),
            "voxel_iou": mean(rows["agreement"], "voxel_iou"),
            "chamfer_m": mean(rows["agreement"], "chamfer_m"),
            "f_score": mean(rows["agreement"], "f_score"),
            "hd95_m": mean(rows["agreement"], "hd95_m"),
            "hausdorff_m": mean(rows["agreement"], "hausdorff_m"),
        }
    return out


__all__ = ["Agreement", "Reprojection", "agreement", "reproject", "run", "summarise"]
