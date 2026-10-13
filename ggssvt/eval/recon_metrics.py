"""Distance metrics for reconstruction quality, where ground truth exists.

The project scores reconstructions two ways and both are overlap measures.
Voxel intersection over union against the Pheno4D ground truth, and silhouette
intersection over union against the rendered masks. Silhouette IoU has already
been shown to rank the two operators *backwards* on all fourteen scans, so the
choice of metric here is not a matter of taste; one of the two in use is known
to be wrong on this data.

Overlap is a harsh and slightly misleading measure for plants specifically.
A stem two centimetres across is under two voxels wide on a 12 mm grid, so a
reconstruction that recovers it one voxel to the left scores close to zero
overlap on that stem while being 12 mm from correct. Foliage is mostly thin
structure, so this is not an edge case; it is most of the plant.

So this adds the distance family that the multi-view literature settles on,
in the form DUSt3R reports for DTU (Wang et al., 2024, sec. 4.5):

``accuracy``
    for each reconstructed point, the distance to the nearest true point. How
    much of what was built is real.
``completeness``
    for each true point, the distance to the nearest reconstructed point. How
    much of what is real was built.
``overall``
    the mean of the two, which is the single number those benchmarks rank on.

Reported as medians as well as means, because a handful of stray points from a
mask leak drags a mean a long way and says little about the surface.

Added to that is the F-score at a distance threshold, which is what the Tanks
and Temples benchmark ranks on and which is easier to interpret than a mean
distance: precision is the fraction of built points within the threshold of
truth, recall is the fraction of true points within the threshold of something
built, and the F-score is their harmonic mean. The threshold is stated rather
than tuned, at one voxel, because a reconstruction cannot be asked to do better
than the grid it is built on.

**These are not a substitute for the density screen.** They need reference
geometry, which exists for the fourteen Pheno4D scans and for none of the 36
specimens. The screen remains the only thing that can be applied to the
project's own captures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import VOXEL_SIZE_M, WORK_DIR

# One voxel. Stated in advance and not tuned: a reconstruction on a 12 mm grid
# cannot be asked to place a surface more finely than the grid allows, so this
# is the tightest threshold that is meaningful rather than a chosen one.
F_THRESHOLD_M = VOXEL_SIZE_M


@dataclass
class DistanceMetrics:
    """One reconstruction against one ground truth."""

    n_reconstructed: int
    n_truth: int
    accuracy_mean_mm: float
    accuracy_median_mm: float
    completeness_mean_mm: float
    completeness_median_mm: float
    overall_mean_mm: float
    precision: float
    recall: float
    f_score: float
    threshold_mm: float

    def as_dict(self) -> dict:
        return asdict(self)


def _nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance from every source point to the closest target point."""
    from scipy.spatial import cKDTree

    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros(0)
    return cKDTree(target).query(source, k=1)[0]


def compare(
    reconstructed: np.ndarray,
    truth: np.ndarray,
    *,
    threshold_m: float = F_THRESHOLD_M,
) -> DistanceMetrics:
    """Accuracy, completeness and F-score between two point sets, in metres in."""
    forward = _nearest(reconstructed, truth)      # accuracy
    backward = _nearest(truth, reconstructed)     # completeness

    if forward.size == 0 or backward.size == 0:
        return DistanceMetrics(
            int(reconstructed.shape[0]), int(truth.shape[0]),
            -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0,
            round(threshold_m * 1000, 1))

    precision = float((forward <= threshold_m).mean())
    recall = float((backward <= threshold_m).mean())
    denom = precision + recall

    mm = lambda x: round(float(x) * 1000.0, 2)
    return DistanceMetrics(
        n_reconstructed=int(reconstructed.shape[0]),
        n_truth=int(truth.shape[0]),
        accuracy_mean_mm=mm(forward.mean()),
        accuracy_median_mm=mm(np.median(forward)),
        completeness_mean_mm=mm(backward.mean()),
        completeness_median_mm=mm(np.median(backward)),
        overall_mean_mm=mm((forward.mean() + backward.mean()) / 2.0),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f_score=round(2 * precision * recall / denom, 4) if denom else 0.0,
        threshold_mm=round(threshold_m * 1000, 1),
    )


def occupancy_points(grid: np.ndarray) -> np.ndarray:
    """Centres of the occupied voxels, in world metres."""
    from ..config import voxel_grid_centres

    return voxel_grid_centres()[np.asarray(grid, dtype=bool)]


def run(
    *,
    source: Path = WORK_DIR / "reports" / "virtual_views.json",
    out: Path = WORK_DIR / "reports" / "recon_metrics.json",
    verbose: bool = True,
) -> dict:
    """Score every Pheno4D scan on the distance metrics and compare rankings.

    The comparison against the existing overlap metrics is the point. If the
    distance family ranks the two operators the same way voxel IoU does, the
    extra metric costs nothing and settles an objection. If it ranks them
    differently, the project has been choosing between operators on a measure
    that does not track being close to the truth.
    """
    from ..data.pheno4d import labelled_scans, load_scan
    from .virtual_views import reconstruct

    if not source.exists():
        raise FileNotFoundError(
            f"{source} is missing; run `cli virtual-views` first, which is what "
            "renders the Pheno4D scans through this project's own operators.")

    previous = {row["scan_id"]: row
                for row in json.loads(source.read_text(encoding="utf-8"))["rows"]}

    paths = {path.stem: path for path in labelled_scans()}

    rows = []
    for scan_id in sorted(previous):
        path = paths.get(scan_id)
        if path is None:
            continue
        scan = load_scan(path)
        result = reconstruct(scan, verbose=False)
        truth = np.asarray(scan.points, dtype=np.float64)

        entry = {"scan_id": scan_id,
                 "species": previous[scan_id].get("species", ""),
                 "carve_iou": previous[scan_id]["carve_iou"],
                 "fused_iou": previous[scan_id]["fused_iou"]}
        for name in ("carve", "fused"):
            grid = getattr(result, f"{name}_occupancy", None)
            if grid is None:
                continue
            entry[name] = compare(occupancy_points(grid), truth).as_dict()
        rows.append(entry)
        if verbose and "carve" in entry and "fused" in entry:
            print(f"  {scan_id:12s} carve F {entry['carve']['f_score']:.3f} "
                  f"overall {entry['carve']['overall_mean_mm']:6.1f} mm   "
                  f"fused F {entry['fused']['f_score']:.3f} "
                  f"overall {entry['fused']['overall_mean_mm']:6.1f} mm", flush=True)

    scored = [r for r in rows if "carve" in r and "fused" in r]
    agree = sum(
        (r["carve"]["f_score"] > r["fused"]["f_score"])
        == (r["carve_iou"] > r["fused_iou"]) for r in scored)

    report = {
        "note": "accuracy, completeness and F-score after the form DUSt3R "
                "reports for DTU; thresholds stated in advance at one voxel",
        "caveat": "these need reference geometry, so they apply to the Pheno4D "
                  "scans and to none of the 36 captured specimens; the implied "
                  "density screen remains the only measure that transfers",
        "threshold_mm": round(F_THRESHOLD_M * 1000, 1),
        "n_scans": len(scored),
        "f_score_agrees_with_voxel_iou": agree,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n  F-score and voxel IoU rank the same operator first on "
              f"{agree} of {len(scored)} scans")
        print(f"  wrote {out}")
    return report


__all__ = ["F_THRESHOLD_M", "DistanceMetrics", "compare", "occupancy_points", "run"]
