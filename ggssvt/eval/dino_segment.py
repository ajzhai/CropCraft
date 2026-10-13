"""Can a foundation model separate plant from pot where excess-green cannot?

The geometric segmenter thresholds 2G-R-B inside a cylinder, which works when
the plant is greener than everything around it and fails when the pot fills most
of the silhouette. E001 to E010 are the failure: they carve as single tapering
cones, and the rim detector refuses on nine of the ten because there is no step
in the profile to find. Whether the plant is genuinely inseparable there, or
merely inseparable *by colour*, is an open question this answers.

The test lifts frozen DINOv2 patch features onto the carved points as DITR does,
clusters them, and asks whether the resulting split lines up with height. It is
deliberately unsupervised: there are no per-point labels in this dataset and
inventing some by thresholding colour would make the comparison circular.

What counts as success is stated before the numbers, because it would otherwise
be tempting to accept whatever the clustering produced. A useful separation
puts the two clusters at clearly different heights, assigns most of the volume
near the floor to one and most of the volume above the rim to the other, and
does so on the batch where colour fails. A clustering that splits each specimen
into top and bottom halves regardless of content would satisfy the first of
those and none of the rest, so height separation alone is reported but not
treated as the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres
from ..geometry.dino_lift import cluster, lift, order_by_height


@dataclass
class SegmentationResult:
    """What the lifted clustering did on one specimen."""

    plant_id: str
    n_points: int
    n_pooled: int                     # points at least two views saw
    height_gap_m: float               # mean height of the upper cluster minus lower
    lower_fraction: float             # share of points in the lower cluster
    agreement_with_rim: float         # fraction matching the rim-based split
    rim_confident: bool
    upper_above_rim: float            # share of the upper cluster above the rim
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "plant_id": self.plant_id,
            "n_points": self.n_points,
            "n_pooled": self.n_pooled,
            "height_gap_m": round(self.height_gap_m, 4),
            "lower_fraction": round(self.lower_fraction, 4),
            "agreement_with_rim": round(self.agreement_with_rim, 4),
            "rim_confident": self.rim_confident,
            "upper_above_rim": round(self.upper_above_rim, 4),
            "notes": self.notes,
        }


def _feature_maps(cached, backbone) -> np.ndarray:
    """DINOv2 patch features for every view, as ``(V, gh, gw, D)``."""
    import torch

    rgb = torch.from_numpy(cached.rgb).float().permute(0, 3, 1, 2) / 255.0
    maps = []
    with torch.no_grad():
        for index in range(cached.n_views):
            tokens, grid_h, grid_w = backbone.patch_tokens(rgb[index : index + 1])
            maps.append(tokens.reshape(grid_h, grid_w, -1).cpu().numpy())
    return np.stack(maps).astype(np.float32)


def segment_specimen(cached, backbone, *, k: int = 2) -> SegmentationResult:
    """Lift, cluster and score one specimen."""
    centres = voxel_grid_centres()
    points = centres[cached.occupancy]
    notes: list[str] = []

    if points.shape[0] < 32:
        return SegmentationResult(cached.plant_id, points.shape[0], 0,
                                  0.0, 0.0, 0.0, cached.pot.confident, 0.0,
                                  ["too few occupied voxels to cluster"])

    lifted = lift(
        points,
        _feature_maps(cached, backbone),
        cached.rotation.astype(np.float64),
        cached.centre.astype(np.float64),
        cached.depth_m,
        crop_top=cached.crop_top,
    )
    labels = order_by_height(cluster(lifted, k=k), lifted.heights)
    pooled = int((labels >= 0).sum())
    if pooled < 32:
        return SegmentationResult(cached.plant_id, points.shape[0], pooled,
                                  0.0, 0.0, 0.0, cached.pot.confident, 0.0,
                                  ["too few points survived the occlusion test"])

    lower = labels == 0
    upper = labels == k - 1
    gap = float(lifted.heights[upper].mean() - lifted.heights[lower].mean()) if upper.any() else 0.0

    # The rim split is the incumbent, not ground truth. Agreement says whether
    # the feature clustering found the same boundary, and disagreement is only
    # interesting where the rim estimate is itself confident.
    rim = cached.pot_height_m
    rim_split = lifted.heights > rim
    scored = labels >= 0
    agreement = float((rim_split[scored] == upper[scored]).mean())

    above = float((lifted.heights[upper] > rim).mean()) if upper.any() else 0.0
    if not cached.pot.confident:
        notes.append("rim fell back to the constant; agreement is not meaningful")

    return SegmentationResult(
        plant_id=cached.plant_id,
        n_points=int(points.shape[0]),
        n_pooled=pooled,
        height_gap_m=gap,
        lower_fraction=float(lower.sum() / max(pooled, 1)),
        agreement_with_rim=agreement,
        rim_confident=bool(cached.pot.confident),
        upper_above_rim=above,
        notes=notes,
    )


def run(
    plant_ids: list[str] | None = None,
    *,
    cache_dir: Path = WORK_DIR / "cache",
    variant: str = "base",
    backbone_kind: str = "dinov2",
    out: Path = WORK_DIR / "reports" / "dino_segment.json",
    verbose: bool = True,
) -> list[SegmentationResult]:
    """Lift and cluster every specimen, and write the table."""
    from ..data.preprocess import load_cached, usable_plant_ids
    from ..models.backbones import build_backbone

    plant_ids = plant_ids or usable_plant_ids(cache_dir)
    # Which backbone lifts the features is the open question this module was
    # written to answer and could not, because DINOv3 was gated when it was
    # written. It no longer is.
    backbone = build_backbone(backbone_kind, variant=variant)

    results = []
    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, cache_dir)
        result = segment_specimen(cached, backbone)
        results.append(result)
        if verbose:
            flag = "" if result.rim_confident else "  (rim fallback)"
            print(
                f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
                f"pooled {result.n_pooled:6d}/{result.n_points:6d}  "
                f"gap {result.height_gap_m:+.3f} m  "
                f"agree {result.agreement_with_rim:.3f}{flag}"
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8"
    )
    return results


def summarise(results: list[SegmentationResult]) -> dict:
    """Grouped by batch, since the question is about one batch in particular."""
    def batch(plant_id: str) -> str:
        if plant_id.startswith("M"):
            return "Mango"
        if plant_id.startswith("V"):
            return "V001-V008"
        return "E001-E010" if int(plant_id[1:]) <= 10 else "E011-E020"

    groups: dict[str, list[SegmentationResult]] = {}
    for r in results:
        groups.setdefault(batch(r.plant_id), []).append(r)

    return {
        name: {
            "n": len(rows),
            "mean_height_gap_m": round(float(np.mean([r.height_gap_m for r in rows])), 4),
            "mean_agreement": round(float(np.mean([r.agreement_with_rim for r in rows])), 4),
            "mean_upper_above_rim": round(
                float(np.mean([r.upper_above_rim for r in rows])), 4
            ),
            "rim_confident": sum(r.rim_confident for r in rows),
        }
        for name, rows in sorted(groups.items())
    }


__all__ = ["SegmentationResult", "run", "segment_specimen", "summarise"]
