"""How many views the carve actually needs, judged physically.

The sampling question -- would four images at 90 degrees do? -- was previously
answered with geometric quality scores, which degrade gently and politely as
views are removed. Multi-view agreement falls from 0.608 at twelve views to 0.424
at four, a decline that looks like a trade-off worth considering.

Dividing measured mass by reconstructed above-ground volume tells a different
story. At four views the median implied bulk density is 9.2 kg per cubic metre --
lighter than expanded polystyrene, thirty to ninety times below fresh plant
tissue -- from hulls averaging 126 litres for plants of at most 2.35 kg. Not one
of twenty-five specimens is physically capable of weighing what it weighs. The
reconstructions have not degraded; they have stopped being reconstructions.

The reason is geometric. Four views at 90 degrees is the visual-hull minimum for
a *convex* object, and a plant is the opposite of convex: every unsampled azimuth
leaves a prism of empty space that nothing carves away.

Which is the argument for reporting the plausibility check beside a geometric
quality metric rather than instead of one. Agreement measures whether the views
concur; it cannot notice that they concur on something the size of a wardrobe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres
from .plausibility import classify, summarise

# Cache directory per view count. The 12-view carve lives in the default cache;
# the rest are built with `preprocess --views N --cache-dir ...`.
VIEW_CACHES: dict[int, str] = {
    3: "cache_v3",
    4: "cache_v4",
    6: "cache_v6",
    12: "cache",
}


@dataclass(frozen=True)
class ViewCountResult:
    """One view count's reconstruction quality, geometric and physical."""

    n_views: int
    n_usable: int
    n_total: int
    agreement: float
    coverage: float
    mean_above_ground_l: float
    n_plausible: int
    median_density_kg_m3: float

    def as_dict(self) -> dict:
        return {
            "n_views": self.n_views,
            "usable": f"{self.n_usable}/{self.n_total}",
            "agreement": round(self.agreement, 3),
            "coverage": round(self.coverage, 3),
            "mean_above_ground_l": round(self.mean_above_ground_l, 1),
            "plausible": f"{self.n_plausible}/{self.n_usable}",
            "median_density_kg_m3": round(self.median_density_kg_m3, 1),
        }


def evaluate_view_count(
    cache_dir: Path, *, n_views: int, n_total: int = 38
) -> ViewCountResult:
    """Score one already-built cache."""
    from ..data.preprocess import load_cached, load_quality, usable_plant_ids

    plant_ids = usable_plant_ids(cache_dir)
    quality = load_quality(cache_dir)
    heights = voxel_grid_centres()[..., 2]

    checks, volumes_l = [], []
    for plant_id in plant_ids:
        cached = load_cached(plant_id, cache_dir)
        above = cached.occupancy & (heights > cached.pot_height_m)
        volume = float(above.sum()) * cached.voxel_size_m ** 3
        volumes_l.append(volume * 1000.0)
        checks.append(classify(plant_id, float(cached.target_kg), volume))

    summary = summarise(checks)
    return ViewCountResult(
        n_views=n_views,
        n_usable=len(plant_ids),
        n_total=n_total,
        agreement=float(np.mean([quality[p].multiview_agreement for p in plant_ids])),
        coverage=float(np.mean([quality[p].surface_coverage for p in plant_ids])),
        mean_above_ground_l=float(np.mean(volumes_l)) if volumes_l else 0.0,
        n_plausible=summary["n_plausible"],
        median_density_kg_m3=summary["median_density_kg_m3"] or 0.0,
    )


def run_ablation(
    work_dir: Path = WORK_DIR, *, verbose: bool = True
) -> list[ViewCountResult]:
    """Score every view-count cache that has been built.

    Missing caches are reported rather than raising -- building them is a
    separate, slow step, and a partial sweep is still worth seeing.
    """
    results = []
    for n_views, name in sorted(VIEW_CACHES.items()):
        cache_dir = work_dir / name
        if not (cache_dir / "quality.json").exists():
            if verbose:
                print(
                    f"  {n_views:2d} views: no cache at {cache_dir}. Build it with\n"
                    f"      python -m ggssvt.cli preprocess --views {n_views} "
                    f"--cache-dir {cache_dir}"
                )
            continue
        results.append(evaluate_view_count(cache_dir, n_views=n_views))
    return results


def format_table(results: list[ViewCountResult]) -> str:
    lines = [
        (
            f"{'views':>5s} {'usable':>8s} {'agreement':>10s} {'coverage':>9s} "
            f"{'mean hull':>11s} {'plausible':>10s} {'median kg/m3':>13s}"
        ),
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r.n_views:5d} {f'{r.n_usable}/{r.n_total}':>8s} {r.agreement:10.3f} "
            f"{r.coverage:9.3f} {r.mean_above_ground_l:9.1f} L "
            f"{f'{r.n_plausible}/{r.n_usable}':>10s} {r.median_density_kg_m3:13.1f}"
        )
    return "\n".join(lines)


__all__ = [
    "VIEW_CACHES",
    "ViewCountResult",
    "evaluate_view_count",
    "format_table",
    "run_ablation",
]
