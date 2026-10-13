"""External validation: does the regression transfer off our own 36 specimens?

Our biomass numbers cannot answer the question they were built for. FINDINGS 7l
shows why: predicting mass from a specimen's capture batch alone beats every
method we have, so a leave-one-out score on our data measures how separable the
capture sessions were, not how well geometry predicts mass. No re-analysis fixes
that, and there will be no further capture.

This runs the same question on somebody else's plants. The 4TU greenhouse lettuce
set is 388 destructively harvested plants across four cultivars and a seven-week
growth series, so its mass range is continuous by construction rather than
clustered into sessions, 1.4 g to 459.7 g with nothing missing in between.

**Three things are reported, in this order, and the order matters.**

1. *Does the measurement work at all.* Their Height, Diameter and LeafArea were
   measured destructively on the same plants, so the depth-derived versions can
   be correlated against them before anything is regressed. A pipeline whose
   diameter does not track a ruler has no business predicting mass.

2. *Which plants to screen out.* The same discipline the rest of the project
   uses: a criterion fixed in advance, applied before the regression, reported
   with the count it costs. Here it is agreement with the measured diameter,
   because a segmentation that fails produces a diameter that disagrees.

3. *Then the regression*, under leave-one-out and under leave-one-cultivar-out.
   The second is this dataset's analogue of leave-one-batch-out: it asks whether
   the fit survives a cultivar it has never seen, which is the transfer question
   an operational deployment actually faces.

Only the image-only half of our pipeline can run here: one top-down view means
no carve and no fusion. That is not a limitation dodged: `direct 2D` and
`2D + profile` are the two methods that currently win on our own data, so what is
being tested externally is what is being claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import WORK_DIR

# A segmentation is accepted when its diameter lands within this factor of the
# measured one. Fixed before the numbers were looked at, and wide because a
# destructive "principal diameter" and a top-down projection are related but not
# the same measurement: the criterion is meant to catch failures, not to tune.
DIAMETER_TOLERANCE = 0.40

# Below this the plant is a seedling of a few grams, its projection is a handful
# of pixels, and the depth camera resolves nothing useful.
MIN_PIXELS = 400


def cache_path(root: Path) -> Path:
    return root / "lettuce_features.npz"


def extract(
    root: Path,
    *,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Measure every plant and cache the result. Around four minutes for 388."""
    from PIL import Image

    from ..data.lettuce import FEATURE_NAMES, load_ground_truth, measure

    records = load_ground_truth(root)
    if limit:
        records = records[:limit]

    rows, ids, varieties, missing = [], [], [], []
    for index, record in enumerate(records, start=1):
        # The archive as distributed is one file short of its own ground truth:
        # Image332 names RGB_332.png, which is absent, while an unreferenced
        # RGB_322.png sits in the folder and no Image322 record exists. That is
        # very likely a misnamed file. Pairing it with the orphaned Depth_332
        # was tested by overlapping the RGB's saturated region with the depth's
        # raised region; the result, 0.163, sits inside the range known-correct
        # pairs span (0.151 to 0.321), so the check does not discriminate.
        # Substituting on a hunch would put an unverified plant into the
        # validation set, so the record is skipped and counted instead.
        if not record.rgb.exists() or not record.depth.exists():
            missing.append(record.image_id)
            continue
        rgb = np.asarray(Image.open(record.rgb).convert("RGB"))
        depth = np.asarray(Image.open(record.depth))
        measured = measure(rgb, depth)
        rows.append([measured[name] for name in FEATURE_NAMES]
                    + [measured["n_pixels"], measured["tray_depth_m"]])
        ids.append(record.image_id)
        varieties.append(record.variety)
        if verbose and index % 50 == 0:
            print(f"  measured {index}/{len(records)}", flush=True)

    if verbose and missing:
        print(f"  skipped {len(missing)} record(s) with no image on disk: "
              f"{', '.join(missing)}")

    kept = [r for r in records if r.image_id not in set(missing)]
    return {
        "missing": np.array(missing),
        "ids": np.array(ids),
        "varieties": np.array(varieties),
        "features": np.array(rows, dtype=np.float64),
        "names": np.array(list(FEATURE_NAMES) + ["n_pixels", "tray_depth_m"]),
        "fresh_weight_g": np.array([r.fresh_weight_g for r in kept]),
        "dry_weight_g": np.array([r.dry_weight_g for r in kept]),
        "height_cm": np.array([r.height_cm for r in kept]),
        "diameter_cm": np.array([r.diameter_cm for r in kept]),
        "leaf_area_cm2": np.array([r.leaf_area_cm2 for r in kept]),
    }


def load_or_extract(root: Path, *, force: bool = False, verbose: bool = True) -> dict:
    """The cached measurements, extracting them once if they are not there."""
    path = cache_path(root)
    if path.exists() and not force:
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    payload = extract(root, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def correlate(measured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Pearson r and the mean absolute difference, for a measurement check."""
    keep = np.isfinite(measured) & np.isfinite(reference)
    if keep.sum() < 3:
        return {"r": float("nan"), "mae": float("nan"), "n": int(keep.sum())}
    a, b = measured[keep], reference[keep]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return {"r": float("nan"), "mae": float("nan"), "n": int(keep.sum())}
    return {
        "r": float(np.corrcoef(a, b)[0, 1]),
        "mae": float(np.abs(a - b).mean()),
        "n": int(keep.sum()),
    }


def run(
    *,
    root: Path | None = None,
    out: Path = WORK_DIR / "reports" / "external_lettuce.json",
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Measure, screen, then regress. Nothing here needs a GPU."""
    from ..data.lettuce import (
        DATASET_DIR,
        FEATURE_NAMES,
        OUTLINE_NAMES,
        SURFACE_NAMES,
    )
    from .batch_holdout import cross_validate
    from .metrics import paired_bootstrap_difference

    root = root or DATASET_DIR
    data = load_or_extract(root, force=force, verbose=verbose)

    names = [str(n) for n in data["names"]]
    features = data["features"]
    column = {name: features[:, i] for i, name in enumerate(names)}
    ids = [str(i) for i in data["ids"]]
    varieties = [str(v) for v in data["varieties"]]

    # ---- 1. does the measurement work
    checks = {
        "diameter_cm": correlate(column["diameter_cm"], data["diameter_cm"]),
        "height_cm": correlate(column["height_cm"], data["height_cm"]),
        "area_cm2": correlate(column["area_cm2"], data["leaf_area_cm2"]),
    }
    if verbose:
        print("\n  measurement against the destructive record")
        for trait, stat in checks.items():
            print(f"    {trait:14s} r = {stat['r']:+.3f}  MAE {stat['mae']:8.2f}  "
                  f"n = {stat['n']}")

    # ---- 2. screen, on a criterion fixed in advance
    reference = data["diameter_cm"]
    ratio = np.divide(column["diameter_cm"], np.maximum(reference, 1e-6))
    passed = (
        (column["n_pixels"] >= MIN_PIXELS)
        & (np.abs(ratio - 1.0) <= DIAMETER_TOLERANCE)
    )
    if verbose:
        print(f"\n  screen: {int(passed.sum())} of {len(passed)} plants agree with "
              f"the measured diameter within {DIAMETER_TOLERANCE:.0%}")

    kept = np.flatnonzero(passed)
    if kept.size < 20:
        raise ValueError(f"only {kept.size} plants survive the screen")

    # ---- 3. regress, both schemes
    targets = data["fresh_weight_g"][kept] / 1000.0        # kilograms, as ours are
    kept_varieties = [varieties[i] for i in kept]
    kept_ids = [ids[i] for i in kept]

    # Nested on purpose: each set adds one kind of information to the one above,
    # so the comparison isolates what that kind is worth. The surface set is the
    # only genuinely three-dimensional one: it is taken off the back-projected
    # point cloud rather than off the silhouette, and whether it earns its place
    # is the question our own specimens cannot answer through the batch confound.
    sets = {
        "direct 2D": ["area_cm2", "diameter_cm", "compactness", "elongation"],
        "2D + profile": list(OUTLINE_NAMES),
        "surface only": list(SURFACE_NAMES),
        "2D + profile + surface": list(FEATURE_NAMES),
        "volume only": ["volume_l"],
        "hull volume only": ["hull_volume_l"],
    }

    # The screen uses their measured diameter, which correlates with mass, so a
    # screened score is selected partly on the label. That is the same exposure
    # our own density criterion has, implied density is mass over volume, but
    # it means the screened number alone is not reportable. Both are computed and
    # both go in the record.
    everything = np.arange(len(ids))
    all_targets = data["fresh_weight_g"] / 1000.0

    rows, gaps = [], {}
    predictions: dict[str, np.ndarray] = {}
    for label, columns in sets.items():
        matrix = np.column_stack([column[c][kept] for c in columns])
        loocv, _ = cross_validate(
            matrix, targets, kept_ids, condition=label, scheme="loocv",
            components=min(10, len(columns)))
        cultivar, cultivar_predictions = cross_validate(
            matrix, targets, kept_ids, condition=label, scheme="lobo",
            groups=kept_varieties, components=min(10, len(columns)))
        predictions[label] = cultivar_predictions

        unscreened_matrix = np.column_stack([column[c][everything] for c in columns])
        unscreened, _ = cross_validate(
            unscreened_matrix, all_targets, ids,
            condition=label + ", unscreened", scheme="lobo",
            groups=varieties, components=min(10, len(columns)))
        rows.extend([loocv.as_dict(), cultivar.as_dict(), unscreened.as_dict()])
        gaps[label] = {
            "loocv_rmse_kg": loocv.rmse_kg,
            "loocv_r2": loocv.r2,
            "held_out_cultivar_rmse_kg": cultivar.rmse_kg,
            "held_out_cultivar_r2": cultivar.r2,
            "inflation_kg": round(cultivar.rmse_kg - loocv.rmse_kg, 4),
            "unscreened_cultivar_rmse_kg": unscreened.rmse_kg,
            "unscreened_cultivar_r2": unscreened.r2,
        }
        if verbose:
            print(f"    {label:14s} LOOCV {loocv.rmse_kg:.4f} (R2 {loocv.r2:+.3f})"
                  f"   held-out cultivar {cultivar.rmse_kg:.4f} (R2 {cultivar.r2:+.3f})"
                  f"   unscreened {unscreened.rmse_kg:.4f} (R2 {unscreened.r2:+.3f})")

    # Does the surface earn its place, or is the improvement sampling noise?
    # The project's standing rule: an interval on the paired difference, not two
    # point estimates side by side. Scored on the held-out cultivar, because a
    # feature set that only helps in-distribution has not helped.
    reference = "2D + profile"
    comparisons = {}
    for label, predicted in predictions.items():
        if label == reference:
            continue
        paired = paired_bootstrap_difference(
            predicted, predictions[reference], targets)
        comparisons[label] = {
            k: round(v, 4) for k, v in paired.items()
            if isinstance(v, (int, float))
        }
        if verbose:
            print(f"    {label:24s} vs {reference}: "
                  f"{paired['difference'] * 1000:+.1f} g "
                  f"[{paired['low'] * 1000:+.1f}, {paired['high'] * 1000:+.1f}], "
                  f"p = {paired['p_direction']:.4f}")

    result = {
        "dataset": "3rd Autonomous Greenhouse Challenge lettuce, DOI 10.4121/15023088",
        "paired_vs_2d_profile": comparisons,
        "n_plants": len(ids),
        "skipped_no_image": [str(m) for m in data.get("missing", [])],
        "n_after_screen": int(kept.size),
        "screen": {
            "diameter_tolerance": DIAMETER_TOLERANCE,
            "min_pixels": MIN_PIXELS,
            "note": "criterion fixed before the numbers were read; a failed "
                    "segmentation shows up as a diameter that disagrees. It "
                    "uses their measured diameter, which correlates with mass, "
                    "so the unscreened rows are reported alongside",
        },
        "measurement_checks": checks,
        "mean_mass_g": float(data["fresh_weight_g"].mean()),
        "mass_range_g": [float(data["fresh_weight_g"].min()),
                         float(data["fresh_weight_g"].max())],
        "cultivars": {v: kept_varieties.count(v) for v in sorted(set(kept_varieties))},
        "rows": rows,
        "gaps": gaps,
        "note": "held-out cultivar is this dataset's analogue of leave-one-batch-out: "
                "the fit is scored on a variety it never saw",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = [
    "DIAMETER_TOLERANCE", "MIN_PIXELS", "cache_path", "correlate", "extract",
    "load_or_extract", "run",
]
