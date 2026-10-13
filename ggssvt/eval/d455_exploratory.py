"""Exploratory biomass-model comparison for fixed-camera D455 turntable captures.

This deliberately does not claim calibrated reconstruction or plant-only
segmentation. It pools handcrafted RGB-D statistics from a fixed central image
crop, then compares small-data regressors with nested plant-level LOOCV. The
crop can contain greenhouse structure and the pot, so outputs are pilot signals
only and must not be compared directly with the registered Kinect pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_ANGLES = tuple(range(0, 360, 15))
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def _row_features(rgb: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Summarise one plant-centred crop without pretending it is a mask."""
    import cv2

    height, width = depth_m.shape
    x0, x1 = int(0.30 * width), int(0.70 * width)
    y0, y1 = int(0.05 * height), int(0.90 * height)
    crop_rgb = rgb[y0:y1, x0:x1]
    crop_depth = depth_m[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2HSV)

    features: list[float] = []
    for channel in cv2.split(crop_rgb):
        hist = cv2.calcHist([channel], [0], None, [8], [0, 256]).ravel()
        features.extend((hist / max(1, hist.sum())).tolist())
    for channel, bins, upper in (
        (hsv[:, :, 0], 12, 180),
        (hsv[:, :, 1], 4, 256),
        (hsv[:, :, 2], 4, 256),
    ):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, upper]).ravel()
        features.extend((hist / max(1, hist.sum())).tolist())

    valid = np.isfinite(crop_depth) & (crop_depth > 0)
    values = crop_depth[valid]
    features.append(float(1.0 - valid.mean()))
    depth_hist, _ = np.histogram(values, bins=np.linspace(0.3, 8.3, 17))
    features.extend((depth_hist / max(1, depth_hist.sum())).tolist())
    if values.size:
        features.extend(np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist())
    else:
        features.extend([0.0] * 5)

    row_medians = []
    for band in np.array_split(crop_depth, 8, axis=0):
        band_values = band[np.isfinite(band) & (band > 0)]
        row_medians.append(float(np.median(band_values)) if band_values.size else 0.0)
    features.extend(row_medians)
    return np.asarray(features, dtype=np.float64)


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _ridge_predict(train: np.ndarray, target: np.ndarray, test: np.ndarray, alpha: float) -> float:
    train_scaled, test_scaled = _standardise(train, test)
    target_mean = float(target.mean())
    centered_target = target - target_mean
    gram = train_scaled @ train_scaled.T + alpha * np.eye(train_scaled.shape[0])
    weights = train_scaled.T @ np.linalg.solve(gram, centered_target)
    return float(test_scaled[0] @ weights + target_mean)


def _nested_loocv(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Tune ridge strength only within each outer training fold."""
    predictions = np.empty(target.size, dtype=np.float64)
    baseline = np.empty(target.size, dtype=np.float64)
    selected_alphas: list[float] = []

    for held_out in range(target.size):
        outer_train = np.flatnonzero(np.arange(target.size) != held_out)
        baseline[held_out] = target[outer_train].mean()
        alpha_errors = []
        for alpha in ALPHAS:
            errors = []
            for inner_position, validation_index in enumerate(outer_train):
                inner_train = np.delete(outer_train, inner_position)
                prediction = _ridge_predict(
                    features[inner_train],
                    target[inner_train],
                    features[validation_index : validation_index + 1],
                    alpha,
                )
                errors.append((prediction - target[validation_index]) ** 2)
            alpha_errors.append(float(np.mean(errors)))
        alpha = ALPHAS[int(np.argmin(alpha_errors))]
        selected_alphas.append(alpha)
        predictions[held_out] = _ridge_predict(
            features[outer_train],
            target[outer_train],
            features[held_out : held_out + 1],
            alpha,
        )

    return predictions, baseline, selected_alphas


def _candidate_models(n_train: int) -> list[tuple[str, dict[str, float | int | str]]]:
    """Small, fixed model grids for inner-fold selection."""
    models: list[tuple[str, dict[str, float | int | str]]] = [
        ("ridge", {"alpha": alpha}) for alpha in ALPHAS
    ]
    models.extend(
        ("pls", {"n_components": n_components})
        for n_components in range(1, min(4, n_train - 1) + 1)
    )
    models.extend(
        (
            "svr",
            {"C": c_value, "gamma": gamma, "epsilon": epsilon},
        )
        for c_value in (0.1, 10.0, 100.0)
        for gamma in ("scale", 0.01, 0.1)
        for epsilon in (5.0, 15.0)
    )
    models.extend(
        ("knn", {"n_neighbors": neighbors, "weights": weights, "p": 2})
        for neighbors in (1, 3, 5)
        if neighbors < n_train
        for weights in ("uniform", "distance")
    )
    for model_name in ("random_forest", "extra_trees"):
        models.extend(
            (
                model_name,
                {
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": max_features,
                },
            )
            for depth in (2, None)
            for leaf in (1, 3)
            for max_features in (0.5, 1.0)
        )
    models.extend(
        (
            "gradient_boosting",
            {
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "learning_rate": learning_rate,
            },
        )
        for depth in (1, 2)
        for leaf in (1, 3)
        for learning_rate in (0.03, 0.1)
    )
    return models


def _make_estimator(
    model_name: str, parameters: dict[str, float | int | str]
):
    """Build one candidate; scaling is learned only from its training fold."""
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge

    if model_name == "ridge":
        model = Ridge(alpha=float(parameters["alpha"]))
    elif model_name == "pls":
        model = PLSRegression(n_components=int(parameters["n_components"]), max_iter=500)
    elif model_name == "svr":
        model = SVR(
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            epsilon=float(parameters["epsilon"]),
        )
    elif model_name == "knn":
        model = KNeighborsRegressor(
            n_neighbors=int(parameters["n_neighbors"]),
            weights=str(parameters["weights"]),
            p=int(parameters["p"]),
        )
    elif model_name in ("random_forest", "extra_trees"):
        forest = RandomForestRegressor if model_name == "random_forest" else ExtraTreesRegressor
        model = forest(
            n_estimators=100,
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=float(parameters["max_features"]),
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=int(parameters["max_depth"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            learning_rate=float(parameters["learning_rate"]),
            loss="huber",
            random_state=42,
        )
    else:
        raise ValueError(f"unknown D455 exploratory model {model_name!r}")

    return make_pipeline(StandardScaler(), model)


def _nested_model_comparison(
    features: np.ndarray, target: np.ndarray
) -> tuple[dict[str, dict], np.ndarray]:
    """Compare models with outer LOOCV and inner LOOCV hyperparameter selection."""
    n_samples = target.size
    predictions = {
        "mean_baseline": np.empty(n_samples, dtype=np.float64),
        "ridge": np.empty(n_samples, dtype=np.float64),
        "pls": np.empty(n_samples, dtype=np.float64),
        "svr": np.empty(n_samples, dtype=np.float64),
        "knn": np.empty(n_samples, dtype=np.float64),
        "random_forest": np.empty(n_samples, dtype=np.float64),
        "extra_trees": np.empty(n_samples, dtype=np.float64),
        "gradient_boosting": np.empty(n_samples, dtype=np.float64),
    }
    fold_parameters: dict[str, list[dict]] = {name: [] for name in predictions}

    for held_out in range(n_samples):
        outer_train = np.flatnonzero(np.arange(n_samples) != held_out)
        outer_validation = np.array([held_out])
        predictions["mean_baseline"][held_out] = float(target[outer_train].mean())
        candidates = _candidate_models(len(outer_train))
        best_inner_rmse: dict[str, float] = {
            model_name: float("inf") for model_name in predictions if model_name != "mean_baseline"
        }
        best_models: dict[str, tuple[dict, object]] = {}

        for model_name, parameters in candidates:
            squared_errors = []
            for inner_position, validation_index in enumerate(outer_train):
                inner_train = np.delete(outer_train, inner_position)
                estimator = _make_estimator(model_name, parameters)
                estimator.fit(features[inner_train], target[inner_train])
                prediction = float(
                    np.asarray(estimator.predict(features[validation_index : validation_index + 1])).ravel()[0]
                )
                squared_errors.append((prediction - target[validation_index]) ** 2)

            inner_rmse = float(np.sqrt(np.mean(squared_errors)))
            if inner_rmse < best_inner_rmse[model_name]:
                best_inner_rmse[model_name] = inner_rmse
                best_models[model_name] = (parameters, _make_estimator(model_name, parameters))

        for model_name, (parameters, estimator) in best_models.items():
            estimator.fit(features[outer_train], target[outer_train])
            predictions[model_name][held_out] = float(
                np.asarray(estimator.predict(features[outer_validation])).ravel()[0]
            )
            fold_parameters[model_name].append(
                {"held_out_plant_index": held_out, "parameters": parameters}
            )
        fold_parameters["mean_baseline"].append(
            {"held_out_plant_index": held_out, "parameters": {}}
        )

    summary = {
        name: {
            "metrics": _metrics(model_predictions, target),
            "fold_parameters": fold_parameters[name],
            "predictions_g": model_predictions.tolist(),
        }
        for name, model_predictions in predictions.items()
    }
    return summary, predictions["mean_baseline"]


def _metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = predicted - target
    total = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse_g": float(np.sqrt(np.mean(residual**2))),
        "mae_g": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / total) if total else float("nan"),
        "bias_g": float(np.mean(residual)),
    }


def _casefold_path(parent: Path, name: str) -> Path:
    matches = [path for path in parent.iterdir() if path.name.casefold() == name.casefold()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one case-insensitive match for {name!r} in {parent}, "
            f"found {len(matches)}"
        )
    return matches[0]


def run(dataset_root: Path, plant_ids: tuple[str, ...] = tuple(f"EU{i:02}" for i in range(1, 11))) -> dict:
    """Extract central-crop descriptors and evaluate fresh-mass predictions."""
    import cv2

    plants_root = dataset_root / "plants"
    labels_path = dataset_root / "ground_truth.csv"
    with labels_path.open(newline="", encoding="utf-8") as stream:
        label_rows = {
            row["plant_id"].strip().casefold(): row
            for row in csv.DictReader(stream)
            if row.get("plant_id", "").strip()
        }

    feature_rows, targets, specimen_reports = [], [], []
    for plant_id in plant_ids:
        plant_dir = _casefold_path(plants_root, plant_id)
        row = label_rows.get(plant_id.casefold())
        if row is None:
            raise ValueError(f"missing ground-truth row for {plant_id}")
        target = float(row["shoot_fresh_mass_g"])
        if target <= 0:
            raise ValueError(f"{plant_id} has no positive shoot_fresh_mass_g label")

        manifest = json.loads((plant_dir / "capture_manifest.json").read_text(encoding="utf-8"))
        camera = manifest.get("camera", {})
        depth_scale = float(camera.get("depth_scale_m_per_unit", 0))
        if depth_scale <= 0:
            raise ValueError(f"{plant_id} manifest has no positive D455 depth scale")

        per_view, depth_valid_fractions = [], []
        missing_angles = []
        for angle in EXPECTED_ANGLES:
            stem = f"{angle:03d}.png"
            rgb_path, depth_path = plant_dir / "rgb" / stem, plant_dir / "depth" / stem
            if not rgb_path.is_file() or not depth_path.is_file():
                missing_angles.append(angle)
                continue
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or raw_depth is None or raw_depth.ndim != 2:
                raise ValueError(f"failed to decode aligned RGB-D view {plant_id}/{stem}")
            if rgb.shape[:2] != raw_depth.shape:
                raise ValueError(f"RGB/depth dimensions disagree for {plant_id}/{stem}")
            depth_m = raw_depth.astype(np.float32) * depth_scale
            height, width = depth_m.shape
            crop = depth_m[
                int(0.05 * height) : int(0.90 * height),
                int(0.30 * width) : int(0.70 * width),
            ]
            depth_valid_fractions.append(float(np.mean(crop > 0)))
            per_view.append(_row_features(rgb, depth_m))

        if missing_angles:
            raise ValueError(f"{plant_id} is missing capture angles: {missing_angles}")
        view_features = np.stack(per_view)
        feature_rows.append(np.concatenate([view_features.mean(axis=0), view_features.std(axis=0)]))
        targets.append(target)
        specimen_reports.append(
            {
                "plant_id": plant_id,
                "capture_folder": plant_dir.name,
                "views": len(per_view),
                "camera_serial_number": str(camera.get("serial_number", "unknown")),
                "depth_scale_m_per_unit": depth_scale,
                "mean_roi_depth_valid_fraction": float(np.mean(depth_valid_fractions)),
                "shoot_fresh_mass_g": target,
                "shoot_dry_mass_g": float(row.get("shoot_dry_mass_g") or 0),
                "notes": row.get("notes", ""),
            }
        )

    features = np.stack(feature_rows)
    target_array = np.asarray(targets, dtype=np.float64)
    model_comparison, baseline = _nested_model_comparison(features, target_array)
    best_model = min(
        model_comparison,
        key=lambda name: model_comparison[name]["metrics"]["rmse_g"],
    )
    return {
        "name": "d455_eu01_eu10_central_rgbd_model_comparison_exploratory",
        "status": "exploratory_only",
        "target": "shoot_fresh_mass_g",
        "n": len(plant_ids),
        "plant_ids": list(plant_ids),
        "feature_description": (
            "Per-view RGB/HSV histograms and depth histograms/quantiles in a fixed "
            "central crop (x=30%-70%, y=5%-90%); pooled mean and standard deviation "
            "over the 24 turntable angles. No plant segmentation, pot-rim exclusion, "
            "DINO features, or calibrated pose is used."
        ),
        "validation": (
            "Outer leave-one-plant-out predictions; model hyperparameters selected "
            "by inner leave-one-out within each outer training fold. Scaling is fitted "
            "within each training fold. All models use identical specimen-level features."
        ),
        "model_comparison": model_comparison,
        "best_model_by_outer_loocv_rmse": best_model,
        "best_model_selection_caveat": (
            "The lowest-RMSE model among several candidates has additional selection "
            "optimism at n=10; treat it as a pilot winner and confirm on an independent cohort."
        ),
        "folds": [
            {
                "held_out_plant": plant_ids[index],
                "target_g": float(target_array[index]),
                "best_model_prediction_g": float(
                    model_comparison[best_model]["predictions_g"][index]
                ),
                "training_mean_prediction_g": float(baseline[index]),
                "ridge_alpha": model_comparison["ridge"]["fold_parameters"][index][
                    "parameters"
                ]["alpha"],
            }
            for index in range(len(plant_ids))
        ],
        "data_quality": {
            "all_expected_views_present": True,
            "camera_serial_numbers": sorted(
                {row["camera_serial_number"] for row in specimen_reports}
            ),
            "warnings": [
                "Capture manifests contain intrinsics and turntable labels but no calibrated camera-to-object poses.",
                "The fixed crop is not a plant mask; greenhouse objects and pot pixels can dominate the features.",
                "No confident pot-rim estimate is available, so pot exclusion is not claimed.",
                "The ten labels cover a narrow fresh-mass range and are not an external validation cohort.",
                "Dry-mass labels should be audited: EU01-EU10 are unusually close to fresh masses; EU05 also has contradictory notes and dimensions.",
            ],
        },
        "specimens": specimen_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work_dirs/ab_ssvit/d455-exploratory/eu01_eu10_model_comparison.json"),
    )
    args = parser.parse_args()
    result = run(args.dataset_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Best D455 pilot model: {result['best_model_by_outer_loocv_rmse']} "
        f"(RMSE={result['model_comparison'][result['best_model_by_outer_loocv_rmse']]['metrics']['rmse_g']:.1f} g); "
        f"mean-baseline RMSE={result['model_comparison']['mean_baseline']['metrics']['rmse_g']:.1f} g"
    )
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
