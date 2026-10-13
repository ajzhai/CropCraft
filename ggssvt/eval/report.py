"""Result tables and figures.

Everything here writes to ``work_dirs/ggssvt/reports``. Tables are emitted as
both Markdown (for reading) and CSV (for the dissertation's LaTeX pipeline), and
every number carries the sample size it was computed from, because with
twenty-eight specimens the sample size is part of the result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres
from ..data.preprocess import load_cached, load_quality
from .baselines import evaluate_baselines, load_features
from .metrics import RegressionMetrics, bootstrap_interval, regression_metrics
from .plausibility import classify, summarise

REPORT_DIR = WORK_DIR / "reports"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _markdown_table(header: list[str], rows: list[list]) -> str:
    widths = [
        max(len(str(header[i])), *(len(str(row[i])) for row in rows)) if rows else len(str(header[i]))
        for i in range(len(header))
    ]
    lines = [
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        )
    return "\n".join(lines)


def _above_ground_volume_m3(cached) -> float:
    """Volume above this specimen's own pot rim, not the global constant.

    Using each specimen's measured rim is what makes the implied density
    comparable across batches: V pots weigh 17-32 kg against E's 0.7-2.2, so a
    shared cut height counts a different amount of pot as plant in each.
    """
    centres = voxel_grid_centres()
    above = cached.occupancy & (centres[..., 2] > cached.pot_height_m)
    return float(above.sum()) * cached.voxel_size_m ** 3


def dataset_table(plant_ids: list[str]) -> tuple[list[str], list[list]]:
    """Per-specimen geometry and ground truth."""
    quality = load_quality()
    header = [
        "plant",
        "species",
        "mass_kg",
        "above_ground_L",
        "height_m",
        "coverage",
        "agreement",
        "connected",
        "pot_rim_m",
        "kg_m3",
        "verdict",
        "usable",
    ]
    rows = []
    for plant_id in plant_ids:
        q = quality[plant_id]
        cached = load_cached(plant_id)
        volume = _above_ground_volume_m3(cached)
        check = classify(plant_id, float(cached.target_kg), volume)
        rows.append(
            [
                plant_id,
                cached.species,
                f"{cached.target_kg:.3f}",
                f"{volume * 1000:.2f}",
                f"{q.height_m:.2f}",
                f"{q.surface_coverage:.3f}",
                f"{q.multiview_agreement:.3f}",
                f"{q.connected_fraction:.2f}",
                f"{cached.pot_height_m:.3f}" + ("" if cached.pot.confident else "*"),
                f"{check.density_kg_m3:.0f}",
                check.verdict,
                "yes" if q.is_usable() else "no",
            ]
        )
    return header, rows


def comparison_table(
    results: dict[str, tuple[RegressionMetrics, np.ndarray]],
    targets: np.ndarray,
) -> tuple[list[str], list[list]]:
    """Method comparison with bootstrap intervals on RMSE."""
    header = ["method", "RMSE_kg", "RMSE_95CI", "MAE_kg", "MARE_%", "R2", "bias_kg", "n"]
    rows = []
    for name, (metrics, predictions) in results.items():
        low, high = bootstrap_interval(predictions, targets)
        rows.append(
            [
                name,
                f"{metrics.rmse_kg:.3f}",
                f"[{low:.3f}, {high:.3f}]",
                f"{metrics.mae_kg:.3f}",
                f"{metrics.mare * 100:.1f}",
                f"{metrics.r2:.3f}",
                f"{metrics.bias_kg:+.3f}",
                metrics.n,
            ]
        )
    return header, rows


def scatter_svg(
    predicted: np.ndarray,
    target: np.ndarray,
    labels: list[str],
    *,
    title: str,
    width: int = 480,
    height: int = 480,
) -> str:
    """Predicted-versus-true scatter as a standalone SVG.

    Written by hand rather than through matplotlib so the evaluation stage adds
    no plotting dependency to ``requirements.txt``.
    """
    limit = float(max(predicted.max(), target.max())) * 1.1
    pad = 56

    def sx(value: float) -> float:
        return pad + (value / limit) * (width - 2 * pad)

    def sy(value: float) -> float:
        return height - pad - (value / limit) * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-size="15">{title}</text>',
        f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(limit)}" y2="{sy(limit)}" '
        'stroke="#bbb" stroke-dasharray="4 4"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 14}" text-anchor="middle" font-size="12">'
        "true mass (kg)</text>",
        f'<text x="16" y="{height / 2}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16 {height / 2})">predicted mass (kg)</text>',
    ]

    for step in range(5):
        value = limit * step / 4
        parts.append(
            f'<text x="{sx(value)}" y="{height - pad + 16}" text-anchor="middle" '
            f'font-size="10" fill="#555">{value:.1f}</text>'
        )
        parts.append(
            f'<text x="{pad - 8}" y="{sy(value) + 4}" text-anchor="end" '
            f'font-size="10" fill="#555">{value:.1f}</text>'
        )

    for x, y, label in zip(target, predicted, labels):
        colour = "#2a7" if label.startswith("M") else "#37c"
        parts.append(
            f'<circle cx="{sx(float(x)):.1f}" cy="{sy(float(y)):.1f}" r="4.5" '
            f'fill="{colour}" fill-opacity="0.75"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def write_report(
    plant_ids: list[str] | None = None,
    *,
    model_predictions: dict[str, float] | None = None,
    report_dir: Path = REPORT_DIR,
) -> Path:
    """Run the baselines, fold in any model predictions, and write the report.

    Args:
        plant_ids: specimens to include. Defaults to every usable specimen.
        model_predictions: per plant id, GG-SSVT's leave-one-out prediction.

    Returns:
        The path of the Markdown report.
    """
    from ..data.preprocess import usable_plant_ids

    plant_ids = plant_ids or usable_plant_ids()
    features = load_features(plant_ids)
    targets = np.array([f.target_kg for f in features])

    results = evaluate_baselines(features)

    if model_predictions:
        aligned = np.array([model_predictions.get(f.plant_id, np.nan) for f in features])
        if not np.isnan(aligned).any():
            results["GG-SSVT"] = (regression_metrics(aligned, targets), aligned)

    report_dir.mkdir(parents=True, exist_ok=True)

    data_header, data_rows = dataset_table(plant_ids)
    comp_header, comp_rows = comparison_table(results, targets)

    _write_csv(report_dir / "dataset.csv", data_header, data_rows)
    _write_csv(report_dir / "comparison.csv", comp_header, comp_rows)

    best_name = min(results, key=lambda k: results[k][0].rmse_kg)
    best_predictions = results[best_name][1]
    (report_dir / "scatter.svg").write_text(
        scatter_svg(
            best_predictions,
            targets,
            [f.plant_id for f in features],
            title=f"{best_name}: predicted vs true fresh mass",
        ),
        encoding="utf-8",
    )

    species = sorted({load_cached(pid).species for pid in plant_ids})
    per_species_sections = []
    for name in species:
        subset = [
            f for f in features if load_cached(f.plant_id).species == name
        ]
        if len(subset) < 6:
            continue
        subset_targets = np.array([f.target_kg for f in subset])
        subset_results = evaluate_baselines(subset)
        header, rows = comparison_table(subset_results, subset_targets)
        per_species_sections.append(
            f"### {name} only (n={len(subset)})\n\n{_markdown_table(header, rows)}\n"
        )

    lines = [
        "# GG-SSVT results",
        "",
        f"Specimens: {len(plant_ids)}. Target: fresh above-ground mass "
        "(`net_weight_g` from `dataset/ground_truth.csv`), not oven-dry biomass.",
        "Protocol: leave-one-out cross-validation for every method in the table.",
        "",
        "## Method comparison",
        "",
        _markdown_table(comp_header, comp_rows),
        "",
        "`mean` predicts the training mean and is the floor any method must clear;",
        "a negative R-squared means the method is worse than that floor.",
        "",
    ]
    if per_species_sections:
        lines += ["## Within species", "", *per_species_sections]

    checks = []
    for plant_id in plant_ids:
        cached = load_cached(plant_id)
        checks.append(
            classify(
                plant_id, float(cached.target_kg), _above_ground_volume_m3(cached)
            )
        )
    summary = summarise(checks)
    lines += [
        "## Can the reconstruction weigh what the plant weighs?",
        "",
        "Measured mass divided by reconstructed above-ground volume. Fresh "
        "above-ground tissue is roughly 300-900 kg/m3, so a specimen far below "
        "the band has a hull enclosing the air between leaves rather than the "
        "plant, and one far above it was barely reconstructed at all.",
        "",
        f"**{summary['n_plausible']} of {summary['n']} specimens** fall inside a "
        f"generous {summary['band_kg_m3'][0]:.0f}-{summary['band_kg_m3'][1]:.0f} "
        f"kg/m3 band; the median is {summary['median_density_kg_m3']:.0f} kg/m3. "
        f"Envelope (too light): {summary['verdicts'].get('envelope', 0)}. "
        f"Missing material (too heavy): {summary['verdicts'].get('missing', 0)}.",
        "",
        "This bounds what any biomass method here can achieve, independently of "
        "which method wins a regression.",
        "",
        "## Dataset and reconstruction quality",
        "",
        "`pot_rim_m` is estimated per specimen; a `*` means no rim was "
        "detectable and the configured constant was used instead.",
        "",
        _markdown_table(data_header, data_rows),
        "",
    ]

    path = report_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    (report_dir / "metrics.json").write_text(
        json.dumps(
            {name: metrics.as_dict() for name, (metrics, _) in results.items()}, indent=2
        ),
        encoding="utf-8",
    )
    return path


__all__ = [
    "REPORT_DIR",
    "comparison_table",
    "dataset_table",
    "scatter_svg",
    "write_report",
]
