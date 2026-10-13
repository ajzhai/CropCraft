"""Figures for the results that arrived without any.

The batch holdout, the lettuce transfer and the virtual-view reconstruction all
produced JSON and tables and no pictures, and one of them, §7p, is far more
convincing as a picture than as a table. A reader who sees fourteen points on the
wrong side of a diagonal understands the metric inversion immediately; the same
reader given a column of numbers has to be told what to conclude.

Drawn with PIL, matching `paper/figures.py`: no matplotlib in this environment,
and the palette and typography are shared with the architecture diagrams so the
whole set reads as one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..config import WORK_DIR
from .architecture import INK, MUTED, PAPER, RULE, _font

REPORTS = WORK_DIR / "reports"
FIGURES = REPORTS / "figures"

S = 2                          # supersampling, so text stays crisp at print size
CARVE = "#c4622d"              # the same amber the deck gives to the weaker arm
FUSION = "#0e9384"
TRUTH = "#3e4989"
BAND = "#f2f7ed"


def _canvas(width: int, height: int):
    image = Image.new("RGB", (width * S, height * S), PAPER)
    return image, ImageDraw.Draw(image)


def _fonts():
    return {
        "title": _font("bold", 15 * S),
        "label": _font("regular", 12 * S),
        "small": _font("regular", 11 * S),
        "mono": _font("mono", 11 * S),
    }


def _header(draw, fonts, left, title, subtitle):
    draw.text((left * S, 20 * S), title, font=fonts["title"], fill=INK)
    draw.text((left * S, 42 * S), subtitle, font=fonts["small"], fill=MUTED)


def figure_inversion(out: Path | None = None) -> Path:
    """Carve against fusion, scored two ways, on one diagonal.

    Above the diagonal means fusion won. The truth puts every plant above it and
    the reprojection metric puts every plant below, which is the whole of §7p in
    one panel: the two measures do not merely disagree, they are opposites.
    """
    report = json.loads((REPORTS / "virtual_views.json").read_text(encoding="utf-8"))
    rows = report["rows"]

    width, height = 900, 500
    image, draw = _canvas(width, height)
    fonts = _fonts()

    left, right, top, bottom = 95, width - 250, 78, height - 92
    _header(draw, fonts, left,
            "The reprojection metric ranks these reconstructions backwards",
            "Fourteen Pheno4D plants, twelve virtual views each, carved and fused "
            "by the pipeline's own operators.")

    lo, hi = 0.0, 0.75

    def px(value):
        return left + (value - lo) / (hi - lo) * (right - left)

    def py(value):
        return bottom - (value - lo) / (hi - lo) * (bottom - top)

    # The region where fusion scores better than carving.
    draw.polygon(
        [(px(lo) * S, py(lo) * S), (px(hi) * S, py(hi) * S), (px(lo) * S, py(hi) * S)],
        fill=BAND)
    draw.text((px(0.06) * S, py(0.66) * S), "fusion scores better",
              font=fonts["small"], fill="#4a7a3a")
    draw.text((px(0.42) * S, py(0.12) * S), "carving scores better",
              font=fonts["small"], fill=MUTED)

    for tick in (0.0, 0.2, 0.4, 0.6):
        draw.line([px(tick) * S, top * S, px(tick) * S, bottom * S], fill=RULE, width=S)
        draw.line([left * S, py(tick) * S, right * S, py(tick) * S], fill=RULE, width=S)
        draw.text((px(tick) * S - 10 * S, (bottom + 8) * S), f"{tick:.1f}",
                  font=fonts["small"], fill=MUTED)
        draw.text(((left - 32) * S, py(tick) * S - 7 * S), f"{tick:.1f}",
                  font=fonts["small"], fill=MUTED)

    draw.line([px(lo) * S, py(lo) * S, px(hi) * S, py(hi) * S], fill=INK, width=S)

    draw.text(((left + (right - left) / 2 - 70) * S, (bottom + 30) * S),
              "carving", font=fonts["label"], fill=INK)
    draw.text(((left - 78) * S, (top - 24) * S), "fusion", font=fonts["label"], fill=INK)

    radius = 5 * S
    for row in rows:
        for x_key, y_key, colour in (
            ("carve_iou", "fused_iou", TRUTH),
            ("carve_silhouette_iou", "fused_silhouette_iou", CARVE),
        ):
            cx, cy = px(row[x_key]) * S, py(row[y_key]) * S
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=colour, outline=PAPER, width=S)

    legend_x, legend_y = right + 30, top + 10
    for colour, text, detail in (
        (TRUTH, "against the truth", "fusion wins 14 / 14"),
        (CARVE, "silhouette IoU", "fusion wins 0 / 14"),
    ):
        draw.ellipse([legend_x * S - radius, legend_y * S - radius,
                      legend_x * S + radius, legend_y * S + radius], fill=colour)
        draw.text(((legend_x + 14) * S, (legend_y - 7) * S), text,
                  font=fonts["label"], fill=INK)
        draw.text(((legend_x + 14) * S, (legend_y + 9) * S), detail,
                  font=fonts["small"], fill=MUTED)
        legend_y += 46

    summary = report["summary"]
    note = (f"They disagree on {summary['disagreements']} of "
            f"{summary['n_scans']} plants. Exact test on the discordant pairs: "
            f"p = {summary['metric_vs_truth']['p_value']:.1e}.")
    draw.text((left * S, (height - 26) * S), note, font=fonts["small"], fill=MUTED)

    out = out or FIGURES / "metric_inversion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.LANCZOS).save(out)
    return out


def _project(
    points: np.ndarray, size: int, *, extent: float, splat: int = 0
) -> np.ndarray:
    """Orthographic front view, coloured by depth, on a white background.

    ``splat`` widens each point into a square. A laser cloud is dense enough to
    read as a plant one pixel at a time; a voxel grid is not, and drawn as bare
    points a 4.5-pixel voxel looks like a sparse cloud rather than the solid
    block it is, which would hide exactly the volume difference the figure
    exists to show.
    """
    from .architecture import VIRIDIS

    canvas = np.ones((size, size, 3), dtype=np.float64)
    if points.size == 0:
        return canvas

    palette = np.array([
        [int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)] for c in VIRIDIS
    ], dtype=np.float64) / 255.0

    cols = ((points[:, 0] + extent / 2) / extent * (size - 1)).astype(int)
    # z measured up from the soil, so the plant stands on the bottom edge.
    rows = ((1.0 - points[:, 2] / extent) * (size - 1)).astype(int)
    on = (cols >= 0) & (cols < size) & (rows >= 0) & (rows < size)
    cols, rows = cols[on], rows[on]

    depth = points[on, 1]
    if depth.size:
        span = np.ptp(depth)
        shade = (depth - depth.min()) / span if span > 1e-9 else np.zeros_like(depth)
        index = np.clip((shade * (len(palette) - 1)).astype(int), 0, len(palette) - 1)
        # Nearest points last, so they paint over what is behind them.
        order = np.argsort(-depth)
        rows, cols, index = rows[order], cols[order], index[order]
        for dr in range(-splat, splat + 1):
            for dc in range(-splat, splat + 1):
                r = np.clip(rows + dr, 0, size - 1)
                c = np.clip(cols + dc, 0, size - 1)
                canvas[r, c] = palette[index]
    return canvas


def figure_reconstruction(
    plant: str = "Maize01", out: Path | None = None, *, subsample: int = 2
) -> Path:
    """The truth, the carve and the fusion of one plant, side by side.

    The picture the density criterion was invented to substitute for. Reading
    left to right: what the plant is, what a visual hull makes of it, and what
    depth fusion makes of it.
    """
    from ..config import VOXEL_RESOLUTION, VOXEL_SIZE_M
    from ..data.pheno4d import DATASET_DIR, latest_per_plant, load_scan, voxelise
    from ..geometry.carving import (
        carve,
        largest_connected_component,
        voxel_grid_centres,
    )
    from ..geometry.fusion import FUSION_VOXEL_M, fuse
    from .virtual_views import _downsample, _rig_and_segmentations, render

    path = next(p for p in latest_per_plant(DATASET_DIR) if p.parent.name == plant)
    scan = load_scan(path, subsample=subsample)

    truth = voxelise(scan.points, resolution=VOXEL_RESOLUTION,
                     voxel_size_m=VOXEL_SIZE_M)
    rendered = render(scan.points, target_z_m=max(scan.height_m / 2.0, 0.1))
    rig, segmentations = _rig_and_segmentations(scan.scan_id, rendered)
    carved = largest_connected_component(
        carve(rig, segmentations, plant_id=scan.scan_id).occupancy)
    fused = largest_connected_component(_downsample(
        fuse(rendered.depth_m, rendered.rotation, rendered.centre,
             mask=rendered.mask).interior,
        round(VOXEL_SIZE_M / FUSION_VOXEL_M)))

    centres = voxel_grid_centres()
    litres = VOXEL_SIZE_M ** 3 * 1000.0

    # The working volume is 1.5 m across and these plants are under 0.4 m tall,
    # so drawing the whole grid leaves the subject a speck. The frame is cropped
    # to what the three panels actually occupy: one extent for all of them, so
    # the carve's overshoot is visible as size rather than hidden by rescaling.
    spans = [scan.points, centres[carved], centres[fused]]
    reach = max(
        max(float(np.abs(block[:, 0]).max()), float(block[:, 2].max()))
        for block in spans if block.size
    )
    extent = 2.2 * reach
    panels = [
        ("the plant", scan.points, float(truth.sum()) * litres, TRUTH),
        ("silhouette carving", centres[carved], float(carved.sum()) * litres, CARVE),
        ("depth fusion", centres[fused], float(fused.sum()) * litres, FUSION),
    ]

    tile = 300
    width, height = 3 * tile + 60, tile + 132
    image, draw = _canvas(width, height)
    fonts = _fonts()
    _header(draw, fonts, 20,
            f"{plant}: what the plant is, and what each operator makes of it",
            "Twelve virtual views, the pipeline's own carve and fusion, scored "
            "against the cloud they were rendered from.")

    true_litres = panels[0][2]
    for index, (label, points, volume, colour) in enumerate(panels):
        x = 20 + index * (tile + 10)
        # The truth is a point cloud and needs no splat; the two occupancy
        # grids do, at half a voxel in each direction.
        voxels_across = extent / VOXEL_SIZE_M
        splat = 0 if index == 0 else max(1, round(tile / voxels_across / 2))
        rendering = _project(np.asarray(points), tile, extent=extent, splat=splat)
        image.paste(
            Image.fromarray((rendering * 255).astype(np.uint8)).resize(
                (tile * S, tile * S), Image.NEAREST),
            (x * S, 74 * S))
        draw.rectangle([x * S, 74 * S, (x + tile) * S, (74 + tile) * S],
                       outline=RULE, width=S)
        draw.text((x * S, (74 + tile + 10) * S), label, font=fonts["label"], fill=colour)
        ratio = "" if index == 0 else f"   {volume / max(true_litres, 1e-9):.1f}x true"
        draw.text((x * S, (74 + tile + 28) * S), f"{volume:.2f} L{ratio}",
                  font=fonts["small"], fill=MUTED)

    out = out or FIGURES / f"{plant.lower()}_truth_carve_fusion.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.LANCZOS).save(out)
    return out


def figure_batch_holdout(out: Path | None = None) -> Path:
    """What leave-one-out gains by keeping a specimen's own capture batch."""
    report = json.loads((REPORTS / "batch_holdout.json").read_text(encoding="utf-8"))
    by_condition: dict[str, dict[str, float]] = {}
    for row in report["rows"]:
        by_condition.setdefault(row["condition"], {})[row["scheme"]] = row["rmse_kg"]

    # The stored names say which specimen set each row used; on a chart the
    # distinction that matters is which representation, so they are shortened.
    SHORT = {
        "geometric, every usable specimen": "geometric, all 36",
        "geometric (no DINO)": "geometric, shared 33",
        "dinov2-base": "DINOv2 frozen, 33",
        "batch membership only": "batch label alone",
    }
    wanted = [c for c in (
        "geometric, every usable specimen", "geometric (no DINO)", "dinov2-base",
        "batch membership only") if c in by_condition]

    width, height = 900, 360
    image, draw = _canvas(width, height)
    fonts = _fonts()
    left, right, top, bottom = 230, width - 60, 92, height - 58
    _header(draw, fonts, 20,
            "What the capture batch was worth to the score",
            "Leave-one-out keeps the rest of a specimen's session in the "
            "training fold. Leave-one-batch-out does not.")

    hi = 1.25
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
        gx = left + tick / hi * (right - left)
        draw.line([gx * S, top * S, gx * S, bottom * S], fill=RULE, width=S)
        draw.text((gx * S - 12 * S, (bottom + 8) * S), f"{tick:.2f}",
                  font=fonts["small"], fill=MUTED)
    draw.text(((left + (right - left) / 2 - 40) * S, (bottom + 26) * S),
              "RMSE, kilograms", font=fonts["small"], fill=MUTED)

    row_height = (bottom - top) / max(len(wanted), 1)
    for index, condition in enumerate(wanted):
        y = top + index * row_height + 6
        draw.text((20 * S, (y + 12) * S), SHORT.get(condition, condition),
                  font=fonts["small"], fill=INK)
        for offset, (scheme, colour) in enumerate(
                (("loocv", FUSION), ("lobo", CARVE))):
            value = by_condition[condition].get(scheme)
            if value is None:
                continue
            bar_y = y + offset * 16
            draw.rectangle(
                [left * S, bar_y * S,
                 (left + min(value, hi) / hi * (right - left)) * S, (bar_y + 12) * S],
                fill=colour)
            draw.text(((left + min(value, hi) / hi * (right - left) + 8) * S,
                       (bar_y - 1) * S), f"{value:.3f}", font=fonts["small"], fill=MUTED)

    draw.text((20 * S, (height - 26) * S),
              "teal: leave one specimen out    amber: leave one capture batch out"
              "    batch membership alone beats every method under leave-one-out",
              font=fonts["small"], fill=MUTED)

    out = out or FIGURES / "batch_holdout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.LANCZOS).save(out)
    return out


def run(*, verbose: bool = True) -> list[Path]:
    """Emit every figure whose report exists. Missing reports are skipped."""
    made: list[Path] = []
    for name, builder, needs in (
        ("metric inversion", figure_inversion, "virtual_views.json"),
        ("truth vs carve vs fusion", figure_reconstruction, "virtual_views.json"),
        ("batch holdout", figure_batch_holdout, "batch_holdout.json"),
    ):
        if not (REPORTS / needs).exists():
            if verbose:
                print(f"  skipped {name}: no {needs}")
            continue
        path = builder()
        made.append(path)
        if verbose:
            print(f"  wrote {path}")
    return made


__all__ = [
    "FIGURES", "figure_batch_holdout", "figure_inversion",
    "figure_reconstruction", "run",
]
