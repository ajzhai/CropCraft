"""The two result figures the paper needs and the pipeline does not already emit.

Drawn with PIL rather than matplotlib, which is not installed in the working
environment and would be a heavy dependency to add for two charts. The palette
and typography match the project page and the architecture diagrams, so the
figures read as one set.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\Users\user\CropCraft")

from ggssvt.config import voxel_grid_centres  # noqa: E402
from ggssvt.data.preprocess import load_cached, usable_plant_ids  # noqa: E402
from ggssvt.eval.architecture import VIRIDIS, _font  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
S = 2                      # supersampling, so text stays crisp at print size
INK, MUTED, RULE = "#1a1a1a", "#6a6a6a", "#d8d8d8"
BAND_LO, BAND_HI = 200.0, 1000.0


def _densities():
    """Implied bulk density per specimen, carved and fused."""
    root = Path(r"C:\Users\user\CropCraft\work_dirs\ggssvt")
    gt = {
        r["plant_id"]: float(r["net_weight_g"]) / 1000.0
        for r in csv.DictReader(
            open(root.parent.parent / "dataset" / "ground_truth.csv",
                 newline="", encoding="utf-8")
        )
    }
    heights = voxel_grid_centres()[..., 2]
    rows = []
    for pid in usable_plant_ids(root / "cache"):
        mass = gt[pid]
        out = {"id": pid, "mass": mass,
               "species": "Mango" if pid.startswith("M") else "Eucalyptus"}
        for label, cache in (("carve", root / "cache"), ("fuse", root / "cache_tsdf")):
            c = load_cached(pid, cache)
            vol = float((c.occupancy & (heights > c.pot_height_m)).sum()) * c.voxel_size_m ** 3
            out[label] = mass / vol if vol > 0 else float("inf")
        rows.append(out)
    return rows


def figure_density(rows) -> Path:
    """Implied density on a log axis, carve against fusion, with the band shown."""
    W, H = 900, 250
    img = Image.new("RGB", (W * S, H * S), "white")
    d = ImageDraw.Draw(img)
    f = {k: _font(*v) for k, v in {
        "t": ("bold", 15 * S), "l": ("regular", 12 * S),
        "s": ("regular", 11 * S), "m": ("mono", 11 * S)}.items()}

    left, right, top, bot = 120, W - 40, 70, H - 52
    lo, hi = np.log10(3.0), np.log10(30000.0)

    def x(v):
        v = max(v, 3.0)
        return left + (np.log10(v) - lo) / (hi - lo) * (right - left)

    d.text((left * S, 22 * S), "Implied bulk density of the reconstructed shoot",
           font=f["t"], fill=INK)
    d.text((left * S, 44 * S),
           "Measured mass divided by above-rim volume. Fresh plant tissue is "
           "300 to 900 kg/m3.", font=f["s"], fill=MUTED)

    # The plausible band, drawn behind everything.
    d.rectangle([x(BAND_LO) * S, top * S, x(BAND_HI) * S, bot * S],
                fill="#f2f7ed")
    d.text((x(BAND_LO) * S + 6 * S, (top + 6) * S), "plausible",
           font=f["s"], fill="#4a7a3a")

    for decade in (10, 100, 1000, 10000):
        gx = x(decade)
        d.line([gx * S, top * S, gx * S, bot * S], fill=RULE, width=S)
        d.text((gx * S - 12 * S, (bot + 8) * S), f"{decade:,}", font=f["s"], fill=MUTED)
    d.text(((left + (right - left) / 2 - 60) * S, (bot + 30) * S),
           "kg per cubic metre, log scale", font=f["s"], fill=MUTED)

    lanes = [("carve", "Space carving", top + 22), ("fuse", "TSDF fusion", top + 86)]
    for key, label, y in lanes:
        d.text((30 * S, (y - 8) * S), label, font=f["l"], fill=INK)
        d.line([left * S, (y + 30) * S, right * S, (y + 30) * S], fill=RULE, width=S)
        n_ok = 0
        for r in rows:
            v = r[key]
            colour = VIRIDIS[7] if r["species"] == "Mango" else VIRIDIS[3]
            if BAND_LO <= v <= BAND_HI:
                n_ok += 1
            jitter = (hash(r["id"]) % 22) - 11
            cx, cy = x(v) * S, (y + 12 + jitter * 0.9) * S
            d.ellipse([cx - 4 * S, cy - 4 * S, cx + 4 * S, cy + 4 * S],
                      fill=colour, outline="white", width=max(1, S // 2))
        d.text((30 * S, (y + 10) * S), f"{n_ok} of {len(rows)} plausible",
               font=f["m"], fill=VIRIDIS[0])

    # Species key.
    for i, (name, tone) in enumerate((("Eucalyptus", 3), ("Mango", 7))):
        kx = right - 150 + i * 80
        d.ellipse([kx * S - 4 * S, (top + 4) * S, kx * S + 4 * S, (top + 12) * S],
                  fill=VIRIDIS[tone])
        d.text(((kx + 8) * S, (top + 2) * S), name, font=f["s"], fill=MUTED)

    path = OUT / "fig_density.png"
    img.save(path, "PNG", optimize=True)
    return path


def figure_operator() -> Path:
    """RMSE per method, carved against fused, as paired bars."""
    methods = [
        ("geometric features", 0.544, 0.335, True),
        ("volume allometric", 0.592, 0.469, True),
        ("canopy area allometric", 0.598, 0.494, False),
        ("mesh geometry", 0.507, 0.486, False),
        ("direct 2D (control)", 0.469, 0.469, False),
        ("mean predictor", 0.568, 0.568, False),
    ]
    W, H = 900, 400
    img = Image.new("RGB", (W * S, H * S), "white")
    d = ImageDraw.Draw(img)
    f = {k: _font(*v) for k, v in {
        "t": ("bold", 15 * S), "l": ("regular", 12 * S),
        "s": ("regular", 11 * S), "m": ("mono", 11 * S)}.items()}

    left, right, top = 200, W - 150, 76
    row_h, bar_h = 48, 15
    scale = (right - left) / 0.65

    d.text((30 * S, 22 * S), "Biomass RMSE, same features and protocol",
           font=f["t"], fill=INK)
    d.text((30 * S, 44 * S),
           "Only the reconstruction operator differs. Lower is better.",
           font=f["s"], fill=MUTED)

    for i, (name, carved, fused, resolved) in enumerate(methods):
        y = top + i * row_h
        d.text((30 * S, (y + 8) * S), name, font=f["l"], fill=INK)
        for j, (val, tone) in enumerate(((carved, 2), (fused, 7))):
            by = y + j * (bar_h + 3)
            d.rectangle([left * S, by * S, (left + val * scale) * S,
                         (by + bar_h) * S], fill=VIRIDIS[tone])
            d.text(((left + val * scale + 8) * S, (by + 1) * S), f"{val:.3f}",
                   font=f["m"], fill=MUTED)
        if resolved:
            d.text(((right + 46) * S, (y + 8) * S), "resolved",
                   font=f["m"], fill=VIRIDIS[0])

    ky = top - 18
    for i, (name, tone) in enumerate((("carved", 2), ("fused", 7))):
        kx = left + i * 90
        d.rectangle([kx * S, ky * S, (kx + 22) * S, (ky + 10) * S], fill=VIRIDIS[tone])
        d.text(((kx + 28) * S, (ky - 2) * S), name, font=f["s"], fill=MUTED)

    path = OUT / "fig_operator.png"
    img.save(path, "PNG", optimize=True)
    return path


if __name__ == "__main__":
    rows = _densities()
    p1 = figure_density(rows)
    p2 = figure_operator()
    for p in (p1, p2):
        print(f"  {p.name:22s} {Image.open(p).size}  {p.stat().st_size // 1024} KB")
    ok = {k: sum(1 for r in rows if BAND_LO <= r[k] <= BAND_HI)
          for k in ("carve", "fuse")}
    print("  plausible:", ok)
    json.dump(rows, (OUT / "densities.json").open("w"), indent=1, default=str)
