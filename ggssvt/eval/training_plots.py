"""Render saved training histories as dependency-light PNG plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _plot(history: dict, output: Path) -> None:
    series = []
    for stage in ("pretrain", "finetune"):
        run = history.get(stage)
        if run and run.get("epochs"):
            values = [e["losses"].get("total", 0.0) for e in run["epochs"]]
            series.append((stage, values))
    if not series:
        raise ValueError("history contains no epoch losses")

    width, height = 900, 520
    left, right, top, bottom = 80, 30, 55, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    all_values = [v for _, values in series for v in values]
    lo, hi = min(all_values), max(all_values)
    span = max(hi - lo, 1e-9)
    x_span = width - left - right
    y_span = height - top - bottom

    draw.text((left, 18), f"Training loss: {history.get('held_out', '')}", fill="black")
    draw.line((left, top, left, height - bottom), fill="#555")
    draw.line((left, height - bottom, width - right, height - bottom), fill="#555")
    colours = {"pretrain": "#2878b5", "finetune": "#c44e52"}
    for stage, values in series:
        points = []
        for i, value in enumerate(values):
            x = left + (i / max(1, len(values) - 1)) * x_span
            y = top + (hi - value) / span * y_span
            points.append((round(x), round(y)))
        if len(points) > 1:
            draw.line(points, fill=colours[stage], width=3)
        draw.text((width - 150, top + 25 * len(colours)), stage, fill=colours[stage])
        colours.pop(stage)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.history.read_text(encoding="utf-8"))
    _plot(payload, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
