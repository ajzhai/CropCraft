#!/usr/bin/env python
"""Build a browsable map of campaign fold metrics and reconstruction panels."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from ggssvt.config import WORK_DIR
from ggssvt.eval.viz import VizConfig, save


LAYERS = ("rgb", "segmentation", "depth", "occupancy", "points")


def _folds(campaign: Path) -> list[tuple[str, Path]]:
    return sorted(
        (path.stem, path)
        for path in campaign.glob("folds/fold_*.json")
        if path.is_file()
    )


def _run_folds(campaign: Path, run: str) -> list[tuple[str, Path]]:
    folder = campaign / f"{run}_folds"
    return sorted(
        (path.stem, path)
        for path in folder.glob("fold_*.json")
        if path.is_file()
    )


def build(campaign: Path, output: Path, *, size: int, source: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_paths = {
        path.stem: path
        for path in campaign.glob("*.json")
        if not path.name.endswith("_folds.json")
    }
    run_names = set(run_paths)
    run_names.update(
        path.name.removesuffix("_folds")
        for path in campaign.glob("*_folds")
        if path.is_dir()
    )
    for run in sorted(run_names):
        run_path = run_paths.get(run)
        if run_path is None:
            summary = {"name": run, "status": "in_progress"}
        else:
            try:
                summary = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary = {"name": run, "status": "unreadable"}
        run_rows = []
        for fold_name, fold_path in _run_folds(campaign, run):
            fold = json.loads(fold_path.read_text(encoding="utf-8"))
            held_out = fold["held_out"]
            image = output / run / f"{held_out}_{'_'.join(LAYERS)}.png"
            if not image.exists():
                save(
                    held_out,
                    image,
                    VizConfig(
                        layers=LAYERS,
                        size=size,
                        source=source,
                        cache_root=WORK_DIR,
                    ),
                )
            row = {
                "run": run,
                "fold": fold_name,
                "held_out": held_out,
                "prediction": fold["predicted_kg"],
                "target": fold["target_kg"],
                "error": fold["predicted_kg"] - fold["target_kg"],
                "occupancy_iou": fold["occupancy_iou"],
                "occupancy_ap": fold["occupancy_ap"],
                "image": image.relative_to(output).as_posix(),
            }
            rows.append(row)
            run_rows.append(row)
        summary["fold_count"] = len(run_rows)
        summary["folds"] = run_rows
        (output / run / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (output / run / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)
    lines = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>GG-SSVT campaign fold visualizations</title>",
        "<style>body{font:14px system-ui;margin:2rem} table{border-collapse:collapse}"
        "td,th{padding:.35rem;border-bottom:1px solid #ddd} img{width:360px}"
        ".run{margin:2rem 0} .bad{background:#fff0f0}</style>",
        "<h1>GG-SSVT campaign fold visualizations</h1>",
        f"<p>Layers: {', '.join(LAYERS)}; source: {html.escape(source)}; "
        f"render size: {size}px. Generated from completed fold checkpoints.</p>",
    ]
    for run, run_rows in grouped.items():
        lines.append(f"<section class='run'><h2>{html.escape(run)}</h2><table>")
        lines.append(
            "<tr><th>Fold</th><th>Held out</th><th>Prediction</th>"
            "<th>Target</th><th>Error</th><th>IoU</th><th>AP</th><th>Visualization</th></tr>"
        )
        for row in run_rows:
            cls = " class='bad'" if abs(row["error"]) >= 0.5 else ""
            lines.append(
                f"<tr{cls}><td>{html.escape(row['fold'])}</td>"
                f"<td>{html.escape(row['held_out'])}</td>"
                f"<td>{row['prediction']:.3f}</td><td>{row['target']:.3f}</td>"
                f"<td>{row['error']:+.3f}</td><td>{row['occupancy_iou']:.3f}</td>"
                f"<td>{row['occupancy_ap']:.3f}</td>"
                f"<td><a href='{html.escape(row['image'])}'>"
                f"<img loading='lazy' src='{html.escape(row['image'])}'></a></td></tr>"
            )
        lines.append("</table></section>")
    (output / "index.html").write_text("\n".join(lines), encoding="utf-8")
    return output / "index.html"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=WORK_DIR / "campaign")
    parser.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "campaign_visualizations"
    )
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--source", choices=("carve", "fused"), default="carve")
    args = parser.parse_args()
    path = build(args.campaign, args.out, size=args.size, source=args.source)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
