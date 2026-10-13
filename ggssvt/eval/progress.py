"""What has been run, what has not, and when.

The project page reports whatever numbers exist on disk, which makes it easy to
lose track of which experiments produced them and which are still outstanding.
Across a long project that turns into a real problem: a table of six methods
looks complete whether or not four more were meant to be in it.

So this walks the work directory and reports the state of every experiment in
the programme, including the ones that have never run. Each entry names the
artefact that proves it ran, the command that produces it, and where it sits in
the argument. Nothing here computes a result; it only reports whether one exists,
which means it stays cheap enough to run on every page build.

The `stale` flag compares an artefact against the cache it was derived from. A
result older than the specimens it was fitted to is worse than a missing one,
because it looks finished. That has already happened once in this project, when
the ground truth changed and every trained number silently predated it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import WORK_DIR


@dataclass(frozen=True)
class Experiment:
    """One step in the programme, and how to tell whether it has run."""

    key: str
    label: str
    stage: str            # data | reconstruction | classical | learned | check
    command: str
    artefact: Path
    why: str
    needs_gpu: bool = False


@dataclass
class ExperimentStatus:
    """Whether an experiment has run, and what it produced."""

    key: str
    label: str
    stage: str
    command: str
    why: str
    needs_gpu: bool
    done: bool
    stale: bool = False
    ran_at: str | None = None
    headline: str | None = None
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "stage": self.stage,
            "command": self.command,
            "why": self.why,
            "needs_gpu": self.needs_gpu,
            "done": self.done,
            "stale": self.stale,
            "ran_at": self.ran_at,
            "headline": self.headline,
        }


def _programme(work_dir: Path) -> tuple[Experiment, ...]:
    reports = work_dir / "reports"
    return (
        Experiment(
            "preprocess", "Preprocess, geometric carve", "data",
            "python -m ggssvt.cli preprocess",
            work_dir / "cache" / "quality.json",
            "Registers, segments and carves every specimen. Everything else reads "
            "this cache, so nothing runs before it.",
        ),
        Experiment(
            "preprocess_sam3d", "Preprocess, SAM3D masks", "data",
            "python -m ggssvt.cli preprocess --segmenter sam3d "
            "--cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda",
            work_dir / "cache_sam3d" / "quality.json",
            "The segmentation arm of the factorial. Slower, and it drops three "
            "specimens the geometric gate keeps.",
            needs_gpu=True,
        ),
        Experiment(
            "views", "View-count caches, 3, 4 and 6", "data",
            "for v in 3 4 6; do python -m ggssvt.cli preprocess --views $v "
            "--cache-dir work_dirs/ggssvt/cache_v$v; done",
            work_dir / "cache_v4" / "quality.json",
            "Answers whether four images would have done. They would not.",
        ),
        Experiment(
            "fuse", "TSDF depth fusion", "reconstruction",
            "python -m ggssvt.cli fuse --write-cache",
            reports / "fusion.json",
            "The reconstruction that escapes the visual hull. Also writes the "
            "fused cache the campaign and the page both read.",
        ),
        Experiment(
            "baselines", "Classical biomass baselines", "classical",
            "python -m ggssvt.cli baselines",
            reports / "metrics.json",
            "Leave-one-out over every closed-form method. The comparison the "
            "trained model has to beat.",
        ),
        Experiment(
            "mesh", "Mesh arm", "classical",
            "python -m ggssvt.cli mesh",
            reports / "mesh.json",
            "Marching cubes, then surface area and solidity as biomass features.",
        ),
        Experiment(
            "view_ablation", "View-count ablation", "classical",
            "python -m ggssvt.cli views",
            reports / "view_ablation.json",
            "Scores the reduced-view caches on quality and on physical "
            "plausibility.",
        ),
        Experiment(
            "dino_probe", "DINO frozen-feature probe", "classical",
            "python -m ggssvt.cli dino-probe",
            reports / "dino_probe.json",
            "Does a self-supervised ViT backbone carry more than the CNN stem, "
            "before any training.",
            needs_gpu=True,
        ),
        Experiment(
            "factorial", "SAM3D by DINO factorial", "classical",
            "python -m ggssvt.cli factorial",
            reports / "factorial.json",
            "Two by three over segmenter and backbone, through the frozen probe.",
            needs_gpu=True,
        ),
        Experiment(
            "frequency", "Spectral analysis of the target", "classical",
            "python -m ggssvt.cli frequency",
            reports / "frequency.json",
            "H3's premise, measured rather than argued: what the occupancy "
            "contains against what the encoding can represent.",
        ),
        Experiment(
            "label_efficiency", "Label-efficiency curves", "classical",
            "python -m ggssvt.cli label-efficiency",
            reports / "label_efficiency.json",
            "H1's second half. How few labels each representation needs to "
            "reach the baseline's full-label accuracy.",
        ),
        Experiment(
            "viewpoint", "Held-out-view consistency", "check",
            "python -m ggssvt.cli viewpoint",
            reports / "viewpoint.json",
            "H2's viewpoint claim, on a view the reconstruction never saw. "
            "Re-projection into the views it was built from does not measure "
            "this.",
        ),
        Experiment(
            "robustness", "Noise and occlusion sweeps", "check",
            "python -m ggssvt.cli robustness",
            reports / "robustness.json",
            "H4's other two thirds. Depth noise at multiples of the sensor's "
            "own characteristic, and a band across the subject in every view.",
        ),
        Experiment(
            "reciprocity", "Reconstruction refining the masks", "reconstruction",
            "python -m ggssvt.cli reciprocity",
            reports / "reciprocity.json",
            "Closes the loop the pipeline never closed: the reconstruction is "
            "decided from twelve views and each mask from one.",
        ),
        Experiment(
            "quality", "Reconstruction metrics", "reconstruction",
            "python -m ggssvt.cli quality",
            reports / "reconstruction_quality.json",
            "Re-projection into the captured views, and Chamfer, HD95, F-score "
            "and voxel IoU between the two operators. Reported because the "
            "silhouette column ranks the worse reconstruction higher, which is "
            "the argument for the plausibility check.",
        ),
        Experiment(
            "dino_segment", "DITR-style DINO lifting", "classical",
            "python -m ggssvt.cli dino-segment",
            reports / "dino_segment.json",
            "Lifts DINOv2 patch features onto the points and clusters them, in "
            "place of DITR's supervised head. A reported negative: it does not "
            "rescue E001-E010, where one patch spans 42 mm against a 5 to 15 mm "
            "stem.",
        ),
        Experiment(
            "gate", "Acceptance gates", "check",
            "python -m ggssvt.cli gate",
            reports / "gates.json",
            "Mask area, reconstruction sanity, training loss and prediction "
            "spread, over every specimen. Blocking failures stop a run rather "
            "than propagating into a score.",
        ),
        Experiment(
            "gallery", "Reconstruction gallery", "check",
            "python -m ggssvt.cli gallery",
            work_dir / "reports" / "gallery" / "reconstructions.html",
            "Every specimen rendered. Catches what no metric does.",
        ),
        Experiment(
            "campaign", "Training campaign", "learned",
            "python -m ggssvt.campaign --plan core --device cuda "
            "--workers 8 --batch-size 2",
            work_dir / "campaign" / "summary.txt",
            "The seven runs that close H1, H2 and H3, plus the fused-cache "
            "comparison. Eight to ten hours.",
            needs_gpu=True,
        ),
        Experiment(
            "posefree", "Pose-free reconstruction", "check",
            "python -m ggssvt.cli posefree --methods fast3r dust3r mast3r "
            "--device cuda",
            reports / "posefree.json",
            "DUSt3R, MASt3R and Fast3R estimate cameras from images alone, so "
            "they are the independent check on a registration that has never "
            "been validated against a calibration target.",
            needs_gpu=True,
        ),
    )


def _headline(key: str, path: Path) -> tuple[str | None, dict]:
    """One number worth showing beside the entry, read from its artefact."""
    try:
        if path.suffix != ".json":
            return None, {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, {}

    if key == "baselines" and isinstance(data, dict):
        best = min(
            (v for v in data.values() if isinstance(v, dict) and "rmse_kg" in v),
            key=lambda v: v["rmse_kg"],
            default=None,
        )
        if best:
            return f"best RMSE {best['rmse_kg']:.3f} kg", {}
    if key == "fuse" and isinstance(data, dict):
        return f"{len(data)} specimens fused", {}
    if key == "view_ablation" and isinstance(data, list) and data:
        worst = min(data, key=lambda r: r.get("n_views", 99))
        return f"down to {worst.get('n_views')} views", {}
    if key == "frequency" and isinstance(data, dict):
        reach = data.get("encoding_reach_cycles_per_m")
        nyquist = data.get("grid_nyquist_cycles_per_m")
        if reach and nyquist:
            return f"encoding reaches {reach / nyquist:.1f}x the grid Nyquist", {}
    if key == "label_efficiency" and isinstance(data, dict):
        reached = data.get("comparison", {}).get("labels_to_reach", {})
        best = min((v for v in reached.values() if v), default=None)
        return (f"best reaches the bar with {best} labels", {}) if best else (None, {})
    if key == "viewpoint" and isinstance(data, dict):
        summary = data.get("summary", {})
        if summary.get("relative_drop") is not None:
            return f"{summary['relative_drop']:.1%} drop on a held-out view", {}
    if key == "robustness" and isinstance(data, dict):
        rows = data.get("summary", {})
        worst = max((v.get("fragments", 0) for v in rows.values()), default=0)
        return f"up to {worst} fragmented under occlusion", {}
    if key == "reciprocity" and isinstance(data, dict):
        rules = data.get("summary", {})
        best = max(rules.items(), key=lambda kv: kv[1].get("plausible", 0), default=None)
        if best:
            return f"{best[0]}: {best[1]['plausible']}/{best[1]['n']} plausible", {}
    if key == "quality" and isinstance(data, dict):
        carve = data.get("reprojection", {}).get("carve", [])
        fused = data.get("reprojection", {}).get("fused", [])
        if carve and fused:
            def mean_iou(rows):
                return sum(r["silhouette_iou"] for r in rows) / len(rows)
            return (f"silhouette IoU {mean_iou(carve):.3f} carve, "
                    f"{mean_iou(fused):.3f} fused"), {}
    if key == "dino_segment" and isinstance(data, list) and data:
        confident = sum(1 for r in data if r.get("rim_confident"))
        return f"{confident} of {len(data)} rims confident", {}
    if key == "gate" and isinstance(data, list) and data:
        blocked = sum(1 for r in data if r.get("blocked"))
        advisories = sum(len(r.get("failures", [])) for r in data) - blocked
        return f"{blocked} blocked, {max(advisories, 0)} advisory", {}
    if key == "factorial" and isinstance(data, dict):
        cells = data.get("cells", {})
        ran = sum(1 for v in cells.values() if isinstance(v, dict) and "rmse_kg" in v)
        return f"{ran} of {len(cells)} cells", {}
    return None, {}


def survey(work_dir: Path = WORK_DIR) -> list[ExperimentStatus]:
    """State of every experiment in the programme."""
    cache_stamp = (work_dir / "cache" / "quality.json")
    cache_mtime = cache_stamp.stat().st_mtime if cache_stamp.exists() else 0.0

    out: list[ExperimentStatus] = []
    for experiment in _programme(work_dir):
        exists = experiment.artefact.exists()
        ran_at = None
        stale = False
        headline = None

        if exists:
            mtime = experiment.artefact.stat().st_mtime
            ran_at = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            # Derived results must not predate the cache they were fitted to.
            stale = experiment.stage != "data" and mtime < cache_mtime
            headline, _ = _headline(experiment.key, experiment.artefact)

        out.append(
            ExperimentStatus(
                key=experiment.key,
                label=experiment.label,
                stage=experiment.stage,
                command=experiment.command,
                why=experiment.why,
                needs_gpu=experiment.needs_gpu,
                done=exists,
                stale=stale,
                ran_at=ran_at,
                headline=headline,
            )
        )
    return out


def summarise(statuses: list[ExperimentStatus]) -> dict:
    """Counts for the page header."""
    return {
        "total": len(statuses),
        "done": sum(1 for s in statuses if s.done),
        "stale": sum(1 for s in statuses if s.stale),
        "pending_gpu": sum(1 for s in statuses if not s.done and s.needs_gpu),
    }


__all__ = ["Experiment", "ExperimentStatus", "summarise", "survey"]
