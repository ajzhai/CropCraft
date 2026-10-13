"""The full factorial: SAM3D on/off crossed with the appearance backbone.

Two components are claimed to help, and they act at different stages:

* **SAM3D** replaces the geometric cylinder mask. It changes what counts as the
  subject, so it changes the carved occupancy, the self-supervision targets, the
  token pruning and the geometric features -- everything downstream.
* **DINO** replaces the appearance stem. It changes only how pixels become
  tokens.

Because they act at different stages they can interact: a better mask feeds
cleaner patches to the backbone, and a stronger backbone may or may not need the
better mask. A pair of one-factor-at-a-time ablations cannot see that. The full
factorial can, and it is cheap here because the segmenter is a preprocessing
choice, cached once per condition.

Conditions, for segmenters ``S`` and backbones ``B``, are the |S| x |B| grid::

    geometric x cnn      -- neither (the control)
    geometric x dinov2   -- DINO only
    sam3d     x cnn      -- SAM3D only
    sam3d     x dinov2   -- both
    ... plus dinov3 rows once access is granted

Effects are reported three ways, all paired against the same control:

* **main effect of DINO** -- averaged over segmenters
* **main effect of SAM3D** -- averaged over backbones
* **interaction** -- whether the two together do more (or less) than the sum of
  their parts

At twenty-eight specimens every one of these carries a confidence interval wider
than the effect itself. The harness reports the intervals rather than the point
estimates alone, because with four conditions and a small sample the chance of
one arrangement looking good by luck is substantial.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from .metrics import RegressionMetrics, paired_bootstrap_difference, regression_metrics

SEGMENTERS = ("geometric", "sam3d")
BACKBONES = ("cnn", "dinov2", "dinov3")

CACHE_DIRS = {
    "geometric": WORK_DIR / "cache",
    "sam3d": WORK_DIR / "cache_sam3d",
}


def condition_label(segmenter: str, backbone: str, variant: str = "") -> str:
    """Human-readable name for one cell of the factorial."""
    sam = "SAM3D" if segmenter == "sam3d" else "no SAM3D"
    if backbone == "cnn":
        dino = "no DINO"
    else:
        dino = backbone if not variant else f"{backbone}-{variant}"
    return f"{sam} + {dino}"


@dataclass
class Cell:
    """One condition of the factorial."""

    segmenter: str
    backbone: str
    variant: str
    label: str
    metrics: RegressionMetrics | None = None
    predictions: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
    skipped_reason: str = ""

    @property
    def ran(self) -> bool:
        return self.metrics is not None

    @property
    def has_dino(self) -> bool:
        return self.backbone != "cnn"

    @property
    def has_sam(self) -> bool:
        return self.segmenter == "sam3d"


@dataclass
class FactorialReport:
    """Every cell, the main effects, and the interaction."""

    cells: dict[str, Cell]
    targets: np.ndarray
    n_specimens: int = 0
    control_label: str = ""
    notes: list[str] = field(default_factory=list)

    def ran_cells(self) -> dict[str, Cell]:
        return {k: c for k, c in self.cells.items() if c.ran}

    def _find(self, has_sam: bool, has_dino: bool) -> Cell | None:
        for cell in self.ran_cells().values():
            if cell.has_sam == has_sam and cell.has_dino == has_dino:
                return cell
        return None

    def effects(self, metric: str = "rmse_kg") -> dict[str, dict]:
        """Main effects and the interaction, each with a paired interval.

        Every comparison is paired on specimens, so the intervals describe the
        *difference*, not the spread of either condition on its own.
        """
        neither = self._find(False, False)
        dino_only = self._find(False, True)
        sam_only = self._find(True, False)
        both = self._find(True, True)

        out: dict[str, dict] = {}

        if neither and dino_only:
            out["DINO alone (vs neither)"] = paired_bootstrap_difference(
                dino_only.predictions, neither.predictions, self.targets, metric
            )
        if neither and sam_only:
            out["SAM3D alone (vs neither)"] = paired_bootstrap_difference(
                sam_only.predictions, neither.predictions, self.targets, metric
            )
        if neither and both:
            out["both (vs neither)"] = paired_bootstrap_difference(
                both.predictions, neither.predictions, self.targets, metric
            )
        if sam_only and both:
            out["DINO given SAM3D"] = paired_bootstrap_difference(
                both.predictions, sam_only.predictions, self.targets, metric
            )
        if dino_only and both:
            out["SAM3D given DINO"] = paired_bootstrap_difference(
                both.predictions, dino_only.predictions, self.targets, metric
            )

        if all((neither, dino_only, sam_only, both)):
            out["interaction"] = self._interaction(
                neither, dino_only, sam_only, both, metric
            )

        return out

    def _interaction(
        self, neither: Cell, dino: Cell, sam: Cell, both: Cell, metric: str
    ) -> dict:
        """Is the combination more than the sum of its parts?

        Bootstraps the quantity ``(both - dino) - (sam - neither)``. A negative
        value means DINO helps *more* when SAM3D is already in place, i.e. the
        two are synergistic; positive means they are partly redundant.
        """

        def score(predictions: np.ndarray, index: np.ndarray) -> float:
            return regression_metrics(
                predictions[index], self.targets[index]
            ).as_dict()[metric]

        rng = np.random.default_rng(0)
        n = self.targets.size
        observed = (
            score(both.predictions, np.arange(n)) - score(dino.predictions, np.arange(n))
        ) - (
            score(sam.predictions, np.arange(n))
            - score(neither.predictions, np.arange(n))
        )

        samples = []
        for _ in range(3000):
            index = rng.integers(0, n, n)
            if np.unique(self.targets[index]).size < 2:
                continue
            try:
                samples.append(
                    (score(both.predictions, index) - score(dino.predictions, index))
                    - (
                        score(sam.predictions, index)
                        - score(neither.predictions, index)
                    )
                )
            except (ValueError, ZeroDivisionError):
                continue

        if not samples:
            return {"difference": observed, "low": float("nan"), "high": float("nan")}

        values = np.array(samples)
        return {
            "difference": float(observed),
            "low": float(np.quantile(values, 0.025)),
            "high": float(np.quantile(values, 0.975)),
            "p_direction": float(
                2.0 * min((values > 0).mean(), (values < 0).mean())
            ),
        }

    def to_table(self) -> str:
        header = (
            f"{'condition':30s} {'SAM3D':>6s} {'DINO':>7s} {'RMSE':>7s} "
            f"{'MAE':>7s} {'MARE%':>7s} {'R2':>7s}"
        )
        lines = [header, "-" * len(header)]

        for cell in self.cells.values():
            sam = "yes" if cell.has_sam else "no"
            dino = (
                "no"
                if not cell.has_dino
                else (cell.variant and f"{cell.backbone[-1]}-{cell.variant[:1]}" or cell.backbone)
            )
            if not cell.ran:
                lines.append(f"{cell.label:30s} {sam:>6s} {dino:>7s} {'skipped':>7s}")
                continue
            m = cell.metrics
            lines.append(
                f"{cell.label:30s} {sam:>6s} {dino:>7s} {m.rmse_kg:7.3f} "
                f"{m.mae_kg:7.3f} {m.mare * 100:7.1f} {m.r2:7.3f}"
            )

        effects = self.effects()
        if effects:
            lines += [
                "",
                "Paired effects on RMSE (negative = the addition helps)",
                "An interval spanning zero means the sample cannot resolve it.",
            ]
            for name, d in effects.items():
                verdict = (
                    "significant"
                    if d.get("high", 0) < 0 or d.get("low", 0) > 0
                    else "not resolved"
                )
                lines.append(
                    f"  {name:28s} {d['difference']:+.3f} kg  "
                    f"95% CI [{d['low']:+.3f}, {d['high']:+.3f}]  {verdict}"
                )

        skipped = [c for c in self.cells.values() if not c.ran and c.skipped_reason]
        for cell in skipped:
            lines += ["", f"SKIPPED {cell.label}:", cell.skipped_reason]

        lines += self.notes
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "n_specimens": self.n_specimens,
                "cells": {
                    label: (
                        {
                            "segmenter": c.segmenter,
                            "backbone": c.backbone,
                            "variant": c.variant,
                            **c.metrics.as_dict(),
                            **c.extras,
                            "predictions": c.predictions.tolist(),
                        }
                        if c.ran
                        else {
                            "segmenter": c.segmenter,
                            "backbone": c.backbone,
                            "skipped": c.skipped_reason,
                        }
                    )
                    for label, c in self.cells.items()
                },
                "effects": self.effects(),
                "notes": self.notes,
            },
            indent=2,
        )


def _cache_available(segmenter: str, cache_dir: Path | None = None) -> tuple[bool, str, Path]:
    """Whether a segmenter's preprocessed cache exists."""
    path = cache_dir or CACHE_DIRS[segmenter]
    if (path / "quality.json").exists():
        return True, "", path
    return (
        False,
        f"no cache at {path}. Build it with:\n"
        f"  python -m ggssvt.cli preprocess --segmenter {segmenter} "
        f"--cache-dir {path}",
        path,
    )


def probe_factorial(
    *,
    segmenters: tuple[str, ...] = SEGMENTERS,
    backbones: tuple[str, ...] = ("cnn", "dinov2", "dinov3"),
    variant: str = "base",
    n_components: int = 8,
    alpha: float = 1.0,
    cache_dirs: dict[str, Path] | None = None,
    verbose: bool = True,
) -> FactorialReport:
    """Run the factorial with the frozen-feature probe. CPU, minutes.

    The ``cnn`` backbone here means "no DINO": the probe uses the hand-built
    geometric descriptors from that segmenter's reconstruction, which is the
    honest no-DINO condition for a linear probe.
    """
    from ..data.preprocess import usable_plant_ids
    from ..eval.baselines import load_features
    from ..eval.dino_probe import build_descriptors, loocv_probe
    from ..models.backbones import backbone_is_available

    cache_dirs = cache_dirs or {}
    cells: dict[str, Cell] = {}
    notes: list[str] = []
    targets: np.ndarray | None = None
    shared_ids: list[str] | None = None

    # Use the specimens that pass the quality gate under *every* segmenter, so
    # the conditions are compared on the same plants rather than on different
    # subsets that happen to survive each pipeline.
    available: dict[str, Path] = {}
    for segmenter in segmenters:
        ok, reason, path = _cache_available(segmenter, cache_dirs.get(segmenter))
        if not ok:
            notes.append(f"\n{segmenter}: {reason}")
            continue
        available[segmenter] = path
        ids = set(usable_plant_ids(path))
        shared_ids = sorted(ids if shared_ids is None else set(shared_ids) & ids)

    if not available or not shared_ids:
        return FactorialReport(cells={}, targets=np.array([]), notes=notes)

    if verbose:
        print(f"Factorial over {len(shared_ids)} specimens shared by all segmenters\n")

    for segmenter, backbone in itertools.product(segmenters, backbones):
        label = condition_label(segmenter, backbone, variant if backbone != "cnn" else "")

        if segmenter not in available:
            cells[label] = Cell(
                segmenter, backbone, variant, label,
                skipped_reason=f"no cache for segmenter '{segmenter}'",
            )
            continue

        cache_dir = available[segmenter]
        features = load_features(shared_ids, cache_dir)
        if targets is None:
            targets = np.array([f.target_kg for f in features])

        if backbone == "cnn":
            matrix = np.stack([f.geometric_vector() for f in features])
            components = min(n_components, matrix.shape[1])
        else:
            ok, reason = backbone_is_available(backbone, variant)
            if not ok:
                cells[label] = Cell(
                    segmenter, backbone, variant, label, skipped_reason=reason
                )
                if verbose:
                    print(f"  {label}: skipped ({reason.splitlines()[0]})")
                continue
            if verbose:
                print(f"  {label}: extracting features...")
            matrix = build_descriptors(
                shared_ids, backbone, variant=variant, cache_dir=cache_dir, verbose=False
            ).features
            components = n_components

        predictions = loocv_probe(
            matrix, targets, n_components=components, alpha=alpha
        )
        cells[label] = Cell(
            segmenter,
            backbone,
            variant if backbone != "cnn" else "",
            label,
            metrics=regression_metrics(predictions, targets),
            predictions=predictions,
            extras={"n_features": int(matrix.shape[1]), "n_components": components},
        )
        if verbose:
            print(f"  {label}: {cells[label].metrics}")

    return FactorialReport(
        cells=cells,
        targets=targets if targets is not None else np.array([]),
        n_specimens=len(shared_ids),
        control_label=condition_label("geometric", "cnn"),
        notes=notes,
    )


def train_factorial(
    *,
    segmenters: tuple[str, ...] = SEGMENTERS,
    backbones: tuple[str, ...] = ("cnn", "dinov2", "dinov3"),
    variant: str = "base",
    train_config=None,
    epochs: int | None = None,
    tokens_per_view: int = 64,
    out_dir: Path = WORK_DIR / "experiments",
    cache_dirs: dict[str, Path] | None = None,
    device=None,
    verbose: bool = True,
) -> FactorialReport:
    """The factorial with full GG-SSVT training. Needs a GPU.

    One pretrain plus one leave-one-out sweep per cell. With two segmenters and
    three backbones that is six training runs, so budget accordingly -- start
    with :func:`probe_factorial` to decide whether the grid is worth it, and
    trim ``--backbones``/``--segmenters`` to the cells that matter.
    """
    import dataclasses

    import torch

    from ..config import MODEL, TRAIN
    from ..data.preprocess import usable_plant_ids
    from ..eval.baselines import load_features
    from ..models.backbones import backbone_is_available
    from ..models.ggssvt import GGSSVT
    from ..training.dataset import SpecimenDataset
    from ..training.trainer import loocv, resolve_device, train_stage

    cache_dirs = cache_dirs or {}
    train_config = train_config or TRAIN
    device = device or resolve_device(train_config.device)
    epochs = train_config.pretrain_epochs if epochs is None else epochs
    out_dir.mkdir(parents=True, exist_ok=True)

    available: dict[str, Path] = {}
    shared_ids: list[str] | None = None
    notes: list[str] = []

    for segmenter in segmenters:
        ok, reason, path = _cache_available(segmenter, cache_dirs.get(segmenter))
        if not ok:
            notes.append(f"\n{segmenter}: {reason}")
            continue
        available[segmenter] = path
        ids = set(usable_plant_ids(path))
        shared_ids = sorted(ids if shared_ids is None else set(shared_ids) & ids)

    if not available or not shared_ids:
        return FactorialReport(cells={}, targets=np.array([]), notes=notes)

    targets = np.array(
        [f.target_kg for f in load_features(shared_ids, next(iter(available.values())))]
    )
    cells: dict[str, Cell] = {}

    for segmenter, backbone in itertools.product(segmenters, backbones):
        label = condition_label(segmenter, backbone, variant if backbone != "cnn" else "")

        if segmenter not in available:
            cells[label] = Cell(
                segmenter, backbone, variant, label,
                skipped_reason=f"no cache for segmenter '{segmenter}'",
            )
            continue

        ok, reason = backbone_is_available(backbone, variant)
        if not ok:
            cells[label] = Cell(
                segmenter, backbone, variant, label, skipped_reason=reason
            )
            if verbose:
                print(f"\n=== {label}: SKIPPED ===\n{reason}\n")
            continue

        if verbose:
            print(f"\n=== {label} ===")

        cache_dir = available[segmenter]
        config = dataclasses.replace(
            MODEL, backbone=backbone, backbone_variant=variant
        )
        model = GGSSVT(config=config, tokens_per_view=tokens_per_view)

        train_stage(
            model,
            SpecimenDataset(shared_ids, cache_dir=cache_dir, mode="occupancy"),
            stage="pretrain",
            epochs=epochs,
            config=train_config,
            device=device,
            log_every=max(1, epochs // 6),
            verbose=verbose,
        )

        checkpoint = out_dir / f"pretrain_{segmenter}_{backbone}_{variant}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "segmenter": segmenter,
                "backbone": backbone,
                "variant": variant,
            },
            checkpoint,
        )

        folds = loocv(
            shared_ids,
            cache_dir=cache_dir,
            model_config=config,
            train_config=train_config,
            tokens_per_view=tokens_per_view,
            pretrained_state=model.state_dict(),
            device=device,
            verbose=False,
        )
        predicted = np.array([f.predicted_kg for f in folds])
        metrics = regression_metrics(
            predicted, np.array([f.target_kg for f in folds])
        )

        cells[label] = Cell(
            segmenter,
            backbone,
            variant if backbone != "cnn" else "",
            label,
            metrics=metrics,
            predictions=predicted,
            extras={
                "n_parameters": model.n_parameters(False),
                "n_trainable": model.n_parameters(True),
                "occupancy_ap": float(np.mean([f.occupancy_ap for f in folds])),
                "checkpoint": str(checkpoint),
            },
        )
        if verbose:
            print(f"  {metrics}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return FactorialReport(
        cells=cells,
        targets=targets,
        n_specimens=len(shared_ids),
        control_label=condition_label("geometric", "cnn"),
        notes=notes,
    )


__all__ = [
    "BACKBONES",
    "CACHE_DIRS",
    "SEGMENTERS",
    "Cell",
    "FactorialReport",
    "condition_label",
    "probe_factorial",
    "train_factorial",
]
