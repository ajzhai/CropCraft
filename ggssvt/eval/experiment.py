"""The backbone comparison: with DINO versus without.

Two experiments, at very different costs, answering related questions.

:func:`probe_experiment`
    Frozen features, linear probe, leave-one-out. Runs on a CPU in minutes and
    measures how much biomass information sits in the pretrained representation
    itself. Use it to decide whether the GPU experiment is worth running.

:func:`backbone_experiment`
    The full thing: GG-SSVT pretrained and cross-validated once per backbone,
    everything downstream of the stem held identical. Needs a GPU.

Both report a **paired bootstrap against the no-DINO control** rather than two
point estimates side by side. At twenty-eight specimens the sampling noise on
RMSE is larger than any plausible backbone effect, so an unpaired comparison
would let almost any ranking look meaningful. The paired interval is the number
that decides whether the difference is real.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import MODEL, TRAIN, WORK_DIR, ModelConfig, TrainConfig
from .baselines import load_features
from .metrics import RegressionMetrics, paired_bootstrap_difference, regression_metrics

CONTROL_LABEL = "cnn (no DINO)"


@dataclass
class ConditionResult:
    """One backbone condition."""

    label: str
    metrics: RegressionMetrics | None
    predictions: np.ndarray | None = None
    extras: dict = field(default_factory=dict)
    skipped_reason: str = ""

    @property
    def ran(self) -> bool:
        return self.metrics is not None


@dataclass
class ComparisonReport:
    """Every condition, plus the paired comparisons against the control."""

    conditions: dict[str, ConditionResult]
    targets: np.ndarray
    control: str = CONTROL_LABEL
    n_specimens: int = 0

    def paired_against_control(self, metric: str = "rmse_kg") -> dict[str, dict]:
        """Paired bootstrap of every condition against the control."""
        if self.control not in self.conditions or not self.conditions[self.control].ran:
            return {}

        baseline = self.conditions[self.control].predictions
        out: dict[str, dict] = {}
        for label, condition in self.conditions.items():
            if label == self.control or not condition.ran:
                continue
            out[label] = paired_bootstrap_difference(
                condition.predictions, baseline, self.targets, metric
            )
        return out

    def to_table(self) -> str:
        header = (
            f"{'condition':34s} {'RMSE':>7s} {'MAE':>7s} "
            f"{'MARE%':>7s} {'R2':>7s} {'n_feat':>7s}"
        )
        lines = [header, "-" * len(header)]
        for label, condition in self.conditions.items():
            if not condition.ran:
                lines.append(f"{label:34s} {'skipped':>7s}")
                continue
            m = condition.metrics
            features = condition.extras.get("n_features", "")
            lines.append(
                f"{label:34s} {m.rmse_kg:7.3f} {m.mae_kg:7.3f} "
                f"{m.mare * 100:7.1f} {m.r2:7.3f} {features!s:>7s}"
            )

        paired = self.paired_against_control()
        if paired:
            lines += [
                "",
                f"Paired bootstrap against '{self.control}'",
                "(negative favours the DINO condition; the interval decides)",
            ]
            for label, d in paired.items():
                verdict = (
                    "significant"
                    if d["high"] < 0 or d["low"] > 0
                    else "NOT significant"
                )
                lines.append(
                    f"  {label:32s} dRMSE {d['difference']:+.3f} kg  "
                    f"95% CI [{d['low']:+.3f}, {d['high']:+.3f}]  "
                    f"p~{d['p_direction']:.2f}  {verdict}"
                )

        skipped = [
            (label, c.skipped_reason)
            for label, c in self.conditions.items()
            if not c.ran and c.skipped_reason
        ]
        for label, reason in skipped:
            lines += ["", f"SKIPPED {label}:", reason]

        return "\n".join(lines)

    def to_json(self) -> str:
        payload = {
            "n_specimens": self.n_specimens,
            "control": self.control,
            "conditions": {
                label: (
                    {
                        **c.metrics.as_dict(),
                        **c.extras,
                        "predictions": c.predictions.tolist(),
                    }
                    if c.ran
                    else {"skipped": c.skipped_reason}
                )
                for label, c in self.conditions.items()
            },
            "paired_vs_control": self.paired_against_control(),
        }
        return json.dumps(payload, indent=2)


def probe_experiment(
    plant_ids: list[str],
    *,
    backbones: tuple[str, ...] = ("dinov2", "dinov3"),
    variant: str = "base",
    cache_dir: Path = WORK_DIR / "cache",
    n_components: int = 8,
    alpha: float = 1.0,
    verbose: bool = True,
) -> ComparisonReport:
    """Frozen-feature linear probe: DINO backbones against no-DINO geometry."""
    from .dino_probe import run_probe_experiment

    results, skipped = run_probe_experiment(
        plant_ids,
        backbones=backbones,
        variant=variant,
        cache_dir=cache_dir,
        n_components=n_components,
        alpha=alpha,
        verbose=verbose,
    )
    targets = np.array([f.target_kg for f in load_features(plant_ids, cache_dir)])

    conditions: dict[str, ConditionResult] = {}
    for name, result in results.items():
        label = CONTROL_LABEL if name.startswith("geometry only") else name
        conditions[label] = ConditionResult(
            label=label,
            metrics=result.metrics,
            predictions=result.predictions,
            extras={
                "n_features": result.n_features,
                "n_components": result.n_components,
            },
        )

    for note in skipped:
        kind, _, reason = note.partition(": ")
        conditions[kind] = ConditionResult(
            label=kind, metrics=None, skipped_reason=reason
        )

    return ComparisonReport(
        conditions=conditions, targets=targets, n_specimens=len(plant_ids)
    )


def backbone_experiment(
    plant_ids: list[str],
    *,
    backbones: tuple[str, ...] = ("cnn", "dinov2", "dinov3"),
    variant: str = "base",
    cache_dir: Path = WORK_DIR / "cache",
    model_config: ModelConfig = MODEL,
    train_config: TrainConfig = TRAIN,
    epochs: int | None = None,
    tokens_per_view: int = 64,
    out_dir: Path = WORK_DIR / "experiments",
    device=None,
    verbose: bool = True,
) -> ComparisonReport:
    """Train and cross-validate GG-SSVT once per backbone.

    Everything except the stem is held identical across conditions, so the
    difference in results is attributable to the backbone alone. Pretraining is
    re-run per condition -- reusing one checkpoint across backbones would be
    meaningless, since the stems produce different features.
    """
    import torch

    from ..models.backbones import backbone_is_available
    from ..models.ggssvt import GGSSVT
    from ..training.dataset import SpecimenDataset
    from ..training.trainer import loocv, resolve_device, train_stage

    device = device or resolve_device(train_config.device)
    epochs = train_config.pretrain_epochs if epochs is None else epochs
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = np.array([f.target_kg for f in load_features(plant_ids, cache_dir)])
    conditions: dict[str, ConditionResult] = {}

    for kind in backbones:
        label = CONTROL_LABEL if kind == "cnn" else f"{kind}-{variant}"

        available, reason = backbone_is_available(kind, variant)
        if not available:
            if verbose:
                print(f"\n=== {label}: SKIPPED ===\n{reason}\n")
            conditions[label] = ConditionResult(
                label=label, metrics=None, skipped_reason=reason
            )
            continue

        if verbose:
            print(f"\n=== {label} ===")

        config = dataclasses.replace(
            model_config, backbone=kind, backbone_variant=variant
        )
        model = GGSSVT(config=config, tokens_per_view=tokens_per_view)
        if verbose:
            print(
                f"  {model.n_parameters(False) / 1e6:.1f}M parameters, "
                f"{model.n_parameters(True) / 1e6:.1f}M trainable, device {device}"
            )

        train_stage(
            model,
            SpecimenDataset(plant_ids, cache_dir=cache_dir, mode="occupancy"),
            stage="pretrain",
            epochs=epochs,
            config=train_config,
            device=device,
            log_every=max(1, epochs // 6),
            verbose=verbose,
        )

        checkpoint = out_dir / f"pretrain_{kind}_{variant}.pt"
        torch.save(
            {"state_dict": model.state_dict(), "backbone": kind, "variant": variant},
            checkpoint,
        )

        folds = loocv(
            plant_ids,
            cache_dir=cache_dir,
            model_config=config,
            train_config=train_config,
            tokens_per_view=tokens_per_view,
            pretrained_state=model.state_dict(),
            device=device,
            verbose=False,
        )

        predicted = np.array([f.predicted_kg for f in folds])
        metrics = regression_metrics(predicted, np.array([f.target_kg for f in folds]))

        conditions[label] = ConditionResult(
            label=label,
            metrics=metrics,
            predictions=predicted,
            extras={
                "n_parameters": model.n_parameters(False),
                "n_trainable": model.n_parameters(True),
                "occupancy_ap": float(np.mean([f.occupancy_ap for f in folds])),
                "occupancy_best_iou": float(
                    np.mean([f.occupancy_best_iou for f in folds])
                ),
                "checkpoint": str(checkpoint),
            },
        )
        if verbose:
            print(f"  {metrics}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return ComparisonReport(
        conditions=conditions, targets=targets, n_specimens=len(plant_ids)
    )


__all__ = [
    "CONTROL_LABEL",
    "ComparisonReport",
    "ConditionResult",
    "backbone_experiment",
    "probe_experiment",
]
