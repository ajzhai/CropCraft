"""Two-stage training and the leave-one-out evaluation protocol.

**Stage 1, self-supervised pretraining.** The encoder, fusion stack and
occupancy decoder are trained against the space-carved pseudo-labels. No weighed
mass is touched. This is the stage that consumes almost all of the compute and
all of the data.

**Stage 2, biomass fine-tuning.** The head is attached and the thirty weighed
specimens are used, with a reduced learning rate on everything stage 1 already
trained.

**Evaluation.** Thirty specimens is too few for a held-out test split to say
anything, so the protocol is leave-one-out cross-validation, as in the
predecessor work on this rig. Stage 2 is re-run per fold on the other
twenty-nine; stage 1 is not, by default.

That default is a deliberate, declarable choice. Pretraining once on every
specimen means the held-out plant's *images* were seen during self-supervised
training, though never its mass. This is standard transductive self-supervision
and it is what makes thirty-specimen LOOCV affordable at all, but it is not the
same as a strictly inductive protocol, and reporting it as such would be wrong.
Pass ``strict=True`` to re-pretrain per fold and get the inductive number; the
gap between the two is itself worth reporting.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import MODEL, TRAIN, WORK_DIR, ModelConfig, TrainConfig
from ..eval.metrics import average_precision, best_threshold_iou
from ..models.ggssvt import GGSSVT
from .dataset import SpecimenBatch, SpecimenDataset, collate
from .losses import compute_loss


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU when CUDA was asked for but is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _carved_above_ground_volume(batch: SpecimenBatch, volume_per_query_m3: float) -> torch.Tensor:
    """Above-pot carved volume for each item, from its query labels."""
    rim = batch.pot_height_m.to(batch.query_points.device).reshape(-1, 1)
    above = (batch.query_points[..., 2] > rim).to(batch.query_labels.dtype)
    return (batch.query_labels * above).sum(dim=1) * volume_per_query_m3


@dataclass
class EpochLog:
    """One epoch of training."""

    epoch: int
    stage: str
    losses: dict[str, float]
    seconds: float
    learning_rate: float


@dataclass
class TrainingRun:
    """The record of one training run."""

    stage: str
    epochs: list[EpochLog] = field(default_factory=list)
    device: str = "cpu"
    n_parameters: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "stage": self.stage,
                "device": self.device,
                "n_parameters": self.n_parameters,
                "epochs": [asdict(e) for e in self.epochs],
            },
            indent=2,
        )


def _build_optimiser(
    model: GGSSVT, stage: str, config: TrainConfig
) -> torch.optim.Optimizer:
    base_lr = config.lr_pretrain if stage == "pretrain" else config.lr_finetune
    groups = []
    for group in model.parameter_groups(stage):
        params = [p for p in group["params"] if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": base_lr * group["lr_scale"]})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _learning_rate_factor(epoch: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay."""
    if warmup > 0 and epoch < warmup:
        return (epoch + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch, on every device.

    ``TrainConfig.seed`` existed for a long time and nothing ever consumed it,
    which is worse than having no seed at all: the config advertises
    reproducibility, so nobody checks, and two runs of the same command produce
    different weights, different shuffling and different dropout. For a project
    whose headline result is a 0.209 kg difference with a bootstrap interval,
    being unable to reproduce your own number is not a small thing.

    Note this does not make CUDA bitwise deterministic. cuDNN picks algorithms by
    benchmarking and some reductions are atomic, so runs stay close rather than
    identical. Forcing full determinism costs real speed and is not worth it
    here; seeding is what removes the large, gratuitous variation.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Give each dataloader worker its own deterministic stream."""
    import random

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_stage(
    model: GGSSVT,
    dataset: SpecimenDataset,
    *,
    stage: str,
    epochs: int,
    config: TrainConfig = TRAIN,
    device: torch.device | None = None,
    log_every: int = 10,
    verbose: bool = True,
) -> TrainingRun:
    """Run one training stage to completion.

    Args:
        stage: ``"pretrain"`` or ``"finetune"``.
        epochs: epochs to run.
        log_every: print progress every this many epochs. 0 silences it.

    Returns:
        The :class:`TrainingRun` record.
    """
    device = device or resolve_device(config.device)
    seed_everything(config.seed)
    model.to(device).train()

    # A seeded generator, so shuffling is reproducible independently of whatever
    # else has consumed the global RNG by the time this stage starts.
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    optimiser = _build_optimiser(model, stage, config)
    base_lrs = [group["lr"] for group in optimiser.param_groups]

    use_amp = config.amp if hasattr(config, "amp") else MODEL.amp
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run = TrainingRun(
        stage=stage, device=str(device), n_parameters=model.n_parameters()
    )

    for epoch in range(epochs):
        started = time.time()
        factor = _learning_rate_factor(epoch, epochs, config.warmup_epochs)
        for group, base in zip(optimiser.param_groups, base_lrs):
            group["lr"] = base * factor

        totals: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(
                    batch.rgb,
                    batch.depth,
                    batch.points_world,
                    batch.subject,
                    batch.query_points,
                    predict_biomass=(stage == "finetune"),
                    voxel_volume_m3=dataset.volume_per_query_m3,
                    pot_height_m=batch.pot_height_m,
                )
                carved = (
                    _carved_above_ground_volume(batch, dataset.volume_per_query_m3)
                    if stage == "finetune"
                    else None
                )
                loss = compute_loss(
                    output, batch, stage=stage, config=config, carved_volume_m3=carved
                )

            scaler.scale(loss.total).backward()
            if config.grad_clip:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            for key, value in loss.items().items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

        averaged = {k: v / max(1, n_batches) for k, v in totals.items()}
        run.epochs.append(
            EpochLog(
                epoch=epoch,
                stage=stage,
                losses=averaged,
                seconds=time.time() - started,
                learning_rate=optimiser.param_groups[0]["lr"],
            )
        )

        if verbose and log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            terms = "  ".join(f"{k}={v:.4f}" for k, v in averaged.items())
            print(
                f"  [{stage}] epoch {epoch + 1:3d}/{epochs}  {terms}  "
                f"({time.time() - started:.1f}s)"
            )

    return run


@torch.no_grad()
def predict(
    model: GGSSVT,
    dataset: SpecimenDataset,
    *,
    device: torch.device | None = None,
    config: TrainConfig = TRAIN,
) -> dict[str, dict]:
    """Predict mass and occupancy for every specimen in a dataset.

    Returns:
        Per plant id: ``mass_kg``, ``target_kg``, ``volume_m3``,
        ``density_kg_m3`` and the occupancy probabilities at the query points.
    """
    device = device or resolve_device(config.device)
    model.to(device).eval()

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate
    )
    results: dict[str, dict] = {}

    # `.eval()` only switches dropout and normalisation; it does not stop
    # autograd. Without no_grad this loop built a graph over every intermediate
    # activation of a full query grid and held it, which is where the campaign
    # died: pretraining finished its 120 epochs, the first evaluation forward
    # allocated 14 GiB of activations nobody would ever backpropagate through,
    # and the next 576 MiB request failed. The size did not depend on the
    # backbone, which is what gave it away -- a 19.3M CNN and a 105.9M ViT both
    # failed at exactly the same number.
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            output = model(
                batch.rgb,
                batch.depth,
                batch.points_world,
                batch.subject,
                batch.query_points,
                predict_biomass=True,
                voxel_volume_m3=dataset.volume_per_query_m3,
                pot_height_m=batch.pot_height_m,
                # The decoder already knows how to evaluate in pieces, and
                # nothing was asking it to.
                chunk=MODEL.query_chunk,
            )
        for index, plant_id in enumerate(batch.plant_id):
            results[plant_id] = {
                "mass_kg": float(output.biomass["mass_kg"][index]),
                "target_kg": float(batch.target_kg[index]),
                "volume_m3": float(output.biomass["volume_m3"][index]),
                "density_kg_m3": float(output.biomass["density_kg_m3"][index]),
                "occupancy": output.occupancy[index].detach().cpu().numpy(),
                "occupancy_labels": batch.query_labels[index].detach().cpu().numpy(),
            }
    return results


@dataclass
class FoldResult:
    """One leave-one-out fold."""

    held_out: str
    predicted_kg: float
    target_kg: float
    volume_m3: float
    density_kg_m3: float
    occupancy_iou: float
    occupancy_ap: float = float("nan")
    occupancy_best_iou: float = float("nan")

    @property
    def absolute_error_kg(self) -> float:
        return abs(self.predicted_kg - self.target_kg)

    @property
    def relative_error(self) -> float:
        return self.absolute_error_kg / max(self.target_kg, 1e-6)


def _occupancy_iou(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    predicted = probabilities > threshold
    truth = labels > 0.5
    union = np.logical_or(predicted, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted, truth).sum() / union)


def loocv(
    plant_ids: list[str],
    *,
    cache_dir: Path = WORK_DIR / "cache",
    model_config: ModelConfig = MODEL,
    train_config: TrainConfig = TRAIN,
    tokens_per_view: int = 64,
    geometry_grounded: bool = True,
    strict: bool = False,
    pretrained_state: dict | None = None,
    resume_dir: Path | None = None,
    history_dir: Path | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[FoldResult]:
    """Leave-one-out cross-validation over the labelled specimens.

    Args:
        strict: re-run self-supervised pretraining inside each fold, excluding
            the held-out specimen entirely. Slower by roughly the number of
            folds, and gives the inductive rather than transductive number.
        pretrained_state: a stage-1 checkpoint to start every fold from. Ignored
            when ``strict``.
        resume_dir: directory where completed fold results are stored. Existing
            fold files are loaded and skipped, allowing a campaign to resume
            after a process or power interruption.
        history_dir: optional directory for per-fold epoch-loss histories.

    Returns:
        One :class:`FoldResult` per specimen.
    """
    device = device or resolve_device(train_config.device)
    results: list[FoldResult] = []
    if resume_dir is not None:
        resume_dir.mkdir(parents=True, exist_ok=True)
    if history_dir is not None:
        history_dir.mkdir(parents=True, exist_ok=True)

    for fold, held_out in enumerate(plant_ids, start=1):
        fold_path = resume_dir / f"fold_{fold:03d}_{held_out}.json" if resume_dir else None
        if fold_path is not None and fold_path.exists():
            try:
                stored = FoldResult(**json.loads(fold_path.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                stored = None
            if stored is not None and stored.held_out == held_out:
                results.append(stored)
                if verbose:
                    print(f"[fold {fold}/{len(plant_ids)}] {held_out}: already done, skipping")
                continue

        train_ids = [pid for pid in plant_ids if pid != held_out]

        if verbose:
            print(f"[fold {fold}/{len(plant_ids)}] holding out {held_out}")

        model = GGSSVT(
            config=model_config,
            tokens_per_view=tokens_per_view,
            geometry_grounded=geometry_grounded,
        )

        if strict:
            pretrain_set = SpecimenDataset(
                train_ids, cache_dir=cache_dir, mode="occupancy"
            )
            pretrain_run = train_stage(
                model,
                pretrain_set,
                stage="pretrain",
                epochs=train_config.pretrain_epochs,
                config=train_config,
                device=device,
                verbose=verbose,
                log_every=0,
            )
        elif pretrained_state is not None:
            model.load_state_dict(pretrained_state, strict=False)
            pretrain_run = None
        else:
            pretrain_run = None

        finetune_set = SpecimenDataset(train_ids, cache_dir=cache_dir, mode="biomass")
        finetune_run = train_stage(
            model,
            finetune_set,
            stage="finetune",
            epochs=train_config.finetune_epochs,
            config=train_config,
            device=device,
            verbose=verbose,
            log_every=0,
        )
        if history_dir is not None:
            history = {
                "held_out": held_out,
                "pretrain": None if pretrain_run is None else json.loads(pretrain_run.to_json()),
                "finetune": json.loads(finetune_run.to_json()),
            }
            path = history_dir / f"fold_{fold:03d}_{held_out}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
            temporary.replace(path)

        held_out_set = SpecimenDataset([held_out], cache_dir=cache_dir, mode="biomass")
        prediction = predict(model, held_out_set, device=device, config=train_config)[held_out]

        result = FoldResult(
            held_out=held_out,
            predicted_kg=prediction["mass_kg"],
            target_kg=prediction["target_kg"],
            volume_m3=prediction["volume_m3"],
            density_kg_m3=prediction["density_kg_m3"],
            occupancy_iou=_occupancy_iou(
                prediction["occupancy"], prediction["occupancy_labels"]
            ),
            occupancy_ap=average_precision(
                prediction["occupancy"], prediction["occupancy_labels"]
            ),
            occupancy_best_iou=best_threshold_iou(
                prediction["occupancy"], prediction["occupancy_labels"]
            )[0],
        )
        results.append(result)
        if fold_path is not None:
            temporary = fold_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(asdict(result), indent=2), encoding="utf-8"
            )
            temporary.replace(fold_path)

        if verbose:
            print(
                f"    predicted {result.predicted_kg:.3f} kg vs "
                f"{result.target_kg:.3f} kg  "
                f"(error {result.relative_error * 100:.1f}%, "
                f"occupancy IoU {result.occupancy_iou:.3f} / "
                f"best {result.occupancy_best_iou:.3f}, "
                f"AP {result.occupancy_ap:.3f})"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


__all__ = [
    "EpochLog",
    "FoldResult",
    "TrainingRun",
    "loocv",
    "predict",
    "resolve_device",
    "train_stage",
]
