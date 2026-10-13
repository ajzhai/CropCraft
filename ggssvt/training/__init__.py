"""Training: dataset assembly, losses, and the two-stage schedule."""

from .dataset import SamplingConfig, SpecimenBatch, SpecimenDataset, collate
from .losses import LossTerms, biomass_loss, compute_loss, occupancy_loss
from .trainer import (
    FoldResult,
    TrainingRun,
    loocv,
    predict,
    resolve_device,
    train_stage,
)

__all__ = [
    "FoldResult",
    "LossTerms",
    "SamplingConfig",
    "SpecimenBatch",
    "SpecimenDataset",
    "TrainingRun",
    "biomass_loss",
    "collate",
    "compute_loss",
    "loocv",
    "occupancy_loss",
    "predict",
    "resolve_device",
    "train_stage",
]
