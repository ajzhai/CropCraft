"""Losses for the two training stages.

Stage 1 is purely geometric: a class-weighted occupancy loss against the carved
pseudo-labels. Stage 2 adds mass supervision and a consistency term.

The consistency term deserves a note. Nothing in the occupancy loss stops the
network from inflating the field slightly to make the mass regression easier --
with thirty labels that is a real risk, and it would quietly destroy the
reconstruction while improving the headline biomass number. The volume
consistency term anchors the predicted volume to the carved volume, so the head
has to earn accuracy through the density modulation rather than by rewriting the
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import TRAIN


@dataclass
class LossTerms:
    """Individual loss terms and their weighted total."""

    total: torch.Tensor
    occupancy: torch.Tensor
    biomass: torch.Tensor | None = None
    volume_consistency: torch.Tensor | None = None

    def items(self) -> dict[str, float]:
        out = {
            "total": self.total.detach().item(),
            "occupancy": self.occupancy.detach().item(),
        }
        if self.biomass is not None:
            out["biomass"] = self.biomass.detach().item()
        if self.volume_consistency is not None:
            out["volume_consistency"] = self.volume_consistency.detach().item()
        return out


def occupancy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float = TRAIN.occupancy_pos_weight,
) -> torch.Tensor:
    """Binary cross-entropy against carved occupancy, weighted toward the positives.

    Even after balanced sampling the positives are the harder and rarer class,
    and an unweighted loss converges to a field that is systematically too thin
    -- which for a branching plant means losing whole twigs.
    """
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)


def biomass_loss(
    predicted_kg: torch.Tensor,
    target_kg: torch.Tensor,
    *,
    relative: bool = True,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Mass regression loss.

    Defaults to a relative (Huber-on-log-ratio) form. The specimens span
    0.40 kg to 2.35 kg, nearly a factor of six, so an absolute loss would let
    the heaviest few dominate the gradient and leave the light eucalyptus
    specimens systematically overpredicted.
    """
    if not relative:
        return F.smooth_l1_loss(predicted_kg, target_kg)

    safe_pred = predicted_kg.clamp(min=eps)
    safe_target = target_kg.clamp(min=eps)
    return F.smooth_l1_loss(safe_pred.log(), safe_target.log(), beta=0.1)


def volume_consistency_loss(
    predicted_volume_m3: torch.Tensor,
    carved_volume_m3: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Keep the predicted volume near the carved volume, in log space."""
    predicted = predicted_volume_m3.clamp(min=eps).log()
    carved = carved_volume_m3.clamp(min=eps).log()
    return F.smooth_l1_loss(predicted, carved, beta=0.2)


def compute_loss(
    output,
    batch,
    *,
    stage: str,
    config=TRAIN,
    carved_volume_m3: torch.Tensor | None = None,
) -> LossTerms:
    """Assemble the loss for one step.

    Args:
        output: a :class:`~ggssvt.models.ggssvt.GGSSVTOutput`.
        batch: a :class:`~ggssvt.training.dataset.SpecimenBatch`.
        stage: ``"pretrain"`` or ``"finetune"``.
        carved_volume_m3: ``(B,)`` carved above-ground volume, needed for the
            consistency term during fine-tuning.
    """
    occupancy = occupancy_loss(
        output.occupancy_logits, batch.query_labels, config.occupancy_pos_weight
    )

    if stage == "pretrain":
        return LossTerms(total=config.lambda_occupancy * occupancy, occupancy=occupancy)

    if output.biomass is None:
        raise ValueError("fine-tuning requires a forward pass with predict_biomass=True")

    mass = biomass_loss(output.biomass["mass_kg"], batch.target_kg)
    total = config.lambda_occupancy * occupancy + config.lambda_biomass * mass

    consistency = None
    if carved_volume_m3 is not None and config.lambda_volume_consistency > 0:
        consistency = volume_consistency_loss(
            output.biomass["volume_m3"], carved_volume_m3
        )
        total = total + config.lambda_volume_consistency * consistency

    return LossTerms(
        total=total,
        occupancy=occupancy,
        biomass=mass,
        volume_consistency=consistency,
    )


__all__ = [
    "LossTerms",
    "biomass_loss",
    "compute_loss",
    "occupancy_loss",
    "volume_consistency_loss",
]
