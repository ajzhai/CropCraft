"""Physically-grounded biomass head.

Regressing mass straight from a pooled feature vector throws away the one thing
the reconstruction earned: a metric volume. This head keeps it. Mass is predicted
as a density times an integrated occupancy volume::

    m = rho(z) integrated over the occupancy field, above the pot rim
      + a bounded learned residual

The density is not a single constant. Fresh bulk density varies systematically
with how open a canopy is -- the same litre of visual hull holds far more tissue
in a dense mango crown than in a thin eucalyptus sapling, which is visible in
this dataset as a ten-fold spread in hull density between species. So the head
predicts a density *modulation* from the pooled geometry features, around a
learnable global prior.

Two properties follow from this construction and both matter for a
label-efficient method. The volume integral is differentiable with respect to
occupancy, so the biomass loss also trains the reconstruction. And with only
thirty labelled specimens, most of the mapping is carried by physics rather than
by fitted parameters, which is what keeps the head from simply memorising the
training set.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import MODEL, POT_HEIGHT_M


class BiomassHead(nn.Module):
    """Predicts fresh above-ground mass in kilograms from occupancy and features.

    Args:
        embed_dim: width of the fused tokens.
        hidden: width of the head MLP.
        density_prior_kg_m3: initial global bulk density.
        learn_density: whether the global density is trainable.
        max_density_ratio: the density modulation is constrained to
            ``[1 / r, r]`` around the prior, so the head cannot escape the
            physical interpretation by predicting an arbitrary scale factor.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        hidden: int = MODEL.head_hidden,
        density_prior_kg_m3: float = MODEL.density_prior_kg_m3,
        learn_density: bool = MODEL.learn_density,
        max_density_ratio: float = 8.0,
    ):
        super().__init__()
        self.max_density_ratio = max_density_ratio

        self.log_density = nn.Parameter(
            torch.tensor(float(density_prior_kg_m3)).log(),
            requires_grad=learn_density,
        )

        self.pool_norm = nn.LayerNorm(embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.density_modulation = nn.Linear(hidden, 1)
        self.residual = nn.Linear(hidden, 1)

        # Start as a pure physical model: no modulation, no residual.
        nn.init.zeros_(self.density_modulation.weight)
        nn.init.zeros_(self.density_modulation.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @property
    def global_density_kg_m3(self) -> torch.Tensor:
        return self.log_density.exp()

    def forward(
        self,
        occupancy: torch.Tensor,
        query_points: torch.Tensor,
        voxel_volume_m3: float,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
        pot_height_m: float | torch.Tensor = POT_HEIGHT_M,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            occupancy: ``(B, Q)`` occupancy probabilities for the query points.
            query_points: ``(B, Q, 3)`` world coordinates of those points.
            voxel_volume_m3: volume each query point stands for. Query points
                must be a uniform sample of the working volume for the integral
                to be meaningful.
            tokens: ``(B, N, C)`` fused view tokens.
            token_valid: ``(B, N)``.
            pot_height_m: heights below this are pot and soil, not plant.
                Either a scalar or ``(B,)``, one rim per specimen -- pot mass
                spans 0.7-32 kg across the three batches, so a single cut
                height is only ever right for one of them.

        Returns:
            Dict with ``mass_kg``, ``volume_m3``, ``density_kg_m3`` and
            ``residual_kg``, each ``(B,)`` except ``volume_m3``.
        """
        if torch.is_tensor(pot_height_m):
            rim = pot_height_m.to(query_points.device).reshape(-1, 1)
        else:
            rim = pot_height_m
        above_pot = (query_points[..., 2] > rim).to(occupancy.dtype)
        weighted = occupancy * above_pot

        volume = weighted.sum(dim=1) * voxel_volume_m3            # (B,)

        # Simple shape descriptors, computed from the same occupancy field so the
        # density modulation stays tied to the reconstruction.
        total = weighted.sum(dim=1).clamp(min=1e-6)
        height = (weighted * query_points[..., 2]).sum(dim=1) / total
        radial = torch.linalg.norm(query_points[..., :2], dim=-1)
        spread = (weighted * radial).sum(dim=1) / total
        compactness = volume / (spread.pow(2) * height * torch.pi).clamp(min=1e-9)

        mask = token_valid.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = self.pool_norm(pooled)

        descriptors = torch.stack([height, spread, compactness.clamp(max=10.0)], dim=-1)
        features = self.trunk(torch.cat([pooled, descriptors], dim=-1))

        log_ratio = torch.tanh(self.density_modulation(features).squeeze(-1))
        ratio = (log_ratio * torch.log(torch.tensor(self.max_density_ratio))).exp()
        density = self.global_density_kg_m3 * ratio

        residual = self.residual(features).squeeze(-1)

        mass = density * volume + residual
        return {
            "mass_kg": mass,
            "volume_m3": volume,
            "density_kg_m3": density,
            "residual_kg": residual,
            "mean_height_m": height,
            "mean_spread_m": spread,
        }


__all__ = ["BiomassHead"]
