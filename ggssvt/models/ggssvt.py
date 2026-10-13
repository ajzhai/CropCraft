"""The GG-SSVT model.

Assembles the four components into one network:

1. :class:`~ggssvt.models.encoder.ViewEncoder` -- RGB-D patches to tokens,
   pruned to the subject and grounded with Fourier back-projected positions.
2. :class:`~ggssvt.models.attention.CrossViewFusion` -- 3D-distance-biased
   attention across all views' tokens.
3. :class:`~ggssvt.models.decoder.OccupancyDecoder` -- fused tokens plus a world
   coordinate to an occupancy logit.
4. :class:`~ggssvt.models.head.BiomassHead` -- occupancy integrated to a volume,
   then to mass through a modulated density.

Stage 1 trains 1-3 against space-carved pseudo-labels, which need no manual
annotation. Stage 2 adds the head and the thirty weighed specimens. The split is
the label-efficiency claim: everything expensive to supervise is learned without
supervision, and the labelled data is spent only on the last, smallest mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import MODEL, POT_HEIGHT_M, VOXEL_SIZE_M, ModelConfig
from .attention import CrossViewFusion
from .decoder import OccupancyDecoder
from .encoder import ViewEncoder
from .head import BiomassHead


@dataclass
class GGSSVTOutput:
    """Everything one forward pass produces."""

    occupancy_logits: torch.Tensor          # (B, Q)
    tokens: torch.Tensor                    # (B, N, C)
    anchors: torch.Tensor                   # (B, N, 3)
    token_valid: torch.Tensor               # (B, N)
    biomass: dict[str, torch.Tensor] | None = None

    @property
    def occupancy(self) -> torch.Tensor:
        return self.occupancy_logits.sigmoid()

    @property
    def mass_kg(self) -> torch.Tensor | None:
        return None if self.biomass is None else self.biomass["mass_kg"]


class GGSSVT(nn.Module):
    """Geometry-grounded self-supervised volumetric transformer.

    Args:
        config: architecture settings. Defaults to :data:`ggssvt.config.MODEL`.
        tokens_per_view: subject patches kept per view.
        geometry_grounded: set False for the ablation that removes both the
            Fourier back-projected positions and the 3D-distance attention bias,
            leaving an otherwise identical vanilla multi-view transformer.
    """

    def __init__(
        self,
        config: ModelConfig = MODEL,
        tokens_per_view: int = 64,
        geometry_grounded: bool = True,
    ):
        super().__init__()
        self.config = config
        self.geometry_grounded = geometry_grounded

        self.encoder = ViewEncoder(
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            depth=config.encoder_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            tokens_per_view=tokens_per_view,
            backbone=config.backbone,
            backbone_variant=config.backbone_variant,
            freeze_backbone=config.freeze_backbone,
        )
        self.fusion = CrossViewFusion(
            embed_dim=config.embed_dim,
            depth=config.fusion_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            bias_scale=config.distance_bias_scale,
            learn_bias=config.learn_distance_bias and geometry_grounded,
            use_checkpointing=config.use_checkpointing,
        )
        self.decoder = OccupancyDecoder(
            embed_dim=config.embed_dim,
            hidden=config.decoder_hidden,
            depth=config.decoder_depth,
            num_heads=config.num_heads,
            query_chunk=config.query_chunk,
            n_bands=config.fourier_bands,
            max_freq=config.fourier_max_freq,
        )
        self.head = BiomassHead(
            embed_dim=config.embed_dim,
            hidden=config.head_hidden,
            density_prior_kg_m3=config.density_prior_kg_m3,
            learn_density=config.learn_density,
        )

        if not geometry_grounded:
            self._disable_geometry_grounding()

    def _disable_geometry_grounding(self) -> None:
        """Zero and freeze every geometric pathway, for the ablation.

        Removes the distance bias from all attention (by driving gamma to zero)
        and the back-projected positional code from the tokens, so what remains
        is a multi-view transformer with identical capacity and no 3D prior.
        """
        for module in self.modules():
            if hasattr(module, "log_distance_scale"):
                with torch.no_grad():
                    module.log_distance_scale.fill_(float("-inf"))
                module.log_distance_scale.requires_grad_(False)

        grounding = self.encoder.grounding
        for parameter in grounding.project.parameters():
            with torch.no_grad():
                parameter.zero_()
            parameter.requires_grad_(False)

    def encode(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        points_world: torch.Tensor,
        subject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and fuse the views into a flat token set.

        Returns:
            ``(tokens, anchors, valid)`` of shapes ``(B, N, C)``, ``(B, N, 3)``
            and ``(B, N)``, where ``N = views * tokens_per_view``.
        """
        tokens, anchors, valid = self.encoder(rgb, depth, points_world, subject)

        batch, views, per_view, width = tokens.shape
        tokens = tokens.reshape(batch, views * per_view, width)
        anchors = anchors.reshape(batch, views * per_view, 3)
        valid = valid.reshape(batch, views * per_view)

        tokens = self.fusion(tokens, anchors, valid, key_padding_mask=~valid)
        return tokens, anchors, valid

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        points_world: torch.Tensor,
        subject: torch.Tensor,
        query_points: torch.Tensor,
        *,
        predict_biomass: bool = False,
        voxel_volume_m3: float = VOXEL_SIZE_M ** 3,
        pot_height_m: float | torch.Tensor = POT_HEIGHT_M,
        chunk: int | None = None,
    ) -> GGSSVTOutput:
        """
        Args:
            rgb: ``(B, V, 3, H, W)`` in [0, 1].
            depth: ``(B, V, 1, H, W)`` metres.
            points_world: ``(B, V, 3, H, W)`` world coordinates per pixel.
            subject: ``(B, V, 1, H, W)`` subject mask.
            query_points: ``(B, Q, 3)`` world coordinates to evaluate.
            predict_biomass: also run the biomass head. Requires
                ``query_points`` to be a uniform sample of the working volume.
            voxel_volume_m3: volume represented by each query point.

        Returns:
            A :class:`GGSSVTOutput`.
        """
        tokens, anchors, valid = self.encode(rgb, depth, points_world, subject)
        logits = self.decoder(query_points, tokens, anchors, valid, chunk=chunk)

        biomass = None
        if predict_biomass:
            biomass = self.head(
                logits.sigmoid(),
                query_points,
                voxel_volume_m3,
                tokens,
                valid,
                pot_height_m=pot_height_m,
            )

        return GGSSVTOutput(
            occupancy_logits=logits,
            tokens=tokens,
            anchors=anchors,
            token_valid=valid,
            biomass=biomass,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        points_world: torch.Tensor,
        subject: torch.Tensor,
        grid_points: torch.Tensor,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Occupancy probabilities over an arbitrary grid, evaluated in chunks.

        Args:
            grid_points: ``(B, Q, 3)``, typically a flattened voxel grid.

        Returns:
            ``(B, Q)`` probabilities.
        """
        tokens, anchors, valid = self.encode(rgb, depth, points_world, subject)
        logits = self.decoder(grid_points, tokens, anchors, valid, chunk=chunk)
        return logits.sigmoid()

    def parameter_groups(self, stage: str) -> list[dict]:
        """Parameter groups for the two training stages.

        ``"pretrain"`` trains the encoder, fusion and decoder against the carved
        occupancy. ``"finetune"`` adds the head and lowers the learning rate on
        everything already trained, so thirty labels cannot undo what the
        self-supervised stage learned.
        """
        backbone = [
            p
            for p in list(self.encoder.parameters()) + list(self.fusion.parameters())
            if p.requires_grad
        ]
        decoder = list(self.decoder.parameters())
        head = list(self.head.parameters())

        if stage == "pretrain":
            return [{"params": backbone + decoder, "lr_scale": 1.0}]
        if stage == "finetune":
            return [
                {"params": backbone, "lr_scale": 0.1},
                {"params": decoder, "lr_scale": 0.3},
                {"params": head, "lr_scale": 1.0},
            ]
        raise ValueError(f"unknown stage {stage!r}; expected 'pretrain' or 'finetune'")

    def n_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)


__all__ = ["GGSSVT", "GGSSVTOutput"]
