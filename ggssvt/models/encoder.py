"""Per-view token encoder with geometry-driven token selection.

Box 3 of the system diagram. Three design points matter here.

**A swappable appearance backbone.** The stem is one of the interchangeable
backbones in :mod:`ggssvt.models.backbones` -- a trainable RGB-D CNN (no DINO),
DINOv2, or DINOv3. Everything downstream is identical across the three, so the
difference between their results is attributable to the backbone alone.

**RGB-D patches, not RGB patches.** Every condition sees colour, metric depth and
a validity channel. Depth is what makes a token's 3D anchor computable at all,
and the validity channel stops the network reading the Kinect's zero-return holes
as "surface at range zero" -- those holes are common on the thin stems and dark
pots in this capture set. The frozen DINO backbones cannot ingest depth through
their RGB trunk, so they take it through a parallel stem instead; see
:class:`~ggssvt.models.backbones._DinoBackbone`.

**Subject-driven token pruning.** The plant occupies about four percent of each
512x424 frame; the rest is greenhouse floor and background. Keeping every patch
would mean roughly ten thousand tokens per specimen, whose pairwise distance
matrix does not fit the 4 GB laptop GPU this work is developed on, and would
spend almost all of the attention budget on concrete. Because the views are
registered, the subject mask is already known, so the encoder keeps the ``K``
patches per view with the most subject coverage and discards the rest. This is
not a heuristic bolted on for speed -- it is only available *because* the model
is geometry-grounded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MODEL
from .backbones import Backbone, build_backbone
from .embedding import GeometryGroundedEmbedding, patch_centroids


def _select_top_patches(
    scores: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices of the ``k`` highest-scoring patches per view.

    Args:
        scores: ``(B, V, P)`` subject coverage per patch.
        k: patches to keep.

    Returns:
        ``(index, kept_score)``, both ``(B, V, k)``.
    """
    k = min(k, scores.shape[-1])
    kept_score, index = scores.topk(k, dim=-1)
    return index, kept_score


def _gather_tokens(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather ``(B, V, P, D)`` along the patch axis with a ``(B, V, k)`` index."""
    expanded = index.unsqueeze(-1).expand(*index.shape, source.shape[-1])
    return source.gather(2, expanded)


class ViewEncoder(nn.Module):
    """Encodes every view to a pruned set of geometry-grounded tokens.

    Args:
        embed_dim: token width.
        patch_size: patch side in pixels, for the ``cnn`` backbone only.
        depth: transformer layers applied within each view, before fusion.
        tokens_per_view: how many subject patches to keep per view.
        backbone: ``"cnn"``, ``"dinov2"`` or ``"dinov3"``.
        backbone_variant: DINO size, ``"small"`` / ``"base"`` / ``"large"``.
        freeze_backbone: keep DINO weights fixed.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        patch_size: int = MODEL.patch_size,
        depth: int = MODEL.encoder_depth,
        num_heads: int = MODEL.num_heads,
        mlp_ratio: float = MODEL.mlp_ratio,
        dropout: float = MODEL.dropout,
        tokens_per_view: int = 64,
        backbone: str = MODEL.backbone,
        backbone_variant: str = MODEL.backbone_variant,
        freeze_backbone: bool = MODEL.freeze_backbone,
    ):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.backbone_kind = backbone

        self.stem: Backbone = build_backbone(
            backbone,
            embed_dim=embed_dim,
            patch_size=patch_size,
            variant=backbone_variant,
            freeze=freeze_backbone,
        )
        self.patch_size = self.stem.patch_size

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.within_view = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.grounding = GeometryGroundedEmbedding(embed_dim=embed_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        points_world: torch.Tensor,
        subject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: ``(B, V, 3, H, W)`` in [0, 1].
            depth: ``(B, V, 1, H, W)`` metres, 0 where invalid.
            points_world: ``(B, V, 3, H, W)`` world coordinates per pixel.
            subject: ``(B, V, 1, H, W)`` subject mask, float or bool.

        Returns:
            ``(tokens, anchors, anchor_valid)`` of shapes ``(B, V, K, C)``,
            ``(B, V, K, 3)`` and ``(B, V, K)``.
        """
        batch, views = rgb.shape[:2]
        height, width = rgb.shape[-2:]
        subject = subject.to(rgb.dtype)
        valid = (depth > 0).to(rgb.dtype)

        flat_rgb = rgb.reshape(batch * views, 3, height, width)
        flat_depth = depth.reshape(batch * views, 1, height, width)
        flat_valid = valid.reshape(batch * views, 1, height, width)

        features = self.stem(flat_rgb, flat_depth, flat_valid)
        grid_h, grid_w = features.shape[-2:]
        n_patches = grid_h * grid_w

        tokens = features.reshape(batch, views, -1, n_patches).permute(0, 1, 3, 2)

        # Anchor each token on its *subject* pixels only. Averaging over every
        # valid pixel instead lets a patch that straddles the plant edge take
        # its 3D position partly from the greenhouse wall metres behind, which
        # drags the anchor off the specimen and corrupts the distance bias that
        # the whole architecture rests on.
        centroids, patch_valid = patch_centroids(
            points_world, (depth > 0) & (subject > 0), (grid_h, grid_w)
        )
        coverage = F.adaptive_avg_pool2d(
            (subject * valid).reshape(batch * views, 1, height, width), (grid_h, grid_w)
        ).reshape(batch, views, n_patches)

        index, kept_coverage = _select_top_patches(coverage, self.tokens_per_view)
        tokens = _gather_tokens(tokens, index)
        anchors = _gather_tokens(centroids, index)
        anchor_valid = patch_valid.gather(2, index) & (kept_coverage > 0)

        tokens = self.grounding(tokens, anchors, anchor_valid)

        # Refine within each view before the views are allowed to talk.
        shape = tokens.shape
        tokens = self.within_view(tokens.reshape(batch * views, shape[2], shape[3]))
        tokens = tokens.reshape(shape)

        return tokens, anchors, anchor_valid


__all__ = ["ViewEncoder"]
