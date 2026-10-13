"""Cross-view geometric attention.

Standard self-attention over the concatenated tokens of twelve views is both
expensive and unstructured: every token can attend to every other, and nothing
tells the network that two tokens describing the same twig should interact more
strongly than two tokens a metre apart.

GG-SSVT biases the attention logits by 3D distance::

    logit_ij = (q_i . k_j) / sqrt(d)  -  gamma * ||x_i - x_j||^2

where ``x_i`` is the token's world-frame anchor from
:mod:`ggssvt.models.embedding` and ``gamma`` is a learned, strictly positive
inverse length scale. The bias is soft, not a hard neighbourhood: distant tokens
stay reachable when the appearance term is strong enough, which matters for a
plant whose branches occlude one another. Learning ``gamma`` lets each head pick
its own scale -- in practice early heads settle on wide scales that fuse the
whole specimen and later ones on narrow scales that resolve individual stems.

Tokens whose patch had no valid depth carry no meaningful anchor, so they take
no distance bias at all and fall back to pure appearance attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Imported explicitly. `torch.utils.checkpoint` is a submodule, and whether a
# bare `import torch` also binds it varies by version: it does on 2.13, it does
# not on 2.5.1+cu121. Relying on that is a crash on the first training step,
# because use_checkpointing defaults to True and train_stage puts the model in
# training mode, which is the exact path this line guards.
import torch.utils.checkpoint
from torch import nn

from ..config import MODEL


class CrossViewGeometricAttention(nn.Module):
    """Multi-head attention over all views' tokens, biased by 3D distance.

    Args:
        embed_dim: token width.
        num_heads: attention heads. Each learns its own distance scale.
        bias_scale: initial value of the inverse length scale, in 1/m^2.
        learn_bias: whether the scale is trainable.
        dropout: attention dropout.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        num_heads: int = MODEL.num_heads,
        bias_scale: float = MODEL.distance_bias_scale,
        learn_bias: bool = MODEL.learn_distance_bias,
        dropout: float = MODEL.dropout,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim {embed_dim} must divide evenly among {num_heads} heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out = nn.Linear(embed_dim, embed_dim)

        # Parameterised in log space so the scale stays positive under any step.
        log_scale = torch.full((num_heads,), math.log(max(bias_scale, 1e-6)))
        self.log_distance_scale = nn.Parameter(log_scale, requires_grad=learn_bias)

    @property
    def distance_scale(self) -> torch.Tensor:
        """Per-head inverse length scale, ``gamma``, strictly positive."""
        return self.log_distance_scale.exp()

    # Set by `capture_attention`; None the rest of the time, so the check is one
    # identity comparison per forward and nothing is retained by default. The
    # weights are (B, heads, N, N) and N is 12 views by however many tokens, so
    # holding them unconditionally would be a real memory cost.
    _capture: list | None = None

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        anchor_valid: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: ``(B, N, C)`` tokens from every view, already flattened.
            anchors: ``(B, N, 3)`` world-frame anchor per token.
            anchor_valid: ``(B, N)`` True where the anchor is meaningful.
            key_padding_mask: ``(B, N)`` True for tokens to ignore entirely.

        Returns:
            ``(B, N, C)``.
        """
        batch, n_tokens, _ = tokens.shape

        qkv = self.qkv(tokens).reshape(batch, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]   # (B, H, N, Dh)

        logits = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Squared world distance between every token pair, (B, N, N).
        squared = torch.cdist(anchors, anchors, p=2.0).pow(2)

        # Only penalise pairs where both anchors are real.
        pair_valid = anchor_valid.unsqueeze(2) & anchor_valid.unsqueeze(1)
        squared = squared.masked_fill(~pair_valid, 0.0)

        gamma = self.distance_scale.view(1, self.num_heads, 1, 1)
        logits = logits - gamma * squared.unsqueeze(1)

        if key_padding_mask is not None:
            logits = logits.masked_fill(
                key_padding_mask.view(batch, 1, 1, n_tokens), float("-inf")
            )

        weights = logits.softmax(dim=-1)
        # Stash before dropout: dropout is a training regulariser, and what an
        # interpretability figure should show is what the layer attends to, not
        # which of those attentions a particular step happened to zero.
        if self._capture is not None:
            self._capture.append(weights.detach())
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        fused = (weights @ values).transpose(1, 2).reshape(batch, n_tokens, self.embed_dim)
        return self.out(fused)


class FusionBlock(nn.Module):
    """Pre-norm transformer block using :class:`CrossViewGeometricAttention`."""

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        num_heads: int = MODEL.num_heads,
        mlp_ratio: float = MODEL.mlp_ratio,
        dropout: float = MODEL.dropout,
        bias_scale: float = MODEL.distance_bias_scale,
        learn_bias: bool = MODEL.learn_distance_bias,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = CrossViewGeometricAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias_scale=bias_scale,
            learn_bias=learn_bias,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        anchor_valid: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(
            self.norm1(tokens), anchors, anchor_valid, key_padding_mask
        )
        return tokens + self.mlp(self.norm2(tokens))


class CrossViewFusion(nn.Module):
    """A stack of :class:`FusionBlock` layers, plus the final norm."""

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        depth: int = MODEL.fusion_depth,
        num_heads: int = MODEL.num_heads,
        mlp_ratio: float = MODEL.mlp_ratio,
        dropout: float = MODEL.dropout,
        bias_scale: float = MODEL.distance_bias_scale,
        learn_bias: bool = MODEL.learn_distance_bias,
        use_checkpointing: bool = MODEL.use_checkpointing,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.blocks = nn.ModuleList(
            FusionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                bias_scale=bias_scale,
                learn_bias=learn_bias,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        anchors: torch.Tensor,
        anchor_valid: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block,
                    tokens,
                    anchors,
                    anchor_valid,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, anchors, anchor_valid, key_padding_mask)
        return self.norm(tokens)

    def distance_scales(self) -> torch.Tensor:
        """Per-block, per-head learned inverse length scales, ``(depth, heads)``.

        Read this after training: it is the direct evidence for whether the
        geometry bias is doing anything, and it is what the ablation in the
        dissertation's Section 4.4 compares against.
        """
        return torch.stack([b.attention.distance_scale.detach() for b in self.blocks])


__all__ = ["CrossViewFusion", "CrossViewGeometricAttention", "FusionBlock"]
