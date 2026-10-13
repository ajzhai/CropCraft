"""Implicit occupancy decoder.

The reconstruction is an implicit field rather than an explicit voxel grid: the
decoder answers "is this world coordinate inside the plant?" for any query point,
so resolution is a property of how densely you sample at inference rather than of
the network's output layer. That matters for thin branching structures, where a
fixed 128^3 output would quantise a 6 mm stem into a single voxel column.

A query attends to the fused tokens with the same 3D-distance bias used in the
encoder's fusion stack, so the features that decide a point's occupancy are the
ones anchored near it in space. Queries are processed in chunks, which is what
keeps peak memory flat: evaluating a full 128^3 grid is 2.1 million queries, and
materialising their attention maps at once would need tens of gigabytes.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import MODEL
from .embedding import FourierFeatures, normalise_world

# logit(0.01): the empty-space prior the decoder starts from.
PRIOR_OCCUPANCY_LOGIT = -4.595


class QueryAttention(nn.Module):
    """Cross-attention from query points to fused view tokens, distance-biased.

    True cross-attention, not self-attention over a concatenated sequence. The
    distinction is not cosmetic: concatenating would make both the logits and the
    pairwise distance matrix ``(Q + N)^2``, and at the default chunk of 16384
    queries that is several gigabytes of activations for a decoder whose actual
    work is ``Q x N`` with ``N`` under a thousand. Keeping the two sides separate
    makes cost linear in the number of queries, which is what lets a full 128^3
    grid be evaluated on a 4 GB GPU.

    Queries never attend to each other, so a chunked evaluation is exactly equal
    to a single pass over all queries.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        num_heads: int = MODEL.num_heads,
        bias_scale: float = MODEL.distance_bias_scale,
        learn_bias: bool = MODEL.learn_distance_bias,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(
                f"embed_dim {embed_dim} must divide evenly among {num_heads} heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)

        self.to_query = nn.Linear(embed_dim, embed_dim)
        self.to_key_value = nn.Linear(embed_dim, 2 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        log_scale = torch.full((num_heads,), math.log(max(bias_scale, 1e-6)))
        self.log_distance_scale = nn.Parameter(log_scale, requires_grad=learn_bias)

    @property
    def distance_scale(self) -> torch.Tensor:
        return self.log_distance_scale.exp()

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_points: torch.Tensor,
        context_tokens: torch.Tensor,
        context_anchors: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_tokens: ``(B, Q, C)``
            query_points: ``(B, Q, 3)`` world coordinates
            context_tokens: ``(B, N, C)`` fused tokens
            context_anchors: ``(B, N, 3)``
            context_valid: ``(B, N)``

        Returns:
            ``(B, Q, C)``.
        """
        batch, n_query, _ = query_tokens.shape
        n_context = context_tokens.shape[1]

        queries = self.to_query(self.norm_query(query_tokens))
        queries = queries.reshape(batch, n_query, self.num_heads, self.head_dim)
        queries = queries.transpose(1, 2)                       # (B, H, Q, Dh)

        key_value = self.to_key_value(self.norm_context(context_tokens))
        key_value = key_value.reshape(batch, n_context, 2, self.num_heads, self.head_dim)
        key_value = key_value.permute(2, 0, 3, 1, 4)
        keys, values = key_value[0], key_value[1]               # (B, H, N, Dh)

        logits = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        squared = torch.cdist(query_points, context_anchors, p=2.0).pow(2)  # (B, Q, N)
        squared = squared.masked_fill(~context_valid.unsqueeze(1), 0.0)

        gamma = self.distance_scale.view(1, self.num_heads, 1, 1)
        logits = logits - gamma * squared.unsqueeze(1)

        logits = logits.masked_fill(
            ~context_valid.view(batch, 1, 1, n_context), float("-inf")
        )

        # A query with no valid context anywhere would produce an all -inf row
        # and a NaN softmax. Fall back to a uniform attention for those.
        empty = ~context_valid.any(dim=1)
        if empty.any():
            logits = torch.where(
                empty.view(batch, 1, 1, 1), torch.zeros_like(logits), logits
            )

        weights = logits.softmax(dim=-1)
        fused = (weights @ values).transpose(1, 2).reshape(batch, n_query, self.embed_dim)
        return query_tokens + self.out(fused)


class OccupancyDecoder(nn.Module):
    """Maps a world coordinate plus fused context to an occupancy logit.

    Args:
        embed_dim: token width of the fused context.
        hidden: width of the decoder MLP.
        depth: MLP layers.
        query_chunk: queries evaluated per forward chunk at inference.
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        hidden: int = MODEL.decoder_hidden,
        depth: int = MODEL.decoder_depth,
        num_heads: int = MODEL.num_heads,
        query_chunk: int = MODEL.query_chunk,
        n_bands: int = MODEL.fourier_bands,
        max_freq: float = MODEL.fourier_max_freq,
    ):
        super().__init__()
        self.query_chunk = query_chunk

        self.fourier = FourierFeatures(n_bands=n_bands, max_freq=max_freq)
        self.lift = nn.Linear(self.fourier.out_dim, embed_dim)
        self.attend = QueryAttention(embed_dim=embed_dim, num_heads=num_heads)

        layers: list[nn.Module] = [nn.Linear(embed_dim, hidden), nn.GELU()]
        for _ in range(max(0, depth - 2)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        final = nn.Linear(hidden, 1)
        layers += [final]
        self.mlp = nn.Sequential(*layers)

        # PyTorch's default Linear initialisation is not gain-corrected for
        # GELU, so a four-layer trunk attenuates its input by roughly two orders
        # of magnitude. Left alone, the decoder starts as a near-constant
        # function of its context and has to spend most of a short training
        # budget just recovering signal amplitude. Kaiming initialisation on the
        # hidden layers keeps the variance stable through the depth.
        for module in self.mlp:
            if isinstance(module, nn.Linear) and module is not final:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

        # Start the field almost empty. Under one percent of the working volume
        # is ever occupied, so a zero-initialised bias -- occupancy 0.5
        # everywhere -- would put the volume integral three orders of magnitude
        # too high and hand the biomass loss an enormous initial gradient.
        nn.init.xavier_uniform_(final.weight, gain=0.5)
        nn.init.constant_(final.bias, PRIOR_OCCUPANCY_LOGIT)

    def _forward_chunk(
        self,
        points: torch.Tensor,
        context_tokens: torch.Tensor,
        context_anchors: torch.Tensor,
        context_valid: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = self.lift(self.fourier(normalise_world(points)))
        fused = self.attend(
            query_tokens, points, context_tokens, context_anchors, context_valid
        )
        return self.mlp(fused).squeeze(-1)

    def forward(
        self,
        points: torch.Tensor,
        context_tokens: torch.Tensor,
        context_anchors: torch.Tensor,
        context_valid: torch.Tensor,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Occupancy logits for ``(B, Q, 3)`` world-frame query points.

        Returns:
            ``(B, Q)`` logits. Apply a sigmoid for probabilities.
        """
        chunk = self.query_chunk if chunk is None else chunk
        n_query = points.shape[1]

        if n_query <= chunk:
            return self._forward_chunk(
                points, context_tokens, context_anchors, context_valid
            )

        outputs = []
        for start in range(0, n_query, chunk):
            outputs.append(
                self._forward_chunk(
                    points[:, start : start + chunk],
                    context_tokens,
                    context_anchors,
                    context_valid,
                )
            )
        return torch.cat(outputs, dim=1)


__all__ = ["PRIOR_OCCUPANCY_LOGIT", "OccupancyDecoder", "QueryAttention"]
