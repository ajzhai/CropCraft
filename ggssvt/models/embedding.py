"""Geometry-grounded token embeddings.

A plain vision transformer positions its tokens by where they sit in the image
grid. That is the wrong frame for multi-view reconstruction: the same physical
leaf occupies a different patch index in every view, so the network has to learn
the correspondence from scratch and generalises poorly when the camera moves.

GG-SSVT positions tokens by where they sit *in the world*. Each patch is
back-projected through its median depth into a 3D point, that point is encoded
with a Fourier feature basis, and the result is added to the patch's appearance
embedding. Two tokens describing the same physical structure from opposite sides
of the plant then carry nearly the same positional code, and cross-view
attention (:mod:`ggssvt.models.attention`) can use that directly.

The Fourier basis follows the standard axis-aligned construction used by NeRF
and by implicit-surface decoders: ``[sin(2^k pi x), cos(2^k pi x)]`` over a
geometric ladder of frequencies. Low bands carry the coarse position that
matters for gross structure, high bands the fine detail needed to separate
neighbouring branches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MODEL, VOLUME_EXTENT_M


class FourierFeatures(nn.Module):
    """Axis-aligned Fourier encoding of 3D coordinates.

    Args:
        n_bands: number of frequency bands per axis.
        max_freq: highest frequency in the geometric ladder.
        include_input: append the raw coordinates to the encoding.

    Shape:
        - Input: ``(..., 3)`` coordinates, expected in roughly [-1, 1].
        - Output: ``(..., out_dim)``.
    """

    def __init__(
        self,
        n_bands: int = MODEL.fourier_bands,
        max_freq: float = MODEL.fourier_max_freq,
        include_input: bool = True,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.include_input = include_input

        # Geometric ladder from 1 to max_freq, as in NeRF's positional encoding.
        freqs = 2.0 ** torch.linspace(0.0, float(max_freq), n_bands)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 3 * (2 * self.n_bands + (1 if self.include_input else 0))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 3:
            raise ValueError(f"expected trailing dimension 3, got {coords.shape[-1]}")

        scaled = coords.unsqueeze(-1) * self.freqs * torch.pi   # (..., 3, B)
        encoded = torch.cat(
            [scaled.sin(), scaled.cos()], dim=-1
        ).flatten(start_dim=-2)                                  # (..., 3 * 2B)

        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded


def normalise_world(points: torch.Tensor, extent_m: float = VOLUME_EXTENT_M) -> torch.Tensor:
    """Map world coordinates into [-1, 1] over the working volume.

    ``x`` and ``y`` are centred on the plant axis; ``z`` runs from the floor up,
    so it is shifted before scaling. Keeping the encoder's input in a fixed range
    is what lets one set of Fourier frequencies serve every specimen.
    """
    half = extent_m / 2.0
    scaled = points.clone()
    scaled[..., 0] = scaled[..., 0] / half
    scaled[..., 1] = scaled[..., 1] / half
    scaled[..., 2] = (scaled[..., 2] - half) / half
    return scaled


def patch_centroids(
    points_world: torch.Tensor,
    valid: torch.Tensor,
    grid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a per-pixel world-point map to one 3D anchor per token.

    The anchor is the mean of the token's valid back-projected points. Tokens
    with no valid depth -- background, or a hole in the Kinect return -- get a
    zero anchor and are reported as invalid, so the caller can fall back to a
    learned token rather than pretending the token sits at the world origin.

    Takes a target *grid size* rather than a patch size so the anchors align with
    whatever token grid the appearance backbone produces. DINOv2 uses 14-pixel
    patches and DINOv3 uses 16, and each resizes its input; deriving the grid
    from a fixed patch size would silently misalign anchors with tokens for at
    least one of them.

    Args:
        points_world: ``(B, V, 3, H, W)`` world coordinates per pixel.
        valid: ``(B, V, 1, H, W)`` mask of pixels with usable depth.
        grid_size: ``(grid_h, grid_w)`` token grid to pool onto.

    Returns:
        ``(centroids, patch_valid)`` of shape ``(B, V, P, 3)`` and ``(B, V, P)``
        where ``P = grid_h * grid_w``.
    """
    batch, views, _, height, width = points_world.shape
    flat_points = points_world.reshape(batch * views, 3, height, width)
    flat_valid = valid.reshape(batch * views, 1, height, width).to(flat_points.dtype)

    # Adaptive pooling returns means over each cell; the cell sizes cancel in the
    # ratio, so the result is the mean over valid pixels regardless of whether
    # the grid divides the image evenly.
    summed = F.adaptive_avg_pool2d(flat_points * flat_valid, grid_size)
    counts = F.adaptive_avg_pool2d(flat_valid, grid_size)

    centroids = summed / counts.clamp(min=1e-6)
    patch_valid = counts.squeeze(1) > 0

    n_patches = grid_size[0] * grid_size[1]
    centroids = centroids.reshape(batch, views, 3, n_patches).permute(0, 1, 3, 2)
    patch_valid = patch_valid.reshape(batch, views, n_patches)

    return centroids.contiguous(), patch_valid


class GeometryGroundedEmbedding(nn.Module):
    """Adds a Fourier back-projected positional code to appearance tokens.

    Args:
        embed_dim: token width.
        n_bands, max_freq: Fourier basis settings.

    Shape:
        - ``tokens``: ``(B, V, P, C)``
        - ``centroids``: ``(B, V, P, 3)`` world coordinates
        - ``patch_valid``: ``(B, V, P)``
        - Output: ``(B, V, P, C)``
    """

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        n_bands: int = MODEL.fourier_bands,
        max_freq: float = MODEL.fourier_max_freq,
    ):
        super().__init__()
        self.fourier = FourierFeatures(n_bands=n_bands, max_freq=max_freq)
        self.project = nn.Sequential(
            nn.Linear(self.fourier.out_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Used where a patch has no depth at all, so the model is not forced to
        # treat "unknown position" as "position (0, 0, 0)".
        self.no_geometry = nn.Parameter(torch.zeros(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        centroids: torch.Tensor,
        patch_valid: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.fourier(normalise_world(centroids))
        positional = self.project(encoded)

        positional = torch.where(
            patch_valid.unsqueeze(-1),
            positional,
            self.no_geometry.expand_as(positional),
        )
        return self.norm(tokens + positional)


__all__ = [
    "FourierFeatures",
    "GeometryGroundedEmbedding",
    "normalise_world",
    "patch_centroids",
]
