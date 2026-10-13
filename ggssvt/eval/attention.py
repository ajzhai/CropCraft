"""What the fusion stack looks at, and whether the geometry bias earns its place.

The thesis is about understanding plant morphology, and until now nothing in the
project looked inside the model. The cross-view attention computes a full
``(heads, N, N)`` weight matrix over every token of every view and then throws it
away. This recovers it.

Two readings, and they answer different questions.

**Per-view attention.** Sum the weights arriving at each view's tokens. This says
which of the twelve azimuths the fusion actually draws on when it predicts
occupancy. A model that has learned the rig should attend to neighbouring views
far more than to the opposite side of the plant, because those are the ones whose
frusta overlap.

**Spatial attention.** Reshape one view's incoming weight back onto its token
grid and up to image resolution. This says *where* in a view the fusion looks,
which is the figure that belongs beside a reconstruction.

There is a third thing worth reading and it needs no capture at all:
``CrossViewFusion.distance_scales`` returns the learned inverse length scales per
block and head. If the geometry bias is doing nothing, those collapse toward zero
and the attention is ordinary self-attention wearing a hat. That is a falsifiable
check on H2, and it costs one tensor read.

Nothing here is captured by default. `capture_attention` is a context manager
that switches it on for one forward pass and puts it back, because the weights
are large and retaining them during training would be a real memory cost.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass
class AttentionReading:
    """Attention from one forward pass, already reduced to something readable."""

    per_view: np.ndarray          # (blocks, views) incoming weight per view
    spatial: np.ndarray           # (blocks, views, gh, gw) per-token weight
    distance_scales: np.ndarray   # (blocks, heads) learned inverse length scale
    n_views: int
    grid: tuple[int, int]

    def as_dict(self) -> dict:
        return {
            "n_views": self.n_views,
            "grid": list(self.grid),
            "per_view": [[round(float(v), 5) for v in row] for row in self.per_view],
            "distance_scales": [
                [round(float(v), 5) for v in row] for row in self.distance_scales
            ],
        }

    def neighbour_preference(self) -> float:
        """How much more a view is attended to by its neighbours than by the rest.

        One number for the H2 question. Views sit at 30 degree steps around the
        subject, so index distance is angular distance. Returns the ratio of mean
        weight at index distance 1 to mean weight at distance > 2; above 1 means
        the fusion prefers views whose frusta overlap, which is what a model that
        has learned the rig geometry should do.
        """
        weight = self.per_view.mean(axis=0)
        n = len(weight)
        if n < 5:
            return float("nan")
        # per_view is already a marginal, so rebuild the relation by circular
        # distance from the strongest view, which is the one the pass centred on.
        centre = int(np.argmax(weight))
        distance = np.minimum(
            np.abs(np.arange(n) - centre), n - np.abs(np.arange(n) - centre)
        )
        near = weight[distance == 1]
        far = weight[distance > 2]
        if not near.size or not far.size or far.mean() <= 0:
            return float("nan")
        return float(near.mean() / far.mean())


@contextmanager
def capture_attention(model):
    """Collect attention weights for the forward passes inside this block.

    Yields a dict mapping block index to the list of ``(B, heads, N, N)`` tensors
    that block produced. The flag is cleared on the way out even if the body
    raises, so a failed pass cannot leave a model quietly accumulating weights.
    """
    from ..models.attention import CrossViewGeometricAttention

    blocks = [m for m in model.modules()
              if isinstance(m, CrossViewGeometricAttention)]
    captured: dict[int, list] = {i: [] for i in range(len(blocks))}
    for index, block in enumerate(blocks):
        block._capture = captured[index]
    try:
        yield captured
    finally:
        for block in blocks:
            block._capture = None


def read(model, batch, *, grid: tuple[int, int] | None = None) -> AttentionReading:
    """Run one forward pass and reduce the attention to per-view and spatial maps.

    Args:
        model: a :class:`~ggssvt.models.ggssvt.GGSSVT`.
        batch: one collated :class:`~ggssvt.training.dataset.SpecimenBatch`,
            batch size 1.
        grid: token grid per view as ``(rows, cols)``. Inferred as a square when
            omitted, which is right for the default tokeniser.
    """
    import torch

    model.eval()
    # Unpacked exactly as the trainer does. The forward takes positional
    # tensors, not the batch object, and passing the object silently type-errors
    # four frames down rather than at the call.
    with torch.no_grad(), capture_attention(model) as captured:
        model(
            batch.rgb,
            batch.depth,
            batch.points_world,
            batch.subject,
            batch.query_points,
            pot_height_m=batch.pot_height_m,
        )

    if not any(captured.values()):
        raise RuntimeError(
            "no attention was captured; the model ran no CrossViewGeometricAttention"
        )

    n_views = int(batch.subject.shape[1])
    per_view, spatial = [], []

    for index in sorted(captured):
        passes = captured[index]
        if not passes:
            continue
        # (B, heads, N, N) -> mean over batch and heads, then sum the weight
        # arriving at each key token. That marginal is what "attended to" means.
        weights = passes[0].mean(dim=(0, 1))          # (N, N)
        incoming = weights.sum(dim=0).cpu().numpy()   # (N,)

        n_tokens = incoming.shape[0]
        per_view_tokens = n_tokens // n_views
        if per_view_tokens < 1:
            continue
        trimmed = incoming[: n_views * per_view_tokens].reshape(n_views, -1)
        per_view.append(trimmed.sum(axis=1))

        rows, cols = grid or _square(per_view_tokens)
        if rows * cols == per_view_tokens:
            spatial.append(trimmed.reshape(n_views, rows, cols))

    scales = model.fusion.distance_scales().cpu().numpy() if hasattr(
        model, "fusion"
    ) else np.zeros((0, 0))

    per_view_arr = np.stack(per_view) if per_view else np.zeros((0, n_views))
    # Normalise per block so blocks are comparable; the absolute scale is a
    # function of token count, which is not interesting.
    totals = per_view_arr.sum(axis=1, keepdims=True)
    # out= matters: without it the skipped entries are uninitialised memory
    # rather than zero, which is a real trap in a figure nobody would check.
    per_view_arr = np.divide(
        per_view_arr, totals, out=np.zeros_like(per_view_arr), where=totals > 0
    )

    spatial_arr = (np.stack(spatial) if spatial
                   else np.zeros((0, n_views, 0, 0)))
    return AttentionReading(
        per_view=per_view_arr,
        spatial=spatial_arr,
        distance_scales=scales,
        n_views=n_views,
        grid=grid or (_square(per_view_tokens) if per_view else (0, 0)),
    )


def _square(n: int) -> tuple[int, int]:
    side = round(n ** 0.5)
    return (side, side) if side * side == n else (1, n)


def render_per_view(reading: AttentionReading, *, size: int = 260) -> np.ndarray:
    """A bar chart of attention per view, one row per fusion block.

    Drawn by hand for the same reason every other figure here is: matplotlib is
    not a dependency of this project and adding one for a bar chart would be a
    poor trade.
    """
    from .render import viridis

    blocks, views = reading.per_view.shape
    if blocks == 0:
        raise ValueError("nothing to draw; the reading has no blocks")

    pad, row_height = 26, 34
    height = pad + blocks * row_height
    canvas = np.full((height, size, 3), 255, dtype=np.uint8)

    usable = size - 2 * pad
    bar_width = max(1, usable // views)
    peak = float(reading.per_view.max()) or 1.0

    for block in range(blocks):
        base = pad + block * row_height + row_height - 6
        for view in range(views):
            share = float(reading.per_view[block, view]) / peak
            bar = round(share * (row_height - 12))
            colour = viridis(np.array([share]))[0]
            x0 = pad + view * bar_width
            canvas[base - bar : base, x0 : x0 + bar_width - 1] = colour
        canvas[base : base + 1, pad : pad + views * bar_width] = (150, 150, 150)
    return canvas


__all__ = [
    "AttentionReading",
    "capture_attention",
    "read",
    "render_per_view",
]
