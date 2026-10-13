"""DITR's mechanism, applied to data that can support it.

Knaebel et al. observe that 3D segmentation largely ignores 2D foundation
models even where calibrated images sit beside the point cloud, and propose
DITR: extract frozen DINOv2 patch features, project the 3D points into each
camera to look them up, pool across views, and inject the result into a 3D
segmentation backbone trained with a supervised loss.

The first half of that transfers here directly. This project has twelve
calibrated views, per-pixel depth and a DINOv2 loader already, so lifting patch
features onto the reconstructed points is a short step. The second half does not:
DITR is trained on ScanNet, S3DIS and nuScenes, which supply per-point semantic
labels, and this dataset supplies none. Running the released checkpoints instead
would ask an indoor-furniture label space about a Eucalyptus sapling in a nursery
bag.

So what is implemented here is the lifting, with the supervised head replaced by
clustering. That is a real question rather than a consolation: **can a foundation
model separate plant from pot where excess-green cannot?** It is worth asking
because the geometric segmenter demonstrably fails on one batch. E001 to E010
carve as single tapering cones, and the rim detector refuses on nine of the ten
because no step exists in their profile to find. If DINOv2 features separate
tissue from plastic and soil, that failure is addressable without labels.

SAMa (Fischer et al.) lifts 2D *predictions* rather than features, by projecting
each view into an intermediary point cloud through depth and doing
nearest-neighbour lookups, which makes the result multi-view consistent by
construction. The projection here is the same operation; only what is carried
differs, and the two are complementary rather than alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import KINECT_V2, Intrinsics


@dataclass(frozen=True)
class LiftedFeatures:
    """Per-point features pooled from every view that saw the point."""

    points: np.ndarray          # (N, 3) world coordinates
    features: np.ndarray        # (N, D) pooled patch features
    n_views: np.ndarray         # (N,) how many views contributed
    heights: np.ndarray         # (N,) convenience: world z

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def observed(self, minimum: int = 1) -> np.ndarray:
        """Points at least ``minimum`` cameras could see."""
        return self.n_views >= minimum


def project(
    points: np.ndarray,
    rotation: np.ndarray,
    centre: np.ndarray,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    crop_top: int = 0,
    height: int | None = None,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points into one camera's pixel grid.

    The mapping DITR calls 2D-to-3D, run in the direction the name does not
    suggest: points are pushed into the image so the feature at that pixel can
    be read back.

    Returns:
        ``(col, row, valid)``. Invalid entries are behind the camera or outside
        the frame, and their indices are not meaningful.
    """
    height = height or intrinsics.height
    width = width or intrinsics.width

    cam = (points - centre) @ rotation
    z = cam[:, 2]
    in_front = z > 1e-6

    u = np.full(z.shape, -1.0)
    v = np.full(z.shape, -1.0)
    u[in_front] = cam[in_front, 0] * intrinsics.fx / z[in_front] + intrinsics.cx
    v[in_front] = cam[in_front, 1] * intrinsics.fy / z[in_front] + intrinsics.cy - crop_top

    col = np.round(u).astype(np.int32)
    row = np.round(v).astype(np.int32)
    valid = in_front & (col >= 0) & (col < width) & (row >= 0) & (row < height)
    return col, row, valid


def lift(
    points: np.ndarray,
    feature_maps: np.ndarray,
    rotation: np.ndarray,
    centre: np.ndarray,
    depth_m: np.ndarray,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    crop_top: int = 0,
    occlusion_tol_m: float = 0.03,
    pool: str = "mean",
) -> LiftedFeatures:
    """Pool 2D patch features onto 3D points across every view.

    Args:
        points: ``(N, 3)`` world coordinates to carry features.
        feature_maps: ``(V, gh, gw, D)`` patch features per view.
        depth_m: ``(V, H, W)`` measured depth, used for the occlusion test.
        occlusion_tol_m: a point further behind the measured surface than this
            is hidden in that view and must not take its feature. Without this
            a leaf on the far side of the plant inherits the features of the
            near side, which is the failure that makes naive lifting useless on
            a self-occluding subject.
        pool: ``mean`` or ``max`` across contributing views.

    Returns:
        A :class:`LiftedFeatures`.
    """
    points = np.asarray(points, dtype=np.float64)
    n_views, grid_h, grid_w, dim = feature_maps.shape
    _, img_h, img_w = depth_m.shape

    total = np.zeros((points.shape[0], dim), dtype=np.float32)
    best = np.full((points.shape[0], dim), -np.inf, dtype=np.float32)
    counts = np.zeros(points.shape[0], dtype=np.int32)

    for view in range(n_views):
        col, row, valid = project(
            points, rotation[view], centre[view],
            intrinsics=intrinsics, crop_top=crop_top,
            height=img_h, width=img_w,
        )
        if not valid.any():
            continue

        # Occlusion: compare the point's depth in this camera against what the
        # sensor measured along the same ray.
        cam_z = ((points - centre[view]) @ rotation[view])[:, 2]
        measured = np.zeros(points.shape[0])
        measured[valid] = depth_m[view][row[valid], col[valid]]
        seen = valid & (measured > 0) & (cam_z <= measured + occlusion_tol_m)
        if not seen.any():
            continue

        # Pixel grid to patch grid.
        gy = np.clip((row[seen] * grid_h) // img_h, 0, grid_h - 1)
        gx = np.clip((col[seen] * grid_w) // img_w, 0, grid_w - 1)
        sampled = feature_maps[view][gy, gx]

        if pool == "max":
            best[seen] = np.maximum(best[seen], sampled)
        else:
            total[seen] += sampled
        counts[seen] += 1

    if pool == "max":
        features = np.where(counts[:, None] > 0, best, 0.0).astype(np.float32)
    else:
        features = np.divide(
            total, np.maximum(counts, 1)[:, None],
            out=np.zeros_like(total), where=counts[:, None] > 0,
        )

    return LiftedFeatures(
        points=points,
        features=features,
        n_views=counts,
        heights=points[:, 2].copy(),
    )


def cluster(
    lifted: LiftedFeatures,
    k: int = 2,
    *,
    seed: int = 0,
    iterations: int = 60,
    min_views: int = 2,
) -> np.ndarray:
    """k-means over the lifted features, standing in for DITR's supervised head.

    Returns ``(N,)`` labels, with -1 where too few views saw the point to pool a
    trustworthy feature.

    Plain Lloyd's algorithm on cosine-normalised features. The normalisation
    matters more than the algorithm: DINOv2 features carry a large shared
    component that dominates Euclidean distance, and clustering on raw vectors
    tends to split on illumination rather than on material.
    """
    labels = np.full(lifted.n_points, -1, dtype=np.int32)
    usable = lifted.observed(min_views)
    if usable.sum() < k:
        return labels

    x = lifted.features[usable].astype(np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norm, 1e-9)

    rng = np.random.default_rng(seed)
    centres = x[rng.choice(x.shape[0], k, replace=False)]
    assignment = np.zeros(x.shape[0], dtype=np.int32)

    for _ in range(iterations):
        distances = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = distances.argmin(axis=1).astype(np.int32)
        if np.array_equal(new, assignment):
            break
        assignment = new
        for j in range(k):
            member = assignment == j
            if member.any():
                centres[j] = x[member].mean(axis=0)

    labels[usable] = assignment
    return labels


def order_by_height(labels: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Relabel clusters so 0 is the lowest and k-1 the highest, by mean height.

    k-means labels are arbitrary, and every comparison downstream wants "the pot
    cluster" and "the canopy cluster" to keep the same index between specimens.
    """
    out = np.full_like(labels, -1)
    present = [j for j in np.unique(labels) if j >= 0]
    means = {j: heights[labels == j].mean() for j in present}
    for rank, j in enumerate(sorted(present, key=lambda c: means[c])):
        out[labels == j] = rank
    return out


__all__ = ["LiftedFeatures", "cluster", "lift", "order_by_height", "project"]
