"""What DINOv2 and DINOv3 actually see, side by side.

Both backbones tie on every number this project measures: the frozen-feature
probe puts them within 0.030 kg of each other, and the organ clustering ties
wherever the rim detector is confident. A table of near-identical numbers is a
poor way to understand *why*, and it cannot show the one place they differ,
which is the eleven captures where the geometric segmenter fails.

So this renders the features themselves. Patch tokens are projected onto their
first three principal components and read as RGB, which is the standard way to
look at a self-supervised feature map: structure the model considers similar
takes a similar colour, and the projection is fitted per image so the colours
carry no meaning across panels beyond that.

**The PCA basis is shared between the two backbones for a given view.** Fitting
each separately would give two arbitrary rotations of two different spaces and
any colour difference between the panels would be meaningless. Fitting one basis
on the concatenated features does not make the spaces comparable either, since
they have different dimensions, so each is fitted on its own features and the
sign of each component is then aligned to the other. That does not make the hues
equivalent; it stops them being gratuitously different. Read the panels for the
*boundaries* they draw, not for the colours they choose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import WORK_DIR, voxel_grid_centres

# The stages worth showing. Feature map, then what the lift and clustering make
# of it, because the clustering is where the two backbones diverge.
PANEL_PX = 320
BACKBONES = ("dinov2", "dinov3")

# Which sides to render. Four of the twelve, ninety degrees apart, because the
# question a reader has here is whether the features hold up as the plant turns,
# and four answers it without quadrupling a page that already carries a
# filmstrip. The clustering is a property of the specimen rather than of the
# view, so it is computed once however many sides are drawn.
VIEW_STRIDE = 3


def _components(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centred patch features and their principal directions."""
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    return flat, vt


def foreground(features: np.ndarray, *, keep: float = 0.35) -> np.ndarray:
    """Patches the first principal component separates out, as a flat mask.

    The first component of a self-supervised feature map is reliably an
    object/background split, and this is the standard way to read it. Which side
    of it is the object is not given, and the obvious rule of "take the minority"
    is not a rule at all: a fixed quantile selects the same fraction whichever
    sign it is applied with.

    An earlier version tested ``(first > median(first)).mean() > 0.5``, which is
    0.5 by the definition of a median, so the orientation was a coin flip and the
    selected region landed on the floor about half the time. It scored an
    intersection over union of exactly 0.000 against the segmenter, which is the
    signature of picking the complement rather than of picking badly.

    The sign is chosen instead by which side is more *varied*. Both the dark
    surround and the lit floor are close to featureless, while a plant is not, so
    the subject is the half whose features spread further.
    """
    flat, vt = _components(features)
    first = flat @ vt[0]

    high = first >= np.quantile(first, 1.0 - keep)
    low = first <= np.quantile(first, keep)
    spread = lambda sel: float(flat[sel].var(axis=0).sum()) if sel.any() else -1.0
    return high if spread(high) >= spread(low) else low


def _pca_rgb(features: np.ndarray, *, reference: np.ndarray | None = None,
             keep: float = 0.35) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patch features to RGB by principal components, fitted on the subject.

    Fitted twice, and the second fit is the point. On these captures the first
    component of a single fit is spent almost entirely on the lit floor against
    the dark background, so the plant comes out as a smooth gradient with no
    structure in it at all. Restricting the fit to the patches that component
    selects puts the variance back on the subject, which is where the question
    is.

    Returns the image, the basis (so a second backbone can align component signs
    against the first) and the foreground mask.
    """
    grid_h, grid_w, dim = features.shape
    subject = foreground(features, keep=keep)

    flat = features.reshape(-1, dim).astype(np.float64)
    inside = flat[subject]
    inside = inside - inside.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(inside, full_matrices=False)
    basis = vt[:3]

    if reference is not None and reference.shape == basis.shape:
        # Align signs only. A component and its negation span the same subspace,
        # and an arbitrary flip would read as a different segmentation.
        for k in range(3):
            if float(np.dot(basis[k], reference[k])) < 0:
                basis[k] = -basis[k]

    projected = inside @ basis.T
    lo = np.percentile(projected, 2, axis=0)
    hi = np.percentile(projected, 98, axis=0)
    scaled = np.clip((projected - lo) / np.maximum(hi - lo, 1e-9), 0, 1)

    # Background stays a flat grey rather than taking a colour, so the eye reads
    # the coloured region as "what the model grouped as subject".
    image = np.full((grid_h * grid_w, 3), 232.0)
    image[subject] = scaled * 255.0
    return (image.reshape(grid_h, grid_w, 3).astype(np.uint8), basis,
            subject.reshape(grid_h, grid_w))


def mask_agreement(subject: np.ndarray, cached, view: int) -> dict:
    """Does the feature-derived foreground land on the segmenter's subject?

    This is the measurable behind "could a foundation model segment this". The
    segmenter's mask is not ground truth, so agreement is reported as overlap
    rather than as accuracy, and it is only interesting where the two disagree.
    """
    grid_h, grid_w = subject.shape
    mask = np.asarray(cached.mask)[view]
    h, w = mask.shape

    # Block-average the pixel mask down to the patch grid.
    ys = (np.arange(h) * grid_h // h)
    xs = (np.arange(w) * grid_w // w)
    coarse = np.zeros((grid_h, grid_w), dtype=np.float64)
    counts = np.zeros((grid_h, grid_w), dtype=np.float64)
    np.add.at(coarse, (ys[:, None], xs[None, :]), mask.astype(np.float64))
    np.add.at(counts, (ys[:, None], xs[None, :]), 1.0)
    coarse = coarse / np.maximum(counts, 1.0)
    segmenter = coarse > 0.5

    both = int((subject & segmenter).sum())
    union = int((subject | segmenter).sum())
    return {
        "iou_with_segmenter": round(both / union, 4) if union else 0.0,
        "recall_of_segmenter": round(both / max(int(segmenter.sum()), 1), 4),
        "foreground_patches": int(subject.sum()),
        "segmenter_patches": int(segmenter.sum()),
    }


def _upsample(image: np.ndarray, size: int = PANEL_PX) -> np.ndarray:
    """Nearest-neighbour to the panel size, so patches stay legible as patches."""
    h, w = image.shape[:2]
    factor = max(1, size // max(h, w))
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def _cluster_render(cached, backbone, *, size: int = PANEL_PX) -> tuple[np.ndarray, dict]:
    """The lifted two-way clustering, drawn on the points it labelled."""
    from ..geometry.dino_lift import cluster, lift, order_by_height
    from .dino_segment import _feature_maps
    from .filmstrip import _rotate

    points = voxel_grid_centres()[cached.occupancy]
    lifted = lift(points, _feature_maps(cached, backbone),
                  cached.rotation.astype(np.float64),
                  cached.centre.astype(np.float64), cached.depth_m,
                  crop_top=cached.crop_top)
    labels = order_by_height(cluster(lifted, k=2), lifted.heights)

    kept = labels >= 0
    xyz = lifted.points[kept] if hasattr(lifted, "points") else points[kept]
    tag = labels[kept]

    view = _rotate(xyz - xyz.mean(axis=0), 0.62, 0.18)
    half = max(float(np.abs(view[:, :2]).max()), 1e-6) * 1.06
    px = np.round(size / 2 + view[:, 0] / half * (size / 2)).astype(np.int64)
    py = np.round(size / 2 - view[:, 1] / half * (size / 2)).astype(np.int64)

    # Lower cluster and upper cluster, in the page's own two categorical colours.
    lower = np.array([72, 40, 120], dtype=np.float64)
    upper = np.array([110, 206, 88], dtype=np.float64)
    colour = np.where(tag[:, None] == 0, lower, upper)

    image = np.full((size * size, 3), 255.0)
    order = np.argsort(view[:, 2], kind="stable")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            xs, ys = px[order] + dx, py[order] + dy
            ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
            image[ys[ok] * size + xs[ok]] = colour[order][ok]

    stats = {
        "n_points": int(kept.sum()),
        "upper_fraction": round(float((tag == 1).mean()), 4),
        "upper_mean_height_m": round(float(xyz[tag == 1, 2].mean()), 3)
        if (tag == 1).any() else 0.0,
        "lower_mean_height_m": round(float(xyz[tag == 0, 2].mean()), 3)
        if (tag == 0).any() else 0.0,
    }
    return image.reshape(size, size, 3).astype(np.uint8), stats


def compare(
    plant_id: str,
    *,
    cache_dir: Path,
    out_dir: Path,
    views: tuple[int, ...] | None = None,
    variant: str = "base",
    backbones: dict | None = None,
) -> dict:
    """Render both backbones' features on several sides, and their clusterings.

    ``backbones`` lets a caller hand in already-constructed encoders. Building a
    DINOv3 base takes the better part of a minute, and rebuilding it once per
    specimen dominated the run: for 36 specimens that was over an hour of loading
    weights to do a few seconds of arithmetic each time.
    """
    import torch

    from ..data.preprocess import load_cached
    from ..models.backbones import build_backbone
    from .filmstrip import _rgb_tile, _save

    cached = load_cached(plant_id, cache_dir)
    here = out_dir / plant_id
    rel = lambda name: f"{plant_id}/{name}"

    n_views = int(np.asarray(cached.mask).shape[0])
    if views is None:
        views = tuple(range(0, n_views, VIEW_STRIDE))

    entry = {"plant_id": plant_id, "species": cached.species,
             "mass_kg": round(float(cached.target_kg), 3),
             "views": [], "backbones": {}}

    for view in views:
        name = f"frame_{view:02d}.jpg"
        _save(_upsample(_rgb_tile(cached, view), PANEL_PX), here / name)
        entry["views"].append({
            "view": int(view),
            "azimuth_deg": int(round(view * 360.0 / n_views)),
            "frame": rel(name),
        })

    if backbones is None:
        backbones = {k: build_backbone(k, variant=variant) for k in BACKBONES}

    rgb = torch.from_numpy(cached.rgb).float().permute(0, 3, 1, 2) / 255.0
    reference = None
    for kind, backbone in backbones.items():
        per_view = []
        for view in views:
            with torch.no_grad():
                tokens, grid_h, grid_w = backbone.patch_tokens(rgb[view:view + 1])
            features = tokens.reshape(grid_h, grid_w, -1).cpu().numpy()

            image, basis, subject = _pca_rgb(features, reference=reference)
            if reference is None:
                reference = basis
            name = f"{kind}_features_{view:02d}.jpg"
            _save(_upsample(image, PANEL_PX), here / name)
            per_view.append({
                "view": int(view),
                "features": rel(name),
                **mask_agreement(subject, cached, view),
            })

        # One clustering per specimen: it is lifted from every view at once, so
        # it does not change with the side being drawn.
        clustered, stats = _cluster_render(cached, backbone)
        _save(clustered, here / f"{kind}_cluster.jpg")

        entry["backbones"][kind] = {
            "cluster": rel(f"{kind}_cluster.jpg"),
            "patch_grid": [int(grid_h), int(grid_w)],
            "dim": int(features.shape[-1]),
            "per_view": per_view,
            **stats,
        }
    return entry


def run(
    plant_ids: tuple[str, ...] | None = None,
    *,
    cache_dir: Path | None = None,
    out_dir: Path | None = None,
    out: Path = WORK_DIR / "reports" / "backbone_viz.json",
    views: tuple[int, ...] | None = None,
    variant: str = "base",
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Render the comparison for every usable specimen.

    It used to default to four hand-picked specimens, chosen because each
    illustrated something: one where the rim detector refuses, one where it does
    not, the one DINOv3 rescues, and one where DINOv3 is the worse of the two.
    That is a fine set of examples and a poor default. A reader who wants to
    check a specimen that is not on the list cannot, and a chosen subset invites
    exactly the suspicion that the examples were picked to suit the argument.
    """
    from ..data.preprocess import usable_plant_ids
    from ..models.backbones import build_backbone

    cache_dir = cache_dir or WORK_DIR / "cache"
    out_dir = out_dir or WORK_DIR / "reports" / "backbone_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    plant_ids = tuple(plant_ids or sorted(usable_plant_ids(cache_dir)))
    if limit:
        plant_ids = plant_ids[:limit]

    # Once, not once per specimen.
    if verbose:
        print(f"  loading {', '.join(BACKBONES)} ({variant})", flush=True)
    backbones = {k: build_backbone(k, variant=variant) for k in BACKBONES}

    specimens = []
    for plant_id in plant_ids:
        entry = compare(plant_id, cache_dir=cache_dir, out_dir=out_dir,
                        views=views, variant=variant, backbones=backbones)
        specimens.append(entry)
        if verbose:
            bits = "  ".join(
                f"{k} upper {v['upper_fraction']:.2f} at {v['upper_mean_height_m']:.2f} m"
                for k, v in entry["backbones"].items())
            print(f"  {plant_id:6s} {len(entry['views'])} sides  {bits}", flush=True)

    report = {
        "note": "patch features projected onto their first three principal "
                "components and read as RGB, fitted on the subject rather than "
                "on the whole frame, with component signs aligned between the "
                "two backbones so a sign flip does not read as a different "
                "segmentation",
        "caveat": "the colours are not comparable between panels in any "
                  "stronger sense; read the boundaries the features draw, not "
                  "the hues they take. The clustering is lifted from all twelve "
                  "views at once, so it does not change with the side shown",
        "view_stride": VIEW_STRIDE,
        "n_specimens": len(specimens),
        "specimens": specimens,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if verbose:
        print(f"\n  wrote {out} and panels under {out_dir}")
    return report


__all__ = ["BACKBONES", "compare", "run"]
