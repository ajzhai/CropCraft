"""The 4TU greenhouse lettuce set, as an external validation set.

`3rd Autonomous Greenhouse Challenge: Online Challenge Lettuce Images`,
DOI 10.4121/15023088. 388 plants, four cultivars, weekly destructive harvests,
one RealSense D415 looking straight down, with fresh shoot weight, dry weight,
height, diameter and leaf area measured on every plant.

**Why this set and not another capture.** Our own 36 specimens carry a batch
confound that no modelling change removes: predicting mass from a specimen's
capture batch alone beats every method we have (FINDINGS 7l). A seven-week
growth series in one facility has a continuous mass range by construction --
1.4 g to 459.7 g here, which is what V001-V008 was trying to be, at 388 plants
instead of eight. It is the external validation set, and our specimens become the
method-development set.

**What transfers and what does not.** This is a single top-down view, so the
twelve-view carve and the depth fusion cannot run on it at all. What can run is
the half of the pipeline that turned out to be the strongest: segmentation, and
the projected-area-plus-depth-profile features that `direct 2D` and
`2D + profile` are built from. Those are also the two methods that currently win
on our own data, so the thing being tested externally is the thing being claimed.

Two domain gaps are real and are not hidden: their RealSense D415 against our
Kinect v2, and lettuce against Eucalyptus and Mango. Transfer across both is a
stronger claim than a fit within either.

**The measurement is validated before it is used.** Their Height, Diameter and
LeafArea were measured destructively on the same plants, so the depth-derived
versions can be checked against them per plant. A segmentation that fails shows
up as a diameter that disagrees, and it is screened out on that basis rather than
being carried silently into the regression: the same order the rest of the
project uses: screen first, regress second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset_biomass"

# Excess green is kept for reference and reporting, but it is NOT what segments
# these images. Two of the four cultivars are red-leaf, and a red lettuce has a
# *negative* excess green, on Satine it measured -0.02, indistinguishable from
# the concrete floor. Segmenting on it silently lost an entire cultivar.
EXG_THRESHOLD = 0.06

# Saturation above this counts as plant rather than tray or floor. The white
# tray sits at 0.07 and the concrete at 0.08; both lettuce colours sit above 0.6.
SATURATION_THRESHOLD = 0.25

# How far above the tray surface a pixel must sit. Below this is tray lid and
# sensor noise.
MIN_HEIGHT_M = 0.012

# The tray itself: bright and unsaturated. Its modal depth is the reference
# surface every height is measured from.
TRAY_SATURATION = 0.18
TRAY_VALUE = 120.0

# Removes speckle without eating a seedling, which is about 80 px across.
OPEN_SIZE = 15

# The plant sits at the centre of its tray, which sits at the centre of frame.
# Everything outside this box is greenhouse: hoses, staging, other benches.
ROI = (0.28, 0.72, 0.08, 0.92)      # x_lo, x_hi, y_lo, y_hi as fractions

# Where the tray is looked for first. Tighter than ROI on purpose: see tray_depth.
TRAY_ROI = (0.40, 0.62, 0.28, 0.70)


@dataclass(frozen=True)
class Intrinsics:
    """The camera, from the dataset's own ReadMe and ground-truth JSON."""

    fx: float = 1371.58264160156
    fy: float = 1369.42761230469
    ppx: float = 973.902038574219
    ppy: float = 537.702270507812
    depth_scale: float = 0.00100000004749745     # raw uint16 -> metres

    def pixel_area_m2(self, depth_m: np.ndarray) -> np.ndarray:
        """Ground area one pixel covers at each depth."""
        return depth_m ** 2 / (self.fx * self.fy)


CAMERA = Intrinsics()


@dataclass(frozen=True)
class LettuceRecord:
    """One harvested plant and the pair of images taken of it."""

    image_id: str
    variety: str
    rgb: Path
    depth: Path
    fresh_weight_g: float
    dry_weight_g: float
    height_cm: float
    diameter_cm: float
    leaf_area_cm2: float

    @property
    def fresh_weight_kg(self) -> float:
        return self.fresh_weight_g / 1000.0


def load_ground_truth(root: Path = DATASET_DIR) -> list[LettuceRecord]:
    """Read every measurement, in the numeric order of the image names."""
    path = root / "GroundTruth" / "GroundTruth_All_388_Images.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no ground truth at {path}. Download DOI 10.4121/15023088 and "
            f"unpack it into {root}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))["Measurements"]

    def order(item: tuple[str, dict]) -> int:
        digits = "".join(c for c in item[0] if c.isdigit())
        return int(digits) if digits else 0

    records = []
    for image_id, row in sorted(payload.items(), key=order):
        records.append(LettuceRecord(
            image_id=image_id,
            variety=str(row["Variety"]).strip(),
            rgb=root / "RGBImages" / row["RGB_Image"],
            depth=root / "DepthImages" / row["Depth_Information"],
            fresh_weight_g=float(row["FreshWeightShoot"]),
            dry_weight_g=float(row["DryWeightShoot"]),
            height_cm=float(row["Height"]),
            diameter_cm=float(row["Diameter"]),
            leaf_area_cm2=float(row["LeafArea"]),
        ))
    return records


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """2G - R - B on channels scaled to [0, 1]."""
    channels = rgb.astype(np.float64) / 255.0
    return 2.0 * channels[..., 1] - channels[..., 0] - channels[..., 2]


def saturation(rgb: np.ndarray) -> np.ndarray:
    """HSV saturation, which both lettuce colours share and no surface does."""
    channels = rgb.astype(np.float64)
    high = channels.max(axis=2)
    low = channels.min(axis=2)
    return (high - low) / np.maximum(high, 1.0)


def _roi_mask(
    shape: tuple[int, int], box: tuple[float, float, float, float] | None = None
) -> np.ndarray:
    height, width = shape
    x_lo, x_hi, y_lo, y_hi = box or ROI
    mask = np.zeros(shape, dtype=bool)
    mask[int(y_lo * height):int(y_hi * height),
         int(x_lo * width):int(x_hi * width)] = True
    return mask


def tray_depth(rgb: np.ndarray, depth_m: np.ndarray) -> float:
    """Depth of the tray surface the plant stands in, or 0 if it is not found.

    The modal depth of the bright unsaturated pixels around the plant. Two
    surfaces answer that description: the tray and the concrete floor, and
    they are only ~12 cm apart, so which one wins is decided by which fills more
    of the search box. The floor is greyer but not by much: measured on these
    images the tray reads value 172 saturation 0.07 and the concrete 124 and
    0.08, which no threshold separates reliably.

    So the search starts in a box tight around frame centre, where the tray is
    essentially the only thing that is not plant, and widens only if that box
    holds too little to take a mode from. Searching the wide box first returns
    the floor whenever the tray is small in frame, and every height then comes
    out ~12 cm too large, larger than most of the plants.
    """
    value = rgb.astype(np.float64).max(axis=2)
    unsaturated = (
        (depth_m > 0) & (saturation(rgb) < TRAY_SATURATION) & (value > TRAY_VALUE)
    )
    for box in (TRAY_ROI, ROI):
        candidate = unsaturated & _roi_mask(rgb.shape[:2], box)
        if candidate.sum() < 2000:
            continue
        histogram, edges = np.histogram(depth_m[candidate], bins=200)
        peak = int(np.argmax(histogram))
        return float(0.5 * (edges[peak] + edges[peak + 1]))
    return 0.0


def segment(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    saturation_threshold: float = SATURATION_THRESHOLD,
    min_height_m: float = MIN_HEIGHT_M,
    open_size: int = OPEN_SIZE,
) -> tuple[np.ndarray, float]:
    """The plant, as a boolean mask, with the tray depth it was cut against.

    Colour alone cannot do this. A red-leaf lettuce and the orange crate the tray
    stands on overlap on every simple index, excess green, saturation, and
    green-minus-blue all put them within noise of each other. What separates them
    is that the tray sits *on top of* the crate, so anything raised above the
    tray surface is plant and the crate is below it. Saturation then removes the
    tray's own lid and rim, which are at tray height but unsaturated.

    Returns the mask and the tray depth, because every later measurement is
    relative to that surface and recomputing it would risk the two disagreeing.
    """
    from scipy import ndimage

    tray = tray_depth(rgb, depth_m)
    if tray <= 0:
        return np.zeros(rgb.shape[:2], dtype=bool), 0.0

    plant = (
        _roi_mask(rgb.shape[:2]) & (depth_m > 0)
        & (saturation(rgb) > saturation_threshold)
        & ((tray - depth_m) > min_height_m)
    )
    if not plant.any():
        return plant, tray

    plant = ndimage.binary_opening(plant, structure=np.ones((open_size, open_size)))
    if not plant.any():
        return plant, tray

    labels, count = ndimage.label(plant)
    if count > 1:
        sizes = ndimage.sum(plant, labels, range(1, count + 1))
        plant = labels == (int(np.argmax(sizes)) + 1)
    return plant, tray


def point_cloud(
    mask: np.ndarray,
    depth_m: np.ndarray,
    tray: float,
    *,
    intrinsics: Intrinsics = CAMERA,
) -> np.ndarray:
    """The plant's visible surface as metric points, x-y-z in metres.

    z is height above the tray, so the cloud sits on the plane the plant grows
    from rather than being referenced to the camera. This is a 2.5D surface and
    not a reconstruction: one view sees the top of the canopy and nothing under
    it. What it does carry, which a projected area does not, is how that surface
    is shaped.
    """
    rows, cols = np.nonzero(mask)
    z_camera = depth_m[rows, cols]
    x = (cols - intrinsics.ppx) * z_camera / intrinsics.fx
    y = (rows - intrinsics.ppy) * z_camera / intrinsics.fy
    return np.column_stack([x, y, np.clip(tray - z_camera, 0.0, None)])


def surface_descriptors(
    mask: np.ndarray,
    depth_m: np.ndarray,
    tray: float,
    *,
    intrinsics: Intrinsics = CAMERA,
    projected_area_m2: float,
    max_hull_points: int = 6000,
) -> dict[str, float]:
    """Shape descriptors of the visible surface, beyond its outline.

    ``rugosity`` is the ratio of true surface area to the area of its shadow. A
    flat leaf scores 1; a crumpled Salanova head scores well above it. It is the
    quantity a projected area is blind to, and fresh mass is partly a question of
    how much tissue is folded into a given footprint.

    ``normal_z_mean`` is the mean vertical component of the surface normals,
    which falls as leaves stand up: a proxy for leaf angle.

    ``hull_volume_l`` is the convex hull of the surface points together with
    their shadow on the tray, so a canopy that domes upward encloses more than a
    flat rosette of the same outline. It is an upper bound on the plant, not a
    measurement of it, with one view the underside is assumption, not data.
    """
    from scipy import ndimage

    # Surface area from the depth gradient: each pixel's ground footprint,
    # stretched by the tilt of the surface over it. dz/dx and dz/dy are metres
    # per metre, so the stretch factor is the usual sqrt(1 + |grad z|^2).
    height = np.where(mask, np.clip(tray - depth_m, 0.0, None), 0.0)
    metre_per_px_x = depth_m / intrinsics.fx
    metre_per_px_y = depth_m / intrinsics.fy
    grad_row, grad_col = np.gradient(height)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_x = np.where(metre_per_px_x > 0, grad_col / metre_per_px_x, 0.0)
        slope_y = np.where(metre_per_px_y > 0, grad_row / metre_per_px_y, 0.0)

    # Only interior pixels: the gradient across the silhouette edge is the drop
    # to the tray, which is a boundary, not a surface the plant has.
    interior = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    if not interior.any():
        interior = mask

    stretch = np.sqrt(1.0 + slope_x[interior] ** 2 + slope_y[interior] ** 2)
    footprint = intrinsics.pixel_area_m2(depth_m[interior])
    surface_m2 = float((footprint * stretch).sum())
    shadow_m2 = float(footprint.sum())

    normal_z = float(np.mean(1.0 / stretch)) if stretch.size else 0.0

    points = point_cloud(mask, depth_m, tray, intrinsics=intrinsics)
    heights = points[:, 2]

    hull_volume_m3 = 0.0
    if points.shape[0] >= 8:
        sample = points
        if sample.shape[0] > max_hull_points:
            step = sample.shape[0] // max_hull_points
            sample = sample[::step]
        # The surface and its shadow, so the hull is a solid rather than a sheet.
        floor = sample.copy()
        floor[:, 2] = 0.0
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull_volume_m3 = float(ConvexHull(np.vstack([sample, floor])).volume)
        except QhullError:           # degenerate: coplanar, or too few points
            hull_volume_m3 = 0.0

    return {
        "surface_area_cm2": surface_m2 * 1e4,
        "rugosity": surface_m2 / max(shadow_m2, 1e-12),
        "normal_z_mean": normal_z,
        "hull_volume_l": hull_volume_m3 * 1e3,
        # How the height is distributed, not just how tall the plant is.
        "height_p50_cm": float(np.percentile(heights, 50)) * 100.0,
        "height_p90_cm": float(np.percentile(heights, 90)) * 100.0,
        "height_sd_cm": float(heights.std()) * 100.0,
        # Fill of the smallest upright cylinder the canopy sweeps: a dome fills
        # more of it than a rosette with the same outline and peak.
        "fill_ratio": float(
            (heights.mean() * projected_area_m2)
            / max(heights.max() * projected_area_m2, 1e-12)
        ),
    }


def measure(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    *,
    intrinsics: Intrinsics = CAMERA,
) -> dict[str, float]:
    """Descriptors for one plant, in the units the ground truth uses.

    The reference surface is the tray the plant stands in, taken as the median
    depth of the non-plant pixels inside the region of interest. That is this
    dataset's equivalent of our measured pot rim: heights and the volume are
    integrated above it, not above the camera.
    """
    depth_m = depth_raw.astype(np.float64) * intrinsics.depth_scale
    mask, tray = segment(rgb, depth_m)

    n_pixels = int(mask.sum())
    if n_pixels < 50:
        empty = dict.fromkeys(FEATURE_NAMES, 0.0)
        empty.update({"n_pixels": float(n_pixels), "valid": 0.0,
                      "tray_depth_m": 0.0})
        return empty

    plant_depth = depth_m[mask]
    pixel_area = intrinsics.pixel_area_m2(plant_depth)
    area_m2 = float(pixel_area.sum())

    # Height above the tray. Clipped at zero because a depth outlier below the
    # tray plane is a sensor artefact, not a plant hanging through the bench.
    height_above = np.clip(tray - plant_depth, 0.0, None)
    volume_m3 = float((height_above * pixel_area).sum())

    rows, cols = np.nonzero(mask)
    span_row = (rows.max() - rows.min() + 1)
    span_col = (cols.max() - cols.min() + 1)
    median_depth = float(np.median(plant_depth))
    metres_per_px = median_depth / intrinsics.fx
    extent_m = float(max(span_row, span_col)) * metres_per_px
    minor_m = float(min(span_row, span_col)) * metres_per_px

    surface = surface_descriptors(
        mask, depth_m, tray, intrinsics=intrinsics, projected_area_m2=area_m2)

    return {
        **surface,
        "n_pixels": float(n_pixels),
        "valid": 1.0,
        "area_cm2": area_m2 * 1e4,
        "volume_l": volume_m3 * 1e3,
        # The 99th percentile rather than the maximum: one stray near-camera
        # pixel would otherwise set the height of the whole plant.
        "height_cm": float(np.percentile(height_above, 99)) * 100.0,
        "mean_height_cm": float(height_above.mean()) * 100.0,
        "diameter_cm": extent_m * 100.0,
        # Occupied share of the bounding box: a rosette fills more of it than a
        # sprawling head with gaps between the leaves.
        "compactness": n_pixels / float(span_row * span_col),
        "elongation": extent_m / max(minor_m, 1e-6),
        "exg_mean": float(excess_green(rgb)[mask].mean()),
        "tray_depth_m": tray,
    }


# What a silhouette and one height statistic give: the "2D + profile" set.
OUTLINE_NAMES = (
    "area_cm2", "volume_l", "height_cm", "mean_height_cm",
    "diameter_cm", "compactness", "elongation", "exg_mean",
)

# What the back-projected surface gives on top of that. Kept separate so the
# experiment can ask whether the surface adds anything, which is the question
# H2 asks and which our own specimens cannot answer through the batch confound.
SURFACE_NAMES = (
    "surface_area_cm2", "rugosity", "normal_z_mean", "hull_volume_l",
    "height_p50_cm", "height_p90_cm", "height_sd_cm", "fill_ratio",
)

FEATURE_NAMES = OUTLINE_NAMES + SURFACE_NAMES


def feature_vector(measured: dict[str, float]) -> np.ndarray:
    """The descriptors used for regression, in a fixed order."""
    return np.array([measured[name] for name in FEATURE_NAMES], dtype=np.float64)


__all__ = [
    "DATASET_DIR",
    "EXG_THRESHOLD",
    "FEATURE_NAMES",
    "MIN_HEIGHT_M",
    "OUTLINE_NAMES",
    "SATURATION_THRESHOLD",
    "SURFACE_NAMES",
    "Intrinsics",
    "LettuceRecord",
    "excess_green",
    "feature_vector",
    "load_ground_truth",
    "measure",
    "point_cloud",
    "saturation",
    "segment",
    "surface_descriptors",
    "tray_depth",
]
