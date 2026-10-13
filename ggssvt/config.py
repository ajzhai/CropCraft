"""Central configuration for GG-SSVT.

Every camera constant, rig assumption, voxel-grid setting and hyperparameter
lives here so a single edit propagates through the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
PLANTS_DIR = DATASET_DIR / "plants"
CALIB_DIR = DATASET_DIR / "calib"
GROUND_TRUTH_CSV = DATASET_DIR / "ground_truth.csv"
WORK_DIR = REPO_ROOT / "work_dirs" / "ggssvt"


# ---------------------------------------------------------------------------
# Kinect v2 depth/IR intrinsics at 512x424.
#
# Factory defaults. The registered RGB frames in dataset/plants/*/images share
# this intrinsic matrix because libfreenect2 maps colour into the depth frame,
# which is why RGB and depth are pixel-aligned and both 512x424.
#
# Override at runtime from device.getIrCameraParams() if the units differ.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Intrinsics:
    fx: float = 365.456
    fy: float = 365.456
    cx: float = 254.878
    cy: float = 205.395
    width: int = 512
    height: int = 424

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


KINECT_V2 = Intrinsics()

DEPTH_SCALE_M = 1e-3        # stored uint16 is millimetres
DEPTH_MIN_M = 0.5           # below this the v2 sensor is unreliable
DEPTH_MAX_M = 4.5           # beyond this it is room background


# ---------------------------------------------------------------------------
# Rig protocol.
#
# camA and camB are carried together through 6 manual steps of 30 degrees.
# camB sits 180 degrees opposite camA, so the 12 physical azimuths are
# 0, 30, ... 330. See ggssvt.data.naming for how filenames map onto these --
# two naming conventions exist in the collected data.
# ---------------------------------------------------------------------------
ANGULAR_STEP_DEG = 30
N_SWEEP_STEPS = 6
SWEEP_ANGLES_DEG = tuple(range(0, N_SWEEP_STEPS * ANGULAR_STEP_DEG, ANGULAR_STEP_DEG))
CAMB_OFFSET_DEG = 180
CAMERAS = ("camA", "camB")

# Nominal rig pose, used only to seed the estimator in ggssvt.geometry.rig.
# The real values are recovered per view from the floor plane and the plant
# centroid, because no ChArUco calibration was captured.
NOMINAL_CAM_RADIUS_M = 1.4   # horizontal distance from turntable axis to camera
NOMINAL_CAM_HEIGHT_M = 1.0   # camera optical centre above the floor
NOMINAL_CAM_PITCH_DEG = 0.0  # positive tilts the camera down


# Network input crop. The Kinect frame is 512x424; 424 is not divisible by the
# 16-pixel patch, so four rows are dropped top and bottom. Cropping rather than
# resizing keeps the intrinsics valid -- only the principal point's row shifts,
# which the cached world points already account for.
INPUT_CROP_TOP = 4
INPUT_HEIGHT = 416
INPUT_WIDTH = 512


# ---------------------------------------------------------------------------
# Working volume and voxel grid.
#
# A cube centred on the plant axis at floor level. 128^3 at 12 mm resolution
# spans 1.536 m, which comfortably contains the tallest specimen (E019/E020).
# ---------------------------------------------------------------------------
VOXEL_RESOLUTION = 128
VOXEL_SIZE_M = 0.012
VOLUME_EXTENT_M = VOXEL_RESOLUTION * VOXEL_SIZE_M

# Segmentation ROI, in the plant-centred world frame (metres, z up from floor).
ROI_RADIUS_M = 0.6
ROI_Z_MIN_M = 0.05
ROI_Z_MAX_M = 1.5

# Height above the floor below which points are treated as pot/soil rather
# than above-ground plant material. Applied only when computing the
# above-ground volume feature, never when carving.
POT_HEIGHT_M = 0.28


# ---------------------------------------------------------------------------
# Segmentation thresholds
# ---------------------------------------------------------------------------
FLOOR_RANSAC_ITERS = 400
FLOOR_INLIER_TOL_M = 0.02
FLOOR_MIN_INLIER_FRAC = 0.08
EXCESS_GREEN_THRESHOLD = 0.06   # 2G-R-B on [0,1] channels; >0 favours foliage
SOR_NEIGHBOURS = 16
SOR_STD_RATIO = 2.0


# ---------------------------------------------------------------------------
# Space carving
# ---------------------------------------------------------------------------
# Tuned on E002/E011/M001 against surface coverage (the fraction of measured
# subject points the hull retains) at the smallest hull that keeps it. Raising
# the vote budget past 3 inflates the hull sharply and breaks its connectivity.
CARVE_DEPTH_MARGIN_M = 0.04        # constant part of the free-space tolerance
CARVE_DEPTH_MARGIN_SLOPE = 0.010   # quadratic part; Kinect v2 noise grows as z^2
CARVE_MAX_VOTES = 3                # views allowed to rule a voxel out before it goes
CARVE_MIN_INFORMATIVE_VIEWS = 6    # views that must be able to see it at all
CARVE_MASK_DILATION = 2            # pixels the subject mask is widened by


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """GG-SSVT architecture settings."""

    # Token encoder
    patch_size: int = 16
    embed_dim: int = 384
    encoder_depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    backbone: str = "cnn"           # "cnn" (no DINO) | "dinov2" | "dinov3"
    backbone_variant: str = "small" # DINO size: small | base | large
    freeze_backbone: bool = True

    # Fourier back-projected positional encoding
    fourier_bands: int = 10
    fourier_max_freq: float = 8.0

    # Cross-view geometric attention
    fusion_depth: int = 4
    distance_bias_scale: float = 4.0   # initial value of the learned 1/sigma
    learn_distance_bias: bool = True

    # Implicit occupancy decoder
    decoder_hidden: int = 256
    decoder_depth: int = 4
    query_chunk: int = 16384
    n_train_queries: int = 8192

    # Biomass head
    density_prior_kg_m3: float = 240.0  # fresh canopy bulk density, initial value
    learn_density: bool = True
    head_hidden: int = 128

    # Memory
    use_checkpointing: bool = True
    amp: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings for both stages."""

    pretrain_epochs: int = 120
    finetune_epochs: int = 200
    batch_size: int = 1            # one specimen (12 views) per step
    lr_pretrain: float = 2e-4
    lr_finetune: float = 5e-5
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    occupancy_pos_weight: float = 4.0
    lambda_occupancy: float = 1.0
    lambda_biomass: float = 1.0
    lambda_volume_consistency: float = 0.1
    seed: int = 0
    device: str = "cuda"
    num_workers: int = 0


MODEL = ModelConfig()
TRAIN = TrainConfig()


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
TARGET_COLUMN = "net_weight_g"
SPECIES_COLUMN = "species_breed"
# Excluded from every experiment, with the reason. Both are capture failures
# rather than difficult specimens: a specimen the rig never finished is not
# evidence about the method, and keeping it would put a 9-view reconstruction
# in a 12-view comparison.
EXCLUDED_PLANTS: tuple[str, ...] = (
    "X001",   # 2 views only, single synthetic specimen
    "V011",   # capture stopped at 240 degrees; 9 views of 12, a contiguous
              # 120 degree arc never captured. Dropped rather than re-shot.
)


def voxel_grid_centres(
    resolution: int = VOXEL_RESOLUTION, voxel_size_m: float = VOXEL_SIZE_M
) -> np.ndarray:
    """Return the (R, R, R, 3) world-frame centre of every voxel.

    The grid is centred on the plant axis in x/y and starts at the floor in z.

    The arguments exist so a finer grid can cover the *same extent* as the
    default one. TSDF fusion runs at 6 mm and 256^3, which is 1.536 m either way,
    and keeping the extent identical is what makes a volume computed on one grid
    comparable with a volume computed on the other.
    """
    r = resolution
    extent = resolution * voxel_size_m
    half = extent / 2.0
    xs = np.linspace(-half + voxel_size_m / 2, half - voxel_size_m / 2, r)
    ys = np.linspace(-half + voxel_size_m / 2, half - voxel_size_m / 2, r)
    zs = np.linspace(voxel_size_m / 2, extent - voxel_size_m / 2, r)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1)
