"""Pose-free reconstruction: DUSt3R, MASt3R and Fast3R as an independent check.

Every camera pose in this project is *estimated* from depth, and the azimuth
refinement saturates its search bound on almost every specimen. That makes the
registration the least verified assumption in the pipeline, and nothing inside
the pipeline can test it -- the carve, the mesh and the biomass features all
consume those poses and would agree with each other regardless.

DUSt3R, MASt3R and Fast3R estimate cameras *and* geometry from images alone.
They share no failure mode with a depth-based registration, so comparing their
poses against ours is a genuine second opinion, and it is worth more than the
fourth biomass number that comes with it.

Two pieces of machinery make the comparison meaningful, and both are testable
without the models:

**Similarity alignment.** A pose-free method returns poses in its own arbitrary
frame and, except for MASt3R's metric variant, its own arbitrary scale. Comparing
them to ours requires solving for the rotation, translation and scale that best
align the two camera sets -- the Umeyama problem -- and then reporting what is
*left over*. The residual is the disagreement; everything removed by the
alignment is gauge freedom and means nothing.

**Scale recovery.** For the non-metric methods, scale is fixed by matching the
predicted depth against the Kinect's measured depth over the pixels where both
are valid. A ratio of medians is used rather than a mean because a handful of
background pixels at wildly wrong depth would otherwise set the scale for the
whole specimen.

The adapters below are written against each project's documented interface but
have **not been executed against the real weights** -- none of the three installs
without a GPU, and the machine this was written on has none. Treat them as a
scaffold that needs its first run watched, not as tested code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DUST3R_REPO = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
MAST3R_METRIC_REPO = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
FAST3R_REPO = "jedyang97/Fast3R_ViT_Large_512"

INSTALL_HELP = {
    "dust3r": (
        "DUSt3R needs its GitHub repository, not just the weights:\n"
        "  git clone --recursive https://github.com/naver/dust3r third_party/dust3r\n"
        "The --recursive matters: the CroCo backbone is a submodule.\n"
        "There is no setup.py, so `pip install -e .` fails. Put the repo root on\n"
        "sys.path instead -- a .pth file in site-packages holding its absolute\n"
        "path, or PYTHONPATH. dust3r adds croco to sys.path itself on import.\n"
        "Inference needs only: torch torchvision roma einops opencv-python scipy\n"
        "trimesh 'huggingface-hub[torch]>=0.22'. The rest of requirements.txt is\n"
        "demo and training tooling."
    ),
    "mast3r": (
        "MASt3R needs its GitHub repository, which vendors DUSt3R in turn:\n"
        "  git clone --recursive https://github.com/naver/mast3r third_party/mast3r\n"
        "No setup.py here either; put the repo root on sys.path the same way.\n"
        "Its vendored dust3r/ is byte-identical to the standalone clone, so the\n"
        "two backends share one environment safely despite differing commit SHAs.\n"
        "Use the *metric* checkpoint; the others return arbitrary scale."
    ),
    "fast3r": (
        "Fast3R needs its GitHub repository:\n"
        "  git clone https://github.com/facebookresearch/fast3r third_party/fast3r\n"
        "  pip install --no-deps -e third_party/fast3r\n"
        "This one does have a setup.py. Use --no-deps: requirements.txt pulls\n"
        "deepspeed, open3d, wandb and numpy<2 for training, none of which\n"
        "inference needs and several of which fight a modern environment.\n"
        "Inference imports need only: omegaconf hydra-core lightning rich.\n"
        "It ingests all views in one pass rather than pairwise."
    ),
}


class PoseFreeError(RuntimeError):
    """Raised when a pose-free backend is unavailable or fails."""


@dataclass
class PoseFreeResult:
    """What a pose-free method returns for one specimen."""

    method: str
    position_ids: list[str]
    rotations: np.ndarray          # (V, 3, 3) world_from_cam, OpenCV convention
    centres: np.ndarray            # (V, 3) camera centres
    points: np.ndarray             # (N, 3) reconstructed point cloud
    confidence: np.ndarray | None = None
    is_metric: bool = False
    scale_applied: float = 1.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_views(self) -> int:
        return len(self.position_ids)


# --------------------------------------------------------------------------
# Similarity alignment -- the part that makes any comparison meaningful
# --------------------------------------------------------------------------


def umeyama(
    source: np.ndarray, target: np.ndarray, *, with_scale: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Solves for ``R``, ``t`` and ``s`` minimising ``|| s R x + t - y ||^2``
    (Umeyama 1991), including the reflection guard that a naive SVD solution
    omits -- without it a poorly conditioned point set can produce a mirrored
    "solution" with a lower residual than the correct one, which here would
    silently report a flipped rig as well aligned.

    Args:
        source: ``(N, 3)`` points to be transformed.
        target: ``(N, 3)`` points to align onto.
        with_scale: solve for scale as well as pose.

    Returns:
        ``(rotation, translation, scale)``.
    """
    source = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError(f"shape mismatch: {source.shape} vs {target.shape}")
    if source.shape[0] < 3:
        raise ValueError("need at least 3 correspondences for a similarity fit")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    centred_source = source - mu_source
    centred_target = target - mu_target

    covariance = centred_target.T @ centred_source / source.shape[0]
    u, singular, vh = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[2, 2] = -1.0

    rotation = u @ correction @ vh

    if with_scale:
        variance = (centred_source ** 2).sum() / source.shape[0]
        scale = float((singular * np.diag(correction)).sum() / variance) if variance > 0 else 1.0
    else:
        scale = 1.0

    translation = mu_target - scale * rotation @ mu_source
    return rotation, translation, scale


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two rotations, in degrees."""
    relative = np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64)
    cosine = (np.trace(relative) - 1.0) / 2.0
    return float(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))


@dataclass
class PoseAgreement:
    """How far a pose-free estimate sits from the rig registration."""

    method: str
    n_views: int
    scale: float
    centre_rmse_m: float
    centre_max_m: float
    rotation_rmse_deg: float
    rotation_max_deg: float
    azimuth_rmse_deg: float
    per_view: dict[str, dict]

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "n_views": self.n_views,
            "scale": self.scale,
            "centre_rmse_m": self.centre_rmse_m,
            "centre_max_m": self.centre_max_m,
            "rotation_rmse_deg": self.rotation_rmse_deg,
            "rotation_max_deg": self.rotation_max_deg,
            "azimuth_rmse_deg": self.azimuth_rmse_deg,
        }

    def __str__(self) -> str:  # pragma: no cover - reporting convenience
        return (
            f"{self.method}: centre RMSE {self.centre_rmse_m * 100:.1f} cm "
            f"(max {self.centre_max_m * 100:.1f}), rotation RMSE "
            f"{self.rotation_rmse_deg:.1f} deg (max {self.rotation_max_deg:.1f}), "
            f"azimuth RMSE {self.azimuth_rmse_deg:.1f} deg, scale {self.scale:.3f}"
        )


def compare_poses(
    result: PoseFreeResult,
    rig,
    *,
    with_scale: bool = True,
) -> PoseAgreement:
    """Align a pose-free estimate to the rig and report what is left over.

    The alignment removes the seven degrees of freedom that carry no information
    -- global rotation, translation and scale. What remains is genuine
    disagreement about the rig's shape, and the azimuth component is the one this
    project needs, since that is the parameter the refinement cannot pin down.

    Args:
        result: the pose-free estimate.
        rig: a :class:`~ggssvt.geometry.rig.RigSolution`.
        with_scale: solve for scale. Set False for a metric method to test
            whether its scale is actually right.

    Returns:
        A :class:`PoseAgreement`.
    """
    ours_centres, ours_rotations, ids = [], [], []
    for index, position_id in enumerate(result.position_ids):
        if position_id not in rig.poses:
            continue
        pose = rig.pose(position_id)
        ours_centres.append(pose.centre)
        ours_rotations.append(pose.rotation)
        ids.append((index, position_id))

    if len(ids) < 3:
        raise PoseFreeError(
            f"only {len(ids)} shared views; need at least 3 to align"
        )

    ours_centres = np.array(ours_centres)
    theirs_centres = np.array([result.centres[i] for i, _ in ids])

    rotation, translation, scale = umeyama(
        theirs_centres, ours_centres, with_scale=with_scale
    )
    aligned_centres = scale * theirs_centres @ rotation.T + translation

    centre_errors = np.linalg.norm(aligned_centres - ours_centres, axis=1)

    rotation_errors, azimuth_errors, per_view = [], [], {}
    for slot, (index, position_id) in enumerate(ids):
        aligned_rotation = rotation @ result.rotations[index]
        angle = rotation_angle_deg(ours_rotations[slot], aligned_rotation)
        rotation_errors.append(angle)

        ours_azimuth = math.degrees(
            math.atan2(ours_centres[slot][1], ours_centres[slot][0])
        )
        theirs_azimuth = math.degrees(
            math.atan2(aligned_centres[slot][1], aligned_centres[slot][0])
        )
        azimuth = (theirs_azimuth - ours_azimuth + 180.0) % 360.0 - 180.0
        azimuth_errors.append(azimuth)

        per_view[position_id] = {
            "centre_error_m": float(centre_errors[slot]),
            "rotation_error_deg": float(angle),
            "azimuth_error_deg": float(azimuth),
        }

    rotation_errors = np.array(rotation_errors)
    azimuth_errors = np.array(azimuth_errors)

    return PoseAgreement(
        method=result.method,
        n_views=len(ids),
        scale=float(scale),
        centre_rmse_m=float(np.sqrt((centre_errors ** 2).mean())),
        centre_max_m=float(centre_errors.max()),
        rotation_rmse_deg=float(np.sqrt((rotation_errors ** 2).mean())),
        rotation_max_deg=float(rotation_errors.max()),
        azimuth_rmse_deg=float(np.sqrt((azimuth_errors ** 2).mean())),
        per_view=per_view,
    )


# --------------------------------------------------------------------------
# Scale recovery against the measured depth
# --------------------------------------------------------------------------


def recover_scale_from_depth(
    predicted_depth: np.ndarray,
    measured_depth_m: np.ndarray,
    *,
    subject_mask: np.ndarray | None = None,
    min_pixels: int = 200,
) -> tuple[float, dict]:
    """Metric scale for a scale-free depth prediction, from the Kinect depth.

    Uses the ratio of medians rather than the median of ratios or a least-squares
    fit. A least-squares fit is dominated by the far background, where both
    depths are large and the pose-free prediction is least reliable; a mean of
    ratios is destroyed by the near-zero measured depths at range boundaries.

    Args:
        predicted_depth: ``(H, W)`` depth in arbitrary units.
        measured_depth_m: ``(H, W)`` Kinect depth in metres, 0 where invalid.
        subject_mask: restrict to the plant, which is where the reconstruction
            is actually used.

    Returns:
        ``(scale, diagnostics)``. Multiply the prediction by ``scale`` for metres.
    """
    predicted = np.asarray(predicted_depth, dtype=np.float64)
    measured = np.asarray(measured_depth_m, dtype=np.float64)
    if predicted.shape != measured.shape:
        raise ValueError(
            f"depth shapes differ: {predicted.shape} vs {measured.shape}"
        )

    valid = (measured > 0) & np.isfinite(predicted) & (predicted > 0)
    if subject_mask is not None:
        valid &= subject_mask.astype(bool)

    count = int(valid.sum())
    if count < min_pixels:
        raise PoseFreeError(
            f"only {count} pixels have both a prediction and a measurement; "
            f"need {min_pixels}"
        )

    scale = float(np.median(measured[valid]) / np.median(predicted[valid]))
    residual = measured[valid] - scale * predicted[valid]

    return scale, {
        "n_pixels": count,
        "scale": scale,
        "residual_median_m": float(np.median(np.abs(residual))),
        "residual_p90_m": float(np.percentile(np.abs(residual), 90)),
    }


# --------------------------------------------------------------------------
# Backend availability
# --------------------------------------------------------------------------

_BACKEND_MODULES = {
    "dust3r": "dust3r",
    "mast3r": "mast3r",
    "fast3r": "fast3r",
}


def backend_is_available(method: str) -> tuple[bool, str]:
    """Whether a pose-free backend can be imported.

    Checks the *code*, not the weights. All three sets of weights are open on
    HuggingFace; it is the repositories that have to be cloned, and that is what
    actually blocks a run.
    """
    module = _BACKEND_MODULES.get(method)
    if module is None:
        return False, f"unknown method {method!r}; expected one of {sorted(_BACKEND_MODULES)}"

    import importlib.util

    if importlib.util.find_spec(module) is None:
        return False, f"`{module}` is not importable.\n{INSTALL_HELP[method]}"
    return True, ""


def available_backends() -> dict[str, tuple[bool, str]]:
    """Availability of every pose-free backend."""
    return {name: backend_is_available(name) for name in _BACKEND_MODULES}


__all__ = [
    "DUST3R_REPO",
    "FAST3R_REPO",
    "INSTALL_HELP",
    "MAST3R_METRIC_REPO",
    "PoseAgreement",
    "PoseFreeError",
    "PoseFreeResult",
    "available_backends",
    "backend_is_available",
    "compare_poses",
    "recover_scale_from_depth",
    "rotation_angle_deg",
    "umeyama",
]
