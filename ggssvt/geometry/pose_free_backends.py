"""Adapters for the DUSt3R, MASt3R and Fast3R backends.

**These have not been executed against the real weights**, which needs a GPU. All
three now install and import on CPU, though, and every call below has been
checked against the installed code rather than against documentation --
see :mod:`tests.test_pose_free_api`, which pins each signature and skips when the
repositories are absent.

That check earned its keep immediately. The Fast3R adapter was wrong three ways:
there is no function named ``inference_multiview``, its ``inference`` takes views
before the model and requires a ``dtype``, and camera poses do not ride inside
each prediction the way :func:`_from_multiview` assumed -- they come from a
separate global PnP solve. DUSt3R's and MASt3R's calls were correct as written.

The maths they feed -- :func:`~ggssvt.geometry.pose_free.compare_poses` and
:func:`~ggssvt.geometry.pose_free.recover_scale_from_depth` -- *is* tested, so a
first run should be watched for wrong conventions rather than for wrong numbers.
:doc:`POSEFREE.md <../POSEFREE>` has the install and run plan.

Two conventions decide whether the output is meaningful, and both are handled
here rather than left to the caller:

**Camera convention.** All three return camera-to-world poses in the OpenCV
convention (+x right, +y down, +z forward), the same as
:mod:`ggssvt.geometry.rig`, so no flip is applied. This is asserted rather than
assumed -- :func:`sanity_check_result` verifies the cameras end up looking
roughly at the scene centroid, which catches a convention change in an upstream
release.

**Scale.** DUSt3R and Fast3R return geometry up to an unknown global scale;
MASt3R's ``_metric`` checkpoint does not. Feeding a scale-free reconstruction
into a volume calculation would measure the rescaling rather than the plant, so
the non-metric backends must go through
:func:`~ggssvt.geometry.pose_free.recover_scale_from_depth` first, and
``PoseFreeResult.is_metric`` records which happened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .pose_free import (
    DUST3R_REPO,
    FAST3R_REPO,
    INSTALL_HELP,
    MAST3R_METRIC_REPO,
    PoseFreeError,
    PoseFreeResult,
    backend_is_available,
)

DEFAULT_IMAGE_SIZE = 512


class PoseFreeBackend(ABC):
    """Common interface for a pose-free reconstruction method."""

    name: str = "posefree"
    repo: str = ""
    returns_metric_scale: bool = False

    def __init__(self, device: str = "cuda", image_size: int = DEFAULT_IMAGE_SIZE):
        self.device = device
        self.image_size = image_size
        self._model = None

    def ensure_available(self) -> None:
        available, reason = backend_is_available(self.name)
        if not available:
            raise PoseFreeError(reason)

    @abstractmethod
    def load(self) -> PoseFreeBackend:
        """Load the weights. Idempotent."""

    @abstractmethod
    def reconstruct(self, specimen) -> PoseFreeResult:
        """Run the method over one specimen's views."""

    def _image_paths(self, specimen) -> tuple[list[str], list[str]]:
        views = sorted(specimen.views, key=lambda v: v.azimuth_deg)
        return [str(v.rgb_path) for v in views], [v.position_id for v in views]


class Dust3rBackend(PoseFreeBackend):
    """DUSt3R: pairwise point maps fused by global alignment.

    Pairwise means the cost grows with the square of the view count. Twelve views
    is 66 unordered pairs at ``complete`` pairing -- but ``symmetrize=True``
    below runs each in both directions, so the real cost is **132 forward
    passes per specimen**, and the whole set is 36 times that. The swin or
    oneref scene graphs exist if that proves too slow.
    """

    name = "dust3r"
    repo = DUST3R_REPO
    returns_metric_scale = False

    def load(self) -> Dust3rBackend:
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["dust3r"]) from exc

        self._model = AsymmetricCroCo3DStereo.from_pretrained(self.repo).to(self.device)
        self._model.eval()
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["dust3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1)

        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

        return _from_scene(scene, self.name, position_ids, is_metric=False)


class Mast3rBackend(PoseFreeBackend):
    """MASt3R with the metric checkpoint.

    The metric variant is the one worth running here: the others return arbitrary
    scale, and a volume computed from arbitrary scale measures the rescaling.
    """

    name = "mast3r"
    repo = MAST3R_METRIC_REPO
    returns_metric_scale = True

    def load(self) -> Mast3rBackend:
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["mast3r"]) from exc

        self._model = AsymmetricMASt3R.from_pretrained(self.repo).to(self.device)
        self._model.eval()

        if "metric" not in self.repo:
            raise PoseFreeError(
                f"{self.repo} is not the metric checkpoint. Use "
                f"{MAST3R_METRIC_REPO}, or the reconstruction will carry an "
                "arbitrary scale and its volume will be meaningless."
            )
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
            from dust3r.image_pairs import make_pairs
            from dust3r.inference import inference
            from dust3r.utils.image import load_images
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["mast3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self._model, self.device, batch_size=1)

        scene = global_aligner(
            output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer
        )
        scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

        return _from_scene(scene, self.name, position_ids, is_metric=True)


class Fast3rBackend(PoseFreeBackend):
    """Fast3R: every view in a single forward pass.

    No pairwise blow-up and no global alignment step, which is why it is the one
    to try first on twelve views.
    """

    name = "fast3r"
    repo = FAST3R_REPO
    returns_metric_scale = False

    def load(self) -> Fast3rBackend:
        if self._model is not None:
            return self
        self.ensure_available()
        try:
            from fast3r.models.fast3r import Fast3R
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["fast3r"]) from exc

        self._model = Fast3R.from_pretrained(self.repo).to(self.device)
        self._model.eval()
        return self

    def reconstruct(self, specimen) -> PoseFreeResult:
        self.load()
        try:
            import torch
            from fast3r.dust3r.inference_multiview import inference
            from fast3r.dust3r.utils.image import load_images
            from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
        except ImportError as exc:
            raise PoseFreeError(INSTALL_HELP["fast3r"]) from exc

        paths, position_ids = self._image_paths(specimen)
        images = load_images(paths, size=self.image_size, verbose=False)

        # Positional: the views come first and the model second, and `dtype` is
        # required rather than defaulted. `profiling=False` returns the dict
        # alone; with profiling it returns a (dict, info) tuple instead.
        output = inference(
            images, self._model, self.device, dtype=torch.float32, verbose=False
        )

        # Fast3R does not put a pose in each prediction the way the earlier
        # adapter assumed. The predictions carry point maps only, and camera
        # poses come from a separate PnP step over all of them at once -- which
        # is the whole point of the method: one global solve rather than the
        # pairwise alignment DUSt3R and MASt3R run.
        poses, _focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            output["preds"],
            niter_PnP=100,
            focal_length_estimation_method="first_view_from_global_head",
        )

        return _from_multiview(output, poses[0], self.name, position_ids)


# --------------------------------------------------------------------------
# Output normalisation
# --------------------------------------------------------------------------


def _from_scene(scene, method: str, position_ids: list[str], *, is_metric: bool):
    """Normalise a DUSt3R/MASt3R global-alignment scene."""
    poses = scene.get_im_poses().detach().cpu().numpy()      # (V, 4, 4) cam-to-world
    points = np.concatenate(
        [p.detach().cpu().numpy().reshape(-1, 3) for p in scene.get_pts3d()], axis=0
    )
    confidence = None
    if hasattr(scene, "get_conf"):
        confidence = np.concatenate(
            [c.detach().cpu().numpy().reshape(-1) for c in scene.get_conf()], axis=0
        )

    return PoseFreeResult(
        method=method,
        position_ids=list(position_ids[: poses.shape[0]]),
        rotations=poses[:, :3, :3],
        centres=poses[:, :3, 3],
        points=points,
        confidence=confidence,
        is_metric=is_metric,
        notes=[] if is_metric else ["scale is arbitrary; align to depth before use"],
    )


def _from_multiview(output, poses, method: str, position_ids: list[str]):
    """Normalise a Fast3R multi-view output.

    Args:
        output: the dict returned by ``fast3r.dust3r.inference_multiview.inference``.
        poses: one ``(4, 4)`` camera-to-world matrix per view, from
            ``MultiViewDUSt3RLitModule.estimate_camera_poses``. They arrive
            separately because Fast3R's predictions hold point maps only -- the
            poses come from a single global PnP solve across every view.
    """
    preds = output["preds"] if isinstance(output, dict) else output
    if len(poses) != len(preds):
        raise PoseFreeError(
            f"Fast3R returned {len(preds)} predictions but {len(poses)} poses"
        )

    rotations, centres, clouds = [], [], []
    for view, pose in zip(preds, poses):
        pose = np.asarray(
            pose.detach().cpu() if hasattr(pose, "detach") else pose, dtype=np.float64
        ).reshape(4, 4)
        rotations.append(pose[:3, :3])
        centres.append(pose[:3, 3])

        pts = view.get("pts3d_in_other_view", view.get("pts3d"))
        if pts is not None:
            arr = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
            clouds.append(arr.reshape(-1, 3))

    return PoseFreeResult(
        method=method,
        position_ids=list(position_ids[: len(rotations)]),
        rotations=np.stack(rotations),
        centres=np.stack(centres),
        points=np.concatenate(clouds, axis=0) if clouds else np.zeros((0, 3)),
        is_metric=False,
        notes=["scale is arbitrary; align to depth before use"],
    )


def sanity_check_result(result: PoseFreeResult) -> list[str]:
    """Catch a convention change before it becomes a wrong number.

    All three backends document OpenCV camera-to-world poses, so the optical axis
    (+z in camera coordinates) should point roughly at the scene. If an upstream
    release switches to OpenGL, or to world-to-camera, this is where it shows --
    as a warning rather than as a plausible-looking reconstruction that is
    inside out.
    """
    warnings: list[str] = []

    if result.n_views != result.rotations.shape[0]:
        warnings.append(
            f"{result.n_views} position ids but {result.rotations.shape[0]} poses"
        )

    for index in range(result.rotations.shape[0]):
        determinant = float(np.linalg.det(result.rotations[index]))
        if abs(determinant - 1.0) > 1e-3:
            warnings.append(
                f"view {index}: rotation determinant {determinant:+.3f}, not a proper rotation"
            )

    if result.points.size:
        centroid = result.points.mean(axis=0)
        facing = 0
        for index in range(result.rotations.shape[0]):
            forward = result.rotations[index] @ np.array([0.0, 0.0, 1.0])
            to_centre = centroid - result.centres[index]
            norm = np.linalg.norm(to_centre)
            if norm > 1e-9 and forward @ (to_centre / norm) > 0:
                facing += 1
        if facing < result.rotations.shape[0] / 2:
            warnings.append(
                f"only {facing}/{result.rotations.shape[0]} cameras face the scene "
                "centroid; the pose convention may have changed upstream "
                "(OpenGL vs OpenCV, or world-from-camera vs camera-from-world)"
            )

    return warnings


BACKENDS = {
    "dust3r": Dust3rBackend,
    "mast3r": Mast3rBackend,
    "fast3r": Fast3rBackend,
}


def build_backend(method: str, **kwargs) -> PoseFreeBackend:
    """Construct a pose-free backend by name."""
    if method not in BACKENDS:
        raise PoseFreeError(
            f"unknown method {method!r}; expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[method](**kwargs)


__all__ = [
    "BACKENDS",
    "Dust3rBackend",
    "Fast3rBackend",
    "Mast3rBackend",
    "PoseFreeBackend",
    "build_backend",
    "sanity_check_result",
]
