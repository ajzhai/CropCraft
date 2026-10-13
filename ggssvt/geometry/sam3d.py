"""SAM3D: promptable 2D segmentation lifted to a 3D-consistent subject mask.

This follows the design in the predecessor project's
``neural_geometry/sam3d/sam3d_pipeline.py``: run a promptable segmenter on each
RGB view, back-project the masked depth, and enforce consistency across views
before handing the result to the volumetric stage. It is a *segmentation*
component, not a mesh generator.

The default cylinder segmentation in :mod:`ggssvt.geometry.segment` is purely
geometric -- everything inside a cylinder about the plant axis. It is robust and
free, but it has one systematic failure: anything physically inside that
cylinder is kept, so a rig pole or bench edge directly behind the plant is
labelled subject. SAM3D exists to cut those away on appearance.

**The prompt comes from the geometry.** SAM needs to be told what to segment, and
because the views are already registered, the projected plant axis and the
bounding box of the geometric mask are available for free. That makes SAM3D a
*refinement* of the geometric mask rather than an independent segmenter, which
is both why it works without manual clicks and a limitation worth stating: it
cannot recover a plant the geometric stage missed entirely.

Three consistency rules keep SAM honest:

1. **3D gating.** The SAM mask is intersected with the working cylinder, so a
   mask that leaks onto the far wall is trimmed by geometry.
2. **Coverage guard.** A mask covering implausibly much or little of the prompt
   box is rejected and the geometric mask is kept for that view. SAM
   occasionally returns the whole floor when the prompt straddles a boundary.
3. **Multi-view agreement.** Masks are scored by how well their back-projected
   points land where the other views' points already are; a view that disagrees
   with the rest is reverted.

Model availability, checked 21 August 2026:

* ``facebook/sam-vit-base`` / ``-large`` / ``-huge`` -- open.
* ``facebook/sam2-hiera-*`` -- open.
* ``facebook/sam3`` and ``facebook/sam-3d-objects`` -- **gated**, manual approval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import KINECT_V2, ROI_RADIUS_M, ROI_Z_MAX_M, ROI_Z_MIN_M, Intrinsics
from ..data.io import backproject, depth_validity, project
from .rig import RigSolution, ViewPose
from .segment import ViewSegmentation, segment_view

SAM_REPOS = {
    "base": "facebook/sam-vit-base",
    "large": "facebook/sam-vit-large",
    "huge": "facebook/sam-vit-huge",
}
SAM3_REPO = "facebook/sam3"
SAM3D_OBJECTS_REPO = "facebook/sam-3d-objects"

GATED_HELP = (
    "SAM 3 and SAM 3D Objects are gated on HuggingFace; Meta approves access "
    "manually, per account.\n"
    "  1. Open https://huggingface.co/facebook/sam-3d-objects (and /facebook/sam3)\n"
    "  2. Accept the licence and request access; wait for approval.\n"
    "  3. Authenticate: `hf auth login` (or set HF_TOKEN).\n"
    "The open SAM checkpoints (facebook/sam-vit-base/large/huge) need no request "
    "and are what --sam-model uses by default."
)

# A SAM mask is rejected if it covers less or more of its prompt box than this.
MIN_BOX_FILL = 0.02
MAX_BOX_FILL = 0.92


class Sam3DError(RuntimeError):
    """Raised when the SAM backend cannot be loaded."""


@dataclass
class Sam3DStats:
    """What SAM actually changed, per specimen."""

    n_views: int = 0
    n_accepted: int = 0
    n_rejected_coverage: int = 0
    n_rejected_agreement: int = 0
    mean_pixels_before: float = 0.0
    mean_pixels_after: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.n_accepted / max(1, self.n_views)

    @property
    def pixel_change(self) -> float:
        """Fraction of subject pixels SAM removed (negative means it added)."""
        if self.mean_pixels_before <= 0:
            return 0.0
        return 1.0 - self.mean_pixels_after / self.mean_pixels_before

    def as_dict(self) -> dict:
        return {
            "n_views": self.n_views,
            "n_accepted": self.n_accepted,
            "n_rejected_coverage": self.n_rejected_coverage,
            "n_rejected_agreement": self.n_rejected_agreement,
            "acceptance_rate": self.acceptance_rate,
            "pixel_change": self.pixel_change,
        }


def sam_is_available(model: str = "base") -> tuple[bool, str]:
    """Whether a SAM checkpoint can be downloaded by this account.

    Delegates to :func:`ggssvt.models.backbones.repo_access`, which asks whether
    *this account* can fetch the files rather than whether the repository is
    gated at all. The distinction matters: a repository stays flagged as gated
    forever, including for accounts that have been granted access.
    """
    from ..models.backbones import repo_access

    repo = SAM_REPOS.get(model, model)
    accessible, reason = repo_access(repo)
    if accessible:
        return True, ""
    return False, f"{reason}\n{GATED_HELP}".rstrip()


class Sam3DSegmenter:
    """Promptable segmentation refined into a 3D-consistent subject mask.

    Args:
        model: key in :data:`SAM_REPOS`, or any HuggingFace SAM repo id.
        device: torch device string.
    """

    def __init__(self, model: str = "base", device: str = "cpu"):
        self.repo = SAM_REPOS.get(model, model)
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> Sam3DSegmenter:
        """Load the SAM weights. Called lazily by :meth:`segment_specimen`."""
        if self._model is not None:
            return self

        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise Sam3DError(
                "SAM3D needs `transformers` and `torch`; install with "
                "`pip install transformers torch`"
            ) from exc

        try:
            self._processor = SamProcessor.from_pretrained(self.repo)
            self._model = SamModel.from_pretrained(self.repo).to(self.device).eval()
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                raise Sam3DError(f"{self.repo} is not accessible.\n\n{GATED_HELP}") from exc
            raise Sam3DError(f"could not load {self.repo}: {message[:200]}") from exc

        self._torch = torch
        return self

    # -- prompting -----------------------------------------------------------

    @staticmethod
    def _prompt_from_geometry(
        pose: ViewPose, intrinsics: Intrinsics, geometric_mask: np.ndarray
    ) -> tuple[list[float] | None, list[list[float]]]:
        """Build a box and point prompt from the registered geometry.

        The box is the bounding box of the geometric mask, padded slightly. The
        point sits on the world plant axis at mid-canopy height, which is inside
        the plant in every view by construction.
        """
        rows, cols = np.nonzero(geometric_mask)
        if rows.size < 30:
            return None, []

        pad = 8
        height, width = geometric_mask.shape
        box = [
            float(max(0, cols.min() - pad)),
            float(max(0, rows.min() - pad)),
            float(min(width - 1, cols.max() + pad)),
            float(min(height - 1, rows.max() + pad)),
        ]

        axis = np.array([[0.0, 0.0, 0.35], [0.0, 0.0, 0.6]])
        uv, depth = project(pose.to_camera(axis), intrinsics)
        points = [
            [float(u), float(v)]
            for (u, v), z in zip(uv, depth)
            if z > 0.2 and 0 <= u < width and 0 <= v < height
        ]
        return box, points

    def _run_sam(
        self, rgb: np.ndarray, box: list[float], points: list[list[float]]
    ) -> np.ndarray:
        """One SAM forward pass. Returns a boolean mask at the input resolution."""
        torch = self._torch
        image = (rgb * 255).astype(np.uint8)

        kwargs = {"input_boxes": [[box]]}
        if points:
            kwargs["input_points"] = [[points]]
            kwargs["input_labels"] = [[[1] * len(points)]]

        inputs = self._processor(image, return_tensors="pt", **kwargs).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        return masks[0][0, 0].numpy().astype(bool)

    # -- consistency ---------------------------------------------------------

    @staticmethod
    def _gate_in_3d(
        mask: np.ndarray,
        depth_m: np.ndarray,
        pose: ViewPose,
        intrinsics: Intrinsics,
        radius_m: float,
    ) -> np.ndarray:
        """Trim a SAM mask to the working cylinder.

        Geometry has the final say: a mask that leaked onto the wall behind the
        plant is cut here even if SAM was confident about it.
        """
        points_cam = backproject(depth_m, intrinsics).reshape(-1, 3).astype(np.float64)
        world = points_cam @ pose.rotation.T + pose.centre

        radial = np.linalg.norm(world[:, :2], axis=1)
        inside = (
            (radial < radius_m)
            & (world[:, 2] > ROI_Z_MIN_M)
            & (world[:, 2] < ROI_Z_MAX_M)
        ).reshape(depth_m.shape)

        return mask & inside & depth_validity(depth_m)

    def segment_specimen(
        self,
        specimen,
        rig: RigSolution,
        *,
        intrinsics: Intrinsics = KINECT_V2,
        radius_m: float = ROI_RADIUS_M,
        agreement_voxel_m: float = 0.03,
        min_agreement: float = 0.25,
        verbose: bool = False,
    ) -> tuple[dict[str, ViewSegmentation], Sam3DStats]:
        """Segment every view with SAM, falling back to geometry where it fails.

        Returns:
            ``(segmentations, stats)``.
        """
        self.load()
        stats = Sam3DStats()

        geometric: dict[str, ViewSegmentation] = {}
        refined: dict[str, ViewSegmentation] = {}

        for view in specimen.views:
            pose = rig.pose(view.position_id)
            depth = view.load_depth()
            rgb = view.load_rgb()

            base = segment_view(depth, pose, rgb=rgb, intrinsics=intrinsics)
            geometric[view.position_id] = base
            stats.n_views += 1

            box, points = self._prompt_from_geometry(pose, intrinsics, base.mask)
            if box is None:
                refined[view.position_id] = base
                continue

            sam_mask = self._run_sam(rgb, box, points)
            gated = self._gate_in_3d(sam_mask, depth, pose, intrinsics, radius_m)

            box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            fill = gated.sum() / box_area
            if not (MIN_BOX_FILL <= fill <= MAX_BOX_FILL) or gated.sum() < 200:
                stats.n_rejected_coverage += 1
                refined[view.position_id] = base
                continue

            flat_world = (
                backproject(depth, intrinsics).reshape(-1, 3).astype(np.float64)
                @ pose.rotation.T
                + pose.centre
            )
            refined[view.position_id] = ViewSegmentation(
                position_id=view.position_id,
                mask=gated,
                depth_m=depth,
                points_world=flat_world.reshape(*depth.shape, 3)[gated].astype(np.float32),
                colours=rgb[gated].astype(np.float32),
            )
            stats.n_accepted += 1

        self._revert_disagreeing_views(
            refined, geometric, stats, agreement_voxel_m, min_agreement, verbose
        )

        stats.mean_pixels_before = float(
            np.mean([s.mask.sum() for s in geometric.values()])
        )
        stats.mean_pixels_after = float(
            np.mean([s.mask.sum() for s in refined.values()])
        )
        return refined, stats

    @staticmethod
    def _revert_disagreeing_views(
        refined: dict[str, ViewSegmentation],
        geometric: dict[str, ViewSegmentation],
        stats: Sam3DStats,
        voxel_m: float,
        min_agreement: float,
        verbose: bool,
    ) -> None:
        """Revert any view whose SAM mask disagrees with the rest in 3D.

        SAM segments each frame independently and has no idea the twelve views
        show one object. This is the step that makes the result *3D*: a mask is
        kept only if its back-projected points land where the other views'
        points already are.
        """
        if len(refined) < 3:
            return

        def keys(segmentation: ViewSegmentation) -> set[tuple[int, int, int]]:
            if segmentation.n_points == 0:
                return set()
            index = np.floor(segmentation.points_world / voxel_m).astype(np.int32)
            return {tuple(k) for k in index}

        key_sets = {pid: keys(s) for pid, s in refined.items()}

        for position_id in list(refined):
            others: set[tuple[int, int, int]] = set()
            for other, other_keys in key_sets.items():
                if other != position_id:
                    others |= other_keys

            own = key_sets[position_id]
            if not own:
                continue
            overlap = len(own & others) / len(own)
            if overlap < min_agreement:
                stats.n_rejected_agreement += 1
                stats.n_accepted = max(0, stats.n_accepted - 1)
                refined[position_id] = geometric[position_id]
                key_sets[position_id] = keys(geometric[position_id])
                if verbose:
                    print(
                        f"    {position_id}: SAM mask reverted, "
                        f"3D agreement {overlap:.2f} < {min_agreement}"
                    )


__all__ = [
    "GATED_HELP",
    "SAM3D_OBJECTS_REPO",
    "SAM3_REPO",
    "SAM_REPOS",
    "Sam3DError",
    "Sam3DSegmenter",
    "Sam3DStats",
    "sam_is_available",
]
