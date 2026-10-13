"""Interchangeable appearance backbones: none, DINOv2, DINOv3.

Box 3 of the system diagram is a DINO vision transformer feeding the 3D feature
volume. This module makes that box swappable so the contribution of the
self-supervised foundation backbone can be measured rather than assumed:

``cnn``
    No DINO. A patch-embedding stem trained from scratch on the RGB-D frames.
    This is the control condition -- it is the *only* one that sees depth and
    validity directly, so it is not a strawman, it is the geometry-only model.

``dinov2``
    DINOv2 patch tokens (Oquab et al., 2023). Open weights.

``dinov3``
    DINOv3 patch tokens. **Gated on HuggingFace**: Meta requires a manual access
    request and licence acceptance per account. Until that is granted, loading
    raises with instructions rather than failing obscurely.

All three expose the same contract::

    tokens = backbone(rgb, depth, valid)     # (N, C, grid_h, grid_w)
    backbone.grid_size(height, width)        # -> (grid_h, grid_w)

Every DINO backbone is frozen by default. With twenty-eight labelled specimens
there is no prospect of fine-tuning 21M+ parameters without memorising the set,
and a frozen backbone is what makes the label-efficiency claim meaningful in the
first place.

Because a frozen RGB backbone cannot ingest depth, the DINO variants embed depth
and validity through a small parallel stem and add it to the DINO tokens. That
keeps the appearance prior intact while the geometry still reaches the encoder,
and it keeps the comparison fair -- every condition sees the same information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MODEL

# DINO models were trained with ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DINOV2_REPOS = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
}
DINOV3_REPOS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

def dinov3_access_help(variant: str = "base") -> str:
    """Instructions for the specific DINOv3 repository that was refused.

    Names the variant actually requested. Pointing at a different size than the
    one that failed sends people to accept a licence that does not unblock them
    -- access is granted per repository, not per model family.
    """
    repo = DINOV3_REPOS.get(variant, DINOV3_REPOS["base"])
    return (
        "DINOv3 is gated as a Gating Group Collection: one approval covers every\n"
        "variant, so you only request once. Meta approves manually, per account.\n"
        f"  1. Open https://huggingface.co/{repo}\n"
        "  2. Accept the licence and submit the access request.\n"
        "  3. Wait for approval. Track it at\n"
        "     https://huggingface.co/settings/gated-repos -- it must read ACCEPTED,\n"
        "     not PENDING. `hf auth login` alone does nothing until it is granted.\n"
        "  4. Confirm the machine is logged in as the account that was approved:\n"
        "     python -c \"from huggingface_hub import whoami; print(whoami()['name'])\"\n"
        "  5. Re-check with:\n"
        "     python -c \"from ggssvt.models.backbones import backbone_is_available; "
        f"print(backbone_is_available('dinov3','{variant}'))\"\n"
        "Until then run --backbones cnn dinov2, which needs no access request."
    )


# Kept as a module constant for callers that want the generic text.
DINOV3_ACCESS_HELP = dinov3_access_help("base")


class BackboneError(RuntimeError):
    """Raised when a backbone cannot be constructed or loaded."""


class Backbone(nn.Module, ABC):
    """Common interface for the appearance stems."""

    name: str = "backbone"
    patch_size: int = 16

    @abstractmethod
    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Map ``(N, 3, H, W)`` RGB plus depth and validity to patch tokens.

        Returns:
            ``(N, C, grid_h, grid_w)``.
        """

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        """Token grid the backbone produces for an input of this size."""
        return height // self.patch_size, width // self.patch_size

    @property
    def n_frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class CnnBackbone(Backbone):
    """Trainable RGB-D patch stem. The no-DINO control condition."""

    name = "cnn"

    def __init__(
        self, embed_dim: int = MODEL.embed_dim, patch_size: int = MODEL.patch_size
    ):
        super().__init__()
        self.patch_size = patch_size
        self.project = nn.Conv2d(5, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        return self.norm(self.project(torch.cat([rgb, depth, valid], dim=1)))


class _DinoBackbone(Backbone):
    """Shared machinery for the DINOv2 and DINOv3 stems."""

    repos: dict[str, str] = {}

    def __init__(
        self,
        embed_dim: int = MODEL.embed_dim,
        variant: str = "small",
        freeze: bool = True,
        depth_stem: bool = True,
    ):
        super().__init__()
        if variant not in self.repos:
            raise BackboneError(
                f"unknown {self.name} variant {variant!r}; expected one of "
                f"{sorted(self.repos)}"
            )

        self.repo = self.repos[variant]
        self.variant = variant
        self.frozen = freeze

        model, hidden_size, patch_size = self._load(self.repo)
        self.backbone = model
        self.patch_size = patch_size

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

        self.project = nn.Linear(hidden_size, embed_dim)
        self.depth_stem = (
            nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)
            if depth_stem
            else None
        )

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @staticmethod
    def _load(repo: str):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise BackboneError(
                "the DINO backbones need `transformers`; install it with "
                "`pip install transformers`"
            ) from exc

        try:
            model = AutoModel.from_pretrained(repo)
        except Exception as exc:
            message = str(exc)
            if "gated" in message.lower() or "401" in message or "403" in message:
                raise BackboneError(f"{repo} is not accessible.\n\n{DINOV3_ACCESS_HELP}") from exc
            raise BackboneError(f"could not load {repo}: {message}") from exc

        config = model.config
        patch_size = getattr(config, "patch_size", 16)
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            raise BackboneError(f"{repo} exposes no hidden_size; unsupported architecture")
        return model, hidden, patch_size

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def _resize_for_backbone(self, rgb: torch.Tensor) -> torch.Tensor:
        """Crop-free resize to the nearest size the patch grid divides evenly."""
        height, width = rgb.shape[-2:]
        target_h = max(self.patch_size, round(height / self.patch_size) * self.patch_size)
        target_w = max(self.patch_size, round(width / self.patch_size) * self.patch_size)
        if (target_h, target_w) == (height, width):
            return rgb
        return F.interpolate(
            rgb, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

    def grid_size(self, height: int, width: int) -> tuple[int, int]:
        target_h = max(self.patch_size, round(height / self.patch_size) * self.patch_size)
        target_w = max(self.patch_size, round(width / self.patch_size) * self.patch_size)
        return target_h // self.patch_size, target_w // self.patch_size

    def patch_tokens(self, rgb: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Frozen DINO patch tokens for ``(N, 3, H, W)`` RGB in [0, 1].

        Returns:
            ``(tokens, grid_h, grid_w)`` with tokens ``(N, P, hidden)``.
        """
        resized = self._resize_for_backbone(rgb)
        normalised = (resized - self.mean) / self.std

        grid_h = resized.shape[-2] // self.patch_size
        grid_w = resized.shape[-1] // self.patch_size

        context = torch.no_grad() if self.frozen else torch.enable_grad()
        with context:
            outputs = self.backbone(pixel_values=normalised)
            hidden = outputs.last_hidden_state

        # Drop the CLS token and any register tokens: whatever is left over
        # beyond grid_h * grid_w sits at the front of the sequence.
        n_patches = grid_h * grid_w
        if hidden.shape[1] < n_patches:
            raise BackboneError(
                f"{self.repo} returned {hidden.shape[1]} tokens for a "
                f"{grid_h}x{grid_w} grid; cannot align patches"
            )
        return hidden[:, hidden.shape[1] - n_patches :], grid_h, grid_w

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        tokens, grid_h, grid_w = self.patch_tokens(rgb)
        tokens = self.project(tokens)
        tokens = tokens.transpose(1, 2).reshape(rgb.shape[0], -1, grid_h, grid_w)

        if self.depth_stem is not None:
            geometry = torch.cat([depth, valid], dim=1)
            geometry = F.interpolate(
                geometry,
                size=(grid_h * self.patch_size, grid_w * self.patch_size),
                mode="nearest",
            )
            tokens = tokens + self.depth_stem(geometry)

        return tokens


class Dinov2Backbone(_DinoBackbone):
    """DINOv2 patch tokens. Open weights, no access request needed."""

    name = "dinov2"
    repos = DINOV2_REPOS


class Dinov3Backbone(_DinoBackbone):
    """DINOv3 patch tokens. Gated -- see :data:`DINOV3_ACCESS_HELP`."""

    name = "dinov3"
    repos = DINOV3_REPOS


BACKBONES = {
    "cnn": CnnBackbone,
    "dinov2": Dinov2Backbone,
    "dinov3": Dinov3Backbone,
}


def build_backbone(
    kind: str = "cnn",
    *,
    embed_dim: int = MODEL.embed_dim,
    patch_size: int = MODEL.patch_size,
    variant: str = "small",
    freeze: bool = True,
) -> Backbone:
    """Construct one of the interchangeable appearance backbones.

    Args:
        kind: ``"cnn"``, ``"dinov2"`` or ``"dinov3"``.
        patch_size: used by ``cnn`` only; the DINO stems use their own.
        variant: DINO size, ``"small"`` / ``"base"`` / ``"large"``.
        freeze: keep DINO weights fixed. Strongly recommended at this sample size.

    Raises:
        BackboneError: on an unknown kind, a missing dependency, or gated weights.
    """
    if kind not in BACKBONES:
        raise BackboneError(
            f"unknown backbone {kind!r}; expected one of {sorted(BACKBONES)}"
        )
    if kind == "cnn":
        return CnnBackbone(embed_dim=embed_dim, patch_size=patch_size)
    return BACKBONES[kind](embed_dim=embed_dim, variant=variant, freeze=freeze)


def repo_access(repo_id: str) -> tuple[bool, str]:
    """Whether *this account* can download a repository's files.

    Uses :func:`huggingface_hub.auth_check`, which is the only call that answers
    the question actually being asked. Two weaker checks look like they work and
    do not:

    * ``model_info`` succeeding proves nothing -- a gated repository serves its
      metadata to anonymous callers.
    * ``info.gated`` describes the *repository's policy*, not the caller's
      permission. It reads ``"manual"`` forever, including for accounts that
      have been granted access. Gating a backbone on this flag means the
      backbone stays skipped even after approval comes through, which is a
      silent and very confusing failure.

    Returns:
        ``(accessible, reason)``. ``reason`` is empty when accessible.
    """
    try:
        from huggingface_hub import auth_check
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        return False, "huggingface_hub >= 0.26 is not installed"

    try:
        auth_check(repo_id)
        return True, ""
    except GatedRepoError:
        return False, f"{repo_id} is gated and this account has not been granted access."
    except RepositoryNotFoundError:
        return False, f"{repo_id} does not exist, or requires a token to see."
    except Exception as exc:
        return False, f"{repo_id} could not be checked: {str(exc)[:160]}"


def backbone_is_available(kind: str, variant: str = "small") -> tuple[bool, str]:
    """Check whether a backbone can actually be loaded, without building it.

    Returns:
        ``(available, reason)``. ``reason`` is empty when available.
    """
    if kind == "cnn":
        return True, ""
    repos = DINOV2_REPOS if kind == "dinov2" else DINOV3_REPOS
    if variant not in repos:
        return False, f"unknown variant {variant!r}"

    accessible, reason = repo_access(repos[variant])
    if accessible:
        return True, ""

    help_text = dinov3_access_help(variant) if kind == "dinov3" else ""
    return False, f"{reason}\n{help_text}".rstrip()


__all__ = [
    "BACKBONES",
    "DINOV3_ACCESS_HELP",
    "Backbone",
    "BackboneError",
    "CnnBackbone",
    "Dinov2Backbone",
    "Dinov3Backbone",
    "backbone_is_available",
    "build_backbone",
    "dinov3_access_help",
    "repo_access",
]
