"""Which reconstruction methods the project page can show, and why.

The page used to offer a two-way toggle labelled "segmenter", because the only
two things it could show were the geometric carve and the SAM3D one. That framing
was too narrow. What the toggle actually selects is *where the occupancy field
came from*, and there are more answers to that than two.

Three kinds of entry live here.

**Segmentation variants.** Same twelve views, same carve, different subject mask.
Geometric and SAM3D.

**Reconstruction operators.** Same views, same masks, a different way of turning
them into occupancy. TSDF fusion is here, held at the carve's own 128^3 and 12 mm
precisely so the comparison isolates the operator rather than the grid.

**Sampling variants.** Same carve, fewer views. These are already built by the
view-count ablation and they belong on the page because the ablation's finding is
visual before it is numerical: at four views the hull balloons to something the
size of a wardrobe, and reading that off a table is far less convincing than
rotating it.

**Pose-free reconstructions.** DUSt3R, MASt3R and Fast3R estimate cameras and
geometry from images alone, so they share no failure mode with the carve. They
are declared here but produce no cache until the GPU run happens, and entries
whose cache is missing are simply skipped.

`UNSUPPORTED` records what cannot go on the page and why, so the question does
not have to be re-answered each time it comes up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import WORK_DIR


@dataclass(frozen=True)
class ReconstructionMethod:
    """One source of occupancy fields the viewer can switch to."""

    key: str
    label: str
    group: str          # segmentation | sampling | pose-free
    note: str           # one line, shown beside the toggle
    cache: Path

    def available(self) -> bool:
        return (self.cache / "quality.json").exists()


METHODS: tuple[ReconstructionMethod, ...] = (
    ReconstructionMethod(
        key="geometric",
        label="Geometric",
        group="segmentation",
        note="Excess-green and depth gating inside a cylinder about the plant axis. "
             "The reference condition for everything else on this page.",
        cache=WORK_DIR / "cache",
    ),
    ReconstructionMethod(
        key="sam3d",
        label="SAM3D",
        group="segmentation",
        note="SAM prompted from the registered geometry, with masks reverted where "
             "they disagree with the rest in 3D. Tighter masks, less coverage: it "
             "drops three specimens the geometric gate keeps.",
        cache=WORK_DIR / "cache_sam3d",
    ),
    ReconstructionMethod(
        key="tsdf",
        label="TSDF fusion",
        group="reconstruction",
        note="The same twelve depth maps integrated as a signed distance field "
             "instead of intersected as silhouette cones. Concavities survive and "
             "unobserved space stays empty, so 25 of 36 reconstructions become "
             "physically capable of weighing their plant, against 8 for the carve "
             "at this same resolution.",
        cache=WORK_DIR / "cache_tsdf",
    ),
    ReconstructionMethod(
        key="views6",
        label="6 views",
        group="sampling",
        note="Every second view discarded. Still carves, but the hull is already "
             "fifteen times too large for the mass it has to weigh.",
        cache=WORK_DIR / "cache_v6",
    ),
    ReconstructionMethod(
        key="views4",
        label="4 views",
        group="sampling",
        note="Four views at 90 degrees, the visual-hull minimum for a convex object. "
             "A plant is the opposite of convex, and not one of the twenty-five that "
             "survive is physically capable of weighing what it weighs.",
        cache=WORK_DIR / "cache_v4",
    ),
    ReconstructionMethod(
        key="views3",
        label="3 views",
        group="sampling",
        note="Three views at 120 degrees. Thirteen specimens fail the quality gate "
             "outright; the rest are envelopes.",
        cache=WORK_DIR / "cache_v3",
    ),
    ReconstructionMethod(
        key="dust3r",
        label="DUSt3R",
        group="pose-free",
        note="Pairwise point maps fused by global alignment, with cameras estimated "
             "from the images rather than from depth.",
        cache=WORK_DIR / "cache_dust3r",
    ),
    ReconstructionMethod(
        key="mast3r",
        label="MASt3R",
        group="pose-free",
        note="The metric checkpoint, so the geometry arrives at real scale instead "
             "of an arbitrary one.",
        cache=WORK_DIR / "cache_mast3r",
    ),
    ReconstructionMethod(
        key="fast3r",
        label="Fast3R",
        group="pose-free",
        note="All twelve views in a single forward pass, with one global PnP solve "
             "for the cameras.",
        cache=WORK_DIR / "cache_fast3r",
    ),
)


# Methods that come up as suggestions and cannot go on this page. Keeping the
# reasons written down is cheaper than re-deriving them.
UNSUPPORTED: dict[str, str] = {
    "DeepVoxels": (
        "Per-scene and image-only. It optimises one feature volume over all "
        "observations of a single object, needs hundreds of views where this "
        "dataset has twelve, and outputs novel-view RGB rather than occupancy, so "
        "there is no volume to read or score. It also wants accurate poses, which "
        "is precisely the assumption this project cannot make. Its place is the "
        "related-work chapter, as the origin of lifting 2D features into a "
        "persistent 3D grid, not the results."
    ),
    "NeRF / splatfacto": (
        "No longer unsupported, only unrun. transforms.json is exported for "
        "every specimen and `cli neural-field` reads a trained field back as a "
        "third operator, sweeping the density threshold rather than choosing "
        "one. Needs the other conda environment and a training run per "
        "specimen, so it is sequenced after the campaign."
    ),
    "SAM 3 / SAM 3D Objects": (
        "Gated on HuggingFace and not granted to this account. The adapter exists "
        "and the check reports the block rather than failing at load."
    ),
}


def available_methods() -> list[ReconstructionMethod]:
    """Every method whose cache has actually been built."""
    return [m for m in METHODS if m.available()]


def cache_dirs() -> dict[str, Path]:
    """Available methods as the mapping the payload builder expects."""
    return {m.key: m.cache for m in available_methods()}


def describe() -> dict[str, dict[str, str]]:
    """Label, group and note per available method, for the page."""
    return {
        m.key: {"label": m.label, "group": m.group, "note": m.note}
        for m in available_methods()
    }


__all__ = [
    "METHODS",
    "UNSUPPORTED",
    "ReconstructionMethod",
    "available_methods",
    "cache_dirs",
    "describe",
]
