"""Architecture diagrams, drawn from the pipeline rather than about it.

Every figure on the project page is generated, and hand-drawn architecture
diagrams are the one thing that reliably drifts: the pipeline changes, the figure
does not, and nobody notices until a reader asks why the paper describes a stage
the code no longer has. These are emitted as SVG from the same constants the
pipeline runs on, so a change to the voxel size or the segmentation threshold
changes the figure with it.

One diagram per methodology, each running the full length of the argument from
image acquisition to a mass in kilograms. They share a layout on purpose: put two
side by side and the stage that differs is the one that does not line up.

SVG rather than PNG because these end up in a dissertation at whatever size the
template decides, and because the text stays selectable and the file stays a few
kilobytes. No external fonts, no embedded raster, nothing to fetch.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

from ..config import (
    ANGULAR_STEP_DEG,
    EXCESS_GREEN_THRESHOLD,
    KINECT_V2,
    N_SWEEP_STEPS,
    ROI_RADIUS_M,
    VOXEL_RESOLUTION,
    VOXEL_SIZE_M,
)
from ..geometry.fusion import FUSION_RESOLUTION, FUSION_VOXEL_M, TRUNCATION_VOXELS

# The page's ramp, so a figure dropped beside the site does not look borrowed.
VIRIDIS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
]
INK = "#1a1a1a"
MUTED = "#6a6a6a"
RULE = "#d8d8d8"
PAPER = "#ffffff"

WIDTH = 900
BOX_W = 470
BOX_X = 210
GAP = 26
PAD_TOP = 96
FONT = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
MONO = "'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', monospace"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


@dataclass
class Stage:
    """One box in the flow."""

    title: str
    detail: str
    module: str = ""
    tone: int = 3                 # index into VIRIDIS
    note: str = ""                # right-hand annotation, the data leaving here
    verdict: str = ""             # a result worth stating at this stage

    def height(self) -> int:
        lines = len(textwrap.wrap(self.detail, 62)) or 1
        return 34 + lines * 15 + (17 if self.module else 0) + 12


@dataclass
class Diagram:
    """A methodology, end to end."""

    key: str
    title: str
    subtitle: str
    stages: list[Stage] = field(default_factory=list)
    outcome: str = ""


def _text(x: float, y: float, s: str, *, size=13, fill=INK, weight="normal",
          family=FONT, anchor="start") -> str:
    return (
        f'<text x="{x:.0f}" y="{y:.0f}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{_esc(s)}</text>'
    )


def render(diagram: Diagram) -> str:
    """One diagram as a standalone SVG string."""
    body: list[str] = []
    y = PAD_TOP

    for index, stage in enumerate(diagram.stages, start=1):
        h = stage.height()
        colour = VIRIDIS[stage.tone]

        # Stage number in the left gutter.
        body.append(_text(BOX_X - 26, y + 26, f"{index:02d}", size=15,
                          fill=MUTED, family=MONO, anchor="end"))

        body.append(
            f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{h}" rx="7" '
            f'fill="{PAPER}" stroke="{RULE}" stroke-width="1"/>'
        )
        # A colour bar carrying the stage's role.
        body.append(
            f'<rect x="{BOX_X}" y="{y}" width="4" height="{h}" rx="2" fill="{colour}"/>'
        )

        ty = y + 24
        body.append(_text(BOX_X + 18, ty, stage.title, size=14, weight="600"))
        ty += 18
        for line in textwrap.wrap(stage.detail, 62) or [""]:
            body.append(_text(BOX_X + 18, ty, line, size=12, fill=MUTED))
            ty += 15
        if stage.module:
            body.append(_text(BOX_X + 18, ty + 3, stage.module, size=11,
                              fill=colour, family=MONO))

        # Right gutter: what leaves this stage, and any result worth stating.
        if stage.note:
            ny = y + 22
            for line in textwrap.wrap(stage.note, 24):
                body.append(_text(BOX_X + BOX_W + 16, ny, line, size=11,
                                  fill=MUTED, family=MONO))
                ny += 14
        if stage.verdict:
            vy = y + h - 10
            for line in reversed(textwrap.wrap(stage.verdict, 24)):
                body.append(_text(BOX_X + BOX_W + 16, vy, line, size=11,
                                  fill=VIRIDIS[0], weight="600"))
                vy -= 14

        y += h
        if index < len(diagram.stages):
            mid = BOX_X + BOX_W / 2
            body.append(
                f'<path d="M{mid:.0f} {y} L{mid:.0f} {y + GAP - 7}" '
                f'stroke="{RULE}" stroke-width="1.5"/>'
                f'<path d="M{mid - 4:.0f} {y + GAP - 11} L{mid:.0f} {y + GAP - 4} '
                f'L{mid + 4:.0f} {y + GAP - 11}" fill="none" stroke="{RULE}" '
                f'stroke-width="1.5" stroke-linecap="round"/>'
            )
            y += GAP

    if diagram.outcome:
        y += 30
        body.append(f'<line x1="{BOX_X}" y1="{y - 16}" x2="{BOX_X + BOX_W}" '
                    f'y2="{y - 16}" stroke="{RULE}"/>')
        for line in textwrap.wrap(diagram.outcome, 72):
            body.append(_text(BOX_X, y, line, size=12.5, fill=INK))
            y += 17

    height = y + 34
    head = [
        f'<rect width="{WIDTH}" height="{height}" fill="{PAPER}"/>',
        _text(BOX_X - 26, 44, diagram.title, size=21, weight="700"),
        _text(BOX_X - 26, 66, diagram.subtitle, size=12.5, fill=MUTED),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height}" viewBox="0 0 {WIDTH} {height}" '
        f'role="img" aria-label="{_esc(diagram.title)}">'
        + "".join(head) + "".join(body) + "</svg>"
    )


# ---------------------------------------------------------------------------
# The methodologies. Every number here is read from config or from a measured
# result, so the figures cannot quietly disagree with the pipeline.
# ---------------------------------------------------------------------------

def _acquisition() -> Stage:
    return Stage(
        "Image acquisition",
        f"Two Kinect v2 units carried together through {N_SWEEP_STEPS} positions "
        f"{ANGULAR_STEP_DEG} degrees apart, giving 12 azimuths. Colour is mapped "
        f"into the depth frame, so both are {KINECT_V2.width} by "
        f"{KINECT_V2.height} and pixel aligned.",
        "dataset/plants/<id>/",
        tone=0,
        note=f"12 x {KINECT_V2.width}x{KINECT_V2.height} RGB-D",
    )


def _registration() -> Stage:
    return Stage(
        "Registration, without calibration",
        "No ChArUco sequence was ever captured, so every pose is estimated from "
        "the depth itself. A RANSAC floor plane per view fixes tilt, roll and "
        "height; the subject axis fixes the origin; azimuth is refined by "
        "coordinate descent.",
        "ggssvt/geometry/rig.py",
        tone=1,
        note="12 poses, cam-to-world",
        verdict="saturates +-8 deg on 25 of 30",
    )


def _segmentation(sam: bool = False) -> Stage:
    if sam:
        return Stage(
            "Segmentation, SAM prompted",
            "SAM ViT-B prompted from the registered geometry, gated to the "
            "working cylinder, with masks reverted where they disagree with the "
            "rest in 3D. Tighter than the geometric mask and less complete.",
            "ggssvt/geometry/sam3d.py",
            tone=2,
            note="12 subject masks",
            verdict="33 of 38 usable",
        )
    return Stage(
        "Segmentation, geometric",
        f"Excess green (2G-R-B) above {EXCESS_GREEN_THRESHOLD} inside a cylinder "
        f"of radius {ROI_RADIUS_M} m about the plant axis, then a statistical "
        f"outlier removal pass and a multi-view consistency check.",
        "ggssvt/geometry/segment.py",
        tone=2,
        note="12 subject masks",
        verdict="36 of 38 usable",
    )


def _pot() -> Stage:
    return Stage(
        "Pot rim, per specimen",
        "The rim is a step in the vertical cross-section, not a slope, so it is "
        "found by comparing medians above and below each candidate height. A "
        "smooth taper has no rim and the estimator says so rather than guessing.",
        "ggssvt/geometry/pot.py",
        tone=6,
        note="rim height, metres",
        verdict="refuses on 9 of E001-E010",
    )


def _features(fused: bool = False) -> Stage:
    return Stage(
        "Shape descriptors",
        "Above-rim volume and its two-thirds power, height, mean and maximum "
        "spread, compactness against the swept cylinder, and floor footprint. "
        "Seven numbers, hand-designed.",
        "ggssvt/eval/baselines.py",
        tone=7,
        note="7-vector per specimen",
    )


def _regression(rmse: str, r2: str, extra: str = "") -> Stage:
    return Stage(
        "Biomass regression",
        "Ridge on standardised descriptors, leave-one-out over every specimen, "
        "with a paired bootstrap on any difference. " + extra,
        "ggssvt/eval/baselines.py",
        tone=8,
        note="mass, kg",
        verdict=f"RMSE {rmse}, R2 {r2}",
    )


def carve() -> Diagram:
    return Diagram(
        "carve",
        "Method A. Space carving",
        "Silhouette intersection. The reference pipeline, and the one the "
        "evidence argues against.",
        [
            _acquisition(),
            _registration(),
            _segmentation(),
            Stage(
                "Space carving",
                f"Silhouette and depth carving into a {VOXEL_RESOLUTION} cubed "
                f"grid at {VOXEL_SIZE_M * 1000:.0f} mm, spanning "
                f"{VOXEL_RESOLUTION * VOXEL_SIZE_M:.3f} m. A voxel survives "
                f"unless enough views vote it out.",
                "ggssvt/geometry/carving.py",
                tone=4,
                note=f"{VOXEL_RESOLUTION}^3 occupancy",
                verdict="the visual hull",
            ),
            _pot(),
            _features(),
            _regression("0.544 kg", "+0.030"),
        ],
        outcome=(
            "The visual hull is the maximal solid consistent with the "
            "silhouettes, so a pot rim and the gap between two leaves are both "
            "filled at any resolution. Only 8 of 36 reconstructions imply a bulk "
            "density inside 200 to 1000 kg per cubic metre. What is measured is "
            "the canopy envelope, not the plant."
        ),
    )


def sam3d() -> Diagram:
    d = carve()
    d.key = "sam3d"
    d.title = "Method B. Space carving, SAM3D masks"
    d.subtitle = "Method A with the segmentation stage replaced."
    d.stages[2] = _segmentation(sam=True)
    d.outcome = (
        "Tighter masks raise multi-view agreement by 0.020 and cost 0.065 of "
        "surface coverage, dropping three specimens the geometric gate keeps. "
        "The hull's ceiling is unchanged, because the mask is not what limits it."
    )
    return d


def fusion() -> Diagram:
    return Diagram(
        "fusion",
        "Method C. TSDF depth fusion",
        "The same depth maps integrated rather than intersected. The change that "
        "resolved.",
        [
            _acquisition(),
            _registration(),
            _segmentation(),
            Stage(
                "TSDF fusion",
                f"Each depth pixel is a surface at a known distance, not a ray "
                f"through an unknown one. Signed distances are accumulated with a "
                f"{TRUNCATION_VOXELS:.0f}-voxel truncation band, at "
                f"{FUSION_RESOLUTION} cubed and "
                f"{FUSION_VOXEL_M * 1000:.0f} mm, which is two depth samples "
                f"across at the working distance.",
                "ggssvt/geometry/fusion.py",
                tone=5,
                note="signed distance field",
                verdict="concavities survive",
            ),
            Stage(
                "Observed interior",
                "Voxels behind an observed surface and inside the truncation. "
                "Space no camera measured stays unknown rather than being "
                "assumed solid, so coverage is reported alongside the volume.",
                "FusionResult.interior",
                tone=6,
                note="occupancy, holes kept",
                verdict="mean coverage 0.12",
            ),
            _pot(),
            _features(fused=True),
            _regression(
                "0.335 kg", "+0.632",
                "Identical features and protocol to Method A.",
            ),
        ],
        outcome=(
            "25 of 36 plausible against the carve's 8 at this same 12 mm grid, "
            "and 31 of 36 at the 6 mm the sensor supports. Holding the rim fixed "
            "at the carve's estimate isolates the occupancy operator alone and "
            "gives 21 of 36, so the rim estimate improves on a fused profile too. "
            "Against Method A the paired "
            "bootstrap is -0.209 kg, 95 percent interval -0.363 to -0.066: the "
            "first resolved biomass improvement in the project. Direct 2D is "
            "unchanged at 0.469, which is the control."
        ),
    )


def posefree() -> Diagram:
    return Diagram(
        "posefree",
        "Method D. Pose-free reconstruction, feed-forward pointmap models",
        "DUSt3R, MASt3R and Fast3R. Cameras and geometry from images alone, so "
        "no failure mode is shared with the carve.",
        [
            _acquisition(),
            Stage(
                "Feed-forward reconstruction",
                "No registration stage. DUSt3R and MASt3R regress pointmaps for "
                "image pairs, 132 forward passes per specimen at complete "
                "symmetric pairing, then a global alignment. Fast3R ingests all "
                "twelve views in one pass.",
                "ggssvt/geometry/pose_free_backends.py",
                tone=3,
                note="poses + point cloud",
                verdict="installed, not yet run",
            ),
            Stage(
                "Scale recovery",
                "DUSt3R and Fast3R return geometry up to an unknown global "
                "scale, so a volume computed from them would measure the "
                "rescaling. Both are aligned to the measured depth first. "
                "MASt3R's metric checkpoint needs no such step.",
                "recover_scale_from_depth",
                tone=5,
                note="metric point cloud",
            ),
            Stage(
                "Registration check",
                "Rotation and translation error per view against the "
                "calibration-free rig estimate, plus a convention check that the "
                "cameras end up looking at the scene rather than away from it.",
                "compare_poses, sanity_check_result",
                tone=6,
                note="per-view pose error",
                verdict="the independent check",
            ),
            _pot(),
            _features(),
            _regression("pending", "pending"),
        ],
        outcome=(
            "The question these answer is not which reconstructs best. It is "
            "whether the estimated rig is trustworthy, and whether the implied "
            "densities stay low with poses that were derived from images rather "
            "than from the same depth the carve used. If they do, the envelope "
            "argument closes and is no longer answerable with 'your poses were "
            "bad'."
        ),
    )


def ggssvt() -> Diagram:
    return Diagram(
        "ggssvt",
        "Method E. GG-SSVT, the learned model",
        "Geometry-grounded self-supervised vision transformer. Carved or fused "
        "occupancy becomes the training target rather than the answer.",
        [
            _acquisition(),
            _registration(),
            _segmentation(),
            Stage(
                "Reconstruction as supervision",
                "The carve, or the fusion, supplies occupancy labels with no "
                "manual annotation. This is what makes the model "
                "self-supervised, and it also caps it: the model can only learn "
                "what the reconstruction contains.",
                "ggssvt/training/dataset.py",
                tone=4,
                note="occupancy targets",
            ),
            Stage(
                "View encoder",
                "Per-view tokens from a CNN stem or a frozen DINOv2 backbone, "
                "each anchored to the world point its patch back-projects to, "
                "and positioned by a Fourier ladder over that coordinate.",
                "models/encoder.py, embedding.py",
                tone=5,
                note="tokens + 3D anchors",
            ),
            Stage(
                "Cross-view geometric fusion",
                "Attention across all twelve views with a bias on the 3D "
                "distance between token anchors, so tokens describing the same "
                "piece of plant attend to each other regardless of which camera "
                "saw them.",
                "ggssvt/models/attention.py",
                tone=6,
                note="fused tokens",
            ),
            Stage(
                "Occupancy decoder",
                "Cross-attention from query points to the fused tokens, "
                "evaluated in chunks so an arbitrary number of queries fits in "
                "memory. Trained against the carved occupancy.",
                "ggssvt/models/decoder.py",
                tone=7,
                note="occupancy at any point",
            ),
            Stage(
                "Biomass head",
                "Volume by integrating occupancy above the rim, multiplied by a "
                "density predicted from pooled tokens and shape descriptors, "
                "plus a residual. Density is learned rather than assumed.",
                "ggssvt/models/head.py",
                tone=8,
                note="mass, kg",
                verdict="never trained",
            ),
        ],
        outcome=(
            "Implemented, tested and never run: the campaign is seven runs and "
            "eight to ten hours on one GPU. The comparison that matters is "
            "baseline_cnn against baseline_fused, which asks whether a trained "
            "model inherits the 0.209 kg the classical features gained when the "
            "reconstruction operator changed."
        ),
    )


def backbones() -> Diagram:
    """The frozen-feature arm, which is the one that is finished."""
    return Diagram(
        "backbones",
        "Method F. Frozen self-supervised features",
        "No training. Patch tokens from a frozen backbone, pooled and ridged "
        "against mass, which is what makes the comparison cheap enough to run "
        "properly.",
        [
            _acquisition(),
            _registration(),
            _segmentation(),
            Stage(
                "Patch tokens per view",
                "Every view through a frozen DINOv2 or DINOv3 encoder. Nothing "
                "is finetuned, so the comparison is of the representations "
                "themselves rather than of how well each one trains.",
                "models/backbones.py",
                tone=4,
                note="(V, gh, gw, D) tokens",
            ),
            Stage(
                "Pool, reduce, ridge",
                "Tokens pooled across views, projected to a handful of "
                "principal components, and regressed on mass with leave-one-out "
                "cross-validation. The no-DINO control uses the seven geometric "
                "descriptors alone.",
                "eval/dino_probe.py",
                tone=6,
                note="mass, kg",
                verdict="DINOv2 0.392 kg, DINOv3 0.394 kg, control 0.458 kg",
            ),
            Stage(
                "Lift onto points, cluster",
                "The same tokens back-projected onto the carved points and "
                "clustered in two, which asks whether the features separate "
                "plant from pot where colour does not.",
                "geometry/dino_lift.py, eval/dino_segment.py",
                tone=7,
                note="a label per point",
                verdict="ties where the rim is confident, DINOv3 ahead where it is not",
            ),
        ],
        outcome=(
            "Finished, and the answer is a null. Paired against each other the "
            "two backbones differ by 0.0018 kg with a 95 percent interval of "
            "-0.025 to +0.030, on predictions correlating at 0.977. The "
            "resolution ledger puts the smallest detectable effect at n=36 at "
            "0.138 kg, so no amount of finetuning could settle a difference this "
            "size on this dataset. The one place they part company is the "
            "eleven captures where the rim detector refuses: DINOv3 is better on "
            "all eleven, and still fails to separate plant from pot there."
        ),
    )


def generative() -> Diagram:
    """The proposed baseline. Scoped, deliberately not implemented."""
    return Diagram(
        "generative",
        "Method G. Single-image generative reconstruction (proposed)",
        "SAM 3D Objects as a control, not as an instrument: what would a "
        "model that never saw the back of the plant report?",
        [
            Stage(
                "One view, one mask",
                "A single frame and its subject mask, which is all this model "
                "takes. Eleven of the twelve views are discarded on purpose: "
                "the point is to measure what one view is worth.",
                "eval/generative.py (proposed)",
                tone=2,
                note="one RGB frame",
            ),
            Stage(
                "Structured latent generation",
                "facebook/sam-3d-objects. A sparse structure generator and a "
                "latent generator, about 12.5 GB of checkpoints, producing a "
                "mesh or a splat. The geometry it returns for the unseen half "
                "of the plant is *invented*, which is the whole reason this arm "
                "exists and the whole reason it must stay a control.",
                "12.5 GB, gated, needs Meta's own package",
                tone=5,
                note="a generated mesh",
                verdict="does not fit 4 GiB without offloading",
            ),
            Stage(
                "Scale and measure",
                "The generated mesh has no metric scale, so it is fitted to the "
                "measured silhouette before any volume is read. Without this the "
                "comparison is of shape only and the volumes are arbitrary.",
                "eval/generative.py (proposed)",
                tone=6,
                note="volume, litres",
            ),
            Stage(
                "Score against the same screen",
                "The implied bulk density screen and the leave-one-batch-out "
                "regression, unchanged, so the number lands in the same ledger "
                "as every other operator.",
                "eval/resolution.py",
                tone=8,
                note="admitted or rejected",
                verdict="never enters a reported volume",
            ),
        ],
        outcome=(
            "The hypothesis is stated before the run, because it is the kind of "
            "result that is easy to rationalise afterwards. If a single-image "
            "generative model predicts mass as well as the twelve-view carve, "
            "that is evidence for the batch confound: it would mean the task is "
            "being solved from apparent size rather than from measured geometry, "
            "and that the rig is not earning its twelve views. If it does worse, "
            "the multi-view capture is justified on its own terms. Either way "
            "the volume it produces is generated rather than measured, so it "
            "never feeds a reported figure and never enters the density screen "
            "as evidence about a specimen."
        ),
    )


DIAGRAMS = (carve, sam3d, fusion, posefree, ggssvt, backbones, generative)


def manifest() -> list[dict]:
    """Key, title and subtitle for every diagram, for the page to list.

    Generated from the diagram factories rather than repeated as markup, so a
    methodology cannot appear in the picker and be missing from the figure, or
    the other way round.
    """
    entries = []
    for factory in DIAGRAMS:
        diagram = factory()
        entries.append({
            "key": diagram.key,
            "title": diagram.title,
            "subtitle": diagram.subtitle,
            "outcome": diagram.outcome,
            "n_stages": len(diagram.stages),
            "image": f"./static/architecture/architecture_{diagram.key}.svg",
        })
    return entries


def write_all(out_dir: Path) -> list[Path]:
    """Render every methodology to its own SVG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for factory in DIAGRAMS:
        diagram = factory()
        path = out_dir / f"architecture_{diagram.key}.svg"
        path.write_text(render(diagram), encoding="utf-8")
        written.append(path)
    return written




# ---------------------------------------------------------------------------
# Raster backend.
#
# The SVG is the primary artefact: sharp at any size, selectable text, a few
# kilobytes. Word will not embed one without a raster fallback, though, and
# neither will most journal submission systems, so the same Stage and Diagram
# objects are drawn a second way here rather than rasterised through a
# dependency that wants a C library Windows does not ship.
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = {
    "regular": ("segoeui.ttf", "calibri.ttf", "arial.ttf", "DejaVuSans.ttf"),
    "bold": ("segoeuib.ttf", "calibrib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
    "mono": ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"),
}
_FONT_DIRS = ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu",
              "/System/Library/Fonts")


def _font(kind: str, size: int):
    """A usable face, or PIL's default if the system has none of them."""
    from PIL import ImageFont

    for directory in _FONT_DIRS:
        for name in _FONT_CANDIDATES[kind]:
            path = Path(directory) / name
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except OSError:
                    continue
    return ImageFont.load_default()


def render_png(diagram: Diagram, scale: int = 2):
    """The same diagram as a PIL image, at ``scale`` times the SVG geometry."""
    from PIL import Image, ImageDraw

    s = scale
    fonts = {
        "title": _font("bold", 21 * s), "sub": _font("regular", 13 * s),
        "head": _font("bold", 14 * s), "body": _font("regular", 12 * s),
        "mono": _font("mono", 11 * s), "num": _font("mono", 15 * s),
        "small": _font("mono", 11 * s), "out": _font("regular", 13 * s),
    }

    # Height has to match the SVG's, so the two artefacts stay the same figure.
    height = 0
    for stage in diagram.stages:
        height += stage.height() + GAP
    height = PAD_TOP + height - GAP
    if diagram.outcome:
        height += 30 + 17 * len(textwrap.wrap(diagram.outcome, 72))
    height += 34

    image = Image.new("RGB", (WIDTH * s, height * s), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text(((BOX_X - 26) * s, 28 * s), diagram.title, font=fonts["title"], fill=INK)
    draw.text(((BOX_X - 26) * s, 56 * s), diagram.subtitle, font=fonts["sub"], fill=MUTED)

    y = PAD_TOP
    for index, stage in enumerate(diagram.stages, start=1):
        h = stage.height()
        colour = VIRIDIS[stage.tone]

        draw.text(((BOX_X - 40) * s, (y + 14) * s), f"{index:02d}",
                  font=fonts["num"], fill=MUTED)
        draw.rounded_rectangle(
            [BOX_X * s, y * s, (BOX_X + BOX_W) * s, (y + h) * s],
            radius=7 * s, fill=PAPER, outline=RULE, width=max(1, s // 2),
        )
        draw.rounded_rectangle(
            [BOX_X * s, y * s, (BOX_X + 4) * s, (y + h) * s],
            radius=2 * s, fill=colour,
        )

        ty = y + 13
        draw.text(((BOX_X + 18) * s, ty * s), stage.title, font=fonts["head"], fill=INK)
        ty += 18
        for line in textwrap.wrap(stage.detail, 62) or [""]:
            draw.text(((BOX_X + 18) * s, ty * s), line, font=fonts["body"], fill=MUTED)
            ty += 15
        if stage.module:
            draw.text(((BOX_X + 18) * s, (ty + 2) * s), stage.module,
                      font=fonts["mono"], fill=colour)

        if stage.note:
            ny = y + 12
            for line in textwrap.wrap(stage.note, 24):
                draw.text(((BOX_X + BOX_W + 16) * s, ny * s), line,
                          font=fonts["small"], fill=MUTED)
                ny += 14
        if stage.verdict:
            lines = textwrap.wrap(stage.verdict, 24)
            vy = y + h - 12 - 14 * (len(lines) - 1)
            for line in lines:
                draw.text(((BOX_X + BOX_W + 16) * s, vy * s), line,
                          font=fonts["small"], fill=VIRIDIS[0])
                vy += 14

        y += h
        if index < len(diagram.stages):
            mid = BOX_X + BOX_W / 2
            draw.line([mid * s, y * s, mid * s, (y + GAP - 7) * s],
                      fill=RULE, width=max(1, s))
            draw.line([(mid - 4) * s, (y + GAP - 11) * s, mid * s, (y + GAP - 4) * s],
                      fill=RULE, width=max(1, s))
            draw.line([(mid + 4) * s, (y + GAP - 11) * s, mid * s, (y + GAP - 4) * s],
                      fill=RULE, width=max(1, s))
            y += GAP

    if diagram.outcome:
        y += 30
        draw.line([BOX_X * s, (y - 16) * s, (BOX_X + BOX_W) * s, (y - 16) * s],
                  fill=RULE, width=max(1, s // 2))
        for line in textwrap.wrap(diagram.outcome, 72):
            draw.text((BOX_X * s, (y - 11) * s), line, font=fonts["out"], fill=INK)
            y += 17

    return image


def write_all_png(out_dir: Path, scale: int = 2) -> list[Path]:
    """Render every methodology to PNG, for Word and for submission systems."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for factory in DIAGRAMS:
        diagram = factory()
        path = out_dir / f"architecture_{diagram.key}.png"
        render_png(diagram, scale=scale).save(path, "PNG", optimize=True)
        written.append(path)
    return written


__all__ = [
    "DIAGRAMS",
    "Diagram",
    "Stage",
    "render",
    "render_png",
    "write_all",
    "write_all_png",
]
