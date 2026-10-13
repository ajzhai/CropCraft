"""Works this project borrows from, and the stage each one informs.

A bibliography that is only a list is furniture. This one records *what* was
taken from each work and *where* it lands in the architecture, because that is
the part a reader needs and the part that is easy to lose. Two of these four
supply a method, one supplies a metric, and one supplies a visualisation.

The distinction between borrowing a method and borrowing a framing matters for
honesty. DINOv3 is used here as a backbone, unchanged. The DUSt3R evaluation
form is adopted as a metric, on data this project did not collect. CHMv2 is
neither: it is a precedent for a decision already taken independently, and it is
cited as corroboration rather than as a source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Reference:
    """One work, and its relationship to this project."""

    key: str
    authors: str
    year: int
    title: str
    venue: str
    arxiv: str
    stage: str            # which architecture stage it informs
    borrowed: str         # what is actually taken
    relation: str         # "method", "metric", "visualisation" or "precedent"

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv}"

    def as_dict(self) -> dict:
        return {**asdict(self), "url": self.url}


REFERENCES: tuple[Reference, ...] = (
    Reference(
        key="caron2021dino",
        authors="Caron, Touvron, Misra, Jégou, Mairal, Bojanowski and Joulin",
        year=2021,
        title="Emerging Properties in Self-Supervised Vision Transformers",
        venue="ICCV",
        arxiv="2104.14294",
        stage="Method F, patch tokens per view",
        borrowed="The finding that a self-supervised vision transformer's "
                 "attention segments objects with no supervision, which is the "
                 "reason to test a frozen backbone here at all, and the way "
                 "those maps are read: project the patch features and look at "
                 "the boundaries rather than at any single head.",
        relation="visualisation",
    ),
    Reference(
        key="wang2024dust3r",
        authors="Wang, Leroy, Cabon, Chidlovskii and Revaud",
        year=2024,
        title="DUSt3R: Geometric 3D Vision Made Easy",
        venue="CVPR",
        arxiv="2312.14132",
        stage="Method D, pose-free reconstruction; and the Pheno4D scoring",
        borrowed="Two things, and they are separable. The method is pointmap "
                 "regression from uncalibrated pairs, which is the family the "
                 "pose-free arm belongs to. The metric is the accuracy, "
                 "completeness and overall triple it reports for DTU, adopted "
                 "unchanged in eval/recon_metrics.py, where accuracy is the "
                 "distance from each reconstructed point to the nearest true "
                 "one and completeness is the reverse.",
        relation="method and metric",
    ),
    Reference(
        key="simeoni2025dinov3",
        authors="Siméoni, Vo, Seitzer, Baldassarre, Oquab and others",
        year=2025,
        title="DINOv3",
        venue="arXiv preprint",
        arxiv="2508.10104",
        stage="Method F, patch tokens per view",
        borrowed="The backbone itself, frozen and unmodified. Used here to ask "
                 "whether a stronger self-supervised representation moves the "
                 "biomass estimate. It does not: paired against DINOv2 the "
                 "difference is 0.0018 kg with a 95 percent interval of -0.025 "
                 "to +0.030.",
        relation="method",
    ),
    Reference(
        key="brandt2026chmv2",
        authors="Brandt, Yi, Tolan, Li, Potapov and others",
        year=2026,
        title="CHMv2: Improvements in Global Canopy Height Mapping using DINOv3",
        venue="arXiv preprint",
        arxiv="2603.06382",
        stage="Method F, pool and ridge; and the batch-holdout evaluation",
        borrowed="A precedent rather than a source, and the closest published "
                 "analogue to this task: DINOv3 features regressed onto "
                 "vegetation structure. Two of its evaluation choices match "
                 "decisions taken here independently. It reports block-R2 over "
                 "50 by 50 pixel blocks rather than a pooled R2, which is "
                 "spatial-block cross-validation and the same argument as "
                 "leave-one-batch-out. And it reports mean bias error "
                 "separately for the tall tail, which is error stratified by "
                 "the magnitude of the target, the thing this project needs "
                 "given that its models separate size classes.",
        relation="precedent",
    ),
    Reference(
        key="nombambela2025",
        authors="Nombambela",
        year=2025,
        title="Plant Mass Estimation Using 3D Modelling",
        venue="EPR402 report, University of Pretoria",
        arxiv="",
        stage="Method H, surface voxel count",
        borrowed="The operator: volume as the count of occupied voxels in a "
                 "registered surface point cloud at 7 mm, with no carving and "
                 "no signed distance field. Reimplemented from the method as "
                 "described rather than from his code, and verified against his "
                 "own plant 1 at 10,177 voxels.",
        relation="method",
    ),
)


def by_stage() -> dict[str, list[Reference]]:
    """References grouped by the architecture stage they inform."""
    grouped: dict[str, list[Reference]] = {}
    for reference in REFERENCES:
        grouped.setdefault(reference.stage, []).append(reference)
    return grouped


def bibtex() -> str:
    """The bibliography, for pasting into a thesis."""
    entries = []
    for r in REFERENCES:
        kind = "inproceedings" if r.venue in {"ICCV", "CVPR"} else "misc"
        lines = [f"@{kind}{{{r.key},",
                 f"  author = {{{r.authors}}},",
                 f"  title  = {{{r.title}}},",
                 f"  year   = {{{r.year}}},"]
        if r.venue:
            field = "booktitle" if kind == "inproceedings" else "howpublished"
            lines.append(f"  {field} = {{{r.venue}}},")
        if r.arxiv:
            lines.append(f"  eprint = {{{r.arxiv}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries)


def payload() -> dict:
    """What the page embeds."""
    return {
        "note": "each work is listed with the stage it informs and what is "
                "actually taken from it, because a bibliography that is only a "
                "list does not tell a reader where a method came from",
        "references": [r.as_dict() for r in REFERENCES],
        "bibtex": bibtex(),
    }


__all__ = ["REFERENCES", "Reference", "bibtex", "by_stage", "payload"]
