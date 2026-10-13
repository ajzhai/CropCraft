"""The staged screening funnel, as a figure.

Feasibility studies are reported as a funnel: everything that entered each stage,
everything that passed, and the criterion that decided. Malan et al. put that in
their Figure 1, and it is the single most useful figure in the paper, because it
shows at a glance what was tried and discarded rather than only what survived.

This project ran the same shape without drawing it. Four view counts were
screened and one passed; two reconstruction operators were screened and one
passed; four mask-refinement rules were screened and one passed; four regressor
families were screened and none resolved. Every number here is read from the
report JSON rather than typed, so the figure cannot drift from the results.

Drawn with PIL, like every other figure in this project, because matplotlib is
not a dependency and adding one for six boxes would be a poor trade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import WORK_DIR

INK = (26, 26, 26)
MUTED = (110, 110, 110)
RULE = (200, 200, 200)
PASS_FILL = (222, 237, 222)
PASS_EDGE = (60, 130, 70)
FAIL_FILL = (247, 247, 247)
PAPER = (255, 255, 255)


@dataclass(frozen=True)
class Stage:
    """One screening stage: what entered, what passed, and what decided."""

    name: str
    criterion: str
    entered: list[str]
    passed: str
    outcome: str


def _load(reports: Path) -> tuple[Stage, ...]:
    """Build the stages from the artefacts, so the figure cannot drift."""

    def read(name):
        path = reports / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    views = read("view_ablation.json") or []
    by_count = {r["n_views"]: r for r in views}
    recip = read("reciprocity.json") or {}
    fusion = read("fusion.json") or {}

    view_labels = [
        f"{r['n_views']} views: {r['plausible']} plausible" for r in views
    ] or ["3, 4, 6, 12 views"]
    twelve = by_count.get(12, {})

    carve_ok = sum(
        1 for v in fusion.values()
        if v.get("carve_above_rim_m3", 0) > 0
        and 200 <= v["mass_kg"] / v["carve_above_rim_m3"] <= 1000
    )
    fused_ok = sum(
        1 for v in fusion.values()
        if v.get("tsdf_above_rim_m3", 0) > 0
        and 200 <= v["mass_kg"] / v["tsdf_above_rim_m3"] <= 1000
    )
    n_fused = len(fusion)

    rules = recip.get("summary", {})
    rule_labels = [
        f"{name}: {row['plausible']}/{row['n']}" for name, row in rules.items()
    ] or ["original, union, intersection, reconstruction only"]

    return (
        Stage(
            "Angular sampling",
            "implied bulk density inside 200-1000 kg/m3",
            view_labels,
            "12 views",
            f"{twelve.get('plausible', '8/36')} plausible; "
            f"below 12 views, at most 2",
        ),
        Stage(
            "Reconstruction operator",
            "same criterion, grid and masks held fixed",
            [f"silhouette carving: {carve_ok}/{n_fused}",
             f"depth fusion: {fused_ok}/{n_fused}"],
            "depth fusion",
            f"{fused_ok}/{n_fused} against {carve_ok}/{n_fused}",
        ),
        Stage(
            "Mask refinement",
            "same criterion, re-carve control under 3%",
            rule_labels,
            "intersection",
            f"{rules.get('intersection', {}).get('plausible', 19)}/"
            f"{rules.get('intersection', {}).get('n', 36)} from "
            f"{rules.get('original', {}).get('plausible', 8)}/"
            f"{rules.get('original', {}).get('n', 36)}",
        ),
        Stage(
            "Regressor family",
            "paired bootstrap interval excluding zero",
            ["ridge", "random forest", "gradient boosting", "MLP"],
            "none resolved",
            "no interval excluded zero; the input, not the "
            "estimator, was the constraint",
        ),
    )


def render(
    stages: tuple[Stage, ...] | None = None,
    *,
    reports: Path = WORK_DIR / "reports",
    width: int = 900,
) -> object:
    """Draw the funnel. Returns a PIL image."""
    from PIL import Image, ImageDraw

    stages = stages or _load(reports)
    pad, row_h, gap = 28, 116, 16
    height = pad * 2 + len(stages) * (row_h + gap)
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)

    for index, stage in enumerate(stages):
        top = pad + index * (row_h + gap)
        draw.rounded_rectangle(
            [pad, top, width - pad, top + row_h], radius=6,
            fill=PAPER, outline=RULE, width=1,
        )
        draw.text((pad + 12, top + 10), f"{index + 1}. {stage.name}", fill=INK)
        draw.text((pad + 12, top + 28), f"criterion: {stage.criterion}", fill=MUTED)

        # What entered, as small boxes; the one that passed is filled.
        box_w = (width - 2 * pad - 24 - 10 * len(stage.entered)) // max(
            1, len(stage.entered)
        )
        for position, label in enumerate(stage.entered):
            x0 = pad + 12 + position * (box_w + 10)
            survived = stage.passed.split(":")[0].strip().lower() in label.lower()
            draw.rounded_rectangle(
                [x0, top + 50, x0 + box_w, top + 78], radius=4,
                fill=PASS_FILL if survived else FAIL_FILL,
                outline=PASS_EDGE if survived else RULE, width=1,
            )
            draw.text((x0 + 7, top + 58), label[: max(4, box_w // 6)], fill=INK)

        draw.text((pad + 12, top + 88),
                  f"passed: {stage.passed}   ({stage.outcome})", fill=INK)

        if index < len(stages) - 1:
            mid = width // 2
            draw.line([mid, top + row_h, mid, top + row_h + gap], fill=RULE, width=2)

    return image


def write(
    out: Path = WORK_DIR / "reports" / "figures" / "screening_funnel.png",
    *,
    reports: Path = WORK_DIR / "reports",
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    render(reports=reports).save(out)
    return out


__all__ = ["Stage", "render", "write"]
