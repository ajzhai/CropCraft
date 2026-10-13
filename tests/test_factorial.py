"""The SAM3D x DINO factorial, and the SAM3D consistency rules."""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.factorial import Cell, FactorialReport, condition_label
from ggssvt.eval.metrics import regression_metrics
from ggssvt.geometry.sam3d import Sam3DSegmenter, Sam3DStats, sam_is_available


def _cell(segmenter, backbone, predictions, targets, variant="base"):
    label = condition_label(segmenter, backbone, variant if backbone != "cnn" else "")
    return label, Cell(
        segmenter=segmenter,
        backbone=backbone,
        variant=variant if backbone != "cnn" else "",
        label=label,
        metrics=regression_metrics(predictions, targets),
        predictions=predictions,
    )


def _report(errors: dict[tuple[bool, bool], float], seed: int = 0) -> FactorialReport:
    """Build a synthetic 2x2 factorial with prescribed noise levels."""
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.4, 2.4, 40)

    cells = {}
    for (has_sam, has_dino), sigma in errors.items():
        segmenter = "sam3d" if has_sam else "geometric"
        backbone = "dinov2" if has_dino else "cnn"
        predictions = targets + rng.normal(0, sigma, targets.size)
        label, cell = _cell(segmenter, backbone, predictions, targets)
        cells[label] = cell

    return FactorialReport(cells=cells, targets=targets, n_specimens=targets.size)


def test_condition_labels_name_all_four_corners():
    assert condition_label("geometric", "cnn") == "no SAM3D + no DINO"
    assert condition_label("sam3d", "cnn") == "SAM3D + no DINO"
    assert condition_label("geometric", "dinov2", "base") == "no SAM3D + dinov2-base"
    assert condition_label("sam3d", "dinov3", "base") == "SAM3D + dinov3-base"


def test_effects_are_computed_for_every_corner_present():
    report = _report({(False, False): 0.5, (False, True): 0.3, (True, False): 0.35, (True, True): 0.2})
    effects = report.effects()

    for name in (
        "DINO alone (vs neither)",
        "SAM3D alone (vs neither)",
        "both (vs neither)",
        "DINO given SAM3D",
        "SAM3D given DINO",
        "interaction",
    ):
        assert name in effects, name


def test_effects_degrade_gracefully_with_a_missing_corner():
    """A gated backbone leaves holes; the harness must not invent them."""
    report = _report({(False, False): 0.5, (False, True): 0.3})
    effects = report.effects()

    assert "DINO alone (vs neither)" in effects
    assert "interaction" not in effects
    assert "SAM3D alone (vs neither)" not in effects


def test_additive_improvements_show_no_interaction():
    """Two independent improvements should give an interaction near zero."""
    rng = np.random.default_rng(3)
    targets = rng.uniform(0.4, 2.4, 60)
    shared = rng.normal(0, 1.0, targets.size)

    def predict(sam_gain: float, dino_gain: float) -> np.ndarray:
        sigma = 0.5 - sam_gain - dino_gain
        return targets + shared * sigma

    cells = {}
    for has_sam, has_dino, sam_gain, dino_gain in (
        (False, False, 0.0, 0.0),
        (False, True, 0.0, 0.1),
        (True, False, 0.1, 0.0),
        (True, True, 0.1, 0.1),
    ):
        label, cell = _cell(
            "sam3d" if has_sam else "geometric",
            "dinov2" if has_dino else "cnn",
            predict(sam_gain, dino_gain),
            targets,
        )
        cells[label] = cell

    report = FactorialReport(cells=cells, targets=targets, n_specimens=targets.size)
    interaction = report.effects()["interaction"]
    assert abs(interaction["difference"]) < 0.02


def test_table_renders_skipped_cells_and_never_crashes():
    report = _report({(False, False): 0.4, (False, True): 0.3})
    report.cells["SAM3D + dinov3-base"] = Cell(
        segmenter="sam3d",
        backbone="dinov3",
        variant="base",
        label="SAM3D + dinov3-base",
        skipped_reason="gated on HuggingFace",
    )

    table = report.to_table()
    assert "skipped" in table
    assert "gated" in table

    import json

    payload = json.loads(report.to_json())
    assert payload["cells"]["SAM3D + dinov3-base"]["skipped"] == "gated on HuggingFace"


def test_sam_availability_check_reports_a_reason_when_unavailable():
    available, reason = sam_is_available("base")
    if not available:
        assert reason


def test_sam_prompt_is_derived_from_the_geometric_mask():
    from ggssvt.config import KINECT_V2
    from ggssvt.geometry.rig import nominal_view_pose

    mask = np.zeros((424, 512), dtype=bool)
    mask[100:300, 200:320] = True

    box, points = Sam3DSegmenter._prompt_from_geometry(
        nominal_view_pose(0), KINECT_V2, mask
    )

    assert box is not None
    assert box[0] < 200 and box[1] < 100          # padded outward
    assert box[2] > 319 and box[3] > 299
    assert all(0 <= u < 512 and 0 <= v < 424 for u, v in points)


def test_sam_prompt_is_declined_when_the_geometric_mask_is_empty():
    from ggssvt.config import KINECT_V2
    from ggssvt.geometry.rig import nominal_view_pose

    box, points = Sam3DSegmenter._prompt_from_geometry(
        nominal_view_pose(0), KINECT_V2, np.zeros((424, 512), dtype=bool)
    )
    assert box is None
    assert points == []


def test_disagreeing_sam_masks_are_reverted_to_geometry():
    """The rule that makes SAM3D three-dimensional rather than per-frame."""
    from ggssvt.geometry.segment import ViewSegmentation

    def make(points: np.ndarray) -> ViewSegmentation:
        return ViewSegmentation(
            position_id="x",
            mask=np.zeros((4, 4), dtype=bool),
            depth_m=np.zeros((4, 4), dtype=np.float32),
            points_world=points.astype(np.float32),
            colours=None,
        )

    agreeing = np.random.default_rng(0).uniform(0, 0.1, size=(200, 3))
    rogue = np.random.default_rng(1).uniform(5.0, 5.1, size=(200, 3))

    refined = {"a": make(agreeing), "b": make(agreeing), "c": make(rogue)}
    geometric = {k: make(agreeing) for k in refined}
    stats = Sam3DStats(n_views=3, n_accepted=3)

    Sam3DSegmenter._revert_disagreeing_views(
        refined, geometric, stats, voxel_m=0.03, min_agreement=0.25, verbose=False
    )

    assert stats.n_rejected_agreement == 1
    assert refined["c"] is geometric["c"]
    assert refined["a"] is not geometric["a"]


def test_sam_stats_report_what_changed():
    stats = Sam3DStats(
        n_views=12, n_accepted=9, mean_pixels_before=1000.0, mean_pixels_after=800.0
    )
    assert stats.acceptance_rate == pytest.approx(0.75)
    assert stats.pixel_change == pytest.approx(0.2)
    assert stats.as_dict()["n_views"] == 12


def test_access_check_asks_about_permission_not_repo_policy():
    """A gated repo stays flagged gated forever, including once you have access.

    `model_info(...).gated` describes the repository's policy, not the caller's
    permission, so gating a backbone on it means the backbone stays skipped even
    after approval arrives -- silent, and very hard to diagnose. The check must
    use `auth_check`, which answers the question actually being asked.
    """
    from ggssvt.models.backbones import repo_access

    accessible, reason = repo_access("facebook/dinov2-base")
    assert accessible, f"an open repo must read as accessible: {reason}"
    assert reason == ""


def test_access_check_reports_a_missing_repo_distinctly():
    from ggssvt.models.backbones import repo_access

    accessible, reason = repo_access("facebook/definitely-not-a-real-model-xyz")
    assert not accessible
    assert "does not exist" in reason or "could not be checked" in reason


def test_dinov3_help_names_the_variant_that_was_refused():
    """Point people at the size they asked for, not a different one."""
    from ggssvt.models.backbones import DINOV3_REPOS, dinov3_access_help

    for variant in ("small", "base", "large"):
        help_text = dinov3_access_help(variant)
        assert DINOV3_REPOS[variant] in help_text
        for other in set(DINOV3_REPOS) - {variant}:
            assert DINOV3_REPOS[other] not in help_text


def test_dinov3_help_covers_the_two_ways_approval_looks_granted_but_is_not():
    """Both traps that make an approved account still read as gated."""
    from ggssvt.models.backbones import dinov3_access_help

    help_text = dinov3_access_help("base")
    # PENDING on the settings page is not ACCEPTED.
    assert "PENDING" in help_text and "gated-repos" in help_text
    # The machine can be logged in as a different account than the approved one.
    assert "whoami" in help_text
