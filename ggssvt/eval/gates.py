"""Acceptance checks for the output of each step.

There are gates in this project already, but they are uneven. Preprocessing has
one, `SpecimenQuality.is_usable`, deliberately loose so it catches a registration
that failed outright rather than quietly dropping specimens that merely
reconstruct poorly. Reconstruction has the implied-density check. Training has
nothing: the campaign records whatever a run produced and marks it done, so a
model that collapsed to predicting the mean, or whose loss never moved, finishes
an overnight run looking exactly like one that worked.

This module is the missing half. Every check states a threshold, reports the
value it measured against it, and says what to do when it fails. Checks are
separated by severity because the two behave differently:

**Blocking.** The output is not usable and whatever consumes it will produce
nonsense. An empty occupancy field, a training run that never beat the mean
predictor, predictions that are all the same number.

**Advisory.** The output is usable but something is worth knowing. A specimen
whose implied density is implausible is still a legitimate input to a comparison
of methods, and dropping it would bias the comparison toward the specimens that
happen to reconstruct well. That is the distinction `is_usable` was careful
about, and it is preserved here.

The point of a threshold that is written down is that it can be argued with.
Every constant below carries the reasoning that set it, so a future reader can
disagree with the number rather than guessing what it was protecting against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds, with the reasoning that set them.
# ---------------------------------------------------------------------------

# A subject mask covering more than this share of the frame has stopped being a
# subject mask. The plant fills perhaps a fifth of a Kinect frame at the working
# distance; half means the segmenter took the background with it.
MAX_MASK_FRACTION = 0.50

# Below this it has lost the plant. E001-E010 are the smallest subjects here and
# still cover more than 1 per cent.
MIN_MASK_FRACTION = 0.002

# Views whose mask is more than this many times smaller than the median view are
# treated as collapsed. Genuine foreshortening between a front and a side view of
# a plant is well within a factor of four.
MASK_COLLAPSE_RATIO = 4.0

# Bulk density outside this range is not a poor reconstruction, it is arithmetic
# on something that is not a plant. Two orders either side of tissue.
SANITY_DENSITY_MIN = 1.0
SANITY_DENSITY_MAX = 100_000.0

# A model whose predictions vary less than this fraction of the target spread has
# collapsed onto the mean, which scores respectably on RMSE while learning
# nothing. Worth blocking, because it is invisible in a metrics table.
MIN_PREDICTION_SPREAD = 0.10

# Loss must fall by at least this fraction between the first and last epoch.
# Anything less is a run that did not train, whatever its final number.
MIN_LOSS_DROP = 0.05


@dataclass(frozen=True)
class Check:
    """One acceptance check and what it found."""

    name: str
    stage: str                  # segmentation | reconstruction | training | regression
    passed: bool
    blocking: bool
    value: float | None = None
    threshold: float | None = None
    message: str = ""

    @property
    def failed_blocking(self) -> bool:
        return self.blocking and not self.passed

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "stage": self.stage,
            "passed": self.passed,
            "blocking": self.blocking,
            "value": None if self.value is None else round(float(self.value), 4),
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class GateReport:
    """Every check for one subject, and the verdict."""

    subject: str
    checks: list[Check] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(c.failed_blocking for c in self.checks)

    @property
    def advisories(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.blocking]

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "blocked": self.blocked,
            "n_checks": len(self.checks),
            "failures": [c.as_dict() for c in self.checks if not c.passed],
        }


def _check(name, stage, passed, *, blocking, value=None, threshold=None, message=""):
    return Check(name, stage, bool(passed), blocking, value, threshold, message)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def check_segmentation(cached) -> GateReport:
    """Is the subject mask a subject mask?

    Catches the two ways segmentation fails silently: taking the background with
    the plant, and losing the plant in some views while keeping it in others. The
    second is the dangerous one, because a carve with one collapsed view still
    produces a volume, just a wrong one.
    """
    report = GateReport(cached.plant_id)
    mask = cached.mask
    per_view = mask.reshape(mask.shape[0], -1).mean(axis=1)

    report.checks.append(_check(
        "mask not empty in any view", "segmentation",
        per_view.min() >= MIN_MASK_FRACTION, blocking=True,
        value=float(per_view.min()), threshold=MIN_MASK_FRACTION,
        message="a view with no subject carves away everything the others found",
    ))
    report.checks.append(_check(
        "mask has not taken the background", "segmentation",
        per_view.max() <= MAX_MASK_FRACTION, blocking=True,
        value=float(per_view.max()), threshold=MAX_MASK_FRACTION,
        message="over half the frame is not a potted plant at this distance",
    ))

    median = float(np.median(per_view))
    ratio = median / max(per_view.min(), 1e-9)
    report.checks.append(_check(
        "no view collapsed relative to the rest", "segmentation",
        ratio <= MASK_COLLAPSE_RATIO, blocking=False,
        value=float(ratio), threshold=MASK_COLLAPSE_RATIO,
        message="foreshortening does not explain a factor this large; inspect the overlay",
    ))
    return report


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def check_reconstruction(cached, *, mass_kg: float | None = None) -> GateReport:
    """Is the occupancy field a reconstruction of something?"""
    from ..config import voxel_grid_centres

    report = GateReport(cached.plant_id)
    occupancy = cached.occupancy
    n_occupied = int(occupancy.sum())

    report.checks.append(_check(
        "occupancy is not empty", "reconstruction",
        n_occupied > 0, blocking=True, value=n_occupied, threshold=1,
        message="nothing survived the carve; check the masks and the rig",
    ))
    if n_occupied == 0:
        return report

    heights = voxel_grid_centres()[..., 2]
    above = occupancy & (heights > cached.pot_height_m)
    volume = float(above.sum()) * cached.voxel_size_m ** 3

    report.checks.append(_check(
        "something sits above the pot rim", "reconstruction",
        volume > 0.0, blocking=True, value=volume, threshold=0.0,
        message="the whole reconstruction is below the rim, so there is no shoot to weigh",
    ))

    top = float(heights[occupancy].max())
    extent = occupancy.shape[2] * cached.voxel_size_m
    report.checks.append(_check(
        "reconstruction fits inside the working volume", "reconstruction",
        top < extent - cached.voxel_size_m, blocking=False,
        value=top, threshold=extent,
        message="the subject reaches the grid ceiling, so it is probably clipped",
    ))

    if mass_kg is not None and volume > 0:
        density = mass_kg / volume
        report.checks.append(_check(
            "implied density is arithmetically sane", "reconstruction",
            SANITY_DENSITY_MIN <= density <= SANITY_DENSITY_MAX, blocking=True,
            value=density, threshold=SANITY_DENSITY_MAX,
            message="mass over volume is off by orders of magnitude; the volume is not the plant",
        ))
        # Advisory rather than blocking on purpose: an implausible specimen is a
        # legitimate input to a comparison of methods, and dropping those would
        # bias the comparison toward whatever reconstructs well.
        report.checks.append(_check(
            "implied density is physically plausible", "reconstruction",
            200.0 <= density <= 1000.0, blocking=False,
            value=density, threshold=1000.0,
            message="outside fresh plant tissue; usable, but it is an envelope not a plant",
        ))

    report.checks.append(_check(
        "pot rim was measured, not assumed", "reconstruction",
        bool(cached.pot.confident), blocking=False,
        message="no step in the profile, so the rim fell back to the configured constant",
    ))
    return report


# ---------------------------------------------------------------------------
# Training and fine-tuning
# ---------------------------------------------------------------------------

def check_training(
    metrics: dict,
    *,
    subject: str = "run",
    losses: list[float] | None = None,
    mean_rmse: float | None = None,
) -> GateReport:
    """Did this run learn anything?

    The failure this exists for is a run that finishes, records a respectable
    RMSE, and is marked done, having collapsed onto the training mean. That
    scores close to the mean predictor and looks like a result in a table. Eight
    hours of GPU time should not end that way unnoticed.
    """
    report = GateReport(subject)

    finite = [
        (k, v) for k, v in metrics.items()
        if isinstance(v, (int, float)) and not np.isfinite(v)
    ]
    report.checks.append(_check(
        "metrics are finite", "training",
        not finite, blocking=True,
        message=f"non-finite: {[k for k, _ in finite]}" if finite else "",
    ))

    predictions = metrics.get("predictions")
    targets = metrics.get("targets")
    if predictions:
        predictions = np.asarray(predictions, dtype=float)
        spread = float(predictions.std())
        reference = float(np.std(targets)) if targets else None
        ratio = spread / reference if reference else None
        report.checks.append(_check(
            "predictions vary", "training",
            ratio is None or ratio >= MIN_PREDICTION_SPREAD, blocking=True,
            value=ratio, threshold=MIN_PREDICTION_SPREAD,
            message="the model collapsed onto the mean; RMSE looks fine and nothing was learned",
        ))

    rmse = metrics.get("rmse_kg")
    if rmse is not None and mean_rmse is not None:
        report.checks.append(_check(
            "beats the mean predictor", "training",
            rmse < mean_rmse, blocking=True,
            value=rmse, threshold=mean_rmse,
            message="worse than predicting the training mean, so the run is not a result",
        ))

    ap = metrics.get("occupancy_ap")
    if ap is not None:
        # Chance average precision equals the positive rate. Carved occupancy is
        # a few per cent of the grid, so anything at that level is guessing.
        report.checks.append(_check(
            "occupancy is better than chance", "training",
            ap > 0.10, blocking=True, value=ap, threshold=0.10,
            message="average precision at the occupied base rate is chance",
        ))

    if losses and len(losses) >= 2:
        first, last = float(losses[0]), float(losses[-1])
        drop = (first - last) / max(abs(first), 1e-9)
        report.checks.append(_check(
            "loss actually fell", "training",
            drop >= MIN_LOSS_DROP, blocking=True,
            value=drop, threshold=MIN_LOSS_DROP,
            message="the loss did not move; check the learning rate and the targets",
        ))
    return report


# ---------------------------------------------------------------------------
# Regression against a recorded baseline
# ---------------------------------------------------------------------------

def check_regression(
    current: dict,
    baseline: dict,
    *,
    tolerance: float = 0.02,
    subject: str = "report",
) -> GateReport:
    """Has a number moved that nobody meant to move?

    Run after any change that should not have altered results. Silent numeric
    drift is how a refactor becomes a retraction, and this project has already
    had one claim withdrawn for a related reason.
    """
    report = GateReport(subject)
    for key, was in sorted(baseline.items()):
        if not isinstance(was, (int, float)):
            continue
        now = current.get(key)
        if now is None:
            report.checks.append(_check(
                f"{key} still reported", "regression", False, blocking=False,
                message="present in the baseline and missing now",
            ))
            continue
        moved = abs(now - was) > max(tolerance * abs(was), 1e-9)
        report.checks.append(_check(
            f"{key} unchanged", "regression", not moved, blocking=False,
            value=now, threshold=was,
            message=f"was {was:.4f}, now {now:.4f}" if moved else "",
        ))
    return report


def summarise(reports: list[GateReport]) -> dict:
    """Counts across many subjects, for a exit code and a one-line verdict."""
    blocked = [r for r in reports if r.blocked]
    advisories = sum(len(r.advisories) for r in reports)
    return {
        "subjects": len(reports),
        "blocked": len(blocked),
        "blocked_subjects": [r.subject for r in blocked],
        "advisories": advisories,
        "checks_run": sum(len(r.checks) for r in reports),
    }


__all__ = [
    "Check",
    "GateReport",
    "check_reconstruction",
    "check_regression",
    "check_segmentation",
    "check_training",
    "summarise",
]
