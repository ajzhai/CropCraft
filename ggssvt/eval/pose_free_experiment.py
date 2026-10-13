"""The pose-free comparison: registration agreement first, biomass second.

Two experiments, and the order matters.

**Registration agreement** is the one worth running even if the biomass numbers
come out useless. Every pose in this project is estimated from depth, and the
azimuth refinement saturates its search bound on almost every specimen, so the
registration is the least verified assumption in the pipeline. A pose-free
method shares no failure mode with it, so the residual after similarity
alignment is a genuine measurement of how wrong the poses are. Nothing else
available produces that number.

**Biomass from the pose-free reconstruction** slots into the same leave-one-out
comparison as every other method, so it is directly comparable. It carries a
caveat the others do not: DUSt3R and Fast3R return arbitrary scale, so their
volumes depend on the scale recovered from the Kinect depth. That makes them only
partly independent of the existing pipeline. MASt3R's metric checkpoint avoids it
and is the one to trust.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from ..geometry.pose_free import (
    PoseAgreement,
    PoseFreeError,
    PoseFreeResult,
    compare_poses,
    recover_scale_from_depth,
)


@dataclass
class SpecimenOutcome:
    """One specimen under one pose-free method."""

    plant_id: str
    method: str
    agreement: dict | None = None
    scale: float | None = None
    scale_diagnostics: dict | None = None
    n_points: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class PoseFreeReport:
    """Every specimen under every available method."""

    outcomes: list[SpecimenOutcome]
    skipped: dict[str, str] = field(default_factory=dict)

    def by_method(self) -> dict[str, list[SpecimenOutcome]]:
        grouped: dict[str, list[SpecimenOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.method, []).append(outcome)
        return grouped

    def to_table(self) -> str:
        lines = []
        grouped = self.by_method()
        if grouped:
            header = (
                f"{'method':10s} {'n':>3s} {'centre RMSE':>12s} {'rot RMSE':>9s} "
                f"{'azimuth RMSE':>13s} {'scale':>8s}"
            )
            lines += ["Registration agreement with the estimated rig", "", header,
                      "-" * len(header)]
            for method, outcomes in grouped.items():
                good = [o for o in outcomes if o.ok and o.agreement]
                if not good:
                    lines.append(f"{method:10s} {'n/a':>3s}  no successful reconstructions")
                    continue
                centre = np.mean([o.agreement["centre_rmse_m"] for o in good])
                rot = np.mean([o.agreement["rotation_rmse_deg"] for o in good])
                azimuth = np.mean([o.agreement["azimuth_rmse_deg"] for o in good])
                scale = np.mean([o.agreement["scale"] for o in good])
                lines.append(
                    f"{method:10s} {len(good):3d} {centre * 100:10.1f} cm "
                    f"{rot:8.1f}° {azimuth:12.1f}° {scale:8.3f}"
                )
            lines += [
                "",
                "Azimuth RMSE is the number to read: it is the parameter the rig",
                "refinement cannot pin down, and it saturates its ±8° search bound",
                "on almost every specimen. A pose-free method disagreeing by less",
                "than that bound would be reassuring; more would not be.",
            ]

        failures = [o for o in self.outcomes if not o.ok]
        if failures:
            lines += ["", f"{len(failures)} specimen/method pairs failed:"]
            for outcome in failures[:8]:
                lines.append(f"  {outcome.plant_id} / {outcome.method}: {outcome.error[:90]}")

        for method, reason in self.skipped.items():
            lines += ["", f"SKIPPED {method}:", reason]

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "outcomes": [asdict(o) for o in self.outcomes],
                "skipped": self.skipped,
            },
            indent=2,
        )


def run_specimen(
    backend,
    specimen,
    rig,
    *,
    align_scale: bool = True,
) -> SpecimenOutcome:
    """Reconstruct one specimen and compare its poses against the rig."""
    from ..geometry.pose_free_backends import sanity_check_result

    outcome = SpecimenOutcome(plant_id=specimen.plant_id, method=backend.name)

    try:
        result: PoseFreeResult = backend.reconstruct(specimen)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome.warnings = sanity_check_result(result)
    outcome.n_points = int(result.points.shape[0])

    # Scale first: a comparison of poses is scale-invariant, but the biomass
    # numbers that follow are not, and a metric method should be checked rather
    # than trusted.
    if align_scale and not result.is_metric:
        try:
            view = specimen.views[0]
            scale, diagnostics = recover_scale_from_depth(
                _predicted_depth(result, 0, view.load_depth().shape),
                view.load_depth(),
            )
            outcome.scale = scale
            outcome.scale_diagnostics = diagnostics
        except (PoseFreeError, ValueError) as exc:
            outcome.warnings.append(f"scale recovery failed: {exc}")

    try:
        agreement: PoseAgreement = compare_poses(
            result, rig, with_scale=not result.is_metric
        )
        outcome.agreement = agreement.as_dict()
    except PoseFreeError as exc:
        outcome.error = f"pose comparison failed: {exc}"

    return outcome


def _predicted_depth(result: PoseFreeResult, index: int, shape: tuple[int, int]):
    """Per-view depth from a pose-free point cloud.

    The backends return points already in world coordinates, so depth for one
    view is the distance along that camera's optical axis. Points are not
    guaranteed to arrive one-per-pixel, so this reshapes only when the count
    matches and otherwise refuses rather than silently producing a wrong map.
    """
    per_view = result.points.shape[0] // max(1, result.rotations.shape[0])
    if per_view != shape[0] * shape[1]:
        raise PoseFreeError(
            f"cannot map {per_view} points per view onto a {shape[0]}x{shape[1]} "
            "frame; the backend returned a different resolution, so scale must be "
            "recovered from its own depth output instead"
        )

    start = index * per_view
    block = result.points[start : start + per_view]
    relative = block - result.centres[index]
    forward = result.rotations[index] @ np.array([0.0, 0.0, 1.0])
    return (relative @ forward).reshape(shape)


def run_experiment(
    plant_ids: list[str] | None = None,
    *,
    methods: tuple[str, ...] = ("fast3r", "dust3r", "mast3r"),
    device: str = "cuda",
    image_size: int = 512,
    cache_dir: Path = WORK_DIR / "cache",
    out_path: Path | None = None,
    verbose: bool = True,
) -> PoseFreeReport:
    """Run every available pose-free method over the specimens."""
    from ..data.dataset import load_specimen
    from ..data.preprocess import usable_plant_ids
    from ..geometry.pose_free import backend_is_available
    from ..geometry.pose_free_backends import build_backend
    from ..geometry.rig import estimate_rig

    plant_ids = plant_ids or usable_plant_ids(cache_dir)
    outcomes: list[SpecimenOutcome] = []
    skipped: dict[str, str] = {}

    for method in methods:
        available, reason = backend_is_available(method)
        if not available:
            skipped[method] = reason
            if verbose:
                print(f"\n=== {method}: SKIPPED ===\n{reason}\n")
            continue

        if verbose:
            print(f"\n=== {method} ===")
        backend = build_backend(method, device=device, image_size=image_size)

        for index, plant_id in enumerate(plant_ids, start=1):
            specimen = load_specimen(plant_id)
            rig = estimate_rig(specimen)
            outcome = run_specimen(backend, specimen, rig)
            outcomes.append(outcome)

            if verbose:
                if outcome.ok and outcome.agreement:
                    print(
                        f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
                        f"azimuth RMSE {outcome.agreement['azimuth_rmse_deg']:5.1f}°  "
                        f"centre RMSE {outcome.agreement['centre_rmse_m'] * 100:5.1f} cm"
                    )
                else:
                    print(f"  [{index:2d}/{len(plant_ids)}] {plant_id}  FAILED: {outcome.error[:70]}")

    report = PoseFreeReport(outcomes=outcomes, skipped=skipped)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_json(), encoding="utf-8")

    return report


__all__ = [
    "PoseFreeReport",
    "SpecimenOutcome",
    "run_experiment",
    "run_specimen",
]
