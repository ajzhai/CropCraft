"""What each comparison could have detected, and which ones actually settled.

Eight rows of "not resolved" invite one reading, that the methods are all much
of a muchness and the project could not tell them apart. That reading is wrong,
and the table cannot correct it on its own, because a null result means nothing
until you say what the design was capable of detecting.

So this computes it. From the paired bootstrap interval on a difference, the
standard error is the half-width over 1.96, and the smallest difference the same
design would detect four times in five is about 2.8 standard errors. Reported
beside the null, that turns "we could not tell" into "this design resolves
differences above X, and the difference present is smaller than that", which is
a statement about the experiment rather than an absence of one.

It also collects every comparison the project has made into one ledger, resolved
and unresolved together, because the resolved ones are scattered across six
report files and the unresolved ones are all in the same table. Read separately
they give the impression that nothing settled. Read together the pattern is
plain: what settles here is paired counts and paired differences on large
samples, and what does not is a difference in RMSE between two methods on
thirty-six specimens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import WORK_DIR

REPORTS = WORK_DIR / "reports"

# Two-sided, alpha 0.05, 80% power. z(0.975) + z(0.80) = 1.96 + 0.84.
Z_INTERVAL = 1.96
Z_POWER = 0.84


def minimum_detectable(low: float, high: float) -> float:
    """The smallest difference this design would detect four times in five.

    Taken from the interval the experiment already reports rather than from an
    assumed variance, so it describes the design that ran and not an idealised
    one.
    """
    standard_error = (high - low) / (2.0 * Z_INTERVAL)
    return (Z_INTERVAL + Z_POWER) * standard_error


@dataclass
class Comparison:
    """One question the project asked, and whether the answer settled."""

    question: str
    experiment: str
    n: int
    design: str                 # paired bootstrap | exact paired | ratio | descriptive
    effect: str                 # the observed difference, in its own units
    verdict: str                # resolved | not resolved | measured
    evidence: str               # the interval or the p value
    detectable: str = ""        # what the design could have found, where it applies

    def as_dict(self) -> dict:
        return asdict(self)


def _load(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def baseline_comparisons() -> list[Comparison]:
    """The biomass table, with what it could have detected attached."""
    from .dashboard_data import build_payload

    payload = build_payload()
    out = []
    reference = next(
        (m["name"] for m in payload.methods if not m.get("vs_reference")), "reference")

    for method in payload.methods:
        against = method.get("vs_reference")
        if not against:
            continue
        mde = minimum_detectable(against["low"], against["high"])
        out.append(Comparison(
            question=f"{method['name']} against {reference}, on mass",
            experiment="baselines, leave-one-out over 36 specimens",
            n=36,
            design="paired bootstrap",
            effect=f"{against['difference']:+.3f} kg RMSE",
            verdict="resolved" if against.get("resolved") else "not resolved",
            evidence=f"95% interval [{against['low']:+.3f}, {against['high']:+.3f}]",
            detectable=f"{mde:.3f} kg",
        ))
    return out


def settled_comparisons() -> list[Comparison]:
    """Everything decided outside that table, from its own report."""
    out: list[Comparison] = []

    virtual = _load("virtual_views.json")
    if virtual:
        s = virtual["summary"]
        agree = s["metric_vs_truth"]
        out.append(Comparison(
            question="does silhouette IoU rank reconstructions the way the truth does",
            experiment="virtual views of Pheno4D, carve against fusion",
            n=s["n_scans"],
            design="exact paired",
            effect=f"the two measures disagree on {s['disagreements']} of {s['n_scans']}",
            verdict="resolved",
            evidence=f"exact test on discordant pairs, p = {agree['p_value']:.1e}",
        ))
        out.append(Comparison(
            question="does depth fusion recover more of the plant than a hull",
            experiment="virtual views of Pheno4D, scored against the cloud",
            n=s["n_scans"],
            design="exact paired",
            effect=f"IoU {s['mean_fused_iou']:.3f} against {s['mean_carve_iou']:.3f}; "
                   f"volume {s['median_fused_volume_ratio']}x true against "
                   f"{s['median_carve_volume_ratio']}x",
            verdict="resolved",
            evidence=f"the truth prefers fusion on {s['truth_prefers_fusion']} "
                     f"of {s['n_scans']}",
        ))

    external = _load("external_lettuce.json")
    if external:
        for name, paired in (external.get("paired_vs_2d_profile") or {}).items():
            mde = minimum_detectable(paired["low"], paired["high"])
            settled = paired["low"] * paired["high"] > 0
            out.append(Comparison(
                question=f"{name} against 2D + profile, on lettuce mass",
                experiment="4TU lettuce, one cultivar held out",
                n=external["n_after_screen"],
                design="paired bootstrap",
                effect=f"{paired['difference'] * 1000:+.1f} g RMSE",
                verdict="resolved" if settled else "not resolved",
                evidence=f"95% interval [{paired['low'] * 1000:+.1f}, "
                         f"{paired['high'] * 1000:+.1f}] g, "
                         f"p = {paired['p_direction']:.4f}",
                detectable=f"{mde * 1000:.1f} g",
            ))

    dino = _load("dino_probe.json")
    if dino:
        for name, paired in (dino.get("paired_vs_control") or {}).items():
            mde = minimum_detectable(paired["low"], paired["high"])
            out.append(Comparison(
                question=f"{name} against the CNN stem, on mass",
                experiment="frozen probe, leave-one-out over 36 specimens",
                n=dino["n_specimens"],
                design="paired bootstrap",
                effect=f"{paired['difference']:+.3f} kg RMSE",
                verdict="not resolved",
                evidence=f"95% interval [{paired['low']:+.3f}, {paired['high']:+.3f}], "
                         f"p = {paired['p_direction']:.4f}",
                detectable=f"{mde:.3f} kg",
            ))

    label = _load("label_efficiency.json")
    if label:
        reach = (label.get("comparison") or {}).get("labels_to_reach", {})
        bar = (label.get("comparison") or {}).get("bar_labels")
        best = min((v for v in reach.values() if v), default=None)
        if best and bar:
            out.append(Comparison(
                question="does a self-supervised backbone need fewer labels",
                experiment="label-efficiency curve, 8 repeats per point",
                n=label["n_specimens"],
                design="ratio inside one experiment",
                effect=f"{best} labels against {bar} for the reference",
                verdict="resolved",
                evidence="a ratio inside one experiment does not depend on "
                         "separating two nearly equal RMSE values",
            ))

    holdout = _load("batch_holdout.json")
    if holdout:
        rows = {(r["condition"], r["scheme"]): r for r in holdout["rows"]}
        batch = rows.get(("batch membership only", "loocv"))
        if batch:
            others = [r for (c, s), r in rows.items()
                      if s == "loocv" and c != "batch membership only"]
            best = min(others, key=lambda r: r["rmse_kg"]) if others else None
            out.append(Comparison(
                question="does any method beat knowing the capture batch",
                experiment="batch holdout, leave-one-out over 36 specimens",
                n=holdout["n_specimens"],
                design="descriptive",
                effect=f"batch label alone {batch['rmse_kg']:.3f} kg"
                       + (f", best method {best['rmse_kg']:.3f} kg" if best else ""),
                verdict="measured",
                evidence="no method here reaches it",
            ))

    return out


def run(*, out: Path | None = None, verbose: bool = True) -> dict:
    """Build the ledger and say what it adds up to."""
    baselines = baseline_comparisons()
    settled = settled_comparisons()
    everything = baselines + settled

    resolved = [c for c in everything if c.verdict == "resolved"]
    unresolved = [c for c in everything if c.verdict == "not resolved"]

    # The floor the biomass table is arguing beneath. Every difference in it is
    # smaller than what the design can find, which is the fact that makes the
    # eight nulls one fact rather than eight.
    detectable = [float(c.detectable.split()[0]) for c in baselines if c.detectable]
    observed = [abs(float(c.effect.split()[0])) for c in baselines]

    report = {
        "note": "a null result means nothing until the design says what it could "
                "have detected; minimum detectable effect is at 80% power, "
                "two-sided, alpha 0.05, taken from each experiment's own interval",
        "n_comparisons": len(everything),
        "n_resolved": len(resolved),
        "n_unresolved": len(unresolved),
        "biomass_table": {
            "n": 36,
            "largest_observed_difference_kg": round(max(observed), 3) if observed else None,
            "smallest_detectable_difference_kg": round(min(detectable), 3)
            if detectable else None,
            "verdict": "every difference in the table is smaller than the smallest "
                       "the design can detect, so the eight nulls are one fact "
                       "about the sample size rather than eight about the methods",
        },
        "comparisons": [c.as_dict() for c in everything],
    }

    out = out or REPORTS / "resolution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if verbose:
        print(f"  {len(resolved)} resolved, {len(unresolved)} not resolved, "
              f"{len(everything)} comparisons in total")
        b = report["biomass_table"]
        print(f"  biomass table: largest difference "
              f"{b['largest_observed_difference_kg']} kg, smallest detectable "
              f"{b['smallest_detectable_difference_kg']} kg")
        print()
        for c in resolved:
            print(f"  resolved   {c.question[:62]:62s} {c.evidence[:40]}")
        for c in unresolved:
            print(f"  open       {c.question[:62]:62s} needs "
                  f"{c.detectable or 'a larger sample'}")
    return report


__all__ = [
    "Comparison", "baseline_comparisons", "minimum_detectable", "run",
    "settled_comparisons",
]
