"""Leave-one-batch-out, beside leave-one-out, so the confound is measured.

Section 4 of FINDINGS establishes that capture batch alone explains most of the
mass variance: R2 0.887 on the two original Eucalyptus batches, still 0.697
across all four. Every number reported from leave-one-out cross-validation is
therefore optimistic by an unknown amount, because for any held-out specimen the
other nine or so members of its own batch are sitting in the training fold,
carrying the batch's mean mass with them.

Leave-one-batch-out removes that. A whole batch is withheld, the model never sees
a single specimen captured in the same session as the one it is scored on, and
what is left is whatever generalises across sessions.

**The gap between the two is the result.** Not the LOBO number on its own, which
will look bad, and not the LOOCV number on its own, which is contaminated. A
method whose LOOCV and LOBO scores are close has learned something about plants;
one whose LOBO score collapses to the mean has learned which batch a specimen
came from. Reporting both turns the project's largest liability into a measured
quantity with a number attached, which is a far better position than reporting
the optimistic figure and hoping the batch structure is not checked.

``batch_only`` is the diagnostic that calibrates the pair. It predicts a
specimen's mass as the mean of the other members of its own batch and nothing
else: no geometry, no image, no features. Under leave-one-out it is the
reproduction of the FINDINGS number. Any real method that fails to beat it under
leave-one-out is doing nothing that batch membership does not already do.

Costs seconds on a CPU. Nothing here trains a network.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import WORK_DIR
from .metrics import regression_metrics

# Specimen identifiers are a letter and a three-digit number, and the capture
# sessions ran in blocks of ten. Splitting on that gives E001-E010, E011-E020,
# M001-M010 and V001-V008 without hard-coding the ranges, so a later capture
# lands in its own batch rather than silently joining an existing one.
BLOCK = 10

_ID = re.compile(r"^([A-Za-z]+)(\d+)$")


def batch_key(plant_id: str) -> tuple[str, int]:
    """The (prefix, block) a specimen belongs to. Unparseable ids get block -1."""
    match = _ID.match(plant_id.strip())
    if not match:
        return (plant_id.strip(), -1)
    prefix, number = match.group(1).upper(), int(match.group(2))
    return (prefix, (number - 1) // BLOCK)


def batch_names(plant_ids: list[str]) -> dict[str, str]:
    """Map each specimen to a readable batch label spanning its actual members.

    The label is built from the members present rather than from the block
    bounds, so a batch of eight reads ``V001-V008`` and not ``V001-V010``.
    """
    members: dict[tuple[str, int], list[str]] = {}
    for pid in plant_ids:
        members.setdefault(batch_key(pid), []).append(pid)

    labels: dict[str, str] = {}
    for group in members.values():
        ordered = sorted(group)
        label = ordered[0] if len(ordered) == 1 else f"{ordered[0]}-{ordered[-1]}"
        for pid in group:
            labels[pid] = label
    return labels


@dataclass
class Scheme:
    """One cross-validation scheme applied to one feature set."""

    condition: str
    scheme: str                       # loocv | lobo
    rmse_kg: float
    mae_kg: float
    r2: float
    bias_kg: float
    n: int
    n_folds: int
    smallest_train: int

    def as_dict(self) -> dict:
        return asdict(self)


def _folds(
    plant_ids: list[str], scheme: str, groups: list[str] | None = None
) -> list[np.ndarray]:
    """Test-index arrays for a scheme. One specimen per fold, or one group.

    ``groups`` names the group each specimen belongs to. Left out, it is derived
    from the identifiers, which is what our own capture batches are. Passed
    explicitly, any grouping works: the lettuce validation set holds out one
    cultivar at a time, which is the same question asked of a different nuisance
    variable.
    """
    n = len(plant_ids)
    if scheme == "loocv":
        return [np.array([i]) for i in range(n)]

    if groups is None:
        labels = batch_names(plant_ids)
        member = [labels[pid] for pid in plant_ids]
    else:
        if len(groups) != n:
            raise ValueError("groups must name every specimen")
        member = list(groups)

    order: list[str] = []
    for name in member:
        if name not in order:
            order.append(name)
    return [
        np.array([i for i, name in enumerate(member) if name == held])
        for held in order
    ]


def _ridge_predict(train_x, train_y, test_x, *, alpha: float, components: int | None):
    """Fit on the training fold only, then predict. Nothing leaks across it.

    The same estimator label_efficiency uses, repeated rather than imported so
    that changing the label-efficiency probe cannot silently change what the
    batch comparison means.
    """
    if components:
        mean = train_x.mean(axis=0)
        k = int(min(components, train_x.shape[0] - 1, train_x.shape[1]))
        _, _, vh = np.linalg.svd(train_x - mean, full_matrices=False)
        basis = vh[:k]
        train_x = (train_x - mean) @ basis.T
        test_x = (test_x - mean) @ basis.T

    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    standardised = (train_x - mu) / sd

    gram = standardised.T @ standardised + alpha * np.eye(standardised.shape[1])
    weights = np.linalg.solve(gram, standardised.T @ (train_y - train_y.mean()))
    return ((test_x - mu) / sd) @ weights + train_y.mean()


def cross_validate(
    features: np.ndarray,
    targets: np.ndarray,
    plant_ids: list[str],
    *,
    condition: str,
    scheme: str,
    groups: list[str] | None = None,
    alpha: float = 1.0,
    components: int | None = 8,
) -> tuple[Scheme, np.ndarray]:
    """Score one feature set under one scheme. Returns the summary and the
    out-of-fold predictions, which the paired bootstrap needs."""
    folds = _folds(plant_ids, scheme, groups)
    predictions = np.zeros(len(targets))
    smallest = len(targets)

    for test in folds:
        train = np.setdiff1d(np.arange(len(targets)), test)
        smallest = min(smallest, int(train.size))
        predictions[test] = _ridge_predict(
            features[train], targets[train], features[test],
            alpha=alpha, components=components,
        )

    m = regression_metrics(predictions, targets)
    return Scheme(
        condition=condition,
        scheme=scheme,
        rmse_kg=round(m.rmse_kg, 4),
        mae_kg=round(m.mae_kg, 4),
        r2=round(m.r2, 4),
        bias_kg=round(m.bias_kg, 4),
        n=m.n,
        n_folds=len(folds),
        smallest_train=smallest,
    ), predictions


def batch_only(
    targets: np.ndarray, plant_ids: list[str], *, scheme: str,
    groups: list[str] | None = None,
) -> tuple[Scheme, np.ndarray]:
    """Predict a specimen's mass from its batch and nothing else.

    Under leave-one-out this is the FINDINGS section 4 number: the held-out
    specimen is predicted by the mean of the rest of its own batch. Under
    leave-one-batch-out the batch is gone, so the only honest prediction left is
    the grand mean of the training batches, which is the point. The information
    that made batch membership powerful is exactly the information the scheme
    removes.
    """
    member = (list(groups) if groups is not None
              else [batch_names(plant_ids)[pid] for pid in plant_ids])
    predictions = np.zeros(len(targets))

    for test in _folds(plant_ids, scheme, groups):
        train = np.setdiff1d(np.arange(len(targets)), test)
        for i in test:
            same = [j for j in train if member[j] == member[i]]
            predictions[i] = targets[same].mean() if same else targets[train].mean()

    m = regression_metrics(predictions, targets)
    return Scheme(
        condition="batch membership only",
        scheme=scheme,
        rmse_kg=round(m.rmse_kg, 4),
        mae_kg=round(m.mae_kg, 4),
        r2=round(m.r2, 4),
        bias_kg=round(m.bias_kg, 4),
        n=m.n,
        n_folds=len(_folds(plant_ids, scheme, groups)),
        smallest_train=0,
    ), predictions


def mcnemar(a_passes: list[bool], b_passes: list[bool]) -> dict:
    """Exact McNemar test for two screens applied to the same specimens.

    The plausibility counts are paired: silhouette carving and depth fusion are
    scored on the same 36 reconstructions, against the same criterion. Comparing
    8 with 31 as though they came from independent samples throws that pairing
    away, and the pairing is where the power is. Only the specimens the two
    methods disagree on carry information; the exact binomial on those
    discordant pairs is the test.

    Returns the two discordant counts and a two-sided exact p-value.
    """
    if len(a_passes) != len(b_passes):
        raise ValueError("paired test needs the same specimens on both sides")

    only_a = sum(1 for a, b in zip(a_passes, b_passes) if a and not b)
    only_b = sum(1 for a, b in zip(a_passes, b_passes) if b and not a)
    discordant = only_a + only_b

    if discordant == 0:
        p = 1.0
    else:
        # Two-sided exact binomial at p=0.5 over the discordant pairs.
        from math import comb
        smaller = min(only_a, only_b)
        tail = sum(comb(discordant, k) for k in range(smaller + 1))
        p = min(1.0, 2.0 * tail / (2 ** discordant))

    return {
        "n_pairs": len(a_passes),
        "only_a": only_a,
        "only_b": only_b,
        "discordant": discordant,
        "p_value": round(p, 6),
        "note": (
            "exact McNemar over discordant pairs; concordant specimens carry no "
            "information about which screen is better"
        ),
    }


def run(
    *,
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "batch_holdout.json",
    exclude: tuple[str, ...] = ("V010",),
    verbose: bool = True,
) -> dict:
    """Score every cached representation under both schemes and report the gap.

    Args:
        exclude: specimens kept out, matching label_efficiency. V010's
            reconstruction fails the plausibility criterion by a factor of
            twenty and dominates any fold it lands in; see FINDINGS 7h.
    """
    from ..data.preprocess import load_cached, usable_plant_ids
    from .baselines import extract_features
    from .metrics import paired_bootstrap_difference

    everything = [p for p in usable_plant_ids(cache_dir) if p not in exclude]
    if len(everything) < 4:
        raise ValueError(f"only {len(everything)} specimens; nothing to hold out")

    cached = {p: load_cached(p, cache_dir) for p in everything}
    geometric = {p: extract_features(c).geometric_vector() for p, c in cached.items()}
    mass = {p: float(c.target_kg) for p, c in cached.items()}

    # The descriptor caches were extracted on whatever specimen set was usable at
    # the time, and those sets differ. Conditions fitted on different specimens
    # cannot be compared, so the comparison runs on the intersection and says
    # which specimens that cost: the rule label_efficiency already uses.
    sources: dict[str, tuple[list[str], np.ndarray]] = {}
    for path in sorted(cache_dir.glob("descriptors_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            stored = [str(pid) for pid in data["plant_ids"]]
            matrix = np.asarray(data["features"], dtype=np.float64)
        name = path.stem.replace("descriptors_", "").replace("_", "-")
        sources[name] = (stored, matrix)

    shared = set(everything)
    for stored, _ in sources.values():
        shared &= set(stored)
    plant_ids = [p for p in everything if p in shared]
    dropped = [p for p in everything if p not in shared]

    if verbose and dropped:
        print(f"  {len(plant_ids)} specimens shared by every condition; "
              f"{len(dropped)} missing from a descriptor cache: "
              f"{', '.join(dropped)}")

    targets = np.array([mass[p] for p in plant_ids])

    conditions: dict[str, np.ndarray] = {
        "geometric (no DINO)": np.stack([geometric[p] for p in plant_ids]),
    }
    for name, (stored, matrix) in sources.items():
        index = {pid: i for i, pid in enumerate(stored)}
        conditions[name] = matrix[[index[pid] for pid in plant_ids]]

    # The geometric features are also scored on every usable specimen, because
    # the rest of the project quotes that n and a reader will look for it here.
    extra_rows: list[dict] = []
    if dropped:
        all_targets = np.array([mass[p] for p in everything])
        all_matrix = np.stack([geometric[p] for p in everything])
        for scheme in ("loocv", "lobo"):
            row, _ = cross_validate(
                all_matrix, all_targets, everything,
                condition="geometric, every usable specimen", scheme=scheme)
            extra_rows.append(row.as_dict())

    labels = batch_names(plant_ids)
    batches: dict[str, int] = {}
    for pid in plant_ids:
        batches[labels[pid]] = batches.get(labels[pid], 0) + 1

    rows: list[dict] = list(extra_rows)
    gaps: dict[str, dict] = {}

    for name, matrix in conditions.items():
        loocv, loocv_pred = cross_validate(
            matrix, targets, plant_ids, condition=name, scheme="loocv")
        lobo, lobo_pred = cross_validate(
            matrix, targets, plant_ids, condition=name, scheme="lobo")
        rows.extend([loocv.as_dict(), lobo.as_dict()])

        paired = paired_bootstrap_difference(lobo_pred, loocv_pred, targets)
        gaps[name] = {
            "loocv_rmse": loocv.rmse_kg,
            "lobo_rmse": lobo.rmse_kg,
            "rmse_inflation_kg": round(lobo.rmse_kg - loocv.rmse_kg, 4),
            "loocv_r2": loocv.r2,
            "lobo_r2": lobo.r2,
            "paired_difference": {k: round(v, 4) for k, v in paired.items()
                                  if isinstance(v, (int, float))},
        }
        if verbose:
            print(f"  {name:24s} LOOCV {loocv.rmse_kg:.3f} (R2 {loocv.r2:+.3f})"
                  f"   LOBO {lobo.rmse_kg:.3f} (R2 {lobo.r2:+.3f})"
                  f"   inflation {lobo.rmse_kg - loocv.rmse_kg:+.3f} kg")

    for scheme in ("loocv", "lobo"):
        summary, _ = batch_only(targets, plant_ids, scheme=scheme)
        rows.append(summary.as_dict())
        if verbose and scheme == "loocv":
            print(f"  {'batch membership only':24s} LOOCV {summary.rmse_kg:.3f} "
                  f"(R2 {summary.r2:+.3f})  <- the confound, reproduced")

    result = {
        "n_specimens": len(plant_ids),
        "excluded": list(exclude),
        "not_shared": dropped,
        "batches": batches,
        "rows": rows,
        "gaps": gaps,
        "note": (
            "loocv leaves one specimen out and leaves the rest of its capture "
            "batch in the training fold; lobo withholds the whole batch. The "
            "difference between them is how much of the loocv score came from "
            "batch membership rather than from the plant"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = [
    "BLOCK", "Scheme", "batch_key", "batch_names", "batch_only",
    "cross_validate", "mcnemar", "run",
]
