"""H1's second half: does the self-supervised backbone need fewer labels?

H1 makes two claims. The first, that a self-supervised transformer beats a
convolutional stem, is a single comparison and the campaign supplies it. The
second is that it "reaches a given accuracy from substantially fewer labelled
examples", and that is the claim which makes the method self-supervised rather
than merely transformer-based. Nothing in this project measured it.

This does, and it does so without the GPU. Label efficiency is conventionally
reported by freezing the representation and fitting a small head on a fraction of
the labels, which is exactly the frozen-feature probe already used here. The
backbone never sees a mass either way, so subsampling labels changes only the
head's training set and stage 1 is not repeated. That turns an experiment that
would cost hours of fine-tuning into one that costs seconds of ridge regression.

**The protocol.** For every held-out specimen, the head is fitted on a random
subsample of the remaining ones at each label fraction, and the prediction for
the held-out specimen is recorded. Standardisation and any rotation are fitted
inside the subsample, so a specimen excluded from the labels is excluded from
everything the head knows. Several seeds per fraction, because at these sample
sizes which specimens you happen to draw matters as much as how many.

**What decides it.** Not whether the curves differ at full labels, which is the
first half of H1 and measured elsewhere. The question here is the *shape*: the
labels each backbone needs to reach the convolutional baseline's full-label
accuracy. Parallel curves mean the transformer is better but not more
label-efficient, and H1's second half fails. That distinction is the whole
experiment, so it is reported as its own number rather than left to a reader
eyeballing a plot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import WORK_DIR

# Fractions of the available labels. 1.0 is the reference every curve is read
# against, so it is always included.
FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# Independent subsamples per fraction. At n in the thirties, which specimens are
# drawn matters as much as how many, and one draw would be a coin flip.
N_REPEATS = 8


@dataclass
class CurvePoint:
    """One backbone at one label fraction."""

    condition: str
    fraction: float
    n_labels: int
    rmse: float
    rmse_sd: float
    r2: float
    floor: float = float("nan")     # the mean predictor, for context

    @property
    def unstable(self) -> bool:
        """Is this point a fit that failed rather than a fit that was poor?

        With few labels and a low-dimensional descriptor the ridge is close to
        singular, and a single draw can produce an RMSE two orders of magnitude
        above the mean predictor. Measured here: the seven geometric features at
        eight labels gave 128.8 kg with a standard deviation of 293.6. That is
        not a label-efficiency measurement, it is a numerical failure, and
        reporting it as a curve point would make any comparison against it
        meaningless.
        """
        spread = self.rmse_sd > self.rmse
        absurd = np.isfinite(self.floor) and self.rmse > 5.0 * self.floor
        return bool(spread or absurd)

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "fraction": self.fraction,
            "n_labels": self.n_labels,
            "rmse": round(self.rmse, 4),
            "rmse_sd": round(self.rmse_sd, 4),
            "r2": round(self.r2, 4),
            "unstable": self.unstable,
        }


@dataclass
class Curve:
    """A backbone's whole label-efficiency curve."""

    condition: str
    points: list[CurvePoint] = field(default_factory=list)

    def at(self, fraction: float) -> CurvePoint | None:
        return next((p for p in self.points if p.fraction == fraction), None)

    def labels_to_reach(self, target_rmse: float) -> int | None:
        """Fewest labels at which this curve reaches ``target_rmse``.

        None when it never does. This is the number H1's second half is about:
        two curves can be far apart everywhere and still need the same labels to
        clear a bar, which would mean the transformer is better but not more
        label-efficient.
        """
        reached = [p for p in sorted(self.points, key=lambda q: q.n_labels)
                   if p.rmse <= target_rmse and not p.unstable]
        return reached[0].n_labels if reached else None


def _ridge_predict(train_x, train_y, test_x, *, alpha: float, components: int | None):
    """Fit on the subsample only, then predict. Nothing leaks from outside it."""
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


def curve(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    condition: str,
    fractions: tuple[float, ...] = FRACTIONS,
    repeats: int = N_REPEATS,
    alpha: float = 1.0,
    components: int | None = 8,
    seed: int = 0,
) -> Curve:
    """Leave-one-out RMSE at each label fraction, averaged over subsamples."""
    n = len(targets)
    rng = np.random.default_rng(seed)
    points: list[CurvePoint] = []

    for fraction in fractions:
        # Drawn from the n-1 training specimens, so the count is what the head
        # actually sees rather than a share of the whole set.
        n_labels = max(2, round(fraction * (n - 1)))
        per_repeat_rmse = []
        pooled = np.zeros((repeats, n))

        for repeat in range(repeats):
            predictions = np.zeros(n)
            for held_out in range(n):
                pool = np.delete(np.arange(n), held_out)
                chosen = (pool if n_labels >= pool.size
                          else rng.choice(pool, n_labels, replace=False))
                predictions[held_out] = float(_ridge_predict(
                    features[chosen], targets[chosen],
                    features[held_out : held_out + 1],
                    alpha=alpha, components=components,
                )[0])
            pooled[repeat] = predictions
            per_repeat_rmse.append(
                float(np.sqrt(np.mean((targets - predictions) ** 2)))
            )

        mean_predictions = pooled.mean(axis=0)
        residual = float(((targets - mean_predictions) ** 2).sum())
        total = float(((targets - targets.mean()) ** 2).sum())
        points.append(CurvePoint(
            condition=condition,
            fraction=fraction,
            n_labels=n_labels,
            rmse=float(np.mean(per_repeat_rmse)),
            rmse_sd=float(np.std(per_repeat_rmse)),
            r2=1.0 - residual / total if total > 0 else float("nan"),
            floor=float(np.sqrt(np.mean((targets - targets.mean()) ** 2))),
        ))
    return Curve(condition=condition, points=points)


def compare(curves: list[Curve], *, reference: str) -> dict:
    """Read the curves against the reference backbone's full-label accuracy.

    The comparison H1 actually makes is not "is it better" but "does it need
    fewer labels to be this good", so the bar is the reference at 100 per cent
    and every curve is asked how few labels clear it.
    """
    ref = next((c for c in curves if c.condition == reference), None)
    full = ref.at(1.0) if ref else None
    if full is None:
        return {"reference": reference, "bar": None, "reached": {}}

    reached = {c.condition: c.labels_to_reach(full.rmse) for c in curves}
    return {
        "reference": reference,
        "bar_rmse": round(full.rmse, 4),
        "bar_labels": full.n_labels,
        "labels_to_reach": reached,
        "note": (
            "labels each condition needs to match the reference's full-label "
            "RMSE; null means it never does. Fewer than the reference's own "
            "count is the label-efficiency claim. Equal counts mean a condition "
            "is better but not more label-efficient, which fails H1's second half"
        ),
    }


def run(
    *,
    cache_dir: Path = WORK_DIR / "cache",
    out: Path = WORK_DIR / "reports" / "label_efficiency.json",
    exclude: tuple[str, ...] = ("V010",),
    verbose: bool = True,
) -> dict:
    """Build a curve for the geometric features and for each cached backbone.

    Args:
        exclude: specimens kept out of the whole experiment. V010 is excluded by
            default because its reconstruction fails the plausibility criterion
            by a factor of twenty and it single-handedly took the frozen probe
            from R2 +0.312 to -27; see FINDINGS section 7h. Excluding a failed
            reconstruction on a criterion fixed in advance is not the same as
            dropping an inconvenient point.
    """
    from ..data.preprocess import load_cached, usable_plant_ids
    from .baselines import extract_features

    wanted = [p for p in usable_plant_ids(cache_dir) if p not in exclude]

    # Every condition's descriptors were extracted on whatever specimen set was
    # usable at the time, and those sets differ: the DINO descriptors were built
    # on the set shared with the SAM3D cache. Curves fitted on different
    # specimens cannot be compared, so the experiment runs on the intersection
    # and says which specimens that cost.
    sources: dict[str, tuple[list[str], np.ndarray]] = {}
    for path in sorted(cache_dir.glob("descriptors_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            stored = [str(p) for p in data["plant_ids"]]
            matrix = np.asarray(data["features"], dtype=np.float64)
        name = path.stem.replace("descriptors_", "").replace("_", "-")
        sources[name] = (stored, matrix)

    shared = set(wanted)
    for stored, _ in sources.values():
        shared &= set(stored)
    plant_ids = [p for p in wanted if p in shared]
    dropped = [p for p in wanted if p not in shared]

    if verbose and dropped:
        print(f"  {len(plant_ids)} specimens shared by every condition; "
              f"{len(dropped)} not in one of the descriptor caches: "
              f"{', '.join(dropped)}")

    cached = [load_cached(p, cache_dir) for p in plant_ids]
    targets = np.array([float(c.target_kg) for c in cached])

    conditions: dict[str, np.ndarray] = {
        "geometric (no DINO)": np.stack(
            [extract_features(c).geometric_vector() for c in cached]
        ),
    }
    for name, (stored, matrix) in sources.items():
        index = {pid: i for i, pid in enumerate(stored)}
        conditions[name] = matrix[[index[pid] for pid in plant_ids]]

    curves = []
    for name, matrix in conditions.items():
        built = curve(matrix, targets, condition=name)
        curves.append(built)
        if verbose:
            line = "  ".join(
                f"{p.fraction:.0%}: {p.rmse:.3f}+-{p.rmse_sd:.3f}"
                + ("*" if p.unstable else "")
                for p in built.points
            )
            print(f"  {name:24s} {line}")

    result = {
        "n_specimens": len(plant_ids),
        "excluded": list(exclude),
        "not_shared": dropped,
        "repeats": N_REPEATS,
        "curves": [{"condition": c.condition,
                    "points": [p.as_dict() for p in c.points]} for c in curves],
        "comparison": compare(curves, reference="geometric (no DINO)"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["FRACTIONS", "N_REPEATS", "Curve", "CurvePoint", "compare", "curve", "run"]
