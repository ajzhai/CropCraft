"""A neural field as a third reconstruction operator.

The threshold that turns density into occupancy has no physical calibration, so
the experiment sweeps it rather than choosing one. What has to hold is that the
sweep cannot manufacture a result: a specimen with no working threshold must
report none, and a per-specimen threshold must not be mistaken for a
calibration.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ggssvt.config import voxel_grid_centres
from ggssvt.eval.neural_field import (
    THRESHOLDS,
    consensus,
    sweep,
    working_thresholds,
)


def _sphere(radius_m=0.10, inside=50.0, outside=0.01, resolution=64, voxel=0.012):
    centres = voxel_grid_centres(resolution, voxel)
    distance = np.linalg.norm(centres - np.array([0.0, 0.0, 0.5]), axis=-1)
    density = np.where(distance < radius_m, inside, outside)
    litres = float((distance < radius_m).sum()) * voxel ** 3 * 1000.0
    return density, litres


def test_the_sweep_recovers_a_known_density():
    """A solid sphere weighed at 400 kg/m3 must be plausible somewhere."""
    density, litres = _sphere()
    mass = litres / 1000.0 * 400.0

    scores = sweep(density, plant_id="SYN", mass_kg=mass, voxel_size_m=0.012)
    assert working_thresholds(scores), "a correctly weighed sphere found nothing"


def test_a_sphere_far_too_light_has_no_working_threshold():
    """The informative outcome: no setting of the free parameter rescues it.

    An envelope holds 4 L for a plant weighing grams. No threshold can help,
    because thresholding only ever removes volume from a field that is already
    solid where it matters.
    """
    density, litres = _sphere()
    mass = litres / 1000.0 * 0.5      # 0.5 kg/m3, lighter than air

    scores = sweep(density, plant_id="SYN", mass_kg=mass, voxel_size_m=0.012)
    assert working_thresholds(scores) == []


def test_raising_the_threshold_never_grows_the_volume():
    """Monotonicity. If this fails the sweep is not measuring what it claims."""
    density, _ = _sphere()
    scores = sweep(density, plant_id="SYN", mass_kg=1.0, voxel_size_m=0.012)
    volumes = [s.volume_l for s in scores]
    assert all(a >= b for a, b in pairwise(volumes))


def test_the_pot_rim_is_honoured_when_given():
    density, _ = _sphere()
    everything = sweep(density, plant_id="S", mass_kg=1.0, voxel_size_m=0.012)
    above = sweep(density, plant_id="S", mass_kg=1.0, voxel_size_m=0.012,
                  pot_height_m=0.55)
    assert above[0].volume_l < everything[0].volume_l


def test_consensus_is_empty_when_specimens_disagree():
    """A per-specimen threshold is a fitted parameter, not a calibration."""
    def scores_for(working):
        density, litres = _sphere(inside=working)
        return sweep(density, plant_id="x", mass_kg=litres / 1000.0 * 400.0,
                     voxel_size_m=0.012)

    a = scores_for(50.0)
    # One specimen that nothing works for empties the intersection.
    density, litres = _sphere()
    b = sweep(density, plant_id="b", mass_kg=litres / 1000.0 * 0.5,
              voxel_size_m=0.012)

    out = consensus({"a": a, "b": b})
    assert out["shared_thresholds"] == []
    assert out["specimens_with_no_working_threshold"] == ["b"]


def test_consensus_of_nothing_does_not_crash():
    assert consensus({})["n_specimens"] == 0


def test_sample_density_refuses_without_the_dataparser_transform(tmp_path):
    """Sampling in the wrong space looks like an empty reconstruction.

    Nerfstudio re-centres and rescales the scene, so the metric grid has to be
    mapped through dataparser_transforms.json. Missing it must be an error, not
    a silently empty result.
    """
    from ggssvt.eval import neural_field

    with pytest.raises((FileNotFoundError, ImportError)):
        neural_field.sample_density(tmp_path)


def test_the_threshold_ladder_is_geometric_and_wide():
    """Densities span orders of magnitude; a linear ladder would miss the answer."""
    ratios = [b / a for a, b in pairwise(THRESHOLDS)]
    assert max(ratios) - min(ratios) < 1e-6, "ladder is not geometric"
    assert THRESHOLDS[-1] / THRESHOLDS[0] >= 1e4
