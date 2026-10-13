"""The per-specimen pot rim estimate."""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.config import POT_HEIGHT_M, VOXEL_SIZE_M
from ggssvt.geometry.pot import estimate_pot_height, vertical_profile


def _volume(resolution: int = 64) -> np.ndarray:
    return np.zeros((resolution, resolution, resolution), dtype=bool)


def _cylinder(volume: np.ndarray, radius: int, z_from: int, z_to: int) -> np.ndarray:
    r = volume.shape[0]
    y, x = np.mgrid[0:r, 0:r]
    disc = (x - r // 2) ** 2 + (y - r // 2) ** 2 <= radius ** 2
    volume[:, :, z_from:z_to] |= disc[:, :, None]
    return volume


def test_finds_the_rim_of_a_pot_with_a_stem_above_it():
    volume = _volume()
    _cylinder(volume, radius=14, z_from=0, z_to=30)     # the pot
    _cylinder(volume, radius=1, z_from=30, z_to=60)     # the stem

    estimate = estimate_pot_height(volume, voxel_size_m=VOXEL_SIZE_M)

    assert estimate.confident
    assert estimate.height_m == pytest.approx(30 * VOXEL_SIZE_M, abs=2 * VOXEL_SIZE_M)
    assert estimate.drop_ratio < 0.2


def test_a_taller_pot_gives_a_taller_rim():
    """The whole point: the answer tracks the specimen, not a constant."""
    short, tall = _volume(), _volume()
    _cylinder(short, radius=14, z_from=0, z_to=20)
    _cylinder(short, radius=1, z_from=20, z_to=60)
    _cylinder(tall, radius=14, z_from=0, z_to=40)
    _cylinder(tall, radius=1, z_from=40, z_to=60)

    assert estimate_pot_height(tall).height_m > estimate_pot_height(short).height_m


def test_a_waist_is_not_a_rim():
    """A pot that narrows and widens again must not be cut at the narrowing."""
    volume = _volume()
    _cylinder(volume, radius=14, z_from=0, z_to=18)
    _cylinder(volume, radius=3, z_from=18, z_to=20)     # brief waist
    _cylinder(volume, radius=14, z_from=20, z_to=34)    # widens back out
    _cylinder(volume, radius=1, z_from=34, z_to=60)

    estimate = estimate_pot_height(volume, voxel_size_m=VOXEL_SIZE_M)

    assert estimate.confident
    assert estimate.height_m > 30 * VOXEL_SIZE_M


def test_falls_back_rather_than_inventing_a_height():
    """No collapse means no measurement, and the caller must be able to tell."""
    volume = _volume()
    _cylinder(volume, radius=14, z_from=0, z_to=60)     # uniform column, no rim

    estimate = estimate_pot_height(volume)

    assert not estimate.confident
    assert estimate.height_m == POT_HEIGHT_M


def test_the_top_of_the_object_is_not_a_rim():
    """A specimen carved as pure pot must report no measurement, not a rim at
    its own summit with nothing above it."""
    volume = _volume()
    _cylinder(volume, radius=14, z_from=0, z_to=24)     # pot and nothing else

    estimate = estimate_pot_height(volume)

    assert not estimate.confident
    assert estimate.height_m == POT_HEIGHT_M


def test_empty_volume_is_not_confident():
    estimate = estimate_pot_height(_volume())

    assert not estimate.confident
    assert estimate.height_m == POT_HEIGHT_M
    assert estimate.body_voxels == 0


def test_a_canopy_cannot_masquerade_as_the_pot_body():
    """A wide crown high up must not be mistaken for the pot it sits above."""
    volume = _volume()
    _cylinder(volume, radius=8, z_from=0, z_to=16)      # modest pot
    _cylinder(volume, radius=1, z_from=16, z_to=40)     # bare stem
    _cylinder(volume, radius=20, z_from=40, z_to=52)    # broad canopy

    estimate = estimate_pot_height(volume, voxel_size_m=VOXEL_SIZE_M)

    assert estimate.confident
    assert estimate.height_m == pytest.approx(16 * VOXEL_SIZE_M, abs=2 * VOXEL_SIZE_M)


def test_profile_counts_voxels_per_slice():
    volume = _volume(8)
    volume[0:2, 0:3, 4] = True

    profile = vertical_profile(volume)

    assert profile.shape == (8,)
    assert profile[4] == 6
    assert profile.sum() == 6
