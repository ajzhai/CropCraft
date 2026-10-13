"""End-to-end checks against the real dataset and cache.

These are integration tests, not unit tests: they read ``dataset/`` and, where
noted, the preprocessed cache. Tests needing the cache skip when it has not been
built, so a fresh clone still passes ``pytest`` without a four-minute
preprocessing run first.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.config import PLANTS_DIR, WORK_DIR
from ggssvt.data import load_dataset, load_ground_truth, load_specimen

CACHE_DIR = WORK_DIR / "cache"
requires_cache = pytest.mark.skipif(
    not (CACHE_DIR / "quality.json").exists(),
    reason="run `python -m ggssvt.cli preprocess` to build the cache",
)
requires_dataset = pytest.mark.skipif(
    not PLANTS_DIR.exists(), reason="dataset/plants is not present"
)


@requires_dataset
def test_every_labelled_specimen_resolves_twelve_distinct_azimuths():
    for specimen in load_dataset():
        azimuths = specimen.azimuths_deg
        assert len(azimuths) == 12, f"{specimen.plant_id} has {len(azimuths)} views"
        assert sorted(azimuths) == list(range(0, 360, 30)), specimen.plant_id


@requires_dataset
def test_ground_truth_is_internally_consistent():
    for plant_id, row in load_ground_truth().items():
        assert row.net_weight_g == pytest.approx(
            row.total_fresh_weight_with_pot_g - row.pot_weight_g, abs=0.5
        ), f"{plant_id}: net weight does not match total minus pot"
        assert row.net_weight_g > 0, plant_id


@requires_dataset
def test_species_labels_are_normalised():
    """E008 is recorded as ``Eucalyptus'`` with a stray apostrophe."""
    species = {row.species for row in load_ground_truth().values()}
    assert "Eucalyptus'" not in species
    assert "Eucalyptus" in species


@requires_dataset
def test_stray_frames_outside_the_manifest_are_ignored():
    """E001 has an orphan camB_000.png that is not in its frames_manifest."""
    on_disk = list((PLANTS_DIR / "E001" / "images").glob("*.png"))
    specimen = load_specimen("E001")
    assert len(on_disk) == 13
    assert specimen.n_views == 12


@requires_cache
def test_cache_round_trips_geometry():
    from ggssvt.data.preprocess import load_cached, usable_plant_ids

    plant_id = usable_plant_ids(CACHE_DIR)[0]
    cached = load_cached(plant_id, CACHE_DIR)

    assert cached.n_views == 12
    assert cached.rgb.shape[1:] == (416, 512, 3)
    assert cached.depth_m.shape == (12, 416, 512)
    assert cached.mask.shape == (12, 416, 512)
    assert cached.occupancy.shape == (128, 128, 128)
    assert cached.target_kg > 0


@requires_cache
def test_registered_subject_points_land_inside_the_working_cylinder():
    from ggssvt.config import ROI_RADIUS_M, ROI_Z_MAX_M
    from ggssvt.data.preprocess import load_cached, usable_plant_ids

    for plant_id in usable_plant_ids(CACHE_DIR)[:3]:
        cached = load_cached(plant_id, CACHE_DIR)
        subject = cached.points_world()[cached.mask]

        radial = np.linalg.norm(subject[:, :2], axis=1)
        assert radial.max() <= ROI_RADIUS_M + 1e-3, plant_id
        assert subject[:, 2].min() >= 0.0, plant_id
        assert subject[:, 2].max() <= ROI_Z_MAX_M + 1e-3, plant_id


@requires_cache
def test_carved_occupancy_is_plausible_for_a_potted_plant():
    from ggssvt.data.preprocess import load_cached, usable_plant_ids

    for plant_id in usable_plant_ids(CACHE_DIR)[:5]:
        cached = load_cached(plant_id, CACHE_DIR)
        occupied = cached.occupancy.sum()

        # A plant fills a small fraction of a 1.5 m cube, but is never empty.
        assert 0 < occupied < 0.05 * cached.occupancy.size, plant_id
        # It stands on the floor rather than floating.
        lowest = np.nonzero(cached.occupancy.any(axis=(0, 1)))[0].min()
        assert lowest * cached.voxel_size_m < 0.30, plant_id


@requires_cache
def test_baselines_beat_the_mean_predictor():
    """The reconstruction must carry some biomass signal at all."""
    from ggssvt.data.preprocess import usable_plant_ids
    from ggssvt.eval.baselines import evaluate_baselines, load_features

    features = load_features(usable_plant_ids(CACHE_DIR), CACHE_DIR)
    results = evaluate_baselines(features)

    mean_rmse = results["mean"][0].rmse_kg
    assert results["geometric features"][0].rmse_kg < mean_rmse


@requires_cache
def test_the_reconstruction_against_image_only_ordering_is_recorded():
    """Research question 3, as a guard rather than as a claim.

    This test has now flipped twice, and both flips were informative:

    * E and M only, n=28: geometric 0.397 beat direct 2D 0.440.
    * Plus V001-V008, n=36: reversed, 0.544 against 0.469.
    * Plus V009 and V010, n=38: reversed again, 0.469 against 0.591.

    **Not one of the three is statistically resolved.** The paired bootstrap
    spans zero every time, and the direction also flips if features are whitened
    before the ridge rather than standardised. An assertion on the ordering would
    therefore be asserting noise, and the earlier version of this test did
    exactly that: it required 3D to lose, and failed the moment two specimens
    were added.

    So it asserts what is actually stable. Both baselines must be computed, both
    must beat nothing worse than a broken fit, and the ordering is printed rather
    than required. If the numbers move, find out what changed in the protocol;
    do not read either direction as a result without the interval.
    """
    import numpy as np

    from ggssvt.data.preprocess import usable_plant_ids
    from ggssvt.eval.baselines import evaluate_baselines, load_features

    features = load_features(usable_plant_ids(CACHE_DIR), CACHE_DIR)
    results = evaluate_baselines(features)

    geometric = results["geometric features"][0].rmse_kg
    image_only = results["direct 2D"][0].rmse_kg
    floor = results["mean"][0].rmse_kg

    assert np.isfinite(geometric) and np.isfinite(image_only)
    # A baseline several times the mean predictor is a failed fit, not a poor
    # one, and that is worth catching however the two happen to be ordered.
    assert geometric < 5 * floor, f"geometric features failed to fit: {geometric:.3f}"
    assert image_only < 5 * floor, f"image-only features failed to fit: {image_only:.3f}"

    ahead = "reconstruction ahead" if geometric < image_only else "pixels ahead"
    print(f"\n  geometric {geometric:.3f} kg, direct 2D {image_only:.3f} kg, "
          f"mean predictor {floor:.3f} kg ({ahead}; unresolved either way)")


@requires_cache
def test_loocv_baseline_never_fits_on_the_held_out_specimen():
    from ggssvt.eval.baselines import MeanPredictor, SpecimenFeatures, loocv_baseline

    def make(mass: float) -> SpecimenFeatures:
        return SpecimenFeatures(
            plant_id=f"P{mass}",
            target_kg=mass,
            above_ground_volume_m3=mass * 1e-3,
            total_volume_m3=mass * 2e-3,
            height_m=1.0,
            mean_spread_m=0.1,
            max_spread_m=0.2,
            compactness=0.1,
            silhouette_area_m2=0.05,
            mean_subject_pixels=1000.0,
            mean_subject_depth_m=1.0,
            subject_pixel_height=100.0,
        )

    features = [make(m) for m in (1.0, 2.0, 3.0)]
    predicted, target = loocv_baseline(MeanPredictor, features)

    # Held out 1.0, the mean of the rest is 2.5 -- not 2.0, which is what a
    # leaky implementation fitted on all three would give.
    assert predicted[0] == pytest.approx(2.5)
    assert np.allclose(target, [1.0, 2.0, 3.0])


@requires_cache
def test_torch_dataset_produces_finite_tensors():
    pytest.importorskip("torch")
    from ggssvt.data.preprocess import usable_plant_ids
    from ggssvt.training.dataset import SpecimenDataset

    dataset = SpecimenDataset(usable_plant_ids(CACHE_DIR)[:2], cache_dir=CACHE_DIR)
    item = dataset[0]

    for key in ("rgb", "depth", "points_world", "subject", "query_points"):
        assert item[key].isfinite().all(), key

    assert item["rgb"].shape == (12, 3, 416, 512)
    assert item["query_points"].shape[1] == 3
    # The sampler must return both classes, or the occupancy loss is degenerate.
    assert 0 < float(item["query_labels"].mean()) < 1


# --- ground truth file integrity -------------------------------------------
#
# Conflict markers were once committed into ground_truth.csv. The symptom was an
# AttributeError on None, several frames from the malformed line and naming
# neither the file nor the row, and it took down every CLI entry point at once.
# These pin the three ways that file can be damaged.

def _write_csv(tmp_path, body: str):
    from pathlib import Path

    path = Path(tmp_path) / "ground_truth.csv"
    header = (
        "plant_id,date,species_breed,total_fresh_weight_with_pot_g,"
        "pot_weight_g,net_weight_g,pot_weight_source,notes\n"
    )
    path.write_text(header + body, encoding="utf-8")
    return path


def test_conflict_markers_are_reported_with_the_line(tmp_path):
    """A short row must name the file and line, not raise AttributeError."""
    import pytest

    from ggssvt.data.dataset import load_ground_truth

    path = _write_csv(
        tmp_path,
        "A001,2026-01-01,Eucalyptus,1000.0,400.0,600.0,measured,\n"
        "<<<<<<< HEAD\n"
        "A002,2026-01-01,Eucalyptus,1200.0,500.0,700.0,measured,\n",
    )
    with pytest.raises(ValueError, match="fields against"):
        load_ground_truth(path)


def test_a_repeated_specimen_is_refused(tmp_path):
    """Two rows for one id means one is superseded; the file has to say which."""
    import pytest

    from ggssvt.data.dataset import load_ground_truth

    path = _write_csv(
        tmp_path,
        "A001,2026-01-01,Eucalyptus,1000.0,400.0,600.0,measured,\n"
        "A001,2026-01-01,Eucalyptus,1000.0,300.0,700.0,estimated,\n",
    )
    with pytest.raises(ValueError, match="repeats A001"):
        load_ground_truth(path)


def test_an_unreadable_weight_names_the_specimen(tmp_path):
    import pytest

    from ggssvt.data.dataset import load_ground_truth

    path = _write_csv(
        tmp_path, "A001,2026-01-01,Eucalyptus,1000.0,,600.0,measured,\n"
    )
    with pytest.raises(ValueError, match="A001.*unreadable weight"):
        load_ground_truth(path)


def test_a_note_containing_a_comma_survives_the_round_trip(tmp_path):
    """V011's note has a comma in it, which must be quoted rather than split."""
    from ggssvt.data.dataset import load_ground_truth

    note = "pot weighed after shoot removal; capture incomplete, 9 of 12 views"
    path = _write_csv(
        tmp_path, f'A001,2026-01-01,Eucalyptus,1000.0,400.0,600.0,measured,"{note}"\n'
    )
    assert load_ground_truth(path)["A001"].notes == note


# --- unlabelled specimens ---------------------------------------------------
#
# Stage 1 fits occupancy against the carve and never reads the mass, so an
# unharvested plant is a valid pretraining example. The risk is the other
# direction: a NaN target reaching a least-squares fit produces NaN
# coefficients and no error at all, so every score downstream would be NaN with
# nothing to say why.

def test_load_features_refuses_a_specimen_with_no_mass(tmp_path, monkeypatch):
    import numpy as np
    import pytest

    from ggssvt.eval import baselines

    class FakeCached:
        plant_id = "U001"
        target_kg = float("nan")

    monkeypatch.setattr(baselines, "load_cached", lambda pid, d: FakeCached())
    monkeypatch.setattr(baselines, "_load_fusion", lambda p: {})
    monkeypatch.setattr(
        baselines, "extract_features",
        lambda cached, fused=None: type("F", (), {
            "plant_id": cached.plant_id, "target_kg": np.float64("nan")})(),
    )
    with pytest.raises(ValueError, match="no weighed mass"):
        baselines.load_features(["U001"], tmp_path)


def test_usable_plant_ids_defaults_to_labelled_only():
    """Every regression path relies on this default; it must not drift."""
    import inspect

    from ggssvt.data.preprocess import usable_plant_ids

    assert inspect.signature(usable_plant_ids).parameters["labelled"].default is True
