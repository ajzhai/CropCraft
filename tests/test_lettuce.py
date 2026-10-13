"""The external validation set's loader and segmenter, on synthetic scenes.

These do not need the download. Each test builds a scene with the geometry the
real images have -- a tray at a known depth, a plant raised above it, an orange
crate below it -- and checks the segmenter picks out the part that is a plant.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ggssvt.data.lettuce import (
    CAMERA,
    Intrinsics,
    excess_green,
    load_ground_truth,
    measure,
    saturation,
    segment,
    tray_depth,
)


def scene(*, plant_colour=(60, 140, 30), plant_radius=90, plant_height_m=0.10):
    """A top-down greenhouse scene: floor, orange crate, white tray, plant."""
    height, width = 1080, 1920
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.uint16)

    rgb[:, :] = (118, 122, 124)            # concrete, near-grey
    depth[:, :] = 1100                     # floor, 1.10 m

    rgb[300:800, 500:1450] = (165, 98, 23)  # orange crate, saturated
    depth[300:800, 500:1450] = 1020         # sits on the floor

    rgb[350:750, 700:1250] = (165, 161, 171)  # white tray, unsaturated
    depth[350:750, 700:1250] = 980            # sits on the crate: 0.98 m

    rows, cols = np.ogrid[:height, :width]
    plant = ((rows - 550) ** 2 + (cols - 975) ** 2) <= plant_radius ** 2
    rgb[plant] = plant_colour
    depth[plant] = int(round((0.980 - plant_height_m) * 1000))
    return rgb, depth


def test_tray_is_found_and_is_not_the_floor():
    rgb, depth = scene()
    found = tray_depth(rgb, depth.astype(np.float64) * CAMERA.depth_scale)
    assert found == pytest.approx(0.980, abs=0.01)


def test_the_crate_is_below_the_tray_and_is_not_segmented():
    rgb, depth = scene()
    mask, tray = segment(rgb, depth.astype(np.float64) * CAMERA.depth_scale)
    assert tray == pytest.approx(0.980, abs=0.01)
    # the crate is saturated too, so only the height rule can exclude it
    assert not mask[400, 600]
    assert mask[550, 975]
    assert mask.sum() == pytest.approx(np.pi * 90 ** 2, rel=0.15)


def test_a_red_cultivar_segments_as_well_as_a_green_one():
    # Satine measures R 80, G 49, B 24: excess green is negative, so the index
    # the rest of the project segments on would lose this plant entirely.
    red = (80, 49, 24)
    rgb, depth = scene(plant_colour=red)
    assert excess_green(rgb)[550, 975] < 0
    assert saturation(rgb)[550, 975] > 0.5

    mask, _ = segment(rgb, depth.astype(np.float64) * CAMERA.depth_scale)
    assert mask[550, 975]
    assert mask.sum() == pytest.approx(np.pi * 90 ** 2, rel=0.15)


def test_height_and_area_come_back_in_the_units_the_ground_truth_uses():
    rgb, depth = scene(plant_radius=120, plant_height_m=0.15)
    measured = measure(rgb, depth)

    assert measured["valid"] == 1.0
    assert measured["height_cm"] == pytest.approx(15.0, abs=1.0)

    # A disc of 120 px at 0.83 m, through the real intrinsics.
    metres_per_px = (0.980 - 0.15) / CAMERA.fx
    expected_cm2 = np.pi * (120 * metres_per_px) ** 2 * 1e4
    assert measured["area_cm2"] == pytest.approx(expected_cm2, rel=0.1)
    assert measured["diameter_cm"] == pytest.approx(
        2 * 120 * metres_per_px * 100, rel=0.1)


def test_an_empty_tray_reports_invalid_rather_than_dividing_by_zero():
    rgb, depth = scene(plant_radius=3)
    measured = measure(rgb, depth)
    assert measured["valid"] == 0.0
    assert measured["area_cm2"] == 0.0


def test_pixel_area_grows_with_the_square_of_depth():
    near = CAMERA.pixel_area_m2(np.array([1.0]))
    far = CAMERA.pixel_area_m2(np.array([2.0]))
    assert far[0] == pytest.approx(4.0 * near[0])


def test_intrinsics_match_the_datasets_own_readme():
    assert CAMERA == Intrinsics()
    assert CAMERA.fx == pytest.approx(1371.58264160156)
    assert CAMERA.depth_scale == pytest.approx(0.001, abs=1e-6)


def test_ground_truth_reads_in_numeric_order_and_converts_to_kilograms(tmp_path):
    payload = {"Measurements": {
        f"Image{i}": {
            "Variety": "Salanova", "RGB_Image": f"RGB_{i}.png",
            "Depth_Information": f"Depth_{i}.png",
            "FreshWeightShoot": 100.0 * i, "DryWeightShoot": 1.0,
            "Height": 10.0, "Diameter": 20.0, "LeafArea": 300.0,
        } for i in (1, 2, 10)
    }}
    (tmp_path / "GroundTruth").mkdir()
    (tmp_path / "GroundTruth" / "GroundTruth_All_388_Images.json").write_text(
        json.dumps(payload), encoding="utf-8")

    records = load_ground_truth(tmp_path)
    # "Image10" sorts before "Image2" as a string; the loader must not do that.
    assert [r.image_id for r in records] == ["Image1", "Image2", "Image10"]
    assert records[0].fresh_weight_kg == pytest.approx(0.1)


def test_a_missing_download_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="10.4121/15023088"):
        load_ground_truth(tmp_path)


def test_a_flat_surface_has_rugosity_one_and_a_crumpled_one_has_more():
    from ggssvt.data.lettuce import surface_descriptors

    flat_rgb, flat_depth = scene(plant_radius=150, plant_height_m=0.10)
    depth_m = flat_depth.astype(np.float64) * CAMERA.depth_scale
    mask, tray = segment(flat_rgb, depth_m)
    flat = surface_descriptors(
        mask, depth_m, tray, projected_area_m2=float(mask.sum()) * 1e-6)

    # A disc at one depth is a plane: its surface is its own shadow.
    assert flat["rugosity"] == pytest.approx(1.0, abs=0.05)
    assert flat["normal_z_mean"] == pytest.approx(1.0, abs=0.05)

    # Corrugate it and the same outline now covers more surface.
    rough_rgb, rough_depth = scene(plant_radius=150, plant_height_m=0.10)
    rows = np.arange(rough_depth.shape[0])[:, None]
    ripple = (6 * np.sin(rows / 4.0)).astype(np.int16)
    plant = rough_depth < 900
    rough_depth = np.where(plant, rough_depth + ripple, rough_depth).astype(np.uint16)

    depth_m = rough_depth.astype(np.float64) * CAMERA.depth_scale
    mask, tray = segment(rough_rgb, depth_m)
    rough = surface_descriptors(
        mask, depth_m, tray, projected_area_m2=float(mask.sum()) * 1e-6)

    assert rough["rugosity"] > flat["rugosity"] + 0.1
    assert rough["normal_z_mean"] < flat["normal_z_mean"]


def test_the_point_cloud_is_metric_and_sits_on_the_tray_plane():
    from ggssvt.data.lettuce import point_cloud

    rgb, depth = scene(plant_radius=120, plant_height_m=0.15)
    depth_m = depth.astype(np.float64) * CAMERA.depth_scale
    mask, tray = segment(rgb, depth_m)
    points = point_cloud(mask, depth_m, tray)

    assert points.shape[1] == 3
    assert points.shape[0] == mask.sum()
    # z is height above the tray, so a plant raised 15 cm sits at 0.15 m.
    assert points[:, 2].max() == pytest.approx(0.15, abs=0.01)
    # x and y span the disc's true width, through the real intrinsics.
    metres_per_px = (0.980 - 0.15) / CAMERA.fx
    assert np.ptp(points[:, 0]) == pytest.approx(2 * 120 * metres_per_px, rel=0.1)


def test_hull_volume_grows_when_the_canopy_domes_at_the_same_outline():
    from ggssvt.data.lettuce import surface_descriptors

    volumes = []
    for height in (0.05, 0.15):
        rgb, depth = scene(plant_radius=120, plant_height_m=height)
        depth_m = depth.astype(np.float64) * CAMERA.depth_scale
        mask, tray = segment(rgb, depth_m)
        volumes.append(surface_descriptors(
            mask, depth_m, tray, projected_area_m2=1.0)["hull_volume_l"])

    assert volumes[1] > volumes[0]
