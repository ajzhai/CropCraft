"""Mesh extraction, its descriptors, and the biomass features built on them."""

from __future__ import annotations

import numpy as np
import pytest

skimage = pytest.importorskip("skimage")

from ggssvt.geometry.mesh import (
    mesh_from_occupancy,
    mesh_metrics,
)


def _sphere(resolution: int = 64, radius: float = 20.0) -> np.ndarray:
    grid = np.zeros((resolution,) * 3, dtype=bool)
    centre = resolution // 2
    z, y, x = np.ogrid[:resolution, :resolution, :resolution]
    grid[(x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2 <= radius ** 2] = True
    return grid


def test_enclosed_volume_matches_an_analytic_sphere():
    """The divergence-theorem volume must be right, not merely plausible."""
    radius = 20.0
    mesh = mesh_from_occupancy(_sphere(radius=radius), voxel_size_m=1.0)
    analytic = 4.0 / 3.0 * np.pi * radius ** 3
    assert mesh.enclosed_volume_m3() == pytest.approx(analytic, rel=0.02)


def test_surface_area_carries_the_known_marching_cubes_bias():
    """Marching cubes on a binary grid overestimates area by roughly 8%.

    Pinned rather than corrected: the bias is consistent across specimens so it
    does not affect a regression feature, but it must not be quoted as an
    absolute measurement, and a future change that silently alters it should
    fail here.
    """
    radius = 20.0
    mesh = mesh_from_occupancy(_sphere(radius=radius), voxel_size_m=1.0)
    ratio = mesh.surface_area_m2() / (4.0 * np.pi * radius ** 2)
    assert 1.03 < ratio < 1.15


def test_volume_scales_with_the_cube_of_the_voxel_size():
    grid = _sphere(resolution=32, radius=10.0)
    coarse = mesh_from_occupancy(grid, voxel_size_m=1.0).enclosed_volume_m3()
    fine = mesh_from_occupancy(grid, voxel_size_m=0.5).enclosed_volume_m3()
    assert fine == pytest.approx(coarse / 8.0, rel=1e-6)


def test_empty_occupancy_gives_an_empty_mesh_not_a_crash():
    mesh = mesh_from_occupancy(np.zeros((16, 16, 16), dtype=bool))
    assert mesh.is_empty
    assert mesh.surface_area_m2() == 0.0
    assert mesh.enclosed_volume_m3() == 0.0
    assert mesh.height_m() == 0.0


def test_cropping_above_a_height_keeps_only_the_upper_faces():
    grid = np.zeros((32, 32, 32), dtype=bool)
    grid[12:20, 12:20, 2:30] = True          # a tall column

    mesh = mesh_from_occupancy(grid, voxel_size_m=0.01)
    upper = mesh.crop_above(0.15)

    assert 0 < upper.n_faces < mesh.n_faces
    assert upper.surface_area_m2() < mesh.surface_area_m2()


def test_solidity_separates_a_solid_block_from_a_sparse_shape():
    """The descriptor that distinguishes a pot from a canopy."""
    solid = np.zeros((40, 40, 40), dtype=bool)
    solid[10:30, 10:30, 10:30] = True

    sparse = np.zeros((40, 40, 40), dtype=bool)
    sparse[10:30, 10:30, 10:30] = True
    sparse[12:28, 12:28, 12:28] = False       # hollow it out

    _, solid_m = mesh_metrics(solid, voxel_size_m=0.01, pot_height_m=0.0)
    _, sparse_m = mesh_metrics(sparse, voxel_size_m=0.01, pot_height_m=0.0)

    assert solid_m.solidity > sparse_m.solidity
    assert sparse_m.area_to_volume > solid_m.area_to_volume


def test_mesh_vector_survives_a_degenerate_convex_hull():
    """NaN solidity must not propagate into the regression design matrix."""
    from ggssvt.eval.mesh_baseline import mesh_vector

    vector = mesh_vector(
        {
            "canopy_area_m2": 0.1,
            "surface_area_m2": 0.2,
            "enclosed_volume_m3": 0.001,
            "solidity": float("nan"),
            "area_to_volume": float("inf"),
            "height_m": 0.5,
        }
    )
    assert np.isfinite(vector).all()


def test_obj_export_round_trips_the_counts():
    mesh = mesh_from_occupancy(_sphere(resolution=24, radius=6.0), voxel_size_m=0.01)
    text = mesh.to_obj()

    assert text.count("\nv ") + text.startswith("v ") == mesh.n_vertices
    assert sum(1 for line in text.splitlines() if line.startswith("f ")) == mesh.n_faces


def test_mesh_faces_index_real_vertices():
    mesh = mesh_from_occupancy(_sphere(resolution=24, radius=6.0))
    assert mesh.faces.min() >= 0
    assert mesh.faces.max() < mesh.n_vertices


def test_a_taller_object_reports_a_greater_height():
    short = np.zeros((32, 32, 32), dtype=bool)
    short[14:18, 14:18, 0:8] = True
    tall = np.zeros((32, 32, 32), dtype=bool)
    tall[14:18, 14:18, 0:24] = True

    assert (
        mesh_from_occupancy(tall, voxel_size_m=0.01).height_m()
        > mesh_from_occupancy(short, voxel_size_m=0.01).height_m()
    )


# --- figure rendering knobs -------------------------------------------------

def test_supersampling_returns_the_requested_size():
    """The multiple is an internal detail; the caller asked for `size`."""
    import numpy as np

    from ggssvt.eval.render import render_volume

    occupancy = np.zeros((32, 32, 32), bool)
    occupancy[10:20, 10:20, 5:25] = True
    for scale in (1, 2, 3):
        assert render_volume(occupancy, size=64, supersample=scale).shape == (64, 64, 3)


def test_supersampling_adds_intermediate_tones():
    """Averaging blocks down is what turns a hard edge into a gradient."""
    import numpy as np

    from ggssvt.eval.render import render_volume

    occupancy = np.zeros((32, 32, 32), bool)
    occupancy[10:20, 10:20, 5:25] = True
    aliased = render_volume(occupancy, size=64, supersample=1)
    smoothed = render_volume(occupancy, size=64, supersample=3)
    assert (len(np.unique(smoothed.reshape(-1, 3), axis=0))
            > len(np.unique(aliased.reshape(-1, 3), axis=0)))


def test_a_larger_point_radius_covers_more_canvas():
    import numpy as np

    from ggssvt.eval.render import render_volume

    occupancy = np.zeros((32, 32, 32), bool)
    occupancy[::4, ::4, ::4] = True          # deliberately sparse
    def ink(image):
        return int((image.reshape(-1, 3) != 255).any(axis=1).sum())
    assert ink(render_volume(occupancy, size=64, point_radius=3)) > \
           ink(render_volume(occupancy, size=64, point_radius=1))


def test_an_empty_grid_still_returns_the_right_shape():
    import numpy as np

    from ggssvt.eval.render import render_volume

    empty = np.zeros((16, 16, 16), bool)
    assert render_volume(empty, size=48, supersample=3).shape == (48, 48, 3)
