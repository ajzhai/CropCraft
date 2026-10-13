import numpy as np

from ggssvt.eval.d455_reconstruction_preview import (
    _mask_view,
    _rotation_about_vertical,
)


def test_vertical_rotation_keeps_pivot_fixed():
    pivot = np.array([0.2, 0.4, 1.5])
    points = np.array([pivot, [0.4, 0.4, 1.5]])
    rotated = _rotation_about_vertical(points, 90, pivot)
    np.testing.assert_allclose(rotated[0], pivot)
    np.testing.assert_allclose(np.linalg.norm(rotated[1] - pivot), 0.2)


def test_mask_keeps_green_foreground_and_connected_stem_without_pot():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[20:50, 48:53] = (0, 180, 0)
    rgb[47:70, 49:52] = (100, 100, 100)
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    background = np.full((100, 100), 3.0, dtype=np.float32)
    mask = _mask_view(
        rgb,
        depth,
        background,
        axis_x=50,
        pot_rim_y=65,
        foreground_margin_m=0.1,
    )
    assert mask[30, 50]
    assert mask[55, 50]
    assert not mask[75, 50]
