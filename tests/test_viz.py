"""The configurable visualisation entry point.

Most of this is drawing, which tests cannot judge. What they can pin is that the
config refuses nonsense loudly rather than rendering something misleading, and
that the deliberate choices, greyscale support and the absence of jet, do not
quietly regress.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ggssvt.eval.viz import COLORMAPS, LAYERS, SOURCES, VizConfig, ramp


def test_an_unknown_layer_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown layer"):
        VizConfig(layers=("occupancy", "hologram"))


def test_an_unknown_source_is_refused():
    with pytest.raises(ValueError, match="unknown source"):
        VizConfig(source="nerf")


def test_jet_is_refused_with_the_reason():
    """A perceptually non-uniform ramp invents boundaries in a depth cue.

    Refusing it in the constructor rather than silently substituting means a
    figure cannot be made with it by accident.
    """
    with pytest.raises(ValueError, match="luminance edges"):
        VizConfig(cmap="jet")


def test_every_advertised_layer_has_a_panel():
    """LAYERS drives the CLI help; a name with no implementation would crash."""
    from ggssvt.eval.viz import _PANELS

    assert set(LAYERS) == set(_PANELS)


def test_every_cache_backed_source_maps_to_a_cache_directory():
    for source in SOURCES:
        if source == "neural":
            continue           # covered separately; it needs a field, not a cache
        assert VizConfig(source=source).cache_dir.name.startswith("cache")


def test_the_neural_source_refuses_without_a_field():
    """A trained field has no occupancy until a threshold is chosen.

    Failing in the constructor means a figure cannot be produced from a source
    that has nothing behind it.
    """
    with pytest.raises(ValueError, match="needs nerf_output"):
        VizConfig(source="neural")


def test_the_neural_source_still_reads_the_capture_from_the_carve_cache(tmp_path):
    """Colour frames and masks are properties of the capture, not the operator.

    Substituting the whole specimen would show what the field imagined instead
    of what the sensor saw, which is the opposite of what a comparison needs.
    """
    config = VizConfig(source="neural", nerf_output=tmp_path)
    assert config.cache_dir.name == "cache"
    assert config.needs_field is True


def test_a_field_of_the_wrong_shape_is_refused(tmp_path):
    """Comparing a 64^3 field against a 128^3 cache would silently mis-scale."""
    import numpy as np

    from ggssvt.eval.neural_field import density_cache_path
    from ggssvt.eval.viz import _with_field_occupancy

    np.savez_compressed(density_cache_path(tmp_path),
                        density=np.zeros((16, 16, 16), np.float32))

    class _Cache:
        occupancy = np.zeros((32, 32, 32), bool)

    with pytest.raises(ValueError, match="cannot be compared"):
        _with_field_occupancy(_Cache(), VizConfig(source="neural",
                                                  nerf_output=tmp_path))


def test_the_caption_states_the_threshold_a_field_was_cut_at(tmp_path):
    """A volume from a field is meaningless without the threshold that made it."""
    import numpy as np

    from ggssvt.eval.viz import _caption

    class _Cache:
        plant_id = "M001"
        species = "Mango"
        target_kg = 0.74
        occupancy = np.ones((8, 8, 8), bool)
        voxel_size_m = 0.012

    text = _caption(_Cache(), VizConfig(source="neural", nerf_output=tmp_path,
                                        threshold=2.5))
    assert "density > 2.5" in text


def test_viridis_is_perceptually_ordered():
    """Monotonic in luminance, which is what makes the depth cue readable."""
    samples = ramp(np.linspace(0.0, 1.0, 16), "viridis").astype(float)
    luminance = samples @ np.array([0.2126, 0.7152, 0.0722])
    assert all(a < b for a, b in pairwise(luminance))


def test_greys_runs_dark_to_light_and_its_reverse_is_the_mirror():
    """A thesis is often printed in greyscale; a colour-only figure fails there."""
    values = np.linspace(0.0, 1.0, 8)
    forward = ramp(values, "greys").astype(int)
    reverse = ramp(values, "greys_r").astype(int)

    assert forward[0, 0] > forward[-1, 0], "greys should darken with value"
    assert np.array_equal(forward[::-1], reverse)


def test_every_colormap_returns_three_uint8_channels():
    for cmap in COLORMAPS:
        out = ramp(np.linspace(0, 1, 5), cmap)
        assert out.shape == (5, 3)
        assert out.dtype == np.uint8


def test_values_outside_the_unit_range_are_clipped_not_wrapped():
    """Wrapping would put the far end of the ramp on an outlier."""
    low = ramp(np.array([-5.0]), "viridis")
    high = ramp(np.array([5.0]), "viridis")
    assert np.array_equal(low, ramp(np.array([0.0]), "viridis"))
    assert np.array_equal(high, ramp(np.array([1.0]), "viridis"))


def test_the_text_backend_returns_lines_of_the_requested_width():
    """Over SSH a PNG on the lab machine is a file you cannot look at."""
    import numpy as np

    from ggssvt.eval.viz import to_text

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[60:140, 60:140] = 255
    lines = to_text(image, width=40).splitlines()

    assert all(len(line) == len(lines[0]) for line in lines), "ragged output"
    assert 35 <= len(lines[0]) <= 45


def test_the_aspect_is_corrected_for_terminal_cells():
    """Cells are about twice as tall as wide; without that everything squashes."""
    import numpy as np

    from ggssvt.eval.viz import to_text

    square = np.zeros((200, 200, 3), dtype=np.uint8)
    lines = to_text(square, width=40).splitlines()
    # Half as many rows as columns, because rows are sampled at half the rate.
    assert len(lines) < len(lines[0])


def test_background_reads_as_empty_space():
    """A white ground must be blank, not solid ink."""
    import numpy as np

    from ggssvt.eval.viz import ASCII_RAMP, to_text

    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[40:60, 40:60] = 0
    text = to_text(image, width=30)

    assert text.splitlines()[0].strip() == "", "the background is being inked"
    assert ASCII_RAMP[-1] in text, "the subject is not drawn"
