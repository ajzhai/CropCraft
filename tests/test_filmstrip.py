"""The shaded render has two failure modes that both look like a working render.

Neither raises. A mesh lit with inward normals comes back as a near-black
silhouette, and a splat radius smaller than the vertex spacing comes back as a
stipple. Both were shipped once, and both are only visible by looking at the
pixels, so these tests look at the pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.filmstrip import (
    N_VIEWS, _view_indices, _vertex_normals, shaded_render,
)
from ggssvt.geometry.mesh import Mesh


def _sphere(radius: float = 0.4, n: int = 40) -> Mesh:
    """A UV sphere, wound so its accumulated normals point inward.

    Inward on purpose: that is the orientation marching cubes hands us here, and
    a test built on an outward-wound mesh would pass whether or not the code
    corrects anything.
    """
    theta = np.linspace(0.0, np.pi, n)
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    vertices = np.column_stack([
        (radius * np.sin(t) * np.cos(p)).ravel(),
        (radius * np.sin(t) * np.sin(p)).ravel(),
        (radius * np.cos(t)).ravel(),
    ])

    faces = []
    for i in range(n - 1):
        for j in range(n):
            a = i * n + j
            b = i * n + (j + 1) % n
            c = (i + 1) * n + j
            d = (i + 1) * n + (j + 1) % n
            faces.append([a, c, b])      # winding chosen to face inward
            faces.append([b, c, d])
    return Mesh(vertices=vertices, faces=np.array(faces))


def test_vertex_normals_are_turned_outward():
    """A mesh wound inward still lights correctly."""
    mesh = _sphere()
    normals = _vertex_normals(mesh.vertices, mesh.faces)

    radial = mesh.vertices - mesh.vertices.mean(axis=0)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    agreement = (normals * radial).sum(axis=1)

    assert agreement.mean() > 0.9, "normals should point away from the centroid"


def test_the_render_is_lit_rather_than_ambient():
    """Inward normals put every visible pixel at the ambient term.

    The symptom is a flat near-black silhouette, so the test asks for range: a
    lit sphere has to show both a highlight and a shadowed side.
    """
    image = shaded_render(_sphere(), size=140, supersample=1)
    painted = image.reshape(-1, 3)
    ink = painted[painted.sum(axis=1) < 720]        # anything but the background
    assert ink.shape[0] > 400, "the sphere should cover a good part of the frame"

    brightness = ink.mean(axis=1)
    assert brightness.max() - brightness.min() > 40, "the surface is not lit"
    assert brightness.max() > 90, "even the highlight is at the ambient term"


def test_the_surface_closes_rather_than_stippling():
    """The splat radius has to follow the vertex spacing.

    Marching cubes rather than the UV sphere above, because the whole point of
    the radius calculation is to cope with *its* edge lengths: they run from near
    zero, where two vertices share a grid point, up to the voxel pitch. A radius
    that suits a uniform mesh proves nothing about this one.

    A stipple leaves background pixels scattered through the interior, so the
    test counts holes along a row that crosses the middle of the shape.
    """
    measure = pytest.importorskip(
        "skimage.measure", reason="the mesh arm needs scikit-image")
    assert measure is not None

    from ggssvt.geometry.mesh import mesh_from_occupancy

    resolution, voxel = 48, 0.012
    centres = (np.indices((resolution,) * 3).T - resolution / 2.0)
    ball = (centres ** 2).sum(axis=-1) < (resolution / 3.0) ** 2
    mesh = mesh_from_occupancy(np.ascontiguousarray(ball.T), voxel_size_m=voxel)

    size = 160
    image = shaded_render(mesh, size=size, supersample=1)
    ink = image.reshape(size, size, 3).sum(axis=2) < 720

    rows = np.flatnonzero(ink.any(axis=1))
    assert rows.size, "nothing was drawn"
    middle = ink[int(rows.mean())]
    gaps = int(np.count_nonzero(middle[:-1] & ~middle[1:]))

    assert gaps <= 1, f"the surface has {gaps} holes across its middle"


def test_views_span_the_turntable():
    """Six of twelve, evenly spaced, so the row is not one side of the plant."""
    picked = _view_indices(12, N_VIEWS)
    assert picked == [0, 2, 4, 6, 8, 10]

    assert _view_indices(4, N_VIEWS) == [0, 1, 2, 3], "fewer views than asked for"


def test_an_empty_mesh_renders_blank_rather_than_raising():
    """A specimen with nothing reconstructed should cost the tile, not the page."""
    empty = Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
    image = shaded_render(empty, size=48)

    assert image.shape == (48, 48, 3)
    assert (image == 255).all()
