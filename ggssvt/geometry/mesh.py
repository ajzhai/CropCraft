"""Meshes from the carved occupancy, and the shape descriptors they expose.

The voxel hull answers "how much space does the plant occupy". A mesh answers
questions the voxel grid structurally cannot, and one of them matters for
biomass: **surface area**.

A leaf contributes mass roughly in proportion to its area and almost nothing to
enclosed volume. A carved hull records leaves as a thin shell whose volume is an
artefact of the voxel size, so a volume-only feature set is blind to exactly the
tissue that dominates a leafy canopy's mass. Mesh surface area is not blind to
it. That makes an area-based biomass feature a real hypothesis rather than a
variation, and it is directly testable against the existing comparison.

Two more things fall out of having a mesh:

* **Enclosed volume by the divergence theorem**, which disagrees with the voxel
  count in a structured way. Voxel volume overcounts thin structures (a 6 mm stem
  still occupies a 12 mm voxel); mesh volume interpolates the surface through the
  voxel. The gap between them is another instance of the surface-versus-volume
  disagreement this project is about, measured rather than argued.
* **Solidity** -- the ratio of the mesh volume to its convex hull volume. A
  compact mango canopy and a sparse eucalyptus sapling can share a volume while
  differing completely in solidity.

Marching cubes on a binary grid overestimates area by roughly 8% (validated
against an analytic sphere in the tests). That bias is consistent and monotone
across specimens, so it does not affect a regression feature, but it should not
be quoted as an absolute measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import POT_HEIGHT_M, VOXEL_SIZE_M


class MeshError(RuntimeError):
    """Raised when a mesh cannot be extracted."""


def _require_skimage():
    try:
        from skimage import measure
    except ImportError as exc:  # pragma: no cover
        raise MeshError(
            "the mesh arm needs scikit-image; install it with "
            "`pip install scikit-image`. The rest of the pipeline does not."
        ) from exc
    return measure


@dataclass
class Mesh:
    """A triangle mesh in world coordinates, metres."""

    vertices: np.ndarray   # (V, 3) float64
    faces: np.ndarray      # (F, 3) int

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.n_faces == 0

    def surface_area_m2(self) -> float:
        """Total triangle area."""
        if self.is_empty:
            return 0.0
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    def enclosed_volume_m3(self) -> float:
        """Volume enclosed by the surface, via the divergence theorem.

        Sums the signed volumes of the tetrahedra formed by each triangle and
        the origin. Exact for a closed, consistently oriented mesh. The absolute
        value is taken because marching cubes' winding direction depends on
        whether the level set is entered from inside or outside, and that choice
        carries no meaning here.
        """
        if self.is_empty:
            return 0.0
        a, b, c = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty:
            zero = np.zeros(3)
            return zero, zero
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def height_m(self) -> float:
        return float(self.bounds()[1][2]) if not self.is_empty else 0.0

    def crop_above(self, z_m: float) -> Mesh:
        """Keep only the faces whose centroid sits above a height.

        A crude cut rather than a proper plane clip: it leaves the mesh open at
        the cut, so ``enclosed_volume_m3`` is no longer meaningful on the result.
        Use it for area, which is well defined either way.
        """
        if self.is_empty:
            return self
        centroids = self.vertices[self.faces].mean(axis=1)
        keep = centroids[:, 2] > z_m
        return Mesh(vertices=self.vertices, faces=self.faces[keep])

    def to_obj(self) -> str:
        """Wavefront OBJ, for MeshLab or Blender."""
        lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in self.vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces]
        return "\n".join(lines) + "\n"


def mesh_from_occupancy(
    occupancy: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    smoothing: int = 0,
) -> Mesh:
    """Extract a mesh from a boolean occupancy grid by marching cubes.

    Args:
        occupancy: ``(R, R, R)`` boolean grid.
        voxel_size_m: edge length of one voxel.
        smoothing: Laplacian smoothing passes. Off by default -- smoothing
            shrinks the surface and changes both area and volume, so it must be
            an explicit choice rather than a silent default.

    Returns:
        A :class:`Mesh` in the same world frame as
        :func:`ggssvt.config.voxel_grid_centres`.
    """
    measure = _require_skimage()

    if not occupancy.any():
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))

    # Pad so structures touching the grid boundary still close.
    padded = np.pad(occupancy.astype(np.float32), 1, constant_values=0.0)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices -= 1.0                                     # undo the pad

    if smoothing:
        vertices = laplacian_smooth(vertices, faces, iterations=smoothing)

    resolution = occupancy.shape[0]
    half = resolution * voxel_size_m / 2.0
    world = np.empty_like(vertices, dtype=np.float64)
    world[:, 0] = vertices[:, 0] * voxel_size_m - half + voxel_size_m / 2
    world[:, 1] = vertices[:, 1] * voxel_size_m - half + voxel_size_m / 2
    world[:, 2] = vertices[:, 2] * voxel_size_m + voxel_size_m / 2

    return Mesh(vertices=world, faces=faces.astype(np.int64))


def laplacian_smooth(
    vertices: np.ndarray, faces: np.ndarray, *, iterations: int = 1, weight: float = 0.5
) -> np.ndarray:
    """Umbrella-operator smoothing. Reduces the marching-cubes staircase."""
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])

    counts = np.bincount(edges[:, 0], minlength=vertices.shape[0]).astype(np.float64)
    counts[counts == 0] = 1.0

    for _ in range(iterations):
        summed = np.zeros_like(vertices)
        np.add.at(summed, edges[:, 0], vertices[edges[:, 1]])
        vertices += weight * (summed / counts[:, None] - vertices)

    return vertices


def convex_hull_volume_m3(points: np.ndarray) -> float:
    """Volume of the convex hull of a point set, or NaN if degenerate."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return float("nan")

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 4:
        return float("nan")
    try:
        return float(ConvexHull(points).volume)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class MeshMetrics:
    """Shape descriptors derived from a mesh."""

    surface_area_m2: float
    canopy_area_m2: float          # above the pot rim
    enclosed_volume_m3: float
    voxel_volume_m3: float
    convex_hull_volume_m3: float
    height_m: float
    n_faces: int

    @property
    def area_to_volume(self) -> float:
        """High for thin, leafy structure; low for a compact blob."""
        if self.enclosed_volume_m3 <= 0:
            return 0.0
        return self.surface_area_m2 / self.enclosed_volume_m3

    @property
    def solidity(self) -> float:
        """Mesh volume as a fraction of its convex hull. Sparse canopies are low."""
        if not np.isfinite(self.convex_hull_volume_m3) or self.convex_hull_volume_m3 <= 0:
            return float("nan")
        return self.enclosed_volume_m3 / self.convex_hull_volume_m3

    @property
    def voxel_to_mesh_volume_ratio(self) -> float:
        """How much the voxel count overstates the interpolated volume.

        Above 1 means the grid is inflating thin structure -- a 6 mm stem still
        claims a whole 12 mm voxel. This is the surface-versus-volume
        disagreement in its most direct form, and it should be largest on the
        thin eucalyptus specimens.
        """
        if self.enclosed_volume_m3 <= 0:
            return float("nan")
        return self.voxel_volume_m3 / self.enclosed_volume_m3

    def as_dict(self) -> dict[str, float]:
        return {
            "surface_area_m2": self.surface_area_m2,
            "canopy_area_m2": self.canopy_area_m2,
            "enclosed_volume_m3": self.enclosed_volume_m3,
            "voxel_volume_m3": self.voxel_volume_m3,
            "convex_hull_volume_m3": self.convex_hull_volume_m3,
            "height_m": self.height_m,
            "area_to_volume": self.area_to_volume,
            "solidity": self.solidity,
            "voxel_to_mesh_volume_ratio": self.voxel_to_mesh_volume_ratio,
            "n_faces": float(self.n_faces),
        }


def mesh_metrics(
    occupancy: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    pot_height_m: float = POT_HEIGHT_M,
    smoothing: int = 0,
) -> tuple[Mesh, MeshMetrics]:
    """Extract a mesh and its shape descriptors from a carved volume."""
    mesh = mesh_from_occupancy(occupancy, voxel_size_m=voxel_size_m, smoothing=smoothing)

    return mesh, MeshMetrics(
        surface_area_m2=mesh.surface_area_m2(),
        canopy_area_m2=mesh.crop_above(pot_height_m).surface_area_m2(),
        enclosed_volume_m3=mesh.enclosed_volume_m3(),
        voxel_volume_m3=float(occupancy.sum()) * voxel_size_m ** 3,
        convex_hull_volume_m3=convex_hull_volume_m3(mesh.vertices),
        height_m=mesh.height_m(),
        n_faces=mesh.n_faces,
    )


__all__ = [
    "Mesh",
    "MeshError",
    "MeshMetrics",
    "convex_hull_volume_m3",
    "laplacian_smooth",
    "mesh_from_occupancy",
    "mesh_metrics",
]
