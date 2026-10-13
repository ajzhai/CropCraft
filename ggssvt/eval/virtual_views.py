"""Score the reconstruction against a plant whose true shape is known.

Every reconstruction claim in this project has been argued rather than measured,
because no reference geometry for our own specimens exists. The implied bulk
density criterion was built to stand in for one (§7b), and §7f goes further and
argues that silhouette IoU ranks our reconstructions *backwards*, which rests
on the density screen disagreeing with the metric, an inference, not a
demonstration.

Pheno4D removes the excuse. Take a laser-scanned plant, render twelve virtual
views of it at our azimuths and through our camera model, run **our** carve and
**our** fusion on those views, and score the results against the cloud they came
from. Nothing here is a reimplementation: `geometry.carving.carve` and
`geometry.fusion.fuse` are called exactly as the pipeline calls them.

**What this can settle that nothing else can.**

*Whether silhouette IoU is misleading.* Both reconstructions are scored two ways
-- against the truth, and by reprojection into their own input silhouettes. If
the reprojection metric orders them opposite to the truth, §7f stops being an
inference.

*What the density criterion is really testing.* With mass fixed, implied density
is mass over reconstructed volume, so a band on density is a band on the **volume
ratio** and nothing else: passing 200-1000 kg/m³ at a tissue density ρ means the
reconstruction's volume lies between ρ/1000 and ρ/200 times the true volume. At
ρ = 600 that is a window of 0.6x to 3.0x. The screen has never been checked
against a known volume; here it can be.

**What this cannot settle.** The virtual views are clean: exact poses, no sensor
noise, no segmentation error, no missing returns. A reconstruction that fails
here fails for geometric reasons alone, which makes this an upper bound on our
real performance rather than an estimate of it. That is the right direction for
the argument: an operator that cannot recover a plant from perfect views will
not recover one from Kinect returns, but it is not a claim about our captures.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import KINECT_V2, VOXEL_RESOLUTION, VOXEL_SIZE_M, WORK_DIR, Intrinsics

# The rig protocol: twelve azimuths thirty degrees apart.
N_VIEWS = 12

# Where the cameras sit. Taken from the rig summaries of our own captures rather
# than chosen: about 1.4 m out and 1.0 m up, tilted down at the subject.
CAMERA_DISTANCE_M = 1.4
CAMERA_HEIGHT_M = 1.0

# Each point is splatted over this radius in pixels. A subsampled cloud leaves
# pin-holes in the depth map otherwise, and a hole reads to the carve as
# background, free space, which would eat the plant for a reason that has
# nothing to do with the operator under test.
SPLAT_RADIUS = 1

# Tissue density assumed when checking what the implied-density screen would do.
# Any value inside the band works; the screen's verdict is a volume ratio, and
# this only fixes where the window sits.
ASSUMED_DENSITY_KG_M3 = 600.0


@dataclass
class Rendered:
    """Twelve synthetic views of one scan."""

    depth_m: np.ndarray         # (V, H, W) float32, 0 where nothing was hit
    mask: np.ndarray            # (V, H, W) bool
    rotation: np.ndarray        # (V, 3, 3) world_from_cam
    centre: np.ndarray          # (V, 3)
    azimuth_deg: np.ndarray     # (V,)


def camera_poses(
    *,
    n_views: int = N_VIEWS,
    distance_m: float = CAMERA_DISTANCE_M,
    height_m: float = CAMERA_HEIGHT_M,
    target_z_m: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cameras on a circle, each looking at a point on the subject axis.

    Returns ``world_from_cam`` rotations, camera centres, and the azimuths, in
    the convention the rest of the pipeline uses: ``x_world = R @ x_cam + c``,
    with the camera looking along its own +z and image rows increasing downward.
    """
    azimuths = np.arange(n_views) * (360.0 / n_views)
    rotations = np.zeros((n_views, 3, 3))
    centres = np.zeros((n_views, 3))

    target = np.array([0.0, 0.0, target_z_m])
    world_up = np.array([0.0, 0.0, 1.0])

    for view, azimuth in enumerate(azimuths):
        angle = np.deg2rad(azimuth)
        centre = np.array([
            distance_m * np.cos(angle), distance_m * np.sin(angle), height_m,
        ])
        forward = target - centre
        forward /= np.linalg.norm(forward)

        right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)

        rotations[view] = np.column_stack([right, down, forward])
        centres[view] = centre

    return rotations, centres, azimuths


def render(
    points: np.ndarray,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    splat_radius: int = SPLAT_RADIUS,
    **pose_kwargs,
) -> Rendered:
    """Z-buffer a point cloud into depth maps and silhouettes.

    Nearest point wins each pixel, which is what a depth sensor reports: a leaf
    in front hides what is behind it. Self-occlusion is therefore modelled, and
    it is most of what makes a plant hard to reconstruct.
    """
    rotations, centres, azimuths = camera_poses(**pose_kwargs)
    n_views = rotations.shape[0]
    height, width = intrinsics.height, intrinsics.width

    depth = np.zeros((n_views, height, width), dtype=np.float32)

    for view in range(n_views):
        # World to camera: the inverse of x_world = R @ x_cam + c.
        cam = (points - centres[view]) @ rotations[view]
        z = cam[:, 2]
        in_front = z > 1e-6
        if not in_front.any():
            continue

        u = cam[in_front, 0] * intrinsics.fx / z[in_front] + intrinsics.cx
        v = cam[in_front, 1] * intrinsics.fy / z[in_front] + intrinsics.cy
        zi = z[in_front]

        buffer = np.full((height, width), np.inf, dtype=np.float32)
        for dv in range(-splat_radius, splat_radius + 1):
            for du in range(-splat_radius, splat_radius + 1):
                cols = np.round(u + du).astype(np.int64)
                rows = np.round(v + dv).astype(np.int64)
                on = (
                    (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                )
                if not on.any():
                    continue
                np.minimum.at(buffer, (rows[on], cols[on]), zi[on])

        buffer[~np.isfinite(buffer)] = 0.0
        depth[view] = buffer

    return Rendered(
        depth_m=depth,
        mask=depth > 0.0,
        rotation=rotations,
        centre=centres,
        azimuth_deg=azimuths,
    )


def _rig_and_segmentations(scan_id: str, rendered: Rendered):
    """Wrap the render in the structures `carve` expects, so the real code runs."""
    from ..geometry.rig import RigSolution, ViewPose
    from ..geometry.segment import ViewSegmentation

    poses, segmentations = {}, {}
    for view in range(rendered.depth_m.shape[0]):
        position = f"v{view:02d}"
        poses[position] = ViewPose(
            position_id=position,
            azimuth_deg=float(rendered.azimuth_deg[view]),
            rotation=rendered.rotation[view],
            centre=rendered.centre[view],
            camera_height_m=float(rendered.centre[view, 2]),
            tilt_deg=0.0,
            subject_distance_m=float(
                np.linalg.norm(rendered.centre[view, :2])),
            floor_inlier_fraction=1.0,
        )
        segmentations[position] = ViewSegmentation(
            position_id=position,
            mask=rendered.mask[view],
            depth_m=rendered.depth_m[view],
            points_world=np.zeros((0, 3), dtype=np.float32),
            colours=None,
        )

    rig = RigSolution(
        plant_id=scan_id, poses=poses, warnings=[],
        subject_distance_m=CAMERA_DISTANCE_M, agreement=1.0,
    )
    return rig, segmentations


def voxel_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def silhouette_iou(occupancy: np.ndarray, rendered: Rendered, *,
                   voxel_size_m: float = VOXEL_SIZE_M,
                   intrinsics: Intrinsics = KINECT_V2) -> float:
    """Agreement between a reconstruction's reprojection and its own input masks.

    The metric §7f is about. A visual hull agrees with the silhouettes it was
    carved from by construction, so this rewards exactly the failure it should
    catch, which is the claim this function exists to test rather than assert.
    """
    from ..geometry.carving import voxel_grid_centres

    resolution = occupancy.shape[0]
    centres = voxel_grid_centres(
        resolution=resolution, voxel_size_m=voxel_size_m)[occupancy]
    if centres.size == 0:
        return 0.0

    scores = []
    for view in range(rendered.depth_m.shape[0]):
        cam = (centres - rendered.centre[view]) @ rendered.rotation[view]
        z = cam[:, 2]
        in_front = z > 1e-6
        if not in_front.any():
            continue
        u = np.round(
            cam[in_front, 0] * intrinsics.fx / z[in_front] + intrinsics.cx
        ).astype(np.int64)
        v = np.round(
            cam[in_front, 1] * intrinsics.fy / z[in_front] + intrinsics.cy
        ).astype(np.int64)
        on = (
            (v >= 0) & (v < intrinsics.height) & (u >= 0) & (u < intrinsics.width)
        )
        projected = np.zeros((intrinsics.height, intrinsics.width), dtype=bool)
        projected[v[on], u[on]] = True
        scores.append(voxel_iou(projected, rendered.mask[view]))

    return float(np.mean(scores)) if scores else 0.0


def _downsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """Any occupied fine voxel makes its coarse parent occupied."""
    resolution = volume.shape[0] // factor
    trimmed = volume[:resolution * factor, :resolution * factor,
                     :resolution * factor]
    return trimmed.reshape(
        resolution, factor, resolution, factor, resolution, factor
    ).any(axis=(1, 3, 5))


@dataclass
class ScanResult:
    """One plant, reconstructed two ways and scored against its own truth."""

    plant_id: str
    scan_id: str
    species: str
    n_points: int
    n_organs: int
    height_m: float
    true_volume_l: float
    carve_volume_l: float
    fused_volume_l: float
    carve_iou: float
    fused_iou: float
    carve_silhouette_iou: float
    fused_silhouette_iou: float
    carve_volume_ratio: float
    fused_volume_ratio: float
    carve_density_passes: bool
    fused_density_passes: bool
    inverted: bool              # silhouette IoU ranks them opposite to the truth

    # The grids themselves, kept so a caller can score them another way without
    # carving the scan a second time. Excluded from as_dict because a boolean
    # volume is not JSON and the report is the same report it always was.
    carve_occupancy: np.ndarray | None = field(default=None, repr=False)
    fused_occupancy: np.ndarray | None = field(default=None, repr=False)
    truth_occupancy: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict:
        skip = {"carve_occupancy", "fused_occupancy", "truth_occupancy"}
        return {k: v for k, v in asdict(self).items() if k not in skip}


def reconstruct(scan, *, verbose: bool = True) -> ScanResult:
    """Render, carve, fuse, and score one scan against its own point cloud."""
    from ..data.pheno4d import voxelise
    from ..geometry.carving import carve, largest_connected_component
    from ..geometry.fusion import FUSION_VOXEL_M, fuse

    voxel_volume_l = VOXEL_SIZE_M ** 3 * 1000.0
    truth = voxelise(
        scan.points, resolution=VOXEL_RESOLUTION, voxel_size_m=VOXEL_SIZE_M)
    true_volume_l = float(truth.sum()) * voxel_volume_l

    rendered = render(scan.points, target_z_m=max(scan.height_m / 2.0, 0.1))
    rig, segmentations = _rig_and_segmentations(scan.scan_id, rendered)

    carved = carve(rig, segmentations, plant_id=scan.scan_id)
    carve_occupancy = largest_connected_component(carved.occupancy)

    fused = fuse(
        rendered.depth_m, rendered.rotation, rendered.centre,
        mask=rendered.mask)
    factor = round(VOXEL_SIZE_M / FUSION_VOXEL_M)
    fused_occupancy = largest_connected_component(
        _downsample(fused.interior, factor))

    carve_volume_l = float(carve_occupancy.sum()) * voxel_volume_l
    fused_volume_l = float(fused_occupancy.sum()) * voxel_volume_l

    carve_iou = voxel_iou(carve_occupancy, truth)
    fused_iou = voxel_iou(fused_occupancy, truth)
    carve_sil = silhouette_iou(carve_occupancy, rendered)
    fused_sil = silhouette_iou(fused_occupancy, rendered)

    # The screen, expressed as what it actually is: a window on the volume ratio.
    mass_kg = ASSUMED_DENSITY_KG_M3 * true_volume_l / 1000.0
    def passes(volume_l: float) -> bool:
        if volume_l <= 0:
            return False
        density = mass_kg / (volume_l / 1000.0)
        return 200.0 <= density <= 1000.0

    result = ScanResult(
        plant_id=scan.plant_id,
        scan_id=scan.scan_id,
        species=scan.species,
        n_points=scan.n_points,
        n_organs=scan.n_organs,
        height_m=round(scan.height_m, 4),
        true_volume_l=round(true_volume_l, 4),
        carve_volume_l=round(carve_volume_l, 4),
        fused_volume_l=round(fused_volume_l, 4),
        carve_iou=round(carve_iou, 4),
        fused_iou=round(fused_iou, 4),
        carve_silhouette_iou=round(carve_sil, 4),
        fused_silhouette_iou=round(fused_sil, 4),
        carve_volume_ratio=round(carve_volume_l / max(true_volume_l, 1e-9), 3),
        fused_volume_ratio=round(fused_volume_l / max(true_volume_l, 1e-9), 3),
        carve_density_passes=passes(carve_volume_l),
        fused_density_passes=passes(fused_volume_l),
        inverted=bool((fused_iou > carve_iou) != (fused_sil > carve_sil)),
        carve_occupancy=carve_occupancy,
        fused_occupancy=fused_occupancy,
        truth_occupancy=truth,
    )
    if verbose:
        print(f"  {result.plant_id:9s} true {true_volume_l:6.3f} L | "
              f"carve {carve_volume_l:7.3f} L IoU {carve_iou:.3f} sil {carve_sil:.3f} | "
              f"fusion {fused_volume_l:6.3f} L IoU {fused_iou:.3f} sil {fused_sil:.3f}"
              + ("   INVERTED" if result.inverted else ""))
    return result


def run(
    *,
    root: Path | None = None,
    out: Path = WORK_DIR / "reports" / "virtual_views.json",
    subsample: int = 2,
    limit: int | None = None,
    verbose: bool = True,
) -> dict:
    """Reconstruct the last scan of every plant and score it against the truth."""
    from ..data.pheno4d import DATASET_DIR, latest_per_plant, load_scan

    paths = latest_per_plant(root or DATASET_DIR)
    if limit:
        paths = paths[:limit]

    results = []
    for path in paths:
        scan = load_scan(path, subsample=subsample)
        results.append(reconstruct(scan, verbose=verbose))

    n = len(results)
    inverted = sum(r.inverted for r in results)
    truth_prefers_fusion = sum(r.fused_iou > r.carve_iou for r in results)
    metric_prefers_fusion = sum(
        r.fused_silhouette_iou > r.carve_silhouette_iou for r in results)

    # Paired: the same plant scored two ways. The exact test over the plants
    # where truth and metric disagree is the same one the operator comparison
    # uses in FINDINGS 7m, applied here to the metric rather than the operator.
    from .batch_holdout import mcnemar
    agreement = mcnemar(
        [r.fused_iou > r.carve_iou for r in results],
        [r.fused_silhouette_iou > r.carve_silhouette_iou for r in results],
    )

    summary = {
        "n_scans": n,
        "metric_vs_truth": agreement,
        "truth_prefers_fusion": truth_prefers_fusion,
        "silhouette_metric_prefers_fusion": metric_prefers_fusion,
        "disagreements": inverted,
        "mean_carve_iou": round(float(np.mean([r.carve_iou for r in results])), 4),
        "mean_fused_iou": round(float(np.mean([r.fused_iou for r in results])), 4),
        "mean_carve_silhouette_iou": round(
            float(np.mean([r.carve_silhouette_iou for r in results])), 4),
        "mean_fused_silhouette_iou": round(
            float(np.mean([r.fused_silhouette_iou for r in results])), 4),
        "median_carve_volume_ratio": round(
            float(np.median([r.carve_volume_ratio for r in results])), 3),
        "median_fused_volume_ratio": round(
            float(np.median([r.fused_volume_ratio for r in results])), 3),
        "carve_passes_density": sum(r.carve_density_passes for r in results),
        "fused_passes_density": sum(r.fused_density_passes for r in results),
    }

    if verbose:
        print(f"\n  truth prefers fusion on {truth_prefers_fusion}/{n}; "
              f"silhouette IoU prefers it on {metric_prefers_fusion}/{n}; "
              f"they disagree on {inverted}/{n}")
        print(f"  median volume ratio: carve {summary['median_carve_volume_ratio']}x "
              f"true, fusion {summary['median_fused_volume_ratio']}x true")
        print(f"  the density screen would pass carve on "
              f"{summary['carve_passes_density']}/{n} and fusion on "
              f"{summary['fused_passes_density']}/{n}")

    report = {
        "dataset": "Pheno4D (Schunck et al., PLOS ONE 2021)",
        "note": "virtual views: exact poses, no sensor noise, no segmentation "
                "error. An upper bound on the operator, not an estimate of our "
                "captures",
        "assumed_density_kg_m3": ASSUMED_DENSITY_KG_M3,
        "n_views": N_VIEWS,
        "voxel_size_m": VOXEL_SIZE_M,
        "summary": summary,
        "rows": [r.as_dict() for r in results],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_meshes(
    plant: str = "Maize01",
    *,
    out_dir: Path | None = None,
    subsample: int = 2,
    smoothing: int = 2,
    verbose: bool = True,
) -> dict[str, Path]:
    """Write the truth, the carve and the fusion of one plant as OBJ files.

    Deliberately a format nothing here is precious about. `.obj` is what
    MeshLab, Blender, three.js and EasyPBR all read, so exporting it decouples
    the question of *what we reconstructed* from the question of *what renders
    it*. The figures in `eval/plots.py` stay dependency-free and deterministic;
    anything prettier can start from these files.

    The truth is meshed from the same voxelisation the scoring uses rather than
    from the raw points, so the three meshes are commensurable: any volume
    difference between them is the operator's, not the mesher's.
    """
    from ..data.pheno4d import DATASET_DIR, latest_per_plant, load_scan, voxelise
    from ..geometry.carving import carve, largest_connected_component
    from ..geometry.fusion import FUSION_VOXEL_M, fuse
    from ..geometry.mesh import mesh_from_occupancy

    out_dir = out_dir or WORK_DIR / "reports" / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = next(p for p in latest_per_plant(DATASET_DIR) if p.parent.name == plant)
    scan = load_scan(path, subsample=subsample)

    rendered = render(scan.points, target_z_m=max(scan.height_m / 2.0, 0.1))
    rig, segmentations = _rig_and_segmentations(scan.scan_id, rendered)

    grids = {
        "truth": voxelise(scan.points, resolution=VOXEL_RESOLUTION,
                          voxel_size_m=VOXEL_SIZE_M),
        "carve": largest_connected_component(
            carve(rig, segmentations, plant_id=scan.scan_id).occupancy),
        "fusion": largest_connected_component(_downsample(
            fuse(rendered.depth_m, rendered.rotation, rendered.centre,
                 mask=rendered.mask).interior,
            round(VOXEL_SIZE_M / FUSION_VOXEL_M))),
    }

    written: dict[str, Path] = {}
    for name, grid in grids.items():
        mesh = mesh_from_occupancy(grid, voxel_size_m=VOXEL_SIZE_M,
                                   smoothing=smoothing)
        target = out_dir / f"{plant.lower()}_{name}.obj"
        target.write_text(mesh.to_obj(), encoding="utf-8")
        written[name] = target
        if verbose:
            print(f"  {name:7s} {mesh.n_vertices:6d} vertices  "
                  f"{mesh.n_faces:6d} faces  -> {target.name}")
    return written


def export_clouds(
    plants: str | list[str] | None = None,
    *,
    out: Path | None = None,
    subsample: int = 2,
    verbose: bool = True,
) -> dict:
    """Encode the truth, the carve and the fusion for the project page.

    Same encoding the specimen viewer already uses, occupied voxel indices as
    byte triples, deflated and base64'd, so the page's existing renderer draws
    these with no new code beyond a panel that shares one camera across three
    canvases. Sharing the camera is the point: read side by side at the same
    angle, the hull is visibly swallowing the gaps between the leaves, which no
    still figure and no pair of numbers conveys as directly.
    """
    from ..data.pheno4d import DATASET_DIR, latest_per_plant, voxelise
    from ..geometry.carving import carve, largest_connected_component
    from ..geometry.fusion import FUSION_VOXEL_M, fuse
    from .dashboard_data import _quantise

    available = latest_per_plant(DATASET_DIR)
    if plants is None:
        wanted = [p.parent.name for p in available]
    elif isinstance(plants, str):
        wanted = [plants]
    else:
        wanted = list(plants)

    entries = []
    for name in wanted:
        entries.append(_one_plant(
            name, available, subsample=subsample, verbose=verbose,
            voxelise=voxelise, carve=carve,
            largest_connected_component=largest_connected_component,
            fuse=fuse, fusion_voxel_m=FUSION_VOXEL_M, quantise=_quantise,
        ))

    payload = {
        "source": "Pheno4D (Schunck et al., PLOS ONE 2021)",
        "n_views": N_VIEWS,
        "voxel_size_m": VOXEL_SIZE_M,
        "plants": entries,
    }
    out = out or WORK_DIR / "reports" / "reconstruction_clouds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    if verbose:
        print(f"  wrote {out} ({out.stat().st_size // 1024} KB, "
              f"{len(entries)} plants)")
    return payload


def _one_plant(
    plant, available, *, subsample, verbose,
    voxelise, carve, largest_connected_component, fuse, fusion_voxel_m, quantise,
) -> dict:
    """Reconstruct one plant three ways and encode all three for the page."""
    from ..data.pheno4d import load_scan

    path = next(p for p in available if p.parent.name == plant)
    scan = load_scan(path, subsample=subsample)

    rendered = render(scan.points, target_z_m=max(scan.height_m / 2.0, 0.1))
    rig, segmentations = _rig_and_segmentations(scan.scan_id, rendered)

    grids = {
        "truth": voxelise(scan.points, resolution=VOXEL_RESOLUTION,
                          voxel_size_m=VOXEL_SIZE_M),
        "carve": largest_connected_component(
            carve(rig, segmentations, plant_id=scan.scan_id).occupancy),
        "fusion": largest_connected_component(_downsample(
            fuse(rendered.depth_m, rendered.rotation, rendered.centre,
                 mask=rendered.mask).interior,
            round(VOXEL_SIZE_M / fusion_voxel_m))),
    }

    litres = VOXEL_SIZE_M ** 3 * 1000.0
    true_volume_l = float(grids["truth"].sum()) * litres
    labels = {
        "truth": ("The plant", "laser scan, voxelised at 12 mm"),
        "carve": ("Silhouette carving", "twelve views, visual hull"),
        "fusion": ("Depth fusion", "twelve views, truncated signed distance"),
    }

    arms = []
    for name, grid in grids.items():
        volume_l = float(grid.sum()) * litres
        title, detail = labels[name]
        arms.append({
            "key": name,
            "title": title,
            "detail": detail,
            "volume_l": round(volume_l, 3),
            "ratio": None if name == "truth" else round(
                volume_l / max(true_volume_l, 1e-9), 2),
            "iou": None if name == "truth" else round(
                voxel_iou(grid, grids["truth"]), 3),
            "silhouette_iou": None if name == "truth" else round(
                silhouette_iou(grid, rendered), 3),
            # No downsampling. The specimen viewer halves resolution because a
            # carved Eucalyptus is tens of thousands of voxels; a maize plant at
            # 12 mm is 342, and halving again leaves 114, which reads as
            # scattered dust rather than a plant.
            "cloud": quantise(grid, downsample=1),
        })
    if verbose:
        ratios = "  ".join(
            f"{a['key']} {a['volume_l']:.2f} L" for a in arms)
        print(f"  {plant:9s} {ratios}")

    return {
        "plant": plant,
        "species": scan.species,
        "n_organs": scan.n_organs,
        "height_m": round(scan.height_m, 3),
        "true_volume_l": round(true_volume_l, 3),
        "arms": arms,
    }


__all__ = [
    "ASSUMED_DENSITY_KG_M3", "CAMERA_DISTANCE_M", "CAMERA_HEIGHT_M", "N_VIEWS",
    "Rendered", "ScanResult", "camera_poses", "export_clouds", "export_meshes",
    "reconstruct",
    "render", "run",
    "silhouette_iou", "voxel_iou",
]
