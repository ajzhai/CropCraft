"""Export estimated rig poses in Nerfstudio format.

``rig_calibration/make_transforms.py`` was written to build a Nerfstudio
``transforms.json`` from ``rig_positions.json`` -- the output of
``calibrate_extrinsics.py``. That calibration was never captured, so the script
has never had an input. :mod:`ggssvt.geometry.rig` recovers the same poses from
the depth data, which is exactly what that file was missing.

This module writes both forms:

* ``transforms.json`` per specimen, ready for ``ns-train`` and ``ns-viewer``.
* ``rig_positions.json`` and per-camera intrinsics, so the existing
  ``make_transforms.py`` path also works unchanged.

**The coordinate convention is the part that goes wrong.** Nerfstudio's
``transform_matrix`` is camera-to-world in the OpenGL/Blender convention: ``+x``
right, ``+y`` **up**, ``-z`` forward. The poses here are OpenCV: ``+x`` right,
``+y`` **down**, ``+z`` forward -- the convention libfreenect2 reports depth in.
Converting means negating the second and third columns of the rotation. Skip it
and the scene trains upside down and back to front, which looks like a broken
reconstruction rather than a broken export.

**On view count.** Twelve views is thin for a radiance field; typical captures
use fifty to two hundred. Two things make it workable here: the depth maps can
supervise geometry directly (``depth-nerfacto``, or splatfacto with depth), and
Nerfstudio's camera optimiser can refine the poses photometrically. That second
point matters -- the azimuth corrections in
:mod:`ggssvt.geometry.refine` saturate their search bound, so letting a radiance
field refine the poses from image evidence is a genuinely independent estimate,
and worth comparing against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import DEPTH_SCALE_M, KINECT_V2, PLANTS_DIR, WORK_DIR, Intrinsics

# Nerfstudio/Blender cameras look down -z with +y up; OpenCV looks down +z with
# +y down. Negating the y and z basis vectors converts between them.
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def opencv_pose_to_nerfstudio(rotation: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Convert an OpenCV camera-to-world pose to Nerfstudio's convention.

    Args:
        rotation: ``(3, 3)`` world-from-camera rotation, OpenCV convention.
        centre: ``(3,)`` camera centre in world coordinates.

    Returns:
        ``(4, 4)`` camera-to-world matrix in the OpenGL/Blender convention.
    """
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(centre, dtype=np.float64)
    return transform @ OPENCV_TO_OPENGL


def nerfstudio_pose_to_opencv(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`opencv_pose_to_nerfstudio`, for round-trip checks."""
    restored = np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(OPENCV_TO_OPENGL)
    return restored[:3, :3], restored[:3, 3]


def intrinsics_dict(
    intrinsics: Intrinsics = KINECT_V2, *, height: int | None = None
) -> dict:
    """Nerfstudio intrinsics block for the Kinect v2 depth/colour frame."""
    return {
        "fl_x": float(intrinsics.fx),
        "fl_y": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "w": int(intrinsics.width),
        "h": int(height if height is not None else intrinsics.height),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def build_transforms(
    specimen,
    rig,
    *,
    intrinsics: Intrinsics = KINECT_V2,
    include_depth: bool = True,
    relative_to: Path | None = None,
) -> dict:
    """Assemble the ``transforms.json`` payload for one specimen.

    Args:
        specimen: a :class:`~ggssvt.data.dataset.Specimen`.
        rig: its :class:`~ggssvt.geometry.rig.RigSolution`.
        relative_to: directory the ``file_path`` entries are relative to.
            Defaults to the specimen's own directory, which is where Nerfstudio
            expects ``transforms.json`` to sit.

    Returns:
        A dict ready to serialise.
    """
    root = relative_to or specimen.root
    frames = []

    for view in specimen.views:
        pose = rig.pose(view.position_id)
        matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)

        frame = {
            "file_path": view.rgb_path.relative_to(root).as_posix(),
            "transform_matrix": [[float(v) for v in row] for row in matrix],
        }
        if include_depth:
            frame["depth_file_path"] = view.depth_path.relative_to(root).as_posix()
        frames.append(frame)

    payload = {
        "camera_model": "OPENCV",
        **intrinsics_dict(intrinsics),
        # The Kinect stores depth as uint16 millimetres.
        "depth_unit_scale_factor": DEPTH_SCALE_M,
        "frames": frames,
    }
    payload["ggssvt"] = {
        "note": (
            "Poses estimated from depth by ggssvt.geometry.rig, not measured by "
            "ChArUco calibration. Train with a camera optimiser enabled."
        ),
        "subject_distance_m": float(rig.subject_distance_m),
        "multiview_agreement": float(rig.agreement),
        "n_rig_warnings": len(rig.warnings),
    }
    return payload


def build_rig_positions(specimen, rig) -> dict:
    """The ``rig_positions.json`` that ``make_transforms.py`` expects.

    Keyed by position id, matching the ids that script looks up. Poses are in
    Nerfstudio convention already, since that is what the script copies straight
    into ``transform_matrix``.
    """
    positions = {}
    for view in specimen.views:
        pose = rig.pose(view.position_id)
        matrix = opencv_pose_to_nerfstudio(pose.rotation, pose.centre)
        positions[view.position_id] = {
            "camera": view.position.camera,
            "azimuth_deg": float(pose.azimuth_deg),
            "transform_matrix": [[float(v) for v in row] for row in matrix],
        }
    return positions


def export_specimen(
    plant_id: str,
    *,
    out_dir: Path | None = None,
    plants_dir: Path = PLANTS_DIR,
    intrinsics: Intrinsics = KINECT_V2,
    include_depth: bool = True,
    write_rig_positions: bool = True,
    seed: int = 0,
) -> dict[str, Path]:
    """Estimate the rig for one specimen and write its Nerfstudio files.

    By default ``transforms.json`` is written **into the specimen directory**,
    beside ``images/`` and ``depth/``, because Nerfstudio resolves ``file_path``
    relative to the transforms file.

    Returns:
        Paths of everything written, keyed by kind.
    """
    from ..data.dataset import load_specimen
    from ..geometry.rig import estimate_rig

    specimen = load_specimen(plant_id, plants_dir=plants_dir)
    rig = estimate_rig(specimen, seed=seed)

    target = out_dir or specimen.root
    target.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    transforms_path = target / "transforms.json"
    transforms_path.write_text(
        json.dumps(
            build_transforms(
                specimen,
                rig,
                intrinsics=intrinsics,
                include_depth=include_depth,
                relative_to=specimen.root,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    written["transforms"] = transforms_path

    if write_rig_positions:
        rig_path = target / "rig_positions.json"
        rig_path.write_text(
            json.dumps(build_rig_positions(specimen, rig), indent=2), encoding="utf-8"
        )
        written["rig_positions"] = rig_path

        for camera in ("camA", "camB"):
            path = target / f"{camera}_intrinsics.json"
            path.write_text(json.dumps(intrinsics_dict(intrinsics), indent=2), encoding="utf-8")
            written[f"{camera}_intrinsics"] = path

    return written


def export_dataset(
    plant_ids: list[str] | None = None,
    *,
    plants_dir: Path = PLANTS_DIR,
    in_place: bool = True,
    out_root: Path = WORK_DIR / "nerfstudio",
    include_depth: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict[str, dict[str, Path]]:
    """Export every specimen.

    Args:
        in_place: write into ``dataset/plants/<id>/``. Set False to write into
            ``out_root/<id>/`` instead, leaving the capture directories untouched
            -- but then the image paths in ``transforms.json`` point back out of
            that directory, so pass the specimen root to ``ns-train`` anyway.
    """
    from ..data.dataset import load_dataset

    if plant_ids is None:
        plant_ids = [s.plant_id for s in load_dataset(plants_dir=plants_dir)]

    results: dict[str, dict[str, Path]] = {}
    for index, plant_id in enumerate(plant_ids, start=1):
        out_dir = None if in_place else out_root / plant_id
        results[plant_id] = export_specimen(
            plant_id,
            out_dir=out_dir,
            plants_dir=plants_dir,
            include_depth=include_depth,
            **kwargs,
        )
        if verbose:
            print(f"[{index:2d}/{len(plant_ids)}] {plant_id} -> {results[plant_id]['transforms']}")

    return results


def training_commands(plant_id: str, plants_dir: Path = PLANTS_DIR) -> list[str]:
    """Suggested Nerfstudio commands for one exported specimen."""
    data = (plants_dir / plant_id).as_posix()
    return [
        "# Gaussian splatting, with the camera optimiser refining the estimated poses",
        f"ns-train splatfacto --data {data} \\",
        "    --pipeline.model.camera-optimizer.mode SO3xR3 \\",
        f"    --experiment-name {plant_id}",
        "",
        "# Depth-supervised NeRF: uses the Kinect depth, which helps at 12 views",
        f"ns-train depth-nerfacto --data {data} \\",
        "    --pipeline.model.camera-optimizer.mode SO3xR3 \\",
        f"    --experiment-name {plant_id}_depth",
        "",
        "# Interactive viewer for a trained run",
        f"ns-viewer --load-config outputs/{plant_id}/splatfacto/<timestamp>/config.yml",
        "",
        "# Export a point cloud to compare against the carved hull",
        f"ns-export pointcloud --load-config outputs/{plant_id}/splatfacto/<timestamp>/config.yml \\",
        f"    --output-dir exports/{plant_id}",
    ]


__all__ = [
    "OPENCV_TO_OPENGL",
    "build_rig_positions",
    "build_transforms",
    "export_dataset",
    "export_specimen",
    "intrinsics_dict",
    "nerfstudio_pose_to_opencv",
    "opencv_pose_to_nerfstudio",
    "training_commands",
]
