"""Capture Kinect v2 RGB-D sweeps with a stationary camera and rotating plant.

For registered camera-to-turntable transforms, mount a known-size checkerboard
rigidly on the turntable, keep it visible in every frame, and pass its inner
corner count and square size. Capture outputs are stored separately from the
existing dual-Kinect dataset.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ggssvt.config import KINECT_V2  # noqa: E402
from rig_calibration import collect_specimen as kinect  # noqa: E402

DATASET_ROOT = REPO_ROOT / "dataset_kinect_stationary"
PLANTS_DIR = DATASET_ROOT / "plants"
BACKGROUND_DIR = DATASET_ROOT / "background"
CAMERA_LABEL = "camA"
KINECT_INTRINSICS = KINECT_V2


def validate_plant_id(value: str) -> str:
    plant_id = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", plant_id):
        raise ValueError("Use letters, numbers, underscores, or hyphens only.")
    return plant_id


def validate_step_degrees(step_degrees: int) -> int:
    if step_degrees <= 0 or 360 % step_degrees:
        raise ValueError("Turntable step must be a positive divisor of 360 degrees.")
    return step_degrees


def validate_actual_angle(value: str | float, *, frame_index: int, step_degrees: int) -> float:
    try:
        angle = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter the measured turntable angle in degrees.") from exc
    if not math.isfinite(angle) or angle < 0 or angle >= 360:
        raise ValueError("Angle must be finite and in the range [0, 360).")
    expected = frame_index * step_degrees
    error = abs((angle - expected + 180.0) % 360.0 - 180.0)
    if error > step_degrees * 0.6:
        raise ValueError(
            f"Measured angle {angle:g} deg is too far from expected {expected} deg; "
            "check the turntable before capturing."
        )
    return angle


def checkerboard_object_points(
    inner_corners: tuple[int, int], square_size_m: float
) -> np.ndarray:
    columns, rows = inner_corners
    if columns < 2 or rows < 2:
        raise ValueError("Checkerboard needs at least 2 inner corners in each direction.")
    if not math.isfinite(square_size_m) or square_size_m <= 0:
        raise ValueError("Checkerboard square size must be a positive number of metres.")
    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size_m
    )
    return points


def checkerboard_pose(
    rgb_bgr: np.ndarray,
    *,
    inner_corners: tuple[int, int],
    square_size_m: float,
) -> dict[str, Any] | None:
    """Estimate camera-from-board pose from the registered Kinect color frame."""
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, inner_corners, flags)
    if not found or corners is None:
        return None
    corners, orientation_marker_detected = _orient_checkerboard_corners(
        rgb_bgr, corners, inner_corners
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (5, 5),
        (-1, -1),
        criteria,
    )
    object_points = checkerboard_object_points(inner_corners, square_size_m)
    camera_matrix = KINECT_INTRINSICS.matrix
    solved, rvec, tvec = cv2.solvePnP(
        object_points,
        refined,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved:
        return None

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    reprojection_error = float(
        np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - refined.reshape(-1, 2)) ** 2, axis=1)))
    )
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = tvec.reshape(3)
    return {
        "camera_from_turntable_board": transform.tolist(),
        "reprojection_rmse_px": reprojection_error,
        "detected_inner_corners": int(len(refined)),
        "orientation_marker_detected": orientation_marker_detected,
    }


def _orient_checkerboard_corners(
    rgb_bgr: np.ndarray,
    corners: np.ndarray,
    inner_corners: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    """Use a blue corner marker to resolve checkerboard's 180-degree ambiguity."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([90, 90, 50], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8),
    )
    contours, _ = cv2.findContours(
        blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = [
        contour
        for contour in contours
        if 8.0 <= cv2.contourArea(contour) <= 2500.0
    ]
    if not candidates:
        return corners, False

    marker = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(marker)
    if moments["m00"] <= 0:
        return corners, False
    marker_xy = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    )
    columns, rows = inner_corners
    grid = corners.reshape(rows, columns, 2)
    endpoints = {
        "top_left": grid[0, 0],
        "top_right": grid[0, -1],
        "bottom_left": grid[-1, 0],
        "bottom_right": grid[-1, -1],
    }
    nearest = min(
        endpoints,
        key=lambda name: float(np.linalg.norm(endpoints[name] - marker_xy)),
    )
    if nearest == "top_left":
        oriented = grid
    elif nearest == "top_right":
        oriented = grid[:, ::-1]
    elif nearest == "bottom_left":
        oriented = grid[::-1]
    else:
        oriented = grid[::-1, ::-1]
    return oriented.reshape(-1, 1, 2).copy(), True


def reference_camera_transforms(
    poses: list[dict[str, Any] | None],
) -> tuple[int | None, list[list[list[float]] | None]]:
    """Map each captured camera frame into the first successful board view."""
    reference_index = next(
        (index for index, pose in enumerate(poses) if pose is not None), None
    )
    if reference_index is None:
        return None, [None] * len(poses)

    reference_from_board = np.asarray(
        poses[reference_index]["camera_from_turntable_board"], dtype=np.float64
    )
    transforms: list[list[list[float]] | None] = []
    for pose in poses:
        if pose is None:
            transforms.append(None)
            continue
        camera_from_board = np.asarray(
            pose["camera_from_turntable_board"], dtype=np.float64
        )
        reference_from_camera = reference_from_board @ np.linalg.inv(camera_from_board)
        transforms.append(reference_from_camera.tolist())
    return reference_index, transforms


def reject_pose_discontinuities(
    poses: list[dict[str, Any] | None], step_degrees: int
) -> tuple[list[dict[str, Any] | None], list[str]]:
    """Reject checkerboard corner-order flips and implausible pose jumps."""
    accepted: list[dict[str, Any] | None] = []
    statuses: list[str] = []
    previous_rotation: np.ndarray | None = None
    jump_limit_deg = max(35.0, min(step_degrees * 2.5, 170.0))
    for pose in poses:
        if pose is None:
            accepted.append(None)
            statuses.append("checkerboard_not_detected")
            continue
        if not pose.get("orientation_marker_detected", False):
            accepted.append(None)
            statuses.append("orientation_marker_not_detected")
            continue

        transform = np.asarray(
            pose["camera_from_turntable_board"], dtype=np.float64
        )
        rotation = transform[:3, :3]
        if previous_rotation is not None:
            relative_rotation = previous_rotation @ rotation.T
            cosine = np.clip(
                (np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0
            )
            rotation_delta = math.degrees(math.acos(float(cosine)))
            if rotation_delta > jump_limit_deg:
                accepted.append(None)
                statuses.append(f"rejected_pose_jump_{rotation_delta:.1f}_deg")
                continue

        accepted.append(pose)
        statuses.append("accepted")
        previous_rotation = rotation
    return accepted, statuses


def _capture_sweep(
    *,
    output_dir: Path,
    sweep_id: str,
    camera: Any,
    step_degrees: int,
    checkerboard_inner_corners: tuple[int, int] | None,
    checkerboard_square_m: float,
    background_sweep_id: str | None = None,
) -> Path:
    validate_step_degrees(step_degrees)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; refusing to overwrite a previous sweep."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    image_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    image_dir.mkdir()
    depth_dir.mkdir()

    angles = tuple(range(0, 360, step_degrees))
    frames: list[dict[str, Any]] = []
    poses: list[dict[str, Any] | None] = []
    print(
        f"\nStarting {len(angles)}-view sweep {sweep_id}. "
        "The camera stays fixed; rotate the turntable and plant together."
    )
    try:
        for frame_index, expected_angle in enumerate(angles):
            while True:
                raw_angle = input(
                    f"\nView {frame_index + 1}/{len(angles)}: set turntable to "
                    f"{expected_angle:03d} deg; measured angle [{expected_angle}]: "
                ).strip()
                try:
                    actual_angle = validate_actual_angle(
                        raw_angle or expected_angle,
                        frame_index=frame_index,
                        step_degrees=step_degrees,
                    )
                    break
                except ValueError as exc:
                    print(f"  {exc}")

            input("  Let the plant settle, then press Enter to capture ... ")
            rgb_bgr, depth_mm = camera.capture()
            expected_shape = (kinect.DEPTH_HEIGHT, kinect.DEPTH_WIDTH)
            if rgb_bgr.shape[:2] != expected_shape or depth_mm.shape != expected_shape:
                raise RuntimeError(
                    f"Kinect returned RGB/depth dimensions {rgb_bgr.shape[:2]} and "
                    f"{depth_mm.shape}; expected registered {expected_shape}."
                )
            stem = f"view_{frame_index:03d}.png"
            rgb_path = image_dir / stem
            depth_path = depth_dir / stem
            if not cv2.imwrite(str(rgb_path), rgb_bgr):
                raise OSError(f"Could not write RGB frame {rgb_path}.")
            if not cv2.imwrite(str(depth_path), depth_mm):
                raise OSError(f"Could not write depth frame {depth_path}.")

            pose = (
                checkerboard_pose(
                    rgb_bgr,
                    inner_corners=checkerboard_inner_corners,
                    square_size_m=checkerboard_square_m,
                )
                if checkerboard_inner_corners is not None
                else None
            )
            poses.append(pose)
            frames.append(
                {
                    "frame_index": frame_index,
                    "target_angle_deg": expected_angle,
                    "actual_angle_deg": actual_angle,
                    "rgb": f"images/{stem}",
                    "depth": f"depth/{stem}",
                    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "turntable_board_pose": pose,
                }
            )
            pose_status = (
                f"checkerboard RMSE {pose['reprojection_rmse_px']:.2f}px; "
                "orientation marker found"
                if pose is not None and pose["orientation_marker_detected"]
                else f"checkerboard RMSE {pose['reprojection_rmse_px']:.2f}px; "
                "orientation marker missing, pose will be rejected"
                if pose is not None
                else "checkerboard pose unavailable"
                if checkerboard_inner_corners is not None
                else "no checkerboard calibration requested"
            )
            print(f"  Saved view {frame_index + 1}/{len(angles)}; {pose_status}.")

    except BaseException:
        print(
            f"\nSweep interrupted. Partial frames remain in {output_dir}; "
            "no complete manifest was written."
        )
        raise

    if checkerboard_inner_corners is not None:
        poses, pose_statuses = reject_pose_discontinuities(poses, step_degrees)
    else:
        pose_statuses = ["calibration_not_requested"] * len(poses)
    reference_index, relative_transforms = reference_camera_transforms(poses)
    for frame, relative_transform in zip(frames, relative_transforms):
        frame["reference_camera_from_camera"] = relative_transform
    for frame, pose_status in zip(frames, pose_statuses):
        frame["pose_tracking_status"] = pose_status

    manifest = {
        "sweep_id": sweep_id,
        "capture_mode": "stationary_kinect_v2_rotating_turntable",
        "camera_stationary": True,
        "object_rotated": True,
        "camera": {
            "model": "Microsoft Kinect v2",
            "serial_number": camera.serial,
            "registered_rgb_depth": True,
            "registration": "libfreenect2 depth-to-color registration into 512x424 frame",
        },
        "camera_intrinsics_registered_frame": {
            "fx": KINECT_INTRINSICS.fx,
            "fy": KINECT_INTRINSICS.fy,
            "cx": KINECT_INTRINSICS.cx,
            "cy": KINECT_INTRINSICS.cy,
            "width": kinect.DEPTH_WIDTH,
            "height": kinect.DEPTH_HEIGHT,
            "depth_unit": "millimetres",
        },
        "turntable": {
            "step_degrees": step_degrees,
            "rotation_direction": "positive, as seen from above",
            "pose_reference_frame_index": reference_index,
            "pose_jump_limit_deg": max(35.0, min(step_degrees * 2.5, 170.0)),
            "checkerboard_inner_corners": (
                list(checkerboard_inner_corners)
                if checkerboard_inner_corners is not None
                else None
            ),
            "checkerboard_square_size_m": (
                checkerboard_square_m
                if checkerboard_inner_corners is not None
                else None
            ),
            "relative_pose_note": (
                "reference_camera_from_camera maps a current registered depth-camera "
                "point into the reference registered depth-camera frame, using the "
                "turntable-attached checkerboard pose."
            ),
        },
        "background_sweep_id": background_sweep_id,
        "frames": frames,
    }
    (output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def _prompt_metadata() -> dict[str, Any]:
    while True:
        try:
            plant_id = validate_plant_id(input("Plant ID: "))
            break
        except ValueError as exc:
            print(f"  Invalid plant ID: {exc}")
    species = input("Species [Eucalyptus]: ").strip() or "Eucalyptus"
    capture_date = input(f"Capture date [{date.today().isoformat()}]: ").strip()
    capture_date = capture_date or date.today().isoformat()
    rim_height = input("Pot-rim height above turntable (mm) [unknown]: ").strip()
    return {
        "plant_id": plant_id,
        "species": species,
        "capture_date": capture_date,
        "pot_rim_height_above_turntable_mm": (
            float(rim_height) if rim_height else None
        ),
    }


def _prompt_optional_mass(label: str) -> float | None:
    while True:
        raw = input(f"{label} in grams [Enter if unavailable]: ").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  Enter a numeric mass or press Enter to leave it unavailable.")
            continue
        if not math.isfinite(value) or value < 0:
            print("  Mass must be finite and nonnegative.")
            continue
        return value


def _connect_camera(serial: str | None):
    camera = kinect.KinectDevice(CAMERA_LABEL, serial or kinect.CAM_A_SERIAL)
    camera.open()
    return camera


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="Kinect v2 serial; defaults to configured camA.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--step-deg", type=int, default=15)
    parser.add_argument(
        "--checkerboard-inner-corners",
        nargs=2,
        type=int,
        metavar=("COLUMNS", "ROWS"),
        help="enable per-view board pose estimation (e.g. 9 6)",
    )
    parser.add_argument(
        "--checkerboard-square-mm",
        type=float,
        default=25.0,
        help="physical checkerboard square size in millimetres (default: 25)",
    )
    parser.add_argument(
        "--background-only",
        action="store_true",
        help="capture an empty-turntable reference sweep for later background subtraction",
    )
    parser.add_argument(
        "--background-sweep-id",
        help="link a plant sweep to a previously captured empty-background sweep",
    )
    parser.add_argument("--setup-id", help="required ID for --background-only.")
    args = parser.parse_args(argv)
    try:
        validate_step_degrees(args.step_deg)
        if args.checkerboard_square_mm <= 0:
            raise ValueError("--checkerboard-square-mm must be positive.")
        if args.background_only and not args.setup_id:
            raise ValueError("--setup-id is required with --background-only.")
        if not args.background_only and args.setup_id:
            raise ValueError("--setup-id is only valid with --background-only.")
    except ValueError as exc:
        parser.error(str(exc))

    board = (
        tuple(args.checkerboard_inner_corners)
        if args.checkerboard_inner_corners
        else None
    )
    if board is not None:
        checkerboard_object_points(board, args.checkerboard_square_mm / 1000.0)
    elif not args.background_only:
        print(
            "WARNING: no checkerboard specified. RGB-D will be saved, but the "
            "manifest will not contain inter-view camera transforms."
        )
    if not kinect.KINECT_AVAILABLE:
        print(
            "\npylibfreenect2 is unavailable. Install libfreenect2 and the Python "
            "binding using rig_calibration/install_libfreenect2.sh.",
            file=sys.stderr,
        )
        return 1

    try:
        if args.background_only:
            sweep_id = validate_plant_id(args.setup_id)
            output_dir = args.dataset_root / "background" / sweep_id
        else:
            metadata = _prompt_metadata()
            sweep_id = metadata["plant_id"]
            output_dir = args.dataset_root / "plants" / sweep_id
            (output_dir.parent).mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                raise FileExistsError(f"{output_dir} already exists.")

        camera = _connect_camera(args.serial)
        try:
            if args.background_only:
                result = _capture_sweep(
                    output_dir=output_dir,
                    sweep_id=sweep_id,
                    camera=camera,
                    step_degrees=args.step_deg,
                    checkerboard_inner_corners=board,
                    checkerboard_square_m=args.checkerboard_square_mm / 1000.0,
                )
            else:
                result = _capture_sweep(
                    output_dir=output_dir,
                    sweep_id=sweep_id,
                    camera=camera,
                    step_degrees=args.step_deg,
                    checkerboard_inner_corners=board,
                    checkerboard_square_m=args.checkerboard_square_mm / 1000.0,
                    background_sweep_id=args.background_sweep_id,
                )
        finally:
            camera.close()

        if not args.background_only:
            metadata["shoot_fresh_mass_g"] = _prompt_optional_mass(
                "Shoot fresh mass after harvest"
            )
            metadata["shoot_dry_mass_g"] = _prompt_optional_mass(
                "Shoot dry mass after drying to constant mass"
            )
            metadata["background_sweep_id"] = args.background_sweep_id
            (result / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        pose_count = sum(
            frame["reference_camera_from_camera"] is not None
            for frame in json.loads(
                (result / "capture_manifest.json").read_text(encoding="utf-8")
            )["frames"]
        )
        print(f"\nCompleted {sweep_id}: {result}")
        if board is not None and pose_count == 0:
            print(
                "WARNING: no checkerboard poses were found; these images cannot "
                "be registered into a common plant frame from this manifest."
            )
        elif board is not None and pose_count < len(range(0, 360, args.step_deg)):
            print(
                f"WARNING: only {pose_count} views have accepted checkerboard poses. "
                "Review missed detections and orientation jumps before reconstruction."
            )
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nCapture cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nStationary Kinect capture failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
