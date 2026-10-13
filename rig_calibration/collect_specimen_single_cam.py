"""
Single-camera fallback for guided specimen collection, for use when only
one of the two Kinect units is working (see collect_specimen.py's docstring
for the normal two-camera protocol).

Captures all 12 rig positions (0-330 deg in 30 deg steps) sequentially with
ONE physical camera, manually moved by hand at each step, instead of camA
and camB capturing 6 simultaneous position-pairs. Output file naming and
frames_manifest.json format are unchanged from the two-camera version --
angles 0-150 keep the "camA" label, angles 180-330 keep the "camB" label --
so the files land in the same rig positions calibrate_extrinsics.py already
solved poses for, and make_transforms.py needs no code changes to read it.

IMPORTANT: every real frame here comes from the same physical sensor. When
running make_transforms.py against a specimen captured this way, point BOTH
--intrinsics camA=... and camB=... at THIS camera's own intrinsics file
(not camB's real one) -- rig_positions.json's per-position "camera" field
only selects which intrinsics get applied to that position, and positions
originally calibrated as "camB" still carry that label. Without aliasing
both to the same file, camB-labeled frames would get camB's real (different)
focal length and distortion applied to pixels that were never actually
captured through that lens.

Usage:
    python rig_calibration/collect_specimen_single_cam.py
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_specimen as cs

CAMERA_LABEL = "camA"          # arbitrary but must match one of the two
CAMERA_SERIAL = cs.CAM_A_SERIAL  # the confirmed-working unit
FULL_SWEEP_ANGLES_DEG = list(range(0, 360, 30))


def label_for_angle(angle: int) -> str:
    return "camA" if angle in cs.HALF_SWEEP_ANGLES_DEG else "camB"


def capture_plant_single_cam(plant_id: str):
    plant_dir = cs.PLANTS_DIR / plant_id
    images_dir = plant_dir / "images"
    depth_dir = plant_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    frames_manifest = {}

    print(f"\n{'=' * 60}")
    print(f"  Capturing {plant_id} -- single-camera fallback, 12 steps "
          f"around the full 360 deg sweep")
    print(f"{'=' * 60}")

    # Open the device once and hold the stream open across all 12 shots,
    # closing only at the end. Re-opening per shot (12x/plant) was the
    # actual cause of captures failing partway through -- each open/close
    # cycle risks leaving the USB device in a degraded state, and the
    # failure probability compounds with every extra cycle, which is why
    # it tended to surface late in a plant's sweep rather than on shot 1.
    cam = cs.KinectDevice(CAMERA_LABEL, CAMERA_SERIAL)
    print(f"  [{CAMERA_LABEL}] opening...")
    cam.open()
    try:
        for i, angle in enumerate(FULL_SWEEP_ANGLES_DEG, start=1):
            label = label_for_angle(angle)
            print(f"\nStep {i}/12: position the camera at {angle:3d} deg.")
            input("  Press Enter when the camera is in position ... ")

            rgb, depth = cam.capture()
            rgb = cs.gamma_correct(rgb)

            rgb_path = images_dir / f"{label}_{angle:03d}.png"
            dep_path = depth_dir / f"{label}_{angle:03d}.png"
            cv2.imwrite(str(rgb_path), rgb)
            cv2.imwrite(str(dep_path), depth)

            pos_id = f"{label}_{angle:03d}"
            frames_manifest[pos_id] = {
                "rgb": f"images/{rgb_path.name}",
                "depth": f"depth/{dep_path.name}",
            }
            print(f"  Captured {pos_id}  ({i}/12 steps done)")
    finally:
        cam.close()

    manifest_path = plant_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps(frames_manifest, indent=2))

    print(f"\n{plant_id}: 12/12 frames saved to {plant_dir}/")
    print(f"  Frames manifest -> {manifest_path}")


def main():
    print("=" * 60)
    print("  CropCraft single-camera specimen collection (fallback)")
    print("=" * 60)

    if not cs.KINECT_AVAILABLE:
        print("\npylibfreenect2 is not installed in this environment.")
        print("Run rig_calibration/install_libfreenect2.sh first, then:")
        print("  pip install pylibfreenect2")
        sys.exit(1)

    try:
        usbfs_mb = int(cs.USBFS_MEMORY_PATH.read_text().strip())
    except OSError:
        usbfs_mb = None
    if usbfs_mb is not None and usbfs_mb < cs.USBFS_MEMORY_MIN_MB:
        print(f"\nusbfs_memory_mb is {usbfs_mb}MB -- raise it first:\n"
              f"  echo 256 | sudo tee {cs.USBFS_MEMORY_PATH}\n"
              f"then re-run this script.")
        sys.exit(1)

    print(f"\nUsing only {CAMERA_LABEL} (serial {CAMERA_SERIAL}) -- the other "
          f"unit is out of service. You'll manually move this one camera "
          f"through all 12 rig positions per plant instead of both cameras "
          f"capturing 6 simultaneous position-pairs.")
    print("\nSee this script's module docstring before running "
          "make_transforms.py on data captured here -- the --intrinsics "
          "camA=/camB= arguments both need to point at this camera's own "
          "intrinsics file.")

    while True:
        row = cs.collect_ground_truth()
        capture_plant_single_cam(row["plant_id"])

        print(f"\n{row['plant_id']} complete: species={row['species_breed']}, "
              f"net weight={row['net_weight_g']}g, 12 frames saved.")

        if not cs.confirm("\nCapture another plant?"):
            break

    print("\nDone for today.")


if __name__ == "__main__":
    main()
