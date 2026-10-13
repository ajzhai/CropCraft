"""
Guided, step-by-step data collection for one or more plant specimens.

Interactively:
  1. Prompts for species, plant ID, total weight (plant+pot), pot weight
     -- computes and shows net plant weight -- and appends a row to
     dataset/ground_truth.csv.
  2. Walks you through the 6-position manual rig protocol (camA at
     0/30/.../150 deg, camB simultaneously at +180 deg), pausing at each
     step for you to reposition the cameras, then captures both.
  3. Saves RGB + registered depth to dataset/plants/{plant_id}/{images,depth}/
     and writes frames_manifest.json for that plant.
  4. Asks whether to capture another plant, looping for the rest of the day.

Usage:
    python rig_calibration/collect_specimen.py

Requires pylibfreenect2 (see rig_calibration/install_libfreenect2.sh) and
both Kinect v2 units connected. Camera serials are set below -- update them
if your units differ (Protonect --help lists connected serials).
"""
import sys
import csv
import json
from pathlib import Path
from datetime import date

import numpy as np
import cv2

USBFS_MEMORY_PATH = Path("/sys/module/usbcore/parameters/usbfs_memory_mb")
USBFS_MEMORY_MIN_MB = 64   # two Kinect v2 depth transfer pools need ~16MB each

try:
    from pylibfreenect2 import (
        Freenect2, SyncMultiFrameListener, FrameType,
        Registration, Frame, OpenGLPacketPipeline,
    )
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False


REPO_ROOT     = Path(__file__).resolve().parent.parent
DATASET_DIR   = REPO_ROOT / "dataset"
PLANTS_DIR    = DATASET_DIR / "plants"
GROUND_TRUTH_CSV = DATASET_DIR / "ground_truth.csv"
GT_FIELDNAMES = [
    "plant_id", "date", "species_breed",
    "total_fresh_weight_with_pot_g", "pot_weight_g", "net_weight_g",
    "pot_weight_source", "notes",
]

HALF_SWEEP_ANGLES_DEG = [0, 30, 60, 90, 120, 150]

CAM_A_SERIAL = "003071363947"   # update from `Protonect --help` if different
CAM_B_SERIAL = "006158144547"

DEPTH_WIDTH, DEPTH_HEIGHT = 512, 424

# Kinect v2's color auto-exposure isn't controllable via libfreenect2 (no
# gain/exposure API), and the greenhouse has natural light only. Measured
# RGB comes out consistently underexposed (~26/255 mean). A fixed gamma
# curve applied identically to every frame recovers shadow visibility
# without introducing shot-to-shot exposure drift across a plant's 12
# views, which an adaptive per-image stretch would (bad for NeRF training,
# which assumes consistent appearance across views of the same scene).
RGB_GAMMA = 2.6
_GAMMA_LUT = (np.linspace(0, 1, 256) ** (1.0 / RGB_GAMMA) * 255).astype(np.uint8)


def gamma_correct(rgb: np.ndarray) -> np.ndarray:
    return cv2.LUT(rgb, _GAMMA_LUT)


# ---------------------------------------------------------------------------
# Small input helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default: str = None) -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"{prompt}{hint}: ").strip()
    return val if val else (default or "")


def ask_float(prompt: str) -> float:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            return float(raw)
        except ValueError:
            print(f"  Not a number: {raw!r} -- try again (e.g. 1234.5)")


def confirm(prompt: str, default_yes: bool = True) -> bool:
    hint = "[Y/n]" if default_yes else "[y/N]"
    raw = input(f"{prompt} {hint}: ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes")


def next_default_plant_id() -> str:
    """Suggest the next sequential plant ID for today, based on existing folders."""
    today = date.today().isoformat()
    existing = [p.name for p in PLANTS_DIR.glob(f"{today}_plant*")] if PLANTS_DIR.exists() else []
    n = 1
    while f"{today}_plant{n:02d}" in existing:
        n += 1
    return f"{today}_plant{n:02d}"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def collect_ground_truth() -> dict:
    print("\n" + "-" * 60)
    print("  Ground truth")
    print("-" * 60)

    plant_id = ask("Plant ID", default=next_default_plant_id())
    species = ask("Species name")
    total_g = ask_float("Total weight, plant + pot (grams)")
    pot_g = ask_float("Pot weight only (grams)")
    net_g = total_g - pot_g

    if pot_g >= total_g:
        print(f"  WARNING: pot weight ({pot_g}g) >= total weight ({total_g}g) -- "
              f"net would be {net_g}g. Double-check these numbers.")

    print(f"\n  Net plant weight = {total_g}g - {pot_g}g = {net_g}g")
    if not confirm("Looks right?"):
        return collect_ground_truth()

    row = {
        "plant_id": plant_id,
        "date": date.today().isoformat(),
        "species_breed": species,
        "total_fresh_weight_with_pot_g": total_g,
        "pot_weight_g": pot_g,
        "net_weight_g": net_g,
        "pot_weight_source": "estimated",
        "notes": "",
    }
    _append_ground_truth(row)
    return row


def _append_ground_truth(row: dict):
    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if GROUND_TRUTH_CSV.exists():
        with GROUND_TRUTH_CSV.open(newline="") as f:
            existing = list(csv.DictReader(f))

    updated = False
    for i, r in enumerate(existing):
        if r.get("plant_id") == row["plant_id"]:
            existing[i] = row
            updated = True
            break
    if not updated:
        existing.append(row)

    with GROUND_TRUTH_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing)

    action = "Updated" if updated else "Added"
    print(f"  {action} ground_truth.csv row for {row['plant_id']}")


# ---------------------------------------------------------------------------
# Kinect camera wrapper
# ---------------------------------------------------------------------------

class KinectDevice:
    def __init__(self, label: str, serial: str):
        self.label = label
        self.serial = serial
        self.fn = None
        self.device = None
        self.listener = None
        self.registration = None

    def open(self, warmup_frames: int = 30):
        if not KINECT_AVAILABLE:
            raise RuntimeError(
                "pylibfreenect2 not installed. See rig_calibration/install_libfreenect2.sh, "
                "then: pip install pylibfreenect2"
            )
        self.fn = Freenect2()
        n = self.fn.enumerateDevices()
        serials = [self.fn.getDeviceSerialNumber(i).decode() for i in range(n)]
        if self.serial not in serials:
            raise RuntimeError(
                f"Camera {self.label}: serial {self.serial} not found among connected "
                f"devices {serials}. Update CAM_A_SERIAL/CAM_B_SERIAL in this script."
            )

        pipeline = OpenGLPacketPipeline()
        self.device = self.fn.openDevice(self.serial.encode(), pipeline=pipeline)
        self.listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)
        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()
        self.registration = Registration(
            self.device.getIrCameraParams(),
            self.device.getColorCameraParams(),
        )

        for i in range(warmup_frames):
            frames = self.listener.waitForNewFrame(milliseconds=10000)
            if frames is None:
                raise RuntimeError(
                    f"Camera {self.label} (serial {self.serial}): no frame arrived within "
                    f"10s during warm-up (frame {i + 1}/{warmup_frames}). The device opened "
                    f"and started but isn't actually delivering frames -- this usually means "
                    f"it's in a wedged state from an earlier crash/kill. Unplug and replug "
                    f"this Kinect's USB cable, then try again."
                )
            self.listener.release(frames)

    def capture(self):
        """Return (rgb, depth): rgb is (424,512,3) BGR uint8, depth is (424,512) uint16 mm."""
        frames = self.listener.waitForNewFrame(milliseconds=10000)
        if frames is None:
            raise RuntimeError(
                f"Camera {self.label} (serial {self.serial}): no frame arrived within 10s. "
                f"Unplug and replug this Kinect's USB cable, then try again."
            )
        color_frame = frames[FrameType.Color]
        depth_frame = frames[FrameType.Depth]

        undistorted = Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
        registered = Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
        # enable_filter suppresses libfreenect2's "flying pixel" artifact at
        # depth discontinuities -- without it, registration leaves the
        # registered RGB frame speckled with black holes wherever a depth
        # sample straddles an edge (worse in the greenhouse's bright ambient
        # IR from sunlight, which already degrades the depth return).
        self.registration.apply(color_frame, depth_frame, undistorted, registered,
                                 enable_filter=True, bigdepth=None, color_depth_map=None)

        rgb = registered.asarray(dtype=np.uint8)[..., :3].copy()
        depth = undistorted.asarray(dtype=np.float32).astype(np.uint16)
        self.listener.release(frames)
        return rgb, depth

    def close(self):
        if self.device:
            try:
                self.device.stop()
                self.device.close()
            except Exception:
                pass
            self.device = None


# One camera open, streaming, and closed again at a time -- avoids running
# both Kinects concurrently (background packet-loss/timing logs from a live
# stream otherwise bury the interactive prompts, and it halves peak USB DMA
# buffer usage). Costs a short re-warmup per shot instead of once per session.
SHOT_WARMUP_FRAMES = 5


def capture_single_shot(label: str, serial: str):
    """Open one camera, grab exactly one frame, close it again."""
    cam = KinectDevice(label, serial)
    print(f"  [{label}] opening...")
    try:
        cam.open(warmup_frames=SHOT_WARMUP_FRAMES)
        return cam.capture()
    finally:
        cam.close()


# ---------------------------------------------------------------------------
# Per-plant capture loop
# ---------------------------------------------------------------------------

def capture_plant(plant_id: str):
    plant_dir = PLANTS_DIR / plant_id
    images_dir = plant_dir / "images"
    depth_dir = plant_dir / "depth"
    images_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    frames_manifest = {}

    print(f"\n{'=' * 60}")
    print(f"  Capturing {plant_id} -- 6 steps, camA then camB at each "
          f"(one stream at a time)")
    print(f"{'=' * 60}")

    for i, angle in enumerate(HALF_SWEEP_ANGLES_DEG, start=1):
        opposite = (angle + 180) % 360
        print(f"\nStep {i}/6: position camera A at {angle:3d} deg, "
              f"camera B at {opposite:3d} deg (directly opposite).")
        input("  Press Enter when both cameras are in position ... ")

        rgb_a, depth_a = capture_single_shot("camA", CAM_A_SERIAL)
        rgb_b, depth_b = capture_single_shot("camB", CAM_B_SERIAL)
        rgb_a = gamma_correct(rgb_a)
        rgb_b = gamma_correct(rgb_b)

        a_rgb_path = images_dir / f"camA_{angle:03d}.png"
        a_dep_path = depth_dir / f"camA_{angle:03d}.png"
        b_rgb_path = images_dir / f"camB_{angle:03d}.png"
        b_dep_path = depth_dir / f"camB_{angle:03d}.png"

        cv2.imwrite(str(a_rgb_path), rgb_a)
        cv2.imwrite(str(a_dep_path), depth_a)
        cv2.imwrite(str(b_rgb_path), rgb_b)
        cv2.imwrite(str(b_dep_path), depth_b)

        pos_a = f"camA_{angle:03d}"
        pos_b = f"camB_{angle:03d}"
        frames_manifest[pos_a] = {"rgb": f"images/{a_rgb_path.name}", "depth": f"depth/{a_dep_path.name}"}
        frames_manifest[pos_b] = {"rgb": f"images/{b_rgb_path.name}", "depth": f"depth/{b_dep_path.name}"}

        print(f"  Captured camA@{angle} deg + camB@{opposite} deg  ({i}/6 steps done, "
              f"{i * 2}/12 frames)")

    manifest_path = plant_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps(frames_manifest, indent=2))

    print(f"\n{plant_id}: 12/12 frames saved to {plant_dir}/")
    print(f"  Frames manifest -> {manifest_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CropCraft guided specimen collection")
    print("=" * 60)

    if not KINECT_AVAILABLE:
        print("\npylibfreenect2 is not installed in this environment.")
        print("Run rig_calibration/install_libfreenect2.sh first, then:")
        print("  pip install pylibfreenect2")
        sys.exit(1)

    try:
        usbfs_mb = int(USBFS_MEMORY_PATH.read_text().strip())
    except OSError:
        usbfs_mb = None
    if usbfs_mb is not None and usbfs_mb < USBFS_MEMORY_MIN_MB:
        print(f"\nusbfs_memory_mb is {usbfs_mb}MB -- a single Kinect v2's depth "
              f"transfer pool needs ~16MB of USB DMA buffer, and the kernel default "
              f"of 16MB leaves no headroom. Raise it first:\n"
              f"  echo 256 | sudo tee {USBFS_MEMORY_PATH}\n"
              f"then re-run this script. (Resets on reboot -- see dataset/README.md "
              f"for making it permanent.)")
        sys.exit(1)

    print("\nOnly one camera streams at a time -- each shot opens its camera, "
          "captures, and closes it again, so neither device is running during "
          "ground-truth entry or while you reposition between steps.")

    while True:
        row = collect_ground_truth()
        capture_plant(row["plant_id"])

        print(f"\n{row['plant_id']} complete: species={row['species_breed']}, "
              f"net weight={row['net_weight_g']}g, 12 frames saved.")

        if not confirm("\nCapture another plant?"):
            break

    print("\nDone for today.")


if __name__ == "__main__":
    main()
