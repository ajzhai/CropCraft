# Rig calibration → Nerfstudio transforms.json

Standalone toolchain for the dual-Kinect, 12-position rig. Not part of
CropCraft's own pipeline (which relies on Polycam/COLMAP structure-from-motion
instead of known poses) — this exists because you know your rig's geometry
and can skip SfM entirely by solving it directly.

Only dependency is `opencv-python`, already in the repo's `requirements.txt`
— no `opencv-contrib`/ArUco needed, since both calibration steps below use a
plain checkerboard.

## Workflow

1. **Per-camera intrinsics** (`calibrate_intrinsics.py`) — run once for camA,
   once for camB. ~15-20 handheld checkerboard shots per camera, varying
   tilt/distance/position in frame.

2. **Per-position extrinsics** (`calibrate_extrinsics.py`) — run once for the
   whole rig. Place a checkerboard flat where the plant pot will sit (this
   defines your world origin) and, **without moving it**, photograph it from
   all 12 rig positions. Solves each position's camera-to-world pose. This
   is reusable for every future specimen *only if* your 12 positions are
   mechanically repeatable (hard stops/detents) — if not, redo this step per
   capture session.

   Use `--visualize` to sanity-check before trusting it: the plotted camera
   arrows should point roughly inward toward the origin in a ring. If they
   don't, something's wrong (board moved between shots, wrong camera/position
   paired in the manifest, etc.) — don't proceed to NeRF training until this
   looks right.

3. **Per-specimen transforms.json** (`make_transforms.py`) — run once per
   plant, combining the fixed rig calibration with that plant's 12 captured
   frames.

4. **Train + export**, using Nerfstudio directly (not CropCraft's
   `train_nerf.py`, which is a thin wrapper around exactly this call but
   hardcoded for COLMAP-derived field data):
   ```bash
   ns-train depth-nerfacto --data ./data/plant_007 \
       --output-dir ./work_dirs --experiment-name plant_007 \
       --max-num-iterations 20000
   ns-export pointcloud --load-config ./work_dirs/plant_007/.../config.yml \
       --output-dir ./data/plant_007 --num-points 1000000 --remove-outliers True
   ```

## Prerequisites this toolchain assumes, not handles

- **Depth-to-RGB registration.** Kinect v2's depth/IR sensor and color
  sensor are physically offset with different resolutions — raw depth must
  already be aligned to the color frame (e.g. via libfreenect2's
  `Registration` class, or the equivalent SDK call) before it reaches
  `make_transforms.py`. This toolchain just points at whatever registered
  depth PNGs you hand it.
- **Depth units.** Confirm what your registration step actually outputs
  (commonly uint16 millimeters) and pass the matching `--depth_unit_scale` —
  don't assume 0.001 without checking.
- **Segmentation** (plant vs. pot vs. background) is not handled here at
  all; that's a separate step, upstream or downstream depending on whether
  you mask before or after reconstruction.

## Stationary Kinect v2 with a rotating plant

The collector for the alternative stationary-camera setup is
[`collect_specimen_stationary_kinect.py`](collect_specimen_stationary_kinect.py).
It uses one Kinect v2 fixed on a tripod while the plant and pot rotate together
on a turntable. Its output is kept in the ignored
`dataset_kinect_stationary/` folder, separate from the existing dual-camera
dataset and its manually repositioned single-camera fallback.

Install the Kinect v2/libfreenect2 stack as described in
[`install_libfreenect2.sh`](install_libfreenect2.sh), then run a background
reference sweep before placing the plant:

```bash
python rig_calibration/collect_specimen_stationary_kinect.py \
  --serial YOUR_KINECT_SERIAL \
  --step-deg 15 \
  --checkerboard-inner-corners 9 6 \
  --checkerboard-square-mm 25 \
  --background-only \
  --setup-id greenhouse_setup_01
```

Attach a known-size checkerboard rigidly to the turntable, outside the pot
footprint, with all corners visible throughout the full rotation. Add a
distinctive blue marker beside one specific outer corner; this resolves the
checkerboard's 180-degree orientation ambiguity. Use the board's **inner
corner count** and measured square size. Keep the camera, tripod, lights,
background, checkerboard and turntable in the same positions for the empty
reference and every plant sweep. The reference captures the empty scene and
rotating board at matching angles.

Capture each plant using the same turntable setup:

```bash
python rig_calibration/collect_specimen_stationary_kinect.py \
  --serial YOUR_KINECT_SERIAL \
  --step-deg 15 \
  --checkerboard-inner-corners 9 6 \
  --checkerboard-square-mm 25 \
  --background-sweep-id greenhouse_setup_01
```

The default is 24 views at 15-degree intervals. At each stop, enter the
measured angle shown by the turntable scale, wait for foliage to settle, and
capture. The manifest records target and measured angles, the per-frame
checkerboard pose, reprojection error, and a transform mapping each registered
depth frame into the first successfully detected board frame. Review all
checkerboard detections and errors before using the transforms. A missed blue
orientation marker or an implausible pose jump is rejected and saved as
unavailable rather than silently interpolated.

Outputs contain `images/`, `depth/`, `capture_manifest.json`, and (for plant
sweeps) `metadata.json`. RGB and depth are saved at the registered
512x424 Kinect v2 resolution; depth values are millimetres. Optional mass
fields can be entered after the scan. Capture data are intentionally
git-ignored.

This is the recommended acquisition geometry for a repeatable single-camera
sweep: it avoids moving the camera between views. It does **not** remove Kinect
v2 limitations: its lower resolution can miss very thin branches, sunlight
and reflective surfaces can degrade infrared depth, and plant motion during
rotation can cause misalignment. The checkerboard-derived transforms depend
on the board being rigidly attached to the turntable and detected accurately.
The new capture format is separate from the current GG-SSVT dual-Kinect
loader; transform/mask validation and an adapter are still required before
these captures can be compared as calibrated reconstructions.
