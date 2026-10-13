# dataset/

Real capture data for the dual-Kinect single-plant biomass study.

## Quick start: guided data collection

For actually collecting today's plants, run:

```bash
cd ~/CropCraft
python rig_calibration/collect_specimen.py
```

This is a single, interactive, step-by-step tool that walks you through the
whole process for one plant, then loops for the next:

1. Prompts for **Plant ID** (suggests the next sequential `<date>_plantNN`,
   just press Enter to accept it), **species name**, **total weight
   (plant+pot, grams)**, and **pot weight (grams)** -- computes and shows
   the net plant weight, lets you re-enter if it looks wrong, then writes
   the row to `ground_truth.csv`.
2. Walks through the 6 manual repositioning steps, printing exactly where
   each camera should be (e.g. "camera A at 90 deg, camera B at 270 deg"),
   pausing for Enter before firing both cameras.
3. Saves all 12 RGB + registered-depth frames to
   `plants/{plant_id}/{images,depth}/` and writes that plant's
   `frames_manifest.json` automatically.
4. Asks if you want to capture another plant -- say yes and it repeats for
   the rest of the day's batch, or no to stop and close the cameras.

Requires `pylibfreenect2` (already installed in the `cropcraft` conda env as
of 2026-07-23) and both Kinect v2 units connected -- see
`rig_calibration/install_libfreenect2.sh` if setting up a new machine.
Camera serials are set near the top of `collect_specimen.py`
(`CAM_A_SERIAL`/`CAM_B_SERIAL`); update them if your units differ.

### Prerequisite: usbfs DMA memory limit

Two Kinect v2 units streaming simultaneously each need ~16MB of USB DMA
buffer for their depth transfer pool. The kernel's `usbfs_memory_mb` limit
defaults to 16MB **total** -- fine for one camera, but the second one hits
`LIBUSB_ERROR_NO_MEM` and streams garbled/lossy frames instead of failing
cleanly. `collect_specimen.py` checks this at startup and refuses to run if
it's too low, but you still need to raise it yourself:

```bash
# Immediate fix (this boot only):
echo 256 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
```

This resets on every reboot. On this machine `usbcore` is compiled into the
kernel (not a loadable module), so `/etc/modprobe.d/` options won't apply --
make it permanent via a GRUB boot parameter instead:

```bash
sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="usbcore.usbfs_memory_mb=256 /' /etc/default/grub
sudo update-grub
sudo reboot
```

## Layout

```
dataset/
  calib/
    intrinsics/
      camA/                       # ~15-20 checkerboard photos, one-time per camera
      camB/
      camA_intrinsics.json        # output of calibrate_intrinsics.py
      camB_intrinsics.json
    extrinsics/
      positions_manifest_template.json   # copy per day, edit paths
      2026-07-23/
        positions/                 # 12 checkerboard-at-plant-center photos for this day
        positions_manifest.json    # copy of the template, paths filled in
        rig_positions.json         # output of calibrate_extrinsics.py for this day
      2026-07-24/
      2026-07-25/
  plants/
    frames_manifest_template.json  # reference only -- collect_specimen.py writes this per plant automatically
    2026-07-23_plant01/
      images/                     # camA_000.png, camB_180.png, camA_030.png, ...
      depth/                      # same naming, registered depth (uint16, mm)
      frames_manifest.json        # written automatically by collect_specimen.py
      transforms.json             # output of make_transforms.py (NeRF path only, see below)
    2026-07-23_plant02/
    ...
  ground_truth.csv                # written automatically by collect_specimen.py
```

## Position naming

12 positions per plant = 6 manual repositioning steps x 2 cameras, matching
the rig: camA and camB start 180 deg apart and are moved together in 30 deg
steps. Position ids are `<camera>_<angle:03d>`:

```
camA_000  camB_180
camA_030  camB_210
camA_060  camB_240
camA_090  camB_270
camA_120  camB_300
camA_150  camB_330
```

`collect_specimen.py` names files this way automatically. These ids must
also match across the day's `rig_positions.json` and the intrinsics camera
keys (`camA`/`camB`) if you use the NeRF path below -- `make_transforms.py`
looks them up by these exact strings.

## Ground truth

`ground_truth.csv` columns, all written automatically by
`collect_specimen.py`:

- `total_fresh_weight_with_pot_g` -- scale reading with the plant still in
  its pot.
- `pot_weight_g` -- pot mass alone.
- `net_weight_g` -- `total_fresh_weight_with_pot_g - pot_weight_g`,
  computed and shown to you at collection time, then stored directly (not
  re-derived later) so the CSV is self-contained.
- `pot_weight_source` -- `estimated` or `weighed`. Pot mass here is
  approximate rather than individually weighed; if all pots are the same
  black pot model, it's cheap to weigh a few empty ones once and use that
  mean (with its spread) instead of a single guessed value -- ground-truth
  provenance is something reviewers will ask about, and this is the
  easiest of the sources to tighten up.

This is **as-collected (fresh) biomass**, not oven-dry above-ground
biomass -- no drying step is used in this protocol. Keep that distinction
in mind if it's ever compared against another study's dry-biomass numbers.

## Calibration cadence (NeRF comparison path only)

The `calib/` + `calibrate_intrinsics.py`/`calibrate_extrinsics.py`/
`make_transforms.py` tools are for an optional second reconstruction method
(Nerfstudio-based), separate from the guided collection above. If/when
that path is wired up:

- **Intrinsics**: once, ever, per camera (fixed lens, no zoom/focus change).
- **Extrinsics**: once per day by default (see `calib/extrinsics/<date>/`),
  since the cameras are hand-repositioned rather than a fixed mechanical
  rig. If you don't have a physical angle/position guide (floor marks,
  fixed-height stand) yet, build one before trusting a single per-day
  calibration to hold across all 10 plants that day -- otherwise consider
  recalibrating more than once per day, or spot-checking partway through.
