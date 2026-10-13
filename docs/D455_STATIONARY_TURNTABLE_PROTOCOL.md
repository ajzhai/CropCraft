# D455 stationary-camera, rotating-plant capture protocol

Use this protocol for the next D455 cohort. Keep the camera fixed and rotate
the plant and pot together on a vertical-axis turntable. The current EU01-EU10
captures are exploratory: their nominal angle labels do not provide calibrated
camera-to-turntable poses, and nearby greenhouse plants and structures make
foreground extraction unreliable.

## 1. Prepare the capture area

- Bolt or clamp the D455 tripod so it cannot move during a sweep. Mark each
  tripod-foot position and the camera height; do not touch focus or framing
  between specimens.
- Place a plain matte backdrop behind and around the plant, far enough away
  that it does not touch foliage. Remove nearby pots, people, hoses, supports,
  and reflective objects from the camera's view. Use even diffuse light and
  avoid direct sun.
- Center the turntable under the camera. Mark its center and zero-degree
  direction on the base. Secure the pot so it cannot slide or tilt while
  turning.
- Frame the entire plant, pot, rim, and turntable, leaving margin at the top.
  Keep the plant at a consistent distance and avoid clipping branches.
- Fix an AprilTag/ArUco marker board rigidly to the turntable, outside the pot
  footprint and visible at every angle. Record the printed marker IDs, measured
  marker dimensions, and their rigid mounting orientation. Keep a second
  stationary fiducial in the scene as a check that the camera and background
  did not move.
- Measure the height of the pot rim above the turntable surface. Record the
  turntable axis position relative to its marker board or calibration target.

## 2. Calibrate and make a background reference

1. Save the D455 color and depth intrinsics, serial number, firmware, depth
   scale, stream resolutions, and exposure settings. Confirm aligned RGB and
   depth frames have matching dimensions.
2. Before placing the plant, capture an **empty turntable sweep** at the same
   angular positions planned for the plant. Keep the marker board and all scene
   objects in place. This is the per-angle background reference; do not use an
   empty-room reference if the turntable, marker, or lighting has changed.
3. Capture the calibration marker at every position. Estimate the
   camera-to-turntable transform from the known marker size and camera
   intrinsics, then fit the common rotation axis across the sweep. Check that
   the stationary fiducial stays fixed. Save the estimated transform,
   per-frame marker pose, reprojection error, and actual angle—not just the
   requested angle.
4. Verify a test rotation returns to zero with the marker in its original
   pose. If marker detection fails, the axis fit is unstable, or the camera
   fiducial moves, fix the setup and repeat calibration before collecting
   plants.

## 3. Capture each plant

1. Photograph the plant label and record plant ID, species, capture date,
   camera serial, pot ID, rim height, and any setup changes.
2. Put the plant and pot on the marked center without moving the camera,
   background, or calibration targets. Allow the foliage to settle.
3. Capture a full-resolution RGB and aligned raw-depth frame at each measured
   turntable angle. The initial protocol is 24 views at 15-degree intervals
   (`0` through `345` degrees); capture at 10-degree intervals only if the
   pilot demonstrates that thin stems or branches are missed at 15 degrees.
4. Rotate the plant and pot together. Pause until branches stop swaying before
   each capture. Record actual marker-derived angles; if an angle differs
   materially from the target, record it rather than silently relabeling the
   image. Do not move the camera between views.
5. Check several views during capture for depth coverage, motion blur, clipping,
   marker visibility, and plant wobble. Repeat the sweep if any are unacceptable.
6. After the scan, measure total plant-plus-pot mass. Harvest and weigh the
   shoot promptly using a tared scale; record cut and weighing times and
   measure fresh leaf and stem masses separately when feasible. Dry samples to
   constant mass and record dry mass, drying temperature, duration, and scale
   resolution. Do not enter unavailable measurements as if they were zero.

## 4. Per-specimen acceptance checks

- All planned RGB/depth pairs decode, have matching dimensions, and include
  valid depth over the plant.
- All calibration markers are detected with low reprojection error, and
  estimated angles cover the full revolution without a duplicate endpoint.
- The stationary fiducial and camera remain fixed across the sweep.
- The empty-turntable references correspond to the same setup and angles.
- The pot rim height and turntable axis are recorded in the calibrated object
  frame.
- A quick aligned point-cloud preview shows one plant with a continuous stem,
  no doubled edges from misregistration, and no large pieces of the backdrop
  or neighboring plants.
- Fresh and dry mass entries are independently checked against the scale
  notes. Review outliers and inconsistent metadata before using the specimen
  for model evaluation.

## 5. Recommended capture metadata

Store these values in the per-plant manifest or in a linked calibration file:

- `camera_stationary`: true
- `object_rotated`: true
- `turntable_axis_in_camera_m` and its coordinate convention
- `turntable_zero_direction` and `rotation_direction`
- `target_angle_deg` and marker-derived `actual_angle_deg` for each frame
- per-frame `camera_from_turntable` transform and marker reprojection error
- marker dictionary/family, IDs, physical dimensions, and board layout
- `empty_background_rgb` and `empty_background_depth` references per angle
- `pot_rim_height_above_turntable_m`
- camera serial, firmware, depth scale, aligned stream intrinsics, and exposure
  settings
- operator, capture timestamp, setup notes, and any rejected/repeated frames

Do not treat turntable angles alone as calibrated camera poses. For this
stationary-camera arrangement, reconstruction should transform each depth
frame into the camera frame, then apply the inverse measured object rotation
about the calibrated turntable axis to express all points in a common
plant-fixed frame. Apply the pot cut in that frame using the measured rim
height. Keep woody stem points; do not use a foliage-only green mask as the
sole plant segmentation.

## 6. Pilot before expanding the cohort

First collect one short test plant with both the empty-background and marker
reference. Validate marker tracking, per-angle background subtraction, stem
continuity, and pot-rim removal by viewing the registered colored point cloud.
Only then capture the full next cohort. Keep EU01-EU10 as the existing
exploratory set; do not silently replace their nominal-angle metadata or
include them in calibrated-reconstruction claims.
