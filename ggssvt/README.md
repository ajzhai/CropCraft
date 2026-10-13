# GG-SSVT

Geometry-Grounded Self-Supervised Vision Transformer for label-efficient
volumetric reconstruction and biomass estimation, implemented against the
dual-Kinect single-plant captures in [`../dataset`](../dataset).

The four components named in the dissertation scaffold:

| Component | Module | What it does |
|---|---|---|
| DINO backbone (box 3) | [`models/backbones.py`](models/backbones.py) | Swappable stem: none, DINOv2, or DINOv3 |
| Fourier back-projected token embeddings | [`models/embedding.py`](models/embedding.py) | Positions tokens by their world coordinate, not their image index |
| Cross-view geometric attention | [`models/attention.py`](models/attention.py) | Biases attention logits by `-gamma * ||x_i - x_j||^2`, gamma learned per head |
| Implicit occupancy decoder | [`models/decoder.py`](models/decoder.py) | World coordinate + fused context to occupancy, evaluated in chunks |
| Space-carving self-supervision | [`geometry/carving.py`](geometry/carving.py) | Depth and silhouettes to the occupancy targets, no manual labels |
| Biomass head | [`models/head.py`](models/head.py) | Volume integration with a modulated density prior |
| SAM3D segmentation | [`geometry/sam3d.py`](geometry/sam3d.py) | Promptable masks made 3D-consistent across views |

Built since, and mostly because the carve turned out to be the limiting
component rather than the model:

| Component | Module | What it does |
|---|---|---|
| TSDF depth fusion | [`geometry/fusion.py`](geometry/fusion.py) | Integrates depth instead of intersecting silhouettes, so concavities survive |
| Per-specimen pot rim | [`geometry/pot.py`](geometry/pot.py) | Finds the rim as a step in the vertical profile, and refuses when there is none |
| Plausibility diagnostic | [`eval/plausibility.py`](eval/plausibility.py) | Mass over volume against the density of plant tissue |
| Acceptance gates | [`eval/gates.py`](eval/gates.py) | Blocking and advisory checks per stage, including a collapsed training run |
| Training campaign | [`campaign.py`](campaign.py) | The whole programme under one command, resumable, target-fingerprinted |
| Reconstruction registry | [`eval/methods.py`](eval/methods.py) | Which reconstruction sources exist, and why some cannot |
| DITR-style feature lifting | [`geometry/dino_lift.py`](geometry/dino_lift.py) | DINOv2 patch features projected onto the points, with an occlusion test |
| View-count ablation | [`eval/view_ablation.py`](eval/view_ablation.py) | Whether four images would have done. They would not |
| Experiment log | [`eval/progress.py`](eval/progress.py) | What has run, what has not, and what is stale |
| Project page | [`eval/site_html.py`](eval/site_html.py) | A GitHub Pages site generated from the artefacts |
| Architecture figures | [`eval/architecture.py`](eval/architecture.py) | One diagram per methodology, drawn from config |

## Where the research stands

**The headline has changed.** The reconstruction step, not the regressor, was the
limiting component. Space carving produces the visual hull, which Laurentini
showed is the maximal solid consistent with the silhouettes, so a pot rim and the
gap between two leaves are both filled at any resolution. Only 8 of 36 carved
specimens imply a bulk density inside a generous 200 to 1000 kg/m³ band.
Integrating the same depth maps as a signed distance field instead raises that to
25 at the identical grid and 31 at the resolution the sensor supports, and moves
biomass RMSE from 0.544 to 0.335 kg with a paired bootstrap of −0.209
[−0.363, −0.066]. That is the first resolved improvement in the project.

**GG-SSVT itself has still never been trained.** Everything above comes from the
geometry pipeline, frozen features, or classical baselines.

- **[RESEARCH_STATUS.md](RESEARCH_STATUS.md)**: every research question and
  hypothesis from the proposal, what answers it, what is pending, and what to do
  about each. Includes the recommended scope change.
- **[HYPOTHESIS_3.md](HYPOTHESIS_3.md)**: the frequency hypothesis, restructured
  into six measurable sub-claims; four are already established.
- **[FINDINGS.md](FINDINGS.md)**: every experiment run, the deductions, the bugs
  found, and what can be reported today.
- **[RERUN_V_BATCH.md](RERUN_V_BATCH.md)**: what the measured pot weights
  changed, including the claim that had to be withdrawn.
- **[POSEFREE.md](POSEFREE.md)**: DUSt3R, MASt3R and Fast3R. Installed, verified,
  not yet run, and the install instructions that were wrong before.

The generated project page lives in `work_dirs/ggssvt/site/` after
`python -m ggssvt.cli dashboard`. Its Experiment log is built by walking the work
directory, so it records what has actually run rather than what was intended.

## Setup and today's experiments

See **[RUNBOOK.md](RUNBOOK.md)** for the environment setup, the ordered list of
experiments with time budgets, and the VRAM fallbacks. Two things from it are
worth repeating here:

- **Two conda environments are required.** Nerfstudio pins torch 2.0.1+cu118 and
  Python 3.8; GG-SSVT needs `transformers >= 4.56`, which will not install there.
  [`environment.yml`](environment.yml) builds the GG-SSVT one.
- **`--finetune-epochs` defaults to 200**: which in a 36-fold leave-one-out sweep
  is well over a day per condition. Override it.

## Quick start

```bash
python -m ggssvt.cli inspect
```

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli gate
```

```bash
python -m ggssvt.cli fuse --write-cache
```

```bash
python -m ggssvt.cli baselines
```

```bash
python -m ggssvt.cli report && python -m ggssvt.cli dashboard
```

`preprocess` takes roughly nine seconds per specimen and writes a ~4 MB archive
each; everything after it reads only the cache. `gate` runs 324 acceptance checks
and exits non-zero if any is blocking, so it is worth putting in front of
anything long. `fuse` is eleven minutes and everything downstream reads the cache
it writes.

For the trained model, do not hand-run `pretrain` and `loocv`: use the campaign,
which queues the whole programme, resumes after an interruption, refuses to reuse
a run fitted to different targets, and blocks a run that learned nothing.

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

```bash
python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2
```

Smoke first: five minutes over four specimens, and it proves the loop, the
checkpointing and the resume before a night is committed to them.
**[RUNBOOK.md](RUNBOOK.md) has the full thirteen-step sequence**; this is the
short version.

## Pipeline

```
dataset/plants/<id>/          12 RGB-D frames, 512x424, registered
        |
        |  geometry/rig.py         floor plane -> tilt, roll, camera height
        |                          subject column -> world origin on the plant axis
        |                          geometry/refine.py -> residual azimuth correction
        v
   registered views
        |
        |  geometry/segment.py     cylinder about the world axis
        v
   subject masks
        |
        |  geometry/carving.py     silhouette + depth carving -> 128^3 occupancy
        v
   pseudo-labels ------------------> stage 1: self-supervised occupancy
        |                                        |
        |                                        v
        +--> ground_truth.csv ------> stage 2: biomass head, LOOCV
```

## Calibration: what is missing, and what is done instead

`dataset/calib` is empty. No ChArUco intrinsics were captured, no per-day
`rig_positions.json` exists, and the `positions/` directories contain no images.
There are therefore no measured extrinsics to load.

[`geometry/rig.py`](geometry/rig.py) estimates them from the depth data instead:

1. **Floor plane** (RANSAC per view) gives tilt, roll and camera height, four
   of six degrees of freedom, with no calibration target. Recovered camera
   heights are consistent to about 3 cm across a sweep, which is a useful
   independent check that the fit is working.
2. **Subject axis** puts the world origin on the plant. Several candidate
   columns are proposed per view and the winner is the one whose registration
   the *other views actually agree with*, scored by how many voxels several
   cameras land on together. This matters: an earlier version picked the
   strongest single-view candidate and silently locked onto background
   structure a metre behind the plant on several specimens.
3. **Azimuth** comes from the filename, then
   [`geometry/refine.py`](geometry/refine.py) corrects it. The correction is
   worth a lot, surface coverage on E011 goes from 0.29 to 0.48, and on M001
   from 0.16 to 0.43.

**This is an estimate, not a measurement.** The azimuth corrections saturate the
±8° search bound on most specimens, which means the true placement error is at
least that large and possibly larger. Capturing the ChArUco sequence that
`dataset/README.md` already specifies would replace all of step 3 with a
measurement and is the single highest-value thing that could be added to the
capture protocol.

## Two naming conventions in the capture set

`rig_calibration/collect_specimen.py` names each camB file with **camA's** step
angle (`camB_000` … `camB_150`) even though it prints the opposite angle to the
console and `dataset/README.md` documents the ids as `camB_180` … `camB_330`.
Both conventions are present in `dataset/plants`, and taking the filename
literally would place camB on top of camA instead of opposite it, collapsing the
12-view rig into a 6-view one.

[`data/naming.py`](data/naming.py) resolves both: camB physically occupies the
upper half-circle, so a camB filename angle below 180 is the step-angle
convention and is rotated by 180 degrees.

## Evaluation protocol

Leave-one-out cross-validation over the usable specimens, for every method in
the comparison table, baselines included.

Stage 1 pretraining runs **once over every specimen** by default. No mass label
is touched, but the held-out plant's images are seen, which makes the protocol
transductive rather than inductive. `--strict` re-runs pretraining inside each
fold for the inductive number; the gap between the two is worth reporting rather
than hiding.

## DINO backbone comparison

Box 3 of the system diagram is a DINO vision transformer. It is swappable, so
its contribution can be measured rather than assumed:

| condition | backbone | trained |
|---|---|---|
| no DINO (control) | RGB-D patch stem | from scratch |
| DINOv2 | `facebook/dinov2-{small,base,large}` | frozen |
| DINOv3 | `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m` | frozen |

Everything downstream of the stem is identical across conditions, so the
difference is attributable to the backbone alone. The control is not a strawman:
it is the only condition whose trunk ingests depth directly, and the frozen DINO
stems take depth through a parallel embedding so every condition sees the same
information.

### DINOv3 needs an access request

DINOv3 weights are **gated** on HuggingFace, Meta approves access manually, per
account. Every command degrades gracefully and tells you what to do:

1. Open <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>
2. Accept the licence and request access.
3. `hf auth login`, or set `HF_TOKEN`.

DINOv2 is open and needs none of this.

### Two experiments

```bash
python -m ggssvt.cli dino-probe --variant base
```

A linear probe on frozen features, leave-one-out. Runs on a CPU in about five
minutes and measures how much biomass information the pretrained representation
holds on its own. Use it to decide whether the GPU run is worth it.

```bash
python -m ggssvt.cli experiment --backbones cnn dinov2 dinov3 --variant base
```

The full comparison: GG-SSVT pretrained and cross-validated once per backbone.
Needs a GPU. Pretraining is re-run per condition, sharing a checkpoint across
backbones would be meaningless, since the stems produce different features.

### Probe results: 36 specimens

| condition | RMSE | R² | dims |
|---|---|---|---|
| no DINO (geometry only) | 0.458 kg | +0.312 | 7 |
| **DINOv2-base** | **0.392 kg** | **+0.495** | 1536 |
| DINOv2-base + geometry | 0.392 kg | +0.496 | 1543 |
| DINOv3 | *gated, not run* | | |

Paired bootstrap, DINOv2-base against the control: **ΔRMSE −0.066 kg, 95% CI
[−0.178, +0.062], p≈0.29, not resolved.** The effect size barely moved between
n=28 and n=36 and neither did the verdict.

**These numbers are not comparable with the baselines table below.** The probe
rotates onto principal components before standardising; the baselines standardise
raw features. On seven features that is a pure rotation, and it changes the
result: the same geometric features score 0.544 there and 0.458 here.

Three things follow, and the third is the one that matters most:

- DINO helps in the point estimate, and the gain grows with backbone size
  (small → base). The direction is consistent across every reduction setting
  tried (PCA-4, PCA-8, un-truncated ridge), so it is not an artefact of the
  probe's hyperparameters.
- **Adding geometry on top of DINO changes nothing** (0.295 → 0.296). The pooled
  DINO descriptor already contains whatever the hand-built geometric features
  encode, which is mostly size.
- **Twenty-eight specimens cannot establish the effect.** The confidence interval
  on the difference spans zero. Reporting "DINOv2 improves R² from 0.62 to 0.74"
  without that interval would be overclaiming. More specimens, not a bigger
  backbone, is what would settle it.

## SAM3D

**Naming, first.** In the predecessor project's
`neural_geometry/sam3d/sam3d_pipeline.py`, "SAM3D" means *SAM plus 3D
back-projection*: promptable 2D masks per view, lifted to 3D, made consistent
across views. That is what [`geometry/sam3d.py`](geometry/sam3d.py) implements.
It is **not** Meta's SAM 3D Objects, which is a single-image mesh generator. If
you meant the mesh generator, that is a different component and a different
experiment, say so and it can be added, but note its weights are gated too.

Availability, checked 21 August 2026:

| model | status |
|---|---|
| `facebook/sam-vit-base` / `-large` / `-huge` | **open**, used by default |
| `facebook/sam2-hiera-*` | open |
| `facebook/sam3` | **gated**, manual approval |
| `facebook/sam-3d-objects` | **gated**, manual approval |

### What it does, and why it is a refinement rather than a replacement

The default cylinder segmentation keeps everything physically inside a cylinder
about the plant axis. That is robust, but anything *inside* that cylinder counts
as subject, a rig pole or bench edge directly behind the plant is kept. SAM3D
cuts those away on appearance.

SAM needs a prompt, and because the views are already registered, the projected
plant axis and the bounding box of the geometric mask are free. So SAM3D refines
the geometric mask rather than replacing it, which is why it needs no manual
clicks, and equally why it cannot recover a plant the geometric stage missed.

Three rules keep SAM honest, and the third is what makes the result *3D* rather
than twelve independent 2D segmentations:

1. **3D gating**. The SAM mask is intersected with the working cylinder, so a
   mask leaking onto the far wall is trimmed by geometry.
2. **Coverage guard**, a mask covering implausibly much or little of its prompt
   box is rejected and that view falls back to geometry.
3. **Multi-view agreement**, a mask whose back-projected points do not land
   where the other views' points already are is reverted.

### Running it

SAM3D changes the subject mask, which changes the carved occupancy, the
self-supervision targets and the geometric features. So it is a *preprocessing*
choice with its own cache, not a flag at training time:

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d
```

Roughly 80 s per specimen on a CPU with SAM ViT-B, most of it SAM's image
encoder; far faster on a GPU with `--sam-device cuda`.

## The full factorial

SAM3D and DINO act at different stages, one changes what counts as the subject,
the other changes how pixels become tokens, so they can interact. One-factor-
at-a-time ablations cannot see that; the factorial can.

```bash
python -m ggssvt.cli factorial
```

| | no DINO | DINOv2 | DINOv3 |
|---|---|---|---|
| **no SAM3D** | control | DINO only | DINO only |
| **SAM3D** | SAM3D only | both | both |

Conditions are compared on the **specimens that pass the quality gate under
every segmenter**, not on whichever subset each pipeline happens to leave
standing, otherwise the conditions would be scored on different plants.

Effects are reported as paired bootstraps against the same control: the main
effect of each factor, each factor *given* the other, and the interaction
`(both − DINO) − (SAM3D − neither)`. A negative interaction means the two are
synergistic; positive means partly redundant.

At n=28 every one of these intervals is wider than the effect it measures. The
harness prints the intervals rather than the point estimates alone, because with
four conditions and a small sample, one arrangement looking good by luck is
likely, not unlikely.

### Factorial results: 33 specimens, frozen-feature probe

|  | no DINO | DINOv2-base |
|---|---|---|
| **no SAM3D** | 0.576 kg / R² −0.080 | **0.385 kg / R² +0.518** |
| **SAM3D** | **0.778 kg / R² −0.967** | 0.390 kg / R² +0.505 |

Paired effects on RMSE (negative = the addition helps):

| effect | ΔRMSE | 95% CI | resolved? |
|---|---|---|---|
| DINO alone | −0.191 kg | [−0.404, +0.021] | no |
| SAM3D alone | +0.201 kg | [−0.017, +0.394] | no |
| **DINO *given* SAM3D** | **−0.387 kg** | **[−0.757, −0.039]** | **yes** |
| SAM3D *given* DINO | +0.005 kg | [−0.007, +0.019] | no effect |
| interaction | −0.196 kg | [−0.393, +0.031] | no |

**The finding is the asymmetry, not the resolved cell.** SAM3D alone drives the
hand-crafted descriptors to R² −0.967, far below the mean-predictor floor, while
DINO moves 0.385 to 0.390, an effect of five grams. The descriptors are fragile
to which segmenter produced the hull; the learned features are indifferent to it.
Read the resolved effect carefully: it is resolved largely *because* SAM3D
without a learned backbone is so poor, so it evidences descriptor fragility more
than it evidences DINO's value.

The 33-specimen shared set is not a random subset of the 36. SAM3D fails the gate
on E015, E019 and V006 on top of the E012/E016 the geometric gate drops, and
losing V006 matters beyond the count, because V is the batch that breaks the
mass/batch confound.

**The one result the factorial found that no one-factor ablation could.**
SAM3D on its own makes things slightly *worse* (+0.011 kg), but SAM3D given DINO
makes them *better* (−0.007 kg). The sign flips. Correspondingly, DINO helps five
times more when SAM3D is present (−0.022) than when it is not (−0.004), and the
interaction term is negative with an interval whose upper bound sits exactly on
zero.

The mechanism is plausible: SAM3D removes about 15% of the subject pixels,
tightening the mask. For hand-built geometric descriptors that is lost volume,
so they get worse. For a DINO descriptor pooled over subject patches it is less
background contamination, so it gets better. Run the two factors separately and
you would conclude "SAM3D doesn't help" and drop it.

**Two caveats that must travel with these numbers.**

*Nothing here is statistically resolved.* Every interval spans zero, the
interaction only just excludes it. This is a hypothesis the factorial generated,
not a finding it established.

*The 26-specimen set is not a random subset of the 28.* Conditions are compared
on the specimens usable under **both** segmenters, and SAM3D fails the quality
gate on E015 and E019, which the geometric pipeline passes. Dropping those two
moved the no-DINO/no-SAM3D control from 0.358 kg (n=28) to 0.306 kg (n=26), a
larger shift than any effect in the table. The comparison between cells is fair
because all four see the same plants; the comparison against the earlier
28-specimen numbers is not.

### SAM3D's effect on the reconstruction itself

| metric | geometric | SAM3D | change |
|---|---|---|---|
| multi-view agreement | 0.625 | 0.637 | **+1.9%** |
| surface coverage | 0.788 | 0.741 | −6.0% |
| connected fraction | 0.870 | 0.842 | −3.2% |
| above-ground volume | 11.0 L | 10.4 L | −5.4% |
| usable specimens | 28/30 | 26/30 | −2 |

SAM accepted 96% of views (minimum 67% on the worst specimen) and removed 15.3%
of subject pixels on average. The masks are tighter and more view-consistent,
which is what it was added to do, but tighter masks also cost coverage and
push two marginal specimens below the quality gate.

## Ablations

```bash
python -m ggssvt.cli pretrain --no-geometry --out work_dirs/ggssvt/checkpoints/ablation.pt
```

`--no-geometry` zeroes and freezes both geometric pathways, the Fourier
back-projected positional code and the 3D-distance attention bias, leaving a
multi-view transformer of the same shape with no 3D prior. Everything else is
identical, so the difference is attributable to geometry grounding alone.

After training, `model.fusion.distance_scales()` returns the learned per-head
`gamma`. That is the direct evidence for whether the distance bias is used at
all, and it belongs in the ablation section.

## Nerfstudio

```bash
python -m ggssvt.cli nerfstudio
```

Writes `transforms.json` into every specimen directory, plus the
`rig_positions.json` and per-camera intrinsics that
[`rig_calibration/make_transforms.py`](../rig_calibration/make_transforms.py)
expects. That script was written against `calibrate_extrinsics.py` output that
was never captured, the estimated rig supplies the same poses, so both paths now
work.

```bash
ns-train splatfacto --data dataset/plants/M001 --pipeline.model.camera-optimizer.mode SO3xR3
```

```bash
ns-train depth-nerfacto --data dataset/plants/M001 --pipeline.model.camera-optimizer.mode SO3xR3
```

```bash
ns-viewer --load-config outputs/M001/splatfacto/<timestamp>/config.yml
```

**Always enable the camera optimiser.** The exported poses are estimated, not
measured, and the azimuth refinement saturates its ±8° search bound on most
specimens. Letting the radiance field refine them from image evidence is an
independent estimate, and comparing the optimised poses against the exported
ones is a direct, quantitative check on the registration this whole pipeline
rests on.

### Three things that will bite

**Coordinate convention.** Nerfstudio's `transform_matrix` is camera-to-world in
OpenGL/Blender convention (+y up, −z forward); the rig poses are OpenCV (+y down,
+z forward). The exporter applies the flip and
[`tests/test_nerfstudio_export.py`](../tests/test_nerfstudio_export.py) pins it.
Get it wrong and the scene trains upside down and back to front, which looks like
a failed reconstruction rather than a failed export.

**Twelve views is thin.** Radiance fields normally want 50–200. The depth maps
help (prefer `depth-nerfacto`, or splatfacto with depth) but expect floaters
between the sparse viewpoints, especially above ~1.15 m where the frames truncate.

**The vendored copy in `nerfstudio/` does not import.** `.gitignore` line 1 is
`data/`, which matches at any depth and silently excluded
`nerfstudio/nerfstudio/data/` (the dataparsers, datasets and pixel samplers) so
it was never committed and is absent on disk. The ignore rule now carries a
negation for that path, but the files still need restoring: install upstream
Nerfstudio (`pip install nerfstudio`) or re-clone it into that directory. Either
way it needs CUDA and `tiny-cuda-nn`, per the root README.

## Mesh-based biomass, and what it revealed

```bash
python -m ggssvt.cli mesh --export work_dirs/ggssvt/meshes
```

Marching cubes on the carved occupancy, then biomass from mesh descriptors:
canopy surface area above the pot rim, enclosed volume by the divergence
theorem, solidity (mesh volume over convex hull volume), area-to-volume ratio,
and height. Validated against an analytic sphere, volume within 0.5%, area
carrying the known ~8% marching-cubes bias on a binary grid.

### Results: 36 specimens, leave-one-out

| method | RMSE | MARE | R² |
|---|---|---|---|
| **direct 2D** | **0.469 kg** | 42.8% | **+0.279** |
| mesh geometry | 0.507 kg | 37.4% | +0.157 |
| geometric features (voxel) | 0.544 kg | 43.8% | +0.030 |
| mean predictor | 0.568 kg | 57.8% | −0.058 |
| volume allometric | 0.592 kg | 53.0% | −0.150 |
| canopy area allometric | 0.598 kg | 47.3% | −0.170 |

**Do not read the ordering as a result.** The 3D-versus-2D difference is not
statistically resolved in either direction (paired bootstrap [−0.051, +0.227])
and it flips if features are whitened before the ridge. The same test on the
earlier n=28 set gives [−0.168, +0.099], so the "reconstruction beats pixels"
that used to head this table was never resolved either.

**What is resolved is the mechanism.** Measured mass over reconstructed
above-ground volume gives an implied bulk density, against 300–900 kg/m³ for
fresh tissue. **Only 8 of 36 specimens** land inside a generous 200–1000 band;
25 imply less, all ten Mango at 26–77 kg/m³. The hull encloses the air between
leaves, so it measures the canopy envelope rather than the plant, and its volume
cannot carry mass information no matter what is fitted to it. Within Eucalyptus
alone the geometric features score R² −0.313, below the mean-predictor floor.

*(Previous n=28 table, before V001–V008 and the measured pot weights: mesh
geometry 0.359 / 0.613, geometric 0.397 / 0.526, direct 2D 0.440 / 0.419.)*

**The hypothesis this was built to test failed.** The reasoning was that a leaf's
mass scales with its area while it encloses almost no volume, so canopy area
should beat canopy volume. It does not: the single-term area law is the *worst*
method tried, and head-to-head against the single-term volume law the difference
is −0.020 kg with an interval spanning zero. The two are equally uninformative.

The mechanism is worth stating, because it generalises. **A visual hull's surface
area is envelope area, not leaf area.** Twelve views at 12 mm voxels cannot
resolve individual leaves, so the mesh measures the outside of the canopy, and
the outside of a canopy carries no more information than its volume does.

The multi-feature mesh set is the best of the 3D methods, though a paired
bootstrap against the voxel features gives −0.037 kg, 95% CI [−0.128, +0.066],
not resolved. Leave-one-feature-out shows **height**
carrying it (removing height costs 0.051 kg; removing canopy area costs 0.001).

### The confound that caps every biomass claim here

Chasing the mesh result down exposed something that applies to *all* the biomass
numbers in this repository, not just the mesh ones.

The Eucalyptus specimens fall into two batches with almost no overlap in mass:

| batch | n | mean mass | character |
|---|---|---|---|
| E001–E010 | 10 | 0.538 kg | small, reconstruct as mostly pot |
| E011–E020 | 8 usable | 1.844 kg | tall thin saplings |
| **V001–V008** | 8 | **1.138 kg** | **spans both; sd 484 g, the widest of any batch** |

**Knowing only which batch a specimen came from explained R² = 0.887 of the
Eucalyptus mass variance**, more than any method achieved. And within either
batch, nothing predicted anything: no method cleared R² = 0.2 on E001–E010 or
E011–E020 taken alone, at n=10 and n=8.

**V001–V008 was collected to break this.** Its masses run 500–1800 g, overlapping
both existing batches rather than forming a third cluster, and it brings the
batch-only R² down to **0.744** across the three Eucalyptus batches and **0.697**
across all four. Still high, but the shortcut is no longer free, and the
apparent 3D advantage went with it, which is the correct trade.

So the models are recovering *which group a plant belongs to*, tall and sparse
versus short and solid, rather than estimating mass among comparable plants.
Height and solidity are doing that work, which is exactly what the
leave-one-feature-out analysis shows.

This does not invalidate the reconstruction pipeline or the comparison harness.
It does mean the honest claim is **"reconstructed geometry separates plant size
classes"**, not "reconstructed geometry estimates biomass". The second claim
needs specimens whose masses overlap across morphologies, the cheapest fix is
a capture batch spanning a continuous mass range within one species, rather than
more specimens of the two clusters already held.

## How many views do you actually need?

```bash
python -m ggssvt.cli preprocess --views 4 --cache-dir work_dirs/ggssvt/cache_v4
```

Only divisors of 12 give a uniform subset, so 2, 3, 4 and 6 are accepted and
anything else is refused rather than approximated, an uneven subset clusters the
views on one side and biases the hull in a direction unrelated to the plant.

### Reconstruction quality, and it is monotone

| views | usable specimens | multi-view agreement | mean above-ground hull |
|---|---|---|---|
| 3 | 17/30 | 0.372 | 133.9 L |
| 4 | 18/30 | 0.447 | 159.3 L |
| 6 | 26/30 | 0.550 | 250.3 L |
| **12** | **28/30** | **0.635** | **19.3 L** |

Usable count and multi-view agreement both improve monotonically with view count,
which justifies the 12-view protocol on its own. The volume column is the blunter
finding: **below twelve views the carve is uselessly loose.** A hull of 130–250 L
above the pot rim, for plants weighing at most 2.35 kg, is not a reconstruction,
four silhouettes simply do not constrain a branching plant. Four views at 90° is
the classic visual-hull minimum for a convex object, and a plant is the opposite
of convex.

*Caveat on that column:* the carve thresholds scale with view count (see below),
so the volumes are not a pure view-count effect. The usable-count and agreement
columns do not depend on that scaling and are clean.

### The biomass comparison across view counts is not informative

Only 15 specimens pass the quality gate under *every* view count, and on that
subset almost every R² is negative, worse than predicting the mean. Twelve views
against four is ΔRMSE −0.000 kg, 95% CI [−0.106, +0.105]. The apparent ordering
is noise and should not be read; the reconstruction metrics above are the result.

### A bug this ablation exposed

`CARVE_MIN_INFORMATIVE_VIEWS = 6` and `CARVE_MAX_VOTES = 3` were tuned against
the 12-view sweep. Held fixed, a four-view carve returns an **empty** volume
rather than a poor one: no voxel can have six informative views when only four
exist, so the first run of this ablation reported 0/30 usable at 3 and 4 views,
which looks like a finding and is a leftover constant. Both thresholds now derive
from the view count (half and a quarter respectively) unless passed explicitly,
with a regression test that carves at 3, 4, 6 and 12 views and asserts none comes
back empty.

## Can the CropCraft pipeline be reused?

Not directly, and it is worth being precise about why, because the repository
sitting alongside this one looks like it should apply.

| CropCraft assumes | This dataset is |
|---|---|
| Crop plants in **field rows** (sequential RANSAC row fitting in `align_and_render.py`) | Single potted plants on a floor, no rows |
| **Maize or soybean** procedural morphology (`morphology/maize_model.py`, `soybean_model.py`) | Mango and Eucalyptus |
| Ground truth as **LAI and leaf-angle distribution** from field surveys (`evaluate.py`) | Fresh mass on a scale, per plant |
| NeRF from SfM poses over a field traverse | 12 registered RGB-D views on a circle |

Applying it would mean writing a mango and a eucalyptus procedural morphology
model, which is the paper's central contribution, not a configuration change.

**But its core idea is the principled fix for the failure documented above.** The
canopy-area hypothesis failed because a visual hull's surface is *envelope* area,
not leaf area: twelve views at 12 mm voxels cannot resolve individual leaves.
Inverse procedural modelling sidesteps that entirely, it fits a biologically
plausible parametric model whose leaves are explicit, so leaf area becomes a
*model parameter* rather than something the sensor has to see. That is exactly
the quantity a leafy canopy's mass scales with.

That is a research direction rather than a code path, and a substantial one. It
is also the most promising route past the ceiling this dataset currently hits.

## Known limitations

- **Registration is estimated: not measured.** See above.
- **Tall specimens are truncated.** At the ~1 m working radius the vertical
  field of view reaches about 1.15 m above the floor. The E011–E020 eucalyptus
  saplings extend past the top of frame, so their canopies are cut off and their
  carved volumes are underestimates. The rig should step back for tall plants,
  or add a raised second tier.
- **Ground truth is fresh mass, not oven-dry AGB.** `net_weight_g` is an
  as-collected weight; every dissertation and proposal draft specifies oven-dry
  above-ground biomass. These are not interchangeable and the distinction has to
  be stated wherever the numbers appear.
- **Pot masses are measured for V001–V008 and estimated for the rest.** Against
  the measured eight the estimates run 10.9% light with a standard deviation of
  1.8 points, so the bias is systematic rather than noisy and can be carried as a
  stated uncertainty: roughly −12% on Mango net mass and −24% on E001–E010, where
  the pot dominates the total. Weighing every pot on the next capture removes the
  largest single uncertainty in the ground truth for ten minutes of work.
- **Two specimens fail the quality gate** (E012, E016) and `X001` has only two
  views. That leaves 36 usable specimens across two species, enough to fit a
  head, not enough for a strong generalisation claim.

## Layout

```
ggssvt/
  config.py            every camera constant, threshold and hyperparameter
  cli.py               command-line entry point
  campaign.py          the whole training programme under one command
  data/
    naming.py          the camB convention fix
    io.py              PNG loading, back-projection, projection
    dataset.py         specimen index joined to ground_truth.csv
    preprocess.py      geometry cache and the quality report
  geometry/            NumPy only, no PyTorch (sam3d.py also needs torch)
    plane.py           RANSAC plane fitting
    rig.py             calibration-free extrinsics
    refine.py          residual azimuth and offset correction
    segment.py         subject segmentation in the world frame
    sam3d.py           SAM-refined, 3D-consistent subject masks
    carving.py         space carving to occupancy
    fusion.py          TSDF depth fusion, the operator that escapes the hull
    pot.py             per-specimen pot rim, and refusal when there is none
    dino_lift.py       DITR-style 2D-to-3D feature projection
    pose_free.py       DUSt3R / MASt3R / Fast3R contracts and install help
    pose_free_backends.py  the adapters themselves
    mesh.py            marching cubes, surface area, solidity
  models/              PyTorch
    embedding.py       Fourier back-projected positional encoding
    attention.py       3D-distance-biased cross-view attention
    encoder.py         RGB-D patch tokens, pruned to the subject
    decoder.py         implicit occupancy decoder
    head.py            biomass head
    ggssvt.py          the assembled model
  training/
    dataset.py         torch dataset and query sampling
    losses.py          occupancy, biomass, volume consistency
    trainer.py         two-stage schedule and LOOCV
  models/
    backbones.py       interchangeable stems: none / DINOv2 / DINOv3
  eval/
    metrics.py         RMSE/MAE/MARE/R2, IoU, Chamfer, F-score, paired bootstrap
    baselines.py       allometric, geometric-feature, direct-2D, mean
    dino_probe.py      frozen-feature linear probe
    experiment.py      backbone comparison harness
    factorial.py       SAM3D x DINO factorial and interaction
    render.py          volume renders, contact sheets, PLY export
    gallery_html.py    interactive reconstruction gallery
    plausibility.py    can the volume physically weigh the mass?
    gates.py           acceptance checks per stage, blocking and advisory
    methods.py         which reconstruction sources exist, and why some cannot
    fusion_features.py TSDF descriptors and the fused cache writer
    view_ablation.py   how many views the carve actually needs
    frequency.py       radial power spectra, for the frequency hypothesis
    progress.py        what has run, what has not, what is stale
    mesh_baseline.py   marching cubes to biomass
    architecture.py    one diagram per methodology, SVG and PNG
    site_html.py       the project page, in the Nerfies layout
    dashboard_data.py  the payload the page reads
    dino_segment.py    can DINO separate plant from pot? Reported negative
    pose_free_experiment.py  the pointmap-model comparison harness
    nerfstudio_export.py  transforms.json from the estimated rig
    report.py          tables and figures
    visualise.py       rig and mask overlays
```

## Tests

```bash
python -m pytest tests/ -q
```

199 tests. Those needing the preprocessed cache skip when it is absent, so a
fresh clone passes without running preprocessing first.

The gate tests are worth knowing about: each one constructs the failure its check
exists for and asserts it is caught, then constructs a healthy case and asserts it
is not. A check that cannot fail is decoration.

## Looking at things

One command, every knob, replacing four that each had their own flags.

```bash
python -m ggssvt.cli show --plants M001 --layers rgb segmentation depth occupancy points --size 400
```

Layers stack top to bottom in the order given: `rgb`, `depth`, `segmentation`,
`occupancy`, `points`, `mesh`. `--source carve|fused|sam3d` picks the
reconstruction, `--view N` picks which captured view the per-view layers use, and
`--size`, `--point-radius`, `--supersample` and `--cmap` control the drawing.

```bash
python -m ggssvt.cli show --plants M001 --layers points --backend matplotlib
```

Opens an interactive 3D scatter. Worth it for point clouds and nothing else:
rotating one makes the difference between a hull and a fused shell obvious in a
way no fixed viewpoint does. Matplotlib is not a dependency and is imported only
for this.

Colormaps are `viridis`, `greys` and `greys_r`. Jet and rainbow are refused by
the constructor, with the reason: they put luminance edges at arbitrary values,
and in a depth-cued plant projection a reader reads those as structure. Greyscale
is there because a printed thesis is often read in it.

### Looking at things over SSH

```bash
python -m ggssvt.cli show --plants M001 --layers occupancy --size 72 --backend text --no-label
```

Prints straight into the terminal. On a lab machine you are holding a shell on,
a PNG is a file you cannot look at, and this is the difference between checking
a reconstruction and guessing at it. Rows are sampled at half the horizontal
rate because terminal cells are about twice as tall as they are wide; without
that every specimen looks like it has been sat on.

`--size` is the character width here rather than pixels. 72 fits a standard
terminal; 110 or 140 if yours is wider.

### Neural fields

A trained Nerfstudio field is a fourth source, and it needs one knob the others
do not: a density threshold.

```bash
python -m ggssvt.cli neural-field                     # samples and caches the grid
python -m ggssvt.cli show --plants M001 --source neural     --nerf-output work_dirs/ggssvt/nerfstudio/outputs/M001     --layers density threshold_sweep occupancy --threshold 1.0
```

Sampling needs nerfstudio and a GPU; looking at the result needs neither, because
`neural-field` caches the grid as `density_grid.npz` beside the training output.

Two layers exist only for fields. `density` is a maximum projection of the raw
field with **no threshold anywhere**, log-scaled because density spans orders of
magnitude: it shows what the field contains before anyone decides what counts as
matter. `threshold_sweep` draws the same field cut at several values side by
side, which is what makes it obvious whether any single cut keeps the canopy and
excludes the haze.

The colour frames, masks and weighed mass still come from the capture cache when
`--source neural` is used. Only the occupancy is substituted, so an `rgb` or
`segmentation` panel beside a field reconstruction shows what the sensor saw
rather than what the field imagined. The threshold is written into the caption,
because a volume from a field is meaningless without it.

The Python API is the same knobs as a config object:

```python
from ggssvt.eval.viz import VizConfig, save
save("M001", Path("figure.png"),
     VizConfig(layers=("segmentation", "occupancy"), size=500, cmap="greys"))
```

