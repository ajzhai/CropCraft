# DUSt3R, MASt3R and Fast3R: install and run plan

Three pose-free reconstruction methods, against the calibration-free rig
estimate. All three are installed and their APIs verified; **none has been run
against real weights**, because that needs the GPU.

**A note on the name.** These are sometimes called "the 3Rs" after the shared
suffix. This project does not use that name for them, because the thesis framing
takes "the three R's" from Malik et al., where it means recognition,
reconstruction and reorganization. Two different things with one name in one
dissertation is a confusion worth spending three words to avoid, so these are
the *pointmap models* throughout.

---

## Why this experiment matters more than it did last week

`dataset/calib` is empty. Every camera pose in this project comes from
`geometry/rig.py` estimating them from the depth data, and the azimuth
corrections **saturate the ±8° search bound on 25 of 30 specimens**, the least
verified assumption in the pipeline.

These three methods estimate pose from images alone. They are the independent
check on that assumption, and there is now a sharper reason to want it: the
[re-run](RERUN_V_BATCH.md) established that only **8 of 36** reconstructions have
an above-ground volume that could physically weigh the measured mass. Two
explanations fit that, and they call for different fixes:

1. **The visual hull is the wrong instrument** for thin, branching, leafy plants,
   at any resolution and with any pose. Then better poses change nothing.
2. **The poses are bad enough to inflate the hull**, and a mis-registered view
   carves away less than it should.

A pose-free method that agrees with the rig estimate points at (1) and closes the
question. One that disagrees materially reopens (2). Either result is worth the
GPU time, which is not true of most remaining experiments.

---

## What is installed, and where

Nothing was installed into the main environment, `pip install -r
requirements.txt` for these repos would fight the working pipeline over numpy and
torch versions. They live in an isolated venv instead:

| | |
|---|---|
| Clones | `third_party/{dust3r,mast3r,fast3r}` (gitignored) |
| Environment | `.venv-posefree/` (gitignored) |
| Path wiring | `.venv-posefree/Lib/site-packages/posefree_repos.pth` |

`posefree_repos.pth` holds three absolute paths: the two repo roots and the
CropCraft root. Run the experiment with that venv's interpreter:

```bash
.venv-posefree/Scripts/python.exe -m ggssvt.cli posefree --check-only
```

### Three things the old install instructions got wrong

**DUSt3R and MASt3R have no `setup.py`.** The runbook said `pip install -e .` for
both; it fails. They are used by putting the repo root on `sys.path`, which is
what the `.pth` file does. Each repo adds its own vendored dependency to
`sys.path` on import (`dust3r.utils.path_to_croco`,
`mast3r.utils.path_to_dust3r`), so two entries cover all four packages.

**Their two copies of `dust3r` do not collide.** MASt3R vendors dust3r at commit
`3cc8c88` while the standalone clone is `4c24a6e`, which looks like a version
conflict waiting to happen. `diff -rq` on the two package trees reports **zero
differing files**, the commits differ elsewhere in the repo. One environment
hosts both safely. Fast3R namespaces its copy as `fast3r.dust3r`, so it never
enters the argument.

**Fast3R's `requirements.txt` is a training environment.** It pulls `deepspeed`,
`open3d`, `wandb` and `numpy<2.0.0`. Inference needs none of that. Install with
`--no-deps` and add only what the imports demand.

### The dependency set that actually works

Verified on Python 3.13.7 with torch 2.13.0+cpu:

```bash
pip install torch torchvision roma einops opencv-python scipy trimesh 'huggingface-hub[torch]>=0.22'
pip install omegaconf hydra-core lightning lightning-bolts rich scikit-learn scikit-image
pip install --no-deps -e third_party/fast3r
```

`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch
version instead` is expected and harmless, DUSt3R falls back to a pure-PyTorch
rotary embedding. Building the CUDA kernel (`croco/models/curope`) on the lab
machine is worth ~15% and is optional.

### The one Python-version constraint

**`open3d` has no Python 3.13 wheel, and Fast3R imports it at module top level**
in `models/multiview_dust3r_module.py`, the module holding `estimate_camera_poses`.

Stubbing `open3d` out confirms the pose path never touches it: it is there for
ICP refinement this project does not use. But the import is unconditional, so on
3.13 that module cannot load.

**Use Python 3.11 for the lab environment.** It is what Fast3R documents, open3d
installs normally, and no upstream code needs patching. DUSt3R and MASt3R are
happy on either.

---

## API drift already found

The adapters were written against documentation and never executed. Checking them
against the installed code found one real error and confirmed the rest.

**Fast3R was wrong in three ways.** There is no function named
`inference_multiview`; the entry point is `inference_multiview.inference`. Its
signature is `(views, model, device, dtype, ...)`, views first, model second,
`dtype` required. And camera poses do **not** ride inside each prediction, which
is what `_from_multiview` assumed: predictions carry point maps only, and poses
come from a separate `MultiViewDUSt3RLitModule.estimate_camera_poses(preds, ...)`
call. That is the method's whole design, one global PnP solve over every view,
rather than the pairwise alignment DUSt3R and MASt3R run.

Fixed, and `tests/test_pose_free_api.py` now pins every call the adapters make.
Those tests skip when the repos are absent, so they cost nothing normally and
fail loudly at the start of a lab session, which is when you want to know.

**DUSt3R and MASt3R were correct**, verified signature by signature:

| call | real signature |
|---|---|
| `load_images` | `(folder_or_list, size, square_ok=False, verbose=True, patch_size=16)` |
| `make_pairs` | `(imgs, scene_graph='complete', prefilter=None, symmetrize=True)` |
| `inference` | `(pairs, model, device, batch_size=8, verbose=True)` |
| `global_aligner` | `(dust3r_output, device, mode=..., **optim_kw)` |
| `compute_global_alignment` | `(self, init=None, niter_PnP=10, **kw)` |

`get_im_poses`, `get_pts3d` and `get_conf` all exist on `PointCloudOptimizer`.

---

## How to run it

### Cost, and why the order matters

The driver is `scene_graph="complete"` with `symmetrize=True`: twelve views is
66 unordered pairs run in both directions, so **132 forward passes per
specimen**, then a 300-iteration global alignment. Measured, not assumed, the
adapter's docstring used to say 66, which is half the real cost. Fast3R
ingests all twelve in one pass instead, so it is the cheap one.

Rough 4080 estimates, **time one specimen before trusting them**:

| method | per specimen | 36 specimens |
|---|---|---|
| Fast3R | ~5–10 s | ~5 min |
| DUSt3R | ~60–120 s | ~1–1.5 h |
| MASt3R | ~60–120 s | ~1–1.5 h |

**Run Fast3R first.** It is twenty times cheaper and exercises the same
comparison code, so if the plumbing is broken you find out in five minutes rather
than ninety.

```bash
python -m ggssvt.cli posefree --check-only
python -m ggssvt.cli posefree --methods fast3r --plants M001 --device cuda
```

Check that one specimen before going further. Then:

```bash
python -m ggssvt.cli posefree --methods fast3r --device cuda
python -m ggssvt.cli posefree --methods dust3r mast3r --device cuda
```

### Scale, which decides whether the numbers mean anything

DUSt3R and Fast3R return geometry up to an **unknown global scale**. A volume
computed from arbitrary scale measures the rescaling, not the plant, so both must
go through `recover_scale_from_depth` first; `PoseFreeResult.is_metric` records
which happened. MASt3R's `_metric` checkpoint is the exception and the reason to
prefer it, use that checkpoint, not the others.

### What to watch on the first run

`sanity_check_result` verifies the cameras end up looking roughly at the scene
centroid, which catches a convention change (OpenCV → OpenGL, or camera-to-world
→ world-to-camera) before it becomes a plausible-looking reconstruction that is
inside out. **Read its warnings on specimen one.** A silent inside-out
reconstruction is the expensive failure here.

---

## What to report

The question is not which method reconstructs best. It is **whether the
calibration-free rig estimate is trustworthy**, so the comparison is against
`geometry/rig.py`, not against each other.

1. **Rotation and translation error per view**, rig versus each method, from
   `compare_poses`. The specific thing to look for: does the disagreement
   concentrate on the specimens whose azimuth correction saturated at ±8°? If it
   does, the saturation is real error rather than a search-bound artefact, and
   that is a finding worth a paragraph in the limitations chapter.

2. **Carve the volume from pose-free poses and re-run the plausibility check.**
   This is the experiment that matters. If the implied densities stay one to two
   orders of magnitude below plant tissue with better poses, then the visual hull
   is simply the wrong instrument for these species, the strongest form of the
   [RERUN_V_BATCH.md](RERUN_V_BATCH.md) §3 argument, and no longer open to "your
   poses were bad".

3. **Biomass RMSE from pose-free reconstructions**, into the same LOOCV harness.
   Report it, but do not expect it to resolve: nothing else in this project's
   method comparisons has, and the plausibility check is the load-bearing number.

4. **Which specimens each method fails on.** Fast3R and DUSt3R may simply fail on
   the thin Eucalyptus that the geometric segmenter also struggles with, and a
   shared failure set is itself evidence about the subject rather than the method.

Record wall-clock per specimen too. If DUSt3R's pairwise cost is what rules it
out for a larger capture, that belongs in the write-up.
