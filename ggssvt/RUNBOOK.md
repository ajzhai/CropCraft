# Running the programme on the RTX 4080

Everything in order, with what it costs.

**Start at [Everything, from an empty work directory](#everything-from-an-empty-work-directory).**
That is the sequence: thirteen numbered steps, about twelve hours end to end,
and the only place step numbers are defined. Everything after it is detail
keyed to those numbers, or reference. Nothing later introduces a step the
sequence does not already contain.

> **⚠ The targets changed on 2026-08-22. Every trained number predates it.**
>
> V001–V008 were added and their pot weights are now *measured* rather than
> estimated, which moved their net masses by 2.1–5.6×. The dataset is 38
> specimens, 36 usable. Any checkpoint or result produced before that date was
> fitted against the old targets and cannot be compared with anything produced
> after it.
>
> **Rebuild the caches before running anything**, then re-run the campaign.
> The campaign fingerprints the targets each run was fitted against and
> re-runs on a mismatch rather than reusing it, so clearing
> `work_dirs/ggssvt/campaign/` is belt and braces rather than required.
> See [RERUN_V_BATCH.md](RERUN_V_BATCH.md).

**Hardware note.** This was first written for an RTX 4060 (8 GB). The lab card is
an **RTX 4080: 16 GB and roughly three times the compute**. For the Titan
session, read [The Titan session: what has never run](#the-titan-session-what-has-never-run)
first: it lists the arms that have no result at all and how to start them in
parallel without repeating the collision that caused the earlier OOM. Same Ada
architecture, so the `cu121` torch install is unchanged, but the VRAM ceiling
that shaped the original settings is gone, and the epoch counts below are raised
accordingly.

**Read this first:** the default `--finetune-epochs 200` in a leave-one-out sweep
is 36 folds × 200 epochs × 35 specimens. Even on this card that is most of a day
per condition. Every command below overrides it. If you copy a command from the
README instead of from here, you will start a multi-day run by accident.

**The bottleneck moves to the CPU.** Each specimen means decompressing a 4 MB
archive and back-projecting 2.5 M points. On a 4060 that overlapped with GPU
work; on a 4080 it will not. Run with `--workers 8` throughout, and if
`nvidia-smi` shows the GPU below about 70% utilisation, raise it further, a
faster card buys nothing while it waits on numpy.

---

## One command, when you are leaving it alone

For an unattended run, do not type the sequence below. Use:

```bash
python -m ggssvt.run_all --device cuda
```

It runs the same steps in the same order, and the difference is what it does
when one fails. Steps are either **required**, meaning later steps read their
output and a failure abandons the rest, or **optional**, meaning a failure is
recorded and the run continues. That distinction is the whole point: the mesh arm
needs scikit-image and the DINOv3 cells need an access grant Meta approves by
hand, and neither is a reason to lose the remaining hours of a night.

Each step is timed and teed to `work_dirs/ggssvt/logs/<timestamp>/`, and a table
at the end says what ran, what did not, and why. Steps whose artefact is already
newer than `ground_truth.csv` are skipped, so an interrupted night resumes rather
than restarts.

```bash
python -m ggssvt.run_all --list
```

```bash
python -m ggssvt.run_all --device cuda --skip posefree mesh
```

`--force` redoes everything, `--only <keys>` runs a subset. Exit status is
non-zero if anything failed, optional included, so it works under `nohup` or in
a systemd unit.

---

## Everything, from an empty work directory

The whole programme in order. Fourteen steps, of which five need the GPU. Total is
about twelve hours, nearly all of it step 11. Each step writes an artefact, and
the **Experiment log** on the project page is generated from those artefacts, so
the page tracks progress on its own: run something, rebuild the page, and the row
turns from pending to done.

Run these from the repository root with the conda environment active.

### Before anything

```bash
python -m ggssvt.cli preflight
```

**Run this first, every time, on every machine.** Seconds, and it is the step
that would have caught every environment failure this project has hit: a CPU
torch wheel that runs at a hundredth of the speed without ever erroring, a
HuggingFace session on the wrong account so every gated cell skips itself,
scikit-image missing so the mesh arm dies after the caches are built, proxy
variables holding a malformed URL, conflict markers in the ground truth.

Findings are `fatal` (the run cannot be right, fix before starting), `degraded`
(it will run but silently do less than you think), or `note`. It exits non-zero
only on fatal, so it works in front of a long command.

```bash
python -m pytest tests/ -q
```

Seven skips before preprocessing is correct. Zero failures is not optional.

```bash
python -m ggssvt.cli inspect
```

Confirms 39 specimens and reports the empty `dataset/calib`. If the target range
is not 0.20 to 2.35 kg, the ground truth is not the corrected one and every
number downstream will be wrong.

### 1 to 3, the caches. About 15 minutes, plus SAM3D

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda
```

```bash
for v in 3 4 6; do python -m ggssvt.cli preprocess --views $v --cache-dir work_dirs/ggssvt/cache_v$v; done
```

Expect 36 of 38 usable on the geometric cache and 33 on SAM3D.

Measured: geometric is about 6 minutes for 38 specimens and each view count
about 5. **SAM3D is the one to watch.** It is roughly 4 minutes on the GPU and
close to an hour on CPU, because it runs SAM over 12 views per specimen, so
give it `--sam-device cuda` or start it before something else.

### 4, the fusion. CPU, 11 minutes

```bash
python -m ggssvt.cli gate
```

Run this after the caches and again after anything that rebuilds them. It is
324 acceptance checks over the 36 specimens and it **exits non-zero when
something is blocked**, so it can sit in front of a long run. Two severities:
blocking means the output is unusable and whatever consumes it will produce
nonsense; advisory means it is usable but worth knowing. Expect **0 blocked
and 39 advisories**, the advisories being the 28 envelope specimens and the
rim fallbacks, both of which are findings rather than faults.

```bash
python -m ggssvt.cli fuse --write-cache
```

Expect 31 of 36 plausible against the carve's 8. This must run before step 11,
because `baseline_fused` reads the cache it writes.

Four plausible counts appear in this project and all four are correct, so check
which one a number refers to before treating two of them as a contradiction:

| | grid | pot rim | plausible |
|---|---|---|---|
| carve | 12 mm | per-specimen | **8/36** |
| fused, operator-only control | 12 mm | held at the carve's | 21/36 |
| **fused cache, what ships** | 12 mm | re-estimated on the fused occupancy | **25/36** |
| fused, native fusion grid | 6 mm | carve's | **31/36** |

`cli fuse` prints the last of these. The biomass numbers are fitted on the third.

```bash
python -m ggssvt.cli quality
```

Reconstruction metrics, about 2 minutes on CPU, and it needs both caches so it
goes here. Re-projection into the captured views for each operator, then Chamfer,
HD95, F-score and voxel IoU between them. Expect carve to score **higher** on
silhouette IoU, 0.407 against 0.219, which is the point: a hull is by
construction consistent with the silhouettes it was carved from, so that metric
prefers the worse reconstruction. Depth error is a tie at 67.9 against 67.4 mm.
Do not read the IoU column as a ranking.

### 5 to 9, the classical arm. About 40 minutes

```bash
python -m ggssvt.cli baselines
```

```bash
python -m ggssvt.cli mesh
```

```bash
python -m ggssvt.cli views
```

```bash
python -m ggssvt.cli dino-probe
```

```bash
python -m ggssvt.cli factorial
```

```bash
python -m ggssvt.cli dino-segment
```

`baselines` should print `2D + profile` at 0.457 and `fused geometry` at 0.465.
If `fused geometry` is missing, step 4 did not complete for every specimen.

`dino-segment` is the DITR-style lifting, about 9 minutes on CPU. It is a
reported negative: DINOv2 patch features reproduce the pot boundary where the
geometric method already finds it (Mango, 0.969 agreement) and do not rescue
E001-E010, where one patch spans 42 mm against a 5 to 15 mm stem. Run it so the
result is on record, not because it is expected to help.

### 10 and 11, look at it

```bash
python -m ggssvt.cli gallery
```

```bash
python -m ggssvt.cli report
```

```bash
python -m ggssvt.cli architecture
```

`architecture` writes one SVG per methodology from the pipeline's own
constants, so the figures in the paper and on the page cannot drift out of
date. Seconds.

Open `work_dirs/ggssvt/reports/gallery/reconstructions.html` and actually look.
Visual inspection has caught things in this project that no metric did.

### 12, the training campaign. GPU, 8 to 10 hours

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

Five minutes, four specimens, two epochs. It proves the loop, the checkpointing
and the resume before a night is committed to them. Then:

```bash
python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2
```

Seven runs, each gated. A run that collapses onto the training mean, never
beats the mean predictor, leaves occupancy at chance or whose loss never moves
is marked `blocked` rather than `done`, with the failing check named in the
summary line and its numbers still shown so you can see how bad it was. That
matters most overnight, when nobody is watching the loss.

Safe to interrupt: the same command resumes, and a run whose targets
no longer match its fingerprint is re-run rather than reused.

*The campaign has its own document: [CAMPAIGN.md](CAMPAIGN.md), covering the
install, the two checks to do first, what each of the seven runs closes, and how
to read the summary.*

### 13, the independent check. GPU, about 2 hours

```bash
python -m ggssvt.cli posefree --check-only
```

```bash
python -m ggssvt.cli posefree --methods fast3r --plants M001 --device cuda
```

Fast3R first on one specimen, because it is twenty times cheaper than DUSt3R and
exercises the same comparison code. Watch the `sanity_check_result` warnings on
that first specimen: they catch a camera-convention flip before it becomes a
plausible-looking reconstruction that is inside out. Then the full sweep.

See [POSEFREE.md](POSEFREE.md) for the install, which is not the one the old
instructions described.

---

## Growing the pretraining set without harvesting anything

Stage 1 fits occupancy against the carve and never reads a mass, so an
unharvested plant is a valid training example for it. Capture costs twenty
minutes; a label costs a destroyed specimen. This is the only cheap axis the
dataset has.

```bash
python -m ggssvt.cli preprocess --include-unlabelled
```

Specimens with no row in `ground_truth.csv` are carved and cached with a NaN
target. `pretrain` picks them up automatically. Every regression keeps the
labelled-only default, and `load_features` refuses a NaN target by name rather
than letting it turn a whole score column into NaN silently.

```bash
python -m ggssvt.cli attention --plant M001
```

Reads what the fusion stack attends to, once a checkpoint exists. Two numbers
worth reporting: **neighbour preference**, above 1 when the fusion prefers views
whose frusta overlap, which is what a model that has learned the rig should do;
and the mean **distance scale**, which collapsing toward zero would mean the
geometry bias is inert and this is ordinary self-attention. Both are evidence on
H2, in either direction.

---

## Keeping the page current

The page is generated, never edited. Two commands rebuild everything it shows:

```bash
python -m ggssvt.cli report && python -m ggssvt.cli dashboard
```

`dashboard` re-surveys the work directory, so the Experiment log, the method
toggle, the biomass table and the specimen browser all follow whatever has
actually run. A method with no cache does not appear; an experiment with no
artefact stays pending.

Run that after any step above, then publish:

```bash
cd work_dirs/ggssvt/site && git init -b main && git add . && git commit -m "results" && git remote add origin git@github.com:<you>/<you>.github.io.git && git push -u origin main
```

After the first push, `git add . && git commit && git push` from that directory
is enough.

**The one thing to watch.** If the ground truth or the caches change, previously
generated results predate them. The Experiment log flags those rows red as
`stale`, and the campaign refuses to reuse a run whose target fingerprint no
longer matches. Trust those two rather than memory.

---

## Two environments, not one

They cannot be merged. Nerfstudio pins torch 2.0.1+cu118, tiny-cuda-nn and
Python 3.8; GG-SSVT needs `transformers >= 4.56` for the DINOv3 and SAM model
classes, which will not install on Python 3.8.

```bash
conda env create -f ggssvt/environment.yml
```

```bash
conda activate ggssvt && pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -r ggssvt/requirements.txt
```

Keep the existing `cropcraft` environment for `ns-train` / `ns-viewer`.

### Verify before anything else

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print a CUDA build and `NVIDIA GeForce RTX 4080`. If `cuda.is_available()`
is False you installed a CPU wheel, and every command below will silently run on
the CPU at roughly a hundredth of the speed rather than failing.

Expected: `2.5.1+cu121 True NVIDIA GeForce RTX 4080`.

```bash
python -m pytest tests/ -q
```

**Expect `N passed, 7 skipped`. That is correct.** The seven are integration
tests in `test_pipeline.py` gated on the preprocessed cache, which does not exist
yet: `work_dirs/` is gitignored because it holds about 120 MB of derived data
that regenerates in minutes. They skip until step 1 has run, then join in.

Read the **failure** count, not the pass count. The total changes whenever tests
are added, so it is not a useful check:

| What you see | What it means |
|---|---|
| `0 failed`, 7 skipped | Correct, before preprocessing |
| `0 failed`, 0 skipped | Correct, after preprocessing |
| any failures | Fix before spending GPU hours |
| errors during *collection* | A module is missing, see below |

Collection errors reading `ModuleNotFoundError: No module named 'ggssvt.data'`
mean the repository is out of date. Check with `git ls-files ggssvt/data/`, which
must list five files, and `git log --oneline -1` against the remote.

---

## Before you start: request the two gated models

Both are manual approvals by Meta and can take hours to days, so submit them
first and let them run in the background.

- DINOv3, <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
- SAM 3 / SAM 3D Objects, <https://huggingface.co/facebook/sam-3d-objects>

```bash
hf auth login
```

Then check what this machine can actually reach:

```bash
python -m ggssvt.cli access
```

It prints the authenticated account, where the token came from, and an OK/BLOCKED
line per model. **Nothing needs a token added to it.** `hf auth login` writes one
to `~/.cache/huggingface`, and `huggingface_hub` and `transformers` pick it up on
their own, no function in this codebase takes a token argument, and no script
should ever contain one.

Two ways an approved model still reads BLOCKED:

**Wrong account.** Approvals are per account. If the settings page says ACCEPTED
but `access` reports BLOCKED, compare the name it prints against the account that
was approved. Fix with `hf auth login --force`.

**`HF_TOKEN` overriding the login.** If that variable is set, in `.bashrc`, a
job script, a conda activation hook. It **takes precedence over `hf auth login`
entirely**, so logging in again changes nothing. `access` reports which source is
in play. To clear it:

```bash
unset HF_TOKEN && python -m ggssvt.cli access
```

For headless or batch jobs, `HF_TOKEN` is the right mechanism, just make sure it
holds the approved account's token, and keep it out of anything committed.

Everything runs without either model; the DINOv3 cells simply report as skipped.

---

## Notes on the classical steps

Detail for the CPU steps above: what each one is for, what to watch, and
what the output should look like. The sequence itself is the section above;
nothing here introduces a step that is not already in it.

### The caches (steps 1 to 3)

`work_dirs/` is not in the repository, so on a fresh machine there is no cache at
all and everything downstream (baselines, probes, factorial, training) has
nothing to read.

**Both commands are required, and the first one is easy to skip.** The default
`--cache-dir` is `work_dirs/ggssvt/cache`, which only the *geometric* run
writes. Running only the SAM3D command leaves that directory empty, and the next
thing you run fails with `no quality report at .../cache/quality.json`.

```bash
python -m ggssvt.cli preprocess
```

```bash
python -m ggssvt.cli preprocess --segmenter sam3d --cache-dir work_dirs/ggssvt/cache_sam3d --sam-device cuda
```

Then confirm the caches are sane before building on them:

```bash
python -m ggssvt.cli inspect && python -m ggssvt.cli baselines
```

`inspect` audits the raw dataset and reports the missing calibration; `baselines`
prints the leave-one-out table the whole day is measured against. If `baselines`
does not roughly reproduce geometric features at **RMSE 0.544 / R² 0.030** and
direct 2D at **0.469 / 0.279**, the preprocessing differed from the reference run
and everything after it will too.

**Those are the post-V001–V008 numbers, and 2D being ahead is expected, though
the difference is not statistically resolved, so do not read it as a result.** If you see
geometric features at 0.397 / 0.526 you are running the old 28-specimen cache
without the V batch, check `dataset/ground_truth.csv` has `measured` in the
`pot_weight_source` column for V001–V008, and re-run `preprocess`.

Expect 36/38 specimens through the quality gate on the geometric cache. About 4 s/specimen for the
geometric pass and 5 s/specimen for SAM3D on the GPU, so under five minutes for
both. Then re-run the tests; the seven skipped ones should join in:

```bash
python -m pytest tests/ -q
```

### Time one epoch before committing to anything

```bash
python -m ggssvt.cli pretrain --epochs 3 --workers 8 --batch-size 2 --device cuda --out /tmp/timing.pt
```

Note the seconds per epoch. **Every estimate below assumes ~12 s/epoch** on this
card. Watch `nvidia-smi` while it runs: if utilisation sits low you are
dataloader-bound and should raise `--workers` before anything else. Scale every
duration below by whatever ratio you actually measure, do not trust these
numbers over your own stopwatch.

### The frozen-feature factorial (step 8)

```bash
python -m ggssvt.cli factorial --variant base --backbones cnn dinov2 dinov3
```

This is the cheap answer to "does DINO or SAM3D help", and it already ran:
DINOv2-base was the best cell, but nothing was statistically resolved at n=26.
Re-running adds the DINOv3 row if your access came through. Descriptors are
cached, so the DINOv2 cells return instantly.

### DINOv2-large, the best use of the extra VRAM

```bash
python -m ggssvt.cli dino-probe --variant large --backbones dinov2
```

Do this one. The probe already shows DINOv2 small to base improving
(RMSE 0.335 to 0.295), but a single pairwise delta at n=28 was not significant.
**A third scale turns two points into a trend**, and a monotone
small/base/large improvement is far more persuasive than any one comparison.
It is evidence the backbone is doing something rather than that one run got
lucky. ViT-L/14 is 300M frozen parameters; it did not fit comfortably in 8 GB and
fits easily in 16.

### View-count ablation (steps 3 and 7)

Answers "would four images do?" with a physical criterion rather than a quality
score. Build the reduced caches, then score them:

```bash
for v in 3 4 6; do python -m ggssvt.cli preprocess --views $v --cache-dir work_dirs/ggssvt/cache_v$v; done
```

```bash
python -m ggssvt.cli views
```

Expect **0 of 25 plausible at four views** against 8 of 36 at twelve, and a
median implied density of 9.2 kg/m³, lighter than polystyrene. Agreement only
falls from 0.608 to 0.424 over the same range, which is why it is worth printing
both.

### TSDF depth fusion (step 4)

The carve cannot be improved past the visual hull, which is why 25 of 36
specimens imply a bulk density one to two orders of magnitude below plant
tissue. Fusing the same depth maps as a signed distance field keeps concavities
and leaves unobserved space empty.

```bash
python -m ggssvt.cli fuse --write-cache
```

Expect **31 of 36 plausible against the carve's 8**, median implied density
529 kg/m3 against 116.8. `--write-cache` also writes `cache_tsdf` at the carve's
own 128^3 and 12 mm, which is the control that isolates the operator from the
grid: 25 of 36 at matched resolution, or 21 with the rim also held fixed, so the method accounts for most of the gain
and the finer grid adds the rest.

Two things depend on this having run. `cli baselines` offers the `fused geometry`
method only when every specimen is present in `reports/fusion.json`, and the
project page shows TSDF in its method toggle only when `cache_tsdf` exists. Do
not run `fuse --plants ...` on a subset and then expect either: a partial report
is treated as no report, deliberately.

### Mesh arm and reconstruction gallery (steps 6 and 10)

```bash
python -m ggssvt.cli mesh --export work_dirs/ggssvt/meshes
```

Marching cubes on the carved occupancy, then biomass from mesh descriptors,
scored under the same leave-one-out protocol as every other method, so the table
is directly comparable. It is currently the **best** method (RMSE 0.359,
R² 0.613, against 0.397 / 0.526 for the voxel features), though the paired
interval spans zero. `--export` also writes OBJ files for MeshLab.

Needs `scikit-image` and `scipy`, both in `requirements.txt`. Nothing else in
the pipeline depends on them.

```bash
python -m ggssvt.cli gallery
```

Contact sheets, PLY point clouds and a self-contained interactive HTML viewer of
every reconstruction under both segmenters. Worth five minutes before committing
to a day of training: it is how the "E001–E010 are mostly pot" problem became
visible in the first place, and it will show immediately if a preprocessing run
on this machine went wrong in a way the quality gate did not catch.

---

## What today's biomass numbers can and cannot say

Read this before interpreting anything the afternoon produces.

**Two things cap every biomass number, and both are now measured.**

*The batch confound, weakened but present.* E001–E010 average 0.538 kg and
E011–E020 average 1.844 kg, with almost no overlap; batch membership alone
explained **R² = 0.887**. V001–V008 was the batch that breaks it, 500–1800 g,
overlapping every other batch, and takes it down to **0.744** across the three
Eucalyptus batches, **0.697** across all four. Still high. The comparison is
partly measuring how well each method separates *size classes*.

*The reconstructions are envelopes, not plants.* Measured mass over reconstructed
above-ground volume implies a bulk density; fresh tissue is 300–900 kg/m³. Only
**8 of 36** specimens land in a generous 200–1000 band, 25 imply less (the hull
enclosing air between leaves; all ten Mango at 26–77) and 3 imply more (thin
stems never carved). No amount of training fixes a volume that is the wrong size
for its mass, and this, not any RMSE ordering, is what to report.

**The 3D-versus-2D RMSE comparison is not resolved in either direction.** Paired
bootstrap gives [−0.051, +0.227] on the current set and [−0.168, +0.099] on the
old n=28 one, and the winner flips depending on whether features are whitened
before the ridge. Do not quote an ordering from it.

Decide the framing now rather than at examination:
[RERUN_V_BATCH.md](RERUN_V_BATCH.md) §3 has the argument and the numbers.

---

## The Titan session: what has never run

**Checked 2026-09-02 against `work_dirs/`.** Everything classical is done and
every trained arm is not. `work_dirs/ggssvt/campaign/` does not exist, so no
condition in the campaign has ever produced a result, and two further GPU steps
have no report file.

| Arm | State |
| --- | --- |
| A, B, C space carving and fusion | run |
| F frozen backbones, both DINOv2 and DINOv3 | run, and it is a null |
| H surface voxels after Nombambela | run |
| **E GG-SSVT, all ten campaign conditions** | **never run** |
| **D pose-free reconstruction** | **never run**, no `posefree.json` |
| **Neural field** | **never run**, no `neural_field.json` |
| G single-image generative | not implemented, scoped only |

The ten conditions, from `campaign.py`:

```
baseline_cnn  baseline_fused  h2_no_geometry  h1_dinov2
h3_bands_8_freq7  h3_bands_6_freq6  h3_bands_16_freq10
sam3d_cnn  sam3d_dinov2  h1_dinov3
```

`h1_dinov3` is now runnable: access was granted on 2026-09-02 to account
`aamsb`. **Use `env -u HF_TOKEN`** on that machine if it also carries an
`HF_TOKEN` for a different account, because `huggingface_hub` prefers the
environment variable over the cached login and the run will report DINOv3 as
gated while looking otherwise healthy.

### Start it, then leave it

```bash
python -m ggssvt.run_all --device cuda --plan full --workers 8
```

That is the whole programme including the campaign, the pose-free step and the
neural field. It resumes rather than restarts, it tees each step to its own log,
and an optional step that fails does not stop the night. If the classical steps
are already fresh it skips straight to the GPU work.

### Running the conditions in parallel

`--only` takes a list, so the campaign shards across processes:

```bash
python -m ggssvt.campaign --plan full --device cuda --batch-size 4 \
  --only baseline_cnn baseline_fused h2_no_geometry > shard_a.log 2>&1 &

python -m ggssvt.campaign --plan full --device cuda --batch-size 4 \
  --only h1_dinov2 h1_dinov3 sam3d_cnn > shard_b.log 2>&1 &

python -m ggssvt.campaign --plan full --device cuda --batch-size 4 \
  --only h3_bands_8_freq7 h3_bands_6_freq6 h3_bands_16_freq10 sam3d_dinov2 \
  > shard_c.log 2>&1 &
```

**Three shards, not ten, and the reason is the OOM you already debugged.** That
failure was job collisions rather than a memory ceiling: the card reported
1.11 GiB reserved of 15.57 when it failed. Independent processes on one GPU each
size their allocator against the *whole* card, so four jobs that each fit
comfortably alone will still evict each other's cached blocks and fail at a
point none of them would fail at on their own. More shards is not more
throughput once they start contending; it is the same work plus a crash.

If you want a hard guarantee rather than a rule of thumb, cap each process:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  python -m ggssvt.campaign ... &
```

and give each shard a fraction of the card by setting
`torch.cuda.set_per_process_memory_fraction(0.30)` at the top of the run. Three
shards at 0.30 leaves headroom for fragmentation, which is what actually bites.

### Measured, so you can size it yourself

On a 4 GiB RTX 2050, one forward and backward of the GG-SSVT model at twelve
views and 224 pixels peaks at:

| batch | tokens/view | peak |
| --- | --- | --- |
| 1 | 16 | 0.73 GiB |
| 4 | 16 | 1.09 GiB |
| 4 | 32 | 1.50 GiB |
| 8 | 32 | 2.55 GiB |

DINOv3 uses marginally *less* than DINOv2 at every setting. On a 15.57 GiB Titan
batch 8 at 32 tokens leaves about 13 GiB spare for one job, which is why three
concurrent shards is comfortable and ten is not.

### What a week buys, and what it does not

The campaign at default epochs is roughly 112 hours of finetuning per condition
on a small card, less on the Titan. A week is enough. But two of these are worth
running for the record rather than for the answer, and it is better to know that
before you start:

- **`h1_dinov3` cannot resolve what it is asking.** The frozen probe bounds
  DINOv3 against DINOv2 at 0.030 kg, and the resolution ledger puts the smallest
  detectable effect at n=36 at 0.138 kg. Finetuning would have to amplify the
  difference more than fourfold before this sample could see it. Run it to close
  the arm, not expecting a result.
- **The comparison that can pay** is `baseline_cnn` against `baseline_fused`. It
  asks whether a trained model inherits the 0.209 kg the classical features
  gained when the reconstruction operator changed, and that effect is larger than
  the detection floor.

`posefree` and the neural field are different: both are unrun and neither has a
prior result to compare against, so anything they produce is new.

---

## Multi-day: the training campaign

With more than a day on the machine, do not drive the training by hand. One
command queues the whole programme, writes a JSON result per run, and **skips
whatever already finished**, so a campaign that dies at 3am is restarted with
the same command and picks up where it stopped.

```bash
python -m ggssvt.campaign --plan smoke --device cuda
```

**Always run `smoke` first.** Two two-epoch runs over four specimens, about ten
minutes on the GPU, and it proves
the loop, the checkpointing, the resume logic and the result writing before you
commit a night to them.

```bash
python -m ggssvt.campaign --plan core --device cuda --workers 8 --batch-size 2
```

```bash
python -m ggssvt.campaign --plan full --device cuda --workers 8 --batch-size 2
```

`--list` prints a plan without running it; `--only NAME [NAME...]` runs a subset;
`--force` re-runs something already marked done.

### What the plans contain, and which question each run closes

Runs are ordered so the hypothesis-answering ones come first, if the campaign is
cut short, what completed is what the write-up needs.

| run | closes | why |
|---|---|---|
| `baseline_cnn` | reference | every comparison below is against this |
| `baseline_fused` | operator | same model on the fused cache. Classical features gained 0.209 kg there with the operator as the only change; whether a trained model inherits that is untested |
| `h2_no_geometry` | **H2** | the geometry-grounding ablation; γ frozen at zero, capacity identical |
| `h1_dinov2` | **H1** | ViT backbone against the CNN stem, inside the trained model |
| `h3_bands_8_freq7` | **H3** | encoding matched to the grid Nyquist, 41.7 cyc/m |
| `h3_bands_6_freq6` | **H3** | half the Nyquist, 20.8 cyc/m, expected to hurt |
| `h3_bands_16_freq10` | **H3** | 8× the Nyquist, 333 cyc/m, expected to add nothing |

**Run `cli fuse --write-cache` before the campaign.** Without `cache_tsdf` the
`baseline_fused` run has nothing to read and fails rather than being skipped.

**Delete `work_dirs/ggssvt/campaign/` before the first re-run.** Resume works by
skipping runs whose status is `done`, which is what you want after a crash and
exactly what you do not want after the targets changed.
| `sam3d_cnn`, `sam3d_dinov2` | factorial | tests whether the probe's interaction survives training |
| `h1_dinov3` | H1 | if access has been granted |

`core` is seven runs, roughly **8–10 hours** at 120 pretrain and 60 fine-tune
epochs. `full` is nine, roughly 13–15 hours. Time one epoch first (step 2) and
scale.

### What to report back from each run

The campaign writes `summary.txt` and a per-run JSON carrying everything below.
For the research status document you need, per run:

- **RMSE, MAE, MARE, R²** with the bootstrap interval on RMSE
- **occupancy AP and best-threshold IoU**: reconstruction quality, threshold-free
- **parameter count**: total and trainable, H3's efficiency claim needs it
- **the paired bootstrap against `baseline_cnn`**: not the two point estimates

And three things that are not in the JSON and must be recorded by hand:

1. **`model.fusion.distance_scales()` after training.** Per-head, per-block γ. If
   it collapses toward zero the distance bias is not being used, and that is a
   finding about H2 regardless of which way the RMSE goes.
2. **Seconds per epoch and total GPU hours** per condition.
3. **Any run that failed or was skipped**, and why. A gated backbone leaving a
   hole in the factorial is a result about availability, not a gap to hide.

---

## Notes on the GPU steps

Budget roughly **three hours** for this block on this card. Do not try to run the full 2×3
factorial with training today; at these settings that is six pretrain runs plus
six LOOCV sweeps.

### Pretraining a single cell, by hand

Stage 1 is self-supervised and uses no mass labels, so one pretrain per cell
serves that cell's whole LOOCV sweep.

```bash
python -m ggssvt.cli pretrain --epochs 120 --workers 8 --batch-size 2 --device cuda --out work_dirs/ggssvt/checkpoints/geo_cnn.pt
```

```bash
python -m ggssvt.cli pretrain --epochs 120 --workers 8 --batch-size 2 --device cuda --cache-dir work_dirs/ggssvt/cache_sam3d --out work_dirs/ggssvt/checkpoints/sam_cnn.pt
```

The full 120 epochs, not the 60 the 8 GB plan called for, on this card you can
afford the config default, so there is no deviation to justify in the methods
section. That is worth more than it sounds: one fewer "reduced for compute
reasons" caveat in the write-up.

For the DINO cells, edit `backbone` in `ggssvt/config.py` to `"dinov2"`, or run
the combined command in step 6 which handles it.

### One LOOCV sweep, to calibrate the cost

```bash
python -m ggssvt.cli loocv --checkpoint work_dirs/ggssvt/checkpoints/geo_cnn.pt --finetune-epochs 60 --workers 8 --device cuda --out work_dirs/ggssvt/reports/folds_geo_cnn.json
```

60 fine-tune epochs rather than 25. Still well short of the 200 default, and that
is defensible on more than time: the biomass head is initialised as an exact
physical model (density × volume, zero residual), so it starts near a sensible
solution rather than at random. Check the fold errors have actually flattened,
if they are still falling at 60, raise it, because you now have the headroom.

Time this. It sets whether step 6 fits before you go home.

### The trained factorial, by hand

```bash
python -m ggssvt.cli factorial --train --backbones cnn dinov2 --variant base --epochs 120 --finetune-epochs 60 --workers 8 --batch-size 2 --device cuda
```

Four cells: geometric×cnn, geometric×dinov2, sam3d×cnn, sam3d×dinov2. It prints
the same paired-effect table as the frozen probe, so the two are directly
comparable, **and that comparison is itself a result.** If the trained factorial
reproduces the probe's sign flip (SAM3D alone hurts, SAM3D given DINO helps), the
finding survives moving from pooled descriptors to the full model. If it does
not, the probe was measuring an artefact of pooling.

Add `dinov3` to `--backbones` only if you have both the access and the hours.

---

## Pose-free reconstruction, the registration check

> **Read [POSEFREE.md](POSEFREE.md) first.** All three backends are installed and
> their APIs verified against the real code; the install instructions that were
> here before were wrong for two of the three (DUSt3R and MASt3R ship no
> `setup.py`, so `pip install -e .` fails). That document has the working
> dependency set, the Python 3.11 constraint that `open3d` imposes, and the run
> order, Fast3R first, because it is twenty times cheaper and exercises the same
> comparison code.

```bash
python -m ggssvt.cli posefree --check-only
```

Reports which of DUSt3R, MASt3R and Fast3R are installed, with the clone command
for each. All three sets of weights are open on HuggingFace; it is the
repositories that must be cloned, with submodules:

```bash
git clone --recursive https://github.com/naver/mast3r && cd mast3r && pip install -r requirements.txt && pip install -e .
```

Then:

```bash
python -m ggssvt.cli posefree --methods mast3r --device cuda
```

**Run this for the poses, not for the biomass.** Every camera pose in the project
is estimated from depth, and the azimuth refinement saturates its ±8° search
bound on almost every specimen, so the registration is the least verified
assumption in the pipeline, and nothing inside the pipeline can test it. A
pose-free method shares no failure mode with a depth-based registration, so the
residual after similarity alignment is a real measurement of how wrong the poses
are. **Azimuth RMSE is the number to read.** Under 8° would be reassuring; more
would not be.

Use **MASt3R's metric checkpoint** first. DUSt3R and Fast3R return arbitrary
scale, so their volumes depend on a scale recovered from the Kinect depth, which
makes them only partly independent of the pipeline they are meant to check.
MASt3R-metric has no such loop.

The adapters are written against each project's documented interface but have
**not been run against the real weights**: no GPU was available where they were
written. The maths they feed is tested; the adapters are not. Watch the first run
for import errors and convention drift rather than assuming quiet success.
`sanity_check_result` warns if the cameras stop facing the scene, which is what a
convention change looks like.

---

## Evening, Nerfstudio, in the other environment

```bash
conda activate cropcraft
```

The vendored copy in `nerfstudio/` does not import: `.gitignore` line 1 is
`data/`, which matched `nerfstudio/nerfstudio/data/` and kept the dataparsers out
of the repository. Install upstream instead:

```bash
pip install nerfstudio
```

**Export the poses first, `ns-train` has nothing to read without them.** This
runs in the `ggssvt` environment, not `cropcraft`, and writes `transforms.json`
into each specimen directory:

```bash
conda activate ggssvt && python -m ggssvt.cli nerfstudio && conda activate cropcraft
```

Skip it and `ns-train` fails on a missing `transforms.json`, which reads like a
Nerfstudio problem rather than a missing step.

Then, on one good specimen and one bad one:

```bash
ns-train splatfacto --data dataset/plants/M003 --pipeline.model.camera-optimizer.mode SO3xR3 --experiment-name M003
```

```bash
ns-train depth-nerfacto --data dataset/plants/E001 --pipeline.model.camera-optimizer.mode SO3xR3 --experiment-name E001_depth
```

M003 reconstructs well and E001 is nearly all pot, so the pair tells you whether
a radiance field recovers canopy the carve missed, or fails the same way.

**Always pass the camera optimiser**, then diff the optimised poses against the
exported ones. The azimuth refinement saturates its ±8° bound on almost every
specimen, so a photometric estimate of the same poses is an independent
measurement of how wrong the registration is, the single most valuable number
available for the price of one training run.

---

## Spending 16 GB

You should not hit an OOM at the settings above. If a run finishes early, the
headroom is best spent in this order:

| Option | Why |
|---|---|
| `--variant large` in the probe | Third scale point turns a delta into a trend |
| `--tokens-per-view 96` | 64 tokens covers ~4% of frame; more captures thin branches |
| `--batch-size 4` | Fewer, better-conditioned gradient steps |
| `dinov3` in the factorial | Only if access came through |

Raising `--tokens-per-view` is the one most likely to change a *result* rather
than a runtime: the encoder keeps the K best subject patches per view, and thin
eucalyptus stems are exactly what falls off the bottom of that ranking.

### If something does OOM anyway

| Symptom | Fix |
|---|---|
| OOM during fine-tuning | drop `--batch-size` to 1 |
| OOM in the decoder | lower `query_chunk` in `config.py` to 8192 |
| OOM with `--variant large` | fall back to `base` |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with fragmentation
across the 28 folds of a LOOCV sweep.

---

## End of day, collect the results

```bash
python -m ggssvt.cli report --folds work_dirs/ggssvt/reports/folds_geo_cnn.json
```

Writes `report.md`, `comparison.csv`, `metrics.json` and a predicted-versus-true
scatter to `work_dirs/ggssvt/reports/`. Pass `--folds` to fold a GG-SSVT
leave-one-out run into the comparison table alongside the baselines; without it
you get the baselines only.

```bash
python -m ggssvt.cli visualise --plants M003 E001 E011
```

Rig overlays (the world axis projected into every view) and mask overlays. Run
these if any registration looks suspect, if the red axis is not on the plant
stem in all twelve frames, every number for that specimen is meaningless.

---

## Every command

| Command | Needs | Roughly |
|---|---|---|
| `inspect` | nothing | seconds, dataset audit, no computation |
| `access` | network | seconds, HuggingFace account and model access |
| `preprocess` | dataset | ~6 min geometric; SAM3D is ~4 min on GPU, ~1 h on CPU |
| `preprocess --views N` | dataset | ~5 min per view count |
| `baselines` | cache | seconds, LOOCV baseline table |
| `mesh` | cache, scikit-image | ~1 min including the comparison |
| `gallery` | cache | ~2 min, contact sheets, PLY, HTML viewer |
| `visualise` | dataset | seconds per specimen |
| `dino-probe` | cache, network | 5–15 min depending on variant |
| `dino-segment` | cache, network | ~9 min, DITR-style feature lifting |
| `fuse --write-cache` | cache | ~11 min, TSDF fusion plus the fused cache |
| `views` | the view caches | seconds, scores whichever exist |
| `quality` | both caches | ~2 min, re-projection and cross-operator metrics |
| `gate` | cache | seconds, acceptance checks; non-zero exit when blocked |
| `architecture` | nothing | seconds, one SVG per methodology |
| `factorial` | both caches | ~10 min frozen; hours with `--train` |
| `pretrain` | cache, **GPU** | ~25 min at 120 epochs |
| `loocv` | cache, checkpoint, **GPU** | ~40 min at 60 fine-tune epochs |
| `experiment` | cache, **GPU** | one pretrain + one LOOCV per backbone; superseded by `ggssvt.campaign` |
| `nerfstudio` | dataset | seconds, writes `transforms.json` |
| `dashboard` | cache, mesh cache | ~1 min, the walkthrough page |
| `posefree` | cloned repos, **GPU** | minutes per specimen per method |
| `ggssvt.campaign` | cache, **GPU** | 8–15 h depending on plan |
| `report` | cache | seconds |

`experiment` is the single-factor backbone comparison; `factorial` supersedes it
by crossing backbone with segmenter. Use `experiment` only when you want
backbones alone on one cache.

---

## What to record as you go

Because these are deviations from the config defaults, and the methods section
needs them:

- epochs actually used for stage 1 and stage 2, and why
- the GPU is an RTX 4080 (16 GB), not the 4060 the first draft assumed
- seconds per epoch, and total GPU hours per condition
- which cells were skipped for gated access
- the transductive-vs-`--strict` distinction, if you run the strict protocol

Everything writes JSON alongside its output, so the numbers survive the day.
