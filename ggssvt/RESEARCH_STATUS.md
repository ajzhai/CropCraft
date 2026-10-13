# Research questions and hypotheses: what is answered, what is not

Against the May 2025 MEng proposal (*Automated Biomass Estimation using
Self-Supervised Vision Transformers*), Section II.A and II.B.

**Read this first.** The GG-SSVT model itself **has never been trained.** Every
number below comes from the geometry pipeline, frozen pretrained features, or
classical baselines. That is not a gap in the plan: the pipeline had to exist
and be validated before training was worth an hour of GPU, but it means no
hypothesis about *the trained model* can be marked answered yet, and several
below are marked pending purely for that reason.

**The second thing that constrains every answer** is the batch confound.
Knowing only which capture batch a Eucalyptus specimen came from explains
R² = 0.887 of its mass variance, more than any method achieves; within either
batch no method reaches R² = 0.2. So every biomass result currently measures
*size-class separation*, not mass estimation among comparable plants. This is
stated once here and applies throughout.

---

## Summary

| | Status | Blocker |
|---|---|---|
| **RQ1** SSL ViTs learn plant representations | Partial | Needs training; confound caps the label-efficiency claim |
| **RQ2** grounding and fusion for 3D reconstruction | Partial | Architecture built and validated, never trained |
| **RQ3** reconstruction → biomass, label-efficiently | **Answered, and now positively** | The carve was the bottleneck; fusing the same depth resolves the first real improvement |
| **H1** ViT beats CNN; label efficiency | Partial | DINOv2 probe done, not significant at n=28; re-running at n=36 |
| **H2** geometry grounding → viewpoint consistency | Partial | Ablation implemented, not run |
| **H3** frequency + geometry grounding | **Best evidenced of the four** | 4 of 6 sub-claims established; see [HYPOTHESIS_3.md](HYPOTHESIS_3.md) |
| **H4** robustness to occlusion, noise, sparse sampling | Partial | Sparse-sampling clause answered; occlusion/noise not |

---

## RQ1: Can self-supervised ViTs learn plant representations capturing geometric structure and physiological traits, with minimal labelled data?

**Status: partial.**

### What is established

A frozen DINOv2 descriptor, pooled over subject patches only and probed
linearly under leave-one-out on 28 specimens:

| representation | RMSE | R² | dims |
|---|---|---|---|
| no DINO (hand-built geometry) | 0.358 kg | 0.616 | 7 |
| DINOv2-small | 0.335 kg | 0.663 | 768 |
| **DINOv2-base** | **0.295 kg** | **0.738** | 1536 |

Paired bootstrap against the control: **ΔRMSE −0.062 kg, 95% CI
[−0.160, +0.036], p ≈ 0.22, not significant.**

Two secondary observations. Adding hand-built geometric features on top of the
DINO descriptor changes nothing (0.295 → 0.296), so the pretrained
representation already contains whatever those features encode. And the result
survives every dimensionality-reduction setting tried (PCA-4, PCA-8, untruncated
ridge), so it is not an artefact of the probe.

### What is not established

The claim has two halves and only one is tested. *Rich representations*, yes,
the direction is consistent and scales with backbone size. *With minimal
labelled data*, untested, because there is no comparison against a fully
supervised model trained on the same 28 specimens. At n=28 a supervised model
cannot be trained meaningfully anyway, which is itself worth saying.

"Physiological traits" is not addressed at all. Nothing in the pipeline measures
a physiological trait; mass is a bulk property.

### How to close it

1. **Run DINOv2-large** (`dino-probe --variant large`). Open weights, ~15 minutes.
   Small → base → large as a monotone trend is far more persuasive at n=28 than
   any single pairwise delta, because a consistent trend across three scales is
   evidence the backbone is doing something rather than one run getting lucky.
   **This is the highest value-per-minute experiment available.**
2. **Run the trained comparison** (`factorial --train --backbones cnn dinov2`).
   The probe bounds what the representation holds when pooled per specimen;
   GG-SSVT uses it per patch with 3D anchors, which is a different question.
3. **Reframe or drop "physiological traits."** Either add a measurable trait, or
   narrow the question to geometric structure, which is what the work does.

---

## RQ2: What transformer-based grounding and fusion strategies enable accurate 3D reconstruction of plant architecture?

**Status: partial. Architecture complete and validated; never trained.**

### What is established

The architecture exists and its claimed mechanisms are individually verified by
test rather than asserted:

- **Fourier back-projected positional encoding**: tokens positioned by world
  coordinate, not image index. Anchors verified to sit on the plant
  (radial extent 0.21–0.41 m, heights 0.06–1.03 m).
- **3D-distance-biased cross-view attention**: a token attends to a 2 cm
  neighbour more than 100× as strongly as to a 1 m one, pinned by test.
- **Implicit occupancy decoder**: chunked evaluation proven identical to a
  single pass (max difference 4.8 × 10⁻⁷), so a full 128³ grid fits on one GPU.
- **Space-carving self-supervision**: occupancy targets with no manual labels.

Reconstruction quality is measured without ground truth, using multi-view
agreement and surface coverage: mean agreement 0.625, coverage 0.788 across 28
specimens.

**A genuine methodological contribution emerged that was not in the proposal.**
`dataset/calib` is empty, no ChArUco calibration was ever captured, so the
extrinsics are recovered from the depth itself: floor plane per view for tilt,
roll and height; a subject-axis hypothesis chosen by cross-view agreement; then
azimuth refinement. This is calibration-free rig registration and it is a
reportable method in its own right.

### What is not established

Nothing about *accuracy*. There is no surface ground truth for any specimen, so
"accurate 3D reconstruction" cannot currently be measured, only self-consistency
can. The comparison between grounding strategies (the actual question) requires
the trained ablation, which has not run.

### How to close it

1. **Train it**, `pretrain` then `loocv`, and the `--no-geometry` ablation that
   zeroes both geometric pathways while leaving capacity identical.
2. **Get an external reconstruction to compare against.** Splatfacto via the
   Nerfstudio export, or MASt3R-metric. Neither gives ground truth, but a second
   method with different failure modes is the closest available substitute.
3. **Read the learned distance scales.** `model.fusion.distance_scales()` after
   training is direct evidence for whether the geometry bias is used at all, if
   γ collapses toward zero the mechanism is not contributing, and that is a
   finding.

---

## RQ3: Can the reconstructed 3D representation predict biomass label-efficiently, and how does reconstruction quality relate to estimation accuracy?

**Status: answered, but by a physical measurement, not by a model comparison.**

> **This section was rewritten after V001–V008 was added.** The n=28 table below
> is kept as the record of what was run then; the current result is the n=36 one
> beneath it, and it points the other way. Full account:
> [RERUN_V_BATCH.md](RERUN_V_BATCH.md).

### The current result: 36 specimens, measured pot weights

| method | RMSE | MARE | R² |
|---|---|---|---|
| **direct 2D (no 3D)** | **0.469 kg** | 42.8% | **+0.279** |
| mesh geometry | 0.507 kg | 37.4% | +0.157 |
| geometric features (voxel) | 0.544 kg | 43.8% | +0.030 |
| mean predictor | 0.568 kg | 57.8% | −0.058 |
| volume allometric | 0.592 kg | 53.0% | −0.150 |
| canopy area allometric | 0.598 kg | 47.3% | −0.170 |

**Do not read that ordering as "2D beats 3D".** A paired bootstrap puts the
3D−2D difference at +0.075 kg with a 95% interval of [−0.051, +0.227], not
resolved, and the ordering flips if the features are whitened before the ridge
rather than merely standardised (3D 0.458 against 2D 0.491, also not resolved).
The same test on the original n=28 condition gives [−0.168, +0.099], so the
earlier "reconstruction beats pixels" was never resolved either. **At these
sample sizes the comparison cannot be settled by RMSE differences.**

What *can* be settled is the mechanism, because it is a measurement rather than a
difference in means: dividing measured mass by reconstructed above-ground
volume gives an implied bulk density, and against the 300–900 kg/m³ of fresh
tissue only **8 of 36** specimens land in a generous 200–1000 band. Twenty-five
imply *less*. The visual hull has enclosed the air between leaves and branches,
with all ten Mango at 26–77 kg/m³, and three imply more, meaning thin stems were
never carved. **The reconstructed object is a canopy envelope, not a plant**, so
its volume cannot carry mass information regardless of what is fitted to it.

**The answer changed on 2026-08-24, and it is worth stating why.** Everything
below was measured on a space carve, and the carve was the limiting component.
Replacing silhouette intersection with TSDF depth fusion, holding the features,
the protocol, the grid, the views and the masks all fixed, moves geometric
features from 0.544 / +0.030 to **0.335 / +0.632**, with a paired bootstrap of
−0.209 [−0.363, −0.066]. **Resolved**, and the first resolved biomass improvement
in the project. Volume allometry moves from −0.150 to +0.278, also resolved.
Direct 2D is unchanged at 0.469, which is the control it has to be.

So RQ3 has a positive answer after all, but a narrower one than the proposal
asked for. Reconstructed geometry does predict biomass better than pixels here,
*provided the reconstruction is not a visual hull*. The hull was never going to
work, for the reason in the next paragraph, and the fix was to change the
operator rather than the regressor. See [FINDINGS.md](FINDINGS.md) section 7c.

**Within species the failure is unambiguous.** Restricted to Eucalyptus (n=26,
the least confounded set available), geometric features score R² **−0.313**,
*worse than predicting the mean*, whose floor is −0.082, while direct 2D reaches
+0.311. Mango alone is the same, −1.971 against the mean's −0.235. Falling below
the mean-predictor floor is a stronger statement than losing a head-to-head: the
feature carries no usable mass signal at this resolution, which is exactly what
an envelope-shaped volume predicts.

This is a real answer to RQ3 rather than a failure to answer it, and it is the
form the answer should take. It does not depend on n, on a regulariser, or on
which method happened to win a run. Stated for the dissertation:

> *Twelve-view visual-hull reconstruction at 12 mm resolution recovers the canopy
> envelope rather than the plant for these species: only 8 of 36 specimens have a
> reconstructed above-ground volume whose implied bulk density is within an order
> of magnitude of plant tissue. Its volume therefore cannot carry mass
> information, and no difference in fitted RMSE against image-only regression is
> resolved at this sample size in either direction.*

### What was established at n=28, before the V batch

Leave-one-out over 28 specimens, identical protocol for every method:

| method | RMSE | MARE | R² |
|---|---|---|---|
| **mesh geometry** | **0.359 kg** | 27.7% | **0.613** |
| geometric features (voxel) | 0.397 kg | 32.8% | 0.526 |
| direct 2D (no 3D) | 0.440 kg | 39.9% | 0.419 |
| mean predictor | 0.598 kg | 62.5% | −0.075 |
| volume allometric | 0.622 kg | 56.9% | −0.162 |
| canopy area allometric | 0.642 kg | 52.1% | −0.236 |

~~**Reconstruction helps.**~~ **Withdrawn.** The 0.397-against-0.440 gap was
never statistically resolved (paired bootstrap 95% CI [−0.168, +0.099]) and it
was reported without that interval. On the enlarged set the point estimate goes
the other way and is equally unresolved.

**Two negative results that are worth as much as the positive one:**

*Volume allometry is worse than predicting the mean* (R² = −0.162). Carved hull
density varies roughly tenfold between a bushy mango and a thin eucalyptus, so a
single volume-to-mass law cannot hold across morphologies. This is the empirical
motivation for a learned density rather than a fixed one.

*Surface area does not beat volume.* The hypothesis was that leaf mass scales
with area while enclosing no volume, so canopy area should win. The single-term
area law is the **worst** method tried, and head-to-head against the volume law
the difference spans zero. The mechanism generalises: **a visual hull's surface
area is envelope area, not leaf area**, twelve views at 12 mm voxels cannot
resolve individual leaves. The implied-density result above extends this from the
hull's surface to its volume, and is what turned the qualification into the
answer.

### The qualification

Leave-one-feature-out on the mesh set shows **height** carrying the result
(removing it costs 0.051 kg RMSE; removing canopy area costs 0.001). Height,
with solidity, is separating the two Eucalyptus batches, and batch membership
alone explained R² = 0.887. So the honest reading of "reconstruction improves
biomass estimation" was **"reconstruction separates size classes better than
pixels do"**.

V001–V008 tested exactly that reading and confirmed it. Its masses span 500–1800
g, overlapping every other batch instead of forming a cluster, which drops the
batch-only R² to 0.744 across the Eucalyptus batches and 0.697 across all four,
and removes the 3D advantage along with it. The size-class explanation was
right.

The label-efficiency half is untested for the same reason as RQ1.

### How to close it

1. **Collect a continuous mass range within one species.** This is the single
   change that would convert a qualified answer into a clean one, and it is data
   collection, not method work. More specimens of the two existing clusters will
   not help.
2. **Weigh the pots**, and measure their heights. Every `pot_weight_source` reads
   `estimated`, and the fixed `POT_HEIGHT_M = 0.28` sits *below* the real rim on
   E001–E010, so roughly half their 3.5 L "above-ground" volume is pot. Cheap,
   and it cleans up the target and the feature at once.
3. **Report the relationship, which is the second half of the question.** Plot
   per-specimen reconstruction quality (agreement, coverage) against biomass
   error. That relationship is answerable *now* with what is cached, and the
   proposal explicitly asks for it.

---

## H1: Geometry-grounded ViTs outperform CNNs; label-efficient biomass prediction

**Status: partial.** Evidence is the DINOv2 probe above, consistent direction,
not significant at n=28. The CNN-versus-ViT comparison inside the trained model
has not run.

**H1 bundles two separable claims** (architecture superiority, and label
efficiency) and should be split before the dissertation. The publication plan
already flags this.

**Note the proposal's phrasing is unsupportable as written.** It claims ViTs will
outperform CNNs "in segmentation accuracy", nothing in this work measures
segmentation accuracy against labelled masks, and no such labels exist.

---

## H2: Geometry-grounded models show higher viewpoint consistency and better reconstruction

**Status: partial, and the supporting evidence is stronger than expected.**

Multi-view agreement improves monotonically with view count
(0.372 → 0.447 → 0.550 → 0.635 for 3, 4, 6, 12 views), and usable specimens rise
with it (17 → 18 → 26 → 28). Registration refinement raises surface coverage
substantially (E011 0.29 → 0.48; M001 0.16 → 0.43).

But this measures *the geometry pipeline*, not *geometry grounding in the
model*, which is what H2 claims. The `--no-geometry` ablation exists and is
tested; it has not been run.

**How to close it:** run `factorial --train` with and without geometry grounding,
and report multi-view agreement plus occupancy AP for each.

---

## H3: Frequency and geometry grounding together improve parameter efficiency

**Status: keep. Four of six sub-claims established. Full treatment in
[HYPOTHESIS_3.md](HYPOTHESIS_3.md).**

**An earlier draft of this document recommended dropping H3, on the grounds that
frequency grounding was never implemented. That was wrong.** The proposal's own
wording, "especially as relates to positional encodings for 3D spectral
features", names the Fourier positional encoding, which *is* implemented and
used by both the token embedding and the occupancy decoder. What was never built
is a separate wavelet branch, which is not what the hypothesis requires.

Established on collected data:

- **Plant spectra are structure-specific.** High-frequency energy doubles from
  the smooth potted specimens (0.128) to branching ones (0.254–0.273).
- **The encoding is over-provisioned**: it reaches 83.3 cycles/m while the 12 mm
  voxel grid tops out at 41.7. Half the ladder describes detail the grid cannot
  carry, the parameter-efficiency claim as a measured quantity.
- **Mango saturates the grid Nyquist** at 41.7 ± 0.0 cycles/m for all ten
  specimens. Resolution, not method, is their binding constraint.
- **Angular sampling requirement is structure-specific and independent of
  bandwidth.** Mango has the highest bandwidth yet all ten are usable at three
  views; E001–E010 has the lowest yet only one in ten is. Bresolin et al.'s
  phenotype-specific acquisition-frequency finding, reproduced in another
  modality, and the direction is the opposite of what the spectra predict.

Pending, one GPU run each: trimming the encoding to the grid Nyquist should cost
nothing (`h3_bands_*` in the campaign), and the Frequency Principle predicts
high-frequency bands converge last, which `eval/frequency.band_error` measures
per epoch.

## H4: Robustness to occlusion, noise, and sparse sampling

**Status: the sparse-sampling clause is answered. The others are not.**

**Sparse sampling: answered, negatively, and now with a physical criterion.**

| views | usable | agreement | mean above-ground hull | **plausible** | median kg/m³ |
|---|---|---|---|---|---|
| 3 | 23/38 | 0.360 | 99.3 L | **1/23** | 9.8 |
| 4 | 25/38 | 0.424 | 126.5 L | **0/25** | 9.2 |
| 6 | 34/38 | 0.521 | 150.4 L | **2/34** | 15.4 |
| **12** | **36/38** | **0.608** | **10.4 L** | **8/36** | **116.8** |

**At four views, zero of twenty-five reconstructions can physically weigh what
the plant weighs**, a median implied bulk density of 9.2 kg/m³, lighter than
expanded polystyrene and thirty to ninety times below plant tissue, from hulls
averaging 126 L for plants of at most 2.35 kg.

Four views at 90° is the visual-hull minimum for a *convex* object, and a plant
is the opposite of convex: every unsampled azimuth leaves a prism of empty space
uncarved. The 12-view protocol is justified, and the predecessor project's
4-view protocol was not merely under-sampling but reconstructing something other
than the plant.

Note that the agreement score would never have shown this. It degrades gently,
0.608 to 0.424, while the reconstructions stop being reconstructions, which is
an argument for reporting the plausibility check alongside any geometric quality
metric.

**Occlusion: not measured**, though the pipeline is built around it (depth
carving treats occluded space as unobserved rather than empty, per Laurentini's
visual-hull bound).

**Noise: not measured.** The depth margin scales with z² to absorb Kinect range
noise, but no noise-injection experiment has run.

**H4 also mixes three separable claims** and should be split or narrowed to the
one with evidence.

---

## Recommended scope change

The proposal carries four hypotheses and three journal papers. Stander's June 2025
note called that PhD-sized and warned of delayed completion; fourteen months on
the evidence supports him.

| Hypothesis | Recommendation |
|---|---|
| H1 | **Split.** Keep the representation claim; drop "segmentation accuracy" (unmeasurable here) |
| H2 | **Keep.** Closest to being cleanly answerable, needs one GPU run |
| H3 | **Keep, restated.** See HYPOTHESIS_3.md, 4 of 6 sub-claims already established |
| H4 | **Narrow to sparse sampling**, which is answered |

That leaves a defensible set of two-and-a-half hypotheses matched to evidence
that exists or is one GPU day away, and it answers Stander's reservation by
describing what the work is rather than what was hoped for.

Two further corrections the proposal needs regardless: the sensor is **Kinect
v2, not Intel RealSense**, and the ground truth is **fresh mass, not oven-dry
above-ground biomass**. Both appear in the abstract.
