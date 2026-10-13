# Everything run, and what it means

A record of every experiment executed, what it showed, and what follows for the
research. Companion to [RESEARCH_STATUS.md](RESEARCH_STATUS.md), which maps this
onto the proposal's questions and hypotheses.

> **Superseded in places by [RERUN_V_BATCH.md](RERUN_V_BATCH.md).** V001–V008
> were added with *measured* pot weights and the pipeline re-run on 36
> specimens. Two results reversed: 3D geometric features no longer beat
> image-only regression, and the batch confound fell from R² 0.887 to 0.697.
> Where a number below is marked n=28 or n=30, the n=36 value in that document
> is the current one.

**The headline, stated once.** The pipeline is built, validated and reproducible
across two machines. The GG-SSVT model **has not been trained**, every number
here comes from the geometry pipeline, frozen pretrained features, or classical
baselines. And a batch confound caps what any of the biomass numbers can claim.

---

## 1. Data audit

| | |
|---|---|
| Specimens | 38 captured, **36 usable** (geometric), was 30/28 before V001–V008 |
| Views | 12 per specimen, 30° apart, dual Kinect v2, 512×424 RGB-D |
| Species | Eucalyptus ×28, Mango ×10, Xylem ×1 (excluded, 2 views) |
| Mass range | 0.20 – 2.35 kg fresh |
| Pot mass | **measured** for V001–V008, estimated for the rest (biased −10.9%) |

### Problems found in the capture set

**Two camB naming conventions.** `collect_specimen.py` names each camB file with
*camA's* step angle (`camB_000`…`camB_150`) while `dataset/README.md` documents
`camB_180`…`camB_330`. Both are present in the data. Taking the filename
literally places camB on top of camA and **silently collapses the 12-view rig
into a 6-view one**. Resolved in `data/naming.py`; the collection script should
be fixed at source.

**No calibration exists.** `dataset/calib` is entirely empty, no ChArUco
intrinsics, no per-day `rig_positions.json`. `make_transforms.py` has never had
an input.

**Ground truth caveats.** Every `pot_weight_source` reads `estimated`. The target
is as-collected *fresh* mass, not oven-dry above-ground biomass. E008's species
carries a stray apostrophe. E001 has an orphan 13th image outside its manifest.

**Tall specimens are truncated.** At the ~1 m working radius the vertical field
of view reaches about 1.15 m; E011–E020 run off the top of frame, so their
carved volumes are underestimates.

---

## 2. Calibration-free rig registration

Because no calibration was captured, extrinsics are recovered from the depth:
RANSAC floor plane per view (tilt, roll, camera height), a subject-axis
hypothesis chosen by **cross-view agreement**, then azimuth refinement by
coordinate descent.

**Result:** mean multi-view agreement 0.625, surface coverage 0.788. Camera
heights recover consistently to ~3 cm across a sweep, an independent check the
fit is working. Refinement raises coverage substantially, E011 0.29 → 0.48,
M001 0.16 → 0.43.

**One failure worth recording.** An earlier version picked the strongest
*single-view* subject candidate and locked onto background structure a metre
behind the plant on several specimens, producing a confident, internally
consistent, completely wrong registration. It was caught only by projecting the
world axis back into the images and looking. Consensus-based selection fixed it.
**Visual verification caught what every numerical diagnostic missed.**

**The standing limitation:** azimuth corrections saturate the ±8° search bound on
**25 of 30 specimens**, so the true placement error is at least that and possibly
more. This is the least verified assumption in the entire pipeline.

---

## 3. Biomass comparison, the core result

> **Superseded.** Re-run on 36 specimens with corrected targets the ordering
> inverts, direct 2D leading at 0.469 kg / R² 0.279, but **neither ordering is
> statistically resolved**, and the one below never was either (paired bootstrap
> [−0.168, +0.099]). See [RERUN_V_BATCH.md](RERUN_V_BATCH.md) §4.

Leave-one-out, 28 specimens, identical protocol for every method.

| method | RMSE | MAE | MARE | R² |
|---|---|---|---|---|
| **mesh geometry** | **0.359 kg** | 0.269 | 27.7% | **0.613** |
| geometric features (voxel) | 0.397 kg | 0.305 | 32.8% | 0.526 |
| direct 2D (no 3D) | 0.440 kg | 0.336 | 39.9% | 0.419 |
| mean predictor | 0.598 kg | 0.511 | 62.5% | −0.075 |
| volume allometric | 0.622 kg | 0.529 | 56.9% | −0.162 |
| canopy area allometric | 0.642 kg | 0.505 | 52.1% | −0.236 |

~~**Reconstruction beats pixels**, 0.397 against 0.440.~~ **Withdrawn.** The
gap was never statistically resolved: paired bootstrap 95% CI [−0.168, +0.099].
It was reported as a finding without the interval that decides it. On the n=36
set the point estimate goes the other way (0.544 against 0.469) and is equally
unresolved, and the direction flips again under a different feature-whitening
choice. The proposal's third research question is answered by the
implied-density diagnostic below instead, which is a measurement rather than a
difference in means.

**Volume allometry is worse than the mean.** Carved hull density varies ~10×
between bushy mango and thin eucalyptus, so one volume-to-mass law cannot span
both. This is the empirical case for a learned density.

**Surface area does not beat volume.** The mesh arm was built to test the
hypothesis that leaf mass scales with area; the area law is the worst method
tried. **A visual hull's surface is envelope area, not leaf area**, 12 views at
12 mm voxels cannot resolve leaves. This is a mechanism, not a tuning failure,
and it generalises to any hull-based method at this resolution.

**And the same is true of its volume.** Dividing measured mass by reconstructed
above-ground volume gives an implied bulk density; fresh tissue is 300–900 kg/m³.
Only **8 of 36** specimens land in a generous 200–1000 band. Twenty-five imply
*less*. The hull has enclosed the air between leaves, all ten Mango at 26–77
kg/m³. Three imply more, meaning thin stems were never carved at all. This is the
mechanism behind every weak biomass number in this document, now measured rather
than inferred: `ggssvt/eval/plausibility.py`.

---

## 4. The batch confound, the finding that caps everything

Leave-one-feature-out on the mesh set showed **height** carrying the result
(removing it costs 0.051 kg; removing canopy area costs 0.001). Chasing that
down:

| batch | n | mean mass | character |
|---|---|---|---|
| E001–E010 | 10 | 0.538 kg | small; reconstruct as mostly pot |
| E011–E020 | 8 usable | 1.844 kg | tall thin saplings |
| **V001–V008** | **8** | **1.138 kg** | **added later; sd 484 g, spans both** |

**Batch membership alone explained R² = 0.887** on the two Eucalyptus batches,
more than any method achieved, including mesh geometry's 0.788 on that subset.
**Within either batch, no method reached R² = 0.2.**

**V001–V008 was collected to break this, and did.** Its range (500–1800 g)
overlaps every other batch instead of forming its own cluster, and the batch-only
R² falls to **0.744** across the three Eucalyptus batches and **0.697** across all
four. The cost is that the 3D advantage went with it, see
[RERUN_V_BATCH.md](RERUN_V_BATCH.md) §4. That is the correct trade: the earlier
comparison was measuring size-class separation.

Every model is recovering *which size class* a plant belongs to, tall-and-sparse
versus short-and-solid, not estimating mass among comparable plants.

**This does not invalidate the pipeline.** It caps the claim. The defensible
statement is *"reconstructed geometry separates plant size classes"*; the
proposal's *"estimates biomass"* is not yet supported.

**The fix was data, and it worked.** A capture batch spanning a continuous mass
range within one species, which is what V001–V008 turned out to be. More
specimens of the two original clusters would have reinforced the confound rather
than broken it.

> **Superseded by §7l, which is worse.** This section regresses mass on batch
> label, which describes the data. §7l measures what the confound costs the
> reported protocol, and finds that batch membership alone *outperforms every
> method here* under leave-one-out. Quote §7l's table, not this paragraph.
>
> **And no further capture is coming.** As of 1 September 2026 the 36 specimens
> are the dataset, so "the fix was data" is closed as a route. See
> [CAMPAIGN.md](CAMPAIGN.md) § "The no-new-data plan" for what replaces it.

---

## 5. DINO backbones

Frozen-feature linear probe, LOOCV, 28 specimens, PCA and standardisation fitted
inside each fold.

| condition | RMSE | R² | dims |
|---|---|---|---|
| no DINO (geometry) | 0.358 kg | 0.616 | 7 |
| DINOv2-small | 0.335 kg | 0.663 | 768 |
| **DINOv2-base** | **0.295 kg** | **0.738** | 1536 |

Paired: **ΔRMSE −0.062 kg, CI [−0.160, +0.036], p ≈ 0.22, not significant.**
The direction is consistent and scales with backbone size; the sample cannot
resolve it. Adding geometry features on top of DINO changes nothing
(0.295 → 0.296).

**DINOv3** is now access-approved but has not been run.

---

## 6. SAM3D segmentation, and the factorial

SAM (ViT-B) prompted from the registered geometry, with three consistency rules:
3D gating to the working cylinder, a prompt-box coverage guard, and reverting
views whose masks disagree with the rest in 3D.

**Effect on the n=38 set:** 96.7% of views accepted, 17.0% of subject pixels
changed. Multi-view agreement **+0.020**; surface coverage **−0.065**; usable
specimens **36 → 33**.

The same trade as before, at the same size: SAM makes the views agree with each
other slightly more and cover the subject noticeably less, and three specimens
fall below the gate as a result. It now drops E015, E019 and V006 on top of the
E012/E016 that the geometric gate drops too. Losing V006 matters more than the
count suggests, V is the batch that breaks the confound, so the SAM3D arm is
running on a marginally more confounded sample than the geometric arm.
*(Previous figures, n=30: 96% accepted, 15.3% removed, agreement +1.9%, coverage
−6.0%, usable 28 → 26.)*

### The 2×2 factorial (33 shared specimens, re-run with V)

|  | no DINO | DINOv2-base |
|---|---|---|
| **no SAM3D** | 0.576 kg / R² −0.080 | **0.385 / +0.518** |
| **SAM3D** | **0.778 / −0.967** | 0.390 / +0.505 |

| effect | ΔRMSE | 95% CI | |
|---|---|---|---|
| DINO alone | −0.191 | [−0.404, +0.021] | not resolved |
| SAM3D alone | **+0.201** | [−0.017, +0.394] | not resolved |
| **DINO given SAM3D** | **−0.387** | **[−0.757, −0.039]** | **resolved** |
| SAM3D given DINO | +0.005 | [−0.007, +0.019] | no effect |
| interaction | −0.196 | [−0.393, +0.031] | not resolved |

**The first resolved effect this project has produced.** DINO features help
significantly *when the hull came from SAM3D*, and the reason is visible in the
table: SAM3D on its own is catastrophic for the hand-crafted descriptors
(0.576 → 0.778, R² −0.967, far below the mean-predictor floor), and DINO simply
does not care which segmenter produced the hull (0.385 → 0.390, an effect of
0.005 kg with an interval of ±0.013).

That asymmetry is the finding. **The hand-crafted geometric descriptors are
fragile to the segmentation; the learned image features are not.** Which follows
from §3: those descriptors summarise a volume that is a canopy envelope rather
than a plant, so perturbing the mask moves them a great deal and costs nothing
real, while DINO reads the images and never depended on the volume being
meaningful.

Read the "resolved" carefully. It is resolved partly because SAM3D + no DINO is
so bad, so it evidences *fragility of the descriptors* more than *value of DINO*.
DINO alone against neither remains unresolved at [−0.404, +0.021].

**The 33-specimen shared set is not a random subset of the 36.** SAM3D fails the
gate on E015, E019 and V006 on top of the E012/E016 the geometric gate drops.
Losing V006 matters beyond the count: V is the batch that breaks the mass/batch
confound, so both arms here run on a marginally more confounded sample than the
n=36 baselines table.

*(Previous run, 26 shared specimens: all four cells within 0.295–0.317 kg and
nothing resolved. The spread has widened enormously, which is what adding a batch
that does not share the others' size structure does to features that were reading
size.)*

---

## 7. View-count ablation

| views | usable | agreement | mean above-ground hull | **physically plausible** | median kg/m³ |
|---|---|---|---|---|---|
| 3 | 23/38 | 0.360 | 99.3 L | **1/23** | 9.8 |
| 4 | 25/38 | 0.424 | 126.5 L | **0/25** | 9.2 |
| 6 | 34/38 | 0.521 | 150.4 L | **2/34** | 15.4 |
| **12** | **36/38** | **0.608** | **10.4 L** | **8/36** | **116.8** |

Usable count and agreement improve monotonically, so the 12-view protocol is
justified on its own. But the plausibility column settles it far more sharply
than the agreement column ever could.

**At four views, zero of twenty-five reconstructions are physically capable of
weighing what the plant weighs.** The median implied bulk density is 9.2 kg/m³,
lighter than expanded polystyrene, and thirty to ninety times below fresh plant
tissue. The hulls average 126 L for plants of at most 2.35 kg. These are not
poor reconstructions, they are not reconstructions of the plant at all.

Four views at 90° is the visual-hull minimum for a *convex* object. A plant is
the opposite of convex, and every unsampled azimuth leaves a prism of empty space
uncarved. Twelve views is where the number becomes non-absurd, 10.4 L and 117
kg/m³, and even there only 8 of 36 clear the bar.

**This is the answer to "could we get away with four images?"** No, and the
reason is measurable rather than a matter of taste. It also disposes of the
biomass comparison across view counts, which was uninformative for a better
reason than small n: below twelve views there is nothing to regress against.

*(Previous figures, n=30 with a fixed 0.28 m pot cut: 17/30, 18/30, 26/30, 28/30
usable and 133.9 / 159.3 / 250.3 / 19.3 L. The hull volumes fall throughout
because each specimen's pot is now cut at its own rim.)*

---

## 7b. TSDF depth fusion, which escapes the hull

Space carving intersects silhouette cones, and Laurentini's result says the
visual hull it produces is the maximal solid consistent with those silhouettes.
That is a ceiling, not a resolution problem. A pot's rim casts no silhouette from
anywhere on a circle around it, so the carve fills it; the gap between two leaves
casts none either, so the carve fills that too. Finer voxels give a smoother
envelope, never a gap.

Depth maps are different evidence. A depth pixel does not say "the subject lies
somewhere along this ray", it says "a surface is at exactly this distance".
Integrated as a truncated signed distance field, concavities survive and
unobserved space stays unknown instead of being filled in. One depth pixel spans
3.0 mm at the working distance, so 6 mm voxels at 256 cubed are justified by the
sensor where the 12 mm carving grid was chosen to keep carving tractable.

**The reconstruction result is decisive**, and it is a measurement rather than a
fitted comparison:

| | plausible | median implied density | verdicts |
|---|---|---|---|
| carve, 12 mm hull | 8/36 | 116.8 kg/m³ | 25 envelope, 3 missing |
| **TSDF, 12 mm fusion** | **25/36** | 271.9 kg/m³ | the shipped cache |
| **TSDF, 6 mm fusion** | **31/36** | **529.2 kg/m³** | 1 envelope, 4 missing |

Two counts, because two things improve. Holding the rim fixed at the carve's estimate isolates the occupancy operator and gives **21 of 36**. Letting the rim be re-estimated from the fused profile, which is what the shipped cache does, gives **25 of 36**: a fused vertical profile has a sharper step, so the rim detector refuses less often. Both are at 12 mm; 31 of 36 is the 6 mm figure.

M001 is the clearest case: 25.79 L of hull above the rim for a 0.74 kg shoot,
against 1.18 L fused. The hull implied 28.7 kg/m³, the fusion 629. Mean coverage
is 0.12, so roughly one eighth of the working volume was ever measured and the
rest is honestly absent rather than assumed solid.

**The biomass result is real but unresolved**, which by now is the expected
outcome at this sample size:

| feature set | all 36 | Eucalyptus only (n=26) |
|---|---|---|
| carved geometry (7) | 0.544 / +0.030 | 0.717 / **−0.313** |
| direct 2D (7) | 0.469 / +0.279 | 0.520 / +0.311 |
| **TSDF geometry (7)** | **0.465 / +0.290** | **0.494 / +0.377** |
| TSDF + 2D (14) | **0.429 / +0.396** | |
| mean predictor | 0.552 | 0.626 |

Paired bootstrap, 20,000 resamples: TSDF against carved geometry is −0.079
[−0.242, +0.066] on all 36 and −0.223 [−0.498, +0.021] on Eucalyptus, and against
direct 2D it is −0.004 [−0.134, +0.133]. **Nothing resolves.** Do not report an
ordering from this table.

What is worth reporting is the sign change. **TSDF geometry is the first 3D
feature set to clear the mean-predictor floor within a species.** Carved geometry
sits at R² −0.313 against a floor of −0.082, meaning it carries no usable mass
signal; the fused features sit at +0.377. Crossing the floor is a different
statement from winning a head-to-head, and it is the one that follows from the
plausibility result rather than from a difference in means.

**The limits, stated plainly.** Twelve views leave most leaf undersides
unobserved, so this does not resolve individual leaves and is not a substitute
for a dense photogrammetric capture. The fused interior is a band one truncation
width deep behind each observed surface, not a filled solid, so its volume is a
proxy rather than the plant's volume. And 6 mm is the sensor's limit, not the
plant's: a Eucalyptus leaf is 0.3 mm thick.

Ten minutes for all 36 on one CPU core. `python -m ggssvt.cli fuse`.

---

## 7c. The reconstruction was the bottleneck, not the regressor

Everything in section 3 and section 7b pointed one way, and this is the test that
closes it. Take the seven hand-crafted descriptors unchanged, the same
leave-one-out protocol, the same 12 mm grid, the same twelve views and the same
masks. Change only the operator that turns them into occupancy.

| | carved | fused | paired bootstrap |
|---|---|---|---|
| **geometric features** | 0.544 / +0.030 | **0.335 / +0.632** | −0.209 [−0.363, **−0.066**] **resolved** |
| volume allometric | 0.592 / −0.150 | 0.469 / +0.278 | −0.123 [−0.202, **−0.034**] **resolved** |
| canopy area allometric | 0.598 / −0.170 | 0.494 / +0.201 | |
| mesh geometry | 0.507 / +0.157 | 0.486 / +0.227 | |
| direct 2D | 0.469 / +0.279 | 0.469 / +0.279 | unchanged, as it must be |
| mean predictor | 0.568 | 0.568 | |

**These are the first resolved improvements this project has produced on biomass.**
Everything before them, including the original "reconstruction beats pixels", had
an interval spanning zero. Direct 2D is identical in both columns, which is the
control: it touches no reconstruction, so it must not move, and it does not.

Two methods that sat *below* the mean-predictor floor now clear it. Volume
allometry went from −0.150 to +0.278 and canopy area from −0.170 to +0.201. That
matters more than the ordering. A single volume-to-mass law was said to be
impossible across morphologies because hull density varied tenfold between a
bushy Mango and a thin Eucalyptus; on a fused reconstruction the same law works,
because the volumes are no longer envelopes of wildly different emptiness.

**The canopy-area hypothesis partially survives.** It was declared dead in
section 3 on the grounds that a visual hull's surface is envelope area rather
than leaf area. On a fused surface it clears the floor. The mechanism claimed
there was right, and it was a statement about the instrument rather than about
the biology.

On the Eucalyptus subset of those same pooled predictions, geometric features go
from 0.610 / +0.049 to **0.350 / +0.687** against a floor of 0.626. Note that
this is the subset of a fit on all 36, not a fit on Eucalyptus alone; the
fit-within-species numbers quoted elsewhere in this document are a different
protocol and the two should not be compared.

What this does not do is make the reconstructions good. Coverage is 0.12, most
leaf undersides were never observed, and 5 of 36 fused specimens still fail the
plausibility check. The claim is narrow and it is the one the evidence supports:
**for these species at this resolution, replacing silhouette intersection with
depth integration improves biomass estimation by more than any change to the
regressor has.**

---

## 7d. DITR-style DINO lifting, and why it does not rescue E001-E010

*This is the project's one experiment on the reciprocity Malik et al. argue for
in "The three R's of computer vision" (Pattern Recognition Letters 72, 2016):
that grouping and recognition inform reconstruction and each other. Semantic
features are used to attempt a **reorganization** (plant against pot) in order to
repair a **reconstruction**. It fails, for a reason that is measured rather than
guessed, and a negative result about that interaction is still a result about
it. The third R, **recognition**, is out of scope for this study: one genus per
batch and no category task means there is nothing for a recognition claim to be
tested against. Note also that "the 3Rs" in `POSEFREE.md` refers to the pointmap
models by their shared suffix, which is a different thing entirely; this document
uses "pointmap models" for those.*


Requested by the supervisor, after Knaebel et al., who observe that 3D
segmentation largely ignores 2D foundation models even where calibrated images
sit beside the point cloud. DITR extracts frozen DINOv2 patch features, projects
the points into each camera to look them up, pools across views, and injects the
result into a 3D backbone trained with a supervised loss.

The first half transfers directly and is implemented in
`ggssvt/geometry/dino_lift.py`. The second half does not: DITR trains on
ScanNet, S3DIS and nuScenes, which supply per-point semantic labels, and this
dataset supplies none. The supervised head is therefore replaced by k-means over
the pooled features, which makes the question **can a foundation model separate
plant from pot where excess-green cannot?**

The success criteria were written down before the run, because otherwise any
clustering looks like a result. A useful separation puts the clusters at
different heights, puts most of the volume near the floor in one and most above
the rim in the other, and does so on the batch where colour fails.

| batch | n | mean height gap | agreement with rim | upper cluster above rim | rim confident |
|---|---|---|---|---|---|
| E001-E010 | 10 | 0.212 m | 0.704 | **0.497** | 1/10 |
| E011-E020 | 8 | 0.472 m | 0.905 | 0.786 | 8/8 |
| Mango | 10 | 0.620 m | **0.969** | **0.971** | 8/10 |
| V001-V008 | 8 | 0.365 m | 0.741 | **0.358** | 8/8 |

**The answer is no.** On Mango the lifted clustering recovers essentially the
same boundary the geometric rim detector found, agreeing on 96.9 per cent of
points with 97.1 per cent of the upper cluster above the rim, and M001 alone
reaches 0.998. Two methods sharing no mechanism agreeing that closely is a real
validation of the rim estimator, and it is the useful half of this result.

But E001-E010, the batch this was run for, gives 0.497: the upper cluster falls
half above and half below the rim, which is what a split unrelated to the
pot boundary looks like. V001-V008 gives 0.358 with confident rims, meaning the
clustering there finds a *different* boundary from the geometric one rather than
a better one. **DINO features confirm the split where it is already findable and
do not find it where it is not.**

**The reason is resolution, again.** DINOv2 with patch 14 on a 512 by 416 frame
gives a 37 by 30 patch grid, so one patch spans 13.8 pixels, which is 42 mm at
the 1.1 m working distance. A Eucalyptus stem is 5 to 15 mm. Every patch that
contains stem also contains pot, soil or background, so no pooling of those
features can separate the two. This is the same argument as the 12 mm voxel and
it has the same shape: the instrument is coarser than the structure.

Worth being precise about what this does and does not rule out. It rules out
patch-level DINOv2 at this capture resolution. It does not rule out DITR itself,
which was never given the labelled 3D data it is built for, nor a
higher-resolution capture, nor SAM-family masks lifted the way SAMa lifts them,
which operate on pixels rather than 42 mm patches.

`python -m ggssvt.cli dino-segment`, about 15 seconds a specimen on CPU.

---

## 7g. Closing Malik's loop: the reconstruction refining the segmentation

The pipeline ran one way. A 2D segmenter decided which pixels were plant, those
masks were carved, and nothing travelled back. Malik et al. argue that is the
wrong shape, and there is a specific reason to expect it to matter here: each
mask is decided from **one** view by a colour threshold, while the reconstruction
is decided from **twelve at once**. Re-projecting the reconstruction into a view
gives it a second opinion formed from evidence it never saw.

Measured on E001, the two opinions differ in both directions: the carve claims
1541 pixels excess-green missed, and rejects 3326 it included. So the loop can
add information, not merely subtract. Four combination rules, all 36 specimens,
refining against the **fused** reconstruction because it only claims surfaces a
camera measured:

| rule | plausible | median kg/m³ | mean volume |
|---|---|---|---|
| original (no refinement) | 8/36 | 116.8 | 10.40 L |
| union | 4/36 | 88.8 | 17.38 L |
| **intersection** | **19/36** | **344.4** | **2.36 L** |
| reconstruction only | 14/36 | 194.8 | 8.50 L |

**Intersection more than doubles the plausible count, 8 to 19, and brings the
median implied density inside the 200-1000 band for the first time on the carve.**
Union goes the other way, 8 down to 4, which is what the direction predicts and
was written into the module docstring before the run: the hull is already too
large, so a rule that grows masks makes it worse.

**The control is the important part of this entry.** A re-carve of the
*unchanged* masks must reproduce the cached volume. The first version of this
experiment skipped `largest_connected_component`, which preprocess applies after
carving, and the control then moved E001 from 4.20 L to 1.88 L while all three
rules landed near 1.8-2.3 L. Every rule looked like it rescued the specimen. It
was the carver disagreeing with itself. With the step restored the control drifts
by **0.08% median and 2.6% worst**, against a 77% volume reduction from
intersection, so the effect is real by a wide margin. The drift is reported per
specimen in `reciprocity.json` and a rule that moved a volume by less than it
would have shown nothing.

**What this does not do.** 19/36 is still well below the fused reconstruction's
32/38. Refining the masks improves the carve substantially; it does not make the
carve competitive with depth fusion, and the operator finding in 7b and 7c
stands. What it adds is evidence, on this data, for the specific reciprocity
Malik argues for, with the direction measured rather than assumed.

`python -m ggssvt.cli reciprocity`. About twelve minutes on one CPU core.

---

## 7e. Does a stronger regressor help? Only while the input is bad

Asked because a previous student used R-squared and RMSE with a ridge-style fit,
and random forests and networks are the obvious alternatives. Measured rather
than argued: the same seven features, the same leave-one-out, only the regressor
changing.

| regressor | carved | fused |
|---|---|---|
| ridge | 0.544 / +0.030 | **0.335 / +0.632** |
| random forest | **0.406 / +0.459** | 0.393 / +0.493 |
| gradient boosting | 0.406 / +0.459 | 0.395 / +0.490 |
| MLP 32-16 | 0.506 / +0.160 | 0.403 / +0.469 |
| mean predictor | 0.552 | 0.552 |

Paired bootstrap against ridge on the same features:

| | difference | 95% interval | |
|---|---|---|---|
| carved, random forest | −0.138 | [−0.281, −0.026] | **resolved** |
| carved, gradient boosting | −0.138 | [−0.288, −0.001] | **resolved** |
| carved, MLP | −0.038 | [−0.114, +0.029] | not resolved |
| fused, random forest | +0.058 | [−0.003, +0.121] | not resolved |
| fused, gradient boosting | +0.059 | [−0.007, +0.127] | not resolved |
| fused, MLP | +0.068 | [+0.007, +0.129] | **resolved, worse** |

**The pattern is the finding.** A nonlinear regressor rescues the carve, resolving
a 0.138 kg improvement, and then stops helping once the reconstruction is fixed.
On the fused features ridge wins and the MLP is resolvably *worse*.

That reads as exactly what it looks like. Hull volume is an envelope whose
relationship to mass is distorted differently for a bushy Mango than for a thin
Eucalyptus, and a forest can carve that space with thresholds where a linear fit
cannot. Once the volumes are no longer envelopes the relationship is closer to
linear, and at n=36 the extra capacity costs more in variance than it buys in
bias. **A stronger model was compensating for a broken input.**

The decision it poses, and the honest answer:

| | RMSE |
|---|---|
| carved + ridge | 0.544 |
| carved + random forest | 0.406 |
| **fused + ridge** | **0.335** |

Fixing the reconstruction beats strengthening the model, and is also the cheaper
claim to defend. But **fused+ridge against carved+RF is −0.071 [−0.175, +0.028],
not resolved**, so the two routes cannot be separated on this data. What can be
said is that fixing the input beats the mean predictor by more, keeps the model
interpretable, and does not spend capacity at a sample size that cannot afford it.

Two cautions. Six comparisons were run and two resolved at 95 per cent, which is
close to what chance alone would produce, so no single row above should be
quoted without that context. And a random forest at n=36 is fitting 35 samples
per fold, which is not where forests are at their best.

**F1 does not apply.** It is a classification metric and above-ground mass is
continuous. Reaching for it would mean binning mass into classes, discarding the
resolution of a ground truth that was obtained destructively, and inventing
boundaries that carry no agronomic meaning. R-squared and RMSE are the right
family; what was missing was never the metric but the interval around it.

---

## 7f. The reconstruction metrics, and why the good reconstruction scores worse

Chamfer distance, Hausdorff and HD95, F-score and PSNR have been in
`eval/metrics.py` since early in the project and were never called once. That was
not an oversight. Those metrics measure distance to a reference, and there is no
reference: destructive harvest produced a mass, not a geometry, and no laser
scan, CT or CAD model of any specimen exists.

Two things they can legitimately do, kept apart here because conflating them
would overstate the result.

**Explanatory power against the captured views.** Project a reconstruction back
into each camera and score what it predicts against what was measured. No
reference needed, because the views are the reference.

| operator | silhouette IoU | depth MAE | depth PSNR | subject pixels explained |
|---|---|---|---|---|
| space carving | **0.407** | 67.9 mm | **32.70 dB** | **0.456** |
| TSDF fusion | 0.219 | 67.4 mm | 32.49 dB | 0.233 |

**The carve wins, and that is the finding.** A visual hull is by construction
consistent with every silhouette it was built from, so a silhouette-agreement
metric structurally favours it. The fusion scores lower because it has holes: it
claims only what a camera measured, and roughly one eighth of the working volume
was ever measured.

Set that beside what the same two operators do on the questions that matter:

| | carve | fusion |
|---|---|---|
| silhouette IoU | **0.407** | 0.219 |
| physically plausible | 8/36 | **25/36** |
| biomass RMSE | 0.544 kg | **0.335 kg** |

**A metric that looks like "reconstruction quality" ranks the worse
reconstruction higher.** This is the clearest argument the project has produced
for the plausibility check: without ground-truth geometry, self-consistency
metrics measure agreement with the input rather than fidelity to the object, and
for a hull that agreement is guaranteed rather than earned. Reporting silhouette
IoU alone would have pointed the whole project in the wrong direction.

Depth error is the honest tie. Where both predict a surface they are equally
accurate, 67.9 against 67.4 mm and 32.70 against 32.49 dB. The two operators
differ in how much they claim, not in how well they place what they claim.

**Agreement between the two operators**, which is a real number and is not
accuracy:

| | value |
|---|---|
| voxel IoU | 0.251 |
| Chamfer distance | 36.6 mm |
| F-score at 20 mm | 0.691 |
| HD95 | 224 mm |
| Hausdorff | 641 mm |

A quarter of the occupied voxels are shared and the worst-case separation is
0.64 m, which is most of the working volume. The two are describing substantially
different objects, and the 0.691 F-score says the disagreement is in the bulk
rather than in a few outliers. Two methods can agree closely and both be wrong;
these do not even agree.

**PSNR without a radiance field.** The conventional use needs rendered views
against captured ones and belongs to the splatfacto arm, which has
`transforms.json` exported for every specimen and has never been trained. The
figure above is depth PSNR from re-projection, which is a different quantity and
is labelled as such wherever it appears.

**One caveat that must travel with the re-projection numbers.** These views built
the reconstruction, so this is self-consistency rather than held-out
generalisation. A volume that fails to explain the images it was carved from is
definitely wrong; one that explains them may still be an envelope. Leave-one-view-
out would be the stronger test and costs twelve carves per specimen.

`python -m ggssvt.cli quality`, about a minute for all 36.

---

## 7k. H2's viewpoint consistency, on a view never seen

`reconstruction_quality.reproject` projects a reconstruction into the twelve
views it was built from, which is self-consistency, not the viewpoint
generalisation H2 claims. This holds each view out in turn: reconstruct from
eleven, predict the twelfth, compare against what the sensor measured there.
432 held-out views across 36 specimens, no GPU.

| | mean silhouette IoU |
|---|---|
| in-sample, the views it was built from | 0.4070 |
| **held out, a view it never saw** | **0.3896** |
| gap | 0.0174, **4.3%** of the in-sample score |

**The carve loses only 4 per cent of its agreement when a view is withheld**, and
that number is more interesting than it looks in either direction.

Read one way it is a positive result: twelve views at 30 degree steps are dense
enough that any eleven of them nearly determine the hull, so the reconstruction
is not fitted to individual silhouettes. That is a real viewpoint-consistency
finding, and it is the first quantitative evidence on H2.

Read the other way it is a warning about the metric. A visual hull built from
eleven silhouettes is barely different from one built from twelve, so a small gap
here is close to guaranteed and does **not** show the reconstruction captured the
subject. The same hull scores 0.407 in-sample while only 8 of 36 specimens can
physically weigh their plant. **Viewpoint consistency and fidelity are
independent, and this measures the first.**

That makes the gap useful as a *comparative* number rather than an absolute one.
Its value is in what the campaign does with it: run the same held-out protocol
against a geometry-grounded model and against the `h2_no_geometry` ablation, and
the question becomes whether grounding shrinks a gap that is already small on a
hull. A baseline of 4.3 per cent is what those runs have to beat.

`python -m ggssvt.cli viewpoint`, about 25 minutes on one CPU core.

---

## 7j. H1's second half: DINOv2 needs a quarter of the labels

H1 makes two claims and the second is the one that makes the method
self-supervised rather than merely transformer-based: it should reach a given
accuracy from substantially fewer labelled examples. Nothing measured it.

Label efficiency is conventionally read off a frozen representation with a small
head fitted on a fraction of the labels, which is the probe already in use here,
so this needed no GPU. For every held-out specimen the head is fitted on a random
subsample of the rest, eight independent draws per fraction, with standardisation
and rotation fitted inside the subsample.

| labels | geometric (7 features) | dinov2-base (1536) |
|---|---|---|
| 8 (25%) | 128.8 ± 293.6 * | **0.463 ± 0.022** |
| 16 (50%) | 1.269 ± 0.767 | 0.444 ± 0.038 |
| 24 (75%) | 0.915 ± 0.364 | 0.401 ± 0.026 |
| 32 (100%) | 0.576 | **0.385** |

**DINOv2 reaches the geometric baseline's full-label accuracy with 8 labels; the
geometric features need 32.** Read as stated, that is H1's second half supported
by a factor of four. Two things have to travel with it.

**The starred point is a numerical failure, not a measurement.** Seven features
fitted on eight labels is close to singular and a single draw produced 128 kg
with a standard deviation of 293. It is flagged and excluded from the comparison
rather than reported as a curve point, because a bar read off it would be
meaningless. The instability is itself informative: a low-dimensional descriptor
does not merely degrade at few labels, it fails.

**The dimensionalities differ and that is a confound.** Both conditions are
reduced to 8 principal components, but 1536 dimensions reduced to 8 is a
different operation from 7 reduced to 7, and part of DINOv2's advantage at small
label counts may be that projection regularising it rather than the
representation carrying more. Separating the two needs a matched-capacity
control, which this experiment does not have. The claim is therefore reported as
supported and confounded, not as clean.

Run on the 33 specimens shared by every condition; E015, E019 and V006 are absent
from the DINO descriptor cache, and V010 is excluded for failing C1.

`python -m ggssvt.cli label-efficiency`, seconds on one CPU core.

---

## 7i. H4, two thirds answered: noise is harmless, occlusion is not

The view-count ablation already answered sparse sampling. These are the other
two, both degradations of the cached inputs, so neither needed the GPU. Scored
by C1, and by whether the reconstruction survived at all.

| degradation | plausible | **fragments** | median kg/m³ | mean volume |
|---|---|---|---|---|
| control | 8/36 | 0 | 116.8 | 10.34 L |
| noise, 1x sensor | 7/36 | 0 | 118.4 | 10.05 L |
| noise, 2x | 7/36 | 0 | 123.4 | 10.10 L |
| noise, 4x | 7/36 | 1 | 145.3 | 9.61 L |
| occlusion, 10% | 6/36 | 5 | 775.2 | 3.11 L |
| occlusion, 25% | 10/36 | 9 | 356.7 | 3.38 L |
| **occlusion, 50%** | **0/36** | **33/36** | 339.4 | 6.55 L |

**Robust to depth noise, not robust to sustained occlusion.** Quadrupling the
sensor's own noise characteristic costs one specimen and 7% of the volume; the
carve uses depth only for free-space votes with a margin, so noise inside that
margin changes nothing. A band across 50% of the subject's height in every view
destroys 33 of 36 reconstructions.

**The fragment column is the finding that nearly did not get recorded.** Without
it, occlusion at 25% appears to *improve* the plausible count from 8 to 17. It
does not. A mid-height band severs the plant, `largest_connected_component` keeps
whichever side is bigger, and the surviving piece lands inside the plausible band
by coincidence. E001 at 50% keeps 5% of its volume and scores 484 kg/m³, which is
squarely "plausible".

**So C1 is necessary and not sufficient**, and that belongs in the methods
section as a stated limitation rather than being discovered by an examiner. A
reconstruction can pass the physical check by being a well-proportioned fragment
of the plant. The surviving fraction is now recorded per specimen and a fragment
is excluded from the plausible count.

The occlusion here is deliberately the pessimistic case: the same band in every
view. Occlusion that moves between viewpoints is partly recovered by the other
eleven, and that variant is not measured.

`python -m ggssvt.cli robustness`, about 20 minutes on one CPU core.

---

## 7h. The CNN control collapse, diagnosed

At n=36 the frozen-feature probe's control scored RMSE 0.458, R² **+0.312**. At
n=38 on the lab machine it scored RMSE 2.981, R² **−26.997**. Adding two
specimens cannot degrade a baseline sixfold, so the number is an artefact, and
until it is explained **every DINO comparison measured against it is
unreportable**, including the four conditions that came back "significant" at
dRMSE −2.54 to −2.56 with intervals barely excluding zero. Four backbones
agreeing to within 0.02 kg is one broken control subtracted four times.

Reproduced locally, on the archived n=36 features, through the probe's own
pipeline:

| | RMSE | R² |
|---|---|---|
| archived, n=36 | 0.458 | **+0.312** |
| plus one specimen with a 190 L hull | 1.397 | **−5.215** |
| plus a *second* large-hull specimen | 0.414 | **+0.459** |

**One extreme specimen destroys it; two repair it.** That is the signature of a
single leverage point, and PCA fitted in-fold amplifies it: with one outlier the
leading component is that specimen, with two there is a direction and the fit
recovers. V010 is the point. Its carve is 190.6 L for a 1.9 kg shoot, an implied
density of 9.8 kg/m³, forty-two times the median hull in this set.

**The fix is not to patch the probe.** V010 fails C1 by a factor of twenty. A
specimen whose reconstruction cannot physically weigh its plant is a failed
reconstruction, not a hard data point, and the criterion that says so was fixed
before any of this was measured. What the episode shows is that the existing
quality gate, which passed V010 on coverage and agreement, is blind to the one
check that mattered.

**Confirmed, and the repaired comparison reverses the conclusion.** Re-running
the probe on the lab machine with V010 excluded, n=37:

| condition | RMSE | R² | vs the CNN control |
|---|---|---|---|
| cnn (no DINO) | 0.471 | **+0.284** | reference |
| dinov2-base | **0.400** | +0.483 | −0.071 [−0.193, +0.062], p~0.29 |
| dinov2-base + geometry | 0.400 | +0.483 | −0.071 [−0.193, +0.062], p~0.29 |
| dinov3-base | 0.409 | +0.459 | −0.062 [−0.188, +0.077], p~0.38 |
| dinov3-base + geometry | 0.412 | +0.451 | −0.058 [−0.186, +0.080], p~0.40 |

The control returns to +0.284 against the +0.312 it scored at n=36, so the −27.0
was one specimen and nothing else.

**Every DINO condition is now unresolved.** The point estimates favour the
transformer by 13 to 15 per cent, and not one interval excludes zero. Compare
that with what the broken control produced: four conditions at dRMSE −2.54 to
−2.56, all "significant" at p~0.02 to 0.05.

That is the whole lesson. **A single leverage point did not merely inflate the
effect, it changed the verdict**, and the verdict it produced was the one the
hypothesis wanted. Had the diagnosis not been run, H1's first half would have
been reported as supported at p~0.02 on the strength of one reconstruction that
fails the plausibility criterion by a factor of twenty.

The honest statement of H1's first half on frozen features is therefore: the
self-supervised backbone is better on the point estimate and the difference is
not resolved at n=37. The campaign's `h1_dinov2` run tests the same claim on a
*trained* model, which is a different and still-open question.

`eval/metrics.py:leverage_report` now reports whether one specimen carries the
error, because a score can look catastrophic for a reason that has nothing to do
with the method under test and no summary statistic says so.

---

## 7l. The confound, measured rather than argued

Section 4 established the confound by regressing mass on batch label. That is a
description of the data. What it never gave was the size of the error every
reported LOOCV number carries because of it, and *that* is the quantity a
reviewer needs. `eval/batch_holdout.py` supplies it.

**The design.** Leave-one-out withholds a specimen and leaves the other nine or
so members of its own capture session in the training fold, carrying that
session's mean mass. Leave-one-batch-out withholds the whole session, so the
model has never seen a plant captured on the same day as the one it is scored
on. Everything else (estimator, standardisation, PCA, alpha) is held fixed.

| condition | LOOCV RMSE | LOOCV R² | LOBO RMSE | LOBO R² | inflation |
|---|---|---|---|---|---|
| geometric, all 36 | 0.458 | +0.312 | 1.151 | −3.339 | **+0.692 kg** |
| geometric, shared 33 | 0.576 | −0.080 | 1.105 | −2.970 | +0.529 kg |
| DINOv2 frozen, 33 | 0.385 | +0.518 | 0.921 | −1.758 | +0.536 kg |
| **batch membership only** | **0.351** | **+0.600** | n/a | n/a | n/a |

**Read the last row first.** Predicting a specimen's mass as the mean of the rest
of its own batch, using no geometry, no image and no features whatsoever, scores
0.351 kg under leave-one-out and **beats every real method on this dataset**. The
best learned representation (DINOv2 at 0.385) does not reach it. Under
leave-one-batch-out every condition falls below the mean predictor.

This is stronger than section 4's statement and less comfortable. Section 4 said
batch membership explains most of the variance; this says batch membership is
*better than our method*, under the very protocol the project has been reporting.

**What it does not touch.** No reconstruction or screening result is a regression
against mass, so §7b, §7f, §7g, §7i, §7k and the view-count sweep are unaffected.
The relocation is confined to the biomass claim, which §10 had already
recommended relocating for a weaker reason.

**How to report it.** Quote both columns, always. A method whose two scores are
close has learned something about plants; one whose LOBO score collapses to the
mean learned which batch a specimen came from. The gap is the finding, and
publishing it is a far better position than having it found.

The n=33 row exists because the descriptor caches were built on the set shared
with the SAM3D cache; E015, E019 and V006 are absent from them. Conditions fitted
on different specimens are not comparable, so the DINO comparison runs on the
intersection and the geometric condition is reported on both sets.

---

## 7m. The paired tests the designs already earned

Three of this project's comparisons are paired: the same specimens, the same
criterion, one thing changed, and all three were being reported as bare ratios,
which throws away exactly where the power is.

**Carving against fusion.** 8 and 31 plausible of the same 36 reconstructions.
Twenty-nine specimens discriminate (3 favour carving, 26 favour fusion); the
concordant seven carry no information about which operator is better. Exact
McNemar on the discordant pairs:

**p = 1.5 × 10⁻⁵.**

That is the most decisive statistic in the project, and it was sitting unused
behind a ratio. `eval/batch_holdout.py:mcnemar`.

The same treatment applies to the reciprocity rules (8 → 19 of 36, §7g) and to
the view-count sweep (§7). Neither is done yet; both are one call each.

**Why it matters more than it looks.** Section 7f's finding, that silhouette
IoU ranks the worse reconstruction higher, rests on the density screen
disagreeing with the metric. A screen comparison at p = 1.5 × 10⁻⁵ is a much
firmer foundation for that argument than "8 versus 31".

---

## 7n. External validation: the method transfers, the dataset did not

7l left the biomass claim unsupportable on our own data and no further capture is
coming, so the question moved to somebody else's plants. The 4TU greenhouse
lettuce set (DOI 10.4121/15023088) is 387 usable RGB-D pairs across four
cultivars and a seven-week growth series, destructively weighed: 1.4 g to 459.7 g
with a mean of 115.2 g, continuous by construction rather than clustered into
sessions. `eval/external.py`; run with `cli external`.

**Step 1, the measurement, before any regression.** Their height, diameter and
leaf area were measured destructively on the same plants, so the depth-derived
versions can be checked against a ruler:

| trait | Pearson r | MAE |
|---|---|---|
| diameter | **+0.920** | 2.67 cm |
| projected area vs leaf area | **+0.925** | 1563 cm² |
| height | +0.592 | 3.74 cm |

Diameter and area are solid. **Height is the weak measurement**, and it is weak
for a reason worth recording: the reference surface is a tray whose height
changes between growth stages, and a top-down camera sees the canopy top, not the
attachment point their ruler starts from. The area MAE is large because projected
area and true leaf area are different quantities (curled leaves hide area from
a camera), so the correlation is what matters there, not the difference.

**Step 2, the screen.** Agreement with the measured diameter to within 40%,
fixed before the numbers were read: **378 of 387 pass**. Nine failures out of 387
is a far better segmentation rate than we achieve on our own captures, which says
more about a controlled greenhouse than about the segmenter.

**Step 3, the regression**, leave-one-out beside leave-one-cultivar-out, this
dataset's leave-one-batch-out, holding out a variety the fit has never seen:

| condition | LOOCV | held-out cultivar | unscreened, held-out cultivar |
|---|---|---|---|
| direct 2D | 48.0 g (R² +0.806) | 55.1 g (R² +0.744) | 56.0 g (R² +0.736) |
| **2D + profile** | **42.7 g (R² +0.846)** | **50.9 g (R² +0.782)** | 54.7 g (R² +0.747) |
| volume only | 54.3 g (R² +0.751) | 58.0 g (R² +0.716) | 58.1 g (R² +0.716) |

**Set that against our own data, same estimator, same code path:**

| | our 36 specimens | lettuce, 378 plants |
|---|---|---|
| leave-one-out R² | +0.312 | **+0.846** |
| grouped holdout R² | **−3.339** | **+0.782** |
| what the holdout costs | catastrophic | 0.064 in R² |

That is the answer to the question the project has been unable to answer. The
regression **does** estimate biomass; it could not be shown to on our specimens
because our specimens were four capture sessions with different mean masses. The
same features, the same ridge, the same leave-one-group-out protocol, hold up on
a cultivar they have never seen.

**Two further things this settles.**

*The method ranking is the same in both datasets.* `2D + profile` beats
`direct 2D` beats volume alone here, exactly the order §3 found on our specimens.
A ranking that survives a change of species, sensor and facility is worth more
than the margin that produced it.

*The screen is not carrying the result.* The screen uses their measured diameter,
which correlates with mass, so a screened score is selected partly on the label
-- the same exposure our density criterion has, since implied density is mass
over volume. Reported unscreened, the held-out-cultivar R² moves from +0.782 to
+0.747. The selection is worth 0.035 in R², not the result.

**What is honestly claimed.** Transfer runs across a sensor change (their
RealSense D415, our Kinect v2), a species change, and a facility change, using
only the image-only half of the pipeline: one top-down view means no carve and
no fusion. Both those methods are the ones that already won on our data, so the
thing validated externally is the thing being claimed. What is *not* validated
here is anything about the reconstruction: the carve, the fusion, the density
screen and the reciprocity loop cannot run on a single view and remain supported
only by §7b, §7f and §7g on our own specimens.

**One record is unusable as distributed.** The ground truth has 388 measurements
and the archive pairs with 387: `Image332` names `RGB_332.png`, which is absent,
while an unreferenced `RGB_322.png` sits in the folder with no matching record.
Pairing them was tested by overlapping the RGB's saturated region with the
depth's raised region; the candidate scored 0.163 where known-correct pairs
scored 0.151 to 0.321, so the check does not discriminate and settles nothing.
The record is skipped and counted rather than repaired.

**And a segmentation finding that would have cost a cultivar.** Two of the four
varieties are red-leaf. Satine measures R 80, G 49, B 24: an excess green of
**−0.02**, indistinguishable from concrete. Segmenting these images with the
index the rest of this project uses would have silently dropped an entire
cultivar, and it would have dropped the *heaviest* plants. No colour index fixes
it: a red lettuce and the orange crate the tray stands on overlap on excess
green, saturation and green-minus-blue alike. What separates them is that the
tray sits on top of the crate, so height above the tray surface is the
discriminator. The lesson generalises to our own work: the excess-green
segmenter in `geometry/segment.py` would fail on any red or purple foliage, and
nothing currently warns about that.

---

## 7o. Does 3D geometry actually help? On confound-free data, yes

§3 found 3D geometric features **not** beating image-only regression on our
specimens, and that has sat as an uncomfortable result ever since: it is the
premise of the whole project. But that comparison was made inside the batch
confound (§7l), where the strongest predictor available to any method was which
session a plant was captured in. It could not settle the question either way.

The lettuce set can. One top-down view cannot be carved or fused, but the depth
map back-projects to a **metric** point cloud, and that surface carries structure
a silhouette does not. Eight descriptors are taken off it (`data/lettuce.py:
surface_descriptors`), the informative ones being:

- **rugosity**, true surface area over the area of its own shadow. A flat leaf
  scores 1; the crumpled cultivars here score 1.9 to 2.2. This is precisely what a
  projected area is blind to, and fresh mass is partly a question of how much
  tissue is folded into a given footprint.
- **normal_z_mean**, the mean vertical component of the surface normals: a proxy
  for leaf angle.
- **hull_volume_l**, the convex hull of the surface and its shadow: an upper
  bound on the plant, not a measurement of it, because with one view the
  underside is assumption rather than data.
- height at the 50th and 90th percentiles, its standard deviation, and the fill
  ratio: how the mass is stacked, not merely how tall the plant is.

**Scored on a held-out cultivar, 378 plants:**

| feature set | RMSE | R² | unscreened R² |
|---|---|---|---|
| direct 2D | 55.1 g | +0.744 | +0.736 |
| 2D + profile | 50.9 g | +0.782 | +0.747 |
| surface only | 52.1 g | +0.771 | +0.755 |
| **2D + profile + surface** | **47.3 g** | **+0.811** | **+0.790** |
| volume only | 58.0 g | +0.716 | +0.716 |
| hull volume only | 61.8 g | +0.678 | +0.680 |

**Paired bootstrap against `2D + profile`, on the held-out cultivar:**

| condition | difference | 95% interval | p |
|---|---|---|---|
| **2D + profile + surface** | **−3.5 g** | **[−5.8, −1.4]** | **0.0000** |
| surface only | +1.3 g | [−0.8, +3.5] | 0.263 |
| direct 2D | +4.3 g | [+1.5, +6.9] | 0.0044 |
| volume only | +7.1 g | [+2.9, +11.3] | 0.0004 |
| hull volume only | +10.9 g | [+6.7, +15.2] | 0.0000 |

**Two statements, and the difference between them matters.**

*The surface adds information the silhouette does not have.* The interval on the
improvement excludes zero and does not come close to it. By the criterion this
project applies everywhere else, that is **resolved**, and it is the first time
any 3D-versus-2D comparison here has been.

*The surface does not replace the silhouette.* On its own it is statistically
indistinguishable from `2D + profile` (p = 0.26). The two describe different
things and the gain is in combining them, which is a more useful finding than
either winning outright.

**What this does to §3.** It does not overturn it. §3 is still what our own
data shows. It explains it. A 3D advantage of about 3.5 g on a 115 g mean is
roughly 3% of RMSE; a confound that lets batch membership alone beat every method
will bury an effect that size without trace. The honest reading is that our
specimens were never able to detect a 3D advantage of the magnitude that actually
exists, and reporting §3 as evidence *against* 3D geometry would have been
reading a null result as a negative one.

**And the ceiling this exposes.** `hull volume only` is the worst condition
tested, below even `volume only`. A convex hull over a single-view surface throws
away the concavity that makes a lettuce a lettuce. That is the same finding as
§7f in different clothes: an envelope is not a plant, and it is the argument
for the Pheno4D virtual-view experiment, where a real volumetric reconstruction
can be scored against a known cloud instead of assumed.

---

## 7p. The metric inversion, demonstrated against known geometry

§7f argued that silhouette IoU ranks our reconstructions backwards. The argument
rested on the density screen disagreeing with the metric, which is an inference:
if the screen were the thing that was wrong, the argument reverses. Pheno4D
settles it. Fourteen laser-scanned plants, twelve virtual views each at our
azimuths and through our camera model, and **our** `carve` and `fuse` run on
those views, not reimplementations, the pipeline's own functions.
`eval/virtual_views.py`; run with `cli virtual-views`.

**The result is not a tendency. It is total.**

| | agrees with the truth |
|---|---|
| truth prefers depth fusion | **14 / 14** |
| silhouette IoU prefers depth fusion | **0 / 14** |
| the two disagree | **14 / 14**, exact p = 1.2 × 10⁻⁴ |

Every plant, both species, no exceptions. Mean IoU against the truth: carve
0.225, fusion 0.494. Mean silhouette IoU: carve 0.483, fusion 0.426. The
reprojection metric is not noisy on this class of subject; it is **systematically
inverted**, and it is inverted by a margin as large as the one it is being used
to measure.

| | median volume, times true | IoU vs truth |
|---|---|---|
| silhouette carving | **4.57×** | 0.225 |
| depth fusion | **2.05×** | 0.494 |

**Why it happens, now measurable rather than argued.** A visual hull agrees with
the silhouettes it was carved from *by construction*: that is the definition of
a hull, not a property of a good one. Reprojecting it therefore measures whether
the carve executed correctly, never whether the shape is right. A maize plant is
mostly gaps between leaves, and no azimuth ever sees through those gaps, so the
hull fills them and reprojects perfectly while being four and a half times too
large.

**The density criterion, vindicated and explained.** With mass fixed, implied
density is mass over reconstructed volume, so the 200-1000 kg/m³ band is a band
on the **volume ratio and nothing else**: at a tissue density ρ it passes
reconstructions between ρ/1000 and ρ/200 times the true volume, a window of 0.6×
to 3.0× at ρ = 600. On these fourteen plants it passes fusion **14/14** and carve
**0/14**, perfect agreement with the truth-based ranking, on the same cases
where the standard metric is wrong every time.

That is a stronger claim than §7b made. The criterion was adopted because no
reference geometry existed; it turns out to be *right* where the conventional
metric is wrong, and what it is really testing is a volume ratio. Both facts are
worth stating in the write-up, and the second one is what makes it defensible
rather than ad hoc.

**What this does not show, and the report says so in its own note field.** These
views are clean: exact poses, no sensor noise, no segmentation error, no missing
returns. A reconstruction that fails here fails for geometric reasons alone, so
this is an **upper bound on the operator**, not an estimate of what our Kinect
captures achieve. The direction is the useful one: an operator that cannot
recover a plant from perfect views will not recover one from real returns, but
it is not a claim about our specimens. Note too that fusion is still 2.05× too
large: better is not correct, and the remaining factor of two is the honest
ceiling of what twelve depth views can do on a plant.

**And it closes §7o's loose end.** The lettuce work found `hull volume only` the
worst feature set tested, below even an extruded volume, which suggested a hull
discards the concavity that defines a plant. This measures that suggestion: the
hull is 4.57× the true volume because the concavity is exactly what it cannot
see.

---

## 7q. Eight nulls are one fact, and it is about the sample size

The biomass table reports eight comparisons and marks nearly all of them "not
resolved". Read as it stands, that says the methods are much of a muchness and
the project could not choose between them. That reading is wrong, and the table
cannot correct it, because **a null result means nothing until the design says
what it could have detected**.

`eval/resolution.py` computes it. From each comparison's own paired bootstrap
interval, the standard error is the half-width over 1.96, and the smallest
difference the same design would find four times in five is 2.8 standard errors.

| comparison against geometric features | observed | this design detects |
|---|---|---|
| 2D + profile | −0.087 kg | 0.205 kg |
| fused geometry | −0.079 kg | 0.226 kg |
| direct 2D | −0.075 kg | 0.203 kg |
| mesh geometry | −0.037 kg | 0.138 kg |
| mean | +0.024 kg | 0.217 kg |
| volume allometric | +0.048 kg | 0.217 kg |
| canopy area allometric | +0.054 kg | 0.246 kg |

**The largest difference in the table is 0.087 kg. The smallest the design can
detect is 0.138 kg.** Every comparison is between one and a half and ten times
below its own detection threshold, so the eight nulls are one fact about n = 36
and not eight facts about the methods.

It follows that **no method in that table is distinguishable from predicting the
mean**, including the reference. That is a much clearer statement than eight
tags, and it is the correct one.

### The pattern across every comparison the project has made

Collected into one ledger, resolved and unresolved together, the split is not
random:

**Resolved (7).** The operator screen, 8 against 31 plausible of the same 36,
McNemar p = 1.5 × 10⁻⁵ (§7m). The metric inversion, 14 of 14 against known
geometry, p = 1.2 × 10⁻⁴ (§7p). Depth fusion recovering more of the plant than a
hull, 14 of 14 (§7p). Three of the four lettuce feature-set comparisons, on 378
plants with a cultivar held out (§7o). Label efficiency, 8 labels against 32
(§7j).

**Not resolved (10).** Seven rows of the biomass table, both DINO probe
comparisons (§5, §7h), and the surface-only condition on lettuce.

Every resolved result is either a **paired count**, where the pairing carries the
power and n = 14 suffices, or a **paired difference on several hundred samples**,
or a **ratio inside one experiment**, which does not require separating two
nearly equal error values at all. Every unresolved result is a difference in RMSE
between two methods on 36 specimens.

**That is a statement about experimental design, not about the methods**, and it
is the most useful thing the null results have to say. Three of them are now
resolved on the lettuce set using the same estimator and the same protocol, which
is the direct demonstration: the comparisons were fine, the sample was not.

### How to report it

Lead with the ledger, not the table. Quote the detectable column beside every
null, so a reader can see the design was asked a question it could not answer
rather than that it answered "no". State once, plainly, that no method on our own
36 specimens separates from the mean, and that batch label alone beats all of
them (§7l). Then give the resolved results their own space, because they are the
findings and they are currently buried among eight grey tags that say nothing.

```bash
python -m ggssvt.cli resolution
```

---

## 7r. The staging defect, and what it explains

Aaron pointed out that E001 and E003 to E010, and the Mango batch, were staged on
an inverted pot used as a pedestal, with the plant in a plastic bag standing on
top of it, and asked why the camera cannot pick up the stem and the leaves.

**It can. The segmentation finds them. The carve discards them.**
`eval/pedestal.py`, run with `cli pedestal`.

On E001 the carve stops at 0.456 m while the segmentation reaches 1.257 m, and
24,020 masked points sit above the carve in a column whose median radius about
the plant axis is 5.8 cm. That is a seedling roughly 66 cm tall, deleted. Across
the set:

| | |
|---|---|
| specimens losing more than 15 cm of segmented plant | **17 of 36** |
| of those, a narrow column rather than mask leak | **14** |
| median height discarded | **0.805 m** |

The worst are E010 at 1.016 m lost, E008 at 0.960 m, E007 at 0.914 m, E019 at
0.897 m and M008 at 0.819 m with 139,600 points thrown away.

**Why a thin stem cannot survive this carve.** A stem two centimetres across is
thinner than a 12 mm voxel, so most of the twelve cameras look straight past it
and return the background behind, which votes the voxel free. A voxel survives
only when at most three of twelve dissent. Broad Mango leaves clear that bar;
Eucalyptus seedlings do not.

**This explains most of §7l.** The ten pedestal specimens report 3.72 to 4.26 L
for masses spanning 0.40 to 0.70 kg. They are measuring the same stand ten times,
so within that batch the reconstruction holds nothing about the plant for a model
to use, and batch membership is the only signal left. The confound is a
consequence of this defect rather than a property of the dataset.

### The two defects are coupled

A first sweep of the carve thresholds scored zero at every setting. The reason is
instructive: it measured volume above the rim the cache carries, and on exactly
these specimens that rim is the 0.28 m fallback, which sits *inside* the stand.
The stand was being counted as plant, so no carve setting could bring the density
inside the band.

Scoring above the top of the stand instead, one hand-checked setting on E001,
`max_carve_votes = 5` with `min_informative_views = 8`, gives 2.32 L at 237 kg/m³,
inside the plausibility band where the current setting gives 0.01 L. **Fixing the
carve without fixing the rim changes nothing, and the reverse is equally true.**
That single setting on one specimen is a direction, not a result;
`eval/recarve.py` runs the sweep with the criterion fixed in advance and positive
controls included.

---

## 7s. The pot masses, and why they cannot be reverse-estimated

Thirty-one of the forty-two captures have an estimated rather than weighed pot,
and the reported plant mass is the total minus that estimate. Eleven were weighed,
all in the V batch, so there is something to calibrate against.
`eval/pot_mass.py`, run with `cli pot-mass`.

The check is §7b's implied-density argument turned on the pot: mass over the
volume the reconstruction puts below the rim.

| batch | source | implied pot density |
|---|---|---|
| V001 to V008 | weighed | **312 to 485 kg/m³** |
| E011 to E020 | estimated | **249 to 321 kg/m³** |
| E001 to E010 | estimated | **42 to 80 kg/m³** |
| M001 to M010 | estimated | **36 to 80 kg/m³** |

Damp potting medium in a plastic pot is a real material. The weighed pots land
where they should, and **E011 to E020 land with them**, so those estimates need no
correction. E001 to E010 and the Mango batch land where no potting medium can be.

**But they cannot be corrected from this, and the reason matters.** Those are
exactly the specimens raised on a stand, and their below-rim hull contains that
stand. The stand was never weighed, because it is not part of the specimen, so
the density is low for a reason that has nothing to do with the estimate. Reverse
estimating a pot mass from a volume containing unweighed furniture would replace
one error with a larger one.

Mass does follow volume where the geometry is clean, at **r = +0.871** across the
eight weighed pots, 250 g per litre with a residual of 2.19 kg. The method is
sound. It is not identifiable for the specimens that raised the question, and it
will not be until the stand can be separated from the pot, which is the same
unsolved problem as finding the rim.

### A separate hazard in the V batch

The V batch pot is **10.5 to 46.4 times the plant**, so plant mass is a small
difference between two large weighings. At 50 g of scale error on each, V002's
500 g plant carries 14% uncertainty and V008's 600 g plant 12%, before any
reconstruction is attempted. The batch collected specifically to break the
confound carries a target noise floor that no method can get beneath, and the
error propagation should be reported alongside its results rather than left
implicit.

---

## 7t. A third operator, after Nombambela (2025)

Odwa Nombambela's EPR402 report, *Plant Mass Estimation Using 3D Modelling*
(University of Pretoria, November 2025, same study leader), reconstructs from
four Kinect v2 views and takes the volume as **the count of occupied voxels in
the registered surface point cloud** at 7 mm. No carving, no signed distance
field. `eval/surface_mesh.py`, run with `cli surface-mesh`.

His sensor and intrinsics are identical to ours, so the operator transfers
directly. It is reimplemented here rather than copied: the operator is one line
of arithmetic once stated, this repository is public and his report is
unpublished coursework. Verified against his own output, where plant 1's
0.00349071 m³ is exactly 10,177 voxels of 7 mm.

**On our 36 specimens, at the same rim and the same density band:**

| operator | inside 200 to 1000 kg/m³ |
|---|---|
| silhouette carving | 8 of 36 |
| **surface voxels, his four views** | **13 of 36** |
| surface voxels, our twelve views | 7 of 36 |

Paired over the same specimens, surface-at-four-views beats the carve on 5 and
loses on none, exact p = 0.063. Suggestive, not resolved, which at 5 discordant
pairs is the most the design can say.

### The finding that matters more than the count

**The volume this operator reports is a property of the sampling, not of the
plant.** Twelve views give a median of **2.00 times** the volume four views give,
range 1.67 to 2.40. Every additional view lays down more surface points and more
points fall in more voxels; nothing about the plant changed.

That is not fatal to his result and it is worth saying why. His protocol fixes
the view count at four for every specimen, so the bias is a constant scale factor
across his set and a regressor fitted on those features absorbs it. It does mean
the figure is not a volume in any transferable sense, and that two studies using
this operator at different view counts cannot be compared to each other.

It is also the mirror image of our own failure. A hull is too large because it
fills what it cannot see; a surface count is whatever the sampling makes it. Both
are wrong in ways the implied-density criterion catches and silhouette agreement
does not.

### What cannot be borrowed

**His ground truth is the plant and its pot together** ("place plant (including
pot) on scale", report p. 72), and no pot is subtracted anywhere in his pipeline.
That is why his 40 masses span only 0.85 to 1.75 kg. His trained regressor
therefore cannot be applied to our specimens at all, and any comparison of
reported accuracy between the two projects is meaningless without stating it.

**His dataset is one plant.** `weights.txt` carries 40 labels; `data_collection`
holds all four depth views for plant 1 alone, and a single RGB array each for
plants 8 to 12. The workflow runs and reproduces his published numbers exactly,
which is more than most shared code manages, but the 40-plant result cannot be
re-derived from what was shared.

---

## 8. Bugs found and fixed

**Evaluation tracked gradients.** `predict()` put the model in `.eval()` and
called it without `torch.no_grad()`. `.eval()` switches dropout and
normalisation; it does not stop autograd. So predicting one specimen built a
graph over every intermediate activation of a full query grid and held it, and
the decoder's existing `chunk` argument was never passed, so the whole grid went
through at once.

All seven campaign runs died the same way: pretraining completed its 120 epochs,
the first evaluation forward allocated **14.19 GiB**, and the next 576 MiB
request failed. **The tell was that the number did not move**: a 19.3M CNN and
a 105.9M ViT both failed at 14.19 GiB, so it was activations and not weights.
Nothing on this machine could have caught it: there is no GPU here, and on CPU
the same code merely runs slowly.

Separately, `execute()` passed `model.state_dict()` into the folds, which hands
over references to live GPU tensors, and released the pretrained model only
*after* LOOCV. The weights are copied to the host and the model freed first now.

**`torch.utils.checkpoint` was never imported.** `attention.py` called
`torch.utils.checkpoint.checkpoint` behind `use_checkpointing`, which defaults
to True, on a path taken whenever the model is in training mode. A bare
`import torch` binds that submodule on torch 2.13 and does not on 2.5.1+cu121,
so the development machine could not see it and the lab machine crashed on the
first forward pass. **This would have killed the campaign in its first training
step**, eight hours before anyone looked. Found by running the test suite on the
machine that will actually train.


Several would have produced confident wrong numbers rather than errors.

| Bug | Consequence had it stood |
|---|---|
| `.gitignore` `data/` matched at any depth | `ggssvt/data/` and `nerfstudio/…/data/` never committed; fresh clones fail at import |
| Carve thresholds did not scale with view count | 4-view carve returns **empty**, reported as 0/30 usable, looks like a finding |
| Subject axis from single-view candidate | Confident registration onto background a metre behind the plant |
| Token anchors averaged background pixels | Anchors dragged off the specimen, corrupting the distance bias |
| Decoder used self-attention over concatenated queries | O((Q+N)²), several GB where O(Q·N) was needed |
| Decoder MLP not gain-corrected for GELU | Signal attenuated 85× at init; near-constant output |
| HF access checked `.gated` not `auth_check` | DINOv3 would stay skipped **after** approval arrived |
| 6-connectivity in component cleanup | Diagonal stems severed; 0.3 m of plant amputated on E002 |

---

## 9. Built but not yet run

All of this needs the lab GPU.

| | Status |
|---|---|
| GG-SSVT training (`pretrain`, `loocv`) | Implemented, tested, never run |
| Geometry-grounding ablation (`--no-geometry`) | Implemented, tested, never run |
| Trained 2×2 / 2×3 factorial | Implemented, never run |
| DINOv3 | Access approved, never run |
| DUSt3R / MASt3R / Fast3R | Comparison maths verified synthetically; adapters written, **never executed against real weights** |
| Nerfstudio splatfacto | `transforms.json` exported for all 30; never trained |

---

## 10. What to change in the research scope

### Change

**Reframe the biomass claim.** "Reconstructed geometry separates plant size
classes" is supported. "Estimates biomass" is not, and an examiner who checks
the batch structure will find this in minutes. Saying it first is far stronger
than being asked.

**Keep H3, restated**, see [HYPOTHESIS_3.md](HYPOTHESIS_3.md). An earlier draft
recommended dropping it on the grounds frequency grounding was never implemented;
that was wrong. The Fourier positional encoding is exactly what the proposal's
wording names, and once measured H3 has four of six sub-claims already
established, including a counter-intuitive one with a published precedent to cite
against.

**Split H1 and H4.** Each bundles separable claims; only some have evidence.

**Fix two factual errors in the proposal abstract:** the sensor is Kinect v2, not
Intel RealSense; the target is fresh mass, not oven-dry AGB.

### Reconsider

**The dataset is the binding constraint, not the method.** Two clusters of
similar plants at n=28 cannot support a biomass estimation claim regardless of
architecture.

> **Decided against, 1 September 2026.** This recommended one more capture
> campaign. There will not be one. The constraint is real and the diagnosis
> stands, but it is now met by measuring the confound (§7l) and by external
> validation on public data rather than by collecting more: Pheno4D for the
> reference geometry this project has never had, and the 4TU lettuce set (388
> plants, RGB-D, destructive fresh mass) for the regression. Both are ungated;
> see [CAMPAIGN.md](CAMPAIGN.md) § "The no-new-data plan".

**Inverse procedural modelling is the principled route past the leaf-area
ceiling.** The area hypothesis failed because a hull's surface is envelope area.
CropCraft's approach, fitting a biologically plausible parametric model whose
leaves are explicit, makes leaf area a *model parameter* rather than something
the sensor must resolve. It needs a mango/eucalyptus morphology model, which is
substantial, but it is the direction that addresses the actual obstacle.

### Report readily, today

These stand on evidence already collected, with intervals:

1. **Calibration-free rig registration** from depth alone, method, diagnostics,
   and the visual-verification failure case. A contribution the proposal did not
   anticipate.
2. **Depth fusion beats silhouette carving on biomass**, 0.335 against 0.544,
   paired bootstrap −0.209 [−0.363, −0.066], with direct 2D unchanged as the
   control. This replaces the withdrawn "reconstruction beats pixels" claim
   struck through in section 3: that comparison never resolved, this one does.
3. **Volume allometry fails across morphologies** (R² = −0.162), motivates a
   learned density.
4. **Surface area does not beat volume, and why**, envelope area is not leaf
   area. A mechanism that generalises to any hull-based method.
5. **View-count requirement**, monotone degradation, and 12 views justified
   against the 4-view protocol used previously.
6. **The batch confound**, reported as a limitation. It is a methodological
   contribution about evaluating small phenotyping datasets.
7. **The F-score / voxel-IoU gap**, implemented and demonstrated on a synthetic
   shell (F-score 1.0 at IoU 0.58); this is Paper 1's thesis.

### Cheap data fixes worth doing before the next capture

- Weigh a sample of empty pots; record real pot heights per specimen
- Capture the ChArUco sequence `dataset/README.md` already specifies
- Step the rig back for tall specimens, or add a raised second tier
- Fix the camB naming in `collect_specimen.py`
- Record a continuous mass range within one species
