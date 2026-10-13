# The V001–V008 re-run: what the measured pot weights changed

Eight specimens were collected on 2026-08-14 and never processed. Their pots were
weighed after the shoots were cut off, so for the first time in this project the
above-ground mass of a specimen is a **measurement** rather than a total minus an
estimate. Everything below is the pipeline re-run end to end on the enlarged,
corrected set.

**Headline: one claim has to be withdrawn, one new diagnostic replaces it with
something firmer, and the new batch is the most useful one collected so far.**

---

## 1. The estimated pot weights were biased, not noisy

| id | total g | pot estimated | pot **measured** | net estimated | net **measured** |
|---|---|---|---|---|---|
| V001 | 25400 | 21900 | 24400 | 3500 | **1000** |
| V002 | 23700 | 21500 | 23200 | 2200 | **500** |
| V003 | 20500 | 17200 | 19400 | 3300 | **1100** |
| V004 | 32200 | 26800 | 30400 | 5400 | **1800** |
| V005 | 22400 | 18600 | 20600 | 3800 | **1800** |
| V006 | 22300 | 18800 | 21300 | 3500 | **1000** |
| V007 | 33800 | 28400 | 32500 | 5400 | **1300** |
| V008 | 22300 | 18950 | 21700 | 3350 | **600** |

Net shoot mass was overstated by **2.1× to 5.6×**.

The useful part is that the error is systematic:

> **pot estimated / pot measured = 0.891, sd 0.018** across all eight.

A consistent 10.9% under-estimate of pot mass with almost no scatter. That is a
calibration, not a bias to be lamented. It puts an error bar on the E and M net
masses, which are **still estimates**:

| group | net as recorded | if the same −10.9% bias applies |
|---|---|---|
| Mango | 1022 g | 911 g (−11%) |
| E001–E010 | 538 g | 409 g (**−24%**) |
| E011–E020 | 1750 g | 1540 g (−12%) |

E001–E010 is worst affected because the pot is most of the total. Carry this as a
stated uncertainty on every result involving those batches, and weigh the pots
directly on the next capture. It is a scale and ten minutes.

---

## 2. A single pot height could not survive this batch

V pots weigh **17–32 kg** against **0.7–2.2 kg** for E001–E020. The fixed
`POT_HEIGHT_M = 0.28` left a slab of pot counted as plant: V001 reported 15.4 L
of above-ground volume for a 1.0 kg shoot, an implied density of 65 kg/m³ where
fresh tissue is 300–900.

`ggssvt/geometry/pot.py` now estimates the rim per specimen from the occupancy
profile. A rim is a **step**, not a slope, so the test is local, the median
cross-section just below a candidate height against the median just above.

**It refuses more often than it answers, and that is the useful part.**

| batch | n | confident | rim height |
|---|---|---|---|
| E001–E010 | 10 | **1** | n/a |
| E011–E020 | 8 | 8 | 0.240 ± 0.008 m |
| Mango | 10 | 8 | 0.468 ± 0.055 m |
| V001–V008 | 8 | 8 | 0.354 ± 0.036 m |

E001–E010 carve as single cones, 645 voxels at the floor decaying evenly to 140
and then stopping, with no pot/plant boundary anywhere. Nine of ten now return
`confident=False` and fall back to the constant. That is the truthful answer:
**those reconstructions do not separate plant from pot at all**, which has been
suspected throughout and is now measured.

An earlier version of this estimator thresholded against the widest slice and
returned three different confident rims (0.432, 0.336, 0.432 m) for three
specimens of the same batch. Worth remembering as a caution: a detector that
always answers is not the same as a detector that works.

---

## 3. The new diagnostic: can the volume physically weigh that much?

Measured mass ÷ reconstructed above-ground volume gives an implied bulk density,
which is checkable against physics rather than against a baseline. Fresh
above-ground tissue is roughly 300–900 kg/m³.

| group | n | plausible (200–1000) | envelope (too light) | missing (too heavy) | median kg/m³ |
|---|---|---|---|---|---|
| E001–E010 | 10 | 1 | 9 | 0 | 127 |
| E011–E020 | 8 | 3 | 2 | 3 | 863 |
| Mango | 10 | **0** | **10** | 0 | **52** |
| V001–V008 | 8 | **4** | 4 | 0 | 248 |
| **all** | **36** | **8** | **25** | **3** | **117** |

**Only 8 of 36 reconstructions are the right size to weigh what the plant
weighs.** Twenty-five are envelopes: the visual hull has enclosed the air between
leaves and branches, so the quantity being regressed is the canopy envelope and
not the plant. Every Mango specimen sits at 26–77 kg/m³, one to two orders of
magnitude short, which no regressor repairs. Three imply *more* than plausible,
meaning thin stems were never carved; E019 implies 22,569 kg/m³ because
essentially none of it was reconstructed.

This is the same failure the canopy-area hypothesis hit, now measured on volume
as well as surface, and it accounts for the whole pattern of biomass results.
**It is the most useful single number this project has produced**, because it
converts "the fit is poor" into "the reconstructed object is the wrong kind of
object, by this factor, for these species".

---

## 4. The 3D-versus-2D claim has to be withdrawn

### It was never resolved, in either direction

The proposal's third research question was answered at n=28 with "reconstruction
beats pixels", 0.397 kg RMSE against 0.440. Re-tested with a paired bootstrap on
exactly that condition:

> **dRMSE −0.043 kg, 95% CI [−0.168, +0.099]. Not resolved.**

The interval spans zero comfortably. **That claim should not have been made**; a
point-estimate gap was reported as a finding without the interval that decides it.

Adding V flips the point estimate, and it stays unresolved:

| set | rim | 3D RMSE / R² | 2D RMSE / R² | ahead |
|---|---|---|---|---|
| E+M only (n=28) | constant 0.28 | 0.397 / +0.526 | 0.440 / +0.419 | 3D |
| E+M only (n=28) | per-specimen | 0.434 / +0.435 | 0.440 / +0.419 | 3D, barely |
| all incl. V (n=36) | constant 0.28 | 0.512 / +0.140 | 0.469 / +0.279 | 2D |
| **all incl. V (n=36)** | **per-specimen** | **0.544 / +0.030** | **0.469 / +0.279** | **2D** |

### And the direction depends on an arbitrary preprocessing choice

The DINO probe whitens by rotating onto principal components before
standardising; the baselines table standardises the raw features. On seven
features PCA keeps all seven, so this is a pure rotation, same data, same ridge
penalty, different regulariser. Run both feature sets through both protocols:

| protocol | 3D RMSE / R² | 2D RMSE / R² | ahead |
|---|---|---|---|
| standardise only | 0.544 / +0.030 | 0.469 / +0.279 | 2D |
| PCA-rotate then standardise | **0.458 / +0.312** | 0.491 / +0.209 | **3D** |

Paired bootstrap, 20,000 resamples, dRMSE = 3D − 2D:

| protocol | dRMSE | 95% CI | |
|---|---|---|---|
| standardise only | +0.075 | [−0.051, +0.227] | not resolved |
| PCA-rotate then standardise | −0.033 | [−0.143, +0.076] | not resolved |

**Neither direction is resolved, and the winner changes with a preprocessing
choice that has no principled justification either way.** The honest statement is
that **at n=36 this comparison cannot be settled by RMSE differences at all**,
not that 3D lost.

This also means the two tables in this project are not comparable: the probe's
"cnn (no DINO)" at R² 0.312 and the baselines table's "geometric features" at
0.030 are the *same seven features* under different preprocessing. Do not put
them in one table.

### What replaces the claim

Section 3. The implied-density result is a physical measurement rather than a
difference in means, so it does not depend on n or on a regulariser: **8 of 36
reconstructions are the right size to weigh what the plant weighs.** That is what
RQ3 should be answered with.

### Within species, where the confound is weakest, 3D fails outright

The pooled comparison mixes species. Restricting to Eucalyptus, now 26
specimens including V, the least confounded set this project has, is the
sharpest available test:

| method | RMSE | R² |
|---|---|---|
| **direct 2D** | **0.520 kg** | **+0.311** |
| mean predictor | 0.651 kg | −0.082 |
| volume allometric | 0.666 kg | −0.131 |
| **geometric features** | **0.717 kg** | **−0.313** |

**Geometric features are worse than predicting the mean.** Mango alone (n=10) is
the same story, at R² −1.971 against the mean's −0.235.

An R² below the mean-predictor floor is a different kind of statement from losing
a head-to-head: it says the feature carries no usable signal about mass at this
resolution, rather than less signal than a rival. It is also exactly what §3
predicts, a volume that is an envelope has nothing to regress against, and it
is consistent across both species. **This, not the pooled ordering, is the
result worth reporting.**

### The batch confound weakened, because V was designed to break it

| set | batch-only R² |
|---|---|
| Eucalyptus, 2 batches (the old n=18) | **0.887** |
| Eucalyptus, 3 batches incl. V (n=26) | 0.744 |
| all four batches (n=36) | **0.697** |

| batch | n | mean | sd | range |
|---|---|---|---|---|
| E001–E010 | 10 | 538 g | 114 | 400–700 |
| E011–E020 | 8 | 1844 g | 348 | 1350–2350 |
| **V001–V008** | 8 | **1138 g** | **484** | **500–1800** |
| Mango | 10 | 1022 g | 284 | 560–1470 |

V has the **widest within-batch spread of any batch** (sd 484 g, CV 43%) and its
range **overlaps every other batch**. That is exactly what the study was missing:
previously the batches sat at separate sizes, so a feature reading batch
membership scored well without measuring mass. V removes that shortcut, and this
is a real improvement to the design regardless of what happens to any RMSE.

---

## 4b. The factorial produced the project's first resolved effect

2×2, 33 shared specimens (SAM3D drops E015, E019, V006 on top of the geometric
gate's E012/E016):

|  | no DINO | DINOv2-base |
|---|---|---|
| **no SAM3D** | 0.576 kg / R² −0.080 | **0.385 / +0.518** |
| **SAM3D** | **0.778 / −0.967** | 0.390 / +0.505 |

| effect | ΔRMSE | 95% CI | |
|---|---|---|---|
| DINO alone | −0.191 | [−0.404, +0.021] | not resolved |
| SAM3D alone | +0.201 | [−0.017, +0.394] | not resolved |
| **DINO given SAM3D** | **−0.387** | **[−0.757, −0.039]** | **resolved** |
| SAM3D given DINO | +0.005 | [−0.007, +0.019] | no effect |

**The hand-crafted descriptors are fragile to the segmentation; the learned
features are not.** SAM3D alone drives the geometric descriptors to R² −0.967,
far below the mean-predictor floor, while DINO moves 0.385 to 0.390, an effect
of 5 grams.

That follows directly from §3. Those descriptors summarise a volume that is a
canopy envelope rather than a plant, so changing the mask moves them a lot and
costs nothing real; DINO reads the images and never depended on the volume
meaning anything.

Read the "resolved" honestly: it is resolved largely *because* SAM3D + no DINO is
so bad, so it evidences descriptor fragility more than it evidences DINO's value.
DINO against neither is still unresolved.

At n=26 all four cells sat within 0.295–0.317 kg with nothing resolved. Adding a
batch that does not share the others' size structure blew the spread wide open,
which is what happens to features that were reading size.

---

## 5. H3 unchanged in direction, strengthened in evidence

V re-run through the spectral characterisation:

| group | n | 95% bandwidth | high-freq share |
|---|---|---|---|
| E001–E010 | 10 | 29.3 ± 2.0 | 0.128 ± 0.010 |
| E011–E020 | 8 | 40.6 ± 2.3 | 0.254 ± 0.045 |
| Mango | 10 | **41.7 ± 0.0** | 0.273 ± 0.020 |
| **V001–V008** | 8 | **41.7 ± 0.0** | 0.257 ± 0.025 |

Grid Nyquist is 41.7 cycles/m. **H3c now covers 18 of 36 specimens, not 10**,
both Mango and V saturate the grid exactly. Resolution, not method, is the
binding constraint for half the dataset. The 12 mm voxel is now the most
defensible thing to change next.

---

## 5b. The four-view question, answered

| views | usable | agreement | mean above-ground hull | **plausible** | median kg/m³ |
|---|---|---|---|---|---|
| 3 | 23/38 | 0.360 | 99.3 L | **1/23** | 9.8 |
| 4 | 25/38 | 0.424 | 126.5 L | **0/25** | 9.2 |
| 6 | 34/38 | 0.521 | 150.4 L | **2/34** | 15.4 |
| **12** | **36/38** | **0.608** | **10.4 L** | **8/36** | **116.8** |

**At four views, zero of twenty-five reconstructions are physically capable of
weighing what the plant weighs.** Median implied bulk density 9.2 kg/m³,
lighter than expanded polystyrene, thirty to ninety times below plant tissue, and
hulls averaging 126 L for plants of at most 2.35 kg.

Four views at 90° is the visual-hull minimum for a *convex* object; a plant is
the opposite, and every unsampled azimuth leaves a prism of empty space uncarved.

So the sampling question has a physical answer rather than a judgement call: **no,
four images will not do**, and the agreement score alone would never have said so
that clearly. It degrades gently from 0.608 to 0.424 while the reconstructions
stop being reconstructions.

---

## 6. Quality gate

**36 of 38 pass, up from 28 of 30.** All eight V specimens pass. X001 is excluded
throughout: 2 views, one specimen, a species of its own.

---

## What to do with this

**Report readily.** The pot-weight bias calibration (§1). The plausibility
diagnostic and the 8-of-36 result (§3). This is a genuine methodological
contribution, not a negative result to bury. The confound weakening (§4).

**Reframe.** Drop "3D reconstruction improves biomass estimation". It was never
resolved and the point estimate is not stable. Replace it with what the data does
support: *visual-hull reconstruction of these species recovers the canopy
envelope rather than the plant, with implied bulk densities one to two orders of
magnitude below plant tissue, so its volume cannot carry mass information.* That
is a stronger chapter, because it rests on a physical measurement instead of a
difference in means at small n, and it explains the weak RMSE numbers rather
than merely reporting them.

**Do next, in order.**
1. Weigh pots directly for any future capture. Ten minutes, removes the largest
   single uncertainty in the ground truth.
2. Re-run the campaign on the GPU against the corrected targets, every trained
   number in `RESEARCH_STATUS.md` predates this correction.
3. Test the resolution prediction from §5: a finer voxel grid on the 18
   Nyquist-saturated specimens.
