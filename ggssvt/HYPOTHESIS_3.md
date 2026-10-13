# H3, restructured: frequency grounding

**Correcting earlier advice.** [RESEARCH_STATUS.md](RESEARCH_STATUS.md)
recommended dropping H3 on the grounds that frequency grounding was never
implemented. That has changed and it is now fully implemented

The proposal's wording is *"the combination of frequency and geometry grounding
improves parameter efficiency **especially as relates to positional encodings for
3D spectral features**."* That mechanism is already in the model:
`FourierFeatures` builds a geometric ladder of frequency bands and both the token
embedding and the occupancy decoder position themselves with it. What was never
implemented is a *separate wavelet branch*, the WaveFormer-style component. I
conflated the two.

Once measured rather than assumed, H3 turns out to be **the best-evidenced
hypothesis in the proposal**: four of its six sub-claims are already established
on collected data, and the two that are not need one GPU run each.

---

## The two frequency axes

The reference papers point at two different meanings of "frequency", and both
are live in this work.

**Spectral frequency**, the Fourier content of the plant's geometry, and the
bandwidth of the encoding that represents it. This is the survey's territory
(*Frequency-Informed Vision and Learning*, and the Frequency Principle) and the
SFAE paper's.

**Angular sampling frequency**, how densely viewpoints are sampled around the
plant. This is Bresolin et al.'s question in *J. Dairy Sci.* 106:664–675, where
the finding was that the optimal image-acquisition frequency is
**phenotype-specific**, not universal. The direct analogue here is the view-count
ablation.

Treating both under one hypothesis is not a stretch; they are the two ways the
sampling and representation of a plant can be band-limited, and the interesting
result below is that **they are independent**.

---

## H3 restated

> Frequency grounding acts on two independent axes, the spectral bandwidth of
> the positional encoding, and the angular sampling rate of acquisition, and the
> requirement on each is **structure-specific rather than universal**. Matching
> the encoding to the representable bandwidth improves parameter efficiency at no
> cost in reconstruction quality.

Six sub-claims, each a separate measurement.

---

## H3a, Plant occupancy spectra are structure-specific ✅ **established**

Radial power spectrum of each carved 128³ volume, 28 specimens.

| group | n | 95% bandwidth | high-frequency share |
|---|---|---|---|
| Eucalyptus E001–E010 (small, mostly pot) | 10 | 29.3 ± 2.0 cycles/m | 0.128 ± 0.010 |
| Eucalyptus E011–E020 (thin saplings) | 8 | 40.6 ± 2.3 cycles/m | 0.254 ± 0.045 |
| Mango (dense canopy) | 10 | **41.7 ± 0.0 cycles/m** | **0.273 ± 0.020** |
| **Eucalyptus V001–V008** | **8** | **41.7 ± 0.0 cycles/m** | **0.257 ± 0.025** |

High-frequency energy **doubles** from the smooth potted specimens to the
branching ones. A pot is a smooth solid of revolution and lives at low
frequency; a canopy of thin leaves does not. The spectrum measures exactly the
structural difference that the rest of this project keeps running into.

## H3b. The encoding is over-provisioned for the grid ✅ **established**

Over the working extent (128 voxels × 12 mm = 1.536 m):

| config | top frequency | encoding dims | model params |
|---|---|---|---|
| 16 bands @ 2¹⁰ | 333.3 cycles/m | 99 | 19,340,578 |
| **10 bands @ 2⁸ (current)** | **83.3 cycles/m** | **63** | **19,326,754** |
| 8 bands @ 2⁷ | **41.7, the grid Nyquist exactly** | 51 | 19,322,146 |
| 6 bands @ 2⁶ | 20.8 cycles/m | 39 | 19,317,538 |
| voxel grid Nyquist (12 mm) | 41.7 cycles/m | n/a | n/a |

**The current encoding reaches one full octave above what the grid can
represent.** Everything above 41.7 cycles/m is spent describing detail the
occupancy field cannot carry, and 2⁷ is the exact match.

**But state the efficiency claim honestly.** Trimming 10 bands to 6 removes 24
encoding dimensions and **9,216 parameters out of 19.3 million, 0.05%**. As a
parameter count that is nothing, because the encoding feeds an MLP whose width
dominates the budget. The defensible version of H3's "parameter efficiency" is
therefore not *fewer weights* but *representational allocation*: whether the
capacity spent on unrepresentable octaves buys anything. H3e measures that, and a
null result there is itself the finding. It would say the encoding's reach is
not what limits this model, which is worth knowing precisely because the proposal
assumed otherwise.

## H3c, Half the dataset saturates the grid Nyquist ✅ **established**

Every Mango specimen **and every V specimen** reports a 95% bandwidth of
**41.7 ± 0.0** cycles/m, exactly the Nyquist limit. Not near the ceiling, *at*
it, for all eighteen.

**The voxel resolution, not the method, is the binding constraint for half the
dataset.** Their true spectral content exceeds what a 12 mm grid can represent.
No architectural change recovers detail the grid cannot hold; only a finer grid
can, and that is a directly testable prediction.

The V batch strengthened this from a Mango quirk into a general property of the
morphologies here, and it arrives alongside the finding that these same
reconstructions are canopy envelopes rather than plants
([RERUN_V_BATCH.md](RERUN_V_BATCH.md) §3). Both point at resolution: a grid too
coarse to resolve a leaf cannot do anything but enclose the space around it.
**Of everything in this project, halving the voxel size is the change with the
clearest predicted effect.**

## H3d. The two axes are independent ✅ **established, and counter-intuitive**

Reconstruction versus angular sampling, by structural group:

| group | 3 views | 4 | 6 | 12 | usable at 3 views |
|---|---|---|---|---|---|
| Euc E001–E010 | 0.308 | 0.312 | 0.534 | 0.651 | **0.10** |
| Euc E011–E020 | 0.374 | 0.398 | 0.491 | 0.579 | 0.62 |
| Mango | 0.376 | 0.475 | 0.563 | 0.646 | **1.00** |

*(multi-view agreement; usable fraction under the quality gate)*

**The optimal angular sampling rate is structure-specific**, Bresolin et al.'s
phenotype-specific finding, reproduced in a completely different modality.

And the direction is the opposite of what the spectra would predict. Mango has
the **highest** spectral bandwidth yet tolerates the **fewest** views: all ten are
usable at three views. E001–E010 has the **lowest** bandwidth yet needs the most:
one in ten is usable at three views.

The two axes measure different things. Spectral bandwidth is about how fine the
detail is *once reconstructed*; angular sampling is about how much silhouette
evidence is needed to constrain the hull *at all*. A large plant filling the frame
gives every view plenty of evidence; a small tuft on a large pot gives very little
per view and needs many of them. **Bandwidth does not predict view requirement**,
and that non-obviousness is what makes the pair worth reporting together.

## H3e, Where the encoding's reach should sit ⏳ **needs training**

The prediction from H3b. Three runs bracket the grid Nyquist against
`baseline_cnn`'s 2⁸:

| run | top frequency | vs. Nyquist | prediction |
|---|---|---|---|
| `h3_bands_8_freq7` | 41.7 cycles/m | matched | no change from baseline |
| `h3_bands_6_freq6` | 20.8 cycles/m | half | should hurt |
| `h3_bands_16_freq10` | 333.3 cycles/m | 8× | should add nothing |

```bash
python -m ggssvt.campaign --plan core --device cuda
```

Report occupancy AP, best-threshold IoU and biomass RMSE for each, with the
paired bootstrap against baseline. The shape of the curve is the result: flat
from 2¹⁰ down to 2⁷ and dropping at 2⁶ confirms the grid Nyquist is the right
place to set the encoding, and that the mechanism is band-limiting rather than
capacity. A curve that is flat *everywhere*, 2⁶ included, says the encoding's
reach is not a binding constraint on this model at all, also a result, and one
that redirects the efficiency argument away from the encoding.

Do not report this as a parameter saving. It is 0.05% of the weights.

## H3f, The Frequency Principle governs what converges ⏳ **needs training**

The F-Principle predicts a network fits low frequencies first. For a branching
plant the low frequencies are the pot and the trunk; the high frequencies are the
branches. The structures this project keeps failing to recover.

`eval/frequency.band_error` decomposes reconstruction error by radial frequency
band. Logging it per epoch during pretraining tests the prediction directly, and
if the high-band error plateaus while the low bands converge, that is a
*mechanistic* explanation for the thin-structure problem rather than an
observation of it, and it connects the spectral argument to Paper 1's
F-score-versus-IoU gap.

Cheap to add: call `band_error` on a held-out specimen every ten epochs.

---

## Why this is worth keeping

H3 was the weakest hypothesis on paper and is now among the strongest in
evidence. Four sub-claims are established on collected data, two need one GPU run
each, and one of them, H3d, is a genuinely counter-intuitive finding with a
published precedent in a different field to cite against.

It also earns its place in the argument rather than sitting beside it:

- **H3c explains a limit.** Mango is resolution-limited, not method-limited. That
  bounds what any method on this data can achieve, which the results chapter
  needs.
- **H3b/H3e are about representational capacity, not means.** That is the kind
  of claim that survives n=28, where a difference in group averages does not.
  The measured parameter saving is negligible, so the claim must be made about
  where the encoding's reach should sit, not about weight count.
- **H3f would explain the thin-structure failure** that the mesh arm ran into and
  that Paper 1 argues about.

**Recommendation: keep H3, restated as above.** Drop only the WaveFormer/DWT
branch, which was never implemented and is not what the proposal's own wording
requires. Reference [27] can stay as context for the frequency-domain literature
rather than as a component claim.

One caveat to carry: the encoding-reach and grid-Nyquist numbers are properties
of the *configuration*, not measurements of the model's behaviour. They predict
what H3e should find; they do not substitute for running it.
