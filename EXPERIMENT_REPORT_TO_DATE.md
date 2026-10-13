# GG-SSVT Experiment Report to Date

**Report date:** 10 September 2026  
**Dataset:** 38 labelled specimens in the current cache (37 specimens in the
frozen-feature probe because V010 was excluded by that analysis)  
**Evaluation protocol:** leave-one-out cross-validation unless otherwise stated  
**Primary metrics:** RMSE, MAE, MARE, and R²

## 1. Executive summary

The experiments completed so far do not support the original claim that the
end-to-end GG-SSVT model is currently the best biomass estimator. The strongest
results come from the reconstruction and regression pipeline:

1. **Frozen DINOv2 features:** RMSE 0.400 kg, R² 0.483 on 37 specimens.
2. **Frozen DINOv2 plus geometry:** RMSE 0.400 kg, R² 0.483 on 37 specimens.
3. **Classical fused geometry:** RMSE 0.430 kg, R² 0.416 on 38 specimens.
4. **Classical geometric features:** RMSE 0.469 kg, R² 0.309 on 38 specimens.

The trained campaign has not produced a scientifically acceptable
end-to-end result yet. `baseline_fused` completed with RMSE 0.557 kg and R²
0.024. The other completed campaign runs were blocked because they failed the
project's quality or "must beat the mean" gates. DINOv2 was skipped in that
campaign because the pretrained model could not be downloaded at run time.

The present evidence indicates that **reconstruction quality and data
calibration are the main limitations**, not simply model capacity.

## 2. Dataset and protocol

The data consist of RGB-D views from three specimen groups:

- **Eucalyptus:** E001-E020, with some identifiers absent from the labelled set.
- **Mango:** M001-M010.
- **Vine:** V001-V010.

The pipeline uses registered RGB-D views, segmentation, space carving or TSDF
fusion, feature extraction, and leave-one-out biomass regression. Preprocessing
and feature standardisation are performed within the intended evaluation
protocol. Results from the trained campaign should not be compared as if they
were interchangeable with the classical and frozen-feature results: they use
different model stages and, in some cases, different specimen counts.

## 3. Classical baseline results

Results from `work_dirs/ggssvt/reports/metrics.json`:

| Method | RMSE (kg) | MAE (kg) | MARE | R² | n |
|---|---:|---:|---:|---:|---:|
| Mean predictor | 0.579 | 0.493 | 0.590 | -0.055 | 38 |
| Volume allometric | 0.619 | 0.514 | 0.535 | -0.207 | 38 |
| Geometric features | 0.469 | 0.375 | 0.402 | 0.309 | 38 |
| Direct 2D | 0.591 | 0.461 | 0.493 | -0.100 | 38 |
| 2D + profile | 0.746 | 0.485 | 0.466 | -0.755 | 38 |
| **Fused geometry** | **0.430** | **0.352** | **0.393** | **0.416** | **38** |

TSDF fusion is therefore a material improvement over the original carved
geometry. It is the best result that uses all 38 specimens in the current
metrics report.

### 4D. What the reconstruction approach achieved

The reconstruction-based biomass approach followed four stages:

1. register the RGB-D views in a common coordinate system;
2. segment the plant and produce either a silhouette-carved or TSDF-fused
   occupancy volume;
3. summarize the reconstruction with volume, height, projected area,
   compactness, and related geometric descriptors; and
4. fit a biomass regressor under specimen-level leave-one-out cross-validation.

This approach produced useful but incomplete biomass prediction. Volume-only
allometry failed to beat the mean predictor (RMSE 0.619 kg, R² -0.207),
because reconstruction volume was not consistently proportional to plant
mass. Geometric features improved over the mean (RMSE 0.469 kg, R² 0.309),
and TSDF-fused geometry performed better still (RMSE 0.430 kg, R² 0.416).
Thus, 3D structure contained biomass signal, but reconstruction quality was a
larger limitation than regression capacity. The later frozen DINOv2 plus
fused-geometry ridge model improved the current result to approximately
RMSE 0.361 kg and R² 0.581, on the probe's 37 specimens.

#### M008 reconstruction example

M008 illustrates why raw reconstructed volume cannot be interpreted directly
as biomass. Its target fresh mass was 0.56 kg. The carved reconstruction
occupied approximately 6.42 L, implying only about 87 kg/m³, whereas the
TSDF-fused reconstruction occupied approximately 0.65 L, implying about
860 kg/m³. The carved result therefore retained substantial excess or
background geometry and made the plant appear unrealistically large and
sparse. Fusion produced a substantially more plausible volume, although it
remains a measured reconstruction rather than a guarantee of true plant
volume. M008 is retained as a reconstruction-quality diagnostic, not removed
from the primary analysis.

### Reproducing the reconstruction and prediction for another sample

From the repository root, activate the `ggssvt` environment and place the
sample's RGB-D capture in the expected dataset layout. The commands below
rebuild the cached preprocessing, generate the TSDF reconstruction, and
produce the classical baseline outputs:

```bash
conda activate ggssvt
python -m ggssvt.cli preprocess \
  --cache-dir work_dirs/ggssvt/cache
python -m ggssvt.cli fuse \
  --cache-dir work_dirs/ggssvt/cache \
  --write-cache work_dirs/ggssvt/cache_tsdf \
  --out work_dirs/ggssvt/reports/fusion.json
python -m ggssvt.cli baselines \
  > work_dirs/ggssvt/reports/baselines.txt
```

To process only selected specimens, add `--plants M008` (or replace it with
one or more plant identifiers) to both `preprocess` and `fuse`. The
preprocessing command writes the registered RGB-D cache; `fuse` writes the
TSDF occupancy and descriptors; `baselines` evaluates the available
geometry-based regressors using leave-one-out cross-validation over the
available labelled set.

To recreate the visual evidence for a sample, render its RGB frames, masks,
depth panels, carved volume, fused volume, measured points, and shaded
illustration:

```bash
python -m ggssvt.cli filmstrip --limit 1
python -m ggssvt.cli show \
  --plants M008 \
  --layers rgb segmentation depth occupancy points \
  --size 400
```

The filmstrip command processes specimens in dataset order and writes one
folder per specimen; use the generated report and static assets under
`work_dirs/ggssvt/reports/filmstrip/` to find the requested identifier. The
shaded render is only an illustration. The occupancy and measured-point panels
are the evidence used to diagnose reconstruction quality.

To reproduce the primary DINOv2-plus-geometry biomass result across the
labelled set, run:

```bash
python -m ggssvt.cli dino-probe \
  --cache-dir work_dirs/ggssvt/cache \
  --backbones dinov2 \
  --variant base \
  --components 8 \
  --alphas 0.1 1.0 10.0 \
  --out work_dirs/ggssvt/reports/dino_probe.json
python -m ggssvt.eval.ab_ssvit \
  --cache-dir work_dirs/ggssvt/cache \
  --variant base \
  --alpha 1.0 \
  --out work_dirs/ab_ssvit/ab_ssvit.json
```

This estimator is not fitted separately to one new unlabelled plant. It
requires labelled specimens to fit the ridge biomass head, while the held-out
specimen is transformed using training-fold PCA and scaling. For a genuinely
new sample, first add its capture and independently measured mass to the
dataset, rebuild the caches, and rerun the full evaluation. Do not report a
single-sample prediction as validated accuracy unless that sample was held out
from fitting.

## 4. Frozen-feature experiments

Results from `work_dirs/ggssvt/reports/dino_probe.json`:

| Method | RMSE (kg) | MAE (kg) | MARE | R² | n |
|---|---:|---:|---:|---:|---:|
| CNN/no DINO | 0.471 | 0.370 | 0.362 | 0.284 | 37 |
| **DINOv2-base** | **0.400** | **0.314** | **0.306** | **0.483** | **37** |
| DINOv2-base + geometry | 0.400 | 0.314 | 0.306 | 0.483 | 37 |
| DINOv3-base | 0.409 | 0.316 | 0.315 | 0.459 | 37 |
| DINOv3-base + geometry | 0.412 | 0.318 | 0.317 | 0.451 | 37 |

The DINOv2 improvement over the CNN control is promising, but its paired
bootstrap interval includes zero (`-0.193` to `0.062` kg RMSE difference).
The added seven geometry features did not materially improve DINOv2.

The label-efficiency analysis found that DINOv2 and DINOv3 reached the
full-label geometric reference performance using approximately 18 labels,
whereas the geometric reference used 36 labels. This is useful evidence for
label efficiency, but it should be treated cautiously because the sample is
small and V010 is excluded.

## 5. End-to-end campaign status

Saved campaign files are under `work_dirs/ggssvt/campaign/`.

| Experiment | Status | RMSE (kg) | R² | Interpretation |
|---|---|---:|---:|---|
| `baseline_cnn` | Blocked | 1.738 | -8.511 | Failed acceptance gates and did not beat the mean |
| `baseline_fused` | Done | 0.557 | 0.024 | Valid run, but weaker than classical fused geometry |
| `h1_dinov2` | Blocked | 1.117 | -2.930 | Pretrained DINOv2 loaded on retry, but end-to-end biomass result failed the mean-predictor gate |
| `h2_no_geometry` | Blocked | 3.111 | -29.482 | Failed badly without geometry |
| `h3_bands_6_freq6` | Blocked | 1.858 | -9.875 | Failed acceptance gates |
| `h3_bands_8_freq7` | Blocked | 3.181 | -30.852 | Failed acceptance gates |

At the time of this report update, the orchestrator is processing the seventh
and final core condition, `h3_bands_16_freq10`. The full plan (`sam3d_cnn`,
`sam3d_dinov2`, and `h1_dinov3`) remains pending. A complete core-plus-full
study cannot be honestly completed within a two-hour window: a single
successful 36--38-fold trained condition previously required roughly 15--17
hours, whereas blocked checkpointed conditions can finish in minutes.

## 6. Ablation architectures and why they are retained

The blocked conditions are not discarded. They isolate specific architectural
claims and provide negative evidence for the dissertation. Their purpose is to
show which ingredients are necessary, not to serve as competing production
estimators.

| Ablation | Architecture | Isolates | Why retain it |
|---|---|---|---|
| `baseline_cnn` | RGB-D views → trainable CNN stem → geometry-grounded occupancy decoder → volume biomass head | End-to-end learned baseline | Establishes whether the proposed training pipeline works without a pretrained ViT |
| `baseline_fused` | TSDF-fused RGB-D → trainable CNN/geometry-grounded decoder → biomass head | Whether the neural model benefits from improved fusion | Completed, but weaker than classical fused geometry |
| `h2_no_geometry` | RGB views → appearance/CNN features → occupancy and biomass heads, without world-coordinate geometry | Contribution of geometric grounding | Its severe failure supports retaining geometry in the main architecture |
| `h1_dinov2` | RGB-D → frozen/pretrained DINOv2 stem inside the end-to-end decoder → biomass head | Contribution of the ViT representation within the original neural model | Distinguishes a strong backbone from a weak integration/training strategy |
| `h3_bands_6_freq6` | Geometry-grounded decoder with reduced Fourier positional encoding | Whether lower positional bandwidth is sufficient | Tests the frequency-efficiency hypothesis |
| `h3_bands_8_freq7` | Geometry-grounded decoder with grid-matched Fourier encoding | Whether matching encoding bandwidth to the voxel Nyquist limit helps | Tests whether excess positional bandwidth caused instability |
| AB-SSViT primary | Frozen DINOv2 descriptors + TSDF geometry → fold-local PCA/scaling → ridge | Small-sample multimodal estimator | Current best practical architecture; avoids millions of trainable parameters |

The ablation comparisons should use the same specimens, targets, and
specimen-level cross-validation wherever possible. A blocked result remains
reported with its metrics, but is labelled unusable as a production estimator
when it fails the acceptance gates.

### Why the acceptance gates remain

The gates prevent a misleading result from being promoted:

- finite metrics prevent silent numerical failure;
- prediction-spread checks detect collapse to a constant;
- the mean-predictor comparison ensures the model contains predictive signal;
- occupancy-above-chance checks protect the self-supervised reconstruction stage;
- loss-decrease checks detect runs that did not train.

Relaxing these gates would not improve the experiment; it would only relabel
failed models as successes. The correct response is to retain the failed
architectures as ablation evidence and use the stable AB-SSViT ridge pipeline
for the primary biomass estimate.

## 7. Specimen-level error and reconstruction findings

The most influential errors in the completed `baseline_fused` trained run
were:

| Specimen | Target (kg) | Prediction (kg) | Error (kg) | Diagnostic evidence |
|---|---:|---:|---:|---|
| **V001** | 1.00 | 3.36 | +2.36 | Fused voxel IoU 0.141; F-score 0.410 |
| **V010** | 1.90 | 3.05 | +1.15 | Fused voxel IoU 0.070; F-score 0.194 |
| **E020** | 1.95 | 0.86 | -1.09 | Silhouette IoU 0.322; coverage 0.338 |
| E013 | 2.35 | 1.44 | -0.91 | Large underprediction |
| E014 | 1.80 | 1.13 | -0.67 | Large underprediction |

V001 is the largest single influence in this run: removing it from the
already-generated prediction vector changes RMSE from 0.557 to approximately
0.410 kg. This is a sensitivity diagnostic, not a justification for deleting
the specimen.

V001 and V010 have especially weak reconstruction diagnostics. E020 is a
thin-sapling edge case with a large target-to-prediction mismatch. The
acceptance reports also identify broader issues such as fallback pot-rim
estimation and envelope-derived densities. These results suggest that the
specimens should first be audited for calibration, masks, depth alignment,
rim detection, and target-label correctness.

## 8. Limitations

- The labelled sample is small, so single specimens can change RMSE
  substantially.
- Results use 38 or 37 specimens depending on the experiment.
- Several trained runs are blocked rather than valid positive or negative
  scientific results.
- The DINO results depend on frozen pretrained features and are not evidence
  that the full GG-SSVT architecture has been successfully trained.
- Camera calibration is estimated from depth because measured calibration
  targets are unavailable.
- V001, V010, and E020 require a data-quality audit before final claims are
  made.

## 9. Recommended next actions

1. **Audit V001 and V010 first.** Recheck camera registration, depth
   alignment, segmentation, pot-rim height, and reconstructed meshes.
2. **Audit E020 and the biomass labels.** Confirm that the thin-sapling
   geometry and measured mass are both valid.
3. **Run the AB-SSViT pipeline** using TSDF geometry plus frozen DINOv2
   features, fold-local scaling/PCA, and ridge regression.
4. **Keep all valid specimens in the primary analysis.** Report exclusions
   only when an independent quality check proves that a specimen is invalid.
5. **Report a sensitivity analysis** with audited-invalid specimens removed,
   clearly labelled as secondary.
6. **Use bootstrap confidence intervals and per-specimen residual tables** in
   the final dissertation results.

## 10. Reproducibility references

- Classical metrics: `work_dirs/ggssvt/reports/metrics.json`
- Frozen-feature results: `work_dirs/ggssvt/reports/dino_probe.json`
- Label-efficiency results: `work_dirs/ggssvt/reports/label_efficiency.json`
- Campaign outputs: `work_dirs/ggssvt/campaign/`
- Reconstruction diagnostics: `work_dirs/ggssvt/reports/reconstruction_quality.json`
- Robustness diagnostics: `work_dirs/ggssvt/reports/robustness.json`
- Per-run trained predictions: `work_dirs/ggssvt/campaign/baseline_fused.json`

New campaign runs save epoch histories to
`work_dirs/ggssvt/campaign/<run>_pretrain_history.json` and
`work_dirs/ggssvt/campaign/<run>_histories/fold_*.json`. Render a loss plot for
a fold with:

```bash
python -m ggssvt.eval.training_plots \
  work_dirs/ggssvt/campaign/baseline_fused_histories/fold_001_E001.json \
  --out work_dirs/ggssvt/reports/figures/baseline_fused_fold_001_loss.png
```

Previously completed runs contain final metrics and checkpoints but not
epoch-by-epoch histories, so their exact loss curves cannot be reconstructed
from the existing artifacts. Rerunning a smoke or campaign condition creates
the histories without changing the evaluation protocol.
