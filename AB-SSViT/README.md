# AB-SSViT

**Ablation-Benchmarked Self-Supervised Vision Transformer** for biomass
estimation on the CropCraft captures.

This package turns the strongest existing evidence into a reproducible,
GPU-friendly experiment:

```text
RGB-D views
    |
    +--> TSDF fused geometry --> interpretable geometry descriptors
    |
    +--> frozen DINOv2 patch descriptors
                                   |
                    concatenate and fit ridge regression
                                   |
                         leave-one-out biomass estimate
```

The model is deliberately a **frozen-representation probe**, not a new
end-to-end neural claim. With 38 specimens and few labels, this avoids fitting
millions of trainable weights to a tiny regression set while preserving a
strict fold-local PCA, standardisation, and ridge fit.

## Architecture sketch

```text
                 specimen s
             RGB images I_s^v
             depth maps D_s^v
                    |
          [A] registration and masks
                    |
       world-aligned RGB-D observations
              /                 \
 [B] TSDF fusion/carving    [C] frozen DINOv2
              |                 |
       occupancy/grid O_s    patch tokens Z_s^v
              |                 |
 [D] geometry descriptors   view/patch pooling
              |                 |
          g_s in R^7        d_s in R^1536
              \                 /
               \               /
          [E] fold-local fusion
                    |
       x_s = [standardise(g_s), PCA(d_s)]
                    |
          [F] ridge biomass head
                    |
             y_hat_s + interval
```

### Submodules and equations

**[A] Registration and segmentation.** A view is transformed into the
plant-centred world frame and restricted to its subject mask:

\[
\mathbf{x}_{s,v,i}=\mathbf{R}_{s,v}\mathbf{p}_{s,v,i}
+\mathbf{t}_{s,v},\qquad
\mathcal{P}_{s,v}=\{(\mathbf{x}_{s,v,i},d_{s,v,i}):
M_{s,v,i}=1,\ d_{s,v,i}>0\}.
\]

The existing rig and pose modules estimate \(\mathbf{R}\) and \(\mathbf{t}\);
biomass labels are not used.

**[B] TSDF fusion.** For voxel centre \(\mathbf{x}\), valid views contribute:

\[
\phi_{s,v}(\mathbf{x})=
\operatorname{clip}\left(
\frac{d_{s,v}(\pi_{s,v}(\mathbf{x}))-z_{s,v}(\mathbf{x})}{\tau},
-1,1\right),
\]
\[
\operatorname{TSDF}_s(\mathbf{x})=
\frac{\sum_v w_{s,v}(\mathbf{x})\phi_{s,v}(\mathbf{x})}
{\sum_v w_{s,v}(\mathbf{x})+\epsilon},\qquad
O_s(\mathbf{x})=\mathbb{1}[\operatorname{TSDF}_s(\mathbf{x})<0].
\]

Here \(\tau\) is the truncation distance, \(w\) is the view-validity weight,
and \(\pi\) projects a world point into a camera.

**[C] Frozen DINOv2.** The pretrained backbone produces patch tokens:

\[
Z_s^v=f_{\mathrm{DINO}}(I_s^v)\in\mathbb{R}^{N_v\times1536}.
\]

Subject-masked pooling gives a view descriptor, followed by robust aggregation:

\[
d_s^v=\frac{\sum_jm_{s,v,j}Z_{s,v,j}}
{\sum_jm_{s,v,j}+\epsilon},\qquad
d_s=\operatorname{median}_v(d_s^v).
\]

The backbone remains frozen; no biomass target is used to adapt it.

**[D] Geometry descriptors.** From \(O_s\), compute:

\[
g_s=[V_s,V_s^{2/3},h_s,\bar r_s,r_s^{\max},c_s,A_s]^\top
\in\mathbb{R}^{7},
\qquad
V_s=\delta^3\sum_{\mathbf{x}}O_s(\mathbf{x}),
\]

where \(V\) is above-ground volume, \(h\) height, \(r\) radial spread,
\(c\) compactness, \(A\) projected canopy area, and \(\delta\) voxel size.

**[E] Fold-local fusion.** In each LOOCV fold, training specimens alone fit
the transformations:

\[
\tilde g_s=(g_s-\mu_g)/(\sigma_g+\epsilon),\qquad
\tilde d_s=U_k^\top(d_s-\mu_d),\qquad
x_s=[\tilde g_s;\tilde d_s].
\]

The held-out specimen uses the training fold's \(\mu,\sigma,U_k\), preventing
feature-distribution leakage.

**[F] Ridge biomass head.** For training features \(X\) and masses \(\mathbf y\):

\[
\hat{\boldsymbol\beta}=(X^\top X+\alpha I)^{-1}X^\top\mathbf y,\qquad
\hat y_s=x_s^\top\hat{\boldsymbol\beta}+b.
\]

The reported estimate is \(\hat y_s^{(-s)}\), generated without using specimen
\(s\)'s target during fitting. Performance is summarised by:

\[
\mathrm{RMSE}=\sqrt{\frac1n\sum_s(\hat y_s-y_s)^2}.
\]

The pipeline should also bootstrap specimens for confidence intervals and
write per-specimen residuals. V001, V010, and E020 remain in the primary
analysis; exclusions belong only in a separately labelled sensitivity analysis
after an independent data-quality audit.

The current implementation intentionally does not train a multimodal
Transformer head. A trainable attention/gating stack was tested as a
diagnostic, but its RMSE was 0.442 kg (R² 0.369) on 37 specimens, worse than
the fold-local ridge formulation. The ridge model is therefore the primary
AB-SSViT estimator for this small dataset.

## Conditions

The runner compares:

1. geometry-only control;
2. frozen DINOv2 features;
3. frozen DINOv2 plus **TSDF fused geometry** with fold-local ridge;
4. existing fused-geometry baselines.

The primary result is condition 3. Report RMSE, MAE, MARE, R², bias, bootstrap
confidence intervals, and predictions for every held-out specimen.

The runner also records an enriched-feature sensitivity condition combining
fused geometry, carved geometry, profile, and image descriptors. This is not
automatically promoted to the primary result: the 37-specimen RTX 4080 smoke
test scored RMSE 0.519 kg (R² 0.130), compared with RMSE 0.361 kg (R² 0.581)
for the DINOv2 plus fused-geometry ridge. Feature enrichment must therefore be
kept as a pre-registered ablation and only adopted if it improves repeated
fold-local validation.

## RTX 4070 setup

From the repository root:

```bash
AB-SSViT/setup_rtx4070.sh
conda activate ab_ssvit
AB-SSViT/run_rtx4070.sh
```

The setup keeps the existing `cropcraft` and `ggssvt` environments untouched.
The runner defaults to `cuda`, uses one inference worker, and stores outputs in
`work_dirs/ab_ssvit/`.

The source implementation is shared with `ggssvt`; this folder contains the
experiment contract and reproducible entry point rather than a second copy of
the geometry/model code.

## Current stopping point and visual evidence

The AB-SSViT pilot stopped after frozen DINOv2/DINOv3 feature probes and
plant-level LOOCV. DINOv2 plus fused geometry is the current candidate (RMSE
0.336 kg, R2 0.645, n=38); this is promising but still provisional because
paired same-fold ablations, multi-seed uncertainty, and campaign-held-out
validation are not complete. DINOv2 is a feature extractor, not a 3D decoder,
so it does not itself emit a reconstruction.

Pot-excluded RGB-D reconstruction videos for E011, M001, and V001 are in
`work_dirs/ab_ssvit/videos/pot-excluded-20261012/index.html`. Per-view subject
and depth masks are intersected with each specimen's confidently detected pot
rim **before** RGB-D points are fused. The video shows kept plant pixels in
green, rim-filtered pot/soil pixels in red, and the rotating fused canopy.
These are deterministic AB-SSViT input-pipeline visualizations, not outputs
predicted by DINOv2. Each video has a colored PLY and JSON processing record.
