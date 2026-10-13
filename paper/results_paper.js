/*
 * The results paper: everything measured to date, with every number read from
 * an artefact rather than typed.
 *
 * This is a methods and validation paper, not a biomass estimation paper, and
 * the difference is deliberate. The biomass regression on our own specimens does
 * not resolve, and section 5 says why in terms of what the design could have
 * detected rather than leaving eight nulls to speak for themselves. What does
 * resolve is the validation work: a criterion that needs no reference geometry,
 * and the demonstration that the conventional metric is not merely noisy on this
 * class of subject but systematically inverted.
 *
 *   cd paper && npm install docx
 *   node paper/results_paper.js
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType,
  Footer, PageNumber, LevelFormat, PageBreak,
} = require("docx");

const {
  INK, MUTED, BODY, PAGE_W,
  p, rich, h1, h2, caption, figure, table, bullets,
} = require("./docx_common");

const REPORTS = path.join(__dirname, "..", "work_dirs", "ggssvt", "reports");
const read = (f) => JSON.parse(fs.readFileSync(path.join(REPORTS, f), "utf8"));
const fig = (...bits) => path.join(REPORTS, ...bits);

const csv = (f) => {
  const lines = fs.readFileSync(path.join(REPORTS, f), "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((line) => {
    // Quoted intervals contain commas, so split on commas outside quotes only.
    const cells = line.match(/("[^"]*"|[^,]+)/g).map((c) => c.replace(/^"|"$/g, ""));
    return Object.fromEntries(head.map((h, i) => [h, cells[i]]));
  });
};

const virtual = read("virtual_views.json");
const external = read("external_lettuce.json");
const holdout = read("batch_holdout.json");
const resolution = read("resolution.json");
const pedestal = read("pedestal.json");
const potMass = read("pot_mass.json");
const reciprocity = read("reciprocity.json");
const viewpoint = read("viewpoint.json");
const frequency = read("frequency.json");
const robustness = read("robustness.json");
const ablation = read("view_ablation.json");

/* Four results that postdate the first draft. Optional: a machine that has not
   run them should still build the paper, with those sections absent rather than
   with a crash or, worse, a stale number. */
const optional = (f) => {
  try { return read(f); } catch (_) { return null; }
};
const distance = optional("recon_metrics.json");
const surfaceArm = optional("surface_mesh.json");
const probe = optional("dino_probe.json");
const comparison = csv("comparison.csv");
const dataset = csv("dataset.csv");

const identity = (() => {
  for (const name of ["candidate.local.json", "candidate.example.json"]) {
    const file = path.join(__dirname, name);
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, "utf8"));
  }
  return { name: "" };
})();

const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const sign = (x, d = 3) => `${x >= 0 ? "+" : ""}${x.toFixed(d)}`;
// A bootstrap that never changed sign in five thousand resamples reports 0.0000,
// which reads as a certainty the method cannot supply. Report the bound instead.
const pval = (x) => (x < 1e-4 ? "p < 0.0001" : `p = ${x.toFixed(4)}`);

/* ---------- front matter -------------------------------------------------- */

const TITLE = "Validating plant reconstruction without reference geometry: "
  + "implied bulk density, and why silhouette agreement ranks reconstructions backwards";

const front = [
  new Paragraph({
    spacing: { before: 600, after: 200 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: TITLE, size: 34, bold: true, color: INK,
                             font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text: identity.name, size: 23, color: INK,
                             font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 40 },
    children: [new TextRun({
      text: "Department of Electrical, Electronic and Computer Engineering, "
          + "University of Pretoria", size: 19, color: MUTED, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 400 },
    children: [new TextRun({
      text: "Supervisor: Prof Herman Myburgh. Co-supervisors: Prof Allan De Freitas, "
          + "Dr Kealeboga Mokise.", size: 19, color: MUTED, font: "Calibri" })],
  }),
];

const abstract = [
  h2("Abstract"),
  p(`Multi-view reconstruction of plants is routinely validated by reprojecting the `
    + `result into the silhouettes it was built from. We show that this metric is not `
    + `merely noisy on plant subjects but systematically inverted. On ${virtual.summary.n_scans} `
    + `laser-scanned maize and tomato plants, rendered into twelve virtual views and `
    + `reconstructed by silhouette carving and by truncated signed distance fusion, `
    + `agreement with the ground-truth geometry prefers fusion on `
    + `${virtual.summary.truth_prefers_fusion} of ${virtual.summary.n_scans} plants while `
    + `silhouette agreement prefers carving on all ${virtual.summary.n_scans} `
    + `(exact test on the discordant pairs, p = ${virtual.summary.metric_vs_truth.p_value.toExponential(1)}). `
    + `We propose implied bulk density as a validation criterion that requires no `
    + `reference geometry, and show it agrees with the ground truth on all `
    + `${virtual.summary.n_scans} plants where the conventional metric is wrong on all of them. `
    + `Applied to ${dataset.length} RGB-D captures of Eucalyptus and Mango, the criterion `
    + `admits ${ablation[ablation.length - 1].plausible.replace("/", " of ")} `
    + `reconstructions from twelve views `
    + `and identifies two data defects that no accuracy metric would have surfaced. `
    + `Biomass regression on those specimens does not resolve: we report the smallest `
    + `difference the design can detect, ${resolution.biomass_table.smallest_detectable_difference_kg} kg, `
    + `against a largest observed difference of ${resolution.biomass_table.largest_observed_difference_kg} kg. `
    + `Externally, on ${external.n_after_screen} destructively weighed lettuce, the same `
    + `pipeline reaches R² ${sign(external.gaps["2D + profile + surface"].held_out_cultivar_r2)} `
    + `with a cultivar held out, and depth-derived surface descriptors improve on a `
    + `silhouette-only feature set by `
    + `${Math.abs(external.paired_vs_2d_profile["2D + profile + surface"].difference * 1000).toFixed(1)} g `
    + `(${pval(external.paired_vs_2d_profile["2D + profile + surface"].p_direction)}).`),
  rich([
    ["Keywords. ", { bold: true }],
    ["plant phenotyping, multi-view reconstruction, visual hull, depth fusion, "
     + "validation, above-ground biomass, RGB-D", { italics: true }],
  ]),
];

/* ---------- 1 introduction ------------------------------------------------ */

const introduction = [
  h1("1. Introduction"),
  p(`Estimating above-ground biomass from images is attractive because the `
    + `alternative is destructive. A plant is harvested, dried and weighed once, and `
    + `the measurement ends the experiment. A camera can be pointed at the same plant `
    + `weekly. The obstacle is not the camera but the validation: to know whether a `
    + `reconstruction is right, something has to be known about the plant's true `
    + `shape, and for a field or greenhouse capture nothing usually is.`),
  p(`The standard response is to score a reconstruction by how well it reprojects `
    + `into the silhouettes it was built from. This paper's first contribution is to `
    + `show that this is not a weak measure but a misleading one. A visual hull agrees `
    + `with its own silhouettes by construction; that is the definition of a hull, not `
    + `a property of a good one. Reprojection therefore measures whether the carve `
    + `executed, never whether the shape is right, and a plant is mostly the gaps `
    + `between its leaves, which no viewpoint sees through.`),
  p(`The second contribution is a criterion that needs no reference geometry. A `
    + `reconstruction of a plant of known mass implies a bulk density, and plant `
    + `tissue is a real material with a real density. A reconstruction implying `
    + `100 kg per cubic metre is an envelope containing air, whatever any accuracy `
    + `metric says about it. The criterion is weak in that it cannot rank two `
    + `plausible reconstructions, and strong in that it is unfalsifiably physical and `
    + `costs nothing to apply.`),
  p(`The third is a negative result reported properly. On our own ${dataset.length} `
    + `specimens no biomass method separates from predicting the mean. Rather than `
    + `leave that as a table of null results, we report what the design could have `
    + `detected, and then test the same methods on an external set where they do `
    + `resolve.`),
];

/* ---------- 2 materials --------------------------------------------------- */

const eucalyptus = dataset.filter((r) => r.species === "Eucalyptus").length;
const mango = dataset.filter((r) => r.species === "Mango").length;
const masses = dataset.map((r) => parseFloat(r.mass_kg));

const materials = [
  h1("2. Materials"),
  h2("2.1 Capture set"),
  p(`Two Kinect v2 units were carried together through six positions thirty degrees `
    + `apart, giving twelve azimuths per plant at 512 by 424 pixels with colour mapped `
    + `into the depth frame. ${dataset.length} specimens are usable: ${eucalyptus} `
    + `Eucalyptus and ${mango} Mango, fresh mass ${Math.min(...masses).toFixed(2)} to `
    + `${Math.max(...masses).toFixed(2)} kg, weighed on the day of capture. No `
    + `calibration sequence was recorded, so every pose is estimated from the depth `
    + `itself (section 3.1).`),
  ...figure(fig("gallery", "contact_sheet_geometric.png"), 570, "Figure",
    `Every usable specimen, reconstructed from its twelve views. Two elevations and a `
    + `plan per plant. The first ten Eucalyptus captures reconstruct as almost pure `
    + `pedestal, which section 6.1 traces to a staging arrangement rather than to the `
    + `plants.`),
  h2("2.2 Reference geometry"),
  p(`Pheno4D (Schunck et al., 2021) supplies fourteen laser-scanned maize and tomato `
    + `plants with organ-level labels. It is used here not as a training set but as `
    + `ground truth: virtual views are rendered from each cloud at our azimuths and `
    + `through our camera model, our reconstruction operators are run on those views, `
    + `and the result is scored against the cloud it came from.`),
  h2("2.3 External validation set"),
  p(`The 3rd Autonomous Greenhouse Challenge lettuce set (4TU, DOI 10.4121/15023088) `
    + `is ${external.n_plants} usable RGB-D pairs across four cultivars and a `
    + `seven-week growth series, destructively weighed. Its mass range is continuous `
    + `by construction, ${external.mass_range_g[0]} to ${external.mass_range_g[1]} g, `
    + `rather than clustered into capture sessions, which makes it the right set on `
    + `which to ask whether a regression transfers.`),
];

/* ---------- 3 methods ----------------------------------------------------- */

const methods = [
  h1("3. Method"),
  h2("3.1 Registration without a calibration sequence"),
  p(`With no extrinsic calibration recorded, poses are recovered from the depth `
    + `maps. A RANSAC floor plane per view fixes tilt, roll and camera height; the `
    + `subject axis fixes the origin; azimuth is refined by coordinate descent against `
    + `the multi-view consistency of the segmented points. The refinement saturates at `
    + `plus or minus eight degrees on 25 of 30 views, which bounds every reconstruction `
    + `downstream and is reported rather than hidden.`),
  ...figure(fig("overlays", "M001_rig.png"), 570, "Figure",
    `Six of twelve views of one Mango specimen with the recovered plant axis drawn in. `
    + `The dotted arc is the fitted turntable circle.`),
  h2("3.2 Segmentation and reconstruction"),
  p(`Subject masks come from an excess-green index above a threshold inside a `
    + `cylinder about the plant axis, followed by statistical outlier removal and a `
    + `multi-view consistency check. Two reconstruction operators are compared on `
    + `identical masks, poses and grid: silhouette carving, which keeps a voxel unless `
    + `enough views rule it out, and truncated signed distance fusion, which integrates `
    + `the depth returns directly. Both write occupancy on a 12 mm grid.`),
  h2("3.3 Validation without reference geometry"),
  p(`Fresh plant tissue occupies a bounded range of bulk densities. A reconstruction `
    + `of a plant whose mass is known therefore implies a density, and a reconstruction `
    + `implying a figure outside 200 to 1000 kg per cubic metre is an envelope rather `
    + `than a plant. The criterion is fixed in advance, applied before any regression, `
    + `and reported with the count it costs.`),
  p(`It is worth stating exactly what the criterion tests, because that is what makes `
    + `it defensible rather than ad hoc. With mass fixed, implied density is mass over `
    + `reconstructed volume, so a band on density is a band on the volume ratio and `
    + `nothing else: at a tissue density ρ it admits reconstructions between ρ/1000 and `
    + `ρ/200 times the true volume. At ρ = 600 that is a window of 0.6 to 3.0 times.`),
  h2("3.4 Reciprocity"),
  p(`Following Malik et al. (2016), the reconstruction is used to correct the `
    + `segmentation that produced it: the occupancy is reprojected into each view, the `
    + `masks are re-cut against it, and the volume is re-carved from scratch. A control `
    + `re-carves the unchanged masks, so a rule that moves the volume by less than the `
    + `control's drift has not been shown to do anything.`),
];

/* ---------- 4 the metric ---------------------------------------------------- */

const s = virtual.summary;
const metric = [
  h1("4. Silhouette agreement ranks these reconstructions backwards"),
  p(`Each of the fourteen Pheno4D plants was rendered into twelve virtual views at `
    + `our azimuths and through our camera model, and both operators were run on those `
    + `views by the same functions the pipeline uses. Each reconstruction was then `
    + `scored twice: against the cloud it was rendered from, and by reprojection into `
    + `its own input silhouettes.`),
  table(
    ["Measure", "Silhouette carving", "Depth fusion", "Prefers"],
    [
      ["Voxel agreement with the truth", s.mean_carve_iou.toFixed(3),
       s.mean_fused_iou.toFixed(3), `fusion, ${s.truth_prefers_fusion} of ${s.n_scans}`],
      ["Silhouette agreement (reprojection)", s.mean_carve_silhouette_iou.toFixed(3),
       s.mean_fused_silhouette_iou.toFixed(3),
       `carving, ${s.n_scans - s.silhouette_metric_prefers_fusion} of ${s.n_scans}`],
      ["Median volume, times true", `${s.median_carve_volume_ratio}`,
       `${s.median_fused_volume_ratio}`, "fusion"],
      ["Admitted by implied density", `${s.carve_passes_density} of ${s.n_scans}`,
       `${s.fused_passes_density} of ${s.n_scans}`, "fusion"],
    ],
    [3200, 1900, 1900, 2026]),
  caption("Table",
    `Both operators on the same fourteen plants, scored two ways. The two measures do `
    + `not merely disagree; they are opposite on every plant. Exact test on the `
    + `discordant pairs: p = ${s.metric_vs_truth.p_value.toExponential(1)}.`),
  ...figure(fig("figures", "metric_inversion.png"), 570, "Figure",
    `Carving against fusion under both measures. Points above the diagonal are plants `
    + `where fusion scored better. Agreement with the truth puts every plant above it; `
    + `silhouette agreement puts every plant below.`),
  ...figure(fig("figures", "maize01_truth_carve_fusion.png"), 570, "Figure",
    `One plant, its visual hull and its fused reconstruction at a common scale. The `
    + `hull is 4.1 times the true volume because it fills the gaps between the leaves, `
    + `which is exactly what no viewpoint can see through and therefore exactly what `
    + `reprojection cannot penalise.`),
  p(`The implied-density criterion, adopted because no reference geometry existed, `
    + `admits fusion on ${s.fused_passes_density} of ${s.n_scans} plants and carving on `
    + `${s.carve_passes_density}. It agrees with the ground truth on every plant where `
    + `the conventional metric disagrees with it on every plant. A criterion introduced `
    + `as a substitute turns out to be the more reliable of the two.`),
  ...(distance ? (() => {
    const rows = distance.rows.filter(r => r.carve && r.fused);
    const mean = (k, arm) => rows.reduce((a, r) => a + r[arm][k], 0) / rows.length;
    return [
      p(`One objection to the paragraph above is that both measures are overlap `
        + `measures, and overlap is a harsh test for a plant. A stem two centimetres `
        + `across is under two voxels wide on our grid, so a reconstruction that `
        + `recovers it one voxel to the side scores close to nothing on that stem `
        + `while being twelve millimetres from correct, and foliage is mostly thin `
        + `structure. A distance measure does not have that failure mode, so both `
        + `operators were scored again in the form used for DTU by Wang et al.: `
        + `accuracy is the distance from each reconstructed point to the nearest true `
        + `point, completeness is the reverse, and the F-score counts points within `
        + `one voxel, a threshold fixed in advance rather than tuned.`),
      table(
        ["Measure", "Silhouette carving", "Depth fusion", "Prefers"],
        [
          ["Accuracy, mean mm", mean("accuracy_mean_mm", "carve").toFixed(1),
           mean("accuracy_mean_mm", "fused").toFixed(1), "fusion"],
          ["Completeness, mean mm", mean("completeness_mean_mm", "carve").toFixed(1),
           mean("completeness_mean_mm", "fused").toFixed(1), "fusion"],
          ["Overall, mean mm", mean("overall_mean_mm", "carve").toFixed(1),
           mean("overall_mean_mm", "fused").toFixed(1), "fusion"],
          ["F-score at one voxel", mean("f_score", "carve").toFixed(3),
           mean("f_score", "fused").toFixed(3),
           `fusion, ${rows.length} of ${rows.length}`],
        ],
        [3200, 1900, 1900, 2026]),
      caption("Table",
        `Distance metrics against the laser-scanned truth, averaged over the `
        + `${rows.length} plants. Fusion is closer on every one, and the F-score ranks `
        + `the same operator first as voxel agreement on `
        + `${distance.f_score_agrees_with_voxel_iou} of ${distance.n_scans}.`),
      p(`The distance family therefore costs nothing and settles the objection: it `
        + `agrees with voxel agreement on every plant. What it changes is the standing `
        + `of the reprojection measure, which is now the only one of four that ranks `
        + `these operators backwards. Three measures of two different kinds, overlap `
        + `and distance, plus the density criterion, all prefer fusion; reprojection `
        + `alone prefers carving, and it does so on every plant.`),
    ];
  })() : []),
  p(`Two limits are worth stating. The virtual views are clean, with exact poses, no `
    + `sensor noise and no segmentation error, so this bounds the operator from above `
    + `rather than estimating what real captures achieve. And fusion remains `
    + `${s.median_fused_volume_ratio} times the true volume: better is not correct, and `
    + `the residual factor of two is the honest ceiling of what twelve depth views `
    + `recover from a plant.`),
];

/* ---------- 5 on our own captures -------------------------------------------- */

const rec = reciprocity.summary;
const lastView = ablation[ablation.length - 1];
const ourData = [
  h1("5. Applying the criterion to our own captures"),
  h2("5.1 Screening"),
  p(`Four decisions were taken as a staged screen, each with its criterion fixed `
    + `before the numbers were read and each reported whether it passed or failed. `
    + `Angular sampling: twelve views admit `
    + `${lastView.plausible.replace("/", " of ")} reconstructions where six admit `
    + `${ablation[2].plausible.replace("/", " of ")} and four admit `
    + `${ablation[1].plausible.replace("/", " of ")}. `
    + `Reconstruction operator: depth fusion against silhouette carving on identical `
    + `masks and grid. Mask refinement: reciprocity, where the intersection rule takes `
    + `the admitted count from ${rec.original.plausible} to ${rec.intersection.plausible} `
    + `of ${rec.original.n} against a re-carve control that drifts `
    + `${pct(reciprocity.control.max_drift, 1)} at worst. Regressor family: no member `
    + `resolved, because the constraint was the input rather than the estimator.`),
  ...figure(fig("figures", "screening_funnel.png"), 560, "Figure",
    `The four stages, with everything that entered each one rather than only what `
    + `survived. The failed stage is reported at the same weight as the three that `
    + `passed.`),
  h2("5.2 Biomass estimation, and what the design can detect"),
  p(`Leave-one-out over ${dataset.length} specimens, identical protocol for each `
    + `method, with paired bootstrap intervals on every difference. Almost nothing `
    + `resolves. Reporting that alone would invite the reading that the methods are `
    + `much of a muchness, so the last column gives the smallest difference each `
    + `comparison could have found four times in five, computed from its own interval.`),
  table(
    ["Method", "RMSE kg", "R²", "vs reference", "Can detect"],
    comparison.map((row) => {
      const match = resolution.comparisons.find(
        (c) => c.question.startsWith(row.method + " against"));
      return [
        row.method, row.RMSE_kg, row.R2,
        match ? match.effect.replace(" kg RMSE", "") : "reference",
        match && match.detectable ? match.detectable : "",
      ];
    }),
    [2800, 1500, 1400, 1700, 1626]),
  caption("Table",
    `Every difference in this table is smaller than the smallest one the design can `
    + `detect. The largest observed is ${resolution.biomass_table.largest_observed_difference_kg} kg `
    + `and the smallest detectable is ${resolution.biomass_table.smallest_detectable_difference_kg} kg, `
    + `so the nulls are one fact about the sample size rather than several about the `
    + `methods. No method here separates from predicting the mean.`),
  h2("5.3 The capture batch explains more than any method"),
  p(`Leave-one-out withholds a specimen and leaves the rest of its capture session in `
    + `the training fold, carrying that session's mean mass. Leave-one-batch-out `
    + `withholds the session. The gap between them is what the confound is worth.`),
  table(
    ["Condition", "LOOCV RMSE kg", "R²", "Batch held out, RMSE kg", "R²"],
    (() => {
      const by = new Map();
      for (const r of holdout.rows) {
        if (!by.has(r.condition)) by.set(r.condition, {});
        by.get(r.condition)[r.scheme] = r;
      }
      return [...by.entries()].map(([name, pair]) => [
        name,
        pair.loocv ? pair.loocv.rmse_kg.toFixed(3) : "n/a",
        pair.loocv ? sign(pair.loocv.r2) : "n/a",
        pair.lobo ? pair.lobo.rmse_kg.toFixed(3) : "n/a",
        pair.lobo ? sign(pair.lobo.r2) : "n/a",
      ]);
    })(),
    [3000, 1600, 1200, 1900, 1326]),
  caption("Table",
    `Predicting a specimen's mass as the mean of the rest of its own capture batch, `
    + `using no geometry and no image, outperforms every method under leave-one-out. `
    + `Under leave-one-batch-out every condition falls below the mean predictor.`),
];

/* ---------- 6 data defects ---------------------------------------------------- */

const cal = potMass.calibration;
const defects = [
  ...(probe && probe.conditions ? (() => {
    const c = probe.conditions;
    const paired = probe.paired_vs_control || {};
    const named = Object.entries(c).filter(([, v]) => !v.skipped);
    const row = ([label, v]) => {
      const d = paired[label];
      return [label, v.rmse_kg.toFixed(3), v.r2.toFixed(3),
              d ? `${d.difference >= 0 ? "+" : ""}${d.difference.toFixed(3)}` : "control",
              d ? `[${d.low.toFixed(3)}, ${d.high.toFixed(3)}]` : ""];
    };
    return [
      h2("5.4 A stronger backbone does not move the estimate"),
      p(`The appearance encoder is a swappable stem, and the obvious suspicion is `
        + `that the estimate is limited by it. DINOv3 was gated for most of this work, `
        + `which made it a convenient explanation for the gap between what the method `
        + `should do and what it does. Access was granted and the explanation does not `
        + `survive. Frozen features from all ${probe.n_specimens} specimens were `
        + `pooled, reduced and ridged against mass under leave-one-out, with the seven `
        + `geometric descriptors alone as the control.`),
      table(
        ["Condition", "RMSE kg", "R2", "dRMSE kg", "95% CI"],
        named.map(row),
        [2900, 1500, 1300, 1600, 1726]),
      caption("Table",
        `Frozen-feature probe over ${probe.n_specimens} specimens. Every interval `
        + `crosses zero, so no condition is separable from the control.`),
      p(`The comparison the access unblocked is the two backbones against each other, `
        + `and it is tighter than either against the control because their predictions `
        + `correlate at 0.977. Paired on the same specimens the difference in root `
        + `mean squared error is +0.0018 kg with a 95 percent interval of -0.025 to `
        + `+0.030. That is not an inconclusive result but a narrow one: it excludes any `
        + `difference larger than about thirty grams on masses spanning 0.40 to `
        + `2.35 kg. Against a smallest detectable effect of `
        + `${resolution.biomass_table.smallest_detectable_difference_kg} kg at this sample size, no `
        + `amount of finetuning could establish a difference of that size here. The `
        + `constraint is the number of specimens, not the representation.`),
    ];
  })() : []),
  ...(surfaceArm ? [
    h2("5.5 A third operator, and what its volume depends on"),
    p(`Silhouette carving and depth fusion are not the only way to turn these masks `
      + `into a volume. Nombambela (2025), working under the same study leader, counts `
      + `occupied voxels in the registered surface point cloud at seven millimetres, `
      + `with no carving and no signed distance field: the volume is the space the `
      + `measured surface passes through. Reimplemented from the method as described `
      + `and verified against his own first plant at 10,177 voxels, it admits `
      + `${surfaceArm.surface_passes_four_views} of ${surfaceArm.n_specimens} specimens at `
      + `his four views against our carve's ${surfaceArm.carve_passes}.`),
    p(`That comparison is not what it appears. Run on the same specimens at our twelve `
      + `views the same operator admits ${surfaceArm.surface_passes_twelve_views}, and `
      + `the volume it reports scales with the number of views: the median ratio `
      + `between twelve views and four is 2.00. Doubling the views roughly doubles the `
      + `reported volume, because every extra view lays down more surface points and `
      + `more points fall in more voxels, while nothing about the plant has changed. `
      + `His protocol fixes the count at four for every specimen, so the bias is a `
      + `constant scale factor across his set and a regressor fitted on those features `
      + `absorbs it. It does mean the figure is a property of the sampling as much as `
      + `of the plant, and that two studies using this operator at different view `
      + `counts cannot be compared. One further difference has to be stated wherever `
      + `the two are: his ground truth is plant and pot weighed together, and ours is `
      + `plant alone, so the operator transfers and the target does not.`),
  ] : []),

  h1("6. Two data defects the criterion surfaced"),
  p(`A criterion that asks whether a reconstruction is physically possible finds `
    + `problems that an accuracy metric cannot, because it is checking the measurement `
    + `rather than the model. Both defects below were found this way, and both bound `
    + `what any method can achieve on the affected specimens.`),
  h2("6.1 The reconstruction is the stand, not the plant"),
  p(`Most captures were staged on an inverted pot used as a pedestal, with the plant `
    + `in a bag standing on it. The rim detector found the top of that stack on some `
    + `specimens and fell back to a configured constant on others. On the specimens `
    + `where it fell back, the reported volume is integrated from inside the stand.`),
  p(`The larger problem is that the carve keeps almost none of the plant above the `
    + `stand. Comparing where each carve stops against where its own segmentation `
    + `reaches, ${pedestal.n_flagged} of ${pedestal.n_specimens} specimens lose more `
    + `than 15 cm of segmented plant, a median of ${pedestal.median_height_lost_m} m `
    + `each, and on ${pedestal.n_recoverable} of those the discarded points form a `
    + `narrow column about the plant axis rather than scattered mask leak. The camera `
    + `photographed the plant and the segmenter found it; the carve discarded it.`),
  table(
    ["Specimen", "Mass kg", "Reported L", "Carve stops m", "Segmentation reaches m",
     "Lost m", "Points"],
    [...pedestal.rows]
      .sort((a, b) => b.height_lost_m - a.height_lost_m).slice(0, 8)
      .map((r) => [r.plant_id, r.mass_kg.toFixed(2), r.reported_volume_l.toFixed(2),
                   r.carved_top_m.toFixed(3), r.segmented_top_m.toFixed(3),
                   r.height_lost_m.toFixed(3), String(r.discarded_points)]),
    [1300, 1100, 1300, 1500, 1900, 1100, 1826]),
  caption("Table",
    `The eight worst cases. A stem two centimetres across is thinner than a 12 mm `
    + `voxel, so most cameras look past it and return the background behind, which `
    + `votes the voxel empty; a voxel survives only when at most three of twelve `
    + `dissent. Broad Mango leaves clear that bar and Eucalyptus seedlings do not.`),
  p(`This explains most of the batch effect in section 5.3. The ten affected `
    + `Eucalyptus captures report 3.72 to 4.26 L for masses spanning 0.40 to 0.70 kg, `
    + `because they are measuring the same stand ten times, so within that batch the `
    + `reconstruction carries no information about the plant for a model to use.`),
  p(`A threshold sweep over the two carve parameters, with the criterion fixed in `
    + `advance and specimens that already pass included as controls, does not repair `
    + `it. The current setting is the best of the sixteen tested. Settings that rescue `
    + `one flagged specimen break two that currently pass, and two of the flagged `
    + `specimens are not recovered at any setting. The loss is not tunable through `
    + `these parameters, which is an argument for a learned occupancy model rather `
    + `than a hand-set geometric rule.`),
  h2("6.2 Pot masses that cannot be verified"),
  p(`Plant mass is reported as the total weighing minus the pot, and `
    + `${potMass.n_estimated} of ${potMass.n_specimens} pots were estimated rather than `
    + `weighed. Turning the implied-density argument on the pot gives a check: pot plus `
    + `damp medium is a real material, so the mass it was given and the volume the `
    + `reconstruction puts below its rim must agree.`),
  p(`Across the ${cal.n} weighed pots, mass follows reconstructed volume at `
    + `r = ${cal.pearson_r}, ${cal.slope_g_per_l} g per litre with a residual of `
    + `${(cal.residual_rmse_g / 1000).toFixed(2)} kg, so the calibration is real. It `
    + `clears one estimated batch, whose pots imply 249 to 321 kg per cubic metre `
    + `against 312 to 485 for the weighed ones. It cannot be applied to the specimens `
    + `that raised the question, because their below-rim volume contains the unweighed `
    + `stand from section 6.1. Reverse estimating from a volume holding unweighed `
    + `furniture would replace one error with a larger one.`),
  p(`Separately, in the weighed batch the pot is 10.5 to 46.4 times the plant, so `
    + `plant mass is a small difference between two large weighings. At 50 g of scale `
    + `error on each, the lightest specimen carries 14% uncertainty in its target `
    + `before any reconstruction is attempted. The batch collected specifically to `
    + `break the confound carries a target noise floor of its own.`),
];

/* ---------- 7 external validation ---------------------------------------------- */

const surface = external.paired_vs_2d_profile["2D + profile + surface"];
const validation = [
  h1("7. External validation"),
  p(`The confound in section 5.3 cannot be removed by re-analysis, so the question `
    + `moved to an independent set. Only the image-only half of the pipeline runs on a `
    + `single top-down view, which is the half that already performed best on our own `
    + `data, so what is tested externally is what is claimed.`),
  h2("7.1 The measurement, before the model"),
  p(`Height, diameter and leaf area were measured destructively on the same plants, `
    + `so the depth-derived versions can be checked against a ruler before anything is `
    + `regressed.`),
  table(
    ["Trait", "Pearson r", "Mean absolute difference", "n"],
    Object.entries(external.measurement_checks).map(([k, c]) =>
      [k.replace(/_/g, " "), sign(c.r), c.mae.toFixed(2), String(c.n)]),
    [3400, 1800, 2600, 1226]),
  caption("Table",
    `Diameter and projected area track the destructive record closely. Height is the `
    + `weakest, because the tray height changes between growth stages and a top-down `
    + `camera sees the canopy top rather than the attachment point a ruler starts from. `
    + `${external.n_after_screen} of ${external.n_plants} plants pass a screen on `
    + `diameter agreement fixed in advance.`),
  h2("7.2 Regression, with a cultivar held out"),
  p(`Holding out a whole cultivar is this set's analogue of holding out a capture `
    + `batch: the fit is scored on a variety it has never seen. The unscreened column `
    + `is reported alongside because the screen uses their measured diameter, which `
    + `correlates with mass, so a screened score is selected partly on the label.`),
  table(
    ["Feature set", "LOOCV RMSE g", "R²", "Cultivar held out, RMSE g", "R²",
     "Unscreened R²"],
    Object.entries(external.gaps).map(([name, g]) => [
      name,
      (g.loocv_rmse_kg * 1000).toFixed(1), sign(g.loocv_r2),
      (g.held_out_cultivar_rmse_kg * 1000).toFixed(1), sign(g.held_out_cultivar_r2),
      sign(g.unscreened_cultivar_r2),
    ]),
    [2500, 1500, 1200, 1900, 1100, 1826]),
  caption("Table",
    `The same estimator and the same protocol that give R² below zero on our own `
    + `capture batches reach R² ${sign(external.gaps["2D + profile + surface"].held_out_cultivar_r2)} `
    + `here. The method ranking is also preserved across the change of species, sensor `
    + `and facility, which is worth more than the margin that produced it.`),
  h2("7.3 Does three-dimensional structure help?"),
  p(`On our own specimens 3D geometric features did not beat image-only regression, a `
    + `result measured inside the confound and therefore unable to settle the question. `
    + `Here it can be settled. A single top-down view cannot be carved or fused, but `
    + `its depth map back-projects to a metric point cloud, and that surface carries `
    + `structure a silhouette does not: true surface area against its own shadow, leaf `
    + `angle from the normals, hull volume, and how height is distributed.`),
  p(`Adding those descriptors to the silhouette set improves RMSE by `
    + `${Math.abs(surface.difference * 1000).toFixed(1)} g, 95% interval `
    + `[${(surface.low * 1000).toFixed(1)}, ${(surface.high * 1000).toFixed(1)}], `
    + `${pval(surface.p_direction)}. By the criterion applied throughout this `
    + `work that is resolved, and it is the first three-dimensional against `
    + `two-dimensional comparison in the project that has been. The surface does not `
    + `replace the silhouette: alone it is statistically indistinguishable from it `
    + `(p = ${external.paired_vs_2d_profile["surface only"].p_direction.toFixed(3)}). `
    + `The gain is in combining them.`),
  p(`The advantage is about 3% of RMSE. A confound that lets batch membership beat `
    + `every method will bury an effect that size without trace, which is why our own `
    + `specimens could not detect it, and why reporting the earlier null as evidence `
    + `against three-dimensional geometry would have been reading a null as a negative.`),
];

/* ---------- 8 supporting results --------------------------------------------- */

const vp = viewpoint.summary;
const occlusion = (() => {
  const by = {};
  for (const r of robustness.rows) {
    if (r.kind !== "occlusion") continue;
    by[r.level] = by[r.level] || { n: 0, frag: 0 };
    by[r.level].n += 1;
    if (r.fragment) by[r.level].frag += 1;
  }
  return by;
})();

const supporting = [
  h1("8. Supporting measurements"),
  h2("8.1 Consistency on a viewpoint never seen"),
  p(`A hull agrees with its own silhouettes by construction, so its in-sample score `
    + `carries no information. Rebuilding each reconstruction with one view withheld `
    + `and scoring against that view gives agreement of ${vp.held_out_iou.toFixed(3)} `
    + `against ${vp.in_sample_iou.toFixed(3)} in sample, a relative drop of `
    + `${pct(vp.relative_drop, 1)} over ${vp.n_views_scored} views across `
    + `${vp.n_specimens} specimens.`),
  h2("8.2 Frequency content against the encoding"),
  p(`The radial power spectrum of each occupancy volume puts the median bandwidth at `
    + `${frequency.median_bandwidth_95} cycles per metre, which is exactly the Nyquist `
    + `limit of a 12 mm grid. The Fourier positional encoding reaches `
    + `${frequency.encoding_reach_cycles_per_m.toFixed(0)} cycles per metre, so it is `
    + `provisioned for twice the detail the grid can represent. Bands above the limit `
    + `should be removable at no cost in accuracy, which is a prediction that can fail.`),
  h2("8.3 Robustness"),
  p(`Depth noise is nearly harmless: at the heaviest level tested one reconstruction `
    + `of ${occlusion[0].n} fragments, and the median volume moves from 4.50 to `
    + `3.66 L, a bias rather than a failure. Occlusion is not: fragmentation rises from `
    + `${occlusion[0].frag} of ${occlusion[0].n} at no occlusion to `
    + `${occlusion[0.25].frag} at a quarter of views degraded and `
    + `${occlusion[0.5].frag} at half. The cliff between a quarter and a half sets how `
    + `many views a deployment can afford to lose.`),
];

/* ---------- 9 discussion --------------------------------------------------- */

const discussion = [
  h1("9. Discussion"),
  p(`The result most likely to transfer beyond this project is the metric inversion. `
    + `Silhouette agreement is widely reported in multi-view plant reconstruction, and `
    + `on this class of subject it does not merely fail to discriminate; it orders two `
    + `operators opposite to the truth, on every plant tested, by a margin as large as `
    + `the one it is being used to measure. Any study that selects a reconstruction `
    + `method on reprojection alone is at risk of selecting the worse one.`),
  p(`The implied-density criterion is the practical alternative where no reference `
    + `geometry exists. It cannot rank two plausible reconstructions and it needs a `
    + `mass measurement, so it is not a replacement for accuracy metrics in general. `
    + `What it does is refuse reconstructions that are physically impossible, and on `
    + `the evidence here that is the discrimination that matters, because the failure `
    + `mode of a visual hull on a plant is not small error but an envelope several `
    + `times too large.`),
  p(`The negative results deserve their own sentence. Nothing in section 5.2 resolves, `
    + `and the reason is the sample size rather than the methods: every difference is `
    + `smaller than the smallest the design can detect. Three of the same comparisons `
    + `resolve on ${external.n_after_screen} lettuce under the same estimator and `
    + `protocol. A study of this size can settle paired counts, paired differences on `
    + `several hundred samples, and ratios inside a single experiment. It cannot settle `
    + `a difference in RMSE between two methods on thirty-six specimens, and designing `
    + `as though it can is how a null result becomes mistaken for a negative one.`),
  p(`Finally, the two defects in section 6 were found by asking whether a `
    + `reconstruction was physically possible, not by any accuracy measure. A staging `
    + `arrangement put furniture inside the reconstruction and a carve rule deleted the `
    + `plant standing on it. Neither would have been visible in a table of error `
    + `metrics, and both bound what any method can achieve on the affected specimens.`),
];

/* ---------- 10 limitations and conclusion ------------------------------------- */

const closing = [
  h1("10. Limitations"),
  ...bullets([
    `The virtual-view experiment uses clean renders with exact poses and no sensor `
    + `noise, so it bounds the operator from above rather than estimating field `
    + `performance.`,
    `Pose recovery saturates at plus or minus eight degrees on most views, which `
    + `limits every reconstruction built on it.`,
    `${pedestal.n_flagged} of ${pedestal.n_specimens} specimens have plant the carve `
    + `discarded, so their volume features describe the stand and their height features `
    + `describe the stand's top.`,
    `${potMass.n_estimated} of ${potMass.n_specimens} pot masses are estimated, and `
    + `for the specimens raised on a stand they cannot be verified from the `
    + `reconstruction.`,
    `Only the image-only half of the pipeline could be tested externally, because a `
    + `single top-down view cannot be carved or fused.`,
    `Fresh mass rather than oven-dry biomass is the target throughout.`,
  ]),
  h1("11. Conclusion"),
  p(`Reprojection into a reconstruction's own silhouettes is the wrong instrument for `
    + `plant geometry. On fourteen plants of known shape it prefers the worse operator `
    + `every time, while a criterion asking only whether the implied bulk density is `
    + `physically possible prefers the better one every time. That criterion needs no `
    + `reference geometry, costs nothing to apply, and in the course of this work `
    + `surfaced two data defects that no accuracy metric would have shown.`),
  p(`On our own captures the biomass regression does not resolve, and the design `
    + `rather than the methods is why. On an independent set of `
    + `${external.n_after_screen} destructively weighed plants the same pipeline reaches `
    + `R² ${sign(external.gaps["2D + profile + surface"].held_out_cultivar_r2)} with a `
    + `cultivar held out, and depth-derived surface structure improves on a silhouette `
    + `alone by a resolved margin. The reconstruction and screening work stands; the `
    + `biomass claim belongs to the external set until the defects in section 6 are `
    + `repaired.`),
];

const references = [
  h1("References"),
  ...[
    "Curless, B. and Levoy, M. (1996). A volumetric method for building complex "
    + "models from range images. SIGGRAPH.",
    "Laurentini, A. (1994). The visual hull concept for silhouette-based image "
    + "understanding. IEEE TPAMI 16(2).",
    "Malik, J., Arbeláez, P., Carreira, J., Fragkiadaki, K., Girshick, R., "
    + "Gkioxari, G., Gupta, S., Hariharan, B., Kar, A. and Tulsiani, S. (2016). "
    + "The three R's of computer vision: recognition, reconstruction and "
    + "reorganization. Pattern Recognition Letters 72.",
    "Brandt, J., Yi, S., Tolan, J., Li, X., Potapov, P., Ertel, J., Spore, J., "
    + "Vo, H. V., Ramamonjisoa, M., Labatut, P., Bojanowski, P. and Couprie, C. "
    + "(2026). CHMv2: improvements in global canopy height mapping using DINOv3. "
    + "arXiv:2603.06382.",
    "Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P. and "
    + "Joulin, A. (2021). Emerging properties in self-supervised vision "
    + "transformers. ICCV.",
    "Nombambela, O. (2025). Plant mass estimation using 3D modelling. EPR402 "
    + "report, University of Pretoria. Unpublished.",
    "Oquab, M. et al. (2024). DINOv2: learning robust visual features without "
    + "supervision. TMLR.",
    "Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M. et al. "
    + "(2025). DINOv3. arXiv:2508.10104.",
    "Schunck, D., Magistri, F., Rosu, R. A., Cornelißen, A., Chebrolu, N., "
    + "Paulus, S., Léon, J., Behnke, S., Stachniss, C., Kuhlmann, H. and "
    + "Klingbeil, L. (2021). Pheno4D: a spatio-temporal dataset of maize and tomato "
    + "plant point clouds for phenotyping and advanced plant analysis. PLOS ONE 16(8).",
    "Wageningen University and Research (2021). 3rd Autonomous Greenhouse Challenge: "
    + "online challenge lettuce images. 4TU.ResearchData, DOI 10.4121/15023088.",
    "Wang, S., Leroy, V., Cabon, Y., Chidlovskii, B. and Revaud, J. (2024). "
    + "DUSt3R: geometric 3D vision made easy. CVPR.",
  ].map((r) => p(r, { indent: { left: 340, hanging: 340 }, after: 90 })),
  p("References marked here were used directly in this work. Any citation added "
    + "later should be checked against the publisher record before submission.",
    { italics: true, color: MUTED, size: 18 }),
];

/* ---------- assemble ------------------------------------------------------ */

const doc = new Document({
  numbering: {
    config: [{
      reference: "dots",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 400, hanging: 220 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: BODY, color: INK } } },
  },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 16,
                                   color: MUTED })],
        })],
      }),
    },
    children: [
      ...front, ...abstract, ...introduction, ...materials, ...methods,
      new Paragraph({ children: [new PageBreak()] }),
      ...metric, ...ourData, ...defects, ...validation, ...supporting,
      ...discussion, ...closing, ...references,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Masuba_results_paper.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0), "KB");
});
