/* The progress deck.
 *
 * Generated rather than hand-built, for the reason the paper and the proposals
 * are: the numbers move. Everything quantitative on these slides is read out of
 * `work_dirs/ggssvt/reports` at build time, so a rebuild after a run cannot
 * disagree with the artefacts. Prose that interprets those numbers is written
 * here; figures come from the gallery and the overlays the pipeline already
 * emits.
 *
 *   cd paper && npm install pptxgenjs
 *   node paper/presentation.js
 */

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
/* The candidate's name only. `proposal_common.js` holds the same block but
 * pulls in `docx` to do it, and a deck build should not need the Word toolchain
 * installed. The split is the one that module documents: the personal half
 * lives in the gitignored `candidate.local.json`, the departmental half stays
 * in the open. */
const loadIdentity = () => {
  for (const name of ["candidate.local.json", "candidate.example.json"]) {
    const file = path.join(__dirname, name);
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, "utf8"));
  }
  throw new Error("no candidate.local.json or candidate.example.json in paper/");
};

const CANDIDATE = {
  ...loadIdentity(),
  supervisor: "Prof Herman Myburgh",
  cosupervisors: ["Prof Allan De Freitas", "Dr Kealeboga Mokise"],
};

const R = path.join(__dirname, "..", "work_dirs", "ggssvt", "reports");
const OUT = path.join(__dirname, "Masuba_progress_presentation.pptx");
const DATE = "31 August 2026";

const read = (f) => JSON.parse(fs.readFileSync(path.join(R, f), "utf8"));
const img = (p) => path.join(R, p);

const dino = read("dino_probe.json");
const label = read("label_efficiency.json");
const viewpoint = read("viewpoint.json");
const robust = read("robustness.json");
const freq = read("frequency.json");
const recip = read("reciprocity.json");
const recon = read("reconstruction_quality.json");
const ablation = read("view_ablation.json");
const holdout = read("batch_holdout.json");

// ---------------------------------------------------------------- palette
// Categorical order validated with the dataviz validator:
//   node scripts/validate_palette.js "0E9384,7B3FA8,C4622D,2563C9" --mode light
//   -> all six checks PASS (worst adjacent CVD dE 16.2 deutan, normal-vision 25.9)
const TEAL = "0E9384";
const PURPLE = "7B3FA8";
const AMBER = "C4622D";
const BLUE = "2563C9";
const SERIES = [TEAL, PURPLE, AMBER, BLUE];

const DARK = "2A1B3D";   // deep aubergine, the dark end of the viridis ramp the renders use
const DARK2 = "3E2C55";
const INK = "241C33";
const MUTED = "6B6478";
const CARD = "F3F1F6";
const PAPER = "FFFFFF";
const ONDARK = "EDEAF2";
const ONDARK_MUTED = "B4A8C6";

const HEAD = "Cambria";
const BODY = "Calibri";

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = CANDIDATE.name;
pres.title = "Automated Biomass Estimation Using Self-Supervised Vision Transformers";

const L = 0.7;             // left margin
const RIGHT = 12.63;       // right content edge
const W = RIGHT - L;       // 11.93

let slideNo = 0;

// --------------------------------------------------------------- furniture
function chrome(slide, dark) {
  slideNo += 1;
  slide.addText("GG-SSVT  ·  progress review  ·  " + DATE, {
    x: L, y: 6.95, w: 7.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 9, color: dark ? ONDARK_MUTED : MUTED,
  });
  slide.addText(String(slideNo), {
    x: RIGHT - 1.0, y: 6.95, w: 1.0, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 9, color: dark ? ONDARK_MUTED : MUTED, align: "right",
  });
}

// The repeated motif: a small solid teal disc ahead of every eyebrow line.
function heading(slide, eyebrow, title, opts) {
  const dark = (opts || {}).dark;
  slide.addShape(pres.ShapeType.ellipse, {
    x: L, y: 0.47, w: 0.15, h: 0.15, fill: { color: TEAL },
  });
  slide.addText(eyebrow.toUpperCase(), {
    x: L + 0.26, y: 0.4, w: W - 0.26, h: 0.3, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10.5, bold: true, charSpacing: 1.6,
    color: dark ? TEAL : TEAL,
  });
  slide.addText(title, {
    x: L, y: 0.75, w: (opts && opts.titleW) || W, h: 0.85, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: (opts && opts.titleSize) || 30, bold: true,
    color: dark ? ONDARK : INK, valign: "top",
  });
}

function card(slide, x, y, w, h, dark) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: dark ? DARK2 : CARD },
    line: { color: dark ? DARK2 : CARD, width: 0 },
  });
}

function stat(slide, x, y, w, value, label, opts) {
  const o = opts || {};
  const dark = o.dark;
  const h = o.h || 1.85;
  card(slide, x, y, w, h, dark);
  slide.addText(value, {
    x: x + 0.24, y: y + 0.22, w: w - 0.48, h: 0.78, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: o.size || 34, bold: true, color: o.color || TEAL, valign: "top",
  });
  slide.addText(label, {
    x: x + 0.24, y: y + 1.02, w: w - 0.48, h: h - 1.18, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 11.5, color: dark ? ONDARK_MUTED : MUTED, valign: "top", lineSpacing: 15,
  });
}

function body(slide, x, y, w, h, text, opts) {
  const o = opts || {};
  slide.addText(text, {
    x, y, w, h, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: o.size || 13.5, color: o.color || INK,
    valign: "top", lineSpacing: o.lineSpacing || 19,
  });
}

function bullets(slide, x, y, w, h, items, opts) {
  const o = opts || {};
  const runs = items.map((t, i) => ({
    text: t, options: {
      bullet: true, breakLine: i !== items.length - 1,
      paraSpaceAfter: o.gap === undefined ? 7 : o.gap,
    },
  }));
  slide.addText(runs, {
    x, y, w, h, isTextBox: true, margin: 0, valign: "top",
    fontFace: BODY, fontSize: o.size || 13, color: o.color || INK, lineSpacing: o.lineSpacing || 18,
  });
}

function caption(slide, x, y, w, text) {
  slide.addText(text, {
    x, y, w, h: 0.42, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 10, italic: true, color: MUTED, valign: "top", lineSpacing: 13,
  });
}

const chartFrame = (extra) => Object.assign({
  showTitle: false,
  showLegend: false,
  catAxisLabelColor: MUTED, catAxisLabelFontSize: 11, catAxisLabelFontFace: BODY,
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFontFace: BODY,
  catGridLine: { style: "none" },
  valGridLine: { color: "E6E3EC", size: 1 },
  catAxisLineShow: false, valAxisLineShow: false,
  dataLabelColor: INK, dataLabelFontSize: 11, dataLabelFontFace: BODY, dataLabelFontBold: true,
  chartColors: SERIES,
}, extra || {});

// ================================================================== 1 title
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addShape(pres.ShapeType.ellipse, { x: L, y: 1.62, w: 0.19, h: 0.19, fill: { color: TEAL } });
  s.addText("MEng Computer Engineering  ·  University of Pretoria", {
    x: L + 0.32, y: 1.55, w: 10.0, h: 0.32, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 12, bold: true, charSpacing: 1.6, color: TEAL,
  });
  s.addText("Automated Biomass Estimation\nUsing Self-Supervised Vision Transformers", {
    x: L, y: 2.05, w: 11.4, h: 2.0, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 42, bold: true, color: ONDARK, lineSpacing: 50,
  });
  s.addText("A progress review: every experiment run to date, and what each one settled.", {
    x: L, y: 4.15, w: 10.4, h: 0.4, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 16, color: ONDARK_MUTED,
  });
  s.addText([
    { text: CANDIDATE.name, options: { bold: true, breakLine: true } },
    { text: "Supervisor: " + CANDIDATE.supervisor, options: { breakLine: true } },
    { text: "Co-supervisors: " + CANDIDATE.cosupervisors.join(" · "), options: {} },
  ], {
    x: L, y: 5.15, w: 7.2, h: 1.2, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: ONDARK, lineSpacing: 19,
  });
  s.addText("36 specimens · 12 views each\n302 automated tests · 29-step pipeline", {
    x: 8.6, y: 5.15, w: 4.03, h: 1.2, isTextBox: true, margin: 0,
    fontFace: BODY, fontSize: 13, color: ONDARK_MUTED, align: "right", lineSpacing: 19,
  });
  chrome(s, true);
  s.addNotes("Framing line: the pipeline is built, validated and reproducible on two machines. The learned model is training right now; every result in this deck comes from the geometry pipeline, frozen pretrained features, or classical baselines.");
}

// ============================================================== 2 where it stands
{
  const s = pres.addSlide();
  heading(s, "Status", "Where the work stands");
  const cw = 2.78, gap = 0.27;
  const xs = [L, L + cw + gap, L + 2 * (cw + gap), L + 3 * (cw + gap)];
  stat(s, xs[0], 1.75, cw, "36 / 42", "specimens usable of everything captured. Two dropped by name, four by a failed quality gate.");
  stat(s, xs[1], 1.75, cw, "302", "automated tests passing, on Windows and on the Titan, from the same commit.", { color: PURPLE });
  stat(s, xs[2], 1.75, cw, "29", "pipeline steps that run unattended from one command and record their own gates.", { color: AMBER });
  stat(s, xs[3], 1.75, cw, "7", "training runs in flight tonight, the first time the learned model is fitted at all.", { color: BLUE });

  card(s, L, 3.95, W, 1.95);
  s.addText("The one caveat that governs everything else", {
    x: L + 0.35, y: 4.18, w: W - 0.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 16, bold: true, color: INK,
  });
  body(s, L + 0.35, 4.6, W - 0.7, 1.45,
    "The GG-SSVT model has not yet produced a result. Every number in this deck comes from the geometry pipeline, " +
    "from frozen pretrained features, or from classical baselines. That is not a gap in the work. It is what made the " +
    "campaign worth running, because the pipeline had to be shown correct before a model trained on its output could mean anything.",
    { size: 14, color: INK, lineSpacing: 21 });
  chrome(s, false);
  s.addNotes("Say the caveat before anyone asks for it. The screening work is the contribution so far; the model is the next chapter, not the current one.");
}

// ============================================================== 3 hypotheses
{
  const s = pres.addSlide();
  heading(s, "Scope", "Four hypotheses, and how far each has moved");
  const rows = [
    ["H1", "Self-supervised features beat a CNN stem", "DINOv2 lowers RMSE by 0.066 kg and reaches the baseline's accuracy on a quarter of the labels. The accuracy half is not resolved; the label half is.", "Half resolved", AMBER],
    ["H2", "Geometry grounding survives a new viewpoint", "Measured on 432 held-out views: agreement falls 4.3% relative when the reconstruction is scored against a view it never saw.", "Measured", TEAL],
    ["H3", "The encoding is matched to the geometry's frequency content", "The grid resolves 41.7 cycles/m; the Fourier ladder reaches 83.3. The encoding is two times over-provisioned for the signal present.", "Measured", TEAL],
    ["H4", "Robust to sensor noise and to occlusion", "Noise is nearly harmless. Occlusion is not: at half the views degraded, 33 of 36 reconstructions fragment.", "Two thirds answered", PURPLE],
  ];
  let y = 1.72;
  rows.forEach(([tag, title, text, chip, chipColor]) => {
    card(s, L, y, W, 1.18);
    s.addShape(pres.ShapeType.ellipse, { x: L + 0.28, y: y + 0.3, w: 0.58, h: 0.58, fill: { color: chipColor } });
    s.addText(tag, {
      x: L + 0.28, y: y + 0.42, w: 0.58, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(title, {
      x: L + 1.06, y: y + 0.2, w: 7.5, h: 0.34, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: INK,
    });
    body(s, L + 1.06, y + 0.56, 8.6, 0.56, text, { size: 11.5, color: MUTED, lineSpacing: 15 });
    s.addText(chip, {
      x: RIGHT - 2.5, y: y + 0.44, w: 2.2, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, color: chipColor, align: "right",
    });
    y += 1.32;
  });
  chrome(s, false);
  s.addNotes("H2 and H3 are measured but not yet contested by a trained model. H1's two halves came apart, which is the most interesting thing on this slide.");
}

// ============================================================== 4 capture set
{
  const s = pres.addSlide();
  heading(s, "Data", "Everything that was captured");
  body(s, L, 1.85, 5.7,
    1.7,
    "Two Kinect v2 units carried together through six positions thirty degrees apart, giving twelve azimuths per plant at " +
    "512 by 424, colour mapped into the depth frame. Fresh mass was weighed on the same day as the capture.",
    { size: 13.5, lineSpacing: 19 });
  bullets(s, L, 3.7, 5.7, 2.9, [
    "26 Eucalyptus, 10 Mango",
    "Mass 0.40 kg to 2.35 kg, fresh",
    "Four capture batches; V001–V008 added specifically to break a confound",
    "V011 dropped, X001 excluded by name",
  ], { size: 12.5 });
  s.addImage({ path: img("gallery/contact_sheet_geometric.png"), x: 6.71, y: 1.72, w: 5.92, h: 5.0 });
  caption(s, 6.71, 6.8, 5.92, "Every usable specimen, carved from its twelve views. Two elevations and a plan per plant.");
  chrome(s, false);
  s.addNotes("The contact sheet is the honest picture of the dataset: the E001-E010 batch reconstructs as mostly pot, the V batch as sprawling canopy. That difference is the confound that appears later.");
}

// ============================================================== 5 registration
{
  const s = pres.addSlide();
  heading(s, "Method", "Registration, without a calibration sequence");
  s.addImage({ path: img("overlays/M001_rig.png"), x: L, y: 1.62, w: W, h: 1.65 });
  caption(s, L, 3.32, W, "Six of twelve views of M001, with the recovered plant axis drawn in. The dotted arc is the fitted turntable circle.");
  const cw = (W - 0.5) / 2;
  card(s, L, 3.85, cw, 1.95);
  s.addText("The problem", {
    x: L + 0.3, y: 4.05, w: cw - 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK,
  });
  body(s, L + 0.3, 4.42, cw - 0.6, 1.55,
    "No ChArUco sequence was ever captured, so there is no extrinsic calibration to fall back on. Without poses there is no " +
    "reconstruction, and the dataset would have been unusable as recorded.",
    { size: 12.5, color: MUTED, lineSpacing: 17 });
  card(s, L + cw + 0.5, 3.85, cw, 1.95);
  s.addText("What replaced it", {
    x: L + cw + 0.8, y: 4.05, w: cw - 0.6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK,
  });
  body(s, L + cw + 0.8, 4.42, cw - 0.6, 1.55,
    "A RANSAC floor plane per view fixes tilt, roll and height; the subject axis fixes the origin; azimuth is refined by " +
    "coordinate descent. The refinement saturates at plus or minus eight degrees on 25 of 30 views, which is the honest limit of it.",
    { size: 12.5, color: MUTED, lineSpacing: 17 });
  chrome(s, false);
  s.addNotes("Worth stating plainly that the saturation is a limitation, not a success. It bounds how good any reconstruction downstream can be.");
}

// ============================================================== 6 pipeline
{
  const s = pres.addSlide();
  heading(s, "Method", "From twelve photographs to a volume");
  const steps = [
    ["Segment", "Excess green above a threshold inside a cylinder about the plant axis, then outlier removal and a multi-view consistency check."],
    ["Reconstruct", "Silhouette carving or TSDF depth fusion on a 12 mm grid, keeping only the largest connected component."],
    ["Measure", "Volume integrated above the measured pot rim, with canopy height, surface and spread."],
    ["Screen", "Implied bulk density decides whether the result is a plant or an envelope, before any regression sees it."],
  ];
  let y = 1.78;
  steps.forEach(([t, d], i) => {
    s.addShape(pres.ShapeType.ellipse, { x: L, y: y + 0.02, w: 0.44, h: 0.44, fill: { color: SERIES[i] } });
    s.addText(String(i + 1), {
      x: L, y: y + 0.11, w: 0.44, h: 0.28, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 13, bold: true, color: "FFFFFF", align: "center",
    });
    s.addText(t, {
      x: L + 0.62, y: y, w: 5.3, h: 0.3, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 15, bold: true, color: INK,
    });
    body(s, L + 0.62, y + 0.34, 5.35, 0.85, d, { size: 12, color: MUTED, lineSpacing: 16 });
    y += 1.28;
  });
  s.addImage({ path: img("figures/M001_segmentation_occupancy.png"), x: 6.77, y: 1.7, w: 5.86, h: 4.6 });
  caption(s, 6.77, 6.38, 5.86, "M001: the segmented point cloud it starts from, and the occupancy it ends as.");
  chrome(s, false);
  s.addNotes("Point at step 4. Screening before regression is the design decision the whole project turns on.");
}

// ============================================================== 7 density criterion
{
  const s = pres.addSlide();
  heading(s, "Validation", "Judging a reconstruction with no reference geometry", { titleSize: 28 });
  body(s, L, 1.72, 5.1, 1.5,
    "There is no laser scan to compare against, so correctness cannot be measured directly. What can be measured is whether the " +
    "reconstruction implies a physically possible plant: mass divided by reconstructed volume must land inside the range fresh tissue occupies.",
    { size: 13, lineSpacing: 18 });
  stat(s, L, 3.35, 5.1, "200 – 1000", "kg per cubic metre. Outside that band the reconstruction is an envelope, not a plant, and the number it produces is not a measurement.", { h: 1.9, size: 32 });
  card(s, L, 5.45, 5.1, 1.1);
  body(s, L + 0.28, 5.66, 4.54, 0.75,
    "Applied to twelve views: 8 of 36 land inside. Below twelve views, at most 2 do.",
    { size: 12.5, color: INK, lineSpacing: 17 });

  const cats = ablation.map((a) => a.n_views + " views");
  const vals = ablation.map((a) => Number(String(a.plausible).split("/")[0]));
  s.addChart(pres.ChartType.bar, [{ name: "Physically plausible", labels: cats, values: vals }], chartFrame({
    x: 6.35, y: 1.85, w: 6.28, h: 4.3,
    barDir: "col", barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMaxVal: 10, valAxisMajorUnit: 2,
    valAxisTitle: "specimens inside the density band", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    chartColors: [BLUE, BLUE, BLUE, TEAL],
  }));
  caption(s, 6.35, 6.28, 6.28, "Angular sampling sweep. Usable specimens also rise with view count (23, 25, 34, 36 of 38).");
  chrome(s, false);
  s.addNotes("This criterion is the substitute for ground-truth geometry, and it is what lets every later comparison be decided rather than argued.");
}

// ============================================================== 8 funnel
{
  const s = pres.addSlide();
  heading(s, "Validation", "Four screening stages, decided in advance");
  s.addImage({ path: img("figures/screening_funnel.png"), x: L, y: 1.68, w: 7.55, h: 4.9 });
  body(s, 8.55, 1.72, 4.08, 2.4,
    "Each stage had its criterion written before the numbers arrived, and each is reported at equal weight whether it passed or failed.",
    { size: 13, lineSpacing: 18 });
  bullets(s, 8.55, 3.0, 4.08, 3.4, [
    "Twelve views passed; fewer did not",
    "Depth fusion passed against carving",
    "Intersection refinement passed",
    "No regressor family resolved: the input, not the estimator, was the constraint",
  ], { size: 12, gap: 9 });
  chrome(s, false);
  s.addNotes("Stage 4 failing is reported as prominently as the three that passed. That is the feasibility-study discipline borrowed from the Journal of Voice paper.");
}

// ============================================================== 9 operator
{
  const s = pres.addSlide();
  heading(s, "Finding", "The reconstruction operator decides the result");
  s.addChart(pres.ChartType.bar, [{
    name: "Physically plausible, of 36",
    labels: ["Silhouette carving", "TSDF depth fusion"],
    values: [8, 31],
  }], chartFrame({
    x: L, y: 1.85, w: 6.6, h: 4.15,
    barDir: "col", barGapWidthPct: 90,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMaxVal: 36, valAxisMajorUnit: 9,
    chartColors: [AMBER, TEAL],
  }));
  caption(s, L, 6.12, 6.6, "Same grid, same masks, same criterion. Only the operator changed. Exact McNemar over the 29 discordant specimens: p = 0.000015.");

  stat(s, 7.7, 1.85, 4.93, "0.407  vs  0.219", "mean silhouette IoU, carving against fusion. The standard reprojection metric ranks the worse reconstruction higher.", { h: 1.95, size: 26, color: AMBER });
  card(s, 7.7, 4.0, 4.93, 1.8);
  s.addText("Why that matters", {
    x: 7.98, y: 4.2, w: 4.37, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK,
  });
  body(s, 7.98, 4.57, 4.37, 1.3,
    "A hull agrees with its own silhouettes by construction, so silhouette IoU rewards exactly the failure it should catch. " +
    "Reporting it alone would have inverted the conclusion. The counts are paired, so McNemar settles them at p = 1.5 × 10⁻⁵.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("This is the single most transferable result so far: a widely used reconstruction metric is actively misleading on this class of subject.");
}

// ============================================================== 10 biomass comparison
{
  const s = pres.addSlide();
  heading(s, "Finding", "Biomass estimation, leave-one-out over 36 specimens", { titleSize: 28 });
  const rows = [
    ["Volume allometric", 592],
    ["Predict the mean", 568],
    ["3D geometric features", 544],
    ["Direct 2D", 469],
    ["Fused geometry", 465],
    ["2D + profile", 457],
  ];
  s.addChart(pres.ChartType.bar, [{
    name: "RMSE (grams)",
    labels: rows.map((r) => r[0]),
    values: rows.map((r) => r[1]),
  }], chartFrame({
    x: L, y: 1.8, w: 7.5, h: 4.3,
    barDir: "bar", barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMaxVal: 700, valAxisMajorUnit: 100,
    valAxisTitle: "RMSE, grams", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    chartColors: [AMBER, MUTED, AMBER, BLUE, BLUE, TEAL],
  }));
  caption(s, L, 6.2, 7.5, "Lower is better. Predicting the mean of the training fold is the floor any method has to clear.");
  body(s, 8.55, 1.85, 4.08, 1.5,
    "The 95% intervals overlap almost everywhere. The best method beats the mean by 0.111 kg, on intervals that touch.",
    { size: 13, lineSpacing: 18 });
  stat(s, 8.55, 3.5, 4.08, "0.457 kg", "best RMSE, from an image-only method with a depth profile. 3D geometry does not beat it.", { h: 1.7, size: 30, color: TEAL });
  card(s, 8.55, 5.4, 4.08, 1.15);
  body(s, 8.83, 5.6, 3.52, 0.8,
    "R² = 0.317 at best. Honest, and not yet a biomass estimator.",
    { size: 12, color: INK, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("The 3D pipeline does not currently beat 2D. Saying so first is much stronger than being asked about it.");
}

// ============================================================== 11 batch confound
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  heading(s, "The finding that caps everything", "Batch membership explained more than any method did", { dark: true, titleSize: 28 });
  const cw = 3.75, gap = 0.34;
  const batchOnly = holdout.rows.find(
    (r) => r.condition === "batch membership only" && r.scheme === "loocv");
  stat(s, L, 1.95, cw, "R² = 0.887", "explained by knowing only which capture batch a plant came from, more than mesh geometry managed on the same specimens.", { dark: true, h: 2.0, size: 30, color: AMBER });
  stat(s, L + cw + gap, 1.95, cw, "R² = 0.697", "after V001–V008 was captured to span a continuous mass range. The confound fell, and the 3D advantage fell with it.", { dark: true, h: 2.0, size: 30, color: AMBER });
  stat(s, L + 2 * (cw + gap), 1.95, cw, batchOnly.rmse_kg.toFixed(3) + " kg", "what batch membership alone predicts mass to, using no geometry and no image. It beats every method in this deck.", { dark: true, h: 2.0, size: 30, color: AMBER });
  card(s, L, 4.25, W, 1.8, true);
  s.addText("What can be claimed, and what cannot", {
    x: L + 0.35, y: 4.48, w: W - 0.7, h: 0.32, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 16, bold: true, color: ONDARK,
  });
  body(s, L + 0.35, 4.9, W - 0.7, 1.2,
    "Reconstructed geometry separates plant size classes, tall and sparse against short and solid. That is supported. " +
    "\"Estimates biomass\" is not, and the next slide puts a number on why. There will be no further capture, so the confound is " +
    "met by measuring it and by validating on public data rather than by collecting more.",
    { size: 13, color: ONDARK, lineSpacing: 19 });
  chrome(s, true);
  s.addNotes("Lead with this rather than defending it. It reframes the contribution from a biomass estimator to a validated reconstruction and screening pipeline, which is what the evidence supports.");
}

// ========================================== 12 leave-one-batch-out
{
  const s = pres.addSlide();
  heading(s, "Finding", "What the confound costs the protocol", { titleSize: 29 });

  const row = (condition, scheme) => holdout.rows.find(
    (r) => r.condition === condition && r.scheme === scheme);
  const geo = { loocv: row("geometric, every usable specimen", "loocv"),
                lobo: row("geometric, every usable specimen", "lobo") };
  const dinoRow = { loocv: row("dinov2-base", "loocv"), lobo: row("dinov2-base", "lobo") };

  s.addChart(pres.ChartType.bar, [
    { name: "Leave one specimen out",
      labels: ["Geometric features", "DINOv2 frozen", "Batch label alone"],
      values: [Math.round(geo.loocv.rmse_kg * 1000),
               Math.round(dinoRow.loocv.rmse_kg * 1000),
               Math.round(row("batch membership only", "loocv").rmse_kg * 1000)] },
    { name: "Leave one capture batch out",
      labels: ["Geometric features", "DINOv2 frozen", "Batch label alone"],
      values: [Math.round(geo.lobo.rmse_kg * 1000),
               Math.round(dinoRow.lobo.rmse_kg * 1000), null] },
  ], chartFrame({
    x: L, y: 1.85, w: 7.3, h: 4.15,
    barDir: "col", barGapWidthPct: 55,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMinVal: 0, valAxisMaxVal: 1300, valAxisMajorUnit: 250,
    valAxisTitle: "RMSE, grams", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    showLegend: true, legendPos: "b", legendColor: MUTED,
    legendFontSize: 11, legendFontFace: BODY,
    chartColors: [MUTED, AMBER],
  }));
  caption(s, L, 6.12, 7.3, "Same estimator, same folds-within-folds, only the withholding rule changes. Batch-label-alone has nothing left to predict from once its batch is withheld.");

  stat(s, 8.35, 1.85, 4.28, "351 g", "what knowing only the capture batch achieves. No geometry, no image and no features, yet it beats every real method here.", { h: 1.95, size: 34, color: AMBER });
  card(s, 8.35, 4.0, 4.28, 2.05);
  s.addText("Why both columns, always", {
    x: 8.63, y: 4.2, w: 3.72, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
  });
  body(s, 8.63, 4.58, 3.72, 1.35,
    "Leave-one-out keeps the rest of a plant's capture session in the training fold, carrying that session's mean mass. "
    + "The gap between the two bars is how much of the score came from the session rather than the plant.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("Do not soften this. The honest reading is that the biomass regression is not yet a result, and the reconstruction and screening work is. Every experiment on the surrounding slides is unaffected, because none of them regress against mass.");
}

// ============================================================== 12 H1 backbone
{
  const s = pres.addSlide();
  heading(s, "H1 · accuracy", "Does a self-supervised backbone beat the CNN stem?", { titleSize: 28 });
  const c = dino.conditions;
  const names = ["cnn (no DINO)", "dinov2-base", "dinov2-base + geometry"];
  s.addChart(pres.ChartType.bar, [{
    name: "RMSE (kg)",
    labels: ["CNN stem", "DINOv2 frozen", "DINOv2 + geometry"],
    values: names.map((n) => Math.round(c[n].rmse_kg * 1000)),
  }], chartFrame({
    x: L, y: 1.85, w: 6.9, h: 4.2,
    barDir: "col", barGapWidthPct: 80,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMinVal: 0, valAxisMaxVal: 600, valAxisMajorUnit: 100,
    valAxisTitle: "RMSE, grams", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    chartColors: [AMBER, TEAL, TEAL],
  }));
  caption(s, L, 6.15, 6.9, "Frozen features into a ridge probe, leave-one-out over 36 specimens. Exactly 0.458, 0.392 and 0.392 kg.");
  const p = dino.paired_vs_control["dinov2-base"];
  stat(s, 8.0, 1.85, 4.63, "not resolved", "The paired bootstrap interval on the difference crosses zero, so the ranking is not established at this sample size.", { h: 1.9, size: 28, color: AMBER });
  card(s, 8.0, 3.95, 4.63, 2.1);
  s.addText("The interval, stated exactly", {
    x: 8.28, y: 4.15, w: 4.07, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 15, bold: true, color: INK,
  });
  body(s, 8.28, 4.55, 4.07, 1.4,
    "Difference " + p.difference.toFixed(3) + " kg, 95% interval [" + p.low.toFixed(3) + ", " + p.high.toFixed(3) + "], " +
    "p = " + p.p_direction.toFixed(2) + ". An earlier version of this comparison read as significant; a single leverage point was carrying it.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("Be explicit that this reversed. A 190 litre reconstruction for a 1.9 kg plant was dominating the control, and repairing it moved the verdict from significant to unresolved.");
}

// ============================================================== 13 H1 label efficiency
{
  const s = pres.addSlide();
  heading(s, "H1 · labels", "The half of H1 that is resolved");
  const g = label.curves.find((c) => c.condition.indexOf("geometric") === 0).points;
  const d = label.curves.find((c) => c.condition.indexOf("dinov2") === 0).points;
  const labels = d.map((p) => String(p.n_labels));
  s.addChart(pres.ChartType.line, [
    { name: "Geometric features", labels, values: g.map((p) => (p.unstable ? null : Math.round(p.rmse * 1000))) },
    { name: "DINOv2 frozen", labels, values: d.map((p) => Math.round(p.rmse * 1000)) },
  ], chartFrame({
    x: L, y: 1.85, w: 7.3, h: 4.15,
    showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 11, legendFontFace: BODY,
    lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 8,
    valAxisMaxVal: 1400, valAxisMinVal: 0, valAxisMajorUnit: 200,
    valAxisTitle: "RMSE, grams", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    catAxisTitle: "labelled specimens used for fitting", showCatAxisTitle: true,
    catAxisTitleColor: MUTED, catAxisTitleFontSize: 11, catAxisTitleFontFace: BODY,
    chartColors: [AMBER, TEAL],
  }));
  caption(s, L, 6.12, 7.3, "33 specimens, 8 repeats per point. The geometric curve at 8 labels was unstable (RMSE 128.8 kg) and is off-scale.");
  stat(s, 8.35, 1.85, 4.28, "8  vs  32", "labels needed to reach the geometric baseline's full-data accuracy. Frozen DINOv2 gets there on a quarter of them.", { h: 1.95, size: 32, color: TEAL });
  card(s, 8.35, 4.0, 4.28, 2.05);
  body(s, 8.63, 4.25, 3.72, 1.6,
    "This is the claim that survives the sample size. Label efficiency is a ratio inside one experiment, so it does not " +
    "depend on separating two nearly equal RMSE values, which is exactly where the accuracy half failed.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("Destroying and rebuilding a dataset is expensive; needing a quarter of the labels is the practically useful result even though the accuracy comparison is unresolved.");
}

// ============================================================== 14 H2 viewpoint
{
  const s = pres.addSlide();
  const v = viewpoint.summary;
  heading(s, "H2 · consistency", "Scored against a view it was never shown");
  s.addChart(pres.ChartType.bar, [{
    name: "Silhouette IoU",
    labels: ["Views it was built from", "A view held out"],
    values: [Number((100 * v.in_sample_iou).toFixed(1)), Number((100 * v.held_out_iou).toFixed(1))],
  }], chartFrame({
    x: L, y: 1.85, w: 6.5, h: 4.2,
    barDir: "col", barGapWidthPct: 95,
    showValue: false,
    valAxisMinVal: 0, valAxisMaxVal: 50, valAxisMajorUnit: 10,
    valAxisTitle: "silhouette IoU, percent", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    chartColors: [MUTED, TEAL],
  }));
  caption(s, L, 6.15, 6.5, v.n_views_scored + " views across " + v.n_specimens + " specimens, each rebuilt with one view withheld. IoU " + v.in_sample_iou.toFixed(3) + " against " + v.held_out_iou.toFixed(3) + ".");
  stat(s, 7.6, 1.85, 5.03, (100 * v.relative_drop).toFixed(1) + "%", "relative drop in agreement when the view is one the reconstruction never saw. The absolute gap is " + v.iou_gap.toFixed(4) + " IoU.", { h: 1.95, size: 34, color: TEAL });
  card(s, 7.6, 4.0, 5.03, 2.05);
  s.addText("Why the in-sample number is not evidence", {
    x: 7.88, y: 4.2, w: 4.47, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
  });
  body(s, 7.88, 4.58, 4.47, 1.3,
    "A hull is consistent with its own silhouettes by construction. Only the held-out column carries information, " +
    "which is why the experiment withholds a view rather than reporting agreement with all twelve.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("Small gap is the good outcome here: geometry grounding generalises across viewpoint. The caveat is that this measures the reconstruction, not yet the trained model.");
}

// ============================================================== 15 H3 frequency
{
  const s = pres.addSlide();
  heading(s, "H3 · encoding", "Is the positional encoding matched to the signal?", { titleSize: 28 });
  s.addChart(pres.ChartType.bar, [{
    name: "cycles per metre",
    labels: ["Geometry present\n(median bandwidth)", "Grid Nyquist\n(12 mm voxels)", "Fourier ladder reach\n(10 bands)"],
    values: [freq.median_bandwidth_95, freq.grid_nyquist_cycles_per_m, freq.encoding_reach_cycles_per_m].map(Math.round),
  }], chartFrame({
    x: L, y: 1.85, w: 7.1, h: 4.2,
    barDir: "col", barGapWidthPct: 70,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
    valAxisTitle: "cycles per metre", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: BODY,
    chartColors: [TEAL, BLUE, AMBER],
  }));
  caption(s, L, 6.15, 7.1, "Radial power spectrum per occupancy volume, over 36 specimens. Exactly 41.67, 41.67 and 83.33 cycles per metre.");
  stat(s, 8.2, 1.85, 4.43, "2× over", "The encoding reaches twice as high a frequency as the grid can represent. Half the ladder describes detail that cannot be there.", { h: 1.95, size: 34, color: AMBER });
  card(s, 8.2, 4.0, 4.43, 2.05);
  s.addText("The testable consequence", {
    x: 8.48, y: 4.2, w: 3.87, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
  });
  body(s, 8.48, 4.58, 3.87, 1.3,
    "Bands above the Nyquist limit should be removable at no cost in accuracy. The campaign running tonight sweeps the band " +
    "count to find out, which turns a measurement into a prediction that can fail.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("H3 was nearly dropped on the grounds that frequency grounding was never implemented. That was wrong: the Fourier ladder is exactly the mechanism the proposal names.");
}

// ============================================================== 16 H4 robustness
{
  const s = pres.addSlide();
  heading(s, "H4 · robustness", "Noise is survivable. Occlusion is not.");
  const occ = {};
  robust.rows.forEach((r) => {
    if (r.kind !== "occlusion") return;
    occ[r.level] = occ[r.level] || { n: 0, frag: 0 };
    occ[r.level].n += 1;
    if (r.fragment) occ[r.level].frag += 1;
  });
  const levels = Object.keys(occ).map(Number).sort((a, b) => a - b);
  s.addChart(pres.ChartType.line, [{
    name: "Reconstructions that fragment",
    labels: levels.map((l) => Math.round(l * 100) + "%"),
    values: levels.map((l) => occ[l].frag),
  }], chartFrame({
    x: L, y: 1.85, w: 6.9, h: 4.2,
    lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0",
    valAxisMaxVal: 36, valAxisMajorUnit: 9,
    catAxisTitle: "share of views degraded by occlusion", showCatAxisTitle: true,
    catAxisTitleColor: MUTED, catAxisTitleFontSize: 11, catAxisTitleFontFace: BODY,
    chartColors: [AMBER],
  }));
  caption(s, L, 6.15, 6.9, "A reconstruction counts as fragmented when the largest connected component holds under half the occupied voxels.");
  stat(s, 8.0, 1.85, 4.63, "33 of 36", "reconstructions fragment once half the views are occluded. Below a quarter, most survive.", { h: 1.75, size: 32, color: AMBER });
  stat(s, 8.0, 3.8, 4.63, "1 of 36", "fragments under the heaviest depth noise tested. Median volume moves 4.50 L to 3.66 L, a bias rather than a failure.", { h: 1.75, size: 32, color: TEAL });
  card(s, 8.0, 5.75, 4.63, 0.8);
  body(s, 8.28, 5.92, 4.07, 0.5, "Two thirds of H4 answered. Lighting variation is the untested third.", { size: 12, color: INK, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("The occlusion cliff between 25 and 50 percent is the operationally useful number: it sets how many views a field deployment can afford to lose.");
}

// ============================================================== 17 reciprocity
{
  const s = pres.addSlide();
  const sum = recip.summary;
  heading(s, "Method", "Closing Malik's loop: the reconstruction refines the segmentation", { titleSize: 24 });
  const order = ["original", "union", "intersection", "reconstruction_only"];
  const nice = { original: "Original masks", union: "Union", intersection: "Intersection", reconstruction_only: "Reconstruction only" };
  s.addChart(pres.ChartType.bar, [{
    name: "Physically plausible, of 36",
    labels: order.map((k) => nice[k]),
    values: order.map((k) => sum[k].plausible),
  }], chartFrame({
    x: L, y: 1.85, w: 7.0, h: 4.2,
    barDir: "col", barGapWidthPct: 65,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisMaxVal: 24, valAxisMajorUnit: 6,
    chartColors: [MUTED, AMBER, TEAL, BLUE],
  }));
  caption(s, L, 6.15, 7.0, "Each rule re-projects the reconstruction back into the images and re-cuts the masks, then re-carves from scratch.");
  stat(s, 8.1, 1.85, 4.53, "8  →  19", "specimens brought inside the density band by letting the reconstruction correct the segmentation that produced it.", { h: 1.85, size: 34, color: TEAL });
  card(s, 8.1, 3.9, 4.53, 2.15);
  s.addText("The control that makes it a result", {
    x: 8.38, y: 4.1, w: 3.97, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
  });
  body(s, 8.38, 4.5, 3.97, 1.4,
    "Re-carving unchanged masks moves the volume by " + (100 * recip.control.median_drift).toFixed(2) + "% at the median and " +
    (100 * recip.control.max_drift).toFixed(1) + "% at worst. Any rule that moves it less than that has not been shown to do anything.",
    { size: 12, color: MUTED, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("This is reciprocity in Malik's three-Rs sense: recognition and reconstruction each improving the other. The control was added after an early version of this experiment produced a false positive.");
}

// ============================================================== 18 engineering
{
  const s = pres.addSlide();
  heading(s, "Infrastructure", "What makes any of this repeatable");
  const items = [
    ["302 tests, two machines", "The same commit passes on Windows and on the Titan. Regressions surface as failures, not as puzzling numbers weeks later."],
    ["One command, 29 steps", "The whole programme runs unattended, records which gates each step passed, and skips what already finished."],
    ["Preflight before compute", "Torch and CUDA, ground truth integrity, dependencies, disk. An eight-hour run should not fail on a missing file."],
    ["Seeded end to end", "The seed is written into every checkpoint alongside the run config, because a result that cannot be reproduced cannot be defended."],
    ["Configurable visualisation", "Point clouds, meshes, reconstructions, segmentations and overlays, with layers, views and resolution selectable, including straight into a terminal over SSH."],
    ["Campaign runner", "Per-run results and completion markers, so a job that dies at 3am restarts where it stopped rather than from the beginning."],
  ];
  const cw = (W - 0.4) / 2;
  items.forEach(([t, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = L + col * (cw + 0.4);
    const y = 1.95 + row * 1.6;
    s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.04, w: 0.36, h: 0.36, fill: { color: SERIES[i % 4] } });
    s.addText(t, {
      x: x + 0.52, y, w: cw - 0.52, h: 0.32, isTextBox: true, margin: 0,
      fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
    });
    body(s, x + 0.52, y + 0.38, cw - 0.52, 1.15, d, { size: 11.5, color: MUTED, lineSpacing: 15 });
  });
  chrome(s, false);
  s.addNotes("Worth a slide because it is the difference between a set of scripts and a result an examiner can re-run.");
}

// ============================================================== 19 running now
{
  const s = pres.addSlide();
  heading(s, "In flight", "What is training on the Titan tonight");
  body(s, L, 1.75, 7.2, 0.9,
    "Seven runs, ordered so that the ones answering a hypothesis finish first. Whatever survives an interrupted night is the part the write-up needs.",
    { size: 13.5, lineSpacing: 19 });
  bullets(s, L, 2.6, 7.2, 3.7, [
    "baseline_cnn: the reference condition for every comparison below it",
    "baseline_fused: does a trained model inherit the fusion's advantage?",
    "h2_no_geometry: the geometry-grounding ablation",
    "h1_dinov2: the backbone comparison, this time end to end rather than as a frozen probe",
    "h3_fourier_*: the band sweep that tests the 2× over-provisioning prediction",
  ], { size: 12.5, gap: 8 });
  stat(s, 8.3, 1.75, 4.33, "1.11 / 15.57", "GiB reserved on the card at peak. Every out-of-memory failure this week was two jobs colliding, not a real limit.", { h: 1.95, size: 28, color: TEAL });
  card(s, 8.3, 3.9, 4.33, 2.45);
  s.addText("Three bugs found before the run", {
    x: 8.58, y: 4.1, w: 3.77, h: 0.3, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 14.5, bold: true, color: INK,
  });
  bullets(s, 8.58, 4.5, 3.77, 1.7, [
    "Inference built a graph it never used, 14 GiB of retained activations",
    "The pretrained model stayed on the card through all 38 folds",
    "A five-minute smoke test wrote the marker that says the eight-hour run is done",
  ], { size: 10.5, color: MUTED, gap: 5, lineSpacing: 13 });
  chrome(s, false);
  s.addNotes("The third bug is the dangerous one: it would have reported success without training anything.");
}

// ============================================================== 20 scope
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  heading(s, "Honest scope", "What the evidence supports, and what it does not", { dark: true, titleSize: 28 });
  const cw = (W - 0.45) / 2;
  card(s, L, 1.95, cw, 4.35, true);
  s.addText("Supported today", {
    x: L + 0.35, y: 2.18, w: cw - 0.7, h: 0.34, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: TEAL,
  });
  bullets(s, L + 0.35, 2.68, cw - 0.7, 3.4, [
    "Poses can be recovered from depth alone when no calibration sequence exists",
    "Implied bulk density screens reconstructions with no reference geometry",
    "Depth fusion beats silhouette carving, 31 against 8 of 36, paired McNemar p = 0.000015",
    "Silhouette IoU ranks these reconstructions backwards",
    "Reconstruction refining segmentation more than doubles the plausible count",
    "Frozen DINOv2 features reach the baseline's accuracy on a quarter of the labels",
    "Reconstructed geometry separates plant size classes, with the leave-one-batch-out gap quoted",
  ], { size: 11.5, color: ONDARK, gap: 7, lineSpacing: 15 });
  card(s, L + cw + 0.45, 1.95, cw, 4.35, true);
  s.addText("Not supported yet", {
    x: L + cw + 0.8, y: 2.18, w: cw - 0.7, h: 0.34, isTextBox: true, margin: 0,
    fontFace: HEAD, fontSize: 17, bold: true, color: AMBER,
  });
  bullets(s, L + cw + 0.8, 2.68, cw - 0.7, 3.4, [
    "That the pipeline estimates biomass: batch label alone beats every method under leave-one-out",
    "That a self-supervised backbone is more accurate; the interval crosses zero",
    "That 3D geometry beats image-only regression on this data",
    "Anything at all about the trained GG-SSVT model, which has produced no result yet",
    "Robustness to lighting variation, the untested third of H4",
    "Leaf area, which a hull's surface cannot represent",
  ], { size: 11.5, color: ONDARK, gap: 7, lineSpacing: 15 });
  chrome(s, true);
  s.addNotes("Presenting both columns at equal weight is the point. The left column is a real contribution even though the right column is long.");
}

// ============================================================== 21 next
{
  const s = pres.addSlide();
  heading(s, "Next", "What happens after tonight");
  const steps = [
    ["Tonight", "Seven training runs complete on the Titan", "The first learned results the project has ever had, at last."],
    ["Monday", "Paired bootstraps on every campaign comparison", "Numbers propagate into the findings record, the results section and both proposals."],
    ["Then", "Twelve virtual views from a Pheno4D cloud", "Run the whole pipeline on rendered views of a known laser cloud, and the density screen is calibrated against ground truth at last."],
    ["Beyond", "External validation on two public sets", "Pheno4D supplies the reference geometry this project has never had; 388 lettuce with destructive fresh mass supply the labels ours cannot."],
  ];
  let y = 1.85;
  steps.forEach(([when, what, why], i) => {
    s.addShape(pres.ShapeType.ellipse, { x: L, y: y + 0.12, w: 0.2, h: 0.2, fill: { color: SERIES[i] } });
    s.addText(when, {
      x: L + 0.36, y: y, w: 1.5, h: 0.32, isTextBox: true, margin: 0,
      fontFace: BODY, fontSize: 11.5, bold: true, charSpacing: 1.2, color: SERIES[i],
    });
    s.addText(what, {
      x: L + 1.95, y: y - 0.04, w: 5.05, h: 0.78, isTextBox: true, margin: 0, valign: "top",
      fontFace: HEAD, fontSize: 14.5, bold: true, color: INK, lineSpacing: 19,
    });
    body(s, 7.5, y, RIGHT - 7.5, 1.0, why, { size: 12, color: MUTED, lineSpacing: 16 });
    y += 0.99;
  });
  card(s, L, 5.92, W, 0.72);
  body(s, L + 0.32, 6.1, W - 0.64,
    0.45,
    "No further capture. The 36 specimens are the method-development set; public data becomes the external-validation set.",
    { size: 12.5, color: INK, lineSpacing: 16 });
  chrome(s, false);
  s.addNotes("Close on the durable contribution rather than on the pending model, because the pipeline is what stands whatever the campaign returns.");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
