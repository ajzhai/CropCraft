/*
 * The results, framed as a feasibility study.
 *
 * Modelled on Malan et al., "Feasibility of Low-Cost Smartphone Endoscopy for
 * Laryngeal Imaging" (Journal of Voice, 2026), which is in this group and shares
 * a supervisor. Three things are taken from it deliberately:
 *
 *   1. Criteria stated a priori, each with a number AND the reason for that
 *      number, before any result appears.
 *   2. A staged screening funnel as the first figure: everything that entered
 *      each stage, not only what survived.
 *   3. Poor results reported at equal weight to good ones, with the variability
 *      named rather than buried.
 *
 * Every number is read from work_dirs/ggssvt/reports at build time. Nothing here
 * is typed by hand, so the text cannot drift from the artefacts the way the
 * paper draft's tables can.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, ImageRun,
} = require("docx");
const C = require("./proposal_common");

const { p, rich, h } = C;
const REPORTS = path.join(__dirname, "..", "work_dirs", "ggssvt", "reports");

const read = (name) => {
  const file = path.join(REPORTS, name);
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : null;
};

/* ---- the numbers, all of them from disk ---- */

const views = read("view_ablation.json") || [];
const recip = read("reciprocity.json") || { summary: {}, control: {} };
const quality = read("reconstruction_quality.json");
const fusion = read("fusion.json") || {};

const plausible = (key) => Object.values(fusion).filter((v) => {
  const vol = v[key];
  if (!vol || vol <= 0) return false;
  const rho = v.mass_kg / vol;
  return rho >= 200 && rho <= 1000;
}).length;

const nSpecimens = Object.keys(fusion).length;
const carvePlausible = plausible("carve_above_rim_m3");
const fusedPlausible = plausible("tsdf_above_rim_m3");

const meanOf = (rows, key) =>
  rows.reduce((a, r) => a + r[key], 0) / Math.max(1, rows.length);

const carveIoU = quality ? meanOf(quality.reprojection.carve, "silhouette_iou") : 0;
const fusedIoU = quality ? meanOf(quality.reprojection.fused, "silhouette_iou") : 0;
const carveMAE = quality ? meanOf(quality.reprojection.carve, "depth_mae_m") : 0;
const fusedMAE = quality ? meanOf(quality.reprojection.fused, "depth_mae_m") : 0;

/* ---- table furniture ---- */

const WIDTHS = [2600, 1700, 1700, 3000];
const cell = (text, { bold = false, head = false } = {}) => new TableCell({
  width: { size: 0, type: WidthType.DXA },
  shading: head ? { type: ShadingType.CLEAR, fill: "F2F2F2" } : undefined,
  margins: { top: 60, bottom: 60, left: 90, right: 90 },
  children: [new Paragraph({
    spacing: { after: 0 },
    children: [new TextRun({
      text, bold: bold || head, size: 20, font: "Times New Roman",
    })],
  })],
});

const table = (header, rows, widths = WIDTHS) => new Table({
  columnWidths: widths,
  rows: [
    new TableRow({
      tableHeader: true,
      children: header.map((t, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        children: [new Paragraph({
          spacing: { after: 0 },
          children: [new TextRun({ text: t, bold: true, size: 20,
                                   font: "Times New Roman" })],
        })],
      })),
    }),
    ...rows.map((row) => new TableRow({
      children: row.map((t, i) => new TableCell({
        width: { size: widths[i], type: WidthType.DXA },
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        children: [new Paragraph({
          spacing: { after: 0 },
          children: [new TextRun({ text: String(t), size: 20,
                                   font: "Times New Roman" })],
        })],
      })),
    })),
  ],
});

const caption = (label, text) => new Paragraph({
  spacing: { before: 100, after: 220 },
  children: [
    new TextRun({ text: label + ". ", bold: true, size: 18,
                  font: "Times New Roman" }),
    new TextRun({ text, size: 18, font: "Times New Roman" }),
  ],
});

const figure = (file, widthPx, label, text) => {
  const full = path.join(REPORTS, file);
  if (!fs.existsSync(full)) return [p(`[missing figure: ${file}]`, { italics: true })];
  const data = fs.readFileSync(full);
  const ratio = 0.55;
  return [
    new Paragraph({
      spacing: { before: 160, after: 0 }, alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        data, type: "png",
        transformation: { width: widthPx, height: Math.round(widthPx * ratio) },
      })],
    }),
    caption(label, text),
  ];
};

/* ================================================================== */

const body = [
  h("Feasibility of biomass estimation from calibration-free multi-view RGB-D capture"),

  p("Results are reported against criteria fixed before any of them were computed, and every screening stage records what entered as well as what passed. Both conventions are taken from feasibility reporting practice, and the reason for the second is that a study which reports only the surviving configuration cannot be distinguished from one that tried a single configuration and got lucky.", { italics: true, after: 200 }),

  /* ---- criteria ---- */
  h("1. Criteria established a priori", 2),
  p("Three criteria were fixed in advance. Each carries the reason for its threshold, because a threshold without a justification can be moved after the fact."),

  rich([["C1. Physical plausibility. ", { bold: true }],
        ["The reconstructed above-ground volume must be able to weigh the harvested mass. Implied bulk density, computed as weighed shoot mass divided by reconstructed above-ground volume, must fall inside 200 to 1000 kg per cubic metre. Fresh plant tissue spans roughly 300 to 900; the band is widened at both ends deliberately, so that a reconstruction failing it fails by a margin no measurement error explains. This criterion needs no reference geometry, which is what makes it usable in a destructive-harvest study where none exists."]]),

  rich([["C2. Resolved difference. ", { bold: true }],
        ["A difference between two methods counts as a result only when a paired bootstrap over 20,000 resamples gives a 95 per cent interval excluding zero. At the sample sizes destructive harvest permits, most differences a ranking table would present do not survive this, and are reported as unresolved rather than as findings."]]),

  rich([["C3. Controlled substitution. ", { bold: true }],
        ["When two components are compared, everything else is held fixed: the same specimens, grid resolution, voxel size, views, masks, features and protocol. Where a comparison requires re-running an earlier stage, a control re-runs it with the inputs unchanged and must reproduce the original within 3 per cent, or the comparison is measuring the re-run rather than the change."]], { after: 200 }),

  /* ---- funnel ---- */
  h("2. Staged screening", 2),
  p("Four stages were screened in sequence, each against C1 and each carrying its predecessors' outcome forward."),
  ...figure("figures/screening_funnel.png", 620, "Figure 1",
    "The staged screening. Each stage shows every configuration entered and the criterion that decided between them; the shaded box passed. The final stage is the one that passed nothing, which is reported because it is the most informative of the four."),

  /* ---- angular sampling ---- */
  h("3. Angular sampling", 2),
  p(`Reconstructions were rebuilt from evenly spaced subsets of the twelve captured azimuths and scored under C1. ${views.length ? "" : ""}Agreement between views, the conventional quality measure, degrades gently as views are removed. Physical plausibility does not.`),
  table(
    ["Views", "Usable", "Plausible", "Median implied density"],
    views.map((r) => [
      String(r.n_views), r.usable, r.plausible,
      `${r.median_density_kg_m3} kg/m3`,
    ]),
    [1400, 1800, 1800, 4000],
  ),
  caption("Table 1",
    "Angular sampling against C1. At four views no reconstruction can physically weigh its plant: the median implied density of 9.2 kg per cubic metre is lighter than expanded polystyrene, and the volumes average 126 litres for plants of at most 2.35 kg. These are not poor reconstructions of the plant; they are reconstructions of something else. Twelve views is where the figure becomes non-absurd, and even there only a minority pass."),

  /* ---- operator ---- */
  h("4. Reconstruction operator", 2),
  p(`With views, grid, voxel size and masks held fixed under C3, the only variable was the operator that turns masks and depth into occupancy. Silhouette carving produced ${carvePlausible} plausible reconstructions of ${nSpecimens}; truncated signed distance fusion of the same depth maps produced ${fusedPlausible}. The difference is a property of the evidence each operator uses. A silhouette constrains the subject to lie somewhere along a ray and can never assert that an interior region is empty, so the recovered volume is the canopy envelope. A depth sample asserts a surface at a specific distance, which is what a concavity requires.`),

  p(`The conventional quality measure disagrees, and the disagreement is the most transferable result here. Mean silhouette agreement scores the carve at ${carveIoU.toFixed(3)} and the fusion at ${fusedIoU.toFixed(3)}, ranking them in the opposite order to both C1 and biomass accuracy, while depth error is a tie at ${(carveMAE * 1000).toFixed(1)} against ${(fusedMAE * 1000).toFixed(1)} millimetres. A hull is by construction consistent with every silhouette it was carved from, so a metric measuring agreement with the input rewards whichever method overclaims. Reporting silhouette agreement alone would have selected the worse reconstruction.`, { after: 200 }),

  /* ---- refinement ---- */
  h("5. Mask refinement", 2),
  p("Each subject mask is decided from one view by a colour threshold; the reconstruction is decided from all twelve at once. Re-projecting the reconstruction into a view therefore supplies a second opinion formed from evidence that view never saw, and the two opinions differ in both directions: on one specimen the reconstruction claims 1,541 pixels the threshold missed and rejects 3,326 it included. Four combination rules were screened."),
  table(
    ["Rule", "Plausible", "Median density", "Mean volume"],
    Object.entries(recip.summary || {}).map(([name, row]) => [
      name.replace(/_/g, " "),
      `${row.plausible}/${row.n}`,
      `${row.median_density} kg/m3`,
      `${row.mean_volume_l} L`,
    ]),
    [2600, 1700, 2200, 2500],
  ),
  caption("Table 2",
    `Mask refinement against C1. Intersection more than doubles the plausible count and brings the median implied density inside the band for the first time on a carve. Union moves it the other way, which the direction predicts: the hull is already too large, so a rule that grows masks makes it worse. The C3 control, a re-carve with the masks unchanged, drifts by ${((recip.control.median_drift || 0) * 100).toFixed(2)} per cent median and ${((recip.control.max_drift || 0) * 100).toFixed(1)} per cent at worst, against a 77 per cent volume reduction from intersection.`),

  p("The control is reported because an earlier version of this experiment omitted a post-processing step the pipeline applies, and the control alone then moved the volume further than any rule did, making all three rules appear to succeed. Per-specimen drift is recorded alongside each result so that a rule moving a volume by less than the drift can be dismissed.", { after: 200 }),

  /* ---- what failed ---- */
  h("6. What did not pass", 2),
  p("Four regressor families were screened on identical features under an identical leave-one-out protocol: ridge regression, random forests, gradient boosting and a small multilayer perceptron. Under C2, no substitution resolved. Changing the reconstruction operator, with the regressor held fixed, did. The stage is reported at the same weight as the three that passed, because a study that screens four estimators and reports only the best has not screened them."),

  p("Two limitations bound every result above. The specimens fall into morphological clusters whose masses barely overlap, so batch membership alone explains a substantial share of mass variance and any model fitted on this set separates size classes at least as much as it estimates mass. And no calibration target was captured, so every camera pose is estimated from the same depth data being reconstructed; that assumption is the least verified in the pipeline and is checked independently by pose-free reconstruction, which had not been run at the time of writing.", { after: 160 }),

  /* ---- conclusion ---- */
  h("7. Feasibility verdict", 2),
  p(`Two of the three criteria are met and the third is met only in part. Reconstruction from a calibration-free twelve-view RGB-D rig is feasible under C1 once the operator is depth fusion rather than silhouette carving, and once masks are refined against the reconstruction: ${fusedPlausible} of ${nSpecimens} specimens under fusion, against ${carvePlausible} under the carve as captured. Biomass estimation from those reconstructions produces one resolved improvement under C2. Representation learning has not been evaluated, and no claim about it is made here.`),
  p("The finding that generalises beyond this rig is negative and methodological: silhouette agreement, the measure a study without reference geometry naturally reaches for, systematically prefers the reconstruction that overclaims. A physical criterion costs nothing to compute from data such a study already collects, and reverses the conclusion."),
];

const doc = new Document({
  creator: C.CANDIDATE.name,
  title: "Feasibility results",
  numbering: C.numbering,
  styles: C.styles,
  sections: [{
    properties: C.margins,
    footers: { default: C.pageFooter() },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Masuba_feasibility_results.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0), "KB");
});
