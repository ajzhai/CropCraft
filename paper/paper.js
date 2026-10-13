/*
 * A draft paper reporting the results obtained so far.
 *
 * Every number is taken from a measured artefact in work_dirs, and every figure
 * is one the pipeline generates. Where a result is not statistically resolved
 * the text says so rather than reporting the point estimate alone, because that
 * is the mistake this project already made once and had to withdraw.
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, Footer, PageNumber, LevelFormat, TabStopType,
} = require("docx");

const REPO = "C:/Users/user/CropCraft/work_dirs/ggssvt";
const FIG = path.join(__dirname, "figures");
const ARCH = path.join(REPO, "reports", "architecture");

const {
  INK, MUTED, ACCENT, HEAD, RULE, TINT, BODY, PAGE_W,
  p, rich, h1, h2, caption, figure, pngSize, table, bullets,
} = require("./docx_common");

/* ---------- content ------------------------------------------------------ */

const TITLE = "Space Carving Is the Wrong Instrument: Depth Fusion for Above-Ground Biomass Estimation from Twelve-View RGB-D Capture";

const front = [
  new Paragraph({
    spacing: { after: 140 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: TITLE, size: 34, bold: true, color: HEAD, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Aaron Masuba", size: 23, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "Supervisor: Prof. Herman Myburgh    Co-supervisors: Prof. Allan De Freitas, Dr Kealeboga Mokise",
      size: 18, color: MUTED, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 40 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: "Smart Sensing and Intelligent Systems Group, Department of Electrical, Electronic and Computer Engineering",
      size: 18, color: MUTED, font: "Calibri" })],
  }),
  new Paragraph({
    spacing: { after: 260 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "University of Pretoria", size: 18, color: MUTED, font: "Calibri" })],
  }),
];

const abstract = [
  h2("Abstract"),
  p("Non-destructive above-ground biomass estimation from multi-view imaging usually proceeds by reconstructing the plant and regressing mass on the reconstruction's geometry. We report that for potted Eucalyptus and Mango the reconstruction step, not the regressor, is the limiting component, and that the standard choice of reconstruction operator is unsound for these morphologies."),
  p("Thirty-eight specimens were captured with two Kinect v2 units carried through six positions, giving twelve registered RGB-D views each; no calibration target was recorded, so all camera poses are estimated from depth. We introduce a physical plausibility diagnostic: dividing weighed shoot mass by reconstructed above-ground volume yields an implied bulk density that can be checked against the 300 to 900 kg per cubic metre of fresh plant tissue. Under space carving, only 8 of 36 usable specimens fall inside a deliberately generous 200 to 1000 band, and all ten Mango specimens imply 26 to 77. The visual hull is the maximal solid consistent with the silhouettes, so this is a property of the operator rather than of resolution."),
  p("Replacing silhouette intersection with truncated signed distance fusion of the same depth maps, holding features, protocol, grid, views and masks fixed, raises the count to 25 of 36 at the identical 12 mm grid and 31 of 36 at the 6 mm the sensor supports. Biomass root-mean-square error falls from 0.544 to 0.335 kg and the coefficient of determination rises from +0.030 to +0.632, a paired-bootstrap difference of -0.209 kg with a 95 per cent interval of -0.363 to -0.066. An image-only control is unchanged at 0.469 kg. Two allometric laws that sat below the mean-predictor floor on the hull clear it on the fusion."),
  p("We also report a view-count ablation in which no four-view reconstruction is physically capable of weighing its plant, a systematic 10.9 per cent bias in estimated pot masses recovered by direct weighing, and a batch confound that a purpose-collected batch reduced from R-squared 0.887 to 0.697. A previously reported advantage of reconstruction over direct image regression is withdrawn as never statistically resolved.", { after: 220 }),
  rich([["Keywords: ", { bold: true, size: 18 }],
        ["plant phenotyping, above-ground biomass, visual hull, TSDF fusion, RGB-D, self-supervised vision transformers", { size: 18, color: MUTED }]],
       { align: AlignmentType.LEFT, after: 240 }),
];

const introduction = [
  h1("1. Introduction"),
  p("Above-ground biomass is among the most-requested traits in plant phenotyping, and destructive harvest remains the reference method. Non-destructive alternatives estimate it from imaging, and the dominant pipeline is reconstruct-then-regress: recover a three-dimensional representation of the plant, summarise it with shape descriptors, and fit those descriptors to weighed mass. Reviews of high-throughput phenotyping treat plant height and biomass as the principal geometric traits and report biomass estimated from crop surface and crop height models with coefficients of determination between 0.55 and 0.79 for field crops [1]."),
  p("That pipeline has a component that is rarely interrogated. Given silhouettes from a ring of viewpoints, the natural reconstruction is space carving, whose output is the visual hull. Laurentini established that the visual hull is the maximal object consistent with a set of silhouettes [2]: any concavity that casts no silhouette from any viewpoint is filled. For a compact object this is a mild approximation. For a plant, whose above-ground structure is mostly the space between leaves, it is not obvious that it is an approximation at all."),
  p("This paper reports what happens when that assumption is tested rather than inherited. Our contributions are: (i) a physical plausibility diagnostic that checks a reconstruction against the density of the material it claims to represent, independent of any regression; (ii) evidence that space carving fails that check for 28 of 36 specimens and that the failure is a property of the operator; (iii) a controlled substitution of truncated signed distance fusion for silhouette intersection that produces the first statistically resolved improvement in biomass accuracy in this work; and (iv) a set of negative and withdrawn results reported in full, including a claim from an earlier stage of this project that did not survive re-examination."),
];

const related = [
  h1("2. Related work"),
  h2("2.1 Biomass from multi-view imaging"),
  p("Feng and colleagues survey unmanned-aerial phenotyping and organise it by trait, with plant height the most-studied geometric quantity and biomass estimated from height models, vegetation indices, or their combination [1]. That review also defines plant density in its agronomic sense, plants per unit area, which is a stand-level count and distinct from the volumetric density used as a diagnostic here. Amaducci and colleagues report for hemp that plant density affects stem biometry and fibre quality more than it affects biomass yield, and that biomass response is flat across a wide range of planting densities [3]. Both are useful reminders that the agronomic literature's density is not the physical one."),
  p("Acquisition frequency has been studied directly. Bresolin and colleagues report that the optimal image-acquisition frequency for a phenotype is phenotype-specific rather than universal [4]. Our view-count ablation reproduces that conclusion in a different modality and, notably, with the ordering that spectral bandwidth alone would not predict."),
  h2("2.2 Reconstruction operators"),
  p("Silhouette-based reconstruction is bounded above by the visual hull [2]. Volumetric range-image integration provides a different operator: Curless and Levoy accumulate signed distances from depth measurements into a truncated field whose zero crossing is the surface [5], and KinectFusion demonstrated the approach in real time with commodity depth sensors [6]. Marching cubes extracts a surface from the resulting field [7]. The distinction that matters here is evidential. A silhouette constrains the subject to lie somewhere along a ray; a depth sample asserts a surface at a specific distance. Only the second can represent a concavity."),
  h2("2.3 Learned and feed-forward reconstruction"),
  p("DeepVoxels introduced a persistent three-dimensional feature grid optimised over many observations of a single object [8], a lineage continued by neural radiance fields [9] and by three-dimensional Gaussian splatting [10]. These require view counts far beyond the twelve available here and, in the case of DeepVoxels, output novel-view colour rather than geometry, so they are positioned in this work as related architecture rather than as comparators."),
  p("The feed-forward pointmap family is more directly relevant. DUSt3R regresses pointmaps for image pairs without known camera parameters [11]; MASt3R adds metric grounding and matching [12]; Fast3R processes many views in a single pass [13]. Because these estimate cameras from images alone, they share no failure mode with a pipeline whose poses are estimated from the same depth it reconstructs, which makes them the natural independent check on the calibration-free registration used here."),
  h2("2.4 Foundation features and segmentation"),
  p("Self-supervised vision transformers provide general-purpose dense features without labels [14], and promptable segmentation provides subject masks without per-species training [15]. Both are used here in frozen form, as feature extractors and as an alternative segmenter respectively, and neither is fine-tuned."),
];

const methods = [
  h1("3. Materials and methods"),
  h2("3.1 Plant material and capture"),
  p("Thirty-eight potted specimens were captured across four collection batches: Eucalyptus (E001 to E020, V001 to V008) and Mango (M001 to M010). Two Kinect v2 units mounted opposite one another were carried through six positions 30 degrees apart, giving twelve azimuths per specimen. Colour is mapped into the depth frame by the driver, so both streams are 512 by 424 and pixel aligned. Above-ground shoots were cut and weighed fresh."),
  p("Pot mass was estimated for the first three batches and measured directly for V001 to V008 by weighing each pot after shoot removal. Comparing the two on those eight specimens gives a ratio of estimated to measured pot mass of 0.891 with a standard deviation of 0.018, a systematic 10.9 per cent under-estimate with very little scatter. Because net shoot mass is the difference of two large numbers, this propagates to roughly a 24 per cent overstatement of net mass for E001 to E010, where the pot dominates the total. All results below use as-recorded values; the calibration is reported so the uncertainty can be carried."),
  h2("3.2 Calibration-free registration"),
  p("No ChArUco sequence was captured, so camera extrinsics are estimated from the depth data. A RANSAC floor plane per view fixes tilt, roll and height; the subject axis, selected by cross-view agreement, fixes the origin; azimuth is refined by coordinate descent within a plus or minus 8 degree bound. That bound saturates on 25 of 30 specimens in the original batches, which makes registration the least verified assumption in the pipeline and motivates the pose-free check described in Section 6."),
  h2("3.3 Reconstruction operators"),
  p("Two operators are compared, and the comparison is deliberately narrow. Both consume identical views, identical poses and identical subject masks, and both write occupancy on the same 128-cubed grid at 12 mm spanning 1.536 m."),
  p("Space carving (Figure 1) intersects silhouette cones with depth-based free-space carving. TSDF fusion (Figure 2) integrates each depth sample as a signed distance with a three-voxel truncation band, and takes as interior those voxels lying behind an observed surface and within the truncation. Voxels no camera measured remain unknown rather than being assumed solid, so coverage is reported alongside volume; mean coverage is 0.12. A second fusion at 256-cubed and 6 mm is also reported, that being two depth samples across at the 1.1 m working distance, where one depth pixel spans 3.0 mm."),
  h2("3.4 Plausibility diagnostic"),
  p("For each specimen we compute implied bulk density as weighed shoot mass divided by reconstructed above-ground volume, cutting at a per-specimen pot rim estimated from a step in the vertical cross-section profile. Fresh above-ground plant tissue lies in the region of 300 to 900 kg per cubic metre. We adopt a deliberately generous 200 to 1000 band and classify each specimen as plausible, envelope (implied density far below the band, indicating enclosed air) or missing (far above, indicating unreconstructed material). The band edges are conventions; what the diagnostic supports is the direction and the order of magnitude, not a grading."),
  h2("3.5 Biomass regression"),
  p("Seven hand-crafted descriptors are read from each reconstruction: above-rim volume and its two-thirds power, height, mean and maximum radial spread, compactness against the swept cylinder, and floor footprint. Ridge regression on standardised descriptors is evaluated by leave-one-out cross-validation over all usable specimens. Every difference between methods is accompanied by a paired bootstrap over 20,000 resamples, and differences whose interval spans zero are reported as unresolved."),
];

const results = [
  h1("4. Results"),
  h2("4.1 Dataset and quality gate"),
  p("Thirty-six of 38 specimens pass the geometric quality gate; the SAM3D segmentation gate passes 33. X001 is excluded throughout, having two views and being the only specimen of its species."),
  table(
    ["Batch", "n", "Mean shoot mass", "SD", "Range"],
    [["Eucalyptus E001-E010", "10", "538 g", "114", "400-700"],
     ["Eucalyptus E011-E020", "8", "1844 g", "348", "1350-2350"],
     ["Eucalyptus V001-V008", "8", "1138 g", "484", "500-1800"],
     ["Mango M001-M010", "10", "1022 g", "284", "560-1470"]],
    [3000, 700, 1900, 900, 1600]),
  caption("Table 1", "Usable specimens by collection batch. V001 to V008 was captured to span the range occupied by the other batches rather than to form a new cluster, and it is the only batch whose pot masses were measured rather than estimated."),

  h2("4.2 Reconstructions are envelopes, not plants"),
  p("Under space carving, 8 of 36 specimens imply a bulk density inside the plausible band; the median across all specimens is 116.8 kg per cubic metre. Twenty-five imply less, and three imply more. All ten Mango specimens fall between 26 and 77, one to two orders of magnitude below plant tissue."),
  ...figure(path.join(FIG, "fig_density.png"), 620, "Figure 3",
    "Implied bulk density per specimen under each operator, log axis. The shaded band is the plausible range. Space carving places most specimens one to two orders of magnitude below plant tissue; fusion moves the population into the band."),
  p("The magnitude is easiest to see in a single case. Specimen M001, a Mango of weighed shoot mass 0.74 kg, carves to 25.79 L above the rim, implying 28.7 kg per cubic metre. The same views fused give 3.5 L and 214 kg per cubic metre at the same grid."),
  table(
    ["Group", "n", "Plausible", "Envelope", "Missing", "Median kg/m3"],
    [["Eucalyptus E001-E010", "10", "1", "9", "0", "127"],
     ["Eucalyptus E011-E020", "8", "3", "2", "3", "863"],
     ["Eucalyptus V001-V008", "8", "4", "4", "0", "248"],
     ["Mango", "10", "0", "10", "0", "52"],
     ["All, space carving", "36", "8", "25", "3", "117"],
     ["All, TSDF 12 mm", "36", "25", "11", "0", "272"],
     ["All, TSDF 6 mm", "36", "31", "1", "4", "529"]],
    [2600, 600, 1200, 1200, 1100, 1400]),
  caption("Table 2", "Plausibility by group under space carving, with the two fusion conditions for comparison. Holding the pot rim fixed at the carve's estimate isolates the occupancy operator alone and gives 21 of 36, so the rim estimator also improves on a fused profile."),

  h2("4.3 Changing the operator, not the regressor"),
  p("Features, protocol, grid, views and masks are held fixed; only the operator that produces occupancy differs. Two differences resolve."),
  ...figure(path.join(FIG, "fig_operator.png"), 620, "Figure 4",
    "Leave-one-out biomass RMSE per method under each reconstruction operator. Direct 2D uses no reconstruction and is therefore the control: it does not move."),
  table(
    ["Method", "Carved RMSE / R2", "Fused RMSE / R2", "Paired bootstrap"],
    [["geometric features", "0.544 / +0.030", "0.335 / +0.632", "-0.209 [-0.363, -0.066]"],
     ["volume allometric", "0.592 / -0.150", "0.469 / +0.278", "-0.123 [-0.202, -0.034]"],
     ["canopy area allometric", "0.598 / -0.170", "0.494 / +0.201", "not resolved"],
     ["mesh geometry", "0.507 / +0.157", "0.486 / +0.227", "not resolved"],
     ["direct 2D (control)", "0.469 / +0.279", "0.469 / +0.279", "unchanged"],
     ["mean predictor", "0.568 / -0.058", "0.568 / -0.058", "floor"]],
    [2600, 2000, 2000, 2426]),
  caption("Table 3", "Biomass estimation under each operator, leave-one-out over 36 specimens. Paired bootstrap over 20,000 resamples; negative favours fusion. The first two rows are the only statistically resolved improvements this work has produced."),
  p("Two methods that sat below the mean-predictor floor on the hull clear it on the fusion. Volume allometry moves from -0.150 to +0.278 and canopy area from -0.170 to +0.201. This matters more than the ordering. A single volume-to-mass law had appeared impossible across morphologies because hull density varied roughly tenfold between a bushy Mango and a thin Eucalyptus; on a fused reconstruction the same law works, because the volumes are no longer envelopes of differing emptiness. The canopy-area hypothesis, which we had rejected on the grounds that a hull's surface is envelope area rather than leaf area, partially survives once the surface follows the leaves."),

  h2("4.4 View-count ablation"),
  p("Reducing the number of views degrades geometric quality gently and reconstruction validity abruptly."),
  table(
    ["Views", "Usable", "Agreement", "Mean hull", "Plausible", "Median kg/m3"],
    [["3", "23/38", "0.360", "99.3 L", "1/23", "9.8"],
     ["4", "25/38", "0.424", "126.5 L", "0/25", "9.2"],
     ["6", "34/38", "0.521", "150.4 L", "2/34", "15.4"],
     ["12", "36/38", "0.608", "10.4 L", "8/36", "116.8"]],
    [900, 1100, 1400, 1500, 1300, 1600]),
  caption("Table 4", "View-count ablation under space carving. Multi-view agreement falls gently from 0.608 to 0.424 while the reconstructions cease to be physically possible, which is the argument for reporting a plausibility check beside a geometric quality metric rather than instead of one."),
  p("At four views, no specimen of the twenty-five that survive the quality gate is physically capable of weighing its plant. The median implied density is 9.2 kg per cubic metre, from hulls averaging 126 L for plants of at most 2.35 kg. Four views at 90 degrees is the visual-hull minimum for a convex object; a plant is the opposite of convex, and every unsampled azimuth leaves a prism of empty space uncarved."),

  h2("4.5 Frozen-feature backbone comparison"),
  p("A frozen self-supervised backbone improves on a CNN stem in point estimate but not in interval. Over 36 specimens, DINOv2-base reaches 0.392 kg against 0.458 for the CNN stem, a paired difference of -0.066 kg with an interval of -0.178 to +0.062. In a two-by-three factorial over segmenter and backbone, the only resolved effect is that of the backbone given SAM3D masks, -0.387 kg with an interval of -0.757 to -0.039; this is resolved largely because SAM3D without a learned backbone is unusually poor, so it evidences the fragility of hand-crafted descriptors to the segmentation rather than the value of the backbone."),
  p("Note that the probe and the baseline tables use different preprocessing: the probe rotates onto principal components before standardising, the baselines standardise raw features. On seven features this is a pure rotation, and it changes the result. The two tables are therefore not directly comparable, and we report them separately."),

  h2("4.6 Spectral characterisation"),
  p("Radial power spectra of the carved occupancy fields separate the morphologies, and half the dataset saturates the grid."),
  table(
    ["Group", "n", "95% bandwidth", "High-frequency share"],
    [["Eucalyptus E001-E010", "10", "29.3 +- 2.0", "0.128 +- 0.010"],
     ["Eucalyptus E011-E020", "8", "40.6 +- 2.3", "0.254 +- 0.045"],
     ["Eucalyptus V001-V008", "8", "41.7 +- 0.0", "0.257 +- 0.025"],
     ["Mango", "10", "41.7 +- 0.0", "0.273 +- 0.020"]],
    [2900, 700, 2200, 3226]),
  caption("Table 5", "Spectral bandwidth in cycles per metre. The 12 mm grid has a Nyquist limit of 41.7 cycles per metre, which 18 of 36 specimens reach exactly, so resolution rather than method is their binding constraint."),
];

const discussion = [
  h1("5. Discussion"),
  h2("5.1 A withdrawn claim"),
  p("An earlier stage of this work reported that features from a three-dimensional reconstruction outperform direct image regression, 0.397 against 0.440 kg RMSE on a 28-specimen subset. Re-tested with a paired bootstrap, that difference is -0.043 kg with a 95 per cent interval of -0.168 to +0.099. It was never resolved, and it is withdrawn. On the full 36-specimen set the point estimate reverses and remains unresolved, and it reverses again under a different feature-whitening choice. We report this in full because a point-estimate difference presented without its interval is precisely the error the rest of this paper is organised to avoid."),
  h2("5.2 The batch confound"),
  p("Collection batch alone explains a substantial share of mass variance. Across the two original Eucalyptus batches, batch membership explains R-squared 0.887, more than any method achieves. The V001 to V008 batch was collected specifically to break this, spanning 500 to 1800 g and therefore overlapping both existing clusters rather than forming a third. It reduces the batch-only figure to 0.744 across the three Eucalyptus batches and 0.697 across all four. Any comparison on this dataset is therefore partly a measurement of size-class separation, and we recommend that within-species results be reported alongside pooled ones."),
  h2("5.3 Limitations"),
  ...bullets([
    "Twelve views leave most leaf undersides unobserved. Mean fused coverage is 0.12, so roughly one eighth of the working volume was ever measured. Fusion escapes the visual hull's ceiling; it does not substitute for a dense photogrammetric capture.",
    "The fused interior is a band one truncation width deep behind each observed surface, not a filled solid, so its volume is a proxy rather than the plant's volume.",
    "Camera poses are estimated, not measured, and the azimuth refinement saturates its search bound on most specimens.",
    "The target is as-collected fresh mass, which carries water-content variance that no geometry can explain. Oven-dry biomass would be the stronger target.",
    "Pot mass is measured for eight specimens and estimated for the remaining twenty-eight, with a quantified 10.9 per cent bias on the estimates.",
    "Within any single batch, n is 8 to 10, which is too small to fit seven features. The design cannot test within-batch estimation, and the positive results reported here derive partly from cross-batch size separation.",
  ]),
];

const future = [
  h1("6. Ongoing and future work"),
  p("Three lines follow directly. First, the pose-free pointmap models [11, 12, 13] estimate cameras from images alone and therefore share no failure mode with a pipeline whose poses come from the depth it reconstructs. If implied densities remain low under independently estimated poses, the envelope argument closes and can no longer be answered by questioning the registration."),
  p("Second, a learned volumetric model is implemented and awaits training. The comparison of interest is not against classical features in the abstract but specifically whether a trained model inherits the 0.209 kg that classical features gained when the operator changed."),
  p("Third, the resolution prediction is directly testable: 18 of 36 specimens sit exactly at the grid's Nyquist limit, so a finer grid should move them and should not move the others."),
  p("For future capture we recommend weighing every pot after shoot removal, oven-drying a subsample to establish a fresh-to-dry ratio, recording stem diameter with callipers, and collecting roughly thirty specimens of one species spanning a continuous mass range in a single batch."),
];

const references = [
  h1("References"),
  p("[1] L. Feng, S. Chen, C. Zhang, Y. Zhang, and Y. He, \"A comprehensive review on recent applications of unmanned aerial vehicle remote sensing with various sensors for high-throughput plant phenotyping,\" Computers and Electronics in Agriculture, vol. 182, 106033, 2021.", { align: AlignmentType.LEFT, after: 60 }),
  p("[2] A. Laurentini, \"The visual hull concept for silhouette-based image understanding,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 16, no. 2, pp. 150-162, 1994.", { align: AlignmentType.LEFT, after: 60 }),
  p("[3] S. Amaducci, D. Scordia, F. H. Liu, Q. Zhang, H. Guo, G. Testa, and S. L. Cosentino, \"Key cultivation techniques for hemp in Europe and China,\" Industrial Crops and Products, vol. 68, pp. 2-16, 2015.", { align: AlignmentType.LEFT, after: 60 }),
  p("[4] T. Bresolin et al., \"Assessing optimal frequency for image acquisition in computer vision systems developed to monitor dairy cattle,\" Journal of Dairy Science, vol. 106, pp. 664-675, 2023.", { align: AlignmentType.LEFT, after: 60 }),
  p("[5] B. Curless and M. Levoy, \"A volumetric method for building complex models from range images,\" in Proc. SIGGRAPH, pp. 303-312, 1996.", { align: AlignmentType.LEFT, after: 60 }),
  p("[6] R. A. Newcombe et al., \"KinectFusion: real-time dense surface mapping and tracking,\" in Proc. IEEE ISMAR, pp. 127-136, 2011.", { align: AlignmentType.LEFT, after: 60 }),
  p("[7] W. E. Lorensen and H. E. Cline, \"Marching cubes: a high resolution 3D surface construction algorithm,\" in Proc. SIGGRAPH, pp. 163-169, 1987.", { align: AlignmentType.LEFT, after: 60 }),
  p("[8] V. Sitzmann, J. Thies, F. Heide, M. Niessner, G. Wetzstein, and M. Zollhoefer, \"DeepVoxels: learning persistent 3D feature embeddings,\" in Proc. IEEE/CVF CVPR, 2019.", { align: AlignmentType.LEFT, after: 60 }),
  p("[9] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, \"NeRF: representing scenes as neural radiance fields for view synthesis,\" in Proc. ECCV, 2020.", { align: AlignmentType.LEFT, after: 60 }),
  p("[10] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis, \"3D Gaussian splatting for real-time radiance field rendering,\" ACM Transactions on Graphics, vol. 42, no. 4, 2023.", { align: AlignmentType.LEFT, after: 60 }),
  p("[11] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, \"DUSt3R: geometric 3D vision made easy,\" in Proc. IEEE/CVF CVPR, 2024.", { align: AlignmentType.LEFT, after: 60 }),
  p("[12] V. Leroy, Y. Cabon, and J. Revaud, \"Grounding image matching in 3D with MASt3R,\" in Proc. ECCV, 2024.", { align: AlignmentType.LEFT, after: 60 }),
  p("[13] J. Yang et al., \"Fast3R: towards 3D reconstruction of 1000+ images in one forward pass,\" in Proc. IEEE/CVF CVPR, 2025.", { align: AlignmentType.LEFT, after: 60 }),
  p("[14] M. Oquab et al., \"DINOv2: learning robust visual features without supervision,\" Transactions on Machine Learning Research, 2024.", { align: AlignmentType.LEFT, after: 60 }),
  p("[15] A. Kirillov et al., \"Segment Anything,\" in Proc. IEEE/CVF ICCV, 2023.", { align: AlignmentType.LEFT, after: 60 }),
  p("[16] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma, \"Frequency principle: Fourier analysis sheds light on deep neural networks,\" Communications in Computational Physics, vol. 28, no. 5, pp. 1746-1767, 2020.", { align: AlignmentType.LEFT, after: 240 }),
  new Paragraph({
    spacing: { before: 200, after: 80 },
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 8 } },
    children: [new TextRun({
      text: "Note on references. Bibliographic details above were written from the author's working notes and from the two review articles read directly during this work ([1] and [3]). Verify every entry against the publisher record before submission; page ranges and years in particular have not been machine-checked.",
      size: 17, italics: true, color: MUTED, font: "Calibri" })],
  }),
];

/* Architecture figures live in their own section, one per page, because each is
   a tall single-column flow and shrinking two onto a page makes both unreadable. */
const appendix = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("Appendix A. Method architectures"),
  p("Each diagram runs the full length of the argument from image acquisition to a mass in kilograms. They share a layout deliberately: placed side by side, the stage that differs is the one that fails to line up. All are generated from the pipeline's own configuration constants and measured results, so they cannot drift out of date."),
  ...figure(path.join(ARCH, "architecture_carve.png"), 400, "Figure 1",
    "Method A, space carving. The reference pipeline. Silhouette intersection bounded above by the visual hull."),
  new Paragraph({ children: [new PageBreak()] }),
  ...figure(path.join(ARCH, "architecture_fusion.png"), 400, "Figure 2",
    "Method C, TSDF depth fusion. Identical up to stage 03; from stage 04 the same depth maps are integrated rather than intersected, and an additional stage records which space was actually observed."),
  new Paragraph({ children: [new PageBreak()] }),
  ...figure(path.join(ARCH, "architecture_posefree.png"), 400, "Figure 5",
    "Method D, pose-free reconstruction with feed-forward pointmap models. No registration stage: cameras and geometry are regressed from images, which is why this is the independent check on a registration estimated from depth."),
  new Paragraph({ children: [new PageBreak()] }),
  ...figure(path.join(ARCH, "architecture_ggssvt.png"), 400, "Figure 6",
    "Method E, the learned volumetric model. Reconstruction supplies occupancy targets rather than the answer, which is what makes the model self-supervised and also caps what it can learn."),
  new Paragraph({ children: [new PageBreak()] }),
  ...figure(path.join(REPO, "reports/gallery/contact_sheet_geometric.png"), 620, "Figure 7",
    "All 36 usable specimens as carved, depth-cued and coloured on a perceptually uniform ramp. The E001 to E010 batch reconstructs largely as pot with a small tuft, which the per-specimen rim estimator detects and reports rather than silently cutting at a constant."),
];

/* ---------- assemble ----------------------------------------------------- */

const doc = new Document({
  creator: "Aaron Masuba",
  title: TITLE,
  description: "Draft paper reporting results to date.",
  numbering: {
    config: [{
      reference: "dots",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u2022",
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
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED })],
        })],
      }),
    },
    children: [
      ...front, ...abstract, ...introduction, ...related,
      ...methods, ...results, ...discussion, ...future, ...references,
      ...appendix,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Masuba_biomass_paper_draft.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0), "KB");
});
