/*
 * PhD research proposal, in the department's template.
 *
 * Same template as the MEng version, three differences the template itself
 * stipulates: the summary page carries two anticipated articles rather than one
 * and labels the third field "Research gap that will be addressed", the faculty
 * requires one accepted and one submitted rather than one submitted, and the
 * body may run to ten A4 pages rather than two.
 *
 * The scope is the one the Postgraduate Committee already identified. Its note
 * on the 2025 MEng proposal reads that four hypotheses and three papers
 * "would indicate a study with the scope and complexity of a PhD". This document
 * takes that at face value and proposes the study at the size it actually is,
 * carrying all four hypotheses instead of cutting them.
 */

const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, PageBreak, AlignmentType } = require("docx");
const C = require("./proposal_common");

const { p, rich, h, bullets, claim } = C;
const brk = () => new Paragraph({ children: [new PageBreak()] });

/* ------------------------------------------------------------------ *
 * Summary page. One A4, 11 point, single spacing.
 * ------------------------------------------------------------------ */

const summary = [
  h("Summary Page"),

  h("1. Research questions", 2),
  ...bullets([
    "Can self-supervised vision transformers learn plant representations that capture geometric structure and physiological traits from minimal labelled data?",
    "What transformer-based grounding and fusion strategies enable accurate 3D reconstruction of plant architecture from multi-view RGB-D imagery?",
    "Can the reconstructed 3D representation predict biomass label-efficiently, and what is the relationship between reconstruction quality and estimation accuracy?",
    "Which reconstruction operator suits thin, self-occluding plant structure, and how is that decided with no ground-truth geometry to appeal to?",
  ]),

  h("2. Approach", 2),
  p("Potted specimens are captured with a two-camera rig carried through six positions, giving twelve registered RGB-D views each, and destructively harvested for fresh above-ground mass. Reconstruction operators are compared under one protocol with features, grid, views and masks held fixed, so the operator is the only variable, and every difference carries a paired bootstrap interval. Because harvest yields a mass and not a geometry, validity is established physically: measured mass divided by reconstructed above-ground volume must fall inside the bulk density of fresh plant tissue. Transformer features are then evaluated against hand-crafted descriptors on whichever reconstruction that criterion admits, and the work extends to pose-free reconstruction and a second capture campaign."),

  h("3. Contribution", 2),
  p("A validation criterion for multi-view plant reconstruction that needs no reference geometry, and the programme it opens. Applied to silhouette carving it shows the recovered volume is the canopy envelope rather than the plant, a property of the operator and not of resolution, which bounds what any silhouette-based pipeline can achieve. Replacing carving with depth fusion under the criterion gives a resolved improvement in biomass accuracy, and the criterion also exposes a failure in standard practice: silhouette agreement, the usual proxy for reconstruction quality, ranks the worse reconstruction higher on this data."),

  h("4. Anticipated peer-reviewed journal articles", 2),
  rich([["Target journal: ", { bold: true }],
        ["Computers and Electronics in Agriculture (ISI, IF ~8.3)"]], { after: 60 }),
  rich([["Preliminary title: ", { bold: true }],
        ["Silhouette carving recovers the canopy envelope, not the plant: a physical criterion for validating multi-view reconstruction in biomass phenotyping"]], { after: 60 }),
  rich([["Research gap that will be addressed: ", { bold: true }],
        ["Reconstruct-then-regress phenotyping pipelines adopt silhouette reconstruction without testing whether the recovered volume can represent the plant, because destructive harvest leaves no reference geometry to test against. No criterion exists for validating a plant reconstruction in that setting."]], { after: 110 }),

  rich([["Target journal: ", { bold: true }],
        ["International Journal of Computer Vision"]], { after: 60 }),
  rich([["Preliminary title: ", { bold: true }],
        ["Geometry-grounded self-supervised representations for label-efficient biomass regression from sparse multi-view capture"]], { after: 60 }),
  rich([["Research gap that will be addressed: ", { bold: true }],
        ["Self-supervised transformers supply dense features without labels and pose-free pointmap models supply geometry without calibration, but neither has been evaluated where each label costs a destroyed plant and the sample is small enough that differences must be reported with intervals rather than as rankings."]]),

  brk(),
];

/* ------------------------------------------------------------------ *
 * Full research proposal. Up to ten A4 pages for a PhD.
 * ------------------------------------------------------------------ */

const body = [
  h("Full research proposal"),

  /* ---- 1. Introduction ---- */
  h("1. Introduction", 2),
  p("Above-ground biomass is among the most requested traits in plant phenotyping. It underwrites carbon accounting, yield forecasting and growth modelling, and its reference method has not changed: cut the plant, weigh it. Destructive harvest is accurate, and it ends the measurement series for that individual. A method that reads mass from images instead would allow the same plant to be followed through a season, which is why non-destructive biomass estimation has been pursued continuously for two decades [1]."),
  p("Almost every such method follows one pattern. Reconstruct the plant in three dimensions from images, summarise the reconstruction with shape descriptors such as volume, height, projected area and compactness, and regress those descriptors on weighed mass. The pattern is sound in principle. Its weak point is that the first step is treated as solved. Given silhouettes from a ring of viewpoints, the natural and cheap operator is space carving, and what space carving produces is not the object. Laurentini established that it produces the visual hull, the maximal solid consistent with those silhouettes [2]. Every concavity that no silhouette cuts into is filled."),
  p("For a compact object this is a mild approximation. For a plant it may not be an approximation at all. The above-ground structure of a leafy specimen is mostly the space between leaves, and no silhouette from any azimuth cuts into the gaps behind an outer leaf. The hull that results is the canopy envelope: the shape the plant would have if it were solid. Whether the resulting volume bears a usable relationship to mass is an empirical question, and the literature has largely not asked it, because with destructive harvest there is no reference geometry to compare against and therefore no obvious way to ask."),
  p("This proposal is built around a way to ask it that needs no reference geometry. A reconstruction claims a volume. Harvest measures a mass. Their quotient is a bulk density, and bulk density is a physical property of plant tissue with a known range. A reconstruction whose implied density sits an order of magnitude below balsa wood is not a poor reconstruction of the plant; it is a reconstruction of something else. That single check turns an unfalsifiable modelling choice into a measurable one, and the preliminary work reported in section 5.2 shows it changes the answer."),

  /* ---- 2. Literature ---- */
  h("2. Overview of current literature", 2),

  h("2.1 Biomass estimation and phenotyping practice", 2),
  p("Reviews of high-throughput phenotyping organise the field by trait, with plant height the most studied geometric quantity and biomass estimated from crop height models, canopy surface models, vegetation indices, or their combination [1]. Reported coefficients of determination for field crops fall between 0.55 and 0.79, and the dominant predictors are height and projected canopy area, both of which are two-dimensional or near two-dimensional summaries. Where three-dimensional structure is used, it typically arrives as a point cloud from structure-from-motion or from LiDAR, and the shape descriptors extracted from it are the same volume and area quantities."),
  p("Two practical findings from adjacent work bear on the design here. Acquisition frequency has been studied directly in a livestock imaging context, with the conclusion that the optimal frequency is trait-specific rather than universal [3], which has a direct analogue in how many viewpoints a plant capture needs. And agronomic work on plant density shows it affects stem biometry considerably more than it affects biomass yield [4], a reminder that the agronomic sense of density, plants per unit area, is a different quantity from the volumetric sense used as the diagnostic in this work."),
  p("What the phenotyping literature does not generally supply is an account of how reconstruction error propagates into biomass error. Studies report a regression score, and when that score is poor the response is usually a stronger regressor or more features. The possibility that the input geometry cannot support the mapping at all is rarely separated from the possibility that the mapping was badly fitted."),

  h("2.2 Multi-view reconstruction", 2),
  p("Silhouette-based reconstruction is bounded above by the visual hull [2]. That bound is not a matter of resolution or of viewpoint count: adding cameras shrinks the hull toward the hull of the infinite-view limit, which for a non-convex object still strictly contains it. The bound is a property of the evidence. A silhouette asserts only that the subject lies somewhere along a ray, so a set of silhouettes can carve away space that is definitely empty and can never assert that a particular interior region is empty."),
  p("Volumetric integration of range images supplies evidence of the second kind. Curless and Levoy accumulate signed distances from depth measurements into a truncated field whose zero crossing is the surface [5], and KinectFusion demonstrated the same operator running in real time on commodity depth sensors [6]. A depth sample asserts a surface at a specific distance along the ray and therefore asserts that the space in front of it is empty. Only this second kind of evidence can represent a concavity, which is precisely the structure a canopy consists of."),
  p("Neural scene representations offer a third route. DeepVoxels learns a persistent voxel embedding for view synthesis, and neural radiance fields and Gaussian splatting reconstruct appearance to a high standard. These are appearance models rather than occupancy models: they answer what a new view looks like, not which voxels contain matter, and extracting a watertight surface from them requires a density threshold that has no physical calibration. For a pipeline whose output must be a volume in litres, that threshold is exactly the free parameter the plausibility criterion is designed to remove."),
  p("Most recently, feed-forward pointmap models estimate camera geometry and dense structure directly from uncalibrated images. DUSt3R regresses pointmaps in a common frame without prior camera parameters [7], MASt3R adds metric matching [8], and later variants scale to many views in one pass. Their relevance here is specific: the capture used in this work has no calibration target, so camera poses are estimated from the depth data. Any error in that estimation is shared by every operator that consumes those poses. A pointmap model that estimates cameras from images alone shares no failure mode with it and therefore provides an independent check."),

  h("2.3 Self-supervised representation and its transfer to 3D", 2),
  p("Self-supervised vision transformers now supply general-purpose dense features without labels. DINOv2 patch tokens carry semantic and part-level structure that transfers across domains under a linear probe [9], and promptable segmentation supplies subject masks without per-species training [10]. For a label-poor problem this is the central attraction: every biomass label costs a destroyed plant, so a representation that needs none is worth more here than in settings where annotation is merely expensive."),
  p("Transferring those features to three dimensions is an active question. Lifting two-dimensional features into a voxel grid by projection and occlusion-aware aggregation, in the manner of recent 2D-to-3D transfer work, keeps the representation strong while placing it in the geometry. Whether the result is better than the hand-crafted descriptors it would replace has to be measured under a protocol that does not quietly change the preprocessing between the two, and preliminary work here found exactly that trap: the same seven descriptors score 0.458 or 0.544 depending only on whether they are rotated before standardisation, a difference larger than most of the effects being tested."),

  h("2.4 Recognition, reconstruction and reorganization", 2),
  p("Malik et al. propose the interaction between recognition, reconstruction and reorganization as a unifying frame for computer vision: grouping generates candidates which recognition classifies and which are then refined, while reconstruction both benefits from and informs the other two [13]. Two of the three bear on this work and the third does not, which is worth stating rather than leaving as a silent gap."),
  p("Reconstruction is the subject throughout. Reorganization enters as the segmentation that decides which pixels are plant, and the work contains one experiment squarely in the reciprocal spirit the paper argues for: features from a self-supervised transformer are lifted onto the reconstructed points and clustered, to test whether a grouping informed by semantics separates plant from pot where a colour threshold cannot. It does not, for the measurable reason given in section 5.2, and a negative result about that interaction is still a result about it. Recognition is out of scope: the specimens are one genus per batch and no category label is predicted, so a recognition claim would have nothing to be tested against."),

  h("2.5 Validation without ground-truth geometry", 2),
  p("The gap this proposal addresses sits here. Where reference geometry exists, reconstruction is scored by Chamfer distance, Hausdorff distance, F-score at a tolerance, or volumetric intersection over union. Destructive-harvest phenotyping has no reference geometry, so practice falls back on self-consistency: project the reconstruction back into the captured views and measure agreement with the silhouettes."),
  p("That substitution is not neutral. A visual hull is by construction consistent with every silhouette used to build it, so self-consistency is guaranteed for the hull rather than earned by it. Any operator that claims less than the hull, including a depth fusion that only asserts surfaces a camera actually measured, will score worse on silhouette agreement while being closer to the plant. The metric therefore has a systematic preference for the method that overclaims. Section 5.2 reports this preference measured rather than argued."),

  /* ---- 3. Motivation ---- */
  h("3. Motivation", 2),
  p("Three considerations motivate the work. The first is label poverty. Every training example costs a destroyed plant, which sets the practical ceiling on dataset size for this problem at tens rather than thousands of specimens and makes label-efficient methods valuable in a way they are not elsewhere. It also means that at achievable sample sizes, differences between methods must be reported with confidence intervals, because at n in the tens most differences a ranking table would present as results do not survive a paired bootstrap."),
  p("The second is that the field validates reconstruction by proxy, for the structural reason set out in section 2.4. A pipeline can pass every geometric check available to it and still be reconstructing an object that could not physically weigh what the plant weighs, and nothing in the standard protocol would detect it. The consequence is not hypothetical: preliminary work found silhouette agreement and physical plausibility ranking two operators in opposite orders, and biomass accuracy agreeing with plausibility."),
  p("The third is that the confusion has a cost that compounds. If reconstruction is assumed adequate, poor biomass accuracy is attributed to the regressor, and effort goes into architectures. Preliminary work here tested that attribution directly by holding the features and protocol fixed and swapping the regressor between ridge regression, random forests, gradient boosting and a small network. No swap resolved. Swapping the reconstruction operator, with the regressor held fixed, did. Effort spent on the estimator while the input is an envelope is effort spent on the wrong component, and the plausibility criterion is what tells the two apart before the effort is spent."),

  /* ---- 4. Objective ---- */
  h("4. Objective", 2),
  p("The objective is to establish which reconstruction operators can represent plant structure well enough to support biomass regression, to establish that by a criterion that does not require reference geometry, and to determine what self-supervised representation learning adds once the reconstruction is adequate. The four hypotheses carried from the original proposal are retained, each restated so that a specific measurement decides it."),
  claim("H1.", "Self-supervised vision transformers with geometry and reorganisation grounding outperform convolutional baselines on segmentation and biomass prediction, and reach a given accuracy from substantially fewer labelled examples.",
    "leave-one-out root-mean-square error and coefficient of determination against a convolutional baseline on identical reconstructions, with a paired bootstrap on the difference, plus a label-efficiency curve over subsampled training sets. Recognition, the third of the three R's [13], is deliberately out of scope: one genus per batch and no category task means a recognition claim would have nothing to be tested against."),
  claim("H2.", "Geometry-grounded models show higher consistency across viewpoints and produce reconstructions that are more faithful to the plant.",
    "held-out-view re-projection agreement, reported jointly with implied bulk density, because section 2.4 predicts these two can disagree and the pair is more informative than either."),
  claim("H3.", "Frequency grounding and geometry grounding together improve how a fixed parameter budget is allocated, in particular through positional encoding of 3D coordinates and spectral features.",
    "matched-capacity comparison across Fourier band ladders, with the radial power spectrum of the target measured first so that the band ceiling is set against the data's own Nyquist limit rather than chosen."),
  claim("H4.", "Hybrid models are more robust to occlusion, sensor noise and sparse angular sampling.",
    "reconstruction and estimation rebuilt from evenly spaced view subsets, and under injected depth noise and simulated occlusion, with physical plausibility as the pass criterion at each level rather than a relative score."),
  p("A fifth objective, which the original proposal did not anticipate, is to establish the plausibility criterion itself as a reusable instrument: to characterise its sensitivity, its failure modes, and the density band appropriate for different plant material.", { after: 140 }),

  /* ---- 5. Proposed research ---- */
  h("5. Proposed research", 2),

  h("5.1 Data and capture", 2),
  p("Capture uses two Kinect v2 units on a rig carried through six positions thirty degrees apart, giving twelve azimuths per specimen, with colour mapped into the depth frame. Thirty-eight potted Eucalyptus and Mango specimens have been captured and destructively harvested to date, of which thirty-six carry a complete set of views. Above-ground shoots are weighed fresh and pot mass is weighed after shoot removal, so net mass is measured rather than estimated for the most recent batch."),
  p("Two limitations of the existing set are known and are addressed by the work rather than hidden. No calibration target was recorded, so extrinsics are estimated from the depth data by fitting a floor plane per view and recovering the subject axis by cross-view agreement; work package C provides the independent check on this. And the specimens fall into two morphological clusters whose masses barely overlap, so batch membership alone explains a large share of mass variance. Any model fitted on this set separates size classes at least as much as it estimates mass, and a second campaign spanning a continuous mass range within one species is therefore part of the proposed work rather than an optional extension."),

  h("5.2 Preliminary results", 2),
  p("Work already completed establishes that the problem posed here is real and gives the programme its starting point. Under space carving at a 12 mm grid, eight of thirty-six specimens produce a reconstruction whose implied bulk density falls inside a deliberately generous band of 200 to 1000 kilograms per cubic metre. All ten Mango specimens land between 26 and 77 kilograms per cubic metre, an order of magnitude below fresh plant tissue. A view-count ablation sharpens this: at four views, zero of twenty-five reconstructions are physically capable of weighing what the plant weighs, with a median implied density of 9.2 kilograms per cubic metre, which is lighter than expanded polystyrene."),
  p("Substituting truncated signed distance fusion of the same depth maps, with grid, views, masks, features and protocol all unchanged, raises the plausible count to twenty-five of thirty-six and moves biomass root-mean-square error from 0.544 to 0.335 kilograms, a paired-bootstrap difference of minus 0.209 with a 95 per cent interval of minus 0.363 to minus 0.066. An image-only control that touches no reconstruction is unchanged at 0.469, which establishes that the reconstruction and not the regressor was the limiting component. Two allometric baselines that had sat below the mean-predictor floor cleared it under the same substitution."),
  p("The methodological finding follows from the same run. Silhouette agreement scores the carve at 0.407 and the fusion at 0.219, ranking them in the opposite order to both plausibility and biomass accuracy, while depth error is a tie at 67.9 against 67.4 millimetres. A metric that reads as reconstruction quality prefers the worse reconstruction, exactly as section 2.4 predicts. Reporting it alone would have pointed the project in the wrong direction, and this is the strongest available argument for the plausibility check."),

  h("5.3 Work package A: reconstruction operators and the validity criterion", 2),
  p("Extend the comparison beyond carving and fusion to Poisson surface reconstruction on the fused field, to a learned occupancy decoder, and to inverse procedural modelling in which a parametric plant model with explicit leaves is fitted to the observations. The last of these addresses the specific obstacle preliminary work identified: a hull's surface area is envelope area rather than leaf area, so leaf-area-based allometry cannot work on hull output, whereas a fitted model makes leaf area a parameter instead of something the sensor must resolve. Characterise the plausibility criterion itself across a range of plant material, establish its sensitivity to the pot rim estimate and to grid resolution, and determine the density band appropriate for woody against herbaceous tissue."),

  h("5.4 Work package B: geometry-grounded representation learning", 2),
  p("Evaluate self-supervised transformer features in two forms on whichever reconstructions work package A admits. First as frozen features under a linear probe, giving the label-efficiency curve H1 needs. Second as a geometry-grounded encoder trained with occupancy supervision derived from the reconstruction itself, which requires no manual annotation and is the label-efficient training signal the original proposal named. Two-dimensional features are lifted into the voxel grid by projection with an occlusion test. Every comparison holds preprocessing fixed across arms, because preliminary work showed a preprocessing difference alone can exceed the effect under test. A convolutional encoder of matched capacity provides the H1 baseline, and the frequency-grounding ladder of H3 is run against the measured radial spectrum of the occupancy target."),

  h("5.5 Work package C: pose-free reconstruction and angular sampling", 2),
  p("Reconstruct the same specimens with feed-forward pointmap models that estimate cameras from images alone [7], [8]. This serves two purposes. It tests the least verified assumption in the current pipeline, namely the depth-derived extrinsics, using a method that shares no failure mode with them. And it removes the calibration requirement from the capture protocol, which matters for any field deployment. Angular sampling is studied by rebuilding from evenly spaced subsets and scoring each under the validity criterion, extending the existing three, four, six and twelve view ladder upward with denser capture so that the plausibility curve can be located rather than only bounded from below."),

  h("5.6 Work package D: generalisation, scale and robustness", 2),
  p("Run a second capture campaign spanning a continuous mass range within a single species, which breaks the batch structure that currently caps every claim on the existing set. Add at least one morphologically distinct species so that cross-morphology generalisation can be tested rather than assumed, since preliminary work found one volume-to-mass law cannot span bushy and thin architectures. Robustness for H4 is evaluated by injecting depth noise at measured sensor characteristics and by simulated occlusion, with physical plausibility as the pass criterion at each level."),

  h("5.7 Evaluation protocol", 2),
  p("One protocol governs every comparison. Biomass is estimated by leave-one-out cross-validation over all usable specimens, reporting root-mean-square error, mean absolute error and the coefficient of determination, with a paired bootstrap over twenty thousand resamples on every difference between methods. Differences whose interval spans zero are reported as unresolved rather than as results, and the mean predictor is always shown as the floor. Reconstruction is reported on two axes kept explicitly apart: agreement between operators by Chamfer distance, ninety-fifth-percentile Hausdorff distance, F-score at a fixed metric tolerance and voxel intersection over union, which measures how far apart two methods are and is not accuracy; and explanatory power against the captured views by silhouette agreement, depth error and depth peak signal-to-noise ratio, reported alongside implied bulk density so that the disagreement between them stays visible. Acceptance gates run at each pipeline stage on mask area, reconstruction sanity, training loss and prediction spread, so that a degenerate intermediate result blocks rather than propagates into a regression score."),

  h("5.8 Programme and sequencing", 2),
  p("Year one completes work package A and the second capture campaign of work package D, since the dataset is the binding constraint and the operator comparison is the foundation everything else rests on; the first article is written in this period from results largely in hand. Year two runs work package B in full and work package C alongside it, these being independent. Year three completes the robustness study, the cross-species generalisation test, and the second article, and consolidates the thesis. Each work package terminates in a result that stands whether or not the following package succeeds, which is deliberate: the operator comparison, the label-efficiency curve and the pose-free check are each publishable independently."),

  /* ---- 6. Contribution ---- */
  h("6. Contribution", 2),
  p("The primary contribution is a validation criterion for multi-view plant reconstruction that requires no reference geometry, together with a characterisation of when it applies and what it detects. The criterion is not novel physics; its contribution is that it converts an assumption the phenotyping literature has been unable to test into a measurement any destructive-harvest study can make from data it already collects. Applied to space carving it establishes that the recovered volume is the canopy envelope rather than the plant, that this follows from what a silhouette can assert and is therefore not remedied by resolution or by more viewpoints, and that it bounds what any silhouette-based method can achieve on these morphologies."),
  p("The second contribution is the controlled substitution the criterion licenses, and the evidence that the reconstruction operator rather than the estimator was the limiting component. Holding features, grid, views, masks and protocol fixed and changing only the operator produces a resolved improvement in biomass accuracy, while holding the operator fixed and changing the estimator across four model families produces none. This is a claim about where effort should go in reconstruct-then-regress pipelines, and it is supported by an experiment designed to be able to falsify it."),
  p("The third contribution is methodological and reaches beyond biomass. Silhouette agreement, the conventional stand-in for reconstruction quality when no reference geometry exists, systematically prefers whichever operator overclaims, because it measures consistency with the input rather than fidelity to the object, and for a hull that consistency is guaranteed rather than earned. Demonstrating this with a case where the preferred method is also the physically impossible one is a caution the phenotyping literature does not currently carry, and it generalises to any application that validates reconstruction against the images it was built from."),
  p("The fourth contribution is the label-efficiency result for geometry-grounded self-supervised representations under conditions where labels are genuinely scarce because each one destroys a specimen. Reporting it with intervals at achievable sample sizes, including the comparisons that do not resolve, is itself a contribution to how small phenotyping studies should report, in a subfield where ranking tables at n in the tens are common."),
  p("Two supporting contributions arise from the capture setting. Calibration-free rig registration from depth alone, with its diagnostics and its failure cases, was not anticipated in the original proposal and is directly useful to anyone capturing multi-view data without a calibration target. And the view-count requirement, established by physical validity rather than by relative score, gives a defensible answer to how many viewpoints a plant capture needs, where the previous answer was convention."),

  brk(),
];

/* ------------------------------------------------------------------ */

const refs = [
  h("References"),
  ...[
    "[1] L. Feng, S. Chen, C. Zhang, Y. Zhang, and Y. He, “A comprehensive review on recent applications of unmanned aerial vehicle remote sensing with various sensors for high-throughput plant phenotyping,” Computers and Electronics in Agriculture, vol. 182, 106033, 2021.",
    "[2] A. Laurentini, “The visual hull concept for silhouette-based image understanding,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 16, no. 2, pp. 150–162, 1994.",
    "[3] T. Bresolin et al., “Assessing optimal frequency for image acquisition in computer vision systems developed to monitor dairy cattle,” J. Dairy Sci., vol. 106, pp. 664–675, 2023.",
    "[4] S. Amaducci et al., “Key cultivation techniques for hemp in Europe and China,” Industrial Crops and Products, vol. 68, pp. 2–16, 2015.",
    "[5] B. Curless and M. Levoy, “A volumetric method for building complex models from range images,” in Proc. SIGGRAPH, 1996, pp. 303–312.",
    "[6] R. A. Newcombe et al., “KinectFusion: real-time dense surface mapping and tracking,” in Proc. IEEE ISMAR, 2011, pp. 127–136.",
    "[7] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: geometric 3D vision made easy,” in Proc. IEEE/CVF CVPR, 2024.",
    "[8] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” in Proc. ECCV, 2024.",
    "[9] M. Oquab et al., “DINOv2: learning robust visual features without supervision,” Trans. Machine Learning Research, 2024.",
    "[10] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF ICCV, 2023.",
    "[11] V. Sitzmann, J. Thies, F. Heide, M. Nießner, G. Wetzstein, and M. Zollhöfer, “DeepVoxels: learning persistent 3D feature embeddings,” in Proc. IEEE/CVF CVPR, 2019.",
    "[12] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “NeRF: representing scenes as neural radiance fields for view synthesis,” in Proc. ECCV, 2020.",
    "[13] J. Malik, P. Arbelaez, J. Carreira, K. Fragkiadaki, R. Girshick, G. Gkioxari, S. Gupta, B. Hariharan, A. Kar, and S. Tulsiani, “The three R’s of computer vision: recognition, reconstruction and reorganization,” Pattern Recognition Letters, vol. 72, pp. 4–14, 2016.",
  ].map((t) => p(t, { align: AlignmentType.LEFT, after: 80 })),
  C.verifyNote(),
  ...C.contactBlock(),
];

const doc = new Document({
  creator: C.CANDIDATE.name,
  title: "Research proposal: PhD",
  numbering: C.numbering,
  styles: C.styles,
  sections: [{
    properties: C.margins,
    footers: { default: C.pageFooter() },
    children: [
      ...C.titlePage("PhD", "25 August 2026"), brk(),
      ...summary, ...body, ...refs,
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Masuba_research_proposal_PhD.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024).toFixed(0), "KB");
});
