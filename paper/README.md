# Regenerating the paper and the proposal

The draft is generated, not hand-written, for the same reason the architecture
diagrams are: results change, and a document that has to be edited by hand to
keep up eventually stops keeping up. Every number in it comes from an artefact in
`work_dirs/ggssvt`, so regenerating after a run picks the new values up.

## Build it

Node's `docx` package is not vendored here. Install it once into this directory:

```bash
cd paper && npm install docx
```

Then, from the repository root:

```bash
python paper/figures.py
```

```bash
node paper/paper.js
```

The first writes the two result figures that the pipeline does not already emit,
reading the caches directly so the plausibility counts in the figure cannot
disagree with the text. The second assembles the document, pulling the
architecture diagrams from `work_dirs/ggssvt/reports/architecture` and the
contact sheet from the gallery, so run `cli architecture` and `cli gallery`
first if those are stale.

Both documents land next to their generators, in this directory.

## The feasibility results section

```bash
node paper/feasibility.js
```

The results, framed the way a feasibility study reports them, after Malan et al.
in the Journal of Voice, which shares a supervisor with this project. Three
conventions are taken from it: criteria fixed a priori with the reason for each
threshold, a staged screening funnel showing everything that entered each stage
rather than only what survived, and failed stages reported at the same weight as
successful ones.

Every number is read from `work_dirs/ggssvt/reports` at build time, so unlike
`paper.js`, whose tables are literals, this document cannot drift from the
artefacts. Run `python -m ggssvt.cli reciprocity` and the rest of the pipeline
first; the figure comes from `ggssvt.eval.funnel`, which reads the same reports.

## The results paper

```bash
node paper/results_paper.js
```

Ten pages covering everything measured to date, with every number read from
`work_dirs/ggssvt/reports` at build time. It is a methods and validation paper
rather than a biomass paper, and that is deliberate: the biomass regression on
our own specimens does not resolve, section 5 says why in terms of what the
design could detect, and section 7 tests the same methods on an external set
where they do resolve.

It needs `virtual_views.json`, `external_lettuce.json`, `batch_holdout.json`,
`resolution.json`, `pedestal.json` and `pot_mass.json`, so run `cli
virtual-views`, `cli external`, `cli batch-holdout`, `cli resolution`, `cli
pedestal` and `cli pot-mass` first. Figures come from `reports/figures`,
`reports/gallery` and `reports/overlays`, so `cli plots`, `cli gallery` and `cli
overlay` too.

`docx_common.js` holds the builders both this and `paper.js` use. It was
extracted when the second document needed them, so a change to a table border or
a caption style cannot leave two documents from the same project looking
unrelated.

## The progress deck

```bash
cd paper && npm install pptxgenjs
```

```bash
node paper/presentation.js
```

Twenty-one slides covering everything run to date, with speaker notes. Like
`feasibility.js`, every quantitative claim is read out of
`work_dirs/ggssvt/reports` at build time rather than typed, so the slides cannot
drift from the artefacts; the figures come from the gallery, the overlays and
`figures/screening_funnel.png`, so run `cli gallery` and `cli overlay` if those
are stale.

Two things about it are deliberate and will look wrong if changed without
knowing why. The categorical colours are `0E9384, 7B3FA8, C4622D, 2563C9` **in
that order**, which passes a colour-vision-deficiency separation check that the
obvious teal/green/orange ordering fails. And every chart value is an integer,
RMSE in grams rather than kilograms and IoU in percent, because PowerPoint renders
a decimal separator in the *viewer's* locale, so on a machine set to a comma
convention `0.457` becomes `0,457` and fights the prose on the same slide. A
locale tag in the format code does not override it; only avoiding decimals does.
Exact values with their decimal points live in the captions.

## The two research proposals

Both build on their own, with no figure dependency:

```bash
node paper/proposal.js
```

```bash
node paper/proposal_phd.js
```

They share `proposal_common.js`, which holds the title page, the three signature
blocks, the 11-point single-spaced body style and the contact block, so the two
documents cannot drift apart on formatting or on the candidate's details.

### Contact details are not in this repository

It is public, so the student number, e-mail and telephone number live in
`paper/candidate.local.json`, which is gitignored and stays on the machine that
builds the document. `candidate.example.json` is the tracked shape and the
fallback: a fresh clone still builds both documents, with those four fields
showing as `[student number]` and so on rather than being silently wrong. To
build with the real values, copy the example and fill it in:

```bash
cp paper/candidate.example.json paper/candidate.local.json
```

The departmental fields, title, supervisors, address and research group, stay in
`proposal_common.js` in the open.

The department's template covers both degrees and differs in three places, which
is the whole difference between the two scripts:

| | MEng | PhD |
|---|---|---|
| summary page | 1 A4, 11 pt | same |
| anticipated articles | one, labelled `Description` | two, labelled `Research gap that will be addressed` |
| faculty requirement | one submitted | one accepted and one submitted |
| full proposal | 2 A4 | 10 A4 |

Current lengths, against those caps:

| | summary (cap 1 A4) | body |
|---|---|---|
| MEng | 0.83 | 1.98 of 2 A4 |
| PhD | 0.95 | 5.87 of 10 A4 |

Those numbers come from `paper/check_length.py`, which estimates rendered height
from the XML by counting characters against the 6.27 inch measure and adding
paragraph spacing, because this machine has no LibreOffice to render with:

```bash
python paper/check_length.py
```

It exits non-zero when a section is over its cap, so it works in a pre-send
check. It is an estimate and a pessimistic one, since justified text fits
slightly more than it predicts, so treat anything above about 0.95 as needing a
look in Word rather than as settled.

The MEng body has almost no slack, and the PhD summary has little. Any addition
to either needs a matching cut. The PhD body has room that is deliberately
unused: four more pages of padding would read worse than six that earn their
place.

### Why there are two

The Postgraduate Committee approved the 2025 MEng proposal with a reservation on
the record, that four hypotheses and three papers "would indicate a study with
the scope and complexity of a PhD". The MEng document answers that by narrowing
to what the evidence now supports. The PhD document takes the reservation at face
value and proposes the study at the size the committee already judged it to be,
keeping all four hypotheses and restating each with the measurement that decides
it.

## What has to be updated by hand

The prose. `paper.js` holds the text inline, and the tables are literals rather
than being read from the JSON reports. That is deliberate for a draft: a table
whose numbers are wired to disk cannot carry the sentence explaining why a
particular row is not resolved, and at this stage the explanation matters more
than the automation.

The consequence is that **the numbers in `paper.js` can drift from the pipeline**.
Two are worth checking against `cli baselines` and `cli fuse` before every send:

- the operator comparison in Table 3
- the plausibility counts in Table 2

## Before submitting

**Verify every reference.** Only Feng and Amaducci were read directly during
this work, in either document. The rest were written from working notes and their years and
page ranges have not been checked against the publisher record. The document says
so in a note under the reference list; remove that note only once the checking is
actually done.

**Render it and look at it.** This machine has neither LibreOffice nor pandoc, so
the draft was verified structurally rather than visually: XML well-formed, images
resolving, outline correct. That is not the same as having seen it.

**Decide on section 4.5.** The DINO probe and the baseline tables use different
feature preprocessing and are not comparable. The text says so rather than hiding
it, but cutting the section is also a reasonable call.
