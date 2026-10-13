/*
 * Shared furniture for the two proposal documents.
 *
 * The MEng and PhD proposals use the same departmental template, so the title
 * page, the signature blocks, the 11-point single-spaced body style and the
 * contact block all live here. Only the degree label, the article count and the
 * length of the body differ between them, and those are arguments.
 */

const {
  Paragraph, TextRun, HeadingLevel, AlignmentType, BorderStyle,
  LevelFormat, Footer, PageNumber,
} = require("docx");

const INK = "000000";
const MUTED = "444444";
const BODY = 22;            // 11 point, as the template stipulates
const FONT = "Times New Roman";

const p = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: o.line ?? 240 },   // single spacing
  alignment: o.align ?? AlignmentType.JUSTIFIED,
  children: [new TextRun({
    text, size: o.size ?? BODY, bold: o.bold, italics: o.italics,
    color: o.color ?? INK, font: FONT,
  })],
});

const rich = (parts, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, line: 240 },
  alignment: o.align ?? AlignmentType.JUSTIFIED,
  children: parts.map(([t, x = {}]) => new TextRun({
    text: t, size: x.size ?? BODY, bold: x.bold, italics: x.italics,
    color: x.color ?? INK, font: FONT,
  })),
});

const h = (text, level = 1) => new Paragraph({
  heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
  spacing: { before: level === 1 ? 200 : 140, after: 90 },
  children: [new TextRun({
    text, size: level === 1 ? 24 : 22, bold: true, color: INK, font: FONT,
  })],
});

const centre = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text, size: o.size ?? BODY, bold: o.bold, color: o.color ?? INK, font: FONT,
  })],
});

const bullets = (items) => items.map((t) => new Paragraph({
  numbering: { reference: "b", level: 0 },
  spacing: { after: 90, line: 240 },
  children: [new TextRun({ text: t, size: BODY, font: FONT })],
}));

/** A hypothesis or objective stated in bold, with its test in the same run. */
const claim = (label, statement, test) => rich([
  [label + " ", { bold: true }],
  [statement + " "],
  ["Decided by: ", { bold: true }],
  [test],
], { after: 110 });

const signature = (role) => [
  new Paragraph({
    spacing: { before: 340, after: 0 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: INK, space: 2 } },
    children: [new TextRun({ text: "", size: BODY })],
  }),
  new Paragraph({
    spacing: { after: 60 },
    children: [new TextRun({ text: "(Signature)", size: 18, font: FONT })],
  }),
  new Paragraph({
    spacing: { after: 180 },
    children: [
      new TextRun({ text: role, size: BODY, font: FONT }),
      new TextRun({ text: "\t\t\t\t\tDate", size: BODY, font: FONT }),
    ],
  }),
];

/* The candidate's details.
 *
 * This repository is public, so the personal half of the contact block is not
 * kept here. `candidate.local.json` holds the real student number, e-mail and
 * telephone number, it is gitignored, and it never leaves the machine that
 * builds the document. `candidate.example.json` is its tracked shape, and is
 * what the build falls back to when the local file is absent, so a clone still
 * produces a correct document with the personal fields left visibly blank
 * rather than silently wrong. The rest, being departmental, stays in the open.
 */
const fs = require("fs");
const path = require("path");

const loadIdentity = () => {
  for (const name of ["candidate.local.json", "candidate.example.json"]) {
    const file = path.join(__dirname, name);
    if (fs.existsSync(file)) {
      return JSON.parse(fs.readFileSync(file, "utf8"));
    }
  }
  throw new Error("no candidate.local.json or candidate.example.json in paper/");
};

const CANDIDATE = {
  ...loadIdentity(),
  title: "AUTOMATED BIOMASS ESTIMATION USING SELF-SUPERVISED VISION TRANSFORMERS",
  supervisor: "Prof Herman Myburgh",
  cosupervisors: ["Prof Allan De Freitas", "Dr Kealeboga Mokise"],
  address: "Department of Electrical, Electronic and Computer Engineering, "
    + "University of Pretoria, Private Bag X20, Hatfield, 0028, South Africa",
  group: "Smart Sensing and Intelligent Systems Group",
};

const titlePage = (degree, date) => [
  new Paragraph({
    spacing: { before: 900, after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text: CANDIDATE.title, size: 30, bold: true, font: FONT })],
  }),
  centre("Research proposal: " + degree, { after: 500, size: 24 }),
  centre("Candidate", { bold: true, after: 60 }),
  centre(CANDIDATE.name, { after: 240 }),
  centre("Student number", { bold: true, after: 60 }),
  centre(CANDIDATE.number, { after: 240 }),
  centre("Department of Electrical, Electronic and Computer Engineering", { after: 60 }),
  centre("University of Pretoria", { after: 240 }),
  centre(date, { after: 240 }),
  centre(degree === "PhD" ? "Promoter" : "Supervisor", { bold: true, after: 60 }),
  centre(CANDIDATE.supervisor, { after: 240 }),
  centre(degree === "PhD" ? "Co-promoters" : "Co-supervisors", { bold: true, after: 60 }),
  centre(CANDIDATE.cosupervisors[0], { after: 40 }),
  centre(CANDIDATE.cosupervisors[1], { after: 200 }),
  ...signature("Candidate"),
  ...signature("Supervisor/Promoter"),
  ...signature("Postgraduate Committee"),
];

const contactBlock = () => [
  h("Contact Information"),
  ...[
    ["Postal address", CANDIDATE.address],
    ["Research group", CANDIDATE.group],
    ["E-mail", CANDIDATE.email],
    ["Tel number", CANDIDATE.tel],
    ["Fax number", "n/a"],
    ["Cell number", CANDIDATE.tel],
  ].map(([k, v]) => rich([[k + ": ", { bold: true }], [v]],
    { align: AlignmentType.LEFT, after: 80 })),
];

/* Only Feng and Amaducci were read in full during this work. Saying so in the
 * document is cheaper than an examiner finding a wrong page range. */
const verifyNote = () => new Paragraph({
  spacing: { before: 200, after: 200 },
  border: { top: { style: BorderStyle.SINGLE, size: 6, color: "BBBBBB", space: 8 } },
  children: [new TextRun({
    text: "Note: bibliographic details should be verified against publisher "
      + "records before submission. Feng et al., Amaducci et al. and Malik et al. "
      + "were read in full during this work; the remainder were compiled from "
      + "working notes.",
    size: 18, italics: true, color: MUTED, font: FONT })],
});

const numbering = {
  config: [{
    reference: "b",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 500, hanging: 260 } } },
    }],
  }],
};

const pageFooter = () => new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
  })],
});

const styles = {
  default: { document: { run: { font: FONT, size: BODY } } },
};

const margins = { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } };

module.exports = {
  INK, MUTED, BODY, FONT, CANDIDATE,
  p, rich, h, centre, bullets, claim, signature,
  titlePage, contactBlock, verifyNote,
  numbering, pageFooter, styles, margins,
};
