/*
 * Shared document furniture for the generators in this directory.
 *
 * Extracted from paper.js when the results paper needed the same builders. One
 * definition rather than two, so a change to a table border or a caption style
 * cannot leave two documents from the same project looking unrelated.
 */

const fs = require("fs");
const {
  Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun,
} = require("docx");

const INK = "1A1A1A", MUTED = "5A5A5A", ACCENT = "31688E", HEAD = "440154";
const RULE = "D8D8D8", TINT = "F4F6F8";
const BODY = 21;                       // half-points: 10.5pt
const PAGE_W = 9026;                   // usable width in DXA for A4 + 1" margins

/* ---------- small builders ---------------------------------------------- */

const p = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: opts.line ?? 276 },
  alignment: opts.align ?? AlignmentType.JUSTIFIED,
  indent: opts.indent,
  children: [new TextRun({
    text, size: opts.size ?? BODY, color: opts.color ?? INK,
    italics: opts.italics, bold: opts.bold, font: opts.font ?? "Calibri",
  })],
});

/* Rich paragraph: array of [text, {bold,italics,...}] pairs. */
const rich = (parts, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: 276 },
  alignment: opts.align ?? AlignmentType.JUSTIFIED,
  children: parts.map(([t, o = {}]) => new TextRun({
    text: t, size: o.size ?? BODY, bold: o.bold, italics: o.italics,
    color: o.color ?? INK, font: o.font ?? "Calibri",
  })),
});

const h1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before: 320, after: 140 },
  children: [new TextRun({ text, size: 28, bold: true, color: HEAD, font: "Calibri" })],
});

const h2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before: 220, after: 100 },
  children: [new TextRun({ text, size: 23, bold: true, color: ACCENT, font: "Calibri" })],
});

/* Captions number themselves.

   They used to carry a hardcoded label, which is fine until a section is added
   in the middle: inserting the distance-metric table and the backbone table
   produced two Table 2s and two Table 6s, because every later caption still
   claimed the number it was written with. Passing "Table" or "Figure" and
   letting the counter assign the rest means a caption cannot disagree with its
   position. An explicit label ending in a digit is still honoured, so a document
   that wants to fix a number can. */
const CAPTION_N = { Table: 0, Figure: 0 };

const nextLabel = (label) => {
  const kind = String(label).trim().split(/\s+/)[0];
  if (!(kind in CAPTION_N)) return label;
  if (/\d/.test(label)) return label;          // caller fixed it deliberately
  CAPTION_N[kind] += 1;
  return `${kind} ${CAPTION_N[kind]}`;
};

const caption = (rawLabel, text) => ((label) => new Paragraph({
  spacing: { before: 60, after: 200 },
  alignment: AlignmentType.LEFT,
  children: [
    new TextRun({ text: label + ". ", size: 18, bold: true, color: INK, font: "Calibri" }),
    new TextRun({ text, size: 18, color: MUTED, font: "Calibri" }),
  ],
}))(nextLabel(rawLabel));

function figure(file, widthPx, label, text) {
  const buf = fs.readFileSync(file);
  const meta = pngSize(buf);
  const w = widthPx;
  const h = Math.round((meta.h / meta.w) * w);
  return [
    new Paragraph({
      spacing: { before: 160, after: 0 },
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({ data: buf, type: "png",
                                transformation: { width: w, height: h } })],
    }),
    caption(label, text),
  ];
}

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

/* Tables: dual widths, CLEAR shading, no percentage widths. */
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, { bold, bg, align, size } = {}) => new TableCell({
    width: { size: 0, type: WidthType.DXA },
    shading: bg ? { type: ShadingType.CLEAR, fill: bg, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      alignment: align ?? AlignmentType.LEFT,
      children: [new TextRun({ text: String(text), size: size ?? 18, bold,
                               color: INK, font: "Calibri" })],
    })],
  });

  const mk = (cells, o) => new TableRow({
    children: cells.map((c, i) => {
      const td = cell(c, { ...o, align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT });
      td.options.width = { size: widths[i], type: WidthType.DXA };
      return td;
    }),
  });

  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top:    { style: BorderStyle.SINGLE, size: 6, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE },
      left:   { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      right:  { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical:   { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
    },
    rows: [mk(header, { bold: true, bg: TINT }), ...rows.map((r) => mk(r, {}))],
  });
}

const bullets = (items) => items.map((t) => new Paragraph({
  numbering: { reference: "dots", level: 0 },
  spacing: { after: 80, line: 276 },
  children: [new TextRun({ text: t, size: BODY, color: INK, font: "Calibri" })],
}));


module.exports = {
  INK, MUTED, ACCENT, HEAD, RULE, TINT, BODY, PAGE_W,
  p, rich, h1, h2, caption, figure, pngSize, table, bullets,
};
