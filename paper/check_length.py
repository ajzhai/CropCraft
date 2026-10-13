"""Estimate how many A4 pages each section of a generated .docx will occupy.

The department's template caps the summary page at one A4 and the body at two
for a master's or ten for a PhD, and those caps are enforced by whoever reads
the document rather than by anything in the toolchain. This machine has neither
LibreOffice nor pandoc, so the documents cannot be rendered here to check.

This estimates instead, from the XML. It is deliberately crude and deliberately
pessimistic: characters divided by a measured characters-per-line figure, rounded
up per paragraph, plus each paragraph's own before and after spacing converted
from twentieths of a point into line heights. Justified text and hyphenation both
fit slightly more than this predicts, so an estimate under the cap is safe and an
estimate over it is worth checking in Word rather than trusting.

    python paper/check_length.py
    python paper/check_length.py some_other.docx --cap 10
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from pathlib import Path

# A4 is 11.69 in tall and 8.27 wide; the template's margins are one inch.
USABLE_HEIGHT_IN = 11.69 - 2.0
LINE_HEIGHT_IN = 0.176          # 11 pt Times at single spacing
CHARS_PER_LINE = 95             # measured for 11 pt Times in a 6.27 in measure
CHARS_PER_LINE_BULLET = 83      # bullets lose the indent and the hanging marker

PARAGRAPH = re.compile(r"<w:p>.*?</w:p>|<w:p [^>]*>.*?</w:p>", re.DOTALL)
TEXT = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)
BEFORE = re.compile(r'w:before="(\d+)"')
AFTER = re.compile(r'w:after="(\d+)"')

SECTIONS = ["title page", "summary page", "full proposal", "references + contact"]


def _spacing_lines(match: re.Match | None) -> float:
    """Twentieths of a point to line heights; 240 twips is one single-spaced line."""
    return int(match.group(1)) / 240.0 if match else 0.0


def section_lines(xml: str) -> float:
    total = 0.0
    for para in PARAGRAPH.findall(xml):
        text = "".join(TEXT.findall(para))
        if not text.strip():
            continue
        per_line = CHARS_PER_LINE_BULLET if "numPr" in para else CHARS_PER_LINE
        total += max(1, math.ceil(len(text) / per_line))
        total += _spacing_lines(BEFORE.search(para))
        total += _spacing_lines(AFTER.search(para))
    return total


def report(path: Path, body_cap: int) -> bool:
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    parts = xml.split('<w:br w:type="page"/>')

    print(f"\n{path.name}")
    ok = True
    for index, part in enumerate(parts):
        pages = section_lines(part) * LINE_HEIGHT_IN / USABLE_HEIGHT_IN
        name = SECTIONS[index] if index < len(SECTIONS) else f"section {index}"

        cap = {"summary page": 1, "full proposal": body_cap}.get(name)
        if cap is None:
            print(f"  {name:22s} {pages:5.2f} A4")
            continue

        verdict = "OK" if pages <= cap else "OVER"
        ok &= pages <= cap
        print(f"  {name:22s} {pages:5.2f} A4   cap {cap}   {verdict}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--cap", type=int, default=None,
                        help="body page cap; inferred from the filename otherwise")
    args = parser.parse_args()

    here = Path(__file__).parent
    files = args.files or sorted(here.glob("Masuba_research_proposal*.docx"))
    if not files:
        print("nothing to check; build a proposal first", file=sys.stderr)
        return 1

    ok = True
    for path in files:
        cap = args.cap if args.cap else (10 if "PhD" in path.name else 2)
        ok &= report(path, cap)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
