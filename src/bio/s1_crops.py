"""Bio stage 1: crop borders off every biography page.

Reads the classified biography pages (:func:`src.s2_classify_pages.load_bio_pages`),
trims the outer printed frame with :func:`src.s3_segment.trim_borders`, and -- on
the *first* bio page of each subgraph only -- trims the right-hand marker band (the
big vertical ``传禄房系``-style branch label plus the ``派``/``世`` generation-marker
columns). That band is printed only on a subgraph's opening page, so a blanket
right-trim would eat real bio content on the following pages; here we detect its
left boundary as the leftmost full-height vertical rule inside the right margin and
cut there.

The left title band (``武城曾氏重修族谱`` + page number) is not present on the pages
measured for Book 3's ``0_1`` subgraph -- ``trim_borders`` already removes the frame
and the left edge is bio content -- so no extra left trim is applied.

Output: ``books/{book}/bio/1_crops/{page}.png`` (binary ink grid, 0=ink/1=bg).
QA: ``books/{book}/qa/bio_s1/{page}.png`` -- a downscaled view of the frame-trimmed
page with a red box around the kept (cropped) region.

Horizontal rule lines are intentionally left in place; the merge/band-split stage
uses them to slice the 5 generation-bands.

Run:
    PYTHONPATH=. python -m src.bio.s1_crops --book book3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
from PIL import Image

from src.imaging import get_image, save_image
from src.s2_classify_pages import CLASSIFY_DIR, PAGE_TYPES_FILE, load_bio_pages

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
CROPS_DIR = "1_crops"

# The right marker band (vertical branch label + 派/世 columns) is printed on the
# FIRST page of each bio section and nowhere else. The book alternates graph run
# then bio section 1:1 (verified Book 3: 15 graph runs, 15 bio sections), so a
# section's first page is simply the first bio page following a graph page -- see
# section_first_pages(). We trim the band ONLY on those pages, so there is no need
# to detect band-vs-no-band per page.
#
# On a section-first page, the band lies right of the leftmost full-height vertical
# rule; the rule's column is the cut point (its exact position varies ~50px across
# pages, so we find it rather than fix a width). Search only the right margin.
RIGHT_BAND_SEARCH_PX = 1000
# Column ink-coverage for a full-height vertical rule. Rules run ~0.75-1.0 on clean
# scans but drop to ~0.36 where the ADF faded the line (page 66); 0.25 catches the
# faint case. Only ever evaluated on the 15 known band pages, so no false-positive
# risk from bio content.
RULE_COL_COVERAGE = 0.25

# Page frame geometry (verified Book 3; William 2026-09-17). Every side has a thin
# black rule ~x px in from the edge, running around the whole page. The THICK side
# additionally carries a ~y-px label band whose inner edge is another thin rule; the
# thick side is the LEFT on even pages and the RIGHT on odd pages. A section-start
# page also has a ~z-px marker band on the right (whitespace + two 派/世 columns).
#
# We do NOT hardcode these widths: on each side we find the frame rules (high-coverage
# lines near the edge) and cut just past the INNERMOST one -- so content is never
# clipped even as the exact widths drift. The reference values below are used only to
# SANITY-CHECK the detected trim per side and flag any page that deviates.
#
# The graph pipeline's src.s3_segment.trim_borders (fixed ~87/348px insets) is left
# untouched; it overshot by ~300px and clipped left-edge bio text (page 18's 广堂).
BORDER_SEARCH_PX = 400   # frame rules sit within this margin of each edge
BORDER_HI = 0.50         # a frame rule is a line >= this ink coverage
BORDER_LO = 0.30         # ... continued inward while coverage stays >= this
BORDER_MARGIN = 8        # px of air kept just inside the innermost rule

# Reference frame widths (px) for the sanity check, and tolerance.
REF_X = 40    # thin border inset (all sides)
REF_Y = 220   # label band, added to x on the thick side
REF_Z = 730   # marker band, added on the right of a section-start page
# A detected side trim must be within this of its expected width. Generous because
# the upstream 1_pages extraction may itself over/under-trim a page's edges.
FRAME_TOL = 100

QA_DIR = "qa"
QA_SUBDIR = "bio_s1"
QA_SCALE = 4  # downscale factor for QA previews


def section_first_pages(book: str, bio_pages: set[int],
                        books_dir: str = "books") -> set[int]:
    """Return the first bio page of each bio section.

    The book alternates graph run then bio section 1:1, so a bio section begins at
    the first bio page immediately following a graph page. Walking all pages in
    order and marking each bio page that follows a graph page yields exactly one
    first-page per section -- and those are precisely the pages carrying the right
    marker band.

    This is derived from page *ordering* (graph vs bio), not from the per-page
    ``follows_graph`` field, which is unreliable: pages whose rules the classifier
    missed get ``follows_graph=None`` and break the chain, making the *next* page
    look like a spurious new section.
    """
    path = os.path.join(books_dir, book, CLASSIFY_DIR, PAGE_TYPES_FILE)
    with open(path) as fh:
        info = json.load(fh)
    graph_pages = set(info["graph_pages"])

    firsts: set[int] = set()
    prev_was_graph = False
    for page in sorted(graph_pages | bio_pages):
        if page in graph_pages:
            prev_was_graph = True
        else:  # bio page
            if prev_was_graph:
                firsts.add(page)
            prev_was_graph = False
    return firsts


def _frame_lines(cov: np.ndarray) -> list[tuple[int, int]]:
    """Frame rules near one edge, as ``(start, end)`` runs, innermost last.

    ``cov`` is per-line ink coverage running inward FROM the edge. A rule is a run
    starting at coverage >= :data:`BORDER_HI`, continued while >= :data:`BORDER_LO`
    (hysteresis, so a thick line stays one run). Only the outer
    :data:`BORDER_SEARCH_PX` are scanned.
    """
    runs: list[tuple[int, int]] = []
    i = 0
    while i < BORDER_SEARCH_PX:
        if cov[i] >= BORDER_HI:
            start = i
            while i < len(cov) and cov[i] >= BORDER_LO:
                i += 1
            runs.append((start, i))
        else:
            i += 1
    return runs


def _thin_inset(cov: np.ndarray) -> int:
    """Inset past a THIN side's single frame rule (the ~x line).

    Cuts just past the innermost detected rule within the search window. Falls back
    to ``REF_X`` when no rule is found (a faded border on a near-empty page).
    """
    runs = _frame_lines(cov)
    if not runs:
        return REF_X
    return runs[-1][1] + BORDER_MARGIN


def _thick_inset(cov: np.ndarray) -> int:
    """Inset past a THICK side's label band -- just past the label's inner rule.

    The thick side is: x-rule, then the ~y-px label band, then the label's inner
    rule. We want to cut past that inner rule (at ~x+y). Detect it as the innermost
    rule that sits beyond the x-rule (i.e. past ~REF_X), searching a window wide
    enough to include x+y. The label's inner rule can fade below :data:`BORDER_HI`
    on ADF-degraded pages, so when no rule is found near x+y, fall back to the
    reference ``REF_X + REF_Y`` position (the sanity check will still flag a page
    whose true geometry differs).
    """
    runs = _frame_lines(cov)
    # The label-inner rule sits near x+y; require it well past the x double-line
    # (which can itself appear as a second run ~50-70px in). Anything closer is the
    # frame's own thin lines, not the label edge.
    label_min = REF_X + REF_Y // 2
    inner = [r for r in runs if r[0] > label_min]
    if inner:
        return inner[-1][1] + BORDER_MARGIN
    return REF_X + REF_Y


def _frame_box(a: np.ndarray, page: int) -> tuple[int, int, int, int]:
    """The outer-frame crop rectangle ``(top, bottom, left, right)`` for ``a``.

    Top and bottom are thin sides (~x). The thick side (label band, ~x+y) is the
    LEFT on even pages and the RIGHT on odd pages; the other horizontal side is thin
    (~x). Cuts are made just past each side's innermost frame rule so text is never
    clipped. Returned in ORIGINAL page coordinates; the crop is
    ``a[top:bottom, left:right]``. The marker band (z) is handled separately.
    """
    rows, cols = a.shape
    col_cov = np.sum(1 - a, axis=0) / rows   # per column (density down the page)
    row_cov = np.sum(1 - a, axis=1) / cols   # per row (density across the page)
    top = _thin_inset(row_cov)
    bottom = rows - _thin_inset(row_cov[::-1])
    if page % 2 == 0:  # even -> thick left, thin right
        left = _thick_inset(col_cov)
        right = cols - _thin_inset(col_cov[::-1])
    else:              # odd -> thin left, thick right
        left = _thin_inset(col_cov)
        right = cols - _thick_inset(col_cov[::-1])
    return top, bottom, left, right


def check_frame(box: tuple[int, int, int, int], shape: tuple[int, int],
                page: int, is_first: bool) -> list[str]:
    """Sanity-check a page's per-side trim against the x/y/z frame model.

    Expected trim widths: x on top/bottom; x+y on the thick side (left if page even,
    right if odd); x (+z if section-start) on the thin/right side. Returns a list of
    human-readable warnings for sides that deviate by more than :data:`FRAME_TOL`.
    """
    top, bottom, left, right = box
    rows, cols = shape
    even = page % 2 == 0
    exp = {
        "top": REF_X,
        "bottom": REF_X,
        "left": REF_X + (REF_Y if even else 0),
        "right": REF_X + (0 if even else REF_Y) + (REF_Z if is_first else 0),
    }
    got = {"top": top, "bottom": rows - bottom,
           "left": left, "right": cols - right}
    warns = []
    for side, e in exp.items():
        if abs(got[side] - e) > FRAME_TOL:
            warns.append(f"{side} trim {got[side]}px, expected ~{e}px")
    return warns


def right_band_cut(a: np.ndarray) -> int | None:
    """Column at which to cut off the right marker band, or ``None`` if absent.

    ``a`` is the frame-trimmed page. Scans the rightmost :data:`RIGHT_BAND_SEARCH_PX`
    columns for full-height vertical rules (>= :data:`RULE_COL_COVERAGE` ink). The
    marker band lies to the right of the *leftmost* such rule, so that rule's column
    is the cut point (the kept region is ``a[:, :cut]``). Returns ``None`` when no
    rule is found.
    """
    rows, cols = a.shape
    lo = max(0, cols - RIGHT_BAND_SEARCH_PX)
    cov = np.sum(1 - a[:, lo:], axis=0) / rows
    rule_cols = np.nonzero(cov >= RULE_COL_COVERAGE)[0]
    if rule_cols.size == 0:
        return None
    return lo + int(rule_cols.min())


def crop_box(a: np.ndarray, page: int, trim_right_band: bool
             ) -> tuple[int, int, int, int]:
    """Full crop rectangle ``(top, bottom, left, right)`` in ORIGINAL coordinates.

    Combines the outer-frame trim (parity-aware, :func:`_frame_box`) with the
    optional right marker-band trim. The band cut is measured on the frame-trimmed
    page, so it is offset back by ``left`` into original coordinates. Single source
    of truth for both the saved crop (:func:`crop_page`) and the QA overlay.
    """
    top, bottom, left, right = _frame_box(a, page)
    if trim_right_band:
        cut = right_band_cut(a[top:bottom, left:right])
        if cut is not None:
            right = left + cut
    return top, bottom, left, right


def crop_page(a: np.ndarray, page: int, trim_right_band: bool
              ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a page to its bio content. Returns the ink grid and its crop box."""
    top, bottom, left, right = crop_box(a, page, trim_right_band)
    return a[top:bottom, left:right], (top, bottom, left, right)


def _save_qa(a: np.ndarray, box: tuple[int, int, int, int], out_path: str) -> None:
    """Save a downscaled view of the ORIGINAL page with a red kept-region box.

    ``a`` is the untouched ``1_pages`` image and ``box`` is the crop rectangle in
    its coordinates, so the overlay shows everything that was trimmed -- the outer
    frame and (on section-first pages) the right marker band.
    """
    top, bottom, left, right = box
    rows, cols = a.shape
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((cols // QA_SCALE, rows // QA_SCALE))
    px = img.load()
    assert px is not None
    x0, y0 = left // QA_SCALE, top // QA_SCALE
    x1, y1 = right // QA_SCALE - 1, bottom // QA_SCALE - 1
    red = (255, 0, 0)
    for t in range(3):  # 3px-thick box
        for x in range(x0, x1 + 1):
            for y in (y0 + t, y1 - t):
                if 0 <= x < img.width and 0 <= y < img.height:
                    px[x, y] = red
        for y in range(y0, y1 + 1):
            for x in (x0 + t, x1 - t):
                if 0 <= x < img.width and 0 <= y < img.height:
                    px[x, y] = red
    img.save(out_path)


def crop_book(book: str, books_dir: str = "books", qa: bool = True) -> int:
    """Crop all biography pages of ``book``. Returns the number of pages written."""
    bio_pages = load_bio_pages(book, books_dir=books_dir)
    if not bio_pages:
        logger.warning("no biography pages for %s", book)
        return 0
    first_pages = section_first_pages(book, bio_pages, books_dir=books_dir)

    pages_dir = os.path.join(books_dir, book, "1_pages")
    out_dir = os.path.join(books_dir, book, BIO_DIR, CROPS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    n = 0
    flagged = 0
    for page in sorted(bio_pages):
        src_path = os.path.join(pages_dir, f"{page}.png")
        if not os.path.exists(src_path):
            logger.warning("missing page image %s", src_path)
            continue
        a = get_image(src_path)
        is_first = page in first_pages
        cropped, box = crop_page(a, page, trim_right_band=is_first)
        save_image(cropped, os.path.join(out_dir, f"{page}.png"))
        if qa:
            _save_qa(a, box, os.path.join(qa_dir, f"{page}.png"))
        warns = check_frame(box, (a.shape[0], a.shape[1]), page, is_first)
        if warns:
            flagged += 1
            logger.warning("page %d: frame sanity check failed: %s",
                           page, "; ".join(warns))
        logger.info("page %d: %s  (%s)", page,
                    "x".join(map(str, cropped.shape[::-1])),
                    "section-first" if is_first else "body")
        n += 1
    logger.info("wrote %d cropped bio pages to %s (%d flagged by sanity check)",
                n, out_dir, flagged)
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.s3_segment import BOOK_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, choices=sorted(BOOK_CONFIGS))
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--no-qa", action="store_true", help="skip QA previews")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    crop_book(args.book, books_dir=args.books_dir, qa=not args.no_qa)


if __name__ == "__main__":
    main()
