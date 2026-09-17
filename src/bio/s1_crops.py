"""Bio stage 1: crop the printed frame off every biography page.

For each classified biography page (:func:`src.s2_classify_pages.load_bio_pages`) we
detect the page frame by run-length line detection and cut just inside the innermost
rule on each side, then -- on a bio section's first page -- also trim the right marker
band. See the module functions for the frame geometry (x/y/z model) and detection.

Output: ``books/{book}/bio/1_crops/{page}.png`` (binary ink grid, 0=ink/1=bg).
QA: ``books/{book}/qa/bio_s1/{page}.png`` -- the ORIGINAL page downscaled with a red
box around the kept (cropped) region.

Horizontal rule lines are intentionally left in place; the merge/band-split stage
uses them to slice the 5 generation-bands.

KNOWN DATA GAP -- Book 4 page 214 (printed pg 215): this page starts the 庆炆房系
branch (庆炆 -> 繁炉 -> 祥亮/祥成) and carries its own right marker band, but that
branch's TREE-GRAPH page is missing from the parsed graphs -- graph 210_210 is 庆炎's
subtree only, and there is no 庆炆 graph between graphs 206/210/215. Printed page
numbers are contiguous, so the 庆炆 graph was dropped/unparsed upstream (graph
pipeline), not physically missing. Because it has no graph page, section_first_pages()
does not treat p214 as a section start, so its marker band would be left in the crop.
We SKIP p214 here (see ``SKIP_PAGES``); resolve the missing 庆炆 graph upstream later.

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

# Bio pages to skip entirely (not cropped) -- pages whose branch has no parsed tree
# graph, so they cannot be section-classified or attached downstream. See the module
# docstring's KNOWN DATA GAP note. Keyed by book.
SKIP_PAGES: dict[str, set[int]] = {
    "book4": {214},  # 庆炆房系 branch start; its tree-graph page is missing upstream
}

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
# We do NOT hardcode these widths: on each side we DETECT the frame rules and cut
# just past the relevant one, so content is never clipped as widths drift. The
# reference values below are used only to SANITY-CHECK the detected trim per side.
#
# A frame rule is a full-length straight line. We detect it by its LONGEST CONTINUOUS
# INK RUN along the line, NOT by total ink density: an ADF-faded line (or one crossed
# by the horizontal cell rules) can have only ~40% total coverage yet still contain a
# single unbroken 2000px+ run, while the densest CONTENT column tops out ~200px. So a
# run-length test separates lines from text with a ~10x margin, where a density
# threshold missed faded lines (Book 3 p30/p60 label rules at 0.40 coverage). Small
# gaps (breaks, crossing rules) are bridged by LINE_GAP.
#
# The graph pipeline's src.s3_segment.trim_borders (fixed ~87/348px insets) is left
# untouched; it overshot by ~300px and clipped left-edge bio text (page 18's 广堂).
BORDER_SEARCH_PX = 400   # frame rules sit within this margin of each edge
LINE_MIN_FRAC = 0.20     # a line's longest run must span >= this fraction of the axis
LINE_GAP = 40            # px of gap bridged within one run (ADF breaks, crossing rules)
BORDER_MARGIN = 8        # px of air kept just inside the detected rule

# Reference frame widths (px) for the sanity check, and tolerance.
REF_X = 40    # thin border inset (all sides)
REF_Y = 220   # label band, added to x on the thick side
REF_Z = 730   # marker band, added on the right of a section-start page
# A detected side trim must be within this of its expected width. Generous because
# the upstream 1_pages extraction may itself over/under-trim a page's edges.
FRAME_TOL = 100

# Post-crop invariant: after a correct crop no FRAME LINE remains in the outer band.
# We flag a leftover line, not mere ink -- legitimate bio text can run right up to a
# correctly-cut frame (near-empty pages), so a plain ink-density test false-positives.
# A leftover frame line is unmistakable by its continuous run (>= this fraction of the
# perpendicular dimension); text in the band tops out far below that.
EDGE_LINE_COLS = 40       # width of the outer band checked on each side
EDGE_LINE_FRAC = 0.50     # a run this fraction of the axis in the band = leftover line

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


def _longest_run(mask: np.ndarray, gap: int = LINE_GAP) -> int:
    """Longest run of ``True`` in ``mask``, bridging gaps of up to ``gap`` False."""
    best = cur = 0
    misses = gap + 1
    for v in mask:
        if v:
            if misses <= gap:
                cur += misses  # count the bridged gap as part of the run
            cur += 1
            misses = 0
            if cur > best:
                best = cur
        else:
            misses += 1
            if misses > gap:
                cur = 0
    return best


def _line_positions(a: np.ndarray, axis: int) -> np.ndarray:
    """Indices (along ``axis``) of full-length straight lines in ``a``.

    ``axis==1`` finds vertical lines (indexed by column, each spanning the height);
    ``axis==0`` finds horizontal lines (indexed by row, each spanning the width). A
    line is present when its longest continuous ink run spans at least
    :data:`LINE_MIN_FRAC` of its length. Returns the sorted line indices.
    """
    n = a.shape[axis]                    # number of candidate lines
    span = a.shape[1 - axis]             # length each line runs
    min_run = int(LINE_MIN_FRAC * span)
    hits = []
    for i in range(n):
        line = (1 - (a[:, i] if axis == 1 else a[i, :])).astype(bool)
        if _longest_run(line) >= min_run:
            hits.append(i)
    return np.array(hits, dtype=int)


def _line_groups(positions: np.ndarray, tol: int = 6) -> list[tuple[int, int]]:
    """Collapse adjacent line indices into rule spans ``(start, end)`` (end inclusive)."""
    if positions.size == 0:
        return []
    groups: list[list[int]] = [[int(positions[0])]]
    for p in positions[1:]:
        if int(p) - groups[-1][-1] <= tol:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    return [(g[0], g[-1]) for g in groups]


def _side_rules(a: np.ndarray, side: str, win: int) -> list[tuple[int, int]]:
    """Frame-rule spans within ``win`` px of one ``side``, in ABSOLUTE coordinates.

    ``(start, end)`` runs ordered from the edge inward.
    """
    rows, cols = a.shape
    if side == "left":
        return _line_groups(_line_positions(a[:, :win], axis=1))
    if side == "right":
        base = cols - win
        return [(base + s, base + e)
                for s, e in _line_groups(_line_positions(a[:, -win:], axis=1))]
    if side == "top":
        return _line_groups(_line_positions(a[:win, :], axis=0))
    base = rows - win
    return [(base + s, base + e)
            for s, e in _line_groups(_line_positions(a[-win:, :], axis=0))]


def _cut(a: np.ndarray, side: str, fallback: int, win: int) -> int:
    """Crop coordinate for ``side``: just inside its INNERMOST frame rule.

    The correct crop line is the innermost detected rule span within ``win`` of the
    edge -- the ~x thin rule on a thin side, the label's inner rule (~x+y) on a thick
    side, or (on a near-empty page where the ADF smeared the border and x-rule into
    one blob) that single span. ``win`` is capped per side by the caller so a marker-
    band rule (only on the right of a section-start page, handled separately) is never
    mistaken for the frame. We cut at that span's CONTENT-facing edge plus a small
    margin, so no rule stays in the crop and text is never clipped. Falls back to a
    thin ``fallback`` inset when no rule is detected at all.

    Returns an absolute row/col: a low coord for left/top sides, high for right/bottom.
    """
    rows, cols = a.shape
    spans = _side_rules(a, side, win)
    if not spans:
        if side in ("left", "top"):
            return fallback
        return (cols if side == "right" else rows) - fallback
    if side in ("left", "top"):
        return spans[-1][1] + BORDER_MARGIN            # inner edge of innermost span
    return spans[0][0] - BORDER_MARGIN                 # right/bottom: innermost = first


def _frame_box(a: np.ndarray, page: int) -> tuple[int, int, int, int]:
    """The outer-frame crop rectangle ``(top, bottom, left, right)`` for ``a``.

    Each side is cut just inside its innermost frame rule (:func:`_cut`), found by
    run-length line detection -- so the label band on the thick side is removed and
    text is never clipped. The per-side search window is capped to that side's maximum
    frame width so a marker-band rule is never mistaken for the frame: a thin side
    caps at x, a thick side at x+y. The marker side is the RIGHT of section-start
    pages, where the band (z, further in) is trimmed separately in :func:`crop_box`;
    its frame width is x (even) or x+y (odd). Parity feeds only these window caps
    (and the sanity check), not the cut rule itself. Returned in ORIGINAL coordinates.
    """
    thin = REF_X + FRAME_TOL              # covers a thin (x) side
    thick = REF_X + REF_Y + FRAME_TOL     # covers a thick (x+y) side
    even = page % 2 == 0
    left_win = thick if even else thin
    right_win = thin if even else thick
    top = _cut(a, "top", REF_X, thin)
    bottom = _cut(a, "bottom", REF_X, thin)
    left = _cut(a, "left", REF_X, left_win)
    right = _cut(a, "right", REF_X, right_win)
    return top, bottom, left, right


def check_frame(box: tuple[int, int, int, int], cropped: np.ndarray,
                orig_shape: tuple[int, int], page: int, is_first: bool) -> list[str]:
    """Sanity-check a page's crop. Returns human-readable warnings (empty if clean).

    Two independent checks:
    1. Per-side trim vs the x/y/z frame model -- x on top/bottom; x+y on the thick
       side (left if page even, right if odd); x (+z if section-start) on the right --
       flagging any side off by more than :data:`FRAME_TOL`.
    2. Post-crop invariant -- no FRAME LINE remains in the outer
       :data:`EDGE_LINE_COLS` band of the CROPPED image (a continuous run >=
       :data:`EDGE_LINE_FRAC` of the axis). Legitimate bio text may sit in that band,
       so this checks for a leftover line, not mere ink.
    """
    top, bottom, left, right = box
    rows, cols = orig_shape
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

    crows, ccols = cropped.shape
    k = EDGE_LINE_COLS
    for side in ("left", "right", "top", "bottom"):
        if side in ("left", "right"):
            band = cropped[:, :k] if side == "left" else cropped[:, -k:]
            axis_len, line_axis = crows, 1
        else:
            band = cropped[:k, :] if side == "top" else cropped[-k:, :]
            axis_len, line_axis = ccols, 0
        pos = _line_positions_thresh(band, line_axis, int(EDGE_LINE_FRAC * axis_len))
        if pos:
            warns.append(f"{side} edge: frame line left in crop")
    return warns


def _line_positions_thresh(a: np.ndarray, axis: int, min_run: int) -> bool:
    """True if any line along ``axis`` in ``a`` has an ink run >= ``min_run``."""
    n = a.shape[axis]
    for i in range(n):
        line = (1 - (a[:, i] if axis == 1 else a[i, :])).astype(bool)
        if _longest_run(line) >= min_run:
            return True
    return False


def right_band_cut(a: np.ndarray) -> int | None:
    """Column at which to cut off the right marker band, or ``None`` if absent.

    ``a`` is the frame-trimmed page. The marker band's 派/世 columns are bounded by
    full-height vertical rules; its LEFT boundary is the leftmost such rule within
    :data:`RIGHT_BAND_SEARCH_PX` of the right edge, and that is the cut point (kept
    region ``a[:, :cut]``). Rules are found by run-length (the band's outer rule can
    fade / bound empty cells, so a density test missed it and cut the band short --
    Book 4 p227/p229/p241); returns ``None`` when no rule is found.
    """
    cols = a.shape[1]
    lo = max(0, cols - RIGHT_BAND_SEARCH_PX)
    spans = _line_groups(_line_positions(a[:, lo:], axis=1))
    if not spans:
        return None
    return lo + spans[0][0]  # leftmost rule's left edge = band's left boundary


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
        sub_w = right - left
        detected = right_band_cut(a[top:bottom, left:right])
        # The marker band's outer rule can be too broken to detect when its 派/世 cells
        # are empty (Book 4 p229/p282). We KNOW a section-first page has the band, so
        # if detection finds nothing or cuts it short of the reference z, fall back to
        # z from the right edge (the sanity check's tolerance then confirms).
        if detected is not None and sub_w - detected >= REF_Z - FRAME_TOL:
            cut = detected
        else:
            cut = sub_w - REF_Z
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
    skip = SKIP_PAGES.get(book, set())

    pages_dir = os.path.join(books_dir, book, "1_pages")
    out_dir = os.path.join(books_dir, book, BIO_DIR, CROPS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    n = 0
    flagged = 0
    for page in sorted(bio_pages):
        if page in skip:
            logger.info("page %d: SKIPPED (known data gap; see SKIP_PAGES)", page)
            continue
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
        warns = check_frame(box, cropped, (a.shape[0], a.shape[1]), page, is_first)
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
