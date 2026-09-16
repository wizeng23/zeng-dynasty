"""Stage 3 (v1 scans): frame-cropped pages -> per-page tree crops.

Restarted from the v0 (glass-scan) crop stage, with every pixel constant scaled up
for the v1 native resolution. v0 ran on a fixed 1300x1950 normalized canvas; the
v1 pages are ~3789x5740 (see ``src.extract_pages``), so pixel thresholds are
scaled by :data:`SCALE` (~2.9x).

This step CROPS each page down to just its tree line-graph and records which
pages start a new subtree. Merging the per-page crops of a multi-page subtree
into one graph is the *next* step (:mod:`src.merge_pages`); splitting them keeps
this intermediate artifact (the per-page crop) reviewable on its own.

Per page:

1. ``trim_borders`` cuts the printed frame off all four sides by fixed insets.
   (v1 crops each page *to* the frame, so the frame lines sit at the very edge;
   the scaled insets clear them into clean margin.)
2. ``is_tree_start_page`` finds the thin vertical text label near the right edge
   (``X公房系世系图``); its x is where the label + everything right is cropped off.
   A page with no label is a continuation of the previous subtree.
3. ``shrink_page`` crops to the tree's bounding box.

Output: ``books/{book}/crops/{i}.png`` (one tree-crop per page) plus
``books/{book}/crops/starts.json`` mapping each page index to whether it
starts a subtree -- the interface :mod:`src.merge_pages` reads.

CLI:
    python -m src.segment --book book1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os

import numpy as np

from src.imaging import get_image, save_image

logger = logging.getLogger(__name__)

# Pixel-constant scale factor from the v0 1300x1950 canvas to the v1 native page
# (~3789x5740). Measured 2.92x (width) / 2.94x (height); 2.9 is the shared value
# every v0 threshold below is multiplied by.
SCALE = 2.9


def _s(px: int) -> int:
    """Scale a v0 pixel constant to v1 native resolution."""
    return int(round(px * SCALE))


# Minimum ink pixels in a column for it to count as "inked" in label detection.
# v0 ran remove_small_islands first; the v1 pipeline dropped that, so faint
# speckle survives and a bare ">0" test would latch the rightmost-run walk onto a
# 1-3px speck instead of the label column. A column of real label ink has
# hundreds of pixels, so this threshold ignores speckle without touching the
# label. (Measured: label columns ~100s of px; speckle <=~10.)
COL_INK_MIN = _s(7)


# Native page width after Stage 1 (see extract_pages). Ratios below are
# calibrated against this; label detection uses the trimmed width it is handed.
PAGE_WIDTH = _s(1300)


@dataclasses.dataclass(frozen=True)
class BookConfig:
    """Per-book Stage-3 configuration (pixel bounds are v1-native, scaled from v0).

    The label geometry is structural: the label is one character column wide,
    sits in the rightmost sliver of the page, and does not run to the very edge
    (a graph line would). Only span and proportional position generalize; the
    absolute column varies page-to-page.
    """

    num_pages: int
    label_min_span: int = _s(40)
    label_max_span: int = _s(90)
    label_min_end_ratio: float = 0.75
    label_edge_margin: int = _s(5)
    label_min_vstart: int = _s(0)
    label_max_vstart: int = _s(200)
    label_min_vend: int = _s(250)
    label_max_vend: int = _s(650)
    label_min_vspan: int = _s(200)
    label_max_vspan: int = _s(600)
    # Whiten the top / bottom margin (clears smear above / the page-number strip
    # below the graph). Both OFF for Book 3:
    #  - bottom: Book 3's long leaf names (龙岗保幼殁, 庆荣幼殁, 庆连编继) hang to the
    #    inner border, so any bottom band clips them (page 15 lost 91px, 80 lost 179px).
    #  - top: the top band is full-width, so on a page whose tree starts high it wipes
    #    the top generation-bar -- page 83's 纪培 fan-out bar sits at row 0, inside the
    #    250px band; blanking it broke shrink_page's left-walk and CROPPED OUT the whole
    #    广锦/广锽/昭油/宪柳/宪柽/庆发/庆亮/庆烙 subtree (never reached the merged graph).
    whiten_top: bool = True
    whiten_bottom: bool = True
    # When > 0, shrink_page keeps the full span between the leftmost and rightmost
    # column with >= this many ink px, instead of only the subtree contiguous with
    # the densest column. Book 4 has pages with several subtrees split by wide
    # whitespace (36/164/254) where the densest-column grow drops whole subtrees.
    # 0 (off) for Books 1-3 keeps their crops unchanged.
    shrink_full_span_ink: int = 0


BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(num_pages=17),
    "book2": BookConfig(num_pages=134),
    "book3": BookConfig(num_pages=292, whiten_top=False, whiten_bottom=False),
    "book4": BookConfig(
        num_pages=316, whiten_top=False, whiten_bottom=False, shrink_full_span_ink=_s(17),
    ),
}


# Top-border detection (see _top_border_cut). A row is part of a border band when
# its ink spans at least this fraction of the page width; bands are the printed
# horizontal frame lines. The v1 ADF scans smeared ink, so some pages carry extra
# whitespace (or a faint phantom line) above the true top border -- a fixed inset
# would leave it, misaligning the graph top across pages at the merge seam. So the
# top is cut past the *bottommost* border band instead of at a fixed offset.
BORDER_ROW_COVERAGE = 0.5
BORDER_SEARCH_ROWS = _s(120)   # look for border bands within the top ~350px


def _top_border_cut(a: np.ndarray) -> int:
    """Row to cut the top at: just past the bottommost full-width border band.

    Scans the top ``BORDER_SEARCH_ROWS`` for bands of rows whose ink covers
    >=``BORDER_ROW_COVERAGE`` of the width (the printed frame lines: a thick outer
    line, then a thin inner line). Returns the row just below the lowest band, so
    both border lines and any smear/whitespace above them are removed. Falls back
    to the fixed ``_s(30)`` inset when no band is found.
    """
    cols = a.shape[1]
    cov = np.sum(1 - a[:BORDER_SEARCH_ROWS], axis=1) / cols
    black = cov >= BORDER_ROW_COVERAGE
    last_band_end = -1
    r = 0
    while r < len(black):
        if black[r]:
            while r < len(black) and black[r]:
                r += 1
            last_band_end = r - 1
        else:
            r += 1
    return last_band_end + 1 if last_band_end >= 0 else _s(30)


def trim_borders(a: np.ndarray) -> np.ndarray:
    """Trim the page's printed border frame off all four sides.

    The bottom inset is always border. The top is cut past the bottommost border
    band (:func:`_top_border_cut`) so the graph top is standardized regardless of
    smear-induced whitespace above the border. For the left/right border, whichever
    side carries more ink in its outer band is the framed side and gets the larger
    trim; the other side gets the smaller. Insets scaled from v0's 30px/120px.
    """
    rows, cols = a.shape
    small = _s(30)
    big = _s(120)
    a = a[_top_border_cut(a) : rows - small, :]
    col_present = np.sum(1 - a, axis=0)
    if np.sum(col_present[:big]) > np.sum(col_present[-big:]):
        return a[:, big : cols - small]
    return a[:, small : cols - big]


# trim_borders leaves the page's TOP edge just past the top inner border and its
# BOTTOM edge at the bottom inner border (the frame is removed). Measured from the
# inner border, the graph sits inside a fixed margin: it starts ~250px below the top
# inner border and ends ~200px above the bottom inner border. Whitening exactly that
# margin (anchored to the trimmed edges = the inner borders) wipes any residual
# smear, page-number text, or faint phantom line there -- which would otherwise
# corrupt shrink_page's column/row analysis (a stray full-width smear row makes the
# horizontal grow-walk latch across the whole width) -- without ever touching the
# graph. These are measured v1-native pixel margins, so they are NOT re-scaled.
# (An earlier fixed ~300px bottom slab, and a density-thresholded adaptive version,
# both cut into low-hanging 3-char leaf names like page 13's 信九郎/宪七郎; anchoring
# to the inner border at the true 200px margin is what fixes it.)
WHITEN_TOP = 250     # px below the top inner border (graph starts here)
WHITEN_BOTTOM = 200  # px above the bottom inner border (graph ends here)

# The bottom whitening is BETTER anchored to the *detected* inner border than to
# the trimmed edge: trim_borders cuts the bottom at a fixed rows-_s(30) inset,
# which lands ~34px ABOVE the inner border, so the fixed WHITEN_BOTTOM only reaches
# ~234px above the border -- 8-12px clearance from the lowest 3-char leaf names
# (Book 2 p42/p96), which then lost their last char. Whitening from
# WHITEN_BOTTOM_FROM_BORDER px above the DETECTED border instead is a fixed,
# border-relative band that clears the page-number/smear strip while leaving the
# names ~60px of air. Lowest Book 2 glyph ends ~246px above the border -> 66px.
WHITEN_BOTTOM_FROM_BORDER = 180


def bottom_inner_border_row(a: np.ndarray) -> int:
    """Row (in the UNTRIMMED page) of the top of the bottom inner border line.

    The printed frame at the page bottom is two full-width bands: a thick outer
    line flush with the bottom edge, and a thinner inner line ~40px above it. We
    want the inner one -- the graph's true bottom boundary. Scans the bottom
    ``BORDER_SEARCH_ROWS`` for full-width bands (>=``BORDER_ROW_COVERAGE`` ink),
    bottom-up, and returns the TOP row of the SECOND band found (the inner line).
    Falls back to the single band's top if only one is found, or to
    ``rows - _s(30)`` (the old trim inset) if none is.
    """
    rows, cols = a.shape
    lo = max(0, rows - BORDER_SEARCH_ROWS)
    cov = np.sum(1 - a[lo:], axis=1) / cols
    black = cov >= BORDER_ROW_COVERAGE
    band_tops: list[int] = []          # top row (absolute) of each band, bottom-up
    r = len(black) - 1
    while r >= 0:
        if black[r]:
            while r >= 0 and black[r]:
                r -= 1
            band_tops.append(lo + r + 1)   # r+1 is the first black row of the band
        else:
            r -= 1
    if len(band_tops) >= 2:
        return band_tops[1]            # the inner line (2nd band up from the bottom)
    if band_tops:
        return band_tops[0]
    return rows - _s(30)


def whiten_margins(
    a: np.ndarray, bottom_anchor: int | None = None,
    whiten_top: bool = True, whiten_bottom: bool = True,
) -> np.ndarray:
    """Blank the graph-free margin inside each inner border (top & bottom).

    ``a`` is a border-trimmed page, so its top edge is the top inner border. The
    graph lives ``WHITEN_TOP`` px below the top; whitening that clears top smear.

    For the bottom: when ``bottom_anchor`` is given (the inner-border row expressed
    in ``a``'s trimmed coordinates, i.e. ``bottom_inner_border_row(page) -
    _top_border_cut(page)``), whiten every row at/below ``bottom_anchor -
    WHITEN_BOTTOM_FROM_BORDER`` -- a border-relative band that clears the
    page-number strip without cutting into low-hanging names. When omitted, fall
    back to the old fixed ``WHITEN_BOTTOM`` px from the trimmed bottom edge.

    ``whiten_top=False`` skips the top band: the full-width band wipes the top
    generation-bar on a page whose tree starts high (Book 3 page 83). ``whiten_bottom
    =False`` skips the bottom band: for a book whose names hang to the inner border
    (Book 3) any band clips them. With both off, the border trim alone bounds the graph.
    """
    a = a.copy()
    if whiten_top:
        a[:WHITEN_TOP, :] = 1
    if not whiten_bottom:
        return a
    if bottom_anchor is not None:
        start = max(0, bottom_anchor - WHITEN_BOTTOM_FROM_BORDER)
        a[start:, :] = 1
    else:
        a[-WHITEN_BOTTOM:, :] = 1
    return a


def _grow(present: np.ndarray, start: int, step: int, gap: int) -> int:
    """Walk from ``start`` in ``step`` direction while ink is present, jumping a
    blank run of <= ``gap`` columns when ink resumes just beyond it.

    Returns the boundary column: with ``gap == 0`` this exactly reproduces the old
    ``while 0 <= i and present[i]: i += step`` walk (it lands ON the bounding blank
    column, or clamps at the array edge), so the caller's ``- pad`` behaves
    identically. With ``gap > 0`` a break of up to ``gap`` blanks is stepped over.
    """
    n = len(present)
    i = start
    while 0 <= i < n and present[i]:
        i += step
    # i is now on the first blank (or off the end). Try to step over short gaps.
    while 0 <= i < n:
        j = i
        blanks = 0
        while 0 <= j < n and not present[j] and blanks < gap:
            j += step
            blanks += 1
        if 0 <= j < n and present[j]:
            i = j
            while 0 <= i < n and present[i]:
                i += step
        else:
            break
    return i


def shrink_page(
    a: np.ndarray, keep_left: int | None = None, col_gap: int = 0,
    full_span_ink: int = 0,
) -> np.ndarray:
    """Shrink a trimmed page to the tightest box around its line-graph.

    Finds the densest ink column, grows left and right while ink is present, pads,
    then trims trailing blank rows off the bottom. Leaves the top intact so the
    graph's first generation-row stays aligned. Pad scaled from v0's 10px.

    ``col_gap`` (default 0 = the strict walk, byte-identical to the old code) lets
    the grow step over a blank run of up to that many columns -- for a hairline break
    in a connecting bar. Left 0 in practice: a break that severs a real left subtree
    (Book 3 page 80) is handled by a targeted ``CROP_KEEP_LEFT`` entry instead, so we
    don't risk walking a low ``col_gap`` across ADF smear into margin (pages 65/280
    have only faint smear to the left, no real subtree).

    ``full_span_ink`` (default 0 = off): when > 0, keep the full horizontal span
    from the leftmost to the rightmost column with at least this many ink px --
    everything between real content is genuine tree. Needed for a page with several
    subtrees separated by wide whitespace: the densest-column grow keeps only the
    subtree it starts in and drops the rest (Book 4 pages 36/164/254 lost whole
    right-side subtrees). The threshold excludes faint ADF smear (which is a few
    px/col), so it never over-grows into margin. The right edge is already bounded
    by the label strip. Off for Books 1-3 to keep their frozen crops unchanged.
    """
    rows, cols = a.shape
    pad = _s(10)

    col_present = np.sum(1 - a, axis=0)
    if full_span_ink > 0:
        inked = np.flatnonzero(col_present >= full_span_ink)
        if inked.size:
            min_col, max_col = int(inked[0]), int(inked[-1])
        else:  # no real ink -- fall back to the densest-column grow
            min_col = max_col = int(np.argmax(col_present))
    else:
        min_col = int(np.argmax(col_present))
        max_col = min_col
        min_col = _grow(col_present, min_col, -1, col_gap)
        max_col = _grow(col_present, max_col, +1, col_gap)
    min_col = max(min_col - pad, 0)
    max_col = min(max_col + pad, cols - 1)
    if keep_left is not None:
        min_col = min(min_col, max(keep_left - pad, 0))
    a = a[:, min_col : max_col + 1]

    row_present = np.sum(1 - a, axis=1)
    max_row = row_present.shape[0] - 1
    while max_row > 0 and not row_present[max_row]:
        max_row -= 1
    max_row = min(max_row + pad, rows - 1)
    return a[:max_row, :]


# Blank-row gap (px) that separates the label's contiguous character block from a
# detached ADF ink smear below it. Real inter-character gaps within the label are
# ~tens of px; a smear sits >1000px below, so this only ever splits label-vs-smear.
LABEL_ROW_GAP = _s(40)


def _largest_ink_run(row_inked: np.ndarray, gap: int) -> tuple[int, int]:
    """(vstart, vend) of the inked-row run carrying the most ink.

    Groups inked rows into runs, bridging blank gaps smaller than ``gap``, and
    returns the run with the most inked rows -- the label body, isolated from a
    detached smear far below it. Returns (0, 0) when no row is inked.
    """
    idx = np.flatnonzero(row_inked)
    if idx.size == 0:
        return 0, 0
    # Split where consecutive inked rows are >= gap apart.
    breaks = np.flatnonzero(np.diff(idx) >= gap)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    best = max(range(len(starts)), key=lambda k: row_inked[starts[k] : ends[k] + 1].sum())
    return int(starts[best]), int(ends[best])


def is_tree_start_page(a: np.ndarray, config: BookConfig) -> int:
    """Return the label's left x if this page starts a subtree, else -1.

    A start page carries a thin vertical text label near the right edge. The
    rightmost ink column must be about one character wide, in the rightmost
    fifth-ish of the page yet not touching the edge, and a short vertical run.
    """
    col_sums = np.sum(1 - a, axis=0)
    width = col_sums.shape[0]
    inked = col_sums > COL_INK_MIN  # ignore speckle columns (no island removal in v1)

    # Walk inked column-runs from the right edge leftward, testing each against the
    # label shape. The label is the rightmost run that IS a label -- but it is not
    # always the *outermost* inked run: an ADF speck or a leftover sliver of the
    # right border frame can sit to the right of the true label (Book 4 pages
    # 120/285/287/293: a 1px speck, or a border remnant with end_ratio=1.0 /
    # right_margin=0). Latching onto that first run and returning -1 misclassifies a
    # real start page as a continuation. Instead, skip a run that fails the
    # horizontal label test and continue to the next run left of it, so a stray
    # right-edge mark no longer masks the label behind it.
    end = width - 1
    while end > 0:
        while end > 0 and not inked[end]:
            end -= 1
        if end <= 0:
            break
        start = end
        while start > 0 and inked[start]:
            start -= 1
        logger.debug("label column candidate: start=%d end=%d width=%d", start, end, width)

        span = end - start
        end_ratio = end / width if width else 0.0
        right_margin = width - 1 - end
        if not (
            config.label_min_span <= span <= config.label_max_span
            and end_ratio >= config.label_min_end_ratio
            and right_margin >= config.label_edge_margin
        ):
            end = start - 1  # this run isn't the label; try the next one to its left
            continue
        start_x = start

        # Vertical extent of the label block, measured over its own columns. The
        # label is one *contiguous* vertical run of characters; take that block, not
        # the outermost inked rows. An ADF ink smear can drop a few rows of
        # >COL_INK_MIN ink far below the label (Book 3 pages 64/78/134/217/247/261:
        # smear rows near y=2200-3200 with 20-60 ink px), which the plain
        # outermost-row walk reads as the label bottom and inflates vend/vspan past
        # label_max_vend. Splitting the inked rows into runs at blank gaps >=
        # LABEL_ROW_GAP and keeping the run with the most ink isolates the label
        # body from a detached smear (the gap is >1000px; inter-character gaps are
        # ~tens of px).
        row_sums = np.sum(1 - a[:, start:end], axis=1)
        row_inked = row_sums > COL_INK_MIN
        vstart, vend = _largest_ink_run(row_inked, LABEL_ROW_GAP)
        logger.debug("label vertical extent: vstart=%d vend=%d", vstart, vend)
        vspan = vend - vstart
        if (
            config.label_min_vstart <= vstart < config.label_max_vstart
            and config.label_min_vend < vend < config.label_max_vend
            and config.label_min_vspan < vspan < config.label_max_vspan
        ):
            return start_x
        end = start - 1  # right shape ok but vertical extent wrong; keep looking left
    return -1


# Per-page left-crop overrides: keep ink out to this column even if shrink_page's
# grow would stop short. shrink_page crops to the ink contiguous with the densest
# column, so a break in the top connecting bar drops the left subtree beyond it.
# Book 3 page 80: a 4px bar break severs 广铨's whole left subtree (纪壎→广铨→昭溦/
# 昭淮/昭汉→宪模→庆煊); its real ink (tall strokes, >100px/col) starts at col 285,
# while faint ADF smear reaches col 1 -- so keep to 285, not further (pages 65/280
# looked cut but their extra-left is only smear, max ~35px/col, correctly excluded).
CROP_KEEP_LEFT: dict[str, dict[int, int]] = {
    "book3": {80: 285},
}


# Per-page subtree-start overrides: {book: {page: label_left_x}}. is_tree_start_page
# occasionally can't classify a real start page -- Book 4 page 287's label
# (庆祺房系世系图) merges into one ~571px-wide run, past label_max_span, so the
# detector rejects it. Forcing the start here (crop off the label + everything right
# of ``label_left_x``) is cheaper and safer than loosening the detector for one page.
FORCE_START: dict[str, dict[int, int]] = {
    "book4": {287: 2522},
}


def segment(
    book: str,
    books_dir: str = "books",
    config: BookConfig | None = None,
    pages_dir: str = "1_pages",
    crops_dir_name: str = "3_crops",
) -> list[str]:
    """Crop each page to its tree line-graph; record which pages start a subtree.

    Reads ``{books_dir}/{book}/{pages_dir}/{i}.png`` for ``i`` in
    ``0..num_pages-1``, trims the frame, crops off the label (and everything right
    of it) when the page starts a subtree, and shrinks to the tree's bounding box.
    Writes one crop per page to
    ``{books_dir}/{book}/{crops_dir_name}/{i}.png`` and a ``starts.json`` in that
    directory mapping page index -> whether it starts a subtree.

    Merging the crops of a multi-page subtree into one graph is done separately by
    :mod:`src.merge_pages`, which reads ``starts.json``.

    Biography pages (Books 3 & 4 interleave them; see :mod:`src.s2_classify_pages`,
    the preceding Stage 2) are skipped -- they are not tree pages, so they get no
    crop and no ``starts.json`` entry. If no ``page_types.json`` sidecar exists
    (Books 1 & 2, which are entirely tree pages), every page is processed as before.
    """
    if config is None:
        if book not in BOOK_CONFIGS:
            raise KeyError(
                f"no BookConfig for {book!r}; known books: {sorted(BOOK_CONFIGS)}"
            )
        config = BOOK_CONFIGS[book]

    # Lazy import avoids a circular dependency (classify_pages imports from here).
    from src.s2_classify_pages import load_bio_pages
    bio_pages = load_bio_pages(book, books_dir=books_dir)

    in_dir = os.path.join(books_dir, book, pages_dir)
    crops_dir = os.path.join(books_dir, book, crops_dir_name)
    os.makedirs(crops_dir, exist_ok=True)

    logger.info("Cropping %s: %d pages in %s -> %s (%d biography pages skipped)",
                book, config.num_pages, in_dir, crops_dir, len(bio_pages))

    written: list[str] = []
    starts: dict[str, bool] = {}
    for i in range(config.num_pages):
        if i in bio_pages:
            continue
        filepath = os.path.join(in_dir, f"{i}.png")
        a = get_image(filepath)
        anchor = bottom_inner_border_row(a) - _top_border_cut(a)
        a = trim_borders(a)
        tree_start_x = FORCE_START.get(book, {}).get(i, is_tree_start_page(a, config))
        if tree_start_x != -1:
            logger.info("Page %d starts a subtree (label at x=%d)", i, tree_start_x)
            a = a[:, :tree_start_x]
        starts[str(i)] = tree_start_x != -1
        a = whiten_margins(a, bottom_anchor=anchor,
                           whiten_top=config.whiten_top, whiten_bottom=config.whiten_bottom)
        keep_left = CROP_KEEP_LEFT.get(book, {}).get(i)
        a = shrink_page(a, keep_left=keep_left,
                        full_span_ink=config.shrink_full_span_ink)
        out_path = os.path.join(crops_dir, f"{i}.png")
        save_image(a, out_path)
        written.append(out_path)

    with open(os.path.join(crops_dir, "starts.json"), "w") as fh:
        json.dump(starts, fh, indent=2)

    n_starts = sum(starts.values())
    logger.info("Cropped %d pages for %s (%d subtree-start pages)",
                len(written), book, n_starts)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, choices=sorted(BOOK_CONFIGS))
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--pages-dir", default="1_pages")
    parser.add_argument("--crops-dir", default="3_crops")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    written = segment(args.book, books_dir=args.books_dir,
                      pages_dir=args.pages_dir, crops_dir_name=args.crops_dir)
    logger.info("Done. %d page crops written for %s.", len(written), args.book)


if __name__ == "__main__":
    main()
