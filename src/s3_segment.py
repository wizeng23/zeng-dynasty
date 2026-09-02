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


BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(num_pages=17),
    "book2": BookConfig(num_pages=134),
    "book3": BookConfig(num_pages=292),
    "book4": BookConfig(num_pages=316),
}


def trim_borders(a: np.ndarray) -> np.ndarray:
    """Trim the page's printed border frame off all four sides.

    The top and bottom insets are always border. For the left/right border,
    whichever side carries more ink in its outer band is the framed side and gets
    the larger trim; the other side gets the smaller. Scaled from v0's
    30px/120px insets.
    """
    rows, cols = a.shape
    small = _s(30)
    big = _s(120)
    a = a[small : rows - small, :]
    col_present = np.sum(1 - a, axis=0)
    if np.sum(col_present[:big]) > np.sum(col_present[-big:]):
        return a[:, big : cols - small]
    return a[:, small : cols - big]


def shrink_page(a: np.ndarray, keep_left: int | None = None) -> np.ndarray:
    """Shrink a trimmed page to the tightest box around its line-graph.

    Finds the densest ink column, grows left and right while ink is present, pads,
    then trims trailing blank rows off the bottom. Leaves the top intact so the
    graph's first generation-row stays aligned. Pad scaled from v0's 10px.
    """
    rows, cols = a.shape
    pad = _s(10)

    col_present = np.sum(1 - a, axis=0)
    min_col = int(np.argmax(col_present))
    max_col = min_col
    while min_col > 0 and col_present[min_col]:
        min_col -= 1
    while max_col < cols - 1 and col_present[max_col]:
        max_col += 1
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


def is_tree_start_page(a: np.ndarray, config: BookConfig) -> int:
    """Return the label's left x if this page starts a subtree, else -1.

    A start page carries a thin vertical text label near the right edge. The
    rightmost ink column must be about one character wide, in the rightmost
    fifth-ish of the page yet not touching the edge, and a short vertical run.
    """
    col_sums = np.sum(1 - a, axis=0)
    width = col_sums.shape[0]
    inked = col_sums > COL_INK_MIN  # ignore speckle columns (no island removal in v1)

    end = width - 1
    while end > 0 and not inked[end]:
        end -= 1
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
        return -1
    start_x = start

    # Vertical extent of the label block, measured over its own columns. Ignore
    # rows with only speckle ink (same reason as COL_INK_MIN): a stray speck far
    # below the label would otherwise stretch vspan to nearly the page height.
    row_sums = np.sum(1 - a[:, start:end], axis=1)
    row_inked = row_sums > COL_INK_MIN
    vend = row_sums.shape[0] - 1
    while vend > 0 and not row_inked[vend]:
        vend -= 1
    vstart = 0
    while vstart < vend and not row_inked[vstart]:
        vstart += 1
    logger.debug("label vertical extent: vstart=%d vend=%d", vstart, vend)
    vspan = vend - vstart
    if not (
        config.label_min_vstart <= vstart < config.label_max_vstart
        and config.label_min_vend < vend < config.label_max_vend
        and config.label_min_vspan < vspan < config.label_max_vspan
    ):
        return -1
    return start_x


# Per-page left-crop overrides (scaled from v0 book2 geometry if reused later).
# shrink_page normally crops to the ink contiguous with the densest column, which
# drops a sparse node group sitting left of the main tree across a blank gap.
CROP_KEEP_LEFT: dict[str, dict[int, int]] = {}


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
        a = trim_borders(a)
        tree_start_x = is_tree_start_page(a, config)
        if tree_start_x != -1:
            logger.info("Page %d starts a subtree (label at x=%d)", i, tree_start_x)
            a = a[:, :tree_start_x]
        starts[str(i)] = tree_start_x != -1
        keep_left = CROP_KEEP_LEFT.get(book, {}).get(i)
        a = shrink_page(a, keep_left=keep_left)
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
