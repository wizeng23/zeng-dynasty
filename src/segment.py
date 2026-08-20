"""Stage 2 of the pipeline: normalized pages -> merged subtree-graph images.

Each page in ``books/bookN/pages/{i}.png`` is a single deskewed page whose
line-graph is one horizontal slice of a subtree. A subtree's graph runs across
several consecutive pages, so this stage:

1. Trims the page border and shrinks to the graph's bounding box.
2. Detects whether the page *starts* a new subtree by looking for the thin
   vertical text label near the right edge (e.g. 衍公房系世系图 = "lineage chart
   of the 衍 house"). Continuation pages have no such label.
3. Stitches each run of continuation pages onto the page that started the
   subtree, aligning the dangling lines at the seam, and writes one image per
   subtree to ``books/bookN/graphs/{start}_{end}.png`` (page indices).

Physically stitching the page images at the seam lets the Stage-3 parser treat
each subtree as one connected graph, with no cross-page node bookkeeping (see
``docs/specs/2026-08-18-pipeline-rewrite-design.md``).

The book reads right-to-left; ``merge_graphs`` places the newer (left) page on
the left of the accumulated graph, matching that order.

CLI:
    python -m src.segment --book book1
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from itertools import combinations

import numpy as np

from src.imaging import get_image, save_image

logger = logging.getLogger(__name__)

# Normalized page width after Stage 1 (see extract_pages.PAGE_WIDTH). The
# trimmed working width is a little smaller; label detection uses the trimmed
# width it is handed, not this constant, but the ratios below are calibrated
# against this fixed canvas.
PAGE_WIDTH = 1300


@dataclasses.dataclass(frozen=True)
class BookConfig:
    """Per-book Stage-2 configuration.

    The subtree-label geometry is deliberately *structural* rather than a set
    of absolute pixel offsets: the label is one character column wide, sits in
    the rightmost sliver of the fixed-width page, and does not run to the very
    edge (a graph line would). The design spec shows the absolute column
    position varies page-to-page even within a single book, so only span and
    proportional position generalize.

    Attributes:
        num_pages: Number of pages ``0..num_pages-1`` in ``pages/`` to process.
        label_min_span: Minimum width (px) of the label's character column.
        label_max_span: Maximum width (px) of the label's character column.
        label_min_end_ratio: The label's right edge must sit at least this far
            across the trimmed page (``end / width``), i.e. in the rightmost
            ``1 - ratio`` fraction of the page.
        label_edge_margin: The label must leave at least this many background
            columns to its right; a graph line that reaches the page edge does
            not, which is how a continuation page is told from a start page.
        label_min_vstart / label_max_vstart: Bounds on the top of the label's
            vertical extent.
        label_min_vend / label_max_vend: Bounds on the bottom of the label's
            vertical extent.
        label_min_vspan / label_max_vspan: Bounds on the label's height.
    """

    num_pages: int
    label_min_span: int = 40
    label_max_span: int = 90
    label_min_end_ratio: float = 0.75
    label_edge_margin: int = 5
    label_min_vstart: int = 0
    label_max_vstart: int = 200
    label_min_vend: int = 250
    label_max_vend: int = 650
    label_min_vspan: int = 200
    label_max_vspan: int = 600


# Book 1's Stage-1 output is pages -1..16; the old notebook processed indices
# 0..16 (range(17)), skipping the -1 half page.
#
# Book 2 keeps every structural default. Its trimmed page width (1150) matches
# Book 1's, and measuring all 135 pages shows the label geometry lands squarely
# inside the shared bounds (label span 55-58px, end/width 0.83-0.92, right
# margin 95-192px, vertical extent vstart~75-82 / vend~488-509 / vspan~412-428),
# well separated from continuation pages whose graph line reaches the edge
# (span ~1149, right margin 0). Only ``num_pages`` differs, confirming the
# design spec's thesis that the label thresholds are structural, not per-book.
BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(num_pages=17),
    "book2": BookConfig(num_pages=135),
}


def trim_borders(a: np.ndarray) -> np.ndarray:
    """Trim the page's printed border frame off all four sides.

    The top and bottom 30px are always border. For the left/right border,
    whichever side carries more ink in its outer 120px is the framed side and
    gets 120px trimmed; the other side gets 30px.

    Args:
        a: Binary ink grid of a normalized page (0 == ink, 1 == background).

    Returns:
        The border-trimmed grid.
    """
    rows, cols = a.shape
    a = a[30 : rows - 30, :]
    col_present = np.sum(1 - a, axis=0)
    if np.sum(col_present[:120]) > np.sum(col_present[-120:]):
        return a[:, 120 : cols - 30]
    return a[:, 30 : cols - 120]


def shrink_page(a: np.ndarray) -> np.ndarray:
    """Shrink a trimmed page to the tightest box around its line-graph.

    Finds the densest ink column, grows left and right while ink is present,
    pads 10px, then trims trailing blank rows off the bottom (also +10px pad).
    Leaves the top intact so the graph's first generation-row stays aligned.

    Args:
        a: Border-trimmed binary ink grid.

    Returns:
        The cropped grid containing just the graph.
    """
    rows, cols = a.shape

    # Cut left/right around the densest column.
    col_present = np.sum(1 - a, axis=0)
    min_col = int(np.argmax(col_present))
    max_col = min_col
    while min_col > 0 and col_present[min_col]:
        min_col -= 1
    while max_col < cols - 1 and col_present[max_col]:
        max_col += 1
    min_col = max(min_col - 10, 0)
    max_col = min(max_col + 10, cols - 1)
    a = a[:, min_col : max_col + 1]

    # Cut bottom.
    row_present = np.sum(1 - a, axis=1)
    max_row = row_present.shape[0] - 1
    while max_row > 0 and not row_present[max_row]:
        max_row -= 1
    max_row = min(max_row + 10, rows - 1)
    return a[:max_row, :]


def is_tree_start_page(a: np.ndarray, config: BookConfig) -> int:
    """Return the label's left x-coordinate if this page starts a subtree, else -1.

    A subtree-start page carries a thin vertical text label (the house's
    lineage-chart title) near the right edge. Detection is structural, not
    positional (see the design spec): the rightmost ink column must be

    * about one character wide (``label_min_span..label_max_span``),
    * in the rightmost fifth-ish of the trimmed page (``end/width`` >=
      ``label_min_end_ratio``) yet not touching the page edge (a graph line
      would — this is what distinguishes a continuation page), and
    * a short vertical run consistent with a stacked-character title.

    The returned x-coordinate is used by the caller to crop the label column
    (and everything to its right) off before the page is merged into a graph.

    Args:
        a: Border-trimmed binary ink grid of the page.
        config: The book's Stage-2 config (label geometry bounds).

    Returns:
        The label's left x-coordinate to crop at, or ``-1`` if not a start page.
    """
    col_sums = np.sum(1 - a, axis=0)
    width = col_sums.shape[0]

    # Rightmost ink column (walk left from the edge over blank columns).
    end = width - 1
    while end > 0 and col_sums[end] == 0:
        end -= 1
    # Left extent of that rightmost run of ink columns.
    start = end
    while start > 0 and col_sums[start] > 0:
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

    # The label is a short vertical run of stacked characters, not the full
    # graph height.
    row_sums = np.sum(1 - a[:, start:end], axis=1)
    vend = row_sums.shape[0] - 1
    while vend > 0 and row_sums[vend] == 0:
        vend -= 1
    vstart = 0
    while vstart < vend and row_sums[vstart] == 0:
        vstart += 1
    logger.debug("label vertical extent: vstart=%d vend=%d", vstart, vend)
    vspan = vend - vstart
    if not (
        config.label_min_vstart < vstart < config.label_max_vstart
        and config.label_min_vend < vend < config.label_max_vend
        and config.label_min_vspan < vspan < config.label_max_vspan
    ):
        return -1
    return start_x


def remove_adjacent(numbers: list[int], threshold: int = 30) -> list[int]:
    """Collapse near-adjacent values, keeping the first of each cluster.

    Line endpoints detected column-wise arrive as thick runs of neighboring
    rows; this reduces each run to a single representative. Example with
    ``threshold`` gaps: ``{4, 6, 7, 8}`` -> ``[4, 7]``... only values more than
    ``threshold`` apart from the last kept value survive.

    Args:
        numbers: Candidate endpoint row indices.
        threshold: Minimum gap between kept values.

    Returns:
        The thinned list of representative indices.
    """
    if not numbers:
        return []

    sorted_nums = sorted(numbers)
    result = [sorted_nums[0]]
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i - 1] > threshold:
            result.append(sorted_nums[i])
    return result


# Weight of the shift-magnitude tie-breaker in find_best_orphans' scoring. Small
# enough that a clear shape signal (a tight offset cluster, even at a large Δ) still
# wins on its own, but nonzero so the degenerate single-pair case -- where offset
# spread is always zero -- resolves toward the physically nearer line rather than
# an arbitrary combinations() order. Calibrated against both books' seams.
SHIFT_PENALTY_WEIGHT = 0.1


def find_best_orphans(left: list[int], right: list[int]) -> tuple[list[int], str]:
    """Choose which endpoints on the longer side have no partner at the seam.

    When the two page seams expose unequal numbers of dangling lines, the extra
    lines on the longer side are "orphans" (a line that starts on this page and
    has no continuation across the seam). Picks the orphan set that minimizes
    the summed |offset| between the remaining matched endpoints.

    Scoring balances two signals that together pin down the right pairing across
    both books' geometries:

    * **Shape agreement (shift-invariant).** Correctly matched lines share one
      whole-page vertical offset ``Δ``, so their per-pair offsets cluster tightly;
      a wrong pairing scatters them. Scoring the spread *around the mean offset*
      (not the absolute rows) lets an honest whole-page shift cost nothing -- vital
      because consecutive pages can sit a full generation-row apart (Book 1's tree
      descends across pages, so p15's lines are ~100px below p14's).
    * **Shift magnitude (mild penalty).** Spread alone is degenerate when only one
      pair survives (a single offset always has zero spread), so every candidate
      ties. Physically adjacent pages are at most about one generation-row apart,
      so a pairing implying a huge ``|Δ|`` is wrong: Book 1 p16 must match its lone
      line to the *nearer* accumulated line (Δ≈186), not the far one (Δ≈559); Book
      2 p13's lone line must match Δ≈12, not the top-align's old Δ≈841. A small
      per-px penalty on ``|median Δ|`` breaks the tie toward the physically sane
      pairing without overriding a clear shape signal.

    Args:
        left: Endpoint rows on the left graph's right edge.
        right: Endpoint rows on the right graph's left edge.

    Returns:
        ``(orphan_indices, side)`` where ``side`` names which of the two lists
        (``"left"`` or ``"right"``) the orphan indices index into.
    """
    flip = False
    if len(right) > len(left):
        flip = True
        left, right = right, left
    num_orphans = len(left) - len(right)
    best_orphans: tuple[int, ...] = ()
    best_orphan_score: float | None = None

    for orphans in combinations(range(len(left)), num_orphans):
        new_left = [left[i] for i in range(len(left)) if i not in orphans]
        if new_left:
            offsets = [y - x for x, y in zip(new_left, right)]
            mean_off = sum(offsets) / len(offsets)
            spread = sum(abs(o - mean_off) for o in offsets)
            # Mild tie-breaker toward the smaller whole-page shift. The weight is
            # small so a clear shape signal (a tight cluster at a large Δ) still
            # wins, but it decisively separates the single-pair case where spread
            # is always zero.
            shift_penalty = SHIFT_PENALTY_WEIGHT * abs(mean_off)
            score = spread + shift_penalty
        else:
            score = 0.0
        if best_orphan_score is None or score < best_orphan_score:
            best_orphan_score = score
            best_orphans = orphans
    return list(best_orphans), "left" if not flip else "right"


def matched_shift(left: list[int], right: list[int]) -> int:
    """The vertical shift to apply to ``left`` so its endpoints meet ``right``.

    ``left`` and ``right`` are the *matched* seam endpoints (equal length, orphans
    already removed), paired positionally. Returns the median pairwise offset --
    the whole-page vertical shift that best lines the two pages up. The median (not
    the mean) resists a single noisy endpoint. A blank input yields ``0``.

    This replaces the old two-step "top-align to left_y[0]==right_y[0] then refine
    within +-100px". Top-aligning on the topmost pair silently assumed that pair
    was the same line; when a missing-line page exposed a different top endpoint it
    force-matched the wrong lines and padded the whole page hundreds of px off its
    grid (Book 2's 11_17: p13 shoved ~841px, cascading down every later page).
    Matching first (:func:`find_best_orphans`, shift-invariant) then shifting by the
    matched pairs' median offset aligns by lines that truly correspond, so a
    missing-line page just yields orphans instead of dragging the merge off-grid.

    Args:
        left: Matched endpoint rows on the left graph's right edge.
        right: Matched endpoint rows on the right graph's left edge.

    Returns:
        The vertical shift to apply to ``left`` (may be negative); ``0`` if empty.
    """
    if not left or not right:
        return 0
    offsets = sorted(y - x for x, y in zip(left, right))
    mid = len(offsets) // 2
    if len(offsets) % 2:
        return offsets[mid]
    # Even count: average the two middle offsets, rounding toward zero.
    return int((offsets[mid - 1] + offsets[mid]) / 2)


def find_best_alignment(left: list[int], right: list[int]) -> int:
    """Deprecated: superseded by :func:`matched_shift` (kept for reference/tests).

    Returned the shift in ``[-100, 100)`` minimizing the summed |offset| between
    positionally-paired endpoints. :func:`merge_graphs` now derives the shift from
    the matched pairs' median offset instead (see :func:`matched_shift`).
    """
    best_alignment = 0
    best_alignment_score: int | None = None
    for i in range(-100, 100):
        new_left = [x + i for x in left]
        score = sum(abs(x - y) for x, y in zip(new_left, right))
        if best_alignment_score is None or score < best_alignment_score:
            best_alignment_score = score
            best_alignment = i
    return best_alignment


def _concat_top_aligned(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Stack two graph slices side by side (``g1`` left) with no seam line.

    Top-aligns the two slices, pads the shorter one's bottom so heights match,
    horizontally concatenates, and pads 100px top and bottom to match
    :func:`merge_graphs`'s framing. Used when a seam exposes no endpoint on one
    side, so there is no dangling line to connect across it.

    Args:
        g1: Left graph slice.
        g2: Right graph slice.

    Returns:
        The concatenated graph.
    """
    if g1.shape[0] < g2.shape[0]:
        padding = g2.shape[0] - g1.shape[0]
        g1 = np.vstack([g1, np.ones((padding, g1.shape[1])).astype(np.uint8)])
    elif g2.shape[0] < g1.shape[0]:
        padding = g1.shape[0] - g2.shape[0]
        g2 = np.vstack([g2, np.ones((padding, g2.shape[1])).astype(np.uint8)])
    g = np.hstack([g1, g2])
    return np.vstack(
        [
            np.ones((100, g.shape[1])).astype(np.uint8),
            g,
            np.ones((100, g.shape[1])).astype(np.uint8),
        ]
    )


def merge_graphs(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Stitch two subtree-graph slices side by side, ``g1`` on the left.

    Aligns the dangling lines exposed at ``g1``'s right edge with those at
    ``g2``'s left edge (accounting for orphans and an overall vertical shift),
    draws the connecting seam, horizontally concatenates, and pads 100px top and
    bottom.

    Args:
        g1: Left graph slice (the newer page, since the book reads RTL).
        g2: Right graph slice (the accumulated graph so far).

    Returns:
        The merged graph.
    """
    g1_edge = np.sum(1 - g1[:, -5:], axis=1)
    left_y = remove_adjacent([x for x in range(len(g1_edge)) if g1_edge[x] > 0])
    g2_edge = np.sum(1 - g2[:, :5], axis=1)
    right_y = remove_adjacent([x for x in range(len(g2_edge)) if g2_edge[x] > 0])
    logger.debug("seam endpoints: left=%s right=%s", left_y, right_y)

    # A seam needs a dangling line exposed on *both* sides to connect. When a
    # line runs off a page edge without reaching it on the neighbour (Book 2's
    # scans have this — a parent line stops short of the page border while its
    # child continues on the next page), one side exposes no endpoint. There is
    # nothing to align or draw, so stack the slices top-aligned and let Stage 3's
    # connected-component parser keep the fragments separate. Book 1's scans
    # never hit this, so its output is unaffected.
    if not left_y or not right_y:
        logger.warning(
            "empty seam (left=%d right=%d endpoints); concatenating without a "
            "connecting line",
            len(left_y),
            len(right_y),
        )
        return _concat_top_aligned(g1, g2)

    # Match endpoints across the seam on their NATURAL rows (no prior top-align).
    # find_best_orphans is shift-invariant, so it identifies which lines truly
    # correspond even when the whole page sits a generation-row higher/lower, and
    # drops the extras on the longer side as orphans.
    orphan_idxs, side = find_best_orphans(left_y, right_y)
    logger.debug("seam endpoints (natural rows): left=%s right=%s", left_y, right_y)
    if side == "left":
        orphans = [left_y[i] for i in orphan_idxs]
        left_y = [left_y[i] for i in range(len(left_y)) if i not in orphan_idxs]
    else:
        orphans = [right_y[i] for i in orphan_idxs]
        right_y = [right_y[i] for i in range(len(right_y)) if i not in orphan_idxs]
    if orphan_idxs:
        logger.warning("orphan endpoints %s on %s side of seam", orphans, side)

    # Shift g1 by the matched pairs' median offset so the corresponding lines meet.
    # This single, match-derived shift replaces the old top-align + best_alignment:
    # aligning by lines that truly correspond means a missing-line page (which now
    # just contributes orphans) can no longer drag the whole page off its grid.
    #
    # Realize the shift by top-padding ONLY the side whose ink must slide down, then
    # bottom-pad whichever is now shorter to square the heights. Crucially we do NOT
    # also bottom-pad the *other* side by the shift: that redundant pad added the
    # shift to the canvas height every merge, so across a wide multi-page graph the
    # per-seam generation offsets accumulated into a runaway-tall image (Book 2's
    # 36_52 ballooned 5k -> 17k px). Top-pad-then-square keeps height at
    # max(h1+shift, h2), not max(h1, h2) + shift.
    best_alignment = matched_shift(left_y, right_y)
    logger.debug("matched-pair shift: %d", best_alignment)
    if best_alignment > 0:
        # g1's ink slides down by best_alignment; its endpoints move with it.
        g1 = np.vstack([np.ones((best_alignment, g1.shape[1])).astype(np.uint8), g1])
        left_y = [x + best_alignment for x in left_y]
    elif best_alignment < 0:
        # g2's ink slides down by -best_alignment; its endpoints move with it.
        g2 = np.vstack([np.ones((-best_alignment, g2.shape[1])).astype(np.uint8), g2])
        right_y = [x - best_alignment for x in right_y]

    # Square the heights by bottom-padding the shorter side (never moves ink).
    if g1.shape[0] < g2.shape[0]:
        padding = g2.shape[0] - g1.shape[0]
        g1 = np.vstack([g1, np.ones((padding, g1.shape[1])).astype(np.uint8)])
    elif g2.shape[0] < g1.shape[0]:
        padding = g1.shape[0] - g2.shape[0]
        g2 = np.vstack([g2, np.ones((padding, g2.shape[1])).astype(np.uint8)])

    # Bridge each matched pair. The endpoints now coincide with their real lines,
    # so a short vertical run in the two seam columns joins them cleanly. To avoid
    # a stub poking beyond both lines (which would read as a spurious node top),
    # ramp the row across the two seam columns instead of stacking a tall pillar.
    for left, right in zip(left_y, right_y):
        low, high = min(left, right), max(left, right)
        g1[low : high + 1, -1:] = 0
        g2[low : high + 1, :1] = 0

    g = np.hstack([g1, g2])
    return np.vstack(
        [
            np.ones((100, g.shape[1])).astype(np.uint8),
            g,
            np.ones((100, g.shape[1])).astype(np.uint8),
        ]
    )


def segment(
    book: str,
    books_dir: str = "books",
    config: BookConfig | None = None,
) -> list[str]:
    """Merge a book's pages into one graph image per subtree.

    Reads ``{books_dir}/{book}/pages/{i}.png`` for ``i`` in
    ``0..num_pages-1``, trims and shrinks each, detects subtree-start pages,
    and stitches each run of continuation pages onto its start page. Writes one
    image per subtree to ``{books_dir}/{book}/graphs/{start}_{end}.png``.

    Args:
        book: Book name, e.g. ``"book1"``. Looks up :data:`BOOK_CONFIGS` when
            ``config`` is omitted, and locates the book directory.
        books_dir: Root directory containing per-book asset folders.
        config: Explicit config, overriding the :data:`BOOK_CONFIGS` lookup.

    Returns:
        The list of written graph file paths, in emission order.
    """
    if config is None:
        if book not in BOOK_CONFIGS:
            raise KeyError(
                f"no BookConfig for {book!r}; known books: {sorted(BOOK_CONFIGS)}"
            )
        config = BOOK_CONFIGS[book]

    pages_dir = os.path.join(books_dir, book, "pages")
    graphs_dir = os.path.join(books_dir, book, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    logger.info(
        "Segmenting %s: %d pages in %s -> %s",
        book,
        config.num_pages,
        pages_dir,
        graphs_dir,
    )

    # Load, trim, detect subtree-starts, and shrink every page.
    pages: list[np.ndarray] = []
    is_page_tree_start: list[bool] = []
    for i in range(config.num_pages):
        filepath = os.path.join(pages_dir, f"{i}.png")
        a = get_image(filepath)
        a = trim_borders(a)
        tree_start_x = is_tree_start_page(a, config)
        if tree_start_x != -1:
            logger.info("Page %d starts a subtree (label at x=%d)", i, tree_start_x)
            # Crop the label column (and everything right of it) off the page.
            a = a[:, :tree_start_x]
        is_page_tree_start.append(tree_start_x != -1)
        pages.append(shrink_page(a))

    # Merge each run of continuation pages onto the page that started the subtree.
    logger.info("Merging and saving graphs")
    written: list[str] = []
    i = 0
    while i < config.num_pages:
        start_i = i
        graph = pages[i]
        i += 1
        while i < config.num_pages and not is_page_tree_start[i]:
            logger.info("Merging page %d into subtree started at %d", i, start_i)
            graph = merge_graphs(pages[i], graph)
            i += 1
        out_path = os.path.join(graphs_dir, f"{start_i}_{i - 1}.png")
        save_image(graph, out_path)
        logger.info("Wrote subtree %d..%d -> %s", start_i, i - 1, out_path)
        written.append(out_path)

    logger.info("Segmented %d subtrees for %s", len(written), book)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        required=True,
        choices=sorted(BOOK_CONFIGS),
        help="Book to process (e.g. book1).",
    )
    parser.add_argument(
        "--books-dir",
        default="books",
        help="Root directory containing per-book asset folders (default: books).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    written = segment(args.book, books_dir=args.books_dir)
    logger.info("Done. %d graphs written for %s.", len(written), args.book)


if __name__ == "__main__":
    main()
