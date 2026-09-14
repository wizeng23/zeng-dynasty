"""Stage 5 of the pipeline: merged subtree-graph images -> tree JSONL + name crops.

Each image in ``books/{book}/graphs/{start}_{end}.png`` is one subtree's
line-graph (from ``src/merge_pages.py``, Stage 4). Every name sits on a rigid grid:
generation is a vertical row, sibling position is a horizontal column, and a
vertical line drops from each parent down to its children. This stage turns
that geometry into structured :class:`src.model.Node` records:

1. Find the graph's line segments as connected components.
2. Read each segment's endpoints: its top end (where it hangs from its parent)
   and its bottom ends (where its children hang off it).
3. Merge segments into :class:`LineNode`\\ s -- a node's ``top`` is where its own
   name sits and its ``bot`` is where its line fans out to its children -- then
   infer any end that ran off the page.
4. Order nodes right-to-left / eldest-first, assign globally-unique BFS-by-
   generation IDs, crop each name image, and write the domain ``Node`` rows.

The book reads **right-to-left**, so within a generation-row the eldest sibling
is on the *right*. Node ordering, ID assignment and ``children`` arrays are all
RTL / eldest-first (a fix over the old left-to-right logic -- see
``docs/specs/2026-08-18-pipeline-rewrite-design.md``).

Outputs:
    data/{book}.jsonl              one domain ``Node`` per line
    books/{book}/names/{id}.png    the cropped name image for each node

CLI:
    python -m src.build_tree --book book1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os

import numpy as np

from src.imaging import get_image, pad_image, save_image
from src.model import Node

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("build_tree requires opencv-python (cv2)") from exc


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BookConfig:
    """Per-book Stage-5 configuration.

    The parse geometry is otherwise structural, but a few thresholds are kept
    here so a book with different scan geometry can retune them without touching
    the algorithm.

    Attributes:
        line_threshold: A connected component is a "line" if its bounding box
            spans more than this many pixels in width *or* height. Smaller
            components are name-ink and ignored by the line finder.
        end_threshold: When reading a segment's bottom endpoints, rows within
            this distance of the segment's lowest row count as "bottom".
        merge_max_drop: Two segments merge into one node only if the vertical
            gap between them is in ``(0, merge_max_drop)`` -- a real parent/child
            line is roughly one generation-row tall.
        merge_max_shift: ...and only if their columns line up within this many
            pixels (they are the same vertical line, split by a graph artifact).
        node_min_height / node_max_height: A well-formed node's line height
            (``bot - top``) should fall in this band; outside it the node is
            flagged as suspicious by :func:`verify_nodes`.
        gen_row_min / gen_row_max: A child's name row should sit roughly one
            generation-row below its parent's; a vertical drop outside this band
            is flagged by :func:`check_grid_consistency` as a probable mis-merge.
        ignore_regions: Per-graph rectangles of known NON-tree ink to blank
            before parsing, keyed by graph stem (e.g. ``"8_10"``). Each rectangle
            is ``(r0, c0, r1, c1)`` in graph pixel coordinates. Used for one-off
            hand-drawn marks that would otherwise read as phantom lines -- e.g.
            the ink stroke a family member drew in 8_10's margin. Kept explicit
            and localized so it never suppresses real tree ink.
    """

    # Defaults are v1-native (full 600dpi) pixel constants -- ~3x the old v0
    # normalized-canvas values (v0 trimmed width ~1150; v1 native tree ~3500).
    # The parse geometry is structural, so every v1 book shares these defaults
    # (as in v0, where Book 1 used pure defaults and Book 2 changed a single
    # threshold). Per-graph one-offs live in ``ignore_regions``.
    line_threshold: int = 200
    end_threshold: int = 150
    merge_max_drop: int = 600
    merge_max_shift: int = 60
    node_min_height: int = 150
    node_max_height: int = 750
    gen_row_min: int = 800
    gen_row_max: int = 1050
    ignore_regions: dict[str, list[tuple[int, int, int, int]]] = dataclasses.field(
        default_factory=dict
    )


# ``gen_row_min/max`` bound the parent->child *generation drop* (child.top row
# minus parent.top row) -- one full row of vertical grid spacing. That spacing is
# set by the scanner geometry, not the name height, so it is the SAME in both
# books: measured drops cluster tightly at ~300-332px in Book 1 (all 149 edges)
# and Book 2 alike. Hence the band is a shared default. (An earlier default of
# [60,260] was miscalibrated -- it matched a node's own line height, not the
# generation drop -- so it flagged 100% of Book 1's real edges as "possible
# mis-merge"; the QA overlay confirmed the parse was correct and the band wrong.)
# Both books share the scan geometry (trimmed width 1150); Book 2's names are two
# characters stacked, taller per node, but the row-to-row drop is unchanged.
# ``merge_max_shift`` bounds the column drift allowed when fusing a child stub
# into the parent stub below it (same person, two line segments). Book 2's names
# are two characters stacked, and the name-top column vs. the child-bar-hang
# column can drift ~21-23px where the lower connector bar sits slightly off the
# name's centre. At the old shared default of 20 those merges were rejected,
# leaving TWO overlapping nodes per person (an inferred-top phantom up in the bar
# plus a childless name node) -- visible in QA as overlapping boxes in 114_120.
# 25 admits those; Book 2's smallest genuine non-merge shift is 35 (the 8_10
# stray-ink vertical), and Book 1 has no near-miss merges in [15,30), so the
# looser bound is safe and scoped to Book 2 only.
# Every v1 book uses the shared v1-native defaults (BookConfig()); the parse
# geometry is structural across books. A book only needs an entry here if it
# requires a genuine override (e.g. per-graph ``ignore_regions``); unlisted books
# fall back to defaults via ``config_for``.
# Per-graph stray pen marks to blank before parsing (``ignore_regions``, each
# ``(r0, c0, r1, c1)``). These are hand-drawn vertical strokes a family member
# added beside a name -- once red (ignored) but black in the v1 bitonal scans, so
# they read as spurious lines and break the parse. Found by an irregular-vertical-
# beside-a-glyph detector, then each was human-confirmed as a real pen mark (a
# false positive on 13_16's 性 character was excluded). See docs/history.md.
BOOK1_PEN_MARKS: dict[str, list[tuple[int, int, int, int]]] = {
    "2_2": [(2374, 780, 2577, 810)],
    "3_3": [(1460, 1388, 1700, 1428), (2408, 1370, 2598, 1410),
            (3319, 1069, 3524, 1108), (4235, 1096, 4477, 1129),
            (5203, 773, 5426, 807)],
    "5_5": [(4286, 496, 4454, 526), (5180, 197, 5392, 232)],
    "6_6": [(3306, 782, 3466, 816)],
    "7_7": [(1502, 1077, 1654, 1105), (2419, 790, 2689, 827),
            (3365, 807, 3523, 834), (4280, 507, 4459, 538),
            (5214, 491, 5501, 527)],
    "8_8": [(5276, 1395, 5432, 1427)],
    "13_16": [(2372, 7848, 2635, 7890), (5224, 5449, 5532, 5484)],
}

BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(ignore_regions=BOOK1_PEN_MARKS),
}


def config_for(book: str) -> BookConfig:
    """The parse config for a book: its BOOK_CONFIGS entry, or the shared default."""
    return BOOK_CONFIGS.get(book, BookConfig())

# Linear-resolution ratio of a v1 native page to the v0 normalized canvas.
V1_SCALE = 3.0

# Name-crop geometry (v1 native 600dpi). Exactly v0's scheme: the name sits in the
# node's own line segment (top..bot), so each crop grabs that band and is trimmed to
# the ACTUAL name ink with a fixed margin -- both dimensions follow the real name (a
# 3-char name is naturally taller, a wide glyph wider), nothing normalized, clipped,
# or character-counted. (A fixed box inferred the character count from the ink
# height, but the between-character gap and the riser gap are the same size, so it
# oscillated and clipped/over-grew names. A later ink-walk down from the top over-
# reached past the segment into hand-written annotation notes below a name. v0
# cropped the segment and never had either problem.)
#
# v0's unscaled 40px half-window clipped every v1 name -- the ~3x scale to 120 fixes
# that. NAME_TRIM_PAD is v0's 10px margin, scaled ~3x to 30px, kept on every side.
NAME_HALF_WIDTH = int(round(40 * V1_SCALE))   # 120px: half-window for the raw grab
NAME_TRIM_PAD = int(round(10 * V1_SCALE))     # 30px: margin kept around trimmed ink

# Top/bottom trim smooths the row-ink profile before finding the name extent, so a
# stray ADF smear speck (a faint dot far from the name) does not hold the box open
# down to it. NAME_SPECK_WIN: smoothing window (a speck narrower than this averages
# away; a real character stroke, ~200px tall, survives). NAME_ROW_INK_MIN: smoothed
# ink fraction a row must clear to count as real character ink.
NAME_SPECK_WIN = 41           # ~half a character; averages out isolated specks
NAME_ROW_INK_MIN = 0.015      # smoothed ink fraction floor for a real name row

# infer_ends geometry: recovering the missing end of a leaf/root (or page-clipped)
# node by scanning its column. REACH is how far past the known end to look for the
# name's far edge -- v0's 200px scaled ~3x so it clears a full 2-/3-char name at v1
# native resolution (unscaled it reached only one char, truncating 2-char leaves).
INFER_END_REACH = int(round(200 * V1_SCALE))  # 600px: column scan depth
INFER_END_PAD = int(round(10 * V1_SCALE))     # 30px: margin past the recovered ink
INFER_END_PROBE_HALF = int(round(30 * V1_SCALE))  # 90px: +-column probe half-width

# Name-band insets: skip the connection-point pixels at each end of the segment
# before cropping the name (v0's 5px, scaled).
NAME_END_INSET = int(round(5 * V1_SCALE))     # 15px

# sort_nodes generation-band tolerance: two nodes are on the same generation row
# when their ``top`` rows are within this many px (v0's 60px, scaled). Siblings
# hang from one bar so in practice sit on the identical row; the band is generous.
GEN_BAND_TOL = int(round(60 * V1_SCALE))      # 180px

# Blank-crop ink threshold: a name crop with fewer than this many inked pixels is
# effectively empty (a seam stub, not a real name). This counts AREA, so it scales
# with V1_SCALE**2 (a v1 crop has ~9x the pixels of the same v0 crop) -- v0's 30.
BLANK_NAME_MAX_INK = int(round(30 * V1_SCALE * V1_SCALE))  # 270

# Stroke-end read (see find_line_ends). A free stroke end is the top (or bottom)
# edge of a narrow vertical stroke with no ink beyond it. STROKE_MAX_WIDTH bounds
# an ink run that still counts as a stroke rather than a bar (v1 lines are ~7px,
# generation bars hundreds); STROKE_MIN_LEN is how far the run must stay that
# narrow -- a bar corner widens within a few rows and a speck on a bar's ragged
# edge runs out of ink, so neither passes, while a real riser/hang-line does.
STROKE_MAX_WIDTH = int(round(13 * V1_SCALE))  # 40px
STROKE_MIN_LEN = int(round(33 * V1_SCALE))    # 100px
# A stroke may drift this many px sideways over STROKE_MIN_LEN rows (deskew slop).
STROKE_DRIFT = 4


@dataclasses.dataclass
class LineNode:
    """A parse-time geometric node, distinct from the domain :class:`src.model.Node`.

    A ``LineNode`` is a position on the graph grid. ``top`` is the ``(row, col)``
    where the node's own name sits (the top of the line that hangs it from its
    parent); ``bot`` is the ``(row, col)`` where its line fans out to its
    children. Either may be ``None`` until :func:`infer_ends` fills it in.
    ``children`` holds the child ``LineNode`` objects, later ordered eldest-first.
    """

    id: int | None = None
    top: tuple[int, int] | None = None
    bot: tuple[int, int] | None = None
    children: list[LineNode] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return f"LineNode(id={self.id}, top={self.top}, bot={self.bot})"


def find_lines(
    image: np.ndarray, threshold: int = 70
) -> list[set[tuple[int, int]]]:
    """Find the graph's line segments as large connected components.

    Uses ``cv2.connectedComponentsWithStats`` (4-connectivity) on the ink
    foreground and keeps every component whose bounding box spans more than
    ``threshold`` pixels in width or height. This replaces the old hand-rolled
    pixel BFS -- verified to return the identical pixel sets, but in C and with
    bounding boxes for free (see the design spec).

    Args:
        image: Binary ink grid (0 == ink, 1 == background).
        threshold: A component qualifies as a line if its width or height
            exceeds this many pixels.

    Returns:
        A list of components, each a set of ``(row, col)`` ink pixels.
    """
    if image.size == 0:
        return []

    # cv2 labels the nonzero foreground; our ink is 0, so invert to make ink 1.
    foreground = (1 - image).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        foreground, connectivity=4
    )

    results: list[set[tuple[int, int]]] = []
    for label in range(1, num_labels):  # 0 is the background label
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        # find_lines' original bound is (max - min) > threshold; a component's
        # width/height stat is (max - min + 1), so compare against threshold + 1.
        if width > threshold + 1 or height > threshold + 1:
            rows, cols = np.where(labels == label)
            results.append(set(zip(rows.tolist(), cols.tolist())))
    return results


def find_line_ends(
    points: set[tuple[int, int]], threshold: int = 50
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Read a line segment's top and bottom endpoints.

    A segment is one parent's vertical line: it has a single top end (where it
    hangs from its own parent) and one or more bottom ends (one per child). The
    top end is the pixels at the minimum row; the bottom ends are the pixels
    within ``threshold`` rows of the maximum row, each near-adjacent run
    collapsed to a single representative column.

    Two-character stacked names (Book 2) push the horizontal fan-out bar right
    up to the top of the component, with no vertical hang-line rising above it.
    The bar is then flush with ``min_x``, so its two slightly-taller end corners
    read as *two* isolated top columns rather than the single hang-point the
    caller expects (Book 1's vertical hang-line always rose to one point). When
    the strict ``min_x`` top yields more than one endpoint, the whole top band --
    within ``threshold`` rows of ``min_x``, mirroring the bottom logic -- is
    collapsed to a single representative: the parent connection of a bare
    fan-out bar. Single-top segments (Book 1) are untouched.

    That band read assumes a single-level fan-out: parent at the top row, every
    child within ``threshold`` of the bottom row. A **stepped** bar breaks both
    assumptions (Book 2 graph 67_68: 宏羨's hang-line steps left and UP into a
    raised sibling bar, so the parent connection sits ~100 rows below the
    component top and two of the three children hang ~330 rows above the
    bottom). The band read then finds two "parents" -- the raised bar's corner
    and the real hang-line -- or, had the bar's top edge been clean, silently
    picks the bar corner. So the band read is cross-checked against a
    **stroke read** (:func:`_stroke_ends`): the free ends of narrow vertical
    strokes. An upward free end is where the segment hangs from its parent; a
    downward free end is where a child hangs off it. The stroke read decides the
    parent when the band read is ambiguous or disagrees with a single clear
    hang-line, and contributes any child riser the bottom band missed. A bar
    flush with the top has no upward stroke, so it keeps the band read's parent.
    Where the two reads agree (every component of Books 1 and 2 bar 67_68 and
    one 11_17 riser), the band read's exact coordinates are kept.

    Args:
        points: The ``(row, col)`` pixels of one connected component.
        threshold: Rows within this distance of the top/bottom count as ends.

    Returns:
        ``(top_points, bottom_points)``, each a list of ``(row, col)`` tuples
        sorted left-to-right by column.
    """
    if not points:
        return ([], [])

    min_x = min(x for x, _y in points)
    top_ys = {y for x, y in points if x == min_x}
    top_ys_filtered = remove_adjacent(top_ys)
    if len(top_ys_filtered) > 1:
        # Fan-out bar flush with the top: collapse the top band to one point.
        top_band = {y for x, y in points if abs(x - min_x) <= threshold}
        top_ys_filtered = remove_adjacent(top_band)

    max_x = max(x for x, _y in points)
    bottom_ys = {y for x, y in points if abs(x - max_x) <= threshold}
    bottom_ys_filtered = remove_adjacent(bottom_ys)

    top_points = [(min_x, y) for y in sorted(top_ys_filtered)]
    bottom_points = [(max_x, y) for y in sorted(bottom_ys_filtered)]

    ups, downs = _stroke_ends(points)

    def same_end(p: tuple[int, int], q: tuple[int, int]) -> bool:
        # Same stroke: within the band tolerance vertically, one stroke width across.
        return abs(p[0] - q[0]) <= threshold and abs(p[1] - q[1]) <= STROKE_MAX_WIDTH

    if len(ups) == 1:
        if not (len(top_points) == 1 and same_end(top_points[0], ups[0])):
            logger.info(
                "stroke read picks parent %s over band read %s", ups[0], top_points
            )
            top_points = [ups[0]]

    for d in downs:
        if not any(same_end(d, b) for b in bottom_points):
            logger.info("stroke read adds child end %s above the bottom band", d)
            bottom_points.append(d)
    bottom_points.sort(key=lambda p: p[1])
    return (top_points, bottom_points)


def _stroke_ends(
    points: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Free ends of the narrow vertical strokes in one component.

    Returns ``(upward_ends, downward_ends)``: the topmost pixel of every stroke
    whose top is free (no ink above), and the bottommost of every stroke whose
    bottom is free. Ends on the same stroke (ragged edges) are collapsed to one.

    Args:
        points: The ``(row, col)`` pixels of one connected component.
    """
    rows = np.fromiter((p[0] for p in points), dtype=np.int64, count=len(points))
    cols = np.fromiter((p[1] for p in points), dtype=np.int64, count=len(points))
    r0, c0 = int(rows.min()), int(cols.min())
    # 1px background pad so the component's own edges read as free.
    mask = np.zeros((int(rows.max()) - r0 + 3, int(cols.max()) - c0 + 3), dtype=bool)
    mask[rows - r0 + 1, cols - c0 + 1] = True

    ups = _upward_stroke_ends(mask)
    h = mask.shape[0]
    downs = [(h - 1 - r, c) for r, c in _upward_stroke_ends(mask[::-1])]
    return (
        [(r + r0 - 1, c + c0 - 1) for r, c in ups],
        [(r + r0 - 1, c + c0 - 1) for r, c in downs],
    )


def _upward_stroke_ends(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(row, col)`` of each upward free stroke end in a padded ink mask.

    A candidate is a whole ink run on a row with no ink in the row above across
    the run's extent (+-1px) -- a genuine top edge, not a 1px protrusion on a
    stroke's side. The run must be at most :data:`STROKE_MAX_WIDTH` wide and the
    stroke must stay that narrow for :data:`STROKE_MIN_LEN` rows below it.
    """
    ends: list[tuple[int, int]] = []
    h = mask.shape[0]
    for r in range(1, h - 1):
        row = mask[r]
        ink = np.where(row)[0]
        if len(ink) == 0:
            continue
        breaks = np.where(np.diff(ink) > 1)[0]
        starts = np.concatenate([[ink[0]], ink[breaks + 1]])
        stops = np.concatenate([ink[breaks], [ink[-1]]])
        above = mask[r - 1]
        for a, b in zip(starts, stops):
            if b - a + 1 > STROKE_MAX_WIDTH or r + STROKE_MIN_LEN > h:
                continue
            if above[max(0, a - 1):b + 2].any():
                continue
            if _is_narrow_stroke(mask, r, (a + b) // 2):
                ends.append((int(r), int((a + b) // 2)))
    return _collapse_stroke_ends(ends)


def _is_narrow_stroke(mask: np.ndarray, r: int, c: int) -> bool:
    """Does the ink at ``(r, c)`` continue down STROKE_MIN_LEN rows as a stroke?"""
    for rr in range(r, r + STROKE_MIN_LEN):
        row = mask[rr]
        if not row[c]:
            lo = max(0, c - STROKE_DRIFT)
            near = np.where(row[lo:c + STROKE_DRIFT + 1])[0]
            if len(near) == 0:
                return False
            c = lo + int(near[len(near) // 2])
        if _run_width_at(row, c) > STROKE_MAX_WIDTH:
            return False
    return True


def _run_width_at(row: np.ndarray, c: int) -> int:
    """Width of the ink run through column ``c`` (capped just past the max)."""
    lo = max(0, c - STROKE_MAX_WIDTH - 1)
    hi = min(len(row), c + STROKE_MAX_WIDTH + 2)
    win = row[lo:hi]
    ci = c - lo
    left = win[:ci + 1][::-1]
    right = win[ci:]
    n_left = int(np.argmin(left)) if not left.all() else STROKE_MAX_WIDTH + 1
    n_right = int(np.argmin(right)) if not right.all() else STROKE_MAX_WIDTH + 1
    return n_left + n_right - 1


def _collapse_stroke_ends(ends: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse ends within one stroke width of each other; keep the topmost."""
    out: list[tuple[int, int]] = []
    for r, c in sorted(ends, key=lambda p: p[1]):
        if out and c - out[-1][1] <= STROKE_MAX_WIDTH:
            if r < out[-1][0]:
                out[-1] = (r, out[-1][1])
            continue
        out.append((r, c))
    return out


def remove_adjacent(numbers: set[int] | list[int], threshold: int = 30) -> list[int]:
    """Collapse near-adjacent values, keeping the first of each cluster.

    A single logical endpoint is a thick run of neighboring columns; this
    reduces each run to one representative. ``{4, 6, 7, 8}`` -> ``[4, 7]``: only
    values more than ``threshold`` past the last kept value survive.

    Args:
        numbers: Candidate endpoint column indices.
        threshold: Minimum gap between kept values.

    Returns:
        The thinned list of representative indices, ascending.
    """
    if not numbers:
        return []

    sorted_nums = sorted(numbers)
    result = [sorted_nums[0]]
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i - 1] > threshold:
            result.append(sorted_nums[i])
    return result


def build_nodes(
    line_ends: list[tuple[tuple[int, int], list[tuple[int, int]]]],
) -> list[LineNode]:
    """Turn per-segment (parent-end, child-ends) pairs into linked line nodes.

    Each line segment contributes one parent node (its top end fans out at its
    ``bot``) and one node per child (each hangs by its ``top``). Child nodes are
    attached to their parent; the flat list of every node is returned for
    merging.

    Args:
        line_ends: For each segment, ``(parent_point, child_points)`` where
            ``parent_point`` is the segment's single top endpoint and
            ``child_points`` its bottom endpoints.

    Returns:
        Every :class:`LineNode` created, parents and children alike.
    """
    nodes: list[LineNode] = []
    for parent, children in line_ends:
        parent_node = LineNode(bot=parent)
        nodes.append(parent_node)
        for child in children:
            child_node = LineNode(top=child)
            parent_node.children.append(child_node)
            nodes.append(child_node)
    return nodes


def merge_nodes(nodes: list[LineNode], config: BookConfig) -> list[LineNode]:
    """Merge each child stub into the parent stub directly below it.

    A single person appears twice across two line segments: once as a child end
    of the line above (a node with only ``top``) and once as the parent end of
    the line below (a node with only ``bot``). When such a pair lines up -- a
    small vertical drop and near-identical column -- they are the same person and
    are fused: the parent-stub inherits the child-stub's ``bot`` and children.

    Faithfully ports the old O(n^2) restart-on-progress loop.

    Args:
        nodes: The flat node list from :func:`build_nodes` (mutated in place).
        config: Book config supplying the drop/shift merge thresholds.

    Returns:
        The reduced node list (same object as ``nodes``).
    """
    while True:
        made_progress = False
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue
                parent = nodes[i]
                child = nodes[j]
                # Only fuse when parent lacks a bottom and child lacks a top.
                if parent.bot or child.top:
                    continue
                assert parent.top is not None and child.bot is not None
                tx, ty = parent.top
                bx, by = child.bot
                if (
                    0 < bx - tx < config.merge_max_drop
                    and abs(by - ty) < config.merge_max_shift
                ):
                    parent.bot = child.bot
                    parent.children = child.children
                    del nodes[j]
                    made_progress = True
                    break
            if made_progress:
                break
        if not made_progress:
            break
    return nodes


def apply_ignore_regions(
    a: np.ndarray, graph_stem: str, config: BookConfig
) -> np.ndarray:
    """Blank any configured non-tree ink rectangles for this graph.

    Sets the pixels inside each ``config.ignore_regions[graph_stem]`` rectangle to
    background (1), erasing known hand-drawn marks before they reach the line
    finder. Returns ``a`` unchanged (same object) when the graph has no regions,
    so the common path is free.

    Args:
        a: The graph's binary ink grid (0 == ink, 1 == background).
        graph_stem: The graph's filename stem, e.g. ``"8_10"``.
        config: The book config carrying :attr:`BookConfig.ignore_regions`.

    Returns:
        The grid with the configured rectangles blanked (a copy if any applied).
    """
    regions = config.ignore_regions.get(graph_stem)
    if not regions:
        return a
    out = a.copy()
    h, w = out.shape
    for r0, c0, r1, c1 in regions:
        out[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)] = 1
    return out


def drop_blank_leaf_nodes(
    nodes: list[LineNode], a: np.ndarray
) -> list[LineNode]:
    """Drop phantom nodes: a blank name crop AND no children.

    A real node carries information one of two ways: a name above its line, or a
    subtree hanging below it (a cross-graph orphan whose own name/parent lives in
    an adjacent graph still holds real children). A node with NEITHER -- blank
    name, no children -- is noise: a stray ink stroke read as a short line (e.g.
    the hand-drawn vertical in 8_10's margin). Such nodes have no place in the
    tree and produce overlapping empty boxes in QA, so they are removed.

    Nodes with children are kept even when their own name crop is blank, so the
    genuine cross-graph orphans (e.g. 69_82's off-graph parent of 尚澜) survive.

    Args:
        nodes: The merged, end-inferred line nodes.
        a: The graph's binary ink grid, to measure each node's name-crop ink.

    Returns:
        The list with blank childless leaves removed (a new list).
    """
    kept: list[LineNode] = []
    for n in nodes:
        blank = (
            n.top is not None
            and n.bot is not None
            and int((1 - get_name_image(n, a)).sum()) < BLANK_NAME_MAX_INK
        )
        if blank and not n.children:
            logger.info("dropping blank childless phantom node: %s", n)
            continue
        kept.append(n)
    return kept


def infer_ends(nodes: list[LineNode], a: np.ndarray) -> None:
    """Infer the missing end of a node that has only one endpoint.

    A leaf (a child with no line segment of its own below it) has a ``top`` but no
    ``bot``; a subgraph root has a ``bot`` but no ``top``; a line dangling off a
    page edge likewise loses an end. In every case the name still occupies the
    node's column, so this recovers the missing end by scanning the column from
    :data:`INFER_END_REACH` px past the known end and walking back to the last
    inked row -- i.e. the far edge of the name -- then padding
    :data:`INFER_END_PAD` px for a crop margin. Mutates ``nodes`` in place.

    :data:`INFER_END_REACH` must clear a full 2-/3-char name: it is v0's 200px
    scaled to v1 native resolution (v0's ~65px/char made 200px ~3 chars; at v1's
    ~200px/char an unscaled 200 reaches only ONE char, truncating every 2-char
    leaf -- the bug this scaling fixes). The walk stops at the LAST ink within the
    reach, so it takes the whole name but not a distant annotation past a gap.

    Args:
        nodes: The merged line nodes.
        a: The graph's binary ink grid, used to probe for the missing end.
    """
    for n in nodes:
        if n.top is None:
            assert n.bot is not None
            top = max(0, n.bot[0] - INFER_END_REACH)
            y = n.bot[1]
            min_y = max(0, y - INFER_END_PROBE_HALF)
            max_y = min(a.shape[1] - 1, y + INFER_END_PROBE_HALF)
            while top < n.bot[0] and 0 not in a[top][min_y:max_y]:
                top += 1
            top = max(0, top - INFER_END_PAD)
            n.top = (top, y)
        if n.bot is None:
            assert n.top is not None
            bot = min(a.shape[0] - 1, n.top[0] + INFER_END_REACH)
            y = n.top[1]
            min_y = max(0, y - INFER_END_PROBE_HALF)
            max_y = min(a.shape[1] - 1, y + INFER_END_PROBE_HALF)
            while bot > n.top[0] and 0 not in a[bot][min_y:max_y]:
                bot -= 1
            bot = min(bot + INFER_END_PAD, a.shape[0] - 1)
            n.bot = (bot, y)


def verify_nodes(nodes: list[LineNode], config: BookConfig) -> list[LineNode]:
    """Flag nodes whose line height is outside the expected band.

    A malformed merge or a mis-read endpoint tends to produce a node whose
    ``bot - top`` is implausibly short or tall. Such nodes are logged as
    suspicious (the geometry check; :func:`check_grid_consistency` catches the
    orthogonal mis-merge case). Does not raise.

    Args:
        nodes: The merged, end-inferred line nodes.
        config: Book config supplying the node-height band.

    Returns:
        The subset of ``nodes`` that looked suspicious.
    """
    suspicious: list[LineNode] = []
    for n in nodes:
        assert n.top is not None and n.bot is not None
        height = n.bot[0] - n.top[0]
        if not (config.node_min_height < height < config.node_max_height):
            logger.warning("node looks sus (height=%d): %s", height, n)
            suspicious.append(n)
    return suspicious


def check_grid_consistency(
    nodes: list[LineNode], config: BookConfig
) -> list[tuple[LineNode, LineNode]]:
    """Flag parent/child pairs that break the generation-grid geometry.

    Names sit on a rigid grid: a child's name row is one generation-row below
    its parent's. A seam-merge can connect the wrong dangling line to the wrong
    child -- a *wrong-but-valid* topology :func:`verify_nodes` cannot see, since
    each node is individually well-formed. This checks that every child's ``top``
    row sits ``gen_row_min..gen_row_max`` below its parent's ``top`` row and logs
    a warning for any pair outside that band. Does not raise.

    Args:
        nodes: The merged line nodes, with children attached.
        config: Book config supplying the generation-row band.

    Returns:
        The list of ``(parent, child)`` pairs that violated the band.
    """
    violations: list[tuple[LineNode, LineNode]] = []
    for parent in nodes:
        if parent.top is None:
            continue
        for child in parent.children:
            if child.top is None:
                continue
            drop = child.top[0] - parent.top[0]
            if not (config.gen_row_min <= drop <= config.gen_row_max):
                logger.warning(
                    "grid check: child %s is %dpx below parent %s "
                    "(expected %d..%d) -- possible mis-merge",
                    child,
                    drop,
                    parent,
                    config.gen_row_min,
                    config.gen_row_max,
                )
                violations.append((parent, child))
    return violations


def sort_nodes(nodes: list[LineNode]) -> list[LineNode]:
    """Order nodes top-to-bottom by generation-row, right-to-left within a row.

    Nodes are banded into generation-rows (``top`` rows within
    :data:`GEN_BAND_TOL` px of each other), then each band is ordered
    **right-to-left** so the eldest sibling
    (rightmost in the book) comes first. This is the core RTL / eldest-first fix
    over the old left-to-right ordering (see the design spec); it makes the
    subsequent ID assignment BFS-by-generation, right-to-left.

    Args:
        nodes: The merged line nodes.

    Returns:
        A new list in generation-band order, eldest-first within each band.
    """

    def band_key(node: LineNode) -> tuple[int, int]:
        assert node.top is not None
        # Descending column so the rightmost (eldest) sorts first within a band.
        return (node.top[0], -node.top[1])

    nodes_sorted = sorted(nodes, key=band_key)

    result: list[LineNode] = []
    i = 0
    while i < len(nodes_sorted):
        group = [nodes_sorted[i]]
        j = i + 1
        while j < len(nodes_sorted):
            assert nodes_sorted[j].top is not None and nodes_sorted[i].top is not None
            if abs(nodes_sorted[j].top[0] - nodes_sorted[i].top[0]) < GEN_BAND_TOL:
                group.append(nodes_sorted[j])
                j += 1
            else:
                break
        # Rightmost first (eldest-first) within the generation band.
        group.sort(key=lambda node: -node.top[1])
        result.extend(group)
        i = j
    return result


def _tight_ink_box(node: LineNode, a: np.ndarray) -> tuple[int, int, int, int]:
    """Trim to the actual name ink: ``(left, top, right, bottom)`` in graph px.

    Exactly v0's scheme: the name lives in the node's own line segment (its
    ``top..bot`` band), so grab that band -- a +-:data:`NAME_HALF_WIDTH` column
    window between ``top`` and ``bot`` -- and trim to ANY ink with a
    :data:`NAME_TRIM_PAD` margin on every side. Both dimensions follow the real
    name; nothing is normalized, character-counted, or walked past the segment.

    Cropping to the segment (not walking down through nearby ink) is what keeps a
    node with hand-written annotation notes below its name (e.g. 7_7's 文迦, whose
    notes sit just under it) from swallowing them. A degenerate stub (no ink)
    falls back to the raw window so no dimension collapses.
    """
    assert node.top is not None and node.bot is not None
    y = node.top[1]
    top_row = node.top[0] + NAME_END_INSET
    bot_row = max(node.bot[0] - NAME_END_INSET, top_row + 1)
    left0 = max(0, y - NAME_HALF_WIDTH)
    right0 = min(a.shape[1], y + NAME_HALF_WIDTH)
    # ``top`` and ``bot`` are BOTH hard boundaries: ``top`` is where the node hangs
    # from its parent (the name cannot be above it -- above sits the parent's riser
    # tip) and ``bot`` is where the line fans out to children (the name cannot be
    # below it -- below sits this node's own riser). So the measurement band is
    # exactly [top_row, bot_row]; it never pads past either endpoint into a riser.
    # The NAME_END_INSET already steps the band inside the connection points, and the
    # top/bottom margins come from the whitespace between the endpoints and the ink.
    band_top = top_row
    band_bot = bot_row
    band = a[band_top:band_bot, left0:right0]

    col_present = np.sum(1 - band, axis=0)
    if np.any(col_present):
        inked = np.where(col_present > 0)[0]
        left = left0 + max(int(inked[0]) - NAME_TRIM_PAD, 0)
        right = left0 + min(int(inked[-1]) + NAME_TRIM_PAD, band.shape[1] - 1) + 1
    else:
        left, right = left0, right0

    # Top/bottom trim to the real character ink, with the pad. A plain any-ink trim
    # is fooled by a stray ADF speck (a faint smear dot far below the name), keeping
    # a huge empty block down to it -- e.g. 8 (1 char + smear stroke) and 133 (2 char
    # + a speck 250px below). So smooth the row profile over NAME_SPECK_WIN rows and
    # keep only rows whose smoothed ink fraction clears NAME_ROW_INK_MIN: an isolated
    # speck averages away, while a real (even thin) character stroke survives. This
    # keeps thin top strokes like 点's dot while dropping artifact blocks.
    win_w = band.shape[1]
    row_frac = np.sum(1 - band, axis=1) / max(win_w, 1)
    kern = np.ones(NAME_SPECK_WIN) / NAME_SPECK_WIN
    smoothed = np.convolve(row_frac, kern, mode="same")
    real = np.where(smoothed > NAME_ROW_INK_MIN)[0]
    if len(real):
        top = band_top + max(int(real[0]) - NAME_TRIM_PAD, 0)
        bottom = band_top + min(int(real[-1]) + NAME_TRIM_PAD, band.shape[0] - 1) + 1
    else:
        top, bottom = top_row, bot_row
    return left, top, right, bottom


def name_box_coords(node: LineNode, a: np.ndarray) -> tuple[int, int, int, int]:
    """The name box ``(left, top, right, bottom)`` in graph pixels.

    Crops tight to the actual name ink with a :data:`NAME_TRIM_PAD` margin on every
    side -- exactly v0's scheme (its 10px pad scaled ~3x to 30px for v1 native
    resolution). Both dimensions follow the real name: a 3-character name is taller,
    a wide glyph is wider. Nothing is normalized, clipped, or character-counted (an
    earlier fixed box mis-counted characters where the between-character gap matched
    the riser gap, and over-padded narrow glyphs to the rare widest one).

    Single source of truth for the crop: :func:`get_name_image` crops to this box,
    and Stage 5 records it in the parse sidecar so the QA overlay draws the exact
    same box (no re-derivation, no drift).
    """
    return _tight_ink_box(node, a)


def get_name_image(node: LineNode, a: np.ndarray) -> np.ndarray:
    """Crop the name image at a node's top endpoint.

    Crops to :func:`name_box_coords` -- the name ink, height following the name
    length and its width the glyph, both with a :data:`NAME_TRIM_PAD` margin.

    A degenerate node (a ~10px broken-line stub near a page seam) has almost no
    ink; its box is still non-empty. Such nodes are surfaced by
    :func:`verify_nodes`.

    Args:
        node: The line node whose name to crop (``top`` and ``bot`` set).
        a: The graph's binary ink grid.

    Returns:
        The cropped name image (binary ink grid), fixed-box sized; never empty.
    """
    left, top, right, bottom = name_box_coords(node, a)
    return a[top:bottom, left:right]


def _is_empty_name(node: LineNode, a: np.ndarray) -> bool:
    """True if the node's name crop is essentially blank -- a phantom/orphan node.

    A cross-page connector broken at a seam leaves a tiny endpoint that reads as a
    node but has no name under it. Threshold mirrors the QA's old check.
    """
    if node.top is None or node.bot is None:
        return True
    return int((1 - get_name_image(node, a)).sum()) < BLANK_NAME_MAX_INK


def build_parse_sidecar(
    nodes: list[LineNode], a: np.ndarray, scrubbed: list[list[int]] | None = None
) -> dict:
    """The reviewable parse geometry for one graph, for the QA overlay to read.

    Records, per node: its name ``box`` ``[left, top, right, bottom]`` (from
    :func:`name_box_coords`, the same box :func:`get_name_image` crops), its
    ``top``/``bot`` endpoints, whether it is ``empty`` (a nameless orphan), and
    its children's ``top`` points so the QA can draw parent->child edges. Also
    records ``scrubbed`` -- the ``[r0, c0, r1, c1]`` boxes of stray pen marks
    blanked before parsing (the graph's ``ignore_regions`` in its
    :class:`BookConfig`) -- so the QA shows what was removed. This is the single
    source of truth: the QA draws exactly what Stage 5 produced.
    """
    node_recs = []
    for n in nodes:
        if n.top is None or n.bot is None:
            continue
        left, top, right, bottom = name_box_coords(n, a)
        node_recs.append({
            "id": n.id,
            "box": [int(left), int(top), int(right), int(bottom)],
            "top": [int(n.top[0]), int(n.top[1])],
            "bot": [int(n.bot[0]), int(n.bot[1])],
            "empty": _is_empty_name(n, a),
            "children_top": [
                [int(c.top[0]), int(c.top[1])] for c in n.children if c.top is not None
            ],
        })
    return {"nodes": node_recs, "scrubbed": scrubbed or []}


# ---------------------------------------------------------------------------
# Orphan bridging: repair generation bars broken at a page seam.
#
# Stage 4 pastes a subtree's pages side by side. A horizontal generation bar that
# continues from one page onto the next loses the segment the page break ate, so
# its left piece floats: the children hanging off it parse fine, but the bar
# connects to no parent and comes out as an EMPTY node with children -- a
# "bar-orphan" -- and everything under it becomes its own root. William's
# trace-right rule (docs/bridge-ground-truth.md) redraws the missing segment:
# trace the orphan bar RIGHT through hairline scan nicks to its true end, cross
# the one page-break gap to the first ink of the next bar, and fill so the result
# is one traversable line. A candidate is accepted only if the graph still parses
# and the orphan count strictly drops. Ported from the v0 pass (src/v0/segment.py)
# with every pixel constant scaled by V1_SCALE; drawn bridges are recorded to
# ``{stem}.imaginary.json`` so the QA overlay can show them green.
# ---------------------------------------------------------------------------

# Vertical drift tolerance when a bridge looks for the bar it reconnects to: page-
# to-page scan drift can put the next bar this many rows off the orphan's row.
BRIDGE_CONNECT_YTOL = int(round(20 * V1_SCALE))  # 60px
# Largest gap hopped while tracing a bar to its true end: a gap this small is a
# scan nick that does not really break the bar; the first larger gap is the
# page break the bridge fills.
BAR_TRACE_HOP = int(round(40 * V1_SCALE))         # 120px
# A real page-break bridge is always wider than the hop the trace stopped at.
MIN_BRIDGE_SPAN = BAR_TRACE_HOP + 1
# A reconnection landing this close to the graph's right edge means the bar's
# parent is on an adjacent GRAPH (resolved at stitching), not a missing bar here.
GRAPH_EDGE_MARGIN = int(round(40 * V1_SCALE))     # 120px
# Half-height of the row band treated as "the bar" when reading its ink runs.
BAR_BAND_HALF = int(round(8 * V1_SCALE))          # 24px
# How far past a bridge end to look for the bar ink the bridge must touch.
BAR_INK_REACH = int(round(12 * V1_SCALE))         # 36px
# Column slack when matching an orphan's reported column to its bar run.
BAR_RUN_SLACK = int(round(5 * V1_SCALE))          # 15px
# A bar broken across several page breaks is closed by extending one bridge to
# successive reconnections; bound the extension (real bars span <= 17 pages).
MAX_BRIDGE_EXTENSIONS = 8


def _ink_band(a: np.ndarray, row: int, half: int) -> np.ndarray:
    """Per-column: is there any ink within ``half`` rows of ``row``?"""
    lo, hi = max(0, row - half), min(a.shape[0], row + half + 1)
    return (1 - a[lo:hi, :]).sum(axis=0) > 0


def _bar_row_runs(a: np.ndarray, row: int) -> list[tuple[int, int]]:
    """Horizontal ink runs ``(c0, c1)`` in the bar band around ``row``."""
    present = _ink_band(a, row, BAR_BAND_HALF)
    cols = np.where(present)[0]
    if len(cols) == 0:
        return []
    breaks = np.where(np.diff(cols) > 1)[0]
    starts = np.concatenate([[cols[0]], cols[breaks + 1]])
    stops = np.concatenate([cols[breaks], [cols[-1]]])
    return [(int(s), int(e)) for s, e in zip(starts, stops)]


def find_orphans(nodes: list[LineNode], a: np.ndarray) -> list[LineNode]:
    """Bar-orphans: nodes with children whose own name crop is blank."""
    return [
        n for n in nodes
        if n.top is not None and n.bot is not None and n.children
        and _is_empty_name(n, a)
    ]


def bar_true_end(a: np.ndarray, row: int, start_x: int) -> int:
    """Rightmost column the bar at ``row`` reaches from ``start_x``, hopping nicks.

    Walks right through ink within :data:`BRIDGE_CONNECT_YTOL` rows, hopping any
    gap of at most :data:`BAR_TRACE_HOP` columns (a scan nick), and stops at the
    first wider gap -- the page break. Returns the last ink column reached.
    """
    band = _ink_band(a, row, BRIDGE_CONNECT_YTOL)
    w = a.shape[1]
    x = min(start_x, w - 1)
    while x < w:
        if band[x]:
            x += 1
            continue
        j = x
        while j < w and not band[j]:
            j += 1
        if j - x > BAR_TRACE_HOP:
            break
        x = j
    return x - 1


def first_ink_right(a: np.ndarray, row: int, start_x: int) -> int | None:
    """First ink column past the ink at ``start_x`` and the gap after it.

    Skips the bar's own trailing ink, then the empty page-break gap, and returns
    the first column of the next ink within the drift band -- the leftmost pixel
    of the bar the bridge reconnects to. ``None`` if the scan runs off the edge.
    """
    band = _ink_band(a, row, BRIDGE_CONNECT_YTOL)
    w = a.shape[1]
    x = max(0, start_x)
    while x < w and band[x]:
        x += 1
    while x < w and not band[x]:
        x += 1
    return int(x) if x < w else None


def _bar_ink_y(a: np.ndarray, row: int, x: int, look: str) -> int | None:
    """Row of the bar ink just outside a bridge end (``look`` = 'left'/'right').

    Drift can put a bar a few rows off ``row``; the drawn bridge must touch it, so
    scan the columns just beyond the end for the ink row nearest ``row``.
    """
    if look == "right":
        xs = range(x + 1, x + 1 + BAR_INK_REACH)
    else:
        xs = range(x - 1, x - 1 - BAR_INK_REACH, -1)
    h, w = a.shape
    for xx in xs:
        if not 0 <= xx < w:
            continue
        for dy in range(0, BRIDGE_CONNECT_YTOL + 1):
            for y in (row + dy, row - dy):
                if 0 <= y < h and a[y, xx] == 0:
                    return y
    return None


def bridge_candidate(
    a: np.ndarray, row: int, col: int
) -> tuple[int, int, int] | None:
    """The bridge ``(row, true_end, connect)`` for the bar-orphan at ``(row, col)``.

    Anchors at the orphan bar's true right end and reaches across exactly one
    page-break gap to the next bar. ``None`` when there is nothing to reach, when
    the landing is the graph's own right edge (a cross-graph orphan), or when
    the span is too short to be a page break.
    """
    runs = _bar_row_runs(a, row)
    bar_run = next(
        (r for r in runs if r[0] - BAR_RUN_SLACK <= col <= r[1] + BAR_RUN_SLACK), None
    )
    if bar_run is None:
        return None
    true_end = bar_true_end(a, row, bar_run[1])
    connect = first_ink_right(a, row, true_end)
    if connect is None or connect >= a.shape[1] - GRAPH_EDGE_MARGIN:
        return None
    if connect - true_end < MIN_BRIDGE_SPAN:
        return None
    return (row, true_end, connect)


def draw_bridge(a: np.ndarray, row: int, c0: int, c1: int) -> np.ndarray:
    """Copy of ``a`` with the bridge ``c0..c1`` filled so it touches both bars.

    Fills the full row range spanned by ``row`` and the bar ink found just
    outside each end, so drifted bars and the bridge form one traversable line.
    """
    ys = [row]
    for y in (_bar_ink_y(a, row, c0, "left"), _bar_ink_y(a, row, c1, "right")):
        if y is not None:
            ys.append(y)
    out = a.copy()
    out[max(0, min(ys) - 2): min(a.shape[0], max(ys) + 3), c0: c1 + 1] = 0
    return out


def bridge_orphans(
    a: np.ndarray, config: BookConfig
) -> tuple[np.ndarray, list[list[int]]]:
    """Repair the bar-orphans of one merged graph.

    Re-derives the orphans each pass and, for each, proposes the trace-right
    bridge. A bridge is accepted only if the graph still parses (no two-parent
    weld) and the orphan count strictly drops; if one hop does not drop it, the
    same bridge is extended to successive reconnections (a bar broken across
    several page breaks) up to :data:`MAX_BRIDGE_EXTENSIONS` times. The strict-
    drop gate is what rejects a bridge that would weld two subtrees and any
    runaway extension across an already-connected bar.

    Args:
        a: The graph's binary ink grid (ignore regions already blanked).
        config: The book's parse config, for the re-parse gate.

    Returns:
        ``(bridged, imaginary)``: the repaired grid and the bridges drawn, each
        ``[r0, c0, r1, c1]`` in graph pixel coordinates.
    """

    def orphan_count(grid: np.ndarray) -> int:
        return len(find_orphans(parse_graph(grid, config), grid))

    out = a
    imaginary: list[list[int]] = []
    attempted: set[tuple[int, int, int]] = set()
    while True:
        base = orphan_count(out)
        if base == 0:
            break
        progressed = False
        for orphan in find_orphans(parse_graph(out, config), out):
            assert orphan.top is not None and orphan.bot is not None
            row, col = orphan.bot[0], orphan.top[1]
            cand = bridge_candidate(out, row, col)
            if cand is None or cand in attempted:
                continue
            attempted.add(cand)
            r, c0, end = cand
            accepted = None
            for _ext in range(MAX_BRIDGE_EXTENSIONS):
                trial = draw_bridge(out, r, c0, end)
                try:
                    if orphan_count(trial) < base:
                        accepted = (trial, end)
                        break
                except ValueError:
                    break  # two-parent weld -> stop extending
                nxt = first_ink_right(out, r, bar_true_end(out, r, end))
                if nxt is None or nxt >= out.shape[1] - GRAPH_EDGE_MARGIN:
                    break
                end = nxt
            if accepted is None:
                continue
            out, end = accepted
            imaginary.append([int(r), int(c0), int(r), int(end)])
            logger.info("orphan-bridge: row=%d cols %d..%d", r, c0, end)
            progressed = True
            break
        if not progressed:
            break
    return out, imaginary


def _is_multipage(graph_stem: str) -> bool:
    """``"67_68"`` -> True, ``"6_6"`` -> False. Only multi-page graphs have seams."""
    start, end = graph_stem.split("_")
    return start != end


def _graph_files(graphs_dir: str) -> list[str]:
    """Return the graph PNG filenames sorted by their starting page index."""
    files = [f for f in os.listdir(graphs_dir) if f.endswith(".png")]
    return sorted(files, key=lambda x: int(x.split("_")[0]))


def parse_graph(
    a: np.ndarray, config: BookConfig, graph_stem: str | None = None
) -> list[LineNode]:
    """Parse one graph image into ordered, ID-less line nodes.

    Runs the full per-graph pipeline: find line segments, read their endpoints,
    build and merge nodes, infer missing ends, run the geometry and grid
    consistency checks, and order the nodes RTL / eldest-first. IDs are assigned
    later, globally across all graphs.

    Args:
        a: The graph's binary ink grid.
        config: The book's Stage-5 config.
        graph_stem: The graph's filename stem (e.g. ``"8_10"``), used to look up
            any :attr:`BookConfig.ignore_regions` to blank first. Omit when the
            graph has no configured ignore regions.

    Returns:
        The graph's line nodes, ordered eldest-first within each generation band.
    """
    if graph_stem is not None:
        a = apply_ignore_regions(a, graph_stem, config)
    raw_lines = find_lines(a, threshold=config.line_threshold)
    line_ends: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for component in raw_lines:
        parents, children = find_line_ends(component, threshold=config.end_threshold)
        if len(parents) != 1:
            raise ValueError(f"expected exactly one parent endpoint, got {parents}")
        line_ends.append((parents[0], children))

    nodes = build_nodes(line_ends)
    logger.debug("nodes pre-merge: %d", len(nodes))
    nodes = merge_nodes(nodes, config)
    logger.debug("nodes post-merge: %d", len(nodes))

    infer_ends(nodes, a)
    nodes = drop_blank_leaf_nodes(nodes, a)
    verify_nodes(nodes, config)
    check_grid_consistency(nodes, config)

    return sort_nodes(nodes)


def build_tree(
    book: str,
    books_dir: str = "books",
    data_dir: str = "data",
    config: BookConfig | None = None,
    graphs_dir_name: str = "4_graphs",
    names_dir_name: str = "5_names",
    data_stem: str | None = None,
) -> list[Node]:
    """Parse every subtree graph of ``book`` into domain nodes and name crops.

    For each ``graphs/{start}_{end}.png`` (in page order) this parses the graph,
    assigns globally-unique IDs in RTL / BFS-by-generation order, crops each name
    image, infers generations by walking down from each root (root = 1), and
    writes:

    * ``{data_dir}/{book}.jsonl`` -- one domain :class:`src.model.Node` per line.
    * ``{books_dir}/{book}/names/{id}.png`` -- the name crop for each node.

    Args:
        book: Book name, e.g. ``"book1"``. Looks up :data:`BOOK_CONFIGS` when
            ``config`` is omitted, and locates the book directory.
        books_dir: Root directory containing per-book asset folders.
        data_dir: Directory for the output JSONL.
        config: Explicit config, overriding the :data:`BOOK_CONFIGS` lookup.

    Returns:
        Every domain :class:`src.model.Node` written, in ID order.
    """
    if config is None:
        config = config_for(book)

    graphs_dir = os.path.join(books_dir, book, graphs_dir_name)
    names_dir = os.path.join(books_dir, book, names_dir_name)
    os.makedirs(names_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    graph_files = _graph_files(graphs_dir)
    logger.info("Building tree for %s from %d graphs", book, len(graph_files))

    domain_nodes: list[Node] = []
    father_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    # Each entry: (node_id, graph binary grid, LineNode, provenance) so we can
    # crop names after every node has an ID and infer generations tree-wide.
    # ``provenance`` is "{graph}_{local_index}" (e.g. "13_16_54"), recording which
    # Stage-4 graph and within-graph position a node came from -- invaluable for
    # tracing a suspect node back to its source graph when debugging mis-merges.
    node_records: list[tuple[int, np.ndarray, LineNode, str]] = []

    node_idx = 1
    total_sus = 0
    total_grid = 0

    for filepath in graph_files:
        filename = os.path.splitext(filepath)[0]
        a = get_image(os.path.join(graphs_dir, filepath))
        a = apply_ignore_regions(a, filename, config)
        # Record the blanked pen-mark rectangles so the QA draws them (magenta):
        # the ink is gone from the parse, but the reviewer still sees what was removed.
        scrubbed = [list(r) for r in config.ignore_regions.get(filename, [])]
        logger.info("Parsing graph %s", filepath)

        # Repair generation bars broken at page seams (multi-page graphs only --
        # a single page has no seam, so Book 1 is untouched). The drawn bridges go
        # to a sidecar the QA overlay renders green; a stale sidecar is removed.
        imaginary: list[list[int]] = []
        if _is_multipage(filename):
            a, imaginary = bridge_orphans(a, config)
            if imaginary:
                logger.info("%s: bridged %d orphan bar(s)", filepath, len(imaginary))
        imaginary_path = os.path.join(graphs_dir, f"{filename}.imaginary.json")
        if imaginary:
            with open(imaginary_path, "w") as fh:
                json.dump(imaginary, fh)
        elif os.path.exists(imaginary_path):
            os.remove(imaginary_path)

        raw_lines = find_lines(a, threshold=config.line_threshold)
        line_ends: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for component in raw_lines:
            parents, children = find_line_ends(
                component, threshold=config.end_threshold
            )
            if len(parents) != 1:
                raise ValueError(
                    f"{filepath}: expected exactly one parent endpoint, got {parents}"
                )
            line_ends.append((parents[0], children))

        nodes = build_nodes(line_ends)
        logger.debug("%s: nodes pre-merge %d", filepath, len(nodes))
        nodes = merge_nodes(nodes, config)
        logger.debug("%s: nodes post-merge %d", filepath, len(nodes))

        infer_ends(nodes, a)
        nodes = drop_blank_leaf_nodes(nodes, a)
        total_sus += len(verify_nodes(nodes, config))
        total_grid += len(check_grid_consistency(nodes, config))

        # Order RTL / eldest-first, then assign globally-unique BFS-by-generation
        # IDs. Graphs are processed in page order and each graph's nodes are
        # already in generation-band order, so a global running counter yields
        # BFS-by-generation, right-to-left ID assignment.
        nodes = sort_nodes(nodes)
        for n in nodes:
            n.id = node_idx
            node_idx += 1

        # Record parent/child relations by ID. ``local_index`` is the node's
        # position within this graph (post-sort, so RTL/eldest-first order).
        # A child reference can dangle when ``merge_nodes`` folded that child
        # into another node (its object left the flat list, so it never got an
        # id). Skip such stale references rather than crash -- they are an
        # artifact of a noisier parse (common on the bitonal variant, whose 1-bit
        # thresholding fragments connector lines) and the real edge survives via
        # the node the child was merged into.
        id_set = {id(n) for n in nodes}
        for local_index, n in enumerate(nodes):
            assert n.id is not None
            live_children = [c for c in n.children if id(c) in id_set and c.id is not None]
            if len(live_children) != len(n.children):
                logger.warning(
                    "graph %s node %d: dropped %d dangling child ref(s)",
                    filename,
                    n.id,
                    len(n.children) - len(live_children),
                )
            children_of[n.id] = [c.id for c in live_children]
            for c in live_children:
                father_of[c.id] = n.id
            node_records.append((n.id, a, n, f"{filename}_{local_index}"))

        # Write the parse sidecar for this graph -- the QA overlay reads it
        # instead of re-running the parser, so it draws exactly what Stage 5 made.
        sidecar_path = os.path.join(graphs_dir, f"{filename}.parse.json")
        with open(sidecar_path, "w") as fh:
            json.dump(build_parse_sidecar(nodes, a, scrubbed), fh)

    # Infer generations: each root (no father) is generation 1; every child is
    # one generation deeper. Nodes unreachable from any root keep -1.
    generation_of = _infer_generations(children_of, father_of)

    # Emit domain nodes and crop name images.
    data_path = os.path.join(data_dir, f"{data_stem or book}.jsonl")
    with open(data_path, "w") as data_file:
        for node_id, grid, line_node, provenance in node_records:
            name_img = get_name_image(line_node, grid)
            name_path = os.path.join(names_dir, f"{node_id}.png")
            save_image(name_img, name_path)

            node = Node(
                id=node_id,
                name_images=[name_path],
                generation=generation_of.get(node_id, -1),
                father=father_of.get(node_id, -1),
                children=children_of.get(node_id, []),
                notes=provenance,
            )
            data_file.write(json.dumps(dataclasses.asdict(node), ensure_ascii=False) + "\n")
            domain_nodes.append(node)

    logger.info(
        "Wrote %d nodes -> %s (%d sus-node warnings, %d grid warnings)",
        len(domain_nodes),
        data_path,
        total_sus,
        total_grid,
    )
    return domain_nodes


def _infer_generations(
    children_of: dict[int, list[int]], father_of: dict[int, int]
) -> dict[int, int]:
    """Assign a 1-indexed generation to every node by walking down from roots.

    A node with no father is a root at generation 1; each child is one deeper.
    Uses a BFS from every root. Nodes not reachable from any root keep no entry
    (the caller treats a missing entry as generation ``-1``).

    Args:
        children_of: ``node_id -> list of child ids``.
        father_of: ``node_id -> father id`` (missing key means root).

    Returns:
        ``node_id -> generation`` for every reachable node.
    """
    all_ids = set(children_of) | {c for cs in children_of.values() for c in cs}
    roots = [nid for nid in all_ids if nid not in father_of]

    generation_of: dict[int, int] = {}
    queue: list[tuple[int, int]] = [(r, 1) for r in roots]
    while queue:
        node_id, gen = queue.pop(0)
        # Keep the shallowest generation if somehow reached twice.
        if node_id in generation_of:
            continue
        generation_of[node_id] = gen
        for child in children_of.get(node_id, []):
            queue.append((child, gen + 1))
    return generation_of


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        required=True,
        help="Book to process (e.g. book1). Uses BOOK_CONFIGS[book] if present, "
             "else the shared v1 defaults.",
    )
    parser.add_argument(
        "--books-dir",
        default="books",
        help="Root directory containing per-book asset folders (default: books).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for the output JSONL (default: data).",
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
    nodes = build_tree(args.book, books_dir=args.books_dir, data_dir=args.data_dir)
    logger.info("Done. %d nodes written for %s.", len(nodes), args.book)


if __name__ == "__main__":
    main()
