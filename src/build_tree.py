"""Stage 3 of the pipeline: merged subtree-graph images -> tree JSONL + name crops.

Each image in ``books/{book}/graphs/{start}_{end}.png`` is one subtree's
line-graph (see ``src/segment.py``). Every name sits on a rigid grid:
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
    """Per-book Stage-3 configuration.

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
    """

    line_threshold: int = 70
    end_threshold: int = 50
    merge_max_drop: int = 200
    merge_max_shift: int = 20
    node_min_height: int = 60
    node_max_height: int = 250
    gen_row_min: int = 280
    gen_row_max: int = 345


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
BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(),
    "book2": BookConfig(),
}


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


def bridge_horizontal_gaps(foreground: np.ndarray, max_gap: int = 9) -> np.ndarray:
    """Fill short horizontal gaps in a binary ink mask (ink == 1).

    The scans' long horizontal connector lines carry occasional tiny breaks
    (faint ink / where a vertical crosses), which split one line into two
    connected components. The broken-off end then reads as a parent hang-point
    with no name above it -- a phantom empty node the real children mis-attach to.

    A background pixel is filled only when it lies in a horizontal gap of at most
    ``max_gap`` columns with ink on BOTH sides in the SAME row -- i.e. a break in
    a continuing horizontal line. This is safe because distinct elements in the
    grid are spaced far apart: measured across Book 2, real horizontal separations
    are >=~50px while line-breaks are <=~19px, a clean valley. Bridging <10px thus
    can only rejoin a broken line -- it never connects two characters (>=50px
    apart) and never touches vertical strokes (this scans row-wise only), so the
    name glyphs are left intact.

    Args:
        foreground: Binary mask, 1 == ink, 0 == background.
        max_gap: Maximum gap width (columns) to bridge.

    Returns:
        A copy of ``foreground`` with qualifying short horizontal gaps filled.
    """
    out = foreground.copy()
    h, w = foreground.shape
    for r in range(h):
        row = foreground[r]
        ink_cols = np.flatnonzero(row)
        if ink_cols.size < 2:
            continue
        # Between consecutive ink pixels, fill the gap if it is short enough.
        prev = ink_cols[0]
        for c in ink_cols[1:]:
            gap = c - prev - 1
            if 0 < gap <= max_gap:
                out[r, prev + 1 : c] = 1
            prev = c
    return out


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
    foreground = bridge_horizontal_gaps(foreground)
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
    return (top_points, bottom_points)


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


def infer_ends(nodes: list[LineNode], a: np.ndarray) -> None:
    """Fill in any node end that ran off the edge of its page.

    A line dangling off the top or bottom of a page leaves a node with a missing
    ``top`` or ``bot``. This walks inward from the known end along the node's
    column until it meets ink, recovering the missing endpoint (with a small pad
    so the later name crop keeps a margin). Mutates ``nodes`` in place.

    Args:
        nodes: The merged line nodes.
        a: The graph's binary ink grid, used to probe for the missing end.
    """
    for n in nodes:
        if n.top is None:
            assert n.bot is not None
            top = max(0, n.bot[0] - 200)
            y = n.bot[1]
            min_y = max(0, y - 30)
            max_y = min(a.shape[1] - 1, y + 30)
            while top < n.bot[0] and 0 not in a[top][min_y:max_y]:
                top += 1
            top = max(0, top - 10)
            n.top = (top, y)
        if n.bot is None:
            assert n.top is not None
            bot = min(a.shape[0] - 1, n.top[0] + 200)
            y = n.top[1]
            min_y = max(0, y - 30)
            max_y = min(a.shape[1] - 1, y + 30)
            while bot > n.top[0] and 0 not in a[bot][min_y:max_y]:
                bot -= 1
            bot = min(bot + 10, a.shape[0] - 1)
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

    Nodes are banded into generation-rows (``top`` rows within 60px of each
    other), then each band is ordered **right-to-left** so the eldest sibling
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
            if abs(nodes_sorted[j].top[0] - nodes_sorted[i].top[0]) < 60:
                group.append(nodes_sorted[j])
                j += 1
            else:
                break
        # Rightmost first (eldest-first) within the generation band.
        group.sort(key=lambda node: -node.top[1])
        result.extend(group)
        i = j
    return result


def get_name_image(node: LineNode, a: np.ndarray) -> np.ndarray:
    """Crop the name-character image sitting at a node's top endpoint.

    Grabs the band between the node's ``top`` and ``bot`` around the node's
    column, pads it, then trims the surrounding whitespace to a tight box around
    the name ink.

    A degenerate node (a short broken-line fragment near a page seam, height
    ``bot - top`` only ~10px) has no name band and no ink to trim to. Rather than
    crash the save on a zero-size crop -- and rather than silently drop the node,
    which would hide a real mis-parse -- this returns whatever padded band it has
    and never collapses either dimension below one pixel. Such nodes are already
    surfaced by :func:`verify_nodes` (their height falls outside the band).

    Args:
        node: The line node whose name to crop (``top`` and ``bot`` set).
        a: The graph's binary ink grid.

    Returns:
        The cropped, tightly-trimmed name image (binary ink grid); never empty.
    """
    assert node.top is not None and node.bot is not None
    y = node.top[1]
    # Clamp the vertical band so a fragment shorter than the 5px insets still
    # yields a non-empty slice before padding.
    top_row = node.top[0] + 5
    bot_row = max(node.bot[0] - 5, top_row + 1)
    name = a[top_row:bot_row, max(0, y - 40) : min(a.shape[1], y + 40)]
    name = pad_image(name, "udlr")

    # Trim left/right whitespace. If the band is all background (a degenerate
    # fragment), keep the padded image rather than collapsing to zero width.
    col_present = np.sum(1 - name, axis=0)
    if np.any(col_present):
        min_y = 0
        while min_y < len(col_present) and not col_present[min_y]:
            min_y += 1
        min_y = max(min_y - 10, 0)
        max_y = len(col_present) - 1
        while max_y > min_y and not col_present[max_y]:
            max_y -= 1
        max_y = min(max_y + 10, len(col_present) - 1)
        name = name[:, min_y:max_y]

    # Trim top/bottom whitespace, with the same empty-band guard.
    row_present = np.sum(1 - name, axis=1)
    if np.any(row_present):
        min_x = 0
        while min_x < len(row_present) and not row_present[min_x]:
            min_x += 1
        min_x = max(min_x - 10, 0)
        max_x = len(row_present) - 1
        while max_x > 0 and not row_present[max_x]:
            max_x -= 1
        max_x = min(max_x + 10, len(row_present) - 1)
        name = name[min_x:max_x, :]
    return name


def _graph_files(graphs_dir: str) -> list[str]:
    """Return the graph PNG filenames sorted by their starting page index."""
    files = [f for f in os.listdir(graphs_dir) if f.endswith(".png")]
    return sorted(files, key=lambda x: int(x.split("_")[0]))


def parse_graph(
    a: np.ndarray, config: BookConfig
) -> list[LineNode]:
    """Parse one graph image into ordered, ID-less line nodes.

    Runs the full per-graph pipeline: find line segments, read their endpoints,
    build and merge nodes, infer missing ends, run the geometry and grid
    consistency checks, and order the nodes RTL / eldest-first. IDs are assigned
    later, globally across all graphs.

    Args:
        a: The graph's binary ink grid.
        config: The book's Stage-3 config.

    Returns:
        The graph's line nodes, ordered eldest-first within each generation band.
    """
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
    verify_nodes(nodes, config)
    check_grid_consistency(nodes, config)

    return sort_nodes(nodes)


def build_tree(
    book: str,
    books_dir: str = "books",
    data_dir: str = "data",
    config: BookConfig | None = None,
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
        if book not in BOOK_CONFIGS:
            raise KeyError(
                f"no BookConfig for {book!r}; known books: {sorted(BOOK_CONFIGS)}"
            )
        config = BOOK_CONFIGS[book]

    graphs_dir = os.path.join(books_dir, book, "graphs")
    names_dir = os.path.join(books_dir, book, "names")
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
    # Stage-2 graph and within-graph position a node came from -- invaluable for
    # tracing a suspect node back to its source graph when debugging mis-merges.
    node_records: list[tuple[int, np.ndarray, LineNode, str]] = []

    node_idx = 1
    total_sus = 0
    total_grid = 0

    for filepath in graph_files:
        filename = os.path.splitext(filepath)[0]
        a = get_image(os.path.join(graphs_dir, filepath))
        logger.info("Parsing graph %s", filepath)

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
        for local_index, n in enumerate(nodes):
            assert n.id is not None
            child_ids = [c.id for c in n.children]
            children_of[n.id] = child_ids
            for c in n.children:
                assert c.id is not None
                father_of[c.id] = n.id
            node_records.append((n.id, a, n, f"{filename}_{local_index}"))

    # Infer generations: each root (no father) is generation 1; every child is
    # one generation deeper. Nodes unreachable from any root keep -1.
    generation_of = _infer_generations(children_of, father_of)

    # Emit domain nodes and crop name images.
    data_path = os.path.join(data_dir, f"{book}.jsonl")
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
        choices=sorted(BOOK_CONFIGS),
        help="Book to process (e.g. book1).",
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
