"""Stage 4 (v1 scans): per-page tree crops -> merged subtree-graph images.

The previous step (:mod:`src.segment`, Stage 3) cropped each page down to its tree and
recorded which pages start a subtree (``crops/starts.json``). A subtree can
span several consecutive pages; this step stitches each run of continuation pages
onto the page that started the subtree, aligning the dangling lines at each seam,
and writes one image per subtree to ``books/{book}/graphs/{start}_{end}.png``.

Physically stitching at the seam lets the Stage-5 parser (:mod:`src.build_tree`)
treat each subtree as one connected graph. The book reads right-to-left, so
``merge_graphs`` places the newer (left) page on the left of the accumulated graph.

Pixel constants are scaled from the v0 pipeline by :data:`src.segment.SCALE`.
(Orphan-bridging -- repairing generation bars broken across a page seam -- is a
later concern and is not done here.)

CLI:
    python -m src.merge_pages --book book1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from itertools import combinations

import numpy as np

from src.imaging import get_image, save_image
from src.s3_segment import BOOK_CONFIGS, BookConfig, _s

logger = logging.getLogger(__name__)


def remove_adjacent(numbers: list[int], threshold: int | None = None) -> list[int]:
    """Collapse near-adjacent values, keeping the first of each cluster.

    Line endpoints arrive as thick runs of neighboring rows; this reduces each
    run to one representative. Threshold scaled from v0's 30px.
    """
    if threshold is None:
        threshold = _s(30)
    if not numbers:
        return []
    sorted_nums = sorted(numbers)
    result = [sorted_nums[0]]
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] - sorted_nums[i - 1] > threshold:
            result.append(sorted_nums[i])
    return result


SHIFT_PENALTY_WEIGHT = 0.1


def find_best_orphans(left: list[int], right: list[int]) -> tuple[list[int], str]:
    """Choose which endpoints on the longer side have no partner at the seam.

    Picks the orphan set minimizing the summed |offset| between remaining matched
    endpoints (shift-invariant shape score + a mild shift-magnitude penalty).
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
            shift_penalty = SHIFT_PENALTY_WEIGHT * abs(mean_off)
            score = spread + shift_penalty
        else:
            score = 0.0
        if best_orphan_score is None or score < best_orphan_score:
            best_orphan_score = score
            best_orphans = orphans
    return list(best_orphans), "left" if not flip else "right"


def matched_shift(left: list[int], right: list[int]) -> int:
    """Median pairwise offset to apply to ``left`` so its endpoints meet ``right``."""
    if not left or not right:
        return 0
    offsets = sorted(y - x for x, y in zip(left, right))
    mid = len(offsets) // 2
    if len(offsets) % 2:
        return offsets[mid]
    return int((offsets[mid - 1] + offsets[mid]) / 2)


def _concat_top_aligned(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Stack two graph slices side by side (``g1`` left) with no seam line."""
    if g1.shape[0] < g2.shape[0]:
        padding = g2.shape[0] - g1.shape[0]
        g1 = np.vstack([g1, np.ones((padding, g1.shape[1])).astype(np.uint8)])
    elif g2.shape[0] < g1.shape[0]:
        padding = g1.shape[0] - g2.shape[0]
        g2 = np.vstack([g2, np.ones((padding, g2.shape[1])).astype(np.uint8)])
    return np.hstack([g1, g2])


def merge_graphs(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    """Stitch two subtree-graph slices side by side, ``g1`` on the left.

    Aligns the dangling lines at ``g1``'s right edge with those at ``g2``'s left
    edge (orphans + overall vertical shift), draws the connecting seam, and
    concatenates. Seam-band width scaled from v0's 5px.
    """
    band = _s(5)
    g1_edge = np.sum(1 - g1[:, -band:], axis=1)
    left_y = remove_adjacent([x for x in range(len(g1_edge)) if g1_edge[x] > 0])
    g2_edge = np.sum(1 - g2[:, :band], axis=1)
    right_y = remove_adjacent([x for x in range(len(g2_edge)) if g2_edge[x] > 0])
    logger.debug("seam endpoints: left=%s right=%s", left_y, right_y)

    if not left_y or not right_y:
        logger.warning(
            "empty seam (left=%d right=%d endpoints); concatenating without a "
            "connecting line", len(left_y), len(right_y),
        )
        return _concat_top_aligned(g1, g2)

    orphan_idxs, side = find_best_orphans(left_y, right_y)
    if side == "left":
        orphans = [left_y[i] for i in orphan_idxs]
        left_y = [left_y[i] for i in range(len(left_y)) if i not in orphan_idxs]
    else:
        orphans = [right_y[i] for i in orphan_idxs]
        right_y = [right_y[i] for i in range(len(right_y)) if i not in orphan_idxs]
    if orphan_idxs:
        logger.warning("orphan endpoints %s on %s side of seam", orphans, side)

    best_alignment = matched_shift(left_y, right_y)
    logger.debug("matched-pair shift: %d", best_alignment)
    if best_alignment > 0:
        g1 = np.vstack([np.ones((best_alignment, g1.shape[1])).astype(np.uint8), g1])
        left_y = [x + best_alignment for x in left_y]
    elif best_alignment < 0:
        g2 = np.vstack([np.ones((-best_alignment, g2.shape[1])).astype(np.uint8), g2])
        right_y = [x - best_alignment for x in right_y]

    if g1.shape[0] < g2.shape[0]:
        padding = g2.shape[0] - g1.shape[0]
        g1 = np.vstack([g1, np.ones((padding, g1.shape[1])).astype(np.uint8)])
    elif g2.shape[0] < g1.shape[0]:
        padding = g1.shape[0] - g2.shape[0]
        g2 = np.vstack([g2, np.ones((padding, g2.shape[1])).astype(np.uint8)])

    for left, right in zip(left_y, right_y):
        low, high = min(left, right), max(left, right)
        g1[low : high + 1, -1:] = 0
        g2[low : high + 1, :1] = 0

    return np.hstack([g1, g2])


def merge_pages(
    book: str,
    books_dir: str = "books",
    config: BookConfig | None = None,
    crops_dir_name: str = "3_crops",
    graphs_dir_name: str = "4_graphs",
) -> list[str]:
    """Merge each subtree's per-page crops into one graph image.

    Reads ``{books_dir}/{book}/{crops_dir_name}/{i}.png`` and ``starts.json``
    (from :mod:`src.segment`), stitches each run of continuation pages onto its
    start page, and writes ``{graphs_dir_name}/{start}_{end}.png``.
    """
    if config is None:
        if book not in BOOK_CONFIGS:
            raise KeyError(
                f"no BookConfig for {book!r}; known books: {sorted(BOOK_CONFIGS)}"
            )
        config = BOOK_CONFIGS[book]

    crops_dir = os.path.join(books_dir, book, crops_dir_name)
    graphs_dir = os.path.join(books_dir, book, graphs_dir_name)
    os.makedirs(graphs_dir, exist_ok=True)

    with open(os.path.join(crops_dir, "starts.json")) as fh:
        starts = json.load(fh)
    # Only the pages Stage 3 cropped appear in starts.json; biography pages (Books
    # 3 & 4) were skipped and have no crop. Iterate the cropped pages in order --
    # for the all-tree Books 1 & 2 this is simply 0..num_pages-1.
    page_indices = sorted(int(k) for k in starts)
    is_start = {i: bool(starts[str(i)]) for i in page_indices}
    pages = {i: get_image(os.path.join(crops_dir, f"{i}.png")) for i in page_indices}

    logger.info("Merging %s: %d page crops -> %s", book, len(page_indices), graphs_dir)

    written: list[str] = []
    frame = _s(100)
    n = len(page_indices)
    j = 0
    while j < n:
        start_i = page_indices[j]
        graph = pages[start_i]
        j += 1
        while j < n and not is_start[page_indices[j]]:
            cont = page_indices[j]
            logger.info("Merging page %d into subtree started at %d", cont, start_i)
            graph = merge_graphs(pages[cont], graph)
            j += 1

        end_i = page_indices[j - 1]
        graph = np.vstack([
            np.ones((frame, graph.shape[1])).astype(np.uint8),
            graph,
            np.ones((frame, graph.shape[1])).astype(np.uint8),
        ])

        stem = f"{start_i}_{end_i}"
        out_path = os.path.join(graphs_dir, f"{stem}.png")
        save_image(graph, out_path)
        logger.info("Wrote subtree %d..%d -> %s", start_i, end_i, out_path)
        written.append(out_path)

    logger.info("Merged %d subtrees for %s", len(written), book)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, choices=sorted(BOOK_CONFIGS))
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--crops-dir", default="3_crops")
    parser.add_argument("--graphs-dir", default="4_graphs")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    written = merge_pages(args.book, books_dir=args.books_dir,
                          crops_dir_name=args.crops_dir, graphs_dir_name=args.graphs_dir)
    logger.info("Done. %d subtree graphs written for %s.", len(written), args.book)


if __name__ == "__main__":
    main()
