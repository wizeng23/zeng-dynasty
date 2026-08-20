"""Prototype + visualize orphan-bridge detection WITHOUT touching the pipeline.

An orphan is an empty phantom node that still has children: a horizontal sibling
bar whose parent connection is broken, so its children hang off a nameless stub.
For the PAGE-GAP subtype the bar is literally cut by a missing-line page; the true
continuation (and the parent riser) lie to the right across the gap.

This probe, per graph:
  * parses (read-only) and finds orphan bars (empty node, >=1 child),
  * for each, traces its bar row rightward to the next ink after the gap,
  * proposes a green bridge segment [r0,c0,r1,c1] if the trace is safe,
  * renders the graph with RED parse + ORANGE orphan flags + GREEN proposed
    bridges, so each proposed connection can be eyeballed before it is trusted.

Nothing is written back into the graph; this is purely diagnostic. Output PNGs +
an index go to worklog so there is a paper trail of what the algorithm proposes.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

from src import build_tree as bt
from src.imaging import get_image

logger = logging.getLogger(__name__)

RED = (220, 30, 30)
ORANGE = (240, 140, 0)
GREEN = (20, 170, 60)
BLUE = (40, 90, 220)


def _is_empty(node: bt.LineNode, a: np.ndarray) -> bool:
    if node.top is None or node.bot is None:
        return True
    return int((1 - bt.get_name_image(node, a)).sum()) < 30


def bar_row_ink_runs(a: np.ndarray, row: int, half: int = 8) -> list[tuple[int, int]]:
    """Horizontal ink runs in the +-half band around ``row`` (weave-tolerant)."""
    lo, hi = max(0, row - half), min(a.shape[0], row + half + 1)
    present = (1 - a[lo:hi, :]).sum(axis=0) > 0
    runs: list[tuple[int, int]] = []
    c = 0
    n = len(present)
    while c < n:
        if present[c]:
            s = c
            while c < n and present[c]:
                c += 1
            runs.append((s, c - 1))
        else:
            c += 1
    return runs


def has_vertical_between(a: np.ndarray, row: int, c0: int, c1: int, reach: int = 120) -> int:
    """Count columns in (c0,c1) carrying a tall vertical crossing the bar row.

    A real page-gap is empty; if a vertical line sits inside the gap the trace
    would be crossing another branch, so bridging there is unsafe. Returns the
    number of offending columns (0 == safe to bridge).
    """
    lo, hi = max(0, row - reach), min(a.shape[0], row + reach)
    seg = a[lo:hi, c0 + 1 : c1]
    if seg.size == 0:
        return 0
    col_ink = (1 - seg).sum(axis=0)
    # a vertical crossing shows a tall ink column (most of the 2*reach band)
    return int((col_ink > reach).sum())


def find_orphan_bridges(a: np.ndarray, cfg: bt.BookConfig) -> tuple[list, list]:
    """Return (orphans, bridges). Each orphan: (idx, node). Each bridge:
    (idx, [r, c_gap_start, r, c_gap_end], status_str)."""
    nodes = bt.parse_graph(a, cfg)
    orphans = []
    bridges = []
    for i, n in enumerate(nodes):
        if not _is_empty(n, a) or not n.children:
            continue
        orphans.append((i, n))
        row = n.bot[0]  # fan-out (bar) row
        col = n.top[1]
        runs = bar_row_ink_runs(a, row)
        # the run containing the bar
        bar_run = next((r for r in runs if r[0] - 5 <= col <= r[1] + 5), None)
        if bar_run is None:
            bridges.append((i, None, "no-bar-run"))
            continue
        # first run strictly to the right of the bar run
        right_runs = [r for r in runs if r[0] > bar_run[1] + 2]
        if not right_runs:
            bridges.append((i, None, "no-right-continuation (left-edge?)"))
            continue
        nxt = right_runs[0]
        gap = nxt[0] - bar_run[1]
        # guardrails: gap must be page-scale (not a within-bar hairline) and not
        # cross a vertical branch line.
        if gap < 40:
            bridges.append((i, None, f"gap-too-small({gap})"))
            continue
        crossings = has_vertical_between(a, row, bar_run[1], nxt[0])
        status = "OK" if crossings == 0 else f"UNSAFE({crossings} verticals in gap)"
        bridges.append((i, [row, bar_run[1], row, nxt[0]], status))
    return nodes, orphans, bridges


def render(a, nodes, orphans, bridges, out_path):
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    # red edges
    for n in nodes:
        if n.bot is None or not n.children:
            continue
        pr, pc = n.bot
        for child in n.children:
            if child.top is None:
                continue
            cr, cc = child.top
            draw.line([(pc, pr), (pc, pr)], fill=RED, width=3)
            draw.line([(pc, pr), (cc, pr)], fill=RED, width=3)
            draw.line([(cc, pr), (cc, cr)], fill=RED, width=3)
    # orange orphan flags
    for i, n in orphans:
        cx, cy = n.top[1], (n.top[0] + n.bot[0]) // 2
        draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline=ORANGE, width=6)
    # green proposed bridges
    for i, seg, status in bridges:
        if seg is None:
            continue
        r, c0, _r1, c1 = seg
        color = GREEN if status == "OK" else BLUE
        draw.line([(c0, r), (c1, r)], fill=color, width=6)
        draw.ellipse([c1 - 12, r - 12, c1 + 12, r + 12], outline=color, width=4)
    img.save(out_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", default="book2")
    ap.add_argument("--out", default="worklog/2026-08-20/orphan_probe")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("src.build_tree").setLevel(logging.ERROR)
    cfg = bt.BOOK_CONFIGS[args.book]
    graphs_dir = os.path.join("books", args.book, "graphs")
    os.makedirs(args.out, exist_ok=True)
    affected = ["8_10", "11_17", "22_23", "28_30", "31_35", "36_52",
                "58_62", "69_82", "114_120", "121_122", "126_128"]
    summary = []
    for stem in affected:
        a = get_image(os.path.join(graphs_dir, f"{stem}.png"))
        nodes, orphans, bridges = find_orphan_bridges(a, cfg)
        render(a, nodes, orphans, bridges, os.path.join(args.out, f"{stem}.png"))
        for i, seg, status in bridges:
            summary.append((stem, i, status, seg))
            logger.info("%s idx=%d: %s %s", stem, i, status, seg)
    ok = sum(1 for s in summary if s[2] == "OK")
    logger.info("\nProposed %d bridges, %d OK-safe", len(summary), ok)


if __name__ == "__main__":
    main()
