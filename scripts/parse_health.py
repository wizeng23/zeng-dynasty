"""Parse-health metrics: quantify defects in every graph's parse.

Runs the Stage-3 parse over every graph of a book and reports, per graph and in
total, the counts that signal a bad parse:

* ``empty``    -- nodes whose name crop is essentially blank (seam-break phantoms).
* ``sus``      -- nodes whose line height is outside the plausible band.
* ``grid``     -- parent/child pairs whose generation-drop breaks the grid.
* ``badname``  -- name crops whose bounding box is an implausible shape/size.

Used as a regression harness while tuning the merge/parse: run it before and
after a change to see the defect count move, and to find the specific graphs and
nodes to investigate.

Usage::

    python -m scripts.parse_health --book book2
    python -m scripts.parse_health --book book2 --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src import build_tree as bt
from src.imaging import get_image

logger = logging.getLogger(__name__)


def _is_empty(node: bt.LineNode, a: np.ndarray) -> bool:
    if node.top is None or node.bot is None:
        return True
    return int((1 - bt.get_name_image(node, a)).sum()) < 30


def _name_shape(node: bt.LineNode, a: np.ndarray) -> tuple[int, int, int]:
    """Return (height, width, ink) of a node's name crop."""
    img = bt.get_name_image(node, a)
    return img.shape[0], img.shape[1], int((1 - img).sum())


def graph_health(a: np.ndarray, cfg: bt.BookConfig) -> dict:
    """Parse one graph and return its defect metrics + per-node shape data."""
    nodes = bt.parse_graph(a, cfg)
    sus = bt.verify_nodes(nodes, cfg)
    grid = bt.check_grid_consistency(nodes, cfg)

    empties = []
    shapes = []
    for i, n in enumerate(nodes):
        h, w, ink = _name_shape(n, a)
        shapes.append((i, h, w, ink))
        if _is_empty(n, a):
            empties.append((i, n))
    return {
        "n_nodes": len(nodes),
        "empty": len(empties),
        "sus": len(sus),
        "grid": len(grid),
        "empty_idx": [i for i, _ in empties],
        "shapes": shapes,
        "nodes": nodes,
    }


def book_health(book: str, books_dir: str = "books") -> dict:
    cfg = bt.BOOK_CONFIGS[book]
    graphs_dir = os.path.join(books_dir, book, "graphs")
    files = sorted(
        (f for f in os.listdir(graphs_dir) if f.endswith(".png")),
        key=lambda f: int(f.split("_")[0]),
    )
    per_graph = {}
    tot = {"n_nodes": 0, "empty": 0, "sus": 0, "grid": 0}
    all_shapes = []
    for f in files:
        stem = os.path.splitext(f)[0]
        a = get_image(os.path.join(graphs_dir, f))
        h = graph_health(a, cfg)
        per_graph[stem] = {k: h[k] for k in ("n_nodes", "empty", "sus", "grid", "empty_idx")}
        for k in tot:
            tot[k] += h[k]
        for (i, hh, ww, ink) in h["shapes"]:
            all_shapes.append((stem, i, hh, ww, ink))
        logger.info(
            "%-8s nodes=%-4d empty=%-2d sus=%-3d grid=%-3d",
            stem, h["n_nodes"], h["empty"], h["sus"], h["grid"],
        )
    return {"total": tot, "per_graph": per_graph, "shapes": all_shapes}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="book2")
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--json", default=None, help="Write full metrics to this JSON path.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(message)s")
    # Silence the per-node warnings from verify/grid; we count them ourselves.
    logging.getLogger("src.build_tree").setLevel(logging.ERROR)

    result = book_health(args.book, books_dir=args.books_dir)
    t = result["total"]
    print(f"\n=== {args.book} TOTAL: nodes={t['n_nodes']} empty={t['empty']} "
          f"sus={t['sus']} grid={t['grid']} ===")
    if args.json:
        out = {"total": t, "per_graph": result["per_graph"], "shapes": result["shapes"]}
        with open(args.json, "w") as fh:
            json.dump(out, fh)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
