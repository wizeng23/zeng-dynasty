"""Stage 2 (v1 scans): classify each extracted page as a tree graph or a biography.

Books 3 and 4 interleave two kinds of page: **tree** pages (the family-tree
line-graph, the thing Stage 3 crops and Stage 5 parses) and **biography** pages
(dense columns of names + birth/death prose). Only tree pages belong on the graph
path; the biographies are parsed separately later, but their page numbers must be
remembered so a person's tree node can later be linked to their biography.

Detection keys on the printed **cell grid**. A biography page is ruled into 5
stacked cells by exactly 4 full-width horizontal dividers sitting at *fixed*
vertical positions -- the same layout on every biography page in both books,
measured at y-fractions ``BIO_RULE_YFRAC`` (~0.19, 0.40, 0.60, 0.80). A tree page
either has a different number of full-width rules or places them at a different
template (e.g. Book 2's ruled tree pages rule at ~0.15/0.32/0.48/0.64, and most
tree pages have only the odd fan-out bar, which does not span edge to edge).

So a page is a biography iff it carries exactly the four fixed-position full-width
rules and no *other* full-width interior rule. The page's own top/bottom frame
border (a full-width line at y-fraction ~0 or ~1) is ignored -- it is the page
edge, not a cell divider.

This was validated against hand-labelled pages across all four books: Books 1 & 2
(entirely tree, including their ruled tree pages) classify to **zero** biographies;
Books 3 & 4's biography/tree split matches eyeballed ground truth, and every page
Stage 3 independently detects as a subtree-start comes out ``graph`` (0 conflicts).

Output: ``books/{book}/pages/page_types.json`` -- see :func:`classify_pages`.

CLI:
    python -m src.classify_pages --book book3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import cv2
import numpy as np

from src.imaging import get_image
from src.segment import trim_borders

logger = logging.getLogger(__name__)

# Fixed vertical positions (as a fraction of trimmed page height) of the 4
# full-width cell dividers on a biography page. Measured across biography pages in
# Books 3 and 4; deviation between pages is < 0.01, so the tolerance is generous.
BIO_RULE_YFRAC: tuple[float, ...] = (0.193, 0.396, 0.597, 0.802)
BIO_RULE_TOL = 0.03

# A candidate rule is a row whose ink spans at least this fraction of the width.
RULE_COVERAGE_MIN = 0.55
# Ignore rules within this fraction of the top/bottom edge: those are the page's
# own printed frame border, not an interior cell divider.
FRAME_EDGE_MARGIN = 0.05


def _find_full_width_rules(a: np.ndarray) -> list[float]:
    """Return the y-fractions of near-full-width horizontal rules on the page.

    A row counts as a rule when its ink (dilated slightly to bridge scan blips)
    covers at least :data:`RULE_COVERAGE_MIN` of the page width. Adjacent inked
    rows are collapsed to one representative. Rules within
    :data:`FRAME_EDGE_MARGIN` of the top/bottom edge (the page frame) are dropped.
    """
    ink = (1 - a).astype(np.uint8)
    h, w = ink.shape
    if h == 0 or w == 0:
        return []
    # Dilate vertically a hair so a slightly wavy/broken scan line still reads as
    # continuous ink across the row.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(ink, kernel)
    coverage = dilated.sum(axis=1) / w

    rows: list[int] = []
    for y in np.where(coverage > RULE_COVERAGE_MIN)[0]:
        if not rows or y - rows[-1] > 60:
            rows.append(int(y))

    yfracs = [y / h for y in rows]
    return [f for f in yfracs if FRAME_EDGE_MARGIN < f < 1 - FRAME_EDGE_MARGIN]


def is_biography_page(a: np.ndarray) -> tuple[bool, list[float]]:
    """Classify a trimmed page. Returns ``(is_bio, rule_yfracs)``.

    A biography page has exactly the four fixed-position full-width rules
    (:data:`BIO_RULE_YFRAC`) and no other full-width interior rule.
    """
    yfracs = _find_full_width_rules(a)
    at_fingerprint = [
        f for f in yfracs
        if any(abs(f - b) <= BIO_RULE_TOL for b in BIO_RULE_YFRAC)
    ]
    off_fingerprint = [
        f for f in yfracs
        if all(abs(f - b) > BIO_RULE_TOL for b in BIO_RULE_YFRAC)
    ]
    all_matched = all(
        any(abs(f - b) <= BIO_RULE_TOL for f in yfracs) for b in BIO_RULE_YFRAC
    )
    is_bio = all_matched and len(at_fingerprint) == 4 and not off_fingerprint
    return is_bio, [round(f, 3) for f in yfracs]


def classify_pages(
    book: str,
    num_pages: int,
    books_dir: str = "books",
    pages_dir: str = "pages",
) -> dict:
    """Classify every page of a book as ``graph`` or ``bio``; write the sidecar.

    Reads ``{books_dir}/{book}/{pages_dir}/{i}.png`` for ``i`` in
    ``0..num_pages-1`` and writes ``{pages_dir}/page_types.json`` with, per page,
    its type and the detected rule positions, plus flat ``graph_pages`` /
    ``bio_pages`` lists. Each biography page also records ``follows_graph`` -- the
    index of the nearest preceding tree page -- so it can be re-associated with the
    lineage it details when biographies are parsed later.

    Returns the written sidecar dict.
    """
    pages_path = os.path.join(books_dir, book, pages_dir)
    logger.info("Classifying %s: %d pages in %s", book, num_pages, pages_path)

    pages: dict[str, dict] = {}
    graph_pages: list[int] = []
    bio_pages: list[int] = []
    last_graph: int | None = None
    for i in range(num_pages):
        a = trim_borders(get_image(os.path.join(pages_path, f"{i}.png")))
        is_bio, yfracs = is_biography_page(a)
        entry: dict = {"type": "bio" if is_bio else "graph", "rule_yfrac": yfracs}
        if is_bio:
            bio_pages.append(i)
            entry["follows_graph"] = last_graph
        else:
            graph_pages.append(i)
            last_graph = i
        pages[str(i)] = entry

    sidecar = {
        "book": book,
        "num_pages": num_pages,
        "bio_rule_yfrac": list(BIO_RULE_YFRAC),
        "bio_rule_tol": BIO_RULE_TOL,
        "graph_pages": graph_pages,
        "bio_pages": bio_pages,
        "pages": pages,
    }
    out_path = os.path.join(pages_path, "page_types.json")
    with open(out_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)

    logger.info("Classified %s: %d graph, %d bio -> %s",
                book, len(graph_pages), len(bio_pages), out_path)
    return sidecar


def load_bio_pages(book: str, books_dir: str = "books", pages_dir: str = "pages") -> set[int]:
    """Read the sidecar's ``bio_pages`` set, or an empty set if no sidecar exists.

    Books with no ``page_types.json`` (e.g. the all-tree Books 1 & 2, which were
    never classified) return an empty set, so downstream code treats every page as
    a graph page -- preserving their existing behaviour untouched.
    """
    path = os.path.join(books_dir, book, pages_dir, "page_types.json")
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return set(json.load(fh)["bio_pages"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.segment import BOOK_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, choices=sorted(BOOK_CONFIGS))
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--pages-dir", default="pages")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from src.segment import BOOK_CONFIGS

    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    num_pages = BOOK_CONFIGS[args.book].num_pages
    classify_pages(args.book, num_pages,
                   books_dir=args.books_dir, pages_dir=args.pages_dir)


if __name__ == "__main__":
    main()
