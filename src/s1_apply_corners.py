"""Stage 1 final generation: warp every content page using resolved corners.

Two corner sources feed this, in priority order per page:

1. ``books/{book}/pages/corners_review.json`` -- the human review layer from
   the border-QA tool (:mod:`scripts.qa.s1_borders`). An entry with
   ``status == "corners"`` holds hand-placed/confirmed corners and always wins.
2. ``books/{book}/pages/corners.json`` -- the detector output. Pages the two
   detectors agreed on carry their ``corners``; flagged pages carry none and must
   be covered by the review layer.

For each content page in the book's range, the resolved corners are warped with
:func:`src.s1_extract_pages.deskew_to_frame` (same as the detector path) and written
to ``books/{book}/pages/{i}.png``. A page with no corners from either source
is skipped and reported with its page version.

CLI:
    python -m src.s1_apply_corners --book book3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pymupdf

import src.s1_extract_pages as ep
from src.imaging import save_image

logger = logging.getLogger(__name__)


def _valid(c) -> bool:
    """True if a corners dict has real (non-None) coords for all four corners."""
    if not c:
        return False
    for k in ("tl", "tr", "br", "bl"):
        v = c.get(k)
        if not v or v[0] is None or v[1] is None:
            return False
    return True


def _corners_tuple(c: dict) -> tuple:
    return (tuple(c["tl"]), tuple(c["tr"]), tuple(c["br"]), tuple(c["bl"]))


def apply_corners(book: str, books_dir: str = "books") -> tuple[int, list[str]]:
    """Write every content page of ``book`` using review-then-detector corners.

    Returns ``(num_written, missing)`` where ``missing`` lists page tags that had
    no corners from either source.
    """
    out_dir = os.path.join(books_dir, book, "1_pages")
    corners_path = os.path.join(out_dir, "corners.json")
    review_path = os.path.join(out_dir, "corners_review.json")

    with open(corners_path) as f:
        detected = json.load(f)["pages"]
    review = {}
    if os.path.exists(review_path):
        with open(review_path) as f:
            review = json.load(f)

    first, last = ep.BOOK_PAGE_RANGES[book]
    doc = pymupdf.open(os.path.join(books_dir, book, f"{book}.pdf"))

    written = 0
    from_review = 0
    missing: list[str] = []
    for ci, pi in enumerate(range(first, last + 1)):
        tag = f"content {ci} (PDF index {pi}, viewer page {pi + 1})"
        rev = review.get(str(ci))
        det = detected.get(str(ci), {})
        # Priority: valid review corners -> detector's agreed corners -> the
        # detector's Hough corners on a flagged page (a review saved with empty
        # corners falls through here, e.g. book3 pages where flood failed and the
        # QA tool was saved without dragging the Hough-seeded dots).
        if rev and rev.get("status") == "corners" and _valid(rev.get("corners")):
            corners = _corners_tuple(rev["corners"])
            from_review += 1
        elif _valid(det.get("corners")):
            corners = _corners_tuple(det["corners"])
        elif _valid(det.get("hough")):
            corners = _corners_tuple(det["hough"])
            logger.info("%s: no review/agreed corners, using detector Hough", tag)
        else:
            missing.append(tag)
            logger.warning("NO CORNERS for %s -- skipped", tag)
            continue

        a = ep.render_page_binary(doc, pi)
        a = ep.deskew_to_frame(a, corners)
        save_image(a, os.path.join(out_dir, f"{ci}.png"))
        written += 1
    doc.close()

    logger.info("Wrote %d/%d pages for %s (%d from review, %d missing)",
                written, last - first + 1, book, from_review, len(missing))
    for m in missing:
        logger.warning("  missing: %s", m)
    return written, missing


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--books-dir", default="books")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    apply_corners(args.book, books_dir=args.books_dir)


if __name__ == "__main__":
    main()
