"""Bio stage 2: merge each section's crop pages into one wide RTL image.

Bio stage 1 (:mod:`src.bio.s1_crops`) wrote one border-trimmed page per biography
page, keeping the 4 full-width horizontal rules in place. The book alternates a tree
graph then a *biography section* (the pages describing everyone in that subgraph);
a section spans a run of contiguous page numbers. This stage joins each section's
pages into a single wide image so a person's entry -- which runs horizontally and can
cross a page seam -- becomes contiguous, ready for the generation-band split (stage 3).

Two rules of the book drive the join:

* **Right-to-left, eldest first.** The section's FIRST page goes on the RIGHT; later
  pages extend leftward. Block order in the merged image then matches the tree's
  depth-first, eldest-first, right-to-left traversal (what the count gate relies on).
* **The 4 horizontal rules must line up.** Pages are aligned on their TOP rule (padded
  above to a common y). The BOTTOM rule is then the alignment check: once top-aligned,
  every page's bottom rule must sit within :data:`BOTTOM_RULE_TOL` of the median. A
  larger drift means the ruled region is inconsistent across pages (a skewed or
  mis-scaled scan), so the merge **hard-fails** rather than emit a misaligned image.

Rules are located per page by full-width row ink-coverage, snapping any rule too faint
to detect (near-empty pages under-detect the top rules) to the canonical
:data:`~src.s2_classify_pages.BIO_RULE_YFRAC` position.

Output: ``books/{book}/bio/2_merged/{first}_{last}.png`` (binary ink grid, 0=ink),
a sidecar ``{first}_{last}.json`` (rule y-positions, seam x-columns, per-page
bottom-rule offset), and a downscaled QA overlay under ``books/{book}/qa/bio_s2/``.

Run:
    PYTHONPATH=. python -m src.bio.s2_merge --book book3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
from PIL import Image

from src.bio.s1_crops import BIO_DIR, section_first_pages
from src.imaging import get_image, save_image
from src.s2_classify_pages import BIO_RULE_YFRAC, load_bio_pages

logger = logging.getLogger(__name__)

MERGED_DIR = "2_merged"

# A full-width horizontal rule inks at least this fraction of a row. Rules run
# ~0.75-1.0 on clean crops; a page's own prose never fills a whole row.
RULE_ROW_COVERAGE = 0.6
# Adjacent inked rows within this many px are one rule (a rule is several px thick).
RULE_CLUSTER_PX = 20
# A detected rule is accepted for a canonical slot when within this many px of it;
# otherwise that slot falls back to the canonical position (faint/empty page).
RULE_SNAP_TOL = 120

# After top-rule alignment, each page's bottom rule must sit within this of the
# median bottom rule. Real book3 0_1 pages drift <50px; more means an inconsistent
# ruled region (skew / scale mismatch) and the merge hard-fails.
BOTTOM_RULE_TOL = 50

QA_DIR = "qa"
QA_SUBDIR = "bio_s2"
QA_SCALE = 6


def _rule_clusters(a: np.ndarray) -> list[int]:
    """Center rows of full-width horizontal rules in ``a`` (ascending).

    A rule is a run of rows each inking at least :data:`RULE_ROW_COVERAGE` of the
    width; runs within :data:`RULE_CLUSTER_PX` are one rule, reduced to their center.
    """
    cov = (1 - a).sum(axis=1) / a.shape[1]
    rows = np.where(cov >= RULE_ROW_COVERAGE)[0]
    if len(rows) == 0:
        return []
    centers: list[int] = []
    start = prev = rows[0]
    for r in rows[1:]:
        if r - prev > RULE_CLUSTER_PX:
            centers.append((start + prev) // 2)
            start = r
        prev = r
    centers.append((start + prev) // 2)
    return centers


def _expected_positions(h: int, detected: list[int]) -> list[float]:
    """Where the 4 rules are expected on a page of height ``h``.

    The canonical :data:`~src.s2_classify_pages.BIO_RULE_YFRAC` gives the rules'
    relative pattern, but a page's absolute rule positions shift with its top margin
    (crops leave a variable margin, which is exactly why we align on rules). So we do
    not assume a rule sits at ``frac*h``: we estimate a single global vertical shift
    ``d`` -- the median offset of detected rules from their nearest canonical slot --
    and expect each rule at ``frac*h + d``. With no detections, ``d=0``.
    """
    canon = [f * h for f in BIO_RULE_YFRAC]
    if not detected:
        return canon
    offsets = [y - min(canon, key=lambda c: abs(y - c)) for y in detected]
    d = float(np.median(offsets))
    return [c + d for c in canon]


def find_rules(a: np.ndarray) -> list[int]:
    """The 4 horizontal-rule y-positions of a bio page, top to bottom.

    Detects full-width rules and assigns each to its nearest *expected* slot (the
    canonical pattern shifted to the page's estimated top margin -- see
    :func:`_expected_positions`), within :data:`RULE_SNAP_TOL`. A slot that claimed a
    detected rule keeps that rule's real position -- even if somewhat off -- so
    genuine misalignment stays visible to the bottom-rule check in
    :func:`merge_section`. A slot with no detected rule falls back to its expected
    position (near-empty pages under-detect the fainter top rules). Always returns 4
    positions in ascending order.
    """
    detected = _rule_clusters(a)
    expected = _expected_positions(a.shape[0], detected)
    slots: list[int | None] = [None] * len(expected)
    for y in detected:
        si = min(range(len(expected)), key=lambda i: abs(y - expected[i]))
        if abs(y - expected[si]) > RULE_SNAP_TOL:
            continue  # a stray full-width mark, not one of the 4 rules
        prev = slots[si]
        if prev is None or abs(y - expected[si]) < abs(prev - expected[si]):
            slots[si] = int(y)
    rules = [s if s is not None else int(round(e)) for s, e in zip(slots, expected)]
    return sorted(rules)


def _pad_top(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return a
    return np.vstack([np.ones((n, a.shape[1]), dtype=np.uint8), a])


def _pad_bottom_to(a: np.ndarray, h: int) -> np.ndarray:
    n = h - a.shape[0]
    if n <= 0:
        return a
    return np.vstack([a, np.ones((n, a.shape[1]), dtype=np.uint8)])


def merge_section(pages: list[np.ndarray]) -> np.ndarray:
    """Merge a section's pages into one wide RTL image, aligning on the top rule.

    ``pages`` are in reading order (section-first page first). The first page is
    placed on the RIGHT; later pages extend left. Each page is padded above so its
    top rule lands at a common y; pages are then bottom-padded to a common height and
    concatenated. Raises :class:`ValueError` if any page's bottom rule, after
    top-alignment, drifts more than :data:`BOTTOM_RULE_TOL` from the median.
    """
    if not pages:
        raise ValueError("merge_section: no pages")

    rules = [find_rules(p) for p in pages]
    tops = [r[0] for r in rules]
    bottoms = [r[-1] for r in rules]

    # Align top rules: pad each page above by (max_top - its_top) so all top rules
    # share the row ``max_top``.
    max_top = max(tops)
    aligned = [_pad_top(p, max_top - t) for p, t in zip(pages, tops)]
    # After top-alignment the bottom rule sits at (bottom - top) + max_top.
    aligned_bottoms = [(b - t) + max_top for b, t in zip(bottoms, tops)]

    median_bottom = int(np.median(aligned_bottoms))
    for page_idx, ab in enumerate(aligned_bottoms):
        if abs(ab - median_bottom) > BOTTOM_RULE_TOL:
            raise ValueError(
                f"page {page_idx} bottom rule at y={ab} drifts "
                f"{abs(ab - median_bottom)}px from the section median {median_bottom} "
                f"(tol {BOTTOM_RULE_TOL}); the ruled region is inconsistent across "
                f"pages -- refusing to merge a misaligned section"
            )

    common_h = max(p.shape[0] for p in aligned)
    aligned = [_pad_bottom_to(p, common_h) for p in aligned]

    # RTL: first page rightmost. hstack is left-to-right, so reverse the run.
    return np.hstack(list(reversed(aligned)))


def bio_sections_from(bio_pages: set[int], first_pages: set[int]) -> list[list[int]]:
    """Group bio pages into sections: contiguous runs starting at a first page.

    A section begins at each page in ``first_pages`` and runs through the following
    consecutive page numbers up to (but not including) the next section start. A
    non-contiguous jump *within* a section (a missing/misclassified page) raises --
    welding across a page-number gap is a bug, not a wide section.
    """
    ordered = sorted(bio_pages)
    starts = sorted(first_pages)
    sections: list[list[int]] = []
    for si, start in enumerate(starts):
        end = starts[si + 1] if si + 1 < len(starts) else None
        run = [p for p in ordered if p >= start and (end is None or p < end)]
        for a, b in zip(run, run[1:]):
            if b != a + 1:
                raise ValueError(
                    f"bio section starting at page {start}: pages {a} and {b} are "
                    f"not contiguous (gap {a + 1}..{b - 1}); a page is missing or "
                    f"misclassified -- refusing to merge across the gap"
                )
        sections.append(run)
    return sections


def _save_qa(merged: np.ndarray, rules: list[int], seams: list[int],
             out_path: str) -> None:
    """Downscaled merged image with rules (green) and page seams (red) drawn."""
    rows, cols = merged.shape
    img = Image.fromarray((merged * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((max(1, cols // QA_SCALE), max(1, rows // QA_SCALE)))
    px = img.load()
    assert px is not None
    for y in rules:
        yy = y // QA_SCALE
        if 0 <= yy < img.height:
            for x in range(img.width):
                px[x, yy] = (0, 200, 0)
    for x in seams:
        xx = x // QA_SCALE
        if 0 <= xx < img.width:
            for y in range(img.height):
                px[xx, y] = (255, 0, 0)
    img.save(out_path)


def merge_book(book: str, books_dir: str = "books", qa: bool = True) -> list[str]:
    """Merge every bio section of ``book``. Returns the written merged-image paths."""
    bio_pages = load_bio_pages(book, books_dir=books_dir)
    if not bio_pages:
        logger.warning("no biography pages for %s", book)
        return []
    first_pages = section_first_pages(book, bio_pages, books_dir=books_dir)
    sections = bio_sections_from(bio_pages, first_pages)

    crops_dir = os.path.join(books_dir, book, BIO_DIR, "1_crops")
    out_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    os.makedirs(out_dir, exist_ok=True)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    written: list[str] = []
    for run in sections:
        stem = f"{run[0]}_{run[-1]}"
        pages = [get_image(os.path.join(crops_dir, f"{p}.png")) for p in run]
        try:
            merged = merge_section(pages)
        except ValueError as exc:
            # Re-raise with the section and the offending page's real page number,
            # which merge_section (given only arrays) cannot name.
            raise ValueError(f"{book} bio section {stem}: {exc} "
                             f"(pages {run})") from exc

        # Recompute rules + seams on the merged image for the sidecar / QA.
        merged_rules = find_rules(merged)
        # Seam x-columns: cumulative widths of the reversed (RTL) page order.
        widths = [p.shape[1] for p in reversed(pages)]
        seams, acc = [], 0
        for w in widths[:-1]:
            acc += w
            seams.append(acc)

        out_path = os.path.join(out_dir, f"{stem}.png")
        save_image(merged, out_path)
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
            json.dump({
                "book": book,
                "pages_reading_order": run,
                "rules_y": merged_rules,
                "seam_x": seams,
                "size": [int(merged.shape[1]), int(merged.shape[0])],
            }, fh, indent=2)
        if qa:
            _save_qa(merged, merged_rules, seams,
                     os.path.join(qa_dir, f"{stem}.png"))
        logger.info("section %s: %d pages -> %s (%dx%d)", stem, len(run),
                    out_path, merged.shape[1], merged.shape[0])
        written.append(out_path)

    logger.info("merged %d bio sections for %s", len(written), book)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.s3_segment import BOOK_CONFIGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, choices=sorted(BOOK_CONFIGS))
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--no-qa", action="store_true", help="skip QA previews")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(levelname)s %(name)s: %(message)s")
    merge_book(args.book, books_dir=args.books_dir, qa=not args.no_qa)


if __name__ == "__main__":
    main()
