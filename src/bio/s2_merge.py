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

Rules are located per page by run-length line detection (like stage 1) within a
search window around each canonical :data:`~src.s2_classify_pages.BIO_RULE_YFRAC`
y-fraction, so faded (low-density) rules are still found and a truly blank band falls
back to the expected position.

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

from src.bio.s1_crops import BIO_DIR, SKIP_PAGES, _longest_run, section_first_pages
from src.imaging import get_image, save_image
from src.s2_classify_pages import BIO_RULE_YFRAC, load_bio_pages

logger = logging.getLogger(__name__)

MERGED_DIR = "2_merged"

# Horizontal-rule detection mirrors stage 1's border detection: a rule is a
# full-width straight LINE, found by its longest continuous ink run (bridging ADF
# breaks via s1's _longest_run), NOT by row ink-density. An ADF-faded rule can be
# ~30% dense yet keep one unbroken run spanning most of the width, while a page's
# densest prose row tops out well below that -- so run length separates them where a
# density threshold missed faded rules (stage 1 hit this on Book 4). A row is a rule
# when its longest run is at least this fraction of the page width.
RULE_MIN_RUN_FRAC = 0.5
# The canonical BIO_RULE_YFRAC positions are a strong prior (stage 1 verified the 4
# rules sit at fixed y-fractions). We use them as a HINT: search only a window this
# many px around each expected slot for that slot's rule, so a long prose line
# elsewhere can never be taken for a rule, and a faded rule near its slot is found.
RULE_SEARCH_PX = 120

# After top-rule alignment, each page's bottom rule must sit within this of the
# median bottom rule; more means an inconsistent ruled region and the merge hard-fails.
# Set from the data: across all book3 + book4 multi-page sections the bottom-rule drift
# tops out at 55px (book4 207_209, whose p209 was extracted ~1.5% taller so its rule
# spacing is proportionally wider -- a real but small per-page scale difference, not a
# broken page), with the next-highest at 48px. 70 clears these scale near-misses while
# still catching a genuinely mis-extracted page (those drift hundreds of px, cf. the
# graph pipeline's 388/429px seam bugs).
BOTTOM_RULE_TOL = 70

# Side-margin trim before joining. Each page carries ~100px of edge whitespace and
# ~60px inter-column gaps; a naive join doubles a seam-crossing gap to ~200px. Trim
# up to SIDE_TRIM_CAP px of *whitespace* off each side -- scanning inward, stopping
# at the first column denser than SIDE_TRIM_FACTOR x the page's whitespace baseline.
# The baseline is a page's minimum column density (~0.004: the 4 horizontal rules
# crossing an otherwise-blank column). Text columns run ~6-9x baseline (0.025-0.037),
# so a 2x stop halts right at the text ramp without clipping the outermost glyphs.
SIDE_TRIM_CAP = 80
SIDE_TRIM_FACTOR = 2.0
# When a side stops at content BEFORE the cap, keep this much whitespace before the
# glyphs so we never trim right up to the characters. (No pad is withheld when the
# whole cap is whitespace -- there is still whitespace beyond it.)
SIDE_TRIM_PAD = 20

QA_DIR = "qa"
QA_SUBDIR = "bio_s2"
QA_SCALE = 6


def _best_rule_row(a: np.ndarray, center: int, half: int) -> int | None:
    """Row of the strongest horizontal rule within ``center +/- half``, or ``None``.

    Scans rows in the window for the one whose longest continuous ink run (bridging
    ADF breaks, via stage 1's :func:`~src.bio.s1_crops._longest_run`) is longest, and
    returns it when that run spans at least :data:`RULE_MIN_RUN_FRAC` of the width.
    """
    rows, cols = a.shape
    lo, hi = max(0, center - half), min(rows, center + half + 1)
    min_run = int(RULE_MIN_RUN_FRAC * cols)
    best_row, best_run = None, min_run - 1
    for r in range(lo, hi):
        run = _longest_run((1 - a[r, :]).astype(bool))
        if run > best_run:
            best_run, best_row = run, r
    return best_row


def find_rules(a: np.ndarray) -> list[int]:
    """The 4 horizontal-rule y-positions of a bio page, top to bottom.

    The canonical :data:`~src.s2_classify_pages.BIO_RULE_YFRAC` gives each rule's
    expected y-fraction. Because a crop's top margin varies, we first estimate a single
    global shift ``d`` from the rules found in a wide search, then locate each rule by
    its longest ink run within :data:`RULE_SEARCH_PX` of its shifted expected position
    (:func:`_best_rule_row`). A rule found keeps its REAL y -- even if off canonical --
    so genuine misalignment stays visible to the bottom-rule check in
    :func:`merge_section`; a slot with no rule (a truly blank band) falls back to the
    expected position. Always returns 4 positions in ascending order.
    """
    h = a.shape[0]
    canon = [f * h for f in BIO_RULE_YFRAC]
    # Pass 1: find whatever rules we can in a wide window to estimate the top-margin
    # shift d (robust to a missing rule via the median).
    found = [(c, _best_rule_row(a, int(c), RULE_SEARCH_PX)) for c in canon]
    offsets = [row - c for c, row in found if row is not None]
    d = float(np.median(offsets)) if offsets else 0.0
    # Pass 2: locate each rule tightly around its shifted expected position.
    rules = []
    for c in canon:
        expected = c + d
        row = _best_rule_row(a, int(round(expected)), RULE_SEARCH_PX)
        rules.append(row if row is not None else int(round(expected)))
    return sorted(rules)


def _side_trim(col_dens: np.ndarray, baseline: float) -> int:
    """How many leading columns of ``col_dens`` to trim, capped and padded.

    Scans inward from index 0, counting columns whose density is at most
    :data:`SIDE_TRIM_FACTOR` x ``baseline`` (whitespace). If it reaches
    :data:`SIDE_TRIM_CAP` still in whitespace, trims the full cap. If it stops earlier
    at a content column, trims that far minus :data:`SIDE_TRIM_PAD` (floored at 0) so
    :data:`SIDE_TRIM_PAD` px of whitespace is kept before the glyphs.
    """
    limit = baseline * SIDE_TRIM_FACTOR
    n = 0
    while n < SIDE_TRIM_CAP and n < len(col_dens) and col_dens[n] <= limit:
        n += 1
    if n >= SIDE_TRIM_CAP:
        return n
    return max(0, n - SIDE_TRIM_PAD)


def trim_sides(a: np.ndarray) -> np.ndarray:
    """Trim whitespace side-margins off a page so a seam gap is not doubled.

    Removes up to :data:`SIDE_TRIM_CAP` px of whitespace from the left and right
    edges, stopping each side at its first content column (see :func:`_side_trim`).
    The whitespace baseline is the page's minimum column density -- the 4 horizontal
    rules crossing an otherwise-blank column -- so the rules themselves are preserved
    (they span the full height; trimming columns only shortens them).
    """
    col_dens = (1 - a).sum(axis=0) / a.shape[0]
    baseline = float(col_dens.min())
    left = _side_trim(col_dens, baseline)
    right = _side_trim(col_dens[::-1], baseline)
    return a[:, left:a.shape[1] - right]


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


def bio_sections_from(bio_pages: set[int], first_pages: set[int],
                      skip: set[int] | None = None) -> list[list[int]]:
    """Group bio pages into sections: contiguous runs starting at a first page.

    A section begins at each page in ``first_pages`` and runs through the following
    consecutive page numbers up to (but not including) the next section start. A
    non-contiguous jump *within* a section (a missing/misclassified page) raises --
    welding across a page-number gap is a bug, not a wide section.

    ``skip`` pages are DELIBERATE exclusions (e.g. Book 4 p214, whose branch has no
    tree graph -- see :data:`~src.bio.s1_crops.SKIP_PAGES`). Contiguity is checked on
    the full run *including* a skipped page (it is present in ``bio_pages``, so it keeps
    the run contiguous and a genuinely missing page still raises); the skipped page is
    then dropped from the returned run, which may leave an intended gap.
    """
    skip = skip or set()
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
        kept = [p for p in run if p not in skip]
        if kept:
            sections.append(kept)
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
    sections = bio_sections_from(bio_pages, first_pages, skip=SKIP_PAGES.get(book, set()))

    crops_dir = os.path.join(books_dir, book, BIO_DIR, "1_crops")
    out_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    os.makedirs(out_dir, exist_ok=True)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    written: list[str] = []
    for run in sections:
        stem = f"{run[0]}_{run[-1]}"
        pages = [trim_sides(get_image(os.path.join(crops_dir, f"{p}.png")))
                 for p in run]
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
