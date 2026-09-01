"""Stage 1 of the pipeline: two-page spreads -> deskewed single pages.

Each scan in ``books/bookN/original/*.png`` is a camera photo of an open book,
so it contains a left page and a right page side by side. This stage finds the
border corners of each half, applies a perspective transform to deskew and
normalize it onto a fixed canvas, and writes the single pages to
``books/bookN/pages/{i}.png``.

The book reads right-to-left, so within a spread the *right* page is emitted
first and the *left* page second.

Per-book quirks (which spreads to read, how to number the output, whether to
skip a half-page at the very start) are captured in :data:`BOOK_CONFIGS` rather
than baked into the algorithm.

CLI:
    python -m src.extract_pages --book book1
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import os

import numpy as np

from src.imaging import get_image, pad_image, save_image  # noqa: F401  (pad_image re-exported for callers)

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("extract_pages requires opencv-python (cv2)") from exc


logger = logging.getLogger(__name__)

# Normalized output-page dimensions. Every scan is the same book on the same
# scanner, so all pages map onto this fixed canvas (see the design spec).
PAGE_WIDTH = 1300
PAGE_HEIGHT = 1950

# Number of consecutive matching border pixels required before we accept a row
# position as "inside the page border" when scanning inward for the seed pixel.
_BORDER_RUN = 4


@dataclasses.dataclass(frozen=True)
class BookConfig:
    """Per-book Stage-1 configuration.

    Attributes:
        first_spread: Index of the first ``original/{i}.png`` spread to process.
        last_spread: One past the last spread index (exclusive), i.e. spreads
            ``range(first_spread, last_spread)`` are processed.
        start_index: Page number assigned to the very first emitted page. Book 1
            numbers its first (right) page ``-1`` so that the first real page
            lands on ``0``; Book 2 starts at ``0``.
        skip_right_on_first_spread: If True, the right half of the first
            processed spread is skipped entirely (only its left page is
            emitted). Book 2's page-0 spread has no usable right page.
    """

    first_spread: int
    last_spread: int
    start_index: int
    skip_right_on_first_spread: bool


# Faithful to the old notebook: Book 1 read spreads 3..11 and numbered pages
# from -1; Book 2 read spreads 0..67 and skipped the right half of spread 0.
BOOK_CONFIGS: dict[str, BookConfig] = {
    "book1": BookConfig(
        first_spread=3,
        last_spread=12,
        start_index=-1,
        skip_right_on_first_spread=False,
    ),
    "book2": BookConfig(
        first_spread=0,
        last_spread=68,
        start_index=0,
        skip_right_on_first_spread=True,
    ),
}


def remove_small_islands(orig_a: np.ndarray, max_size: int = 10) -> np.ndarray:
    """Remove tiny ink specks (scanning noise) from a binary grid.

    Every connected component of ink pixels no larger than ``max_size`` is
    erased (set to background). Uses 4-connectivity BFS.

    Args:
        orig_a: Binary ink grid (0 == ink, 1 == background).
        max_size: Components with this many pixels or fewer are removed.

    Returns:
        A new grid with small islands erased.
    """
    a = orig_a.copy()  # avoid modifying original
    rows, cols = a.shape
    visited = np.zeros_like(a, dtype=bool)

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for r in range(rows):
        for c in range(cols):
            if a[r, c] == 0 and not visited[r, c]:
                # BFS over this ink component.
                queue = collections.deque([(r, c)])
                visited[r, c] = True
                coords = [(r, c)]

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and a[nr, nc] == 0
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            coords.append((nr, nc))

                if len(coords) <= max_size:
                    for rr, cc in coords:
                        a[rr, cc] = 1

    return a


def get_corners(
    a: np.ndarray, start_i: int, start_j: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Find the four corners of a page by flood-filling its border region.

    Flood-fills the connected background region reachable from
    ``(start_i, start_j)`` (a pixel just inside the page border) and tracks the
    extreme pixels in each diagonal direction to recover the page's corners.

    Args:
        a: Binary ink grid.
        start_i: Seed row, inside the target page.
        start_j: Seed column, inside the target page.

    Returns:
        The corners as ``(top_left, top_right, bottom_right, bottom_left)``,
        each a ``(row, col)`` tuple.
    """
    rows, cols = a.shape
    queue: collections.deque[tuple[int, int]] = collections.deque()
    queue.append((start_i, start_j))

    visited = np.zeros([rows, cols])

    tl = tr = br = bl = (start_i, start_j)

    while queue:
        i, j = queue.popleft()
        if i < 0 or i >= rows or j < 0 or j >= cols:
            continue
        if visited[i][j]:
            continue
        if a[i][j]:
            continue
        visited[i][j] = 1

        if -i - j > -tl[0] - tl[1]:
            tl = i, j
        if -i + j > -tr[0] + tr[1]:
            tr = i, j
        if i - j > bl[0] - bl[1]:
            bl = i, j
        if i + j > br[0] + br[1]:
            br = i, j

        queue.append((i + 1, j))
        queue.append((i - 1, j))
        queue.append((i, j + 1))
        queue.append((i, j - 1))

    return tl, tr, br, bl


def normalize_page(
    a: np.ndarray,
    page_corners: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ],
    width: int = PAGE_WIDTH,
    height: int = PAGE_HEIGHT,
) -> np.ndarray:
    """Deskew a page onto a fixed ``width`` x ``height`` canvas.

    Applies a perspective transform mapping the four detected page corners to
    the corners of a rectangle of the given size.

    Args:
        a: Binary ink grid containing the page.
        page_corners: ``(tl, tr, br, bl)`` as returned by :func:`get_corners`.
        width: Output canvas width.
        height: Output canvas height.

    Returns:
        The warped, normalized page as a binary grid.
    """
    tl, tr, br, bl = page_corners
    # Corners are (row, col); OpenCV wants (x, y) == (col, row).
    src_pts = np.float32(
        [
            [tl[1], tl[0]],
            [tr[1], tr[0]],
            [br[1], br[0]],
            [bl[1], bl[0]],
        ]
    )
    dst_pts = np.float32(
        [
            [0, 0],  # top-left
            [width - 1, 0],  # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1],  # bottom-left
        ]
    )

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(a, matrix, (width, height))


def _seed_from_right(a: np.ndarray) -> tuple[int, int]:
    """Find a seed pixel just inside the right page's border.

    Scans leftward along the vertical middle from the right edge until it
    passes the page's outer border into background.
    """
    rows, cols = a.shape
    i = rows // 2
    j = cols - 20
    # Walk left while any of the next few pixels are ink (the border run).
    while a[i][j] or a[i][j - 1] or a[i][j - 2] or a[i][j - 3]:
        j -= 1
    return i, j


def _seed_from_left(a: np.ndarray) -> tuple[int, int]:
    """Find a seed pixel just inside the left page's border.

    Scans rightward along the vertical middle from the left edge until it
    passes the page's outer border into background.
    """
    rows, _cols = a.shape
    i = rows // 2
    j = 0
    while a[i][j] or a[i][j + 1] or a[i][j + 2] or a[i][j + 3]:
        j += 1
    return i, j


def extract_page_from_side(a: np.ndarray, side: str) -> np.ndarray:
    """Extract and normalize one page (``"right"`` or ``"left"``) from a spread.

    Args:
        a: Binary ink grid of the full spread (already denoised).
        side: Which half to extract, ``"right"`` or ``"left"``.

    Returns:
        The deskewed, normalized single page.
    """
    if side == "right":
        seed = _seed_from_right(a)
    elif side == "left":
        seed = _seed_from_left(a)
    else:
        raise ValueError(f"side must be 'right' or 'left', got {side!r}")

    corners = get_corners(a, *seed)
    logger.debug("%s page corners: %s", side, corners)
    return normalize_page(a, corners)


def extract_pages(
    book: str,
    books_dir: str = "books",
    config: BookConfig | None = None,
) -> list[str]:
    """Split every spread of ``book`` into deskewed single pages.

    Reads ``{books_dir}/{book}/original/{i}.png`` for the spreads named by the
    book's config, deskews the right then left page of each, and writes them to
    ``{books_dir}/{book}/pages/{n}.png``. The output directory is created if
    needed.

    Args:
        book: Book name, e.g. ``"book1"``. Used to look up :data:`BOOK_CONFIGS`
            when ``config`` is not given, and to locate the book directory.
        books_dir: Root directory containing per-book asset folders.
        config: Explicit config, overriding the :data:`BOOK_CONFIGS` lookup.

    Returns:
        The list of output page file paths, in emission order.
    """
    if config is None:
        if book not in BOOK_CONFIGS:
            raise KeyError(
                f"no BookConfig for {book!r}; known books: {sorted(BOOK_CONFIGS)}"
            )
        config = BOOK_CONFIGS[book]

    original_dir = os.path.join(books_dir, book, "original")
    pages_dir = os.path.join(books_dir, book, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    logger.info(
        "Extracting pages for %s: spreads %d..%d -> %s (start index %d)",
        book,
        config.first_spread,
        config.last_spread - 1,
        pages_dir,
        config.start_index,
    )

    written: list[str] = []
    idx = config.start_index

    for spread in range(config.first_spread, config.last_spread):
        filepath = os.path.join(original_dir, f"{spread}.png")
        logger.info("Processing spread %d: %s", spread, filepath)

        a = get_image(filepath)
        a = remove_small_islands(a)

        # Right page first (book reads right-to-left), unless this is the first
        # processed spread and the book skips its right half.
        skip_right = config.skip_right_on_first_spread and spread == config.first_spread
        if skip_right:
            logger.info("Skipping right half of first spread %d per book config", spread)
        else:
            right_page = extract_page_from_side(a, "right")
            out_path = os.path.join(pages_dir, f"{idx}.png")
            save_image(right_page, out_path)
            logger.info("Wrote right page -> %s", out_path)
            written.append(out_path)
            idx += 1

        left_page = extract_page_from_side(a, "left")
        out_path = os.path.join(pages_dir, f"{idx}.png")
        save_image(left_page, out_path)
        logger.info("Wrote left page -> %s", out_path)
        written.append(out_path)
        idx += 1

    logger.info("Extracted %d pages for %s", len(written), book)
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
    written = extract_pages(args.book, books_dir=args.books_dir)
    logger.info("Done. %d pages written for %s.", len(written), args.book)


if __name__ == "__main__":
    main()
