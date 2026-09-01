"""Stage 1 (v1 scans): ADF single-page PDF -> single upright pages.

The v1 scans are automatic-document-feeder captures: one book page per PDF page,
already flat (no perspective distortion) and already upright once the page's
``/Rotate`` flag is honored. This is unlike the v1 glass-top scans, which were
two-page spreads needing corner-detection + perspective deskew. So this stage is
simple: render each content page at full native resolution, binarize, crop away
the blank scanner margin, and write it out. **No downsizing** -- the whole point
of the 600dpi scan is to keep every pixel for line detection and OCR. Downstream
v1 stages carry pixel constants sized for this native resolution.

Two renditions of each book exist; both flow through here identically, differing
only in the source PDF and threshold:

* grayscale (``book1.pdf``) -- 600dpi continuous-tone; binarized at
  :data:`GRAY_THRESHOLD`. The threshold is chosen so every printed stroke (which
  scans as near-black) survives, while lighter hand-pen margin marks (which scan
  as mid-gray) drop out. Preserving true ink is the hard constraint; dropping
  pen marks is a welcome side effect, never pursued at the cost of real pixels.
* bitonal (``book1_bw.pdf``) -- the scanner already thresholded to black/white;
  rendering upsamples it so a few edge grays appear, and the same
  :data:`GRAY_THRESHOLD` re-binarizes it near-losslessly.

Output: ``books/{book}/{pages_dir}/{i}.png`` as an 8-bit binary PNG (0 == ink,
255 == background), page 0 first.

CLI:
    python -m src.extract_pages --book book1 --pdf books/book1/book1.pdf \
        --first-page 6 --last-page 23 --pages-dir pages_gray
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pymupdf
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("src.extract_pages requires opencv-python (cv2)") from exc

from src.imaging import save_image

logger = logging.getLogger(__name__)

# Grayscale binarization cutoff (0..255): pixels this dark or darker are ink.
# 128 keeps all printed strokes (they scan < 128) while dropping lighter pen
# marks (they scan > 128). See the threshold experiment in the design notes.
GRAY_THRESHOLD = 128

# Content page range per book, as inclusive 0-based PDF page indices. Everything
# outside is cover / front matter / blank trailing pages -- ignored. (William's
# ranges, given 1-indexed, converted here: book1 7-23, book2 3-136, book3 7-298,
# book4 14-330.) Used when --first-page/--last-page are not passed explicitly.
BOOK_PAGE_RANGES: dict[str, tuple[int, int]] = {
    "book1": (6, 22),
    "book2": (2, 135),
    "book3": (6, 297),
    "book4": (13, 329),
}

# Render scale: the PDF page is ~595x842 pt; x8.2 lands near the native embedded
# resolution (~4882x6904) without upsampling past it.
RENDER_SCALE = 8.2

# The printed page frame is a closed rectangle whose fitted size must cover at
# least this fraction of the page in each dimension; a smaller detection means
# the flood-fill leaked or caught the wrong ink, so that seed is rejected.
FRAME_MIN_COVER = 0.60

# The four detected corners must form a near-rectangle: each side's angle may
# deviate from horizontal/vertical by at most this many degrees. A leaked fill
# yields a slanted "side" (e.g. a diagonal cutting across the page) that this
# rejects even when the bounding box happens to be large enough.
FRAME_MAX_SIDE_SKEW_DEG = 3.0

# Morphological-open kernel (px) applied to a copy of the ink before frame
# detection. It erases sub-kernel-thickness strokes -- speckle and the thin,
# faint scanning whiskers that trail off a frame corner -- while the thick frame
# rules survive untouched. Without it a stray whisker becomes a false corner and
# skews the whole warp. Detection-only: the warp still uses the original ink.
FRAME_OPEN_KERNEL = 5

# Gap-bridging close applied to the detection copy after the open. Some scans
# (books 3-4) print the frame with small breaks -- a faint or missing stretch of
# a rule -- so a single flood-fill cannot trace the whole box. A *directional*
# close bridges gaps only *along* each rule: a horizontal close reconnects the
# top/bottom rules, a vertical one the sides. Being directional, it cannot fuse
# an interior name to the frame across the margin (that would need bridging
# perpendicular to a rule). Detection-only: corners feed the warp, which reads
# the original ink, so bridged pixels never reach the output. The length is how
# wide a gap to jump; the thickness (3) keeps it hugging the rule.
FRAME_CLOSE_LEN = 15

# Frame detection seeds, one per side: walk inward from the middle of each edge
# to the first ink pixel (the frame rule), then flood-fill from there. Tried in
# order; the first seed whose fill passes the frame heuristic wins. Trying all
# four survives a frame broken on any one side -- the fill from an intact side
# still traces the rest of the box. Each entry is (edge_name, axis, direction).
_FRAME_SEED_EDGES = ("top", "bottom", "left", "right")


def _seed_from_edge(ink: np.ndarray, edge: str) -> tuple[int, int] | None:
    """Walk inward from the middle of ``edge`` to the first ink pixel.

    Args:
        ink: Binary grid, 1 == ink.
        edge: One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``.

    Returns:
        The ``(x, y)`` seed pixel on the frame rule, or ``None`` if the whole
        scan line is background (no frame rule on that side).
    """
    h, w = ink.shape
    if edge in ("top", "bottom"):
        col = w // 2
        rows = range(h) if edge == "top" else range(h - 1, -1, -1)
        for row in rows:
            if ink[row, col]:
                return (col, row)
    else:
        row = h // 2
        cols = range(w) if edge == "left" else range(w - 1, -1, -1)
        for col in cols:
            if ink[row, col]:
                return (col, row)
    return None


def _fit_edge(frame: np.ndarray, side: str) -> tuple[float, float]:
    """Robustly fit the line of one frame side to its outermost pixels.

    For each scan line perpendicular to ``side``, take the frame's *outermost*
    pixel (the leftmost for ``"left"``, topmost for ``"top"``, etc.). Those
    extreme pixels trace the outer frame rule; a Huber line fit through them
    recovers the rule even when it is broken on part of its length (the missing
    rows are simply a minority the robust fit ignores) or when the flood-fill
    wandered inward through touching text (those inner pixels are never the
    outermost, so they do not enter the fit).

    Args:
        frame: Boolean grid, True where the flood-filled frame is.
        side: One of ``"top"``, ``"bottom"``, ``"left"``, ``"right"``.

    Returns:
        ``(m, b)`` for the fitted line. For left/right (near-vertical) it is
        ``x = m*y + b``; for top/bottom (near-horizontal) it is ``y = m*x + b``.
    """
    h, w = frame.shape
    ys, xs = np.where(frame)
    if side in ("left", "right"):
        # Outermost x per row.
        ext = np.full(h, -1 if side == "right" else w, dtype=np.int32)
        if side == "right":
            np.maximum.at(ext, ys, xs)
        else:
            np.minimum.at(ext, ys, xs)
        rows = np.where((ext >= 0) & (ext < w))[0]
        pts = np.column_stack([rows, ext[rows]]).astype(np.float32)  # (y, x)
        vy, vx, y0, x0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        m = float(vx / vy)
        return m, float(x0 - m * y0)  # x = m*y + b
    # Outermost y per column.
    ext = np.full(w, -1 if side == "bottom" else h, dtype=np.int32)
    if side == "bottom":
        np.maximum.at(ext, xs, ys)
    else:
        np.minimum.at(ext, xs, ys)
    cols = np.where((ext >= 0) & (ext < h))[0]
    pts = np.column_stack([cols, ext[cols]]).astype(np.float32)  # (x, y)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    m = float(vy / vx)
    return m, float(y0 - m * x0)  # y = m*x + b


def _intersect(horiz: tuple[float, float], vert: tuple[float, float]) -> tuple[int, int]:
    """Intersect a near-horizontal edge (y = m*x + b) with a near-vertical one
    (x = m*y + b), returning the ``(x, y)`` corner."""
    mh, bh = horiz
    mv, bv = vert
    # y = mh*(mv*y + bv) + bh  ->  y (1 - mh*mv) = mh*bv + bh
    y = (mh * bv + bh) / (1.0 - mh * mv)
    x = mv * y + bv
    return int(round(x)), int(round(y))


def _fill_corners(
    ink: np.ndarray, seed: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    """Flood-fill the frame from ``seed`` and return its corners, or None if bad.

    Fills the connected ink reachable from ``seed`` (the traced frame outline),
    then recovers the four corners by robustly fitting a line to each side's
    outermost pixels (:func:`_fit_edge`) and intersecting adjacent lines. Fitting
    the *outermost* pixels per scan line makes this robust to a frame that is
    broken/truncated on one side or to a fill that wandered inward through
    touching header text -- neither perturbs the outer rule the fit locks onto.
    The fill is validated to cover the page (:data:`FRAME_MIN_COVER`) and the
    fitted corners to form a near-rectangle (:data:`FRAME_MAX_SIDE_SKEW_DEG`).

    Returns:
        ``(tl, tr, br, bl)`` (x, y) corners, or ``None`` if the fill is not a
        plausible frame.
    """
    h, w = ink.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = ink.copy()
    cv2.floodFill(filled, mask, seed, 2, flags=8)
    frame = filled == 2
    ys, xs = np.where(frame)

    if xs.max() - xs.min() < FRAME_MIN_COVER * w or ys.max() - ys.min() < FRAME_MIN_COVER * h:
        return None

    top = _fit_edge(frame, "top")
    bot = _fit_edge(frame, "bottom")
    left = _fit_edge(frame, "left")
    right = _fit_edge(frame, "right")

    tl = _intersect(top, left)
    tr = _intersect(top, right)
    br = _intersect(bot, right)
    bl = _intersect(bot, left)

    # Reject if any fitted side is too far from horizontal/vertical: the top and
    # bottom slopes (dy/dx) and the left/right slopes (dx/dy) are each a tangent.
    tol = np.tan(np.radians(FRAME_MAX_SIDE_SKEW_DEG))
    if max(abs(top[0]), abs(bot[0]), abs(left[0]), abs(right[0])) > tol:
        return None

    return tl, tr, br, bl


def find_frame_corners(
    a: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Locate the four corners of the printed page frame.

    Every page is boxed by a closed rectangular rule (a double line). Flood-fill
    the connected ink from a seed on the frame, then recover the corners by
    fitting a line to each side and intersecting them (:func:`_fill_corners`) --
    giving the true quad, since some frames are a slight trapezoid the downstream
    warp must straighten. The fill touches only the thin rules, so it is
    O(frame perimeter), not O(page area).

    The frame is sometimes *broken* on one side (a printed gap, or the scanner
    cut a corner), so a fill seeded from that side traces only a fragment. To
    survive that, seed from the middle of each of the four edges in turn (walking
    inward to the first ink pixel) and accept the first fill that passes the
    frame heuristic in :func:`_fill_corners`; a break on any single side still
    leaves an intact side to seed from. A morphological open (see
    :data:`FRAME_OPEN_KERNEL`) first removes thin scanning whiskers.

    Args:
        a: Binary ink grid (0 == ink, 1 == background), full resolution.

    Returns:
        The corners as ``(tl, tr, br, bl)``, each an ``(x, y)`` (col, row) pair.

    Raises:
        ValueError: If no seed from any of the four edges yields a plausible
            frame (all fail the cover/rectangle heuristic).
    """
    ink = (1 - a).astype(np.uint8)  # 1 == ink
    ker = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (FRAME_OPEN_KERNEL, FRAME_OPEN_KERNEL)
    )
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, ker)

    # Bridge small breaks in the frame rules so a single fill traces the whole
    # box. Directional: horizontal close for the top/bottom rules, vertical for
    # the sides. Detection-only -- the warp uses the original ink.
    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (FRAME_CLOSE_LEN, 3))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, FRAME_CLOSE_LEN))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, h_ker)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, v_ker)

    for edge in _FRAME_SEED_EDGES:
        seed = _seed_from_edge(ink, edge)
        if seed is None:
            continue
        corners = _fill_corners(ink, seed)
        if corners is not None:
            return corners

    h, w = a.shape
    raise ValueError(
        f"no page frame found: no seed from any edge (top/bottom/left/right) of "
        f"the {w}x{h} page produced a fill covering >={FRAME_MIN_COVER:.0%} and "
        f"forming a rectangle (sides within {FRAME_MAX_SIDE_SKEW_DEG:.0f} deg)"
    )


def deskew_to_frame(
    a: np.ndarray,
    corners: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    """Warp the page so the frame corners become the output page corners.

    A perspective transform mapping the four detected frame corners to a clean
    axis-aligned rectangle. This deskews the page, crops it to exactly the inside
    of the frame, and discards everything outside (the scanner margin) in a
    single step -- the frame line itself is kept, sitting at the page edge. The
    output size is the frame's own averaged side lengths, so no resolution is
    lost. (The ADF adds no perspective distortion, so this is effectively a pure
    rotation; a projective transform is used only because it maps corners
    directly.)

    Args:
        a: Binary ink grid (0 == ink, 1 == background).
        corners: ``(tl, tr, br, bl)`` from :func:`find_frame_corners`.

    Returns:
        The deskewed, frame-cropped binary grid (0 == ink, 1 == background).
    """
    tl, tr, br, bl = corners

    def _dist(p: tuple[int, int], q: tuple[int, int]) -> float:
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    out_w = int(round((_dist(tl, tr) + _dist(bl, br)) / 2))
    out_h = int(round((_dist(tl, bl) + _dist(tr, br)) / 2))
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    mat = cv2.getPerspectiveTransform(src, dst)

    # Warp in ink-space (255 == ink) so the margin outside the frame comes in as
    # 0 (background), then re-binarize to the 0=ink / 1=bg convention.
    ink = ((1 - a) * 255).astype(np.uint8)
    warp = cv2.warpPerspective(
        ink, mat, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return (warp < 128).astype(np.uint8)


# On a correctly normalized page the frame sits at the very edge, so re-detecting
# it should put every corner within this fraction of the page size from its own
# corner of the image. A larger gap means the crop kept margin or sliced content
# (a bad detection), and the page is rejected.
VERIFY_CORNER_TOL = 0.02


def verify_normalized(a: np.ndarray) -> None:
    """Confirm a normalized page's frame really landed at the page edges.

    Re-runs frame detection on the *output* of :func:`deskew_to_frame`: if the
    warp was correct the frame is now the page boundary, so each detected corner
    must fall within :data:`VERIFY_CORNER_TOL` of the matching image corner. This
    catches a mis-detected source frame (e.g. a diagonal side that cropped
    through content) without any human review -- a wrongly cropped page leaves
    its re-detected frame inset or slanted, and a corner drifts past tolerance.

    Args:
        a: A normalized page from :func:`deskew_to_frame` (0 == ink, 1 == bg).

    Raises:
        ValueError: If the frame cannot be re-detected, or any corner sits
            farther than the tolerance from its image corner.
    """
    h, w = a.shape
    try:
        tl, tr, br, bl = find_frame_corners(a)
    except ValueError as exc:
        raise ValueError(f"post-normalize frame re-detection failed: {exc}") from exc

    tol_x = VERIFY_CORNER_TOL * w
    tol_y = VERIFY_CORNER_TOL * h
    targets = {"tl": (tl, (0, 0)), "tr": (tr, (w - 1, 0)),
               "br": (br, (w - 1, h - 1)), "bl": (bl, (0, h - 1))}
    for name, ((cx, cy), (ex, ey)) in targets.items():
        if abs(cx - ex) > tol_x or abs(cy - ey) > tol_y:
            raise ValueError(
                f"normalized frame corner {name} at ({cx},{cy}) is more than "
                f"{VERIFY_CORNER_TOL:.0%} from the page corner ({ex},{ey}) -- "
                f"the source frame was likely mis-detected and the crop is wrong"
            )


def render_page_binary(
    doc: pymupdf.Document,
    page_index: int,
    threshold: int = GRAY_THRESHOLD,
    scale: float = RENDER_SCALE,
) -> np.ndarray:
    """Render one PDF page to a full-resolution binary ink grid.

    Renders via ``get_pixmap`` so the page's ``/Rotate`` flag is applied (the
    ADF alternates 90/270 per sheet; honoring it yields an upright page). The
    grayscale render is thresholded to the pipeline's ink convention.

    Args:
        doc: An open PyMuPDF document.
        page_index: Zero-based page index to render.
        threshold: Pixels <= this (0..255) become ink.
        scale: Render matrix scale factor.

    Returns:
        A ``uint8`` grid, 0 == ink, 1 == background.
    """
    page = doc[page_index]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
    mode = "L" if pix.n == 1 else "RGB"
    gray = np.asarray(
        Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("L")
    )
    # imaging convention: 0 == ink, 1 == background.
    return (gray > threshold).astype(np.uint8)


def extract_pages(
    book: str,
    pdf_path: str,
    pages_dir: str,
    first_page: int | None = None,
    last_page: int | None = None,
    threshold: int = GRAY_THRESHOLD,
    books_dir: str = "books",
) -> list[str]:
    """Render a range of v1 PDF pages to full-resolution single-page PNGs.

    Args:
        book: Book name, e.g. ``"book1"`` (used to locate the output dir).
        pdf_path: Path to the v1 source PDF.
        pages_dir: Output subdirectory name under ``{books_dir}/{book}/``
            (e.g. ``pages_gray`` or ``pages_bw``).
        first_page: First content page index (inclusive, zero-based). Defaults to
            the book's entry in :data:`BOOK_PAGE_RANGES`.
        last_page: Last content page index (inclusive). Defaults to the book's
            entry in :data:`BOOK_PAGE_RANGES`.
        threshold: Grayscale binarization cutoff.
        books_dir: Root directory of per-book asset folders.

    Returns:
        The output page file paths, page 0 first.
    """
    if first_page is None or last_page is None:
        if book not in BOOK_PAGE_RANGES:
            raise KeyError(
                f"no page range for {book!r}; pass --first-page/--last-page or "
                f"add it to BOOK_PAGE_RANGES (known: {sorted(BOOK_PAGE_RANGES)})"
            )
        default_first, default_last = BOOK_PAGE_RANGES[book]
        first_page = default_first if first_page is None else first_page
        last_page = default_last if last_page is None else last_page

    out_dir = os.path.join(books_dir, book, pages_dir)
    os.makedirs(out_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    logger.info(
        "v1 extract %s: %s pages %d..%d -> %s (threshold %d)",
        book,
        pdf_path,
        first_page,
        last_page,
        out_dir,
        threshold,
    )

    written: list[str] = []
    failures: list[str] = []
    corner_meta: dict[str, dict] = {}
    for out_idx, src_idx in enumerate(range(first_page, last_page + 1)):
        # page identifier in every convention, so a failure is checkable at source.
        tag = f"content page {out_idx} (PDF index {src_idx}, viewer page {src_idx + 1})"
        a = render_page_binary(doc, src_idx, threshold=threshold)
        try:
            corners = find_frame_corners(a)
            a = deskew_to_frame(a, corners)
            verify_normalized(a)
        except ValueError as exc:
            failures.append(f"{tag}: {exc}")
            corner_meta[str(out_idx)] = {"pdf_index": src_idx, "viewer_page": src_idx + 1,
                                         "corners": None, "error": str(exc)}
            logger.warning("SKIPPED %s: %s", tag, exc)
            continue
        out_path = os.path.join(out_dir, f"{out_idx}.png")
        save_image(a, out_path)
        tl, tr, br, bl = corners
        corner_meta[str(out_idx)] = {
            "pdf_index": src_idx, "viewer_page": src_idx + 1,
            "corners": {"tl": list(tl), "tr": list(tr), "br": list(br), "bl": list(bl)},
        }
        logger.info("%s  %dx%d -> %s", tag, a.shape[1], a.shape[0], out_path)
        written.append(out_path)

    doc.close()

    # Persist per-page corner coordinates as metadata for every run.
    meta_path = os.path.join(out_dir, "corners.json")
    with open(meta_path, "w") as fh:
        json.dump({"book": book, "pdf": pdf_path, "pages": corner_meta}, fh, indent=2)
    logger.info("Wrote corner metadata -> %s", meta_path)

    logger.info("Wrote %d pages for %s (%d skipped)", len(written), book, len(failures))
    if failures:
        logger.warning("%d page(s) failed verification for %s:", len(failures), book)
        for f in failures:
            logger.warning("  %s", f)
    return written


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True, help="Book name, e.g. book1.")
    parser.add_argument("--pdf", required=True, help="Path to the v1 source PDF.")
    parser.add_argument(
        "--first-page", type=int, default=None,
        help="First content page (0-based, inclusive). Default: book's BOOK_PAGE_RANGES entry.",
    )
    parser.add_argument(
        "--last-page", type=int, default=None,
        help="Last content page (inclusive). Default: book's BOOK_PAGE_RANGES entry.",
    )
    parser.add_argument(
        "--pages-dir", required=True, help="Output subdir under books/{book}/ (e.g. pages_gray)."
    )
    parser.add_argument(
        "--threshold", type=int, default=GRAY_THRESHOLD, help="Grayscale ink cutoff (default 128)."
    )
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    extract_pages(
        args.book,
        args.pdf,
        args.pages_dir,
        first_page=args.first_page,
        last_page=args.last_page,
        threshold=args.threshold,
        books_dir=args.books_dir,
    )


if __name__ == "__main__":
    main()
