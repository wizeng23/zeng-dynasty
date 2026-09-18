"""Bio stage 4: per-person field OCR (ensemble Paddle + vision subagent).

Reads each person BLOCK from a merged bio section (``books/{book}/bio/2_merged/{stem}.png``
+ blocks recomputed via stage 3), tight-crops it, and extracts fields -- P0 = **sons** --
with two independent readers reconciled:

* **Paddle** (PP-OCRv5, detect-then-resort): run Paddle's detector, discard its reading
  order, re-sort the per-character boxes into right-to-left columns (top-to-bottom within
  a column). Deterministic; yields char boxes for structural QA checks. Strong on the
  vertical prose columns; weak on the horizontal father header and short bold name.
* **Vision subagent**: a Claude vision agent transcribes the crop semantically (one line
  per column, RTL; line 1 = father header ``子之X``; line 2 = the person's name; 3+ =
  prose). Catches exactly Paddle's blind spots.

The two reads are reconciled per field; sons are the flagged union (``agreed`` = found by
both). Output: ``books/{book}/bio/4_ocr/{stem}.json`` + QA overlays under
``books/{book}/qa/bio_s4/{stem}/``.

See ``docs/specs/2026-09-18-bio-stage4-field-ocr-design.md``.

Run:
    PYTHONPATH=. python -m src.bio.s4_ocr --book book3 --sections 2_9 --no-vision
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.imaging import get_image
from src.bio.s3_segment import (
    detect_bands,
    find_labels,
    _blocks_from_labels,
    BAND_GENERATIONS,
)

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
MERGED_DIR = "2_merged"
OUT_DIR = "4_ocr"

# tight_crop: a person's entry sits in a contiguous right-side column cluster; a blank
# column run this wide (px) walking left from the right edge ends the entry.
CLUSTER_GAP = 400
PAD = 20


@dataclass
class BlockCrop:
    """One person's tight RGB crop, RTL / eldest-first within a generation band."""

    id: str
    generation: int
    band: int
    k: int
    img: Image.Image


def tight_crop(gray: np.ndarray) -> np.ndarray:
    """Trim a (possibly very wide, mostly blank) block to its content.

    Walk left from the right edge keeping inked columns; stop at the first blank-column
    run >= ``CLUSTER_GAP`` (the gap before the next-younger sibling's reserved space).
    Then trim to the ink bbox and pad. A no-op on already-tight crops, so this stays
    correct once stage 3 emits tight crops itself.
    """
    ink = gray < 128
    has = ink.sum(axis=0) > 3
    n = len(has)
    right = n - 1
    while right >= 0 and not has[right]:
        right -= 1
    if right < 0:
        return gray
    left, blank, i = right, 0, right
    while i >= 0:
        if has[i]:
            left, blank = i, 0
        else:
            blank += 1
            if blank >= CLUSTER_GAP:
                break
        i -= 1
    rows = ink[:, left:right + 1].any(axis=1)
    r0 = int(np.argmax(rows))
    r1 = len(rows) - int(np.argmax(rows[::-1]))
    c0 = max(0, left - PAD)
    c1 = min(gray.shape[1], right + 1 + PAD)
    r0 = max(0, r0 - PAD)
    r1 = min(gray.shape[0], r1 + PAD)
    return gray[r0:r1, c0:c1]


def iter_blocks(book: str, sec: str, books_dir: str = "books") -> list[BlockCrop]:
    """Per-person tight crops for a merged bio section, RTL / eldest-first per band.

    Recomputes the block boxes from the merged image with the stage-3 primitives (rather
    than reading stage 3's PNG files, whose naming is still being finalized).
    """
    merged_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    with open(os.path.join(merged_dir, f"{sec}.json")) as fh:
        meta = json.load(fh)
    a = get_image(os.path.join(merged_dir, f"{sec}.png"))  # 0=ink / 1=bg
    ink = 1 - a
    w, h = meta["size"]
    stem = f"{meta['pages_reading_order'][0]}_{meta['pages_reading_order'][-1]}"
    bands = detect_bands(meta["rules_y"], h)
    out: list[BlockCrop] = []
    for bi, (top, bot) in enumerate(bands):
        labels, _ = find_labels(ink[top:bot, :])
        blocks = _blocks_from_labels(labels, top, bot, BAND_GENERATIONS[bi], bi, stem, w)
        for k, b in enumerate(blocks):
            gray = (a[b.y:b.y + b.height, b.x:b.x + b.width] * 255).astype("uint8")
            gray = tight_crop(gray)
            out.append(
                BlockCrop(b.id, b.generation, bi, k, Image.fromarray(gray).convert("RGB"))
            )
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True)
    parser.add_argument("--sections", nargs="+", default=None)
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-vision", action="store_true", help="Paddle-only run")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit("ocr_book not yet implemented (see plan Task 5)")


if __name__ == "__main__":
    main()
