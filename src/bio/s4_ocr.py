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
import re
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


# --- Paddle reader: detect-then-resort + structural QA (spec §3, §5a) --------------

_ENGINE = None


def _paddle_engine():
    """Cached PP-OCRv5 full detect+recognize pipeline (lazy: paddle is optional)."""
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR

        _ENGINE = PaddleOCR(lang="ch")
    return _ENGINE


def _char_boxes(engine, rgb: np.ndarray) -> list[dict]:
    """Per-character boxes via ``return_word_box=True`` (text_word + text_word_boxes)."""
    res = engine.predict(rgb, return_word_box=True)
    if not res:
        return []
    d = getattr(res[0], "json", None)
    d = d.get("res", d) if isinstance(d, dict) else res[0]
    wtexts = d.get("text_word") or []
    wboxes = d.get("text_word_boxes") or []
    chars: list[dict] = []
    for line_words, line_boxes in zip(wtexts, wboxes):
        for ch, bx in zip(line_words, line_boxes):
            x0, y0, x1, y1 = (float(v) for v in bx)
            chars.append(dict(ch=ch, x0=x0, y0=y0, x1=x1, y1=y1,
                              xc=(x0 + x1) / 2, yc=(y0 + y1) / 2))
    return chars


def _cluster_columns(chars: list[dict], tol_frac: float = 0.7) -> list[list[dict]]:
    """Cluster chars into RTL columns by x-center; sort each column top-to-bottom."""
    if not chars:
        return []
    med_w = float(np.median([c["x1"] - c["x0"] for c in chars])) or 30.0
    tol = max(med_w * tol_frac, 12.0)
    ordered = sorted(chars, key=lambda c: -c["xc"])  # RTL
    cols: list[list[dict]] = []
    centers: list[float] = []
    for c in ordered:
        if cols and abs(centers[-1] - c["xc"]) <= tol:
            cols[-1].append(c)
            centers[-1] = float(np.mean([x["xc"] for x in cols[-1]]))
        else:
            cols.append([c])
            centers.append(c["xc"])
    for ci, col in enumerate(cols):
        col.sort(key=lambda c: c["y0"])
        for c in col:
            c["col"] = ci
    return cols


def _qa_checks(cols: list[list[dict]]) -> list[str]:
    """Structural flags from char boxes: pitch gap / intra-column gap / fill ratio."""
    if not cols:
        return ["no text detected"]
    all_chars = [c for col in cols for c in col]
    med_h = float(np.median([c["y1"] - c["y0"] for c in all_chars])) or 30.0
    centers = [float(np.mean([c["xc"] for c in col])) for col in cols]
    pitches = [abs(centers[i] - centers[i + 1]) for i in range(len(centers) - 1)]
    med_pitch = float(np.median(pitches)) if pitches else 0.0
    flags: list[str] = []
    for i, p in enumerate(pitches):
        if med_pitch and p > 1.6 * med_pitch:
            flags.append(f"col{i}->col{i+1}: pitch gap {p:.0f} (missed column?)")
    for ci, col in enumerate(cols):
        for j in range(len(col) - 1):
            gap = col[j + 1]["y0"] - col[j]["y1"]
            if gap > 1.2 * med_h:
                flags.append(f"col{ci}: y-gap {gap:.0f} after '{col[j]['ch']}' (dropped char?)")
        span = (col[-1]["y1"] - col[0]["y0"]) if col else 0
        if span > 3 * med_h and len(col) * med_h < 0.5 * span:
            flags.append(f"col{ci}: fill ratio low ({len(col)} chars / {span:.0f}px)")
    return flags


def paddle_read(img: Image.Image, engine) -> dict:
    """Read a tight person crop with detect-then-resort + structural QA.

    Returns ``{columns, char_boxes, sons, qa_flags}`` -- ``columns`` are the RTL-ordered
    joined column texts, ``char_boxes`` carry a ``col`` index for the QA overlay.
    """
    chars = _char_boxes(engine, np.asarray(img))
    cols = _cluster_columns(chars)
    col_texts = ["".join(c["ch"] for c in col) for col in cols]
    return {"columns": col_texts, "char_boxes": chars,
            "sons": parse_sons(col_texts), "qa_flags": _qa_checks(cols)}


# --- field parsers (marker-based; sons is the gated P0 field) ---------------------

_COUNT = "一二三四五六七八九十两"   # 生子{N}名
_ORD = "长次三四五六七八九幼元"     # birth-order prefixes on each child


_CN_NUM = {c: i for i, c in enumerate("一二三四五六七八九十", start=1)}


def _count_word(s: str) -> int | None:
    """Parse the ``N`` in ``生子N名`` (e.g. 一->1, 二->2). None if absent/unknown."""
    m = re.search(rf"生子([{_COUNT}]+)名", s)
    if not m:
        return None
    w = m.group(1)
    if w in ("两",):
        return 2
    return _CN_NUM.get(w)


def parse_sons(columns: list[str]) -> list[str]:
    """Son given-names from ``生子…名 <sons>``, RTL/eldest-first, stopping at ``生女``.

    ``columns`` are the ordered (RTL) column/line texts of the entry. Each son name is
    printed in its **own column**, so we locate the ``生子N名`` column and take the next
    columns as sons -- exactly ``N`` when the count word is present, else every column up
    to ``生女`` or a new clause. Ordinal prefixes (长/次/…) are stripped. [] when no
    ``生子`` clause.
    """
    idx = next((i for i, c in enumerate(columns) if re.search(rf"生子[{_COUNT}]*名", c)),
               None)
    if idx is None:
        return []
    n = _count_word(columns[idx])
    sons: list[str] = []
    for col in columns[idx + 1:]:
        if re.search(r"生女", col) or re.match(r"[配继殁歿葬享寿卒]", col):
            break
        name = re.sub(rf"^[{_ORD}]", "", col).strip()
        if name:
            sons.append(name)
        if n is not None and len(sons) >= n:
            break
    return sons


def parse_father(lines: list[str]) -> str | None:
    """Father name-char from the header line ``子[之|次|…]X`` (line 1 of a vision read)."""
    if not lines:
        return None
    m = re.match(rf"子[之{_ORD}]?(.)", lines[0])
    return m.group(1) if m else None


def parse_daughters(columns: list[str]) -> list[str]:
    """Daughter names from ``生女… 长/次…<name>适<place>`` (best-effort)."""
    joined = "".join(columns)
    m = re.search(rf"生女[{_COUNT}]*", joined)
    if not m:
        return []
    tail = joined[m.end():]
    return re.findall(rf"[{_ORD}](.{{1,2}}?)适", tail)


def parse_dates(columns: list[str]) -> dict:
    """Best-effort birth clause after ``生于``."""
    joined = "".join(columns)
    birth = re.search(r"生于([^殁歿葬配继]{2,14})", joined)
    return {"birth": birth.group(1) if birth else None}


# --- vision reader (Claude subagent via the Agent tool; spec §2a) ------------------

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "s4_vision_prompt.txt")


def vision_prompt(crop_path: str) -> str:
    """The validated transcription prompt with the crop path filled in."""
    with open(_PROMPT_PATH) as fh:
        return fh.read().format(crop_path=crop_path)


def parse_vision(text: str) -> dict:
    """Parse a vision subagent's RTL transcription into fields.

    Contract (see ``s4_vision_prompt.txt``): line 1 = father header ``子之X``, line 2 =
    the person's own name, lines 3+ = prose columns (RTL). Returns
    ``{lines, father_char, name, sons}``.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    father = parse_father(lines[:1]) if lines else None
    name = lines[1] if len(lines) > 1 and lines[1] != "?" else None
    sons = parse_sons(lines[2:])
    return {"lines": lines, "father_char": father, "name": name, "sons": sons}


# Dispatch contract for the vision reader
# --------------------------------------
# The image reader is a Claude vision subagent, invoked through the harness Agent tool --
# not an in-process API call. The batch runner (``ocr_book``) therefore writes every
# block crop to ``4_ocr/{stem}/crops/{id}.png`` and returns the crop paths; the executing
# agent dispatches one subagent per crop with ``vision_prompt(crop_path)``, collects the
# returned transcripts into ``{block_id: text}``, and passes that map back as
# ``vision_texts`` to ``ocr_section``. ``parse_vision`` turns each transcript into fields.
# ``--no-vision`` skips this entirely (Paddle-only) for quick iteration.


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
