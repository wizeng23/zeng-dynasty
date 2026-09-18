"""Bio stage 4: per-person field OCR (ensemble Paddle + vision subagent).

**Pure OCR step.** Reads each finalized stage-3 person crop and transcribes it with two
independent readers, saving BOTH raw reads plus a best-effort structured record. It does
**not** consult the tree, assign generations, validate, or stitch -- that is stage 5's
job (father/son cross-check against the graph). Stage 4 loses no OCR information: each
reader's raw output is preserved verbatim.

Readers (chosen by bake-off; see docs/specs/2026-09-18-bio-stage4-field-ocr-design.md):

* **Paddle** (PP-OCRv5, detect-then-resort): one ``predict(rgb, return_word_box=True)``
  on the crop -> per-character boxes; we DISCARD Paddle's reading order and re-sort the
  chars into right-to-left columns (top-to-bottom within a column). Robust to dense blocks
  where columns touch; yields char boxes for structural QA checks.
* **Vision subagent**: a Claude vision agent transcribes semantically, one line per column
  RTL (line 1 = the horizontal father header ``子之X``; line 2 = the person's name; 3+ =
  prose). Catches Paddle's blind spots (header + short name column). Driven by the
  executing agent via the Agent tool (see the dispatch contract below).

Input  (stage-3 final, post-QA): ``books/{book}/bio/3_segment/blocks.jsonl`` -- the
combined final index s3_post writes ("the index stage 4 consumes"), one row per block:
``section, stem, id, band, generation, box, gate_passed`` -- plus the ``{id}.png`` tight
crops beside it. (Per s3_post: block count per section is NOT guaranteed to equal the
subgraph's node count; block->node linking is a later evidence-based step -- so stage 4
does pure OCR, no node mapping.)
Output: ``books/{book}/bio/4_ocr/{stem}.jsonl`` -- one record per block with both raw
reads + best-effort structured fields + structural ``qa_flags``.

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

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
SEGMENT_DIR = "3_segment"
BLOCKS_INDEX = "blocks.jsonl"   # s3_post's combined final index; stage 4 consumes this
OUT_DIR = "4_ocr"


@dataclass
class BlockCrop:
    """One person's tight crop, as produced by stage 3 (RGB), with its metadata."""

    id: str
    box: list[int]
    img: Image.Image
    meta: dict  # the raw stage-3 jsonl record (band, generation, gate_passed, ...)


def _load_index(book: str, books_dir: str) -> list[dict]:
    """All block rows from s3_post's combined ``blocks.jsonl`` (the final index)."""
    path = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR, BLOCKS_INDEX)
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def list_sections(book: str, books_dir: str = "books") -> list[str]:
    """Section stems present in the block index, ordered by first page."""
    secs = {r["section"] for r in _load_index(book, books_dir)}
    return sorted(secs, key=lambda s: int(s.split("_")[0]))


def iter_blocks(book: str, sec: str, books_dir: str = "books") -> list[BlockCrop]:
    """Load stage-3's finalized per-person crops for one section, in index order.

    Reads the combined ``3_segment/blocks.jsonl`` (s3_post's authoritative post-QA index,
    "the index stage 4 consumes") filtered to ``sec``, and the matching ``{id}.png`` tight
    crops. Stage 3 emits crops already bounded to one person, so no re-cropping here.
    """
    seg_dir = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR)
    out: list[BlockCrop] = []
    for rec in _load_index(book, books_dir):
        if rec.get("section") != sec:
            continue
        img = Image.open(os.path.join(seg_dir, f"{rec['id']}.png")).convert("RGB")
        out.append(BlockCrop(rec["id"], rec.get("box", []), img, rec))
    return out


# --- Paddle reader: detect-then-resort + structural QA ------------------------------

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

    Returns ``{columns, char_boxes, sons, name, daughters, birth, qa_flags}``. ``columns``
    are the RTL-ordered joined column texts (the lossless raw read); the structured fields
    are best-effort parses over them.
    """
    chars = _char_boxes(engine, np.asarray(img))
    cols = _cluster_columns(chars)
    col_texts = ["".join(c["ch"] for c in col) for col in cols]
    return {
        "columns": col_texts,
        "char_boxes": chars,
        "name": col_texts[0] if col_texts else None,  # rightmost column ~ header+name
        "sons": parse_sons(col_texts),
        "daughters": parse_daughters(col_texts),
        "birth": parse_dates(col_texts).get("birth"),
        "qa_flags": _qa_checks(cols),
    }


# --- field parsers (marker-based) --------------------------------------------------

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

    Each son name is printed in its **own column**, so we locate the ``生子N名`` column and
    take the next columns as sons -- exactly ``N`` when the count word is present, else
    every column up to ``生女`` or a new clause. Ordinal prefixes (长/次/…) are stripped.
    [] when no ``生子`` clause.
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


# --- vision reader (Claude subagent via the Agent tool) ----------------------------

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "s4_vision_prompt.txt")


def vision_prompt(crop_path: str) -> str:
    """The validated transcription prompt with the crop path filled in."""
    with open(_PROMPT_PATH) as fh:
        return fh.read().format(crop_path=crop_path)


def parse_vision(text: str) -> dict:
    """Parse a vision subagent's RTL transcription into fields.

    Contract (see ``s4_vision_prompt.txt``): line 1 = father header ``子之X``, line 2 =
    the person's own name, lines 3+ = prose columns (RTL). Returns
    ``{text, lines, father_char, name, sons, daughters, birth}``. ``text`` is the raw
    transcript, preserved losslessly.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    father = parse_father(lines[:1]) if lines else None
    name = lines[1] if len(lines) > 1 and lines[1] not in ("?", "") else None
    return {
        "text": text,
        "lines": lines,
        "father_char": father,
        "name": name,
        "sons": parse_sons(lines[2:]),
        "daughters": parse_daughters(lines[2:]),
        "birth": parse_dates(lines[2:]).get("birth"),
    }


# Dispatch contract for the vision reader
# --------------------------------------
# The vision reader is a Claude subagent invoked through the harness Agent tool, not an
# in-process API call. ``save_crops(book, sec)`` writes each block crop and returns
# {id: path}; the executing agent dispatches one subagent per crop with
# ``vision_prompt(path)``, collects {id: transcript}, and passes that as ``vision_texts``
# to ``ocr_section``. ``--no-vision`` skips it (Paddle-only).


EMPTY_VISION = {"text": "", "lines": [], "father_char": None, "name": None,
                "sons": [], "daughters": [], "birth": None}


# --- assemble the lossless per-block record + runners ------------------------------

def block_record(block: BlockCrop, paddle: dict, vision: dict) -> dict:
    """Combine both readers into one lossless record (no tree, no dropping).

    Structured fields keep each reader's read separately (stage 5 merges against the
    tree); ``raw`` preserves each reader's full output verbatim.
    """
    return {
        "id": block.id,
        "box": block.box,
        "band": block.meta.get("band"),
        "generation": block.meta.get("generation"),
        # best-effort structured, per reader (unmerged)
        "name": {"paddle": paddle.get("name"), "vision": vision.get("name")},
        "father_char": {"vision": vision.get("father_char")},
        "sons": {"paddle": paddle.get("sons", []), "vision": vision.get("sons", [])},
        "daughters": {"paddle": paddle.get("daughters", []),
                      "vision": vision.get("daughters", [])},
        "birth": {"paddle": paddle.get("birth"), "vision": vision.get("birth")},
        # lossless raw reads
        "raw": {
            "paddle": {"columns": paddle.get("columns", []),
                       "char_boxes": paddle.get("char_boxes", [])},
            "vision": {"text": vision.get("text", "")},
        },
        "qa_flags": paddle.get("qa_flags", []),
    }


def ocr_section(book: str, sec: str, books_dir: str,
                vision_texts: dict[str, str] | None = None) -> list[dict]:
    """OCR every block of a section with both readers; return lossless records."""
    engine = _paddle_engine()
    records: list[dict] = []
    for b in iter_blocks(book, sec, books_dir):
        paddle = paddle_read(b.img, engine)
        vision = EMPTY_VISION
        if vision_texts and b.id in vision_texts:
            vision = parse_vision(vision_texts[b.id])
        rec = block_record(b, paddle, vision)
        records.append(rec)
        logger.info("%s: paddle_sons=%s vision_sons=%s flags=%d", b.id,
                    rec["sons"]["paddle"], rec["sons"]["vision"], len(rec["qa_flags"]))
    return records


def save_crops(book: str, sec: str, books_dir: str) -> dict[str, str]:
    """Copy stage-3 crops to 4_ocr/{sec}/crops/{id}.png; return {id: path} for vision."""
    out_dir = os.path.join(books_dir, book, BIO_DIR, OUT_DIR, sec, "crops")
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for b in iter_blocks(book, sec, books_dir):
        p = os.path.join(out_dir, f"{b.id}.png")
        b.img.save(p)
        paths[b.id] = p
    return paths


def _save_qa(book: str, sec: str, books_dir: str, blocks: list[BlockCrop],
             records: list[dict], engine) -> None:
    """Per-block overlay: detected char boxes drawn; both readers' sons annotated."""
    from PIL import ImageDraw

    qa_dir = os.path.join(books_dir, book, "qa", "bio_s4", sec)
    os.makedirs(qa_dir, exist_ok=True)
    by_id = {r["id"]: r for r in records}
    for b in blocks:
        cols = _cluster_columns(_char_boxes(engine, np.asarray(b.img)))
        im = b.img.convert("RGB").copy()
        draw = ImageDraw.Draw(im)
        for col in cols:
            for c in col:
                draw.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=(0, 140, 0))
        rec = by_id.get(b.id, {})
        sons = rec.get("sons", {})
        draw.text((4, 4), f"{b.id} P{sons.get('paddle')} V{sons.get('vision')}",
                  fill=(200, 0, 0))
        im.save(os.path.join(qa_dir, f"{b.id}.png"))


def ocr_book(book: str, sections: list[str] | None, books_dir: str,
             vision_texts: dict[str, str] | None = None, qa: bool = True) -> dict:
    """OCR the given (or all) sections; write per-section jsonl sidecars + QA overlays."""
    todo = sections or list_sections(book, books_dir)
    out_base = os.path.join(books_dir, book, BIO_DIR, OUT_DIR)
    os.makedirs(out_base, exist_ok=True)
    engine = _paddle_engine()
    results: dict[str, list[dict]] = {}
    for sec in todo:
        records = ocr_section(book, sec, books_dir, vision_texts=vision_texts)
        with open(os.path.join(out_base, f"{sec}.jsonl"), "w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if qa:
            _save_qa(book, sec, books_dir, iter_blocks(book, sec, books_dir), records, engine)
        logger.info("%s: %d blocks written", sec, len(records))
        results[sec] = records
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True)
    parser.add_argument("--sections", nargs="+", default=None)
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--no-vision", action="store_true", help="Paddle-only run")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # --no-vision => Paddle-only. The ensemble vision pass is driven by the executing
    # agent (save_crops -> dispatch subagents -> re-run with vision_texts).
    ocr_book(args.book, args.sections, books_dir=args.books_dir,
             vision_texts=None, qa=not args.no_qa)


if __name__ == "__main__":
    main()
