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


# --- reconcile + section/book runners (spec §2a, §5, §6) ---------------------------

EMPTY_VISION = {"lines": [], "father_char": None, "name": None, "sons": []}


def reconcile(paddle: dict, vision: dict, tree_name: str | None,
              child_names: set[str] | None = None) -> dict:
    """Merge the two reads into one block record; sons = flagged union (spec §2a).

    ``child_names`` (the tree's names for the next generation, ground truth) demotes a
    **single-reader** son that is NOT a real child name to a ``qa_flag`` instead of the
    union -- this drops OCR junk (garbled glyphs, traditional/simplified variants a reader
    got wrong) while keeping every son that either reader anchored to a real node. Sons
    both readers agree on are always kept.
    """
    p_sons, v_sons = list(paddle["sons"]), list(vision["sons"])
    ps, vs = set(p_sons), set(v_sons)
    union = list(dict.fromkeys(p_sons + v_sons))  # order-preserving dedupe
    flags = list(paddle["qa_flags"])
    sons = []
    for s in union:
        agreed = s in ps and s in vs
        if not agreed and child_names is not None and s not in child_names:
            flags.append(f"son '{s}': single-reader, not a tree child -- dropped")
            continue
        sons.append({"name": s, "agreed": agreed})
    name = vision.get("name") or tree_name
    if vision.get("name") and tree_name and vision["name"] != tree_name:
        flags.append(f"name: vision '{vision['name']}' != tree '{tree_name}'")
    if ps != vs:
        flags.append(f"sons: paddle {sorted(ps)} != vision {sorted(vs)}")
    return {
        "name": name,
        "father_char": vision.get("father_char"),
        "sons": sons,
        "daughters": parse_daughters(paddle["columns"]),
        "birth": parse_dates(paddle["columns"]).get("birth"),
        "paddle": {"columns": paddle["columns"], "sons": p_sons},
        "vision": {"lines": vision["lines"], "sons": v_sons},
        "qa_flags": flags,
    }


def _tree_names_by_gen(data_dir: str, book: str, stem: str) -> dict[int, list[str]]:
    """Tree node names for a subgraph stem, grouped by generation (tree/RTL order)."""
    names: dict[int, list[str]] = {}
    with open(os.path.join(data_dir, f"{book}_stitched.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n = json.loads(line)
            prov = (n.get("notes", "") or "").split(" | ", 1)[0].split("/", 1)[0]
            s = prov.rsplit("_", 1)[0] if prov else ""
            if s == stem:
                names.setdefault(n["generation"], []).append(n["name"])
    return names


def _section_stem(book: str, sec: str, books_dir: str, data_dir: str) -> str:
    """Positional map from a bio section to its tree subgraph stem (spec §2.3)."""
    from src.bio.s3_segment import tree_counts_by_stem, map_sections_to_stems

    merged_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    all_sections = sorted(
        (f[:-4] for f in os.listdir(merged_dir) if f.endswith(".png")),
        key=lambda s: int(s.split("_")[0]),
    )
    tree_by_stem = tree_counts_by_stem(os.path.join(data_dir, f"{book}_stitched.jsonl"))
    return map_sections_to_stems(all_sections, list(tree_by_stem))[sec]


def _validate(records: list[dict], tree_by_gen: dict[int, list[str]]) -> dict:
    """Gate: for each gen G, the union of gen-G sons should equal the gen-(G+1) tree names.

    A subgraph's *leaf* generation lists sons of the NEXT subgraph (not in this tree), so
    we only gate generations G where G+1 exists in this tree. The headline ``gen`` is the
    deepest such valid generation (gen-5 -> gen-6 for a 6-gen Book-3 subgraph = the P0).
    """
    sons_by_gen: dict[int, list[str]] = {}
    for r in records:
        for s in r["sons"]:
            sons_by_gen.setdefault(r["generation"], []).append(s["name"])

    per_gen = {}
    valid_src = [g for g in sons_by_gen if tree_by_gen.get(g + 1)]
    for g in valid_src:
        union = sorted(set(sons_by_gen[g]))
        tree = sorted(set(tree_by_gen.get(g + 1, [])))
        agreed = sum(s["agreed"] for r in records
                     if r["generation"] == g for s in r["sons"])
        per_gen[g] = {
            "child_gen": g + 1,
            "sons_union": union,
            "tree_names": tree,
            "missing_from_ocr": sorted(set(tree) - set(union)),
            "extra_in_ocr": sorted(set(union) - set(tree)),
            "agreed_count": agreed,
            "match": set(union) == set(tree),
        }

    base = {"sons_by_gen": {k: sorted(set(v)) for k, v in sons_by_gen.items()},
            "per_gen": per_gen}
    if not per_gen:
        base.update(match=None, note="no gate-able generation (no sons map to a tree gen)")
        return base
    g = max(per_gen)  # deepest gate-able = the P0 headline
    base.update(gen=per_gen[g]["child_gen"], **{k: per_gen[g][k] for k in
                ("sons_union", "tree_names", "missing_from_ocr", "extra_in_ocr",
                 "agreed_count", "match")})
    return base


def ocr_section(book: str, sec: str, books_dir: str, data_dir: str,
                vision_texts: dict[str, str] | None = None) -> dict:
    """OCR every block of a section; reconcile; validate against the tree oracle."""
    stem = _section_stem(book, sec, books_dir, data_dir)
    tree_by_gen = _tree_names_by_gen(data_dir, book, stem)
    engine = _paddle_engine()
    blocks = iter_blocks(book, sec, books_dir)
    records: list[dict] = []
    for b in blocks:
        # block k in gen g <-> tree node k in gen g (RTL/tree order); count gate passed
        tree_name = None
        gen_names = tree_by_gen.get(b.generation, [])
        if b.k < len(gen_names):
            tree_name = gen_names[b.k]
        paddle = paddle_read(b.img, engine)
        vision = EMPTY_VISION
        if vision_texts and b.id in vision_texts:
            vision = parse_vision(vision_texts[b.id])
        child_names = set(tree_by_gen.get(b.generation + 1, []))
        rec = reconcile(paddle, vision, tree_name, child_names or None)
        rec.update({"block_id": b.id, "generation": b.generation, "band": b.band, "k": b.k})
        records.append(rec)
        logger.info("%s gen%d k%d: sons=%s flags=%d", b.id, b.generation, b.k,
                    [s["name"] for s in rec["sons"]], len(rec["qa_flags"]))
    validation = _validate(records, tree_by_gen)
    return {"section": sec, "stem": stem,
            "engine": "ensemble:paddle+vision" if vision_texts else "paddle",
            "blocks": records, "validation": validation}


def save_crops(book: str, sec: str, books_dir: str) -> dict[str, str]:
    """Write each block's tight crop to 4_ocr/{sec}/crops/{id}.png for the vision pass."""
    out_dir = os.path.join(books_dir, book, BIO_DIR, OUT_DIR, sec, "crops")
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for b in iter_blocks(book, sec, books_dir):
        p = os.path.join(out_dir, f"{b.id}.png")
        b.img.save(p)
        paths[b.id] = p
    return paths


def _save_qa(book: str, sec: str, records: list[dict], books_dir: str,
             blocks: list[BlockCrop], engine) -> None:
    """Per-block overlay: char boxes drawn, columns numbered, extracted sons annotated."""
    from PIL import ImageDraw

    qa_dir = os.path.join(books_dir, book, "qa", "bio_s4", sec)
    os.makedirs(qa_dir, exist_ok=True)
    by_id = {r["block_id"]: r for r in records}
    for b in blocks:
        chars = _char_boxes(engine, np.asarray(b.img))
        cols = _cluster_columns(chars)
        im = b.img.convert("RGB").copy()
        draw = ImageDraw.Draw(im)
        for col in cols:
            for c in col:
                draw.rectangle([c["x0"], c["y0"], c["x1"], c["y1"]], outline=(0, 140, 0))
        rec = by_id.get(b.id, {})
        title = f"{b.id} sons={[s['name'] for s in rec.get('sons', [])]}"
        draw.text((4, 4), title, fill=(200, 0, 0))
        im.save(os.path.join(qa_dir, f"{b.id}.png"))


def ocr_book(book: str, sections: list[str] | None, books_dir: str, data_dir: str,
             vision_texts: dict[str, str] | None = None, qa: bool = True) -> dict:
    """OCR the given (or all) sections; write per-section sidecars + QA overlays."""
    merged_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    all_sections = sorted(
        (f[:-4] for f in os.listdir(merged_dir) if f.endswith(".png")),
        key=lambda s: int(s.split("_")[0]),
    )
    todo = sections or all_sections
    out_base = os.path.join(books_dir, book, BIO_DIR, OUT_DIR)
    os.makedirs(out_base, exist_ok=True)
    results: dict[str, dict] = {}
    engine = _paddle_engine()
    for sec in todo:
        rec = ocr_section(book, sec, books_dir, data_dir, vision_texts=vision_texts)
        with open(os.path.join(out_base, f"{sec}.json"), "w") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        if qa:
            _save_qa(book, sec, rec["blocks"], books_dir, iter_blocks(book, sec, books_dir),
                     engine)
        v = rec["validation"]
        logger.info("%s (%s): match=%s missing=%s extra=%s", sec, rec["stem"],
                    v.get("match"), v.get("missing_from_ocr"), v.get("extra_in_ocr"))
        results[sec] = rec
    return results


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
    # --no-vision => Paddle-only. The ensemble vision pass is driven by the executing
    # agent (dispatches subagents over save_crops output, then re-runs with vision_texts).
    ocr_book(args.book, args.sections, books_dir=args.books_dir, data_dir=args.data_dir,
             vision_texts=None)


if __name__ == "__main__":
    main()
