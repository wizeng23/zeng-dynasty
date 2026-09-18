"""Bio stage 3: segment each merged bio section into per-person blocks.

Input: ``books/{book}/bio/2_merged/{stem}.png`` + ``{stem}.json`` (stage-2 output;
the JSON carries ``rules_y``, ``seam_x``, ``size``, ``pages_reading_order``).

For each section we slice the merged image into 5 generation-bands at the 4 ``rules_y``
rules (the bands are generations 2..6; generation 1 -- the branch root -- has no bio
row), then within each band find one **person block** per printed header label.

Header anchor (why this, not whitespace or a template)
------------------------------------------------------
Every person's entry opens with a short **horizontal header caption** ``子之X`` (子 =
son, 之 = birth-order particle, X = the father's name char) printed near the band top,
above the person's own (bold, larger) name; the prose then runs top-to-bottom below.
Whitespace gaps fail (a gap *within* one person's columns is as wide as the gap
*between* people) and a bare ``子`` match fails (``子`` also occurs in prose, e.g.
生子...). But the header caption has a stable **geometric signature**: a
~385-400px-wide horizontal stripe of ink sitting near the top of the band. We detect
exactly those stripes -- one per person -- which gives the per-generation head count.

The header's vertical offset varies between sections (wider/deeper sections print it
higher), so the label y-window is found adaptively per band (:func:`find_labels`).

Count gate (hard fail)
----------------------
The parsed tree (``data/{book}_stitched.jsonl``) is the oracle: we know exactly how
many people are in each generation of each subgraph. Detected labels per band MUST
equal tree nodes per generation; on any mismatch the section **hard-fails** (per the
design spec) with a QA overlay so the miss can be investigated. We never silently
split/merge to force the count.

This is the PRE-QA stage: it detects blocks and writes them for review, but writes NO
crop images (boxes change in QA, so cutting crops now would be wasted). The 3-step flow:

  1. this stage  -> ``books/{book}/bio/3_segment/qa_input/{a}_{b}.jsonl``  (one block/row)
  2. QA editor   -> ``books/{book}/bio/3_segment/{a}_{b}.jsonl``           (approved copy)
  3. s3_post     -> ``books/{book}/bio/3_segment/{a}_{b}_{c}_{d}.png`` + final jsonl

Block provenance ``{a}_{b}_{c}_{d}``: ``a_b`` = section start/end pages, ``c`` = band
index (0 = top row = gen 2 ... 4 = bottom = gen 6), ``d`` = index within the row (0 =
rightmost/eldest, increasing leftward -- RTL). Each JSONL row is one block, in order,
carrying section context (stem, expected/detected per gen, gate_passed) on every row.

QA overlay: ``books/{book}/qa/bio_s3/{stem}.png`` -- the merged section downscaled with
band rules drawn, each detected header boxed, and a per-row diff (``gen3: 4/5 MISSING 1``,
red when short, green when complete) so missing bios per generation are visible.

Run:
    PYTHONPATH=. python -m src.bio.s3_segment --book book3
    PYTHONPATH=. python -m src.bio.s3_segment --book book3 --sections 2_9 66_77
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from src.imaging import get_image

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
MERGED_DIR = "2_merged"
SEGMENT_DIR = "3_segment"        # final images + QA-approved jsonls live here (post writes them)
QA_INPUT_DIR = "qa_input"        # 3_segment/qa_input/ : pre-QA detections (this stage writes)
QA_DIR = "qa"
QA_SUBDIR = "bio_s3"

# The bio bands are generations 2..6 (gen 1 = branch root, no bio row).
BAND_GENERATIONS = (2, 3, 4, 5, 6)

# --- header-label detection constants (tuned on Book 3, 2026-09-18) ---
# The header caption is a ~385-400px-wide horizontal ink stripe near the band top.
LABEL_WINDOW_H = 120          # height of the y-window we sum ink over
LABEL_MIN_W = 340             # a header stripe is ~385-400px wide; accept 340..440
LABEL_MAX_W = 440
LABEL_MERGE_GAP = 50          # bridge <=50px gaps (whitespace inside/between header chars)
LABEL_COL_INK = 2             # a column "has ink" in the window if > this many ink px
# A real header is solid printed text (peak column ink ~90-106); ADF ink-smear streaks
# (see the v1 scan smear artifact) are ~header-width but faint (peak ~6-16). Require a
# label's densest column to clear this floor, well between the two populations.
LABEL_MIN_PEAK_INK = 40
# The header top-offset varies per section, so scan these window-top offsets and take
# the one yielding the most label-width stripes.
LABEL_Y0_RANGE = range(20, 140, 15)


@dataclass
class Block:
    """One person's bounding box within the merged section (band-relative y kept absolute)."""

    id: str
    generation: int
    band: int
    x: int
    y: int
    width: int
    height: int


def detect_bands(rules_y: list[int], height: int) -> list[tuple[int, int]]:
    """Return the 5 (top, bottom) row ranges = generations 2..6, split at the 4 rules."""
    bounds = [0] + list(rules_y) + [height]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _label_runs(band_ink: np.ndarray, y0: int) -> list[tuple[int, int]]:
    """Header-width column runs whose ink falls in the y-window [y0, y0+H)."""
    window = band_ink[y0:y0 + LABEL_WINDOW_H, :]
    has = window.sum(axis=0) > LABEL_COL_INK
    runs: list[list[int]] = []
    i, n = 0, len(has)
    while i < n:
        if not has[i]:
            i += 1
            continue
        start = i
        while i < n and has[i]:
            i += 1
        runs.append([start, i - 1])
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= LABEL_MERGE_GAP:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    out = []
    for s, e in merged:
        if not (LABEL_MIN_W <= (e - s + 1) <= LABEL_MAX_W):
            continue
        # reject faint smear streaks: a real header has a dense (dark) peak column
        if int(window[:, s:e + 1].sum(axis=0).max()) < LABEL_MIN_PEAK_INK:
            continue
        out.append((s, e))
    return out


def find_labels(band_ink: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """Find header-label x-runs in a band's ink grid (ink=1).

    Scans candidate window-top offsets and returns the (labels, y0) of the offset that
    yields the most header-width stripes -- one per person, left-to-right in pixels.
    """
    best: list[tuple[int, int]] = []
    best_y0 = LABEL_Y0_RANGE.start
    for y0 in LABEL_Y0_RANGE:
        labels = _label_runs(band_ink, y0)
        if len(labels) > len(best):
            best, best_y0 = labels, y0
    return sorted(best), best_y0


# A person's box stops this many px short of the next person's label so adjacent
# boxes don't touch (the header 子之X of the next entry stays outside this box).
BLOCK_GAP = 30


def _blocks_from_labels(
    labels: list[tuple[int, int]], band_top: int, band_bottom: int,
    generation: int, band_idx: int, stem: str, width: int,
) -> list[Block]:
    """Turn header-label x-runs into per-person blocks, RTL / eldest-first.

    The book reads right-to-left: within a band each person's header ``子之X`` sits at
    the RIGHT of their block and their prose fills leftward. So a block's RIGHT edge is
    its own label's right edge, and its LEFT edge is just past the next (younger)
    person's label. The eldest (rightmost label) is block k=0.
    """
    spans = sorted(labels, key=lambda se: -se[1])   # by right edge, rightmost (eldest) first
    blocks = []
    for d, (ls, le) in enumerate(spans):
        right = le                                   # box right edge = this label's right edge
        # left edge = just past the next-younger person's label (or the image left)
        left = (spans[d + 1][1] + 1 + BLOCK_GAP) if d + 1 < len(spans) else 0
        left = min(left, ls)                         # never cut into this person's own label
        # id = {stem}_{c}_{d}: c = band index (0=top row), d = 0 rightmost/eldest, up leftward
        blocks.append(Block(
            id=f"{stem}_{band_idx}_{d}", generation=generation, band=band_idx,
            x=left, y=band_top, width=right - left + 1, height=band_bottom - band_top,
        ))
    return blocks


# --- tree oracle -------------------------------------------------------------------

def _stem_of_notes(notes: str) -> str:
    """Subgraph stem from a node's provenance notes ('{graph}_{idx} | ...')."""
    prov = (notes or "").split(" | ", 1)[0].strip()
    if "/" in prov:
        prov = prov.split("/", 1)[0].strip()
    return prov.rsplit("_", 1)[0] if prov else ""


def tree_counts_by_stem(jsonl_path: str) -> dict[str, Counter]:
    """Map each subgraph stem -> Counter of node count per generation."""
    by: dict[str, Counter] = defaultdict(Counter)
    with open(jsonl_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            node = json.loads(line)
            stem = _stem_of_notes(node.get("notes", ""))
            if stem:
                by[stem][node["generation"]] += 1
    return by


def tree_nodes_by_stem_gen(jsonl_path: str) -> dict[str, dict[int, list[dict]]]:
    """Map stem -> generation -> nodes in DFS eldest-first (RTL) order.

    Bio entries within a generation-band are laid out in the tree's depth-first,
    eldest-first (right-to-left) order (children arrays are already eldest-first). A DFS
    that recurses children in order therefore yields the bio reading order, so block d
    (0 = rightmost/eldest) maps to the d-th node of its generation here. Used by the
    post step to assign a tree node to each bio, once QA has completed the section.
    """
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    by_id = {n["id"]: n for n in rows}
    roots_by_stem: dict[str, list[dict]] = defaultdict(list)
    for n in rows:
        if n.get("generation") == 1 or n.get("father", -1) in (-1, None):
            stem = _stem_of_notes(n.get("notes", ""))
            if stem:
                roots_by_stem[stem].append(n)
    out: dict[str, dict[int, list[dict]]] = {}
    for stem, roots in roots_by_stem.items():
        per_gen: dict[int, list[dict]] = defaultdict(list)
        seen: set[int] = set()

        def dfs(nid: int) -> None:
            if nid in seen or nid not in by_id:
                return
            seen.add(nid)
            n = by_id[nid]
            per_gen[n["generation"]].append(n)
            for c in n["children"]:
                dfs(c)

        for r in sorted(roots, key=lambda n: n["id"]):
            dfs(r["id"])
        out[stem] = per_gen
    return out


def map_sections_to_stems(section_stems: list[str], tree_stems: list[str]) -> dict[str, str]:
    """Positional map: the i-th bio section (by start page) <-> the i-th tree subgraph.

    The book alternates graph run then bio section 1:1, so the ordered lists align even
    though the page numbers differ (graph-page stems vs bio-page stems).
    """
    sec = sorted(section_stems, key=lambda s: int(s.split("_")[0]))
    tre = sorted(tree_stems, key=lambda s: int(s.split("_")[0]))
    if len(sec) != len(tre):
        raise ValueError(f"section/stem count mismatch: {len(sec)} sections vs {len(tre)} subgraphs")
    return dict(zip(sec, tre))


# --- per-section segmentation ------------------------------------------------------

def segment_section(
    merged_path: str, json_path: str, expected_per_gen: list[int],
) -> tuple[list[Block], list[list[tuple[int, int]]], list[tuple[int, int]], bool]:
    """Segment one merged section into blocks and run the count gate.

    Returns (blocks, per_band_labels, bands, gate_passed). ``expected_per_gen`` is the
    tree node count for generations 2..6 (len 5); a band with expected 0 must find 0.
    """
    with open(json_path) as fh:
        meta = json.load(fh)
    rules_y = meta["rules_y"]
    width, height = meta["size"]
    stem = f"{meta['pages_reading_order'][0]}_{meta['pages_reading_order'][-1]}"

    a = get_image(merged_path)
    ink = 1 - a
    bands = detect_bands(rules_y, height)

    blocks: list[Block] = []
    per_band_labels: list[list[tuple[int, int]]] = []
    gate_passed = True
    for band_idx, (top, bottom) in enumerate(bands):
        labels, _y0 = find_labels(ink[top:bottom, :])
        per_band_labels.append(labels)
        gen = BAND_GENERATIONS[band_idx]
        if len(labels) != expected_per_gen[band_idx]:
            gate_passed = False
            logger.warning("%s gen%d: labels=%d nodes=%d", stem, gen, len(labels), expected_per_gen[band_idx])
        for blk in _blocks_from_labels(labels, top, bottom, gen, band_idx, stem, width):
            nl, nt, nr, nb = tighten_box(ink, [blk.x, blk.y, blk.x + blk.width, blk.y + blk.height])
            blk.x, blk.y, blk.width, blk.height = nl, nt, nr - nl, nb - nt
            blocks.append(blk)
    return blocks, per_band_labels, bands, gate_passed


def _save_qa(
    a: np.ndarray, bands: list[tuple[int, int]], per_band_labels: list[list[tuple[int, int]]],
    gate_passed: bool, expected: list[int], detected: list[int], out_path: str, scale: int = 8,
) -> None:
    """Downscaled merged section with band rules, detected label boxes, and a per-row
    diff (expected vs detected, and how many are MISSING) drawn at the left of each band."""
    small = Image.fromarray((a * 255).astype("uint8")).convert("RGB")
    w, h = small.size
    small = small.resize((max(1, w // scale), max(1, h // scale)))
    draw = ImageDraw.Draw(small)
    for _top, bottom in bands[:-1]:
        draw.line([(0, bottom // scale), (small.size[0], bottom // scale)], fill=(0, 0, 220), width=1)
    for band_idx, labels in enumerate(per_band_labels):
        top, bottom = bands[band_idx]
        ok = detected[band_idx] == expected[band_idx]
        color = (0, 160, 0) if ok else (220, 0, 0)
        for s, e in labels:
            y0 = (top + LABEL_Y0_RANGE.start) // scale
            draw.rectangle([s // scale, y0, e // scale, y0 + LABEL_WINDOW_H // scale],
                           outline=color, width=1)
        # per-row diff label at the band's top-left: gen, detected/expected, missing count
        gen = BAND_GENERATIONS[band_idx]
        miss = expected[band_idx] - detected[band_idx]
        tag = f"gen{gen}: {detected[band_idx]}/{expected[band_idx]}"
        if miss:
            tag += f"  MISSING {miss}" if miss > 0 else f"  EXTRA {-miss}"
        draw.text((4, top // scale + 2), tag, fill=color)
    small.save(out_path)


# Crop tightening: a column/row is "text" if its ink exceeds this floor (kills the faint
# ADF smear that otherwise defeats a plain any-ink bounding box), plus a little padding.
TIGHTEN_INK_FLOOR = 12
TIGHTEN_PAD = 40


def tighten_box(ink: np.ndarray, box: list[int]) -> list[int]:
    """Shrink [l,t,r,b] to the block's real text extent within the merged ink grid.

    ``ink`` is the full-section ink grid (1=ink). Columns/rows with ink above
    TIGHTEN_INK_FLOOR are text; everything else (blank + smear) is trimmed, with a small
    pad kept. Returns the original box if no text clears the floor.
    """
    l, t, r, b = box
    sub = ink[t:b, l:r]
    cols = np.where(sub.sum(axis=0) > TIGHTEN_INK_FLOOR)[0]
    rows = np.where(sub.sum(axis=1) > TIGHTEN_INK_FLOOR)[0]
    if len(cols) == 0 or len(rows) == 0:
        return box
    nl = l + max(0, int(cols.min()) - TIGHTEN_PAD)
    nr = l + min(sub.shape[1], int(cols.max()) + 1 + TIGHTEN_PAD)
    nt = t + max(0, int(rows.min()) - TIGHTEN_PAD)
    nb = t + min(sub.shape[0], int(rows.max()) + 1 + TIGHTEN_PAD)
    return [nl, nt, nr, nb]


def _block_row(section: str, stem: str, b: Block, expected: list[int],
               detected: list[int], gate_passed: bool) -> dict:
    """One JSONL row per person block (the shared schema for pre-QA, QA, and final).

    Section-level context (stem, expected/detected per gen, gate_passed) is repeated on
    every row so each JSONL is self-describing without a separate header.
    """
    return {
        "section": section, "stem": stem,
        "id": b.id, "band": b.band, "generation": b.generation,
        "box": [b.x, b.y, b.x + b.width, b.y + b.height],
        "expected_per_gen": expected, "detected_per_gen": detected,
        "gate_passed": gate_passed,
    }


def segment_book(book: str, sections: list[str] | None = None, books_dir: str = "books",
                 data_dir: str = "data", qa: bool = True) -> dict[str, bool]:
    """Segment every (or the given) bio section of a book; return {stem: gate_passed}."""
    merged_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    qa_input_dir = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR, QA_INPUT_DIR)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    all_sections = sorted(
        (f[:-4] for f in os.listdir(merged_dir) if f.endswith(".png")),
        key=lambda s: int(s.split("_")[0]),
    )
    jsonl = os.path.join(data_dir, f"{book}_stitched.jsonl")
    tree_by_stem = tree_counts_by_stem(jsonl)
    sec_to_stem = map_sections_to_stems(all_sections, list(tree_by_stem))
    todo = sections or all_sections

    results: dict[str, bool] = {}
    for sec in todo:
        stem = sec_to_stem[sec]
        expected = [tree_by_stem[stem].get(g, 0) for g in BAND_GENERATIONS]
        merged_path = os.path.join(merged_dir, f"{sec}.png")
        json_path = os.path.join(merged_dir, f"{sec}.json")
        blocks, labels, bands, passed = segment_section(merged_path, json_path, expected)
        results[sec] = passed
        detected = [len(l) for l in labels]

        # Pre-QA output: ONE JSONL per section (one block per row, in order), NO images.
        # Images are deferred to the post script -- boxes (and provenance d-indices) will
        # change in QA, so cutting crops now would just be thrown away. This JSONL is the
        # QA editor's input.
        os.makedirs(qa_input_dir, exist_ok=True)
        with open(os.path.join(qa_input_dir, f"{sec}.jsonl"), "w") as fh:
            for b in blocks:
                fh.write(json.dumps(_block_row(sec, stem, b, expected, detected, passed),
                                    ensure_ascii=False) + "\n")
        if qa:
            _save_qa(get_image(merged_path), bands, labels, passed, expected, detected,
                     os.path.join(qa_dir, f"{sec}.png"))
        logger.info("%s (%s): %s  expected=%s detected=%s", sec, stem,
                    "PASS" if passed else "FAIL", expected, [len(l) for l in labels])
    npass = sum(results.values())
    logger.info("count gate: %d/%d sections pass", npass, len(results))
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True)
    parser.add_argument("--sections", nargs="+", default=None, help="section stems (default: all)")
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--no-qa", action="store_true", help="skip QA overlays")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    segment_book(args.book, sections=args.sections, books_dir=args.books_dir,
                 data_dir=args.data_dir, qa=not args.no_qa)


if __name__ == "__main__":
    main()
