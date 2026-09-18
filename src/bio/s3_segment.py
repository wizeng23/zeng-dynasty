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

Output: ``books/{book}/bio/3_segment/{stem}/`` -- ``blocks.json`` (per-person boxes)
and one ``{stem}_{gen}_{k}.png`` crop per person (RTL / eldest-first within a band).
QA: ``books/{book}/qa/bio_s3/{stem}.png`` -- the merged section downscaled with band
rules drawn and each detected header label boxed (green = gate pass, red = fail).

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
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw

from src.imaging import get_image, save_image

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
MERGED_DIR = "2_merged"
SEGMENT_DIR = "3_segment"
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


def _blocks_from_labels(
    labels: list[tuple[int, int]], band_top: int, band_bottom: int,
    generation: int, band_idx: int, stem: str, width: int,
) -> list[Block]:
    """Turn header-label x-runs into per-person blocks, RTL / eldest-first.

    A block spans from one label's left edge to the next label's left edge (the entry
    reads right-to-left, so the eldest is rightmost). Ordered eldest-first = by
    descending x. IDs are ``{stem}_{gen}_{k}`` with k=0 the eldest (rightmost).
    """
    starts = sorted(s for s, _ in labels)  # left edges, ascending x
    # Right-to-left blocks: block i occupies [starts[i], starts[i+1]) ; last extends to width.
    ranges = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 1 if i + 1 < len(starts) else width - 1
        ranges.append((s, e))
    ranges.sort(key=lambda r: -r[0])  # eldest (rightmost) first
    blocks = []
    for k, (x0, x1) in enumerate(ranges):
        blocks.append(Block(
            id=f"{stem}_{generation}_{k}", generation=generation, band=band_idx,
            x=x0, y=band_top, width=x1 - x0 + 1, height=band_bottom - band_top,
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
            line = line.strip()
            if not line:
                continue
            node = json.loads(line)
            stem = _stem_of_notes(node.get("notes", ""))
            if stem:
                by[stem][node["generation"]] += 1
    return by


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
        blocks.extend(_blocks_from_labels(labels, top, bottom, gen, band_idx, stem, width))
    return blocks, per_band_labels, bands, gate_passed


def _save_qa(
    a: np.ndarray, bands: list[tuple[int, int]], per_band_labels: list[list[tuple[int, int]]],
    gate_passed: bool, out_path: str, scale: int = 8,
) -> None:
    """Downscaled merged section with band rules + detected label boxes drawn."""
    small = Image.fromarray((a * 255).astype("uint8")).convert("RGB")
    w, h = small.size
    small = small.resize((max(1, w // scale), max(1, h // scale)))
    draw = ImageDraw.Draw(small)
    color = (0, 160, 0) if gate_passed else (220, 0, 0)
    for _top, bottom in bands[:-1]:
        draw.line([(0, bottom // scale), (small.size[0], bottom // scale)], fill=(0, 0, 220), width=1)
    for band_idx, labels in enumerate(per_band_labels):
        top, bottom = bands[band_idx]
        for s, e in labels:
            draw.rectangle(
                [s // scale, (top + LABEL_Y0_RANGE.start) // scale, e // scale, (top + LABEL_Y0_RANGE.start + LABEL_WINDOW_H) // scale],
                outline=color, width=1,
            )
    small.save(out_path)


def segment_book(book: str, sections: list[str] | None = None, books_dir: str = "books",
                 data_dir: str = "data", qa: bool = True) -> dict[str, bool]:
    """Segment every (or the given) bio section of a book; return {stem: gate_passed}."""
    merged_dir = os.path.join(books_dir, book, BIO_DIR, MERGED_DIR)
    out_base = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR)
    qa_dir = os.path.join(books_dir, book, QA_DIR, QA_SUBDIR)
    if qa:
        os.makedirs(qa_dir, exist_ok=True)

    all_sections = sorted(
        (f[:-4] for f in os.listdir(merged_dir) if f.endswith(".png")),
        key=lambda s: int(s.split("_")[0]),
    )
    tree_by_stem = tree_counts_by_stem(os.path.join(data_dir, f"{book}_stitched.jsonl"))
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

        out_dir = os.path.join(out_base, sec)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "blocks.json"), "w") as fh:
            json.dump({"section": sec, "stem": stem, "gate_passed": passed,
                       "expected_per_gen": expected,
                       "detected_per_gen": [len(l) for l in labels],
                       "blocks": [asdict(b) for b in blocks]}, fh, indent=2)
        # one crop per person block, for stage-4 field OCR to read
        a = get_image(merged_path)
        for b in blocks:
            crop = a[b.y:b.y + b.height, b.x:b.x + b.width]
            save_image(crop, os.path.join(out_dir, f"{b.id}.png"))
        if qa:
            _save_qa(get_image(merged_path), bands, labels, passed,
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
