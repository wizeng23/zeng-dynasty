"""Bio stage 5: associate each bio entry (stage 4) with its tree-graph node, and fold
the bio fields into the tree.

Association is per **subgraph** (the book alternates a tree subgraph then its bio
section, 1:1 in page order, so the i-th bio section maps to the i-th subgraph). Within a
subgraph both bio blocks and tree nodes are ordered generation-band then RTL/eldest-first,
so for a given generation we match the block list against the node list with a cascade:

  1. **exact name** -- the block's vision name == a tree node's name (80% on Book 3).
  2. **fuzzy name** -- same length and <=1 differing char, or one is a
     traditional/simplified-ish variant (share all-but-one char). Handles OCR variants.
  3. **sons overlap** -- the block's sons (vision or paddle) intersect a node's
     children's names (a strong, independent signal).
  4. **positional** -- remaining blocks <-> remaining nodes in order (RTL/eldest-first),
     used only when counts line up after 1-3.

Each matched node gets a ``bio`` field = the block's structured fields + provenance +
both raw reads (lossless). Unmatched blocks/nodes are reported.

Input:  data/{book}_stitched.jsonl (tree oracle) + books/{book}/bio/4_ocr/*.jsonl +
        books/{book}/bio/3_segment/blocks.jsonl (section->stem map).
Output: data/{book}_bio_linked.jsonl (every tree node, with ``bio`` where linked) +
        data/{book}_bio_link_report.json (per-section match stats + unmatched lists).

Run:
    PYTHONPATH=. python -m src.bio.s5_link --book book3
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


# --- loading ----------------------------------------------------------------------

def _stem_of(node: dict) -> str:
    prov = (node.get("notes", "") or "").split(" | ", 1)[0].split("/", 1)[0]
    return prov.rsplit("_", 1)[0] if prov else ""


def load_tree(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def section_to_stem(book: str, books_dir: str) -> dict[str, str]:
    rows = [json.loads(l) for l in
            open(os.path.join(books_dir, book, "bio", "3_segment", "blocks.jsonl"))]
    return {r["section"]: r["stem"] for r in rows}


def load_bio_sections(book: str, books_dir: str, ocr_dir: str = "4_ocr") -> dict[str, list[dict]]:
    out = {}
    for f in glob.glob(os.path.join(books_dir, book, "bio", ocr_dir, "*.jsonl")):
        sec = os.path.basename(f)[:-6]
        out[sec] = [json.loads(l) for l in open(f) if l.strip()]
    return out


# --- name matching ----------------------------------------------------------------

def _block_name(rec: dict) -> str | None:
    return rec.get("name", {}).get("vision") or rec.get("name", {}).get("paddle")


def _block_sons(rec: dict) -> set[str]:
    s = set(rec.get("sons", {}).get("vision") or [])
    s |= set(rec.get("sons", {}).get("paddle") or [])
    return {x for x in s if x}


def _fuzzy(a: str, b: str) -> bool:
    """True if a and b are the same name modulo <=1 char (OCR variant / trad-simp)."""
    if not a or not b or len(a) != len(b):
        return False
    diff = sum(1 for x, y in zip(a, b) if x != y)
    return diff <= 1


# --- per-subgraph association ------------------------------------------------------

def link_generation(blocks: list[dict], nodes: list[dict],
                    node_children: dict[int, set[str]]):
    """Match bio blocks to tree nodes within one generation.

    Returns (pairs, unmatched_blocks, unmatched_nodes) where each pair is (block, node, how).
    """
    pairs = []
    unblk = list(blocks)
    unnode = list(nodes)

    def take(blk, nd, how):
        pairs.append((blk, nd, how))
        unblk.remove(blk)
        unnode.remove(nd)

    # 1. exact name
    for blk in list(unblk):
        nm = _block_name(blk)
        hit = next((n for n in unnode if n["name"] == nm), None)
        if hit:
            take(blk, hit, "exact")
    # 2. fuzzy name
    for blk in list(unblk):
        nm = _block_name(blk)
        hit = next((n for n in unnode if nm and _fuzzy(n["name"], nm)), None)
        if hit:
            take(blk, hit, "fuzzy")
    # 3. sons overlap
    for blk in list(unblk):
        sons = _block_sons(blk)
        if not sons:
            continue
        hit = next((n for n in unnode if sons & node_children.get(n["id"], set())), None)
        if hit:
            take(blk, hit, "sons")
    # 4. positional (only if the leftovers line up 1:1, preserving order)
    if len(unblk) == len(unnode) and unblk:
        for blk, nd in zip(unblk, unnode):
            pairs.append((blk, nd, "positional"))
        unblk, unnode = [], []
    return pairs, unblk, unnode


def make_bio_field(blk: dict) -> dict:
    """The bio payload folded onto a tree node (structured + provenance + lossless raw)."""
    return {
        "block_id": blk["id"],
        "name_ocr": blk.get("name"),
        "father_char": blk.get("father_char"),
        "sons": blk.get("sons"),
        "daughters": blk.get("daughters"),
        "birth": blk.get("birth"),
        "qa_flags": blk.get("qa_flags", []),
        "raw": blk.get("raw"),
    }


def link_book(book: str, books_dir: str = "books", data_dir: str = "data",
              ocr_dir: str = "4_ocr") -> dict:
    tree = load_tree(os.path.join(data_dir, f"{book}_stitched.jsonl"))
    by_id = {n["id"]: n for n in tree}
    # node -> set of children names (for the sons signal)
    node_children = {n["id"]: {by_id[c]["name"] for c in n.get("children", []) if c in by_id}
                     for n in tree}
    # tree grouped by stem, gen
    tb = collections.defaultdict(lambda: collections.defaultdict(list))
    for n in tree:
        s = _stem_of(n)
        if s:
            tb[s][n["generation"]].append(n)

    sec2stem = section_to_stem(book, books_dir)
    bio = load_bio_sections(book, books_dir, ocr_dir)

    linked_by_node: dict[int, dict] = {}
    report = {"book": book, "sections": {}, "totals": collections.Counter()}
    for sec, recs in sorted(bio.items(), key=lambda kv: int(kv[0].split("_")[0])):
        stem = sec2stem.get(sec)
        byg_blk = collections.defaultdict(list)
        for r in recs:
            byg_blk[r["generation"]].append(r)
        sec_rep = {"stem": stem, "matched": 0, "blocks": len(recs),
                   "how": collections.Counter(), "unmatched_blocks": [], "unmatched_nodes": []}
        for g, blks in byg_blk.items():
            nodes = tb.get(stem, {}).get(g, [])
            pairs, ublk, unode = link_generation(blks, nodes, node_children)
            for blk, nd, how in pairs:
                linked_by_node[nd["id"]] = make_bio_field(blk)
                sec_rep["matched"] += 1
                sec_rep["how"][how] += 1
                report["totals"][how] += 1
            sec_rep["unmatched_blocks"] += [b["id"] for b in ublk]
            sec_rep["unmatched_nodes"] += [n["id"] for n in unode]
        report["sections"][sec] = sec_rep
        report["totals"]["matched"] += sec_rep["matched"]
        report["totals"]["blocks"] += len(recs)

    # fold onto the tree
    out_path = os.path.join(data_dir, f"{book}_bio_linked.jsonl")
    with open(out_path, "w") as fh:
        for n in tree:
            node = dict(n)
            if n["id"] in linked_by_node:
                node["bio"] = linked_by_node[n["id"]]
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")
    rep_path = os.path.join(data_dir, f"{book}_bio_link_report.json")
    report["totals"] = dict(report["totals"])
    for s in report["sections"].values():
        s["how"] = dict(s["how"])
    with open(rep_path, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    t = report["totals"]
    logger.info("%s: linked %d/%d bio blocks to nodes -> %s (%s)", book,
                t.get("matched", 0), t.get("blocks", 0), out_path,
                {k: t[k] for k in ("exact", "fuzzy", "sons", "positional") if k in t})
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--books-dir", default="books")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ocr-dir", default="4_ocr",
                   help="per-section source dir under books/{book}/bio/ (e.g. 5_records "
                        "for human-reviewed records from s5_build_bio)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    link_book(a.book, a.books_dir, a.data_dir, a.ocr_dir)


if __name__ == "__main__":
    main()
