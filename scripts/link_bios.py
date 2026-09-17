"""Best-effort association of biography text to Book-3 tree nodes.

Retrieves per-person bio entries from the Stage-8 per-page Mistral OCR
(`books/{book}/8_bio_ocr/`) and attaches each to its tree node in
`data/{book}_stitched.jsonl`, writing `data/{book}_linked.jsonl`.

Approach (best-effort, per William 2026-09-16 -- "half-decent to show"):
- A person's bio entry opens with a name header, in one of two OCR'd forms after
  whitespace is collapsed:
    A: <father>之<given 1-2>子       (table / first-page form)
    B: 子<order:之次三四五长...><father><given 1-2>   (flat / continuation form)
  Split each page's OCR at these headers -> candidate person-entries.
- Group bio pages by the tree GRAPH they follow (Stage-2 `follows_graph`, mapped to
  each node's provenance graph-stem in `notes`). Within a subgraph, walk the tree's
  nodes generation by generation, eldest-first (RTL == the book's order), and match
  each node to a bio entry by GIVEN NAME (the tree name is the oracle; OCR noise is
  tolerated). Cross-page tails that spilled to a neighbour page may be lost -- accepted.
- Fill `node["biography"]` with the matched entry text; leave unmatched nodes blank.

CLI:  PYTHONPATH=. python -m scripts.link_bios --book book3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

CJK = r"[㐀-鿿〇]"
# Header form A: <father>之<given 1-2>, trailing 子 optional (Book 4 often omits it),
# anchored by a following biography clause (生/配/殁) or end so a 之 inside prose is
# not a false hit. Form B: 子<birth-order><father><given> (flat / continuation form).
HDR_A = re.compile(rf"({CJK})之({CJK}{{1,2}})子?(?=生|配|殁|$)")
HDR_B = re.compile(rf"子[之次三四五六七八九长幼]({CJK})({CJK}{{1,2}})")


def _collapse(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("|", ""))


def extract_entries(book: str, page: int, books_dir: str) -> list[dict]:
    """Split one bio page's OCR markdown into candidate person-entries."""
    path = os.path.join(books_dir, book, "8_bio_ocr", f"{page}.json")
    if not os.path.exists(path):
        return []
    md = _collapse(json.load(open(path))["markdown"])
    marks = []
    for m in HDR_A.finditer(md):
        marks.append((m.start(), m.group(1), m.group(2)))
    for m in HDR_B.finditer(md):
        marks.append((m.start(), m.group(1), m.group(2)))
    marks.sort()
    entries = []
    for i, (pos, father, given) in enumerate(marks):
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(md)
        entries.append({"father": father, "given": given,
                        "text": md[pos:nxt], "page": page})
    return entries


def _graph_stem(node: dict) -> str:
    """The subgraph a node came from, from its provenance notes (e.g. 0_1_7 -> 0_1)."""
    prov = node.get("notes", "").split(" | ")[0].split("/")[0]
    return prov.rsplit("_", 1)[0] if "_" in prov else prov


def link(book: str, books_dir: str = "books", data_dir: str = "data") -> dict:
    nodes = [json.loads(l) for l in open(os.path.join(data_dir, f"{book}_stitched.jsonl"))
             if l.strip()]
    by_id = {n["id"]: n for n in nodes}

    # bio pages grouped by the graph page they follow -> the preceding graph's stem.
    pt = json.load(open(os.path.join(books_dir, book, "2_classify", "page_types.json")))
    # map graph-page-index -> a graph stem that starts at that page (from node provenance)
    stem_start = {}  # first page index of a stem -> stem
    for n in nodes:
        stem = _graph_stem(n)
        if "_" in stem:
            start = int(stem.split("_")[0])
            stem_start.setdefault(start, stem)

    # bio pages per subgraph stem (via follows_graph == the graph's start page)
    pages_for_stem: dict[str, list[int]] = defaultdict(list)
    for k, v in pt["pages"].items():
        if v["type"] != "bio":
            continue
        fg = v.get("follows_graph")
        if fg is None:
            continue
        # find the stem whose start page is the greatest <= fg
        earlier = [st for st in stem_start if st <= fg]
        if not earlier:
            continue
        stem = stem_start[max(earlier)]
        pages_for_stem[stem].append(int(k))

    # nodes per subgraph stem, grouped by generation, eldest-first (id order == RTL)
    nodes_for_stem: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        nodes_for_stem[_graph_stem(n)].append(n)

    def one_off(a: str, b: str) -> bool:
        """Same length, differ in at most one character (single OCR slip)."""
        return len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1

    def take_match(name: str, pool: list[dict]) -> dict | None:
        """Pop the best entry for `name` from `pool`: an exact given-name match if
        any, else the single closest unused entry within 1 OCR-char (the tree name
        is the oracle, so a nearest ≤1-char match is safe within one subgraph)."""
        best = None
        best_d = 2
        for e in pool:
            if e.get("_used"):
                continue
            if e["given"] == name:
                e["_used"] = True
                return e
            if one_off(e["given"], name):
                d = sum(x != y for x, y in zip(e["given"], name))
                if d < best_d:
                    best_d = d
                    best = e
        if best is not None:
            best["_used"] = True
            return best
        return None

    # Build each subgraph's entry pool once; keep a global pool for a fallback pass.
    stem_entries: dict[str, list[dict]] = {}
    global_pool: list[dict] = []
    for stem in nodes_for_stem:
        entries = []
        for pg in sorted(pages_for_stem.get(stem, [])):
            entries += extract_entries(book, pg, books_dir)
        stem_entries[stem] = entries
        global_pool += entries

    matched = 0
    total = 0
    unfilled: list[dict] = []
    # Pass 1: match within the node's own subgraph (exact, then unique fuzzy).
    for stem, gnodes in nodes_for_stem.items():
        pool = stem_entries[stem]
        for n in gnodes:
            if n.get("generation", 1) < 2 or not n.get("name"):
                continue
            total += 1
            e = take_match(n["name"], pool)
            if e:
                n["biography"] = e["text"]
                matched += 1
            else:
                unfilled.append(n)
    # Pass 2: global fallback for still-unfilled nodes (subgraph grouping can strand
    # entries when follows_graph is imperfect). Exact-then-unique-fuzzy over all
    # remaining entries; the tree name keeps this from mis-assigning.
    for n in unfilled:
        e = take_match(n["name"], global_pool)
        if e:
            n["biography"] = e["text"]
            matched += 1

    out = os.path.join(data_dir, f"{book}_linked.jsonl")
    with open(out, "w") as f:
        for n in nodes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    logger.info("Linked %d/%d bio-eligible nodes (%.0f%%) -> %s",
                matched, total, matched / total * 100 if total else 0, out)
    return {"matched": matched, "total": total, "out": out}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="book3")
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")
    link(args.book, books_dir=args.books_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
