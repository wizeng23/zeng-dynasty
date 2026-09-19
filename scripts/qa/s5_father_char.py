"""QA: verify each bio's father HEADER against the tree, within a subgraph.

Every biography opens with a small horizontal header caption ``子<order><father>`` --
e.g. ``子长炯`` = "son (子), eldest (长), of 炯". Three independent facts, all checkable
against the tree (the father and siblings are local nodes in the same subgraph):

  1. **char 1 == 子** -- the "son of" marker. (``女`` = a daughter entry; ``继`` = adopted.)
  2. **char 2 == the person's birth order among sons** -- 长/之 = eldest, 次/二 = 2nd,
     三 = 3rd, 四…八 = 4th…8th. Compared against the node's index in ``father.children``
     (which is eldest-first / RTL). A disagreement means either the order glyph is
     mis-OCR'd OR the sibling ordering in the tree is wrong.
  3. **char 3 == the father's given-name last char** -- an independent confirmation the
     bio linked to the right node.

Stage 4's structured ``father_char`` field keeps only char 3 (``parse_father`` strips 子 and
the order glyph), but the FULL header survives verbatim in the raw vision transcription
(``raw.vision.text`` line 1). This QA reads that raw line, so it can check all three parts --
the check William originally specified. It needs ``data/{book}_bio_linked.jsonl`` (post
:mod:`src.bio.s5_link`, for the tree + which bios linked where) and the Stage-4 OCR under
``books/{book}/bio/4_ocr/*.jsonl`` (for the raw header text, keyed by bio ``block_id``).

Flags cluster, and the cluster says which side is wrong:

  * many children of ONE father sharing a char-3 error  -> the FATHER'S name is mis-OCR'd
    (fix one node, many flags clear);
  * ``order_mismatch`` on a single-graph family         -> mis-OCR'd order glyph, or the
    tree's sibling ordering is off for that family;
  * ``order_multigraph``                                -> the father is a section root
    re-printed across several graph slices (each re-numbering its sons from 长/次/三); the
    slices were never folded into one ordered sibling list, so this is a STITCH artifact,
    not a header error -- expected for the unstitched B4 floater roots (see s5_bio_stitch);
  * char-1 not 子 (and not a daughter/adopted marker)   -> header mis-read.

Output: a triaged terminal summary + ``data/{book}_father_char_qa.json`` (every flagged node
with its parsed header, expected values, and provenance). REPORT ONLY -- nothing downstream
consumes it; read it and correct OCR / links / ordering by hand.

Run::

    python -m scripts.qa.s5_father_char --book book3
    python -m scripts.qa.s5_father_char --book book4
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)

# Birth-order glyph -> 0-based son index (eldest = 0). A set per index: the zupu uses
# 长 and 之 interchangeably for the eldest, 次 and 二 for the second; the rest are the
# plain numerals. Learned empirically from Books 3 & 4 raw headers.
ORDER_TO_INDEX: dict[str, int] = {}
for _idx, _glyphs in enumerate([
    "长之",        # 1st son
    "次二",        # 2nd son
    "三",          # 3rd
    "四", "五", "六", "七", "八", "九", "十",
]):
    for _g in _glyphs:
        ORDER_TO_INDEX[_g] = _idx

# Header char-1 markers that are NOT a plain son-of-father entry, so char-2/3 don't apply.
NON_SON_HEAD = {"女", "继"}  # 女 = daughter entry, 继 = adopted/continued line


def _prov(notes: str) -> str:
    return (notes or "").split(" | ", 1)[0].strip()


def load_headers(book: str, books_dir: str) -> dict[str, str]:
    """{bio block_id -> raw vision header (line 1 of raw.vision.text)}."""
    out: dict[str, str] = {}
    for f in glob.glob(os.path.join(books_dir, book, "bio", "4_ocr", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            text = ((r.get("raw") or {}).get("vision") or {}).get("text") or ""
            out[r["id"]] = text.split("\n", 1)[0] if text else ""
    return out


def _graph_of(prov: str) -> str:
    """The graph key of a provenance ``{tag}:{start}_{end}_{i}`` -> ``{tag}:{start}_{end}``.

    Strips the trailing within-graph index. A stitch trace head (``canon/dup``) is split
    on ``/`` first so a folded node counts under its canonical graph.
    """
    prov = prov.split("/", 1)[0]
    return prov.rsplit("_", 1)[0] if "_" in prov else prov


def check_header(header: str, node: dict, father: dict, multi_graph: bool) -> list[str]:
    """Return a list of problem codes for this header vs the tree (empty = all good).

    ``multi_graph`` is True when the father's children were concatenated from more than one
    printed graph slice (a section root re-printed across graphs, each slice re-numbering its
    sons from 长/次/三). The birth-order glyph is then RELATIVE to a slice, so it can't be
    checked against the merged child index -- an order disagreement there is a STITCH artifact
    (the slices were never folded into one ordered sibling list), not a header/OCR error, and
    is bucketed as ``order_multigraph`` so it doesn't drown out the real signals.
    """
    problems: list[str] = []
    if len(header) < 1:
        return ["no_header"]
    if header[0] in NON_SON_HEAD:
        # a daughter/adopted header -- char-2/3 semantics differ; flag as its own bucket
        return [f"non_son_head:{header[0]}"]
    if header[0] != "子":
        problems.append(f"char1_not_子:{header[0]}")
    if len(header) < 3:
        problems.append(f"header_short:{header}")
        return problems  # can't check order/name without all three chars

    order_glyph, name_char = header[1], header[2]

    # char 2: birth order vs actual index among father's sons
    if node["id"] in father["children"]:
        actual_idx = father["children"].index(node["id"])
        expected_idx = ORDER_TO_INDEX.get(order_glyph)
        if expected_idx is None:
            problems.append(f"order_unknown:{order_glyph}")
        elif expected_idx != actual_idx:
            kind = "order_multigraph" if multi_graph else "order_mismatch"
            problems.append(f"{kind}:{order_glyph}(={expected_idx})!=idx{actual_idx}")

    # char 3: father's name last char (checkable regardless of slicing)
    if father.get("name") and name_char != father["name"][-1]:
        problems.append(f"name_mismatch:{name_char}!={father['name'][-1]}")
    return problems


def check_book(book: str, data_dir: str = "data", books_dir: str = "books") -> dict:
    linked = [json.loads(l) for l in
              open(os.path.join(data_dir, f"{book}_bio_linked.jsonl")) if l.strip()]
    by_id = {n["id"]: n for n in linked}
    headers = load_headers(book, books_dir)

    counts = collections.Counter()
    # flagged nodes grouped by father, so a mis-OCR'd father clusters its children.
    by_father: dict[int, dict] = {}
    problem_kinds = collections.Counter()

    for n in linked:
        bio = n.get("bio")
        if not bio:
            continue
        header = headers.get(bio.get("block_id"), "")
        father = by_id.get(n["father"])
        if not father or not father.get("name"):
            counts["root_or_missing_father"] += 1
            continue
        counts["checked"] += 1
        # does this father's child list mix more than one printed graph slice? if so its
        # per-slice birth-order can't be checked against the merged child index.
        child_graphs = {_graph_of(_prov(by_id[c].get("notes", "")))
                        for c in father["children"] if c in by_id}
        multi_graph = len(child_graphs) > 1
        problems = check_header(header, n, father, multi_graph)
        if not problems:
            counts["clean"] += 1
            continue
        counts["flagged"] += 1
        for p in problems:
            problem_kinds[p.split(":", 1)[0]] += 1
        rec = by_father.setdefault(father["id"], {
            "father_id": father["id"], "father_name": father["name"],
            "father_prov": _prov(father.get("notes", "")), "children_flagged": [],
        })
        rec["children_flagged"].append({
            "id": n["id"], "name": n["name"], "prov": _prov(n.get("notes", "")),
            "header": header, "problems": problems,
        })

    report = {
        "book": book,
        "checked": counts["checked"], "clean": counts["clean"], "flagged": counts["flagged"],
        "root_or_missing_father": counts["root_or_missing_father"],
        "problem_kinds": dict(problem_kinds.most_common()),
        "flagged_by_father": sorted(by_father.values(),
                                    key=lambda r: -len(r["children_flagged"])),
    }
    out = os.path.join(data_dir, f"{book}_father_char_qa.json")
    with open(out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    pct = 100 * counts["clean"] // counts["checked"] if counts["checked"] else 0
    logger.info("%s: %d/%d headers clean (%d%%); %d flagged, %d root/missing-father -> %s",
                book, counts["clean"], counts["checked"], pct, counts["flagged"],
                counts["root_or_missing_father"], out)
    logger.info("  problem kinds: %s", report["problem_kinds"])
    for r in report["flagged_by_father"][:15]:
        kids = ", ".join(f"{c['name']}({c['header']}|{','.join(c['problems'])})"
                         for c in r["children_flagged"])
        logger.info("  father %s [%s]: %d flagged -> %s",
                    r["father_name"], r["father_prov"], len(r["children_flagged"]), kids)
    return report


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--books-dir", default="books")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(message)s")
    check_book(a.book, a.data_dir, a.books_dir)


if __name__ == "__main__":
    main()
