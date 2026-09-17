"""Seed the cross-book stitch pairings -- which node in an earlier book is the
same person as a subtree-root in the next book.

Each book after the first re-prints, as the root of some of its subtrees, a
person who already appears as a *leaf* in the previous book (the book boundary
falls mid-lineage). Stitching folds each such duplicate root into its canonical
earlier-book leaf. This script derives those pairings and writes them, one file
per *child* (later) book, as the child book's stitch-output:

  * ``books/book2/stitch_pairs.json``  -- book1 <- book2 pairings
  * ``books/book3/stitch_pairs.json``  -- book2 <- book3 pairings

File shape::

    {
      "earlier": "book1", "later": "book2",
      "pairs": [["b1:13_16_44", "b2:0_3_0/4_5_0"], ...],   # [survivor_earlier, dup_later]
      "unconnected": ["b2:106_113_0 (贞年)", ...]           # later-book roots with no match
    }

Provenances are namespaced by book (``b1:``/``b2:``/``b3:``) because the bare
``{graph}_{index}`` strings collide across books. The survivor is ALWAYS the
earlier book's node (it is closer to the root 点); the later book's duplicate is
folded into it. These files are hand-editable ground truth: this script SEEDS
them (``--force`` to overwrite) but by default won't clobber an existing file, so
manual corrections survive a re-run.

Two derivations:
  * **book1 <- book2** is *recovered* from the existing ``data/tree_combined.jsonl``
    (the one-off Books-1+2 splice), by finding the single edge where a book2 node's
    father is a book1 node -- i.e. where book2 was grafted onto book1.
  * **book2 <- book3** (and, later, book3 <- book4) is *matched by name*: each
    child-book section-root (``{graph}_0``) is paired to the earlier book's
    same-name leaf. Unique name -> auto-paired. Ambiguous (name recurs) or absent
    -> left for a human, listed under ``unconnected`` with the candidates.

CLI::

    python -m scripts.qa.cross_book_pairs            # seed both, skip existing
    python -m scripts.qa.cross_book_pairs --force    # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

DATA_DIR = "data"
BOOKS_DIR = "books"


def _load(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def _prov(n: dict) -> str:
    """The stable ``{graph}_{localindex}`` provenance = first ' | ' notes segment."""
    return (n.get("notes") or "").split(" | ", 1)[0]


def _tag(book: str, prov: str) -> str:
    """Namespace a bare provenance with its book, e.g. ``b1:13_16_44``."""
    return f"b{book[-1]}:{prov}"


def _pairs_path(child_book: str) -> str:
    return os.path.join(BOOKS_DIR, child_book, "stitch_pairs.json")


def recover_book1_book2() -> dict:
    """Recover the book1 <- book2 pairing from ``tree_combined.jsonl``.

    tree_combined is the existing (un-scripted) Books-1+2 splice. A book2 node is
    tagged ``book2`` in its notes; a book1 node is not. The graft point is the lone
    edge whose child is book2 and whose father is book1 -- there, book2's root was
    folded into a book1 leaf. We read that leaf and book2's own root to recover the
    (survivor_earlier, dup_later) pair.
    """
    tc = _load(os.path.join(DATA_DIR, "tree_combined.jsonl"))
    by_id = {n["id"]: n for n in tc}

    def book_of(n: dict) -> str:
        return "book2" if "book2" in (n.get("notes") or "") else "book1"

    # The cross-book seam: book2 child whose father is a book1 node.
    seam_fathers = {
        by_id[n["father"]]["id"]
        for n in tc
        if n["father"] != -1
        and n["father"] in by_id
        and book_of(n) == "book2"
        and book_of(by_id[n["father"]]) == "book1"
    }
    if len(seam_fathers) != 1:
        logger.warning(
            "expected exactly 1 book1<-book2 graft in tree_combined, found %d",
            len(seam_fathers),
        )

    # The book1 survivor = the graft father; the book2 duplicate = book2's own root.
    b2s = _load(os.path.join(DATA_DIR, "book2_stitched.jsonl"))
    b2_roots = [n for n in b2s if n["father"] == -1]
    # book2's *main* root is the one whose name matches the book1 graft leaf.
    pairs: list[list[str]] = []
    unconnected: list[str] = []
    for fid in seam_fathers:
        b1_leaf = by_id[fid]
        # match book2 root by name to this book1 leaf
        cand = [r for r in b2_roots if r["name"] and r["name"] == b1_leaf["name"]]
        if len(cand) == 1:
            pairs.append([_tag("book1", _prov(b1_leaf)), _tag("book2", _prov(cand[0]))])
        else:
            logger.warning(
                "book1 leaf %r (%s) matched %d book2 roots; leaving unpaired",
                b1_leaf["name"], _prov(b1_leaf), len(cand),
            )
    matched_dups = {p[1] for p in pairs}
    for r in b2_roots:
        if _tag("book2", _prov(r)) not in matched_dups:
            unconnected.append(f"{_tag('book2', _prov(r))} ({r['name']})")

    return {
        "earlier": "book1",
        "later": "book2",
        "pairs": pairs,
        "unconnected": unconnected,
    }


def match_by_name(earlier_book: str, later_book: str) -> dict:
    """Pair each later-book section-root to the earlier book's same-name leaf.

    A section-root is a graph root ``{graph}_0``. Its canonical is the earlier book's
    leaf (childless node) with the same OCR'd name at the earlier book's deepest
    generation (the book boundary is one generation, so the earlier book's *leaves*
    are the people the later book expands). Unique name -> paired. Ambiguous or
    absent -> listed in ``unconnected`` with candidate provenances for a human.
    """
    earlier = _load(os.path.join(DATA_DIR, f"{earlier_book}_stitched.jsonl"))
    later = _load(os.path.join(DATA_DIR, f"{later_book}_stitched.jsonl"))

    # Earlier-book candidates = childless leaves, indexed by name. (We don't restrict
    # to a single generation: the leaves that continue may sit at slightly different
    # depths; a leaf is a leaf.)
    leaves_by_name: dict[str, list[dict]] = {}
    for n in earlier:
        if not n.get("children") and n.get("name"):
            leaves_by_name.setdefault(n["name"], []).append(n)

    # Later-book section roots: a root (father == -1) whose provenance ends _0.
    roots = [
        n for n in later
        if n["father"] == -1 and _prov(n).split("/", 1)[0].endswith("_0")
    ]
    roots.sort(key=lambda n: int(_prov(n).split("_")[0]))

    pairs: list[list[str]] = []
    unconnected: list[str] = []
    for r in roots:
        cands = leaves_by_name.get(r["name"], [])
        if len(cands) == 1:
            pairs.append([_tag(earlier_book, _prov(cands[0])), _tag(later_book, _prov(r))])
        elif not cands:
            unconnected.append(f"{_tag(later_book, _prov(r))} ({r['name']}): NO earlier leaf")
        else:
            opts = ", ".join(_tag(earlier_book, _prov(c)) for c in cands)
            unconnected.append(
                f"{_tag(later_book, _prov(r))} ({r['name']}): {len(cands)} candidates [{opts}]"
            )

    logger.info(
        "%s<-%s: %d/%d roots auto-paired, %d unconnected",
        earlier_book, later_book, len(pairs), len(roots), len(unconnected),
    )
    return {
        "earlier": earlier_book,
        "later": later_book,
        "pairs": pairs,
        "unconnected": unconnected,
    }


def _write(child_book: str, data: dict, force: bool) -> None:
    path = _pairs_path(child_book)
    if os.path.exists(path) and not force:
        logger.info("exists, not overwriting (use --force): %s", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(
        "wrote %s: %d pairs, %d unconnected", path, len(data["pairs"]), len(data["unconnected"])
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing pair files")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    _write("book2", recover_book1_book2(), args.force)
    _write("book3", match_by_name("book2", "book3"), args.force)
    _write("book4", match_by_name("book3", "book4"), args.force)


if __name__ == "__main__":
    main()
