"""Bio stage 5 (fold): attach the linked bios onto the combined cross-book tree.

``data/{book}_bio_linked.jsonl`` carries a ``bio`` field on each per-book node (stage 5
link). The combined graph ``data/tree.jsonl`` renumbers ids but keeps each node's origin
in ``notes`` as ``{book-tag}:{provenance}`` (e.g. ``b3:0_1_0``), while the per-book node's
``notes`` is the bare ``{provenance}`` (``0_1_0``). We key each linked bio by
``{book}:{provenance}`` and fold it onto the matching combined node.

Output: ``data/tree_bio.jsonl`` -- the combined tree with a ``bio`` field where a bio was
linked. Also prints how many combined nodes got a bio.

Run:
    PYTHONPATH=. python -m src.bio.s5_fold_combined
"""
from __future__ import annotations

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# book -> combined notes tag
BOOK_TAG = {"book1": "b1", "book2": "b2", "book3": "b3", "book4": "b4"}


def _prov(node: dict) -> str:
    return (node.get("notes", "") or "").split(" | ", 1)[0].strip()


def build_bio_index(data_dir: str, books: list[str]) -> dict[str, dict]:
    """{ '{tag}:{provenance}' -> bio } from each book's *_bio_linked.jsonl."""
    idx: dict[str, dict] = {}
    for book in books:
        path = os.path.join(data_dir, f"{book}_bio_linked.jsonl")
        if not os.path.exists(path):
            continue
        tag = BOOK_TAG[book]
        for line in open(path):
            n = json.loads(line)
            if "bio" in n:
                idx[f"{tag}:{_prov(n)}"] = n["bio"]
    return idx


def fold(data_dir: str = "data", books=("book3", "book4"),
         combined="tree.jsonl", out="tree_bio.jsonl") -> int:
    idx = build_bio_index(data_dir, list(books))
    src = os.path.join(data_dir, combined)
    dst = os.path.join(data_dir, out)
    n_attached = 0
    with open(dst, "w") as fh:
        for line in open(src):
            node = json.loads(line)
            key = _prov(node)  # already '{tag}:{prov}' in combined notes
            if key in idx:
                node["bio"] = idx[key]
                n_attached += 1
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")
    logger.info("folded %d bios onto %s -> %s (index had %d)", n_attached, combined, out, len(idx))
    return n_attached


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--books", nargs="+", default=["book3", "book4"])
    p.add_argument("--combined", default="tree.jsonl")
    p.add_argument("--out", default="tree_bio.jsonl")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fold(a.data_dir, a.books, a.combined, a.out)


if __name__ == "__main__":
    main()
