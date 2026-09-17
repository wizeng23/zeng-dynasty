"""Stage 8 -- cross-book stitching: fold the per-book trees into one lineage.

Each book covers a generation band; a later book re-prints, as the root of some
subtrees, a person who is a *leaf* in the previous book (the book boundary falls
mid-lineage). :mod:`src.s7_stitch` already collapsed each book's own forest into a
tree (``{book}_stitched.jsonl``); this stage joins those trees end to end.

The pairings -- which later-book root is the same person as which earlier-book
leaf -- are hand-editable ground truth seeded by
:mod:`scripts.qa.cross_book_pairs` and stored in each *child* book's folder as
``books/{child}/stitch_pairs.json``. The survivor is ALWAYS the earlier book's
node (it is nearer the root 点); the later duplicate is folded into it, adopting
its children.

We reuse :func:`src.s7_stitch.stitch_nodes` for the actual fold + renumber -- the
cross-book merge is the same operation as a within-book seam, just spanning two
books' node lists. To make provenances unique across books they are namespaced
``b1:``/``b2:``/``b3:``/``b4:`` before folding (bare ``{graph}_{index}`` strings
collide between books). Afterwards generations are shifted so the root 点 is
**generation 0** (matching the 字辈 naming, where 点 is the 0th 世).

Output is a fresh ``data/tree.jsonl`` -- a full BFS-by-generation, RTL renumber
with contiguous ids. ``data/tree_combined.jsonl`` (the old one-off splice, which
still carries hand overrides) is left untouched.

The book order is fixed (``BOOKS``) and the stage runs over whatever prefix has
both a ``{book}_stitched.jsonl`` and (for every book after the first) a pairs
file, so adding Book 4 is just dropping in its stitched jsonl + pairs file.

CLI::

    python -m src.s8_cross_stitch                    # books 1..N found on disk
    python -m src.s8_cross_stitch --books book1 book2 book3
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os

from src.model import Node
from src.s7_stitch import stitch_nodes

logger = logging.getLogger(__name__)

BOOKS = ["book1", "book2", "book3", "book4"]
DATA_DIR = "data"
BOOKS_DIR = "books"


def _tag(book: str) -> str:
    """The provenance namespace prefix for a book, e.g. ``book2`` -> ``b2:``."""
    return f"b{book[-1]}:"


def _renamespace(node: Node, book: str) -> None:
    """Prefix the node's provenance (in ``notes``) with its book tag, in place.

    ``notes`` is ``{provenance} | {ocr_tags}``; only the provenance head is tagged,
    so ``_provenance`` in s7 still recovers ``b2:0_3_0`` as the addressable key while
    the OCR tags ride along unchanged.
    """
    notes = node.notes or ""
    prov, sep, tags = notes.partition(" | ")
    node.notes = f"{_tag(book)}{prov}{sep}{tags}"


def load_book(book: str) -> list[Node]:
    """Load ``{book}_stitched.jsonl`` and namespace every provenance by book."""
    path = os.path.join(DATA_DIR, f"{book}_stitched.jsonl")
    nodes = [Node(**json.loads(line)) for line in open(path) if line.strip()]
    for n in nodes:
        _renamespace(n, book)
    return nodes


def load_pairs(child_book: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Read a child book's ``stitch_pairs.json`` -> (merges, unconnected).

    ``merges`` are ``(duplicate_later_prov, canonical_earlier_prov)`` -- the argument
    order :func:`stitch_nodes` expects (it folds the *first*, a root, into the
    *second*). The pairs file stores them ``[earlier_survivor, later_duplicate]``, so
    we swap: the later book's root is the duplicate, the earlier book's leaf is
    canonical.
    """
    path = os.path.join(BOOKS_DIR, child_book, "stitch_pairs.json")
    if not os.path.exists(path):
        logger.warning("no pairs file for %s: %s", child_book, path)
        return [], []
    data = json.load(open(path))
    merges = [(later, earlier) for earlier, later in data.get("pairs", [])]
    return merges, data.get("unconnected", [])


def _offset_ids(nodes: list[Node], offset: int) -> None:
    """Shift every id/father/children by ``offset`` so two books' ids don't collide."""
    for n in nodes:
        n.id += offset
        if n.father != -1:
            n.father += offset
        n.children = [c + offset for c in n.children]


def _rebase_floaters(nodes: list[Node]) -> None:
    """Set each disconnected floater's generation from its root's 字辈, in place.

    A floater is a root other than the true root 点 -- a subtree whose exact parent
    isn't paired yet. Its root's name still starts with a generation-name character
    (字辈), and every generation-name sits at exactly one absolute generation in the
    main tree. So we map the floater root's first character to that generation and
    shift the floater's WHOLE subtree to match (root -> that gen, its children one
    below, and so on), so it renders at the right depth on the website even while
    disconnected. A root whose 字辈 isn't found in the main tree is left as-is (and
    warned), since we can't place it.
    """
    by_id = {n.id: n for n in nodes}
    children_of = {n.id: list(n.children) for n in nodes}

    # The main tree = the component under the true root 点 (the largest root's tree,
    # identified as the one containing 点; fall back to lowest-id root).
    roots = [n for n in nodes if n.father == -1]
    main_root = next((r for r in roots if r.name == "点"), min(roots, key=lambda n: n.id))

    def component(root_id: int) -> set[int]:
        seen: set[int] = set()
        stack = [root_id]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(children_of.get(x, []))
        return seen

    main = component(main_root.id)
    # 字辈 char -> the absolute generation its cohort sits at in the main tree. A
    # generation-name can appear at a stray off-generation too (an OCR homograph, or
    # a genuinely different person sharing the char), so we take the MODAL generation
    # -- the one the overwhelming majority share -- not a unique value.
    char_gen_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    for nid in main:
        n = by_id[nid]
        if n.name:
            char_gen_counts[n.name[0]][n.generation] += 1
    char_gen = {c: counts.most_common(1)[0][0] for c, counts in char_gen_counts.items()}

    for root in roots:
        if root.id == main_root.id:
            continue
        c = root.name[0] if root.name else ""
        target = char_gen.get(c)
        if target is None:
            logger.warning(
                "floater root id=%d name=%r: 字辈 %r not found in main tree; "
                "leaving generations as-is", root.id, root.name, c)
            continue
        shift = target - root.generation
        if shift:
            for nid in component(root.id):
                by_id[nid].generation += shift
        logger.info("floater id=%d name=%r placed at gen %d (字辈 %r, shift %+d)",
                    root.id, root.name, target, c, shift)


def cross_stitch(books: list[str]) -> list[Node]:
    """Fold ``books`` (in order) into one tree; return the renumbered node list.

    Combines each book's stitched nodes (ids offset so they stay unique), gathers
    every child book's pairings, and hands the whole lot to
    :func:`stitch_nodes`, which folds the duplicates, recomputes generations from
    the single root, and reassigns BFS/RTL ids. Then shifts generations so the root
    is generation 0.
    """
    combined: list[Node] = []
    merges: list[tuple[str, str]] = []
    all_unconnected: list[str] = []

    offset = 0
    for book in books:
        nodes = load_book(book)
        _offset_ids(nodes, offset)
        combined.extend(nodes)
        offset += 100000  # each book gets its own 100k id band; well clear of counts
        if book != books[0]:
            book_merges, unconnected = load_pairs(book)
            merges.extend(book_merges)
            all_unconnected.extend(f"{book}: {u}" for u in unconnected)

    logger.info(
        "combined %d nodes from %s; applying %d cross-book merges",
        len(combined), ",".join(books), len(merges),
    )
    stitched = stitch_nodes(combined, merges)

    # stitch_nodes roots generations at 1; the 字辈 convention makes 点 the 0th 世.
    for n in stitched:
        if n.generation > 0:
            n.generation -= 1

    _rebase_floaters(stitched)

    roots = [n for n in stitched if n.father == -1]
    logger.info(
        "cross-stitched -> %d nodes, %d root(s) (gen 0..%d)",
        len(stitched), len(roots),
        max((n.generation for n in stitched), default=-1),
    )
    if all_unconnected:
        logger.warning(
            "%d later-book root(s) left unconnected (need manual pairing):\n  %s",
            len(all_unconnected), "\n  ".join(all_unconnected),
        )
    for r in sorted(roots, key=lambda n: n.id):
        logger.info("  root: id=%d name=%r gen=%d notes=%s",
                    r.id, r.name, r.generation, (r.notes or "").split(" | ")[0])
    return stitched


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Stitch per-book trees into one lineage.")
    ap.add_argument("--books", nargs="+", default=None,
                    help="book order, e.g. book1 book2 book3 (default: all present)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "tree.jsonl"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s %(name)s: %(message)s")

    books = args.books or [
        b for b in BOOKS if os.path.exists(os.path.join(DATA_DIR, f"{b}_stitched.jsonl"))
    ]
    logger.info("stitching books: %s", books)
    stitched = cross_stitch(books)

    with open(args.out, "w") as f:
        for n in stitched:
            f.write(json.dumps(dataclasses.asdict(n), ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d nodes)", args.out, len(stitched))


if __name__ == "__main__":
    main()
