"""Stage 3.5 -- cross-graph stitching: connect the per-graph subtrees into one tree.

Stage 3 (:mod:`src.build_tree`) parses each Stage-2 graph independently, so its
output is a *forest*: one subtree per graph, each re-rooted at generation 1. But
a subtree-start page repeats its parent's name at the top to show where the
subtree attaches -- so every graph's root node (except the very first) is a
**duplicate** of a person who already appears as a *leaf* in an earlier graph.

Stitching merges each duplicate root into its canonical leaf: the leaf adopts the
root's children, the duplicate id is dropped, and the whole forest collapses into
one connected lineage. Generations then become absolute (root = 1, counting down
the real tree) and ids are reassigned BFS-by-generation, right-to-left.

**Matching is the hard part.** Identifying which earlier leaf a duplicate root is
the same person as is, in general, a name-recognition problem -- pure name-crop
pixel-matching is unreliable (it recovers only ~4 of Book 1's 13 merges). For now
the merge list is supplied explicitly per book (Book 1's was determined by hand;
see :data:`BOOK_MERGES`). :func:`find_merges` is the seam where an automated
matcher will later plug in. The result is verified against the ``book1_merged``
oracle (``data/oracles/book1_merged.jsonl``) -- see ``scripts/compare_book1.py``.

Nodes are addressed by their **provenance** string (the ``notes`` field written by
Stage 3, ``"{graph}_{local_index}"``, e.g. ``"8_8_13"``) rather than by numeric id,
because ids are reassigned here but provenance is stable across the renumber.

CLI::

    python -m src.stitch --book book1
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os

from src.model import Node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-book merge lists.
#
# Each entry is (duplicate_root_provenance, canonical_leaf_provenance): the
# graph-root that is a duplicate, and the earlier-graph leaf it is the same
# person as. Book 1's 13 merges were determined by hand (matching the archived
# `data/oracles/book1_merged.jsonl`). A future `find_merges` implementation will
# derive these automatically; until then, unknown books produce no merges (the
# forest passes through unchanged) and log a warning.
# ---------------------------------------------------------------------------
BOOK_MERGES: dict[str, list[tuple[str, str]]] = {
    "book1": [
        ("1_1_0", "0_0_7"),
        ("2_2_0", "1_1_6"),
        ("3_3_0", "2_2_7"),
        ("4_4_0", "3_3_13"),
        ("5_5_0", "4_4_7"),
        ("6_6_0", "5_5_6"),
        ("7_7_0", "6_6_7"),
        ("8_8_0", "7_7_8"),
        ("9_9_0", "8_8_18"),
        ("10_10_0", "8_8_17"),
        ("11_11_0", "8_8_14"),
        ("12_12_0", "8_8_13"),
        ("13_16_0", "12_12_7"),
    ],
}


def find_merges(nodes: list[Node], book: str) -> list[tuple[str, str]]:
    """Return the (duplicate_root, canonical_leaf) provenance pairs to merge.

    Currently a lookup into :data:`BOOK_MERGES`. This is the extension point for
    an automated matcher (name-crop similarity + reading-order + grid position);
    swapping the body here is all that's needed to make stitching general.

    Args:
        nodes: The Stage-3 domain nodes (used by a future automated matcher;
            unused by the hardcoded lookup).
        book: Book name, e.g. ``"book1"``.

    Returns:
        List of ``(duplicate_root_provenance, canonical_leaf_provenance)`` pairs.
    """
    merges = BOOK_MERGES.get(book)
    if merges is None:
        logger.warning(
            "no merge list for %s; stitching will pass the forest through "
            "unchanged. Add an entry to BOOK_MERGES or implement automated "
            "matching.",
            book,
        )
        return []
    return merges


def stitch_nodes(nodes: list[Node], merges: list[tuple[str, str]]) -> list[Node]:
    """Apply the merges and return a fresh, connected, renumbered node list.

    For each ``(duplicate_root, canonical_leaf)`` pair the canonical leaf absorbs
    the duplicate root: it adopts the root's children (which are re-parented to
    it) and the duplicate is removed. Generations are then recomputed absolutely
    from the tree's single root, and ids are reassigned BFS-by-generation,
    right-to-left (eldest-first), preserving the project convention.

    Args:
        nodes: Stage-3 domain nodes (a forest).
        merges: ``(duplicate_root_provenance, canonical_leaf_provenance)`` pairs.

    Returns:
        A new list of :class:`Node` forming one connected tree with absolute
        generations and reassigned ids.
    """
    by_prov = {n.notes: n for n in nodes}
    by_id = {n.id: n for n in nodes}

    # 1. Merge: fold each duplicate root into its canonical leaf.
    dropped_ids: set[int] = set()
    for dup_prov, canon_prov in merges:
        dup = by_prov.get(dup_prov)
        canon = by_prov.get(canon_prov)
        if dup is None or canon is None:
            raise KeyError(
                f"merge references unknown provenance: dup={dup_prov!r} "
                f"canon={canon_prov!r}"
            )
        if dup.father != -1:
            raise ValueError(
                f"duplicate {dup_prov!r} (id {dup.id}) is not a root "
                f"(father={dup.father}); merge list is inconsistent"
            )
        # The canonical leaf adopts the duplicate's children.
        canon.children = list(canon.children) + list(dup.children)
        for child_id in dup.children:
            by_id[child_id].father = canon.id
        # Record both provenances so the merge stays traceable (matches the old
        # book1_merged notes convention "canonical/duplicate").
        canon.notes = f"{canon_prov}/{dup_prov}"
        dropped_ids.add(dup.id)

    survivors = [n for n in nodes if n.id not in dropped_ids]

    # 2. Recompute absolute generations from the single root downward.
    children_of = {n.id: list(n.children) for n in survivors}
    roots = [n for n in survivors if n.father == -1]
    if len(roots) != 1:
        logger.warning(
            "expected exactly 1 root after stitching, got %d: %s",
            len(roots),
            [r.id for r in roots],
        )
    generation_of: dict[int, int] = {}
    queue = collections.deque((r.id, 1) for r in roots)
    while queue:
        nid, gen = queue.popleft()
        generation_of[nid] = gen
        for cid in children_of.get(nid, []):
            queue.append((cid, gen + 1))
    for n in survivors:
        n.generation = generation_of.get(n.id, -1)

    # 3. Reassign ids BFS-by-generation, right-to-left (eldest-first). survivors'
    # children lists are already eldest-first, so a BFS that enqueues children in
    # order yields the canonical id ordering.
    old_to_new: dict[int, int] = {}
    next_id = 1
    bfs = collections.deque(sorted((r.id for r in roots)))
    while bfs:
        old_id = bfs.popleft()
        old_to_new[old_id] = next_id
        next_id += 1
        for cid in children_of.get(old_id, []):
            bfs.append(cid)

    result: list[Node] = []
    for n in survivors:
        result.append(
            Node(
                id=old_to_new[n.id],
                name=n.name,
                name_images=n.name_images,
                generation=n.generation,
                father=old_to_new[n.father] if n.father in old_to_new else -1,
                children=[old_to_new[c] for c in n.children],
                biography=n.biography,
                notes=n.notes,
            )
        )
    result.sort(key=lambda n: n.id)
    return result


def stitch(book: str, data_dir: str = "data") -> list[Node]:
    """Read ``{book}.jsonl``, stitch it, and write ``{book}_stitched.jsonl``.

    Args:
        book: Book name, e.g. ``"book1"``.
        data_dir: Directory holding the Stage-3 output and the stitched output.

    Returns:
        The stitched nodes.
    """
    in_path = os.path.join(data_dir, f"{book}.jsonl")
    out_path = os.path.join(data_dir, f"{book}_stitched.jsonl")

    with open(in_path) as f:
        nodes = [Node(**json.loads(line)) for line in f if line.strip()]
    logger.info("Read %d nodes from %s", len(nodes), in_path)

    merges = find_merges(nodes, book)
    logger.info("Applying %d merges", len(merges))
    stitched = stitch_nodes(nodes, merges)

    roots = sum(1 for n in stitched if n.father == -1)
    max_gen = max((n.generation for n in stitched), default=-1)
    logger.info(
        "Stitched -> %d nodes, %d root(s), max generation %d",
        len(stitched),
        roots,
        max_gen,
    )

    with open(out_path, "w") as f:
        for n in stitched:
            f.write(json.dumps(dataclasses.asdict(n), ensure_ascii=False) + "\n")
    logger.info("Wrote %s", out_path)
    return stitched


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch per-graph subtrees into one tree.")
    parser.add_argument("--book", default="book1", help="Book name, e.g. book1")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stitch(args.book, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
