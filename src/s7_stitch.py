"""Stage 7 -- cross-graph stitching: connect the per-graph subtrees into one tree.

Stage 5 (:mod:`src.s5_build_tree`) parses each Stage-4 graph independently, so its
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
matcher will later plug in -- now that Stage 6 (OCR) fills in names before this
stage, that matcher can key on names, not just crop pixels. The result is verified
against the ``book1_merged`` oracle (``data/oracles/book1_merged.jsonl``); see
``scripts/compare_book1.py``.

Nodes are addressed by their **provenance** -- ``"{graph}_{local_index}"`` (e.g.
``"8_8_13"``), the first ``" | "``-separated segment of the ``notes`` field written
by Stage 5 -- rather than by numeric id, because ids are reassigned here but
provenance is stable across the renumber. (Stage 6 appends ``ocr_*`` tags to
``notes`` after the provenance; :func:`_provenance` strips them.)

CLI::

    python -m src.s7_stitch --book book1
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
# Manual per-book merge OVERRIDES (rarely needed).
#
# Each entry is (duplicate_root_provenance, canonical_leaf_provenance). Normally
# :func:`find_merges` derives these automatically by name-matching each duplicate
# root to its earlier same-name leaf (works cleanly for Book 1: 13/13 unique).
# An entry here overrides the matcher for a whole book -- use it only when names
# genuinely can't resolve a book's seams. Empty by default.
#
# (Historical note: Book 1 previously carried a hand-authored list here, but it was
# keyed to the v0 parse's provenances; v1 renumbered graph-8's leaves, so 7 of 13
# pairs pointed at the wrong person. The name-matcher fixes this and self-adapts to
# the parse, so the hardcoded list was removed. See docs/history.md.)
# ---------------------------------------------------------------------------
BOOK_MERGES: dict[str, list[tuple[str, str]]] = {}


def _provenance(notes: str) -> str:
    """The stable ``{graph}_{local_index}`` provenance = first ' | ' segment.

    Stage 5 writes ``notes`` as the bare provenance; Stage 6 appends ``ocr_*``
    tags after a ``" | "`` separator. Either way the provenance is the head.
    """
    return (notes or "").split(" | ", 1)[0]


def _graph_num(prov: str) -> tuple[int, ...]:
    """The numeric graph key of a provenance ``{start}_{end}_{i}`` -> (start, end).

    Graphs are ordered by their starting page; this sorts/compares provenances by
    which graph they belong to (ignoring the within-graph index).
    """
    parts = prov.split("_")
    return tuple(int(p) for p in parts[:-1]) if len(parts) >= 3 else (0,)


def find_merges(nodes: list[Node], book: str) -> list[tuple[str, str]]:
    """Match each duplicate graph-root to its canonical leaf, BY NAME.

    A subtree-start page reprints its parent's name, so every graph's root (except
    the first) duplicates a *leaf* of an earlier graph -- the SAME person, hence the
    same name. Now that Stage 6 (OCR) fills in names, this pairs each duplicate root
    with the earlier-graph leaf that has the same name. When several earlier leaves
    share the name, the nearest (latest) earlier graph wins.

    This supersedes the old hardcoded per-book provenance list, which was authored
    against the v0 parse and mis-paired 7 of Book 1's 13 seams once v1 renumbered
    graph-8's leaves (same provenance string, different person). A manual override
    in :data:`BOOK_MERGES` still wins when present, for any case names can't resolve.

    Args:
        nodes: The Stage-5 domain nodes (a forest), names filled in by Stage 6.
        book: Book name, e.g. ``"book1"``.

    Returns:
        ``(duplicate_root_provenance, canonical_leaf_provenance)`` pairs.
    """
    if book in BOOK_MERGES:
        logger.info("using manual BOOK_MERGES override for %s", book)
        return BOOK_MERGES[book]

    prov_of = {n.id: _provenance(n.notes) for n in nodes}
    # Earlier-graph candidates indexed by name: leaves (the normal seam -- a
    # subtree-start page repeats a person whose OWN subtree was cut off there) and
    # non-leaves (a section that re-prints an ancestor who already has children in
    # an earlier graph, e.g. Book 2's 4_5 repeating 0_3's root 存学). Leaves win
    # when both exist; non-leaves are the fallback.
    leaves_by_name: dict[str, list[str]] = {}
    inner_by_name: dict[str, list[str]] = {}
    for n in nodes:
        if n.name:
            index = leaves_by_name if not n.children else inner_by_name
            index.setdefault(n.name, []).append(prov_of[n.id])

    # Duplicate roots = each graph's own root, ``{graph}_0`` -- the person the
    # subtree-start page reprints -- except the forest's first graph. A graph can
    # hold OTHER roots too: subtrees severed at a page seam (Stage 5's empty
    # phantom bars and the branches under them, provenance index != 0). Those are
    # not duplicates of anyone; name-matching them would weld a severed branch
    # onto whoever shares its OCR'd name (Book 2's 114_120_44 毓棋 -> 101_105).
    # They stay roots until the seam itself is repaired (bridging).
    roots = sorted(
        (n for n in nodes if n.father == -1),
        key=lambda n: _graph_num(prov_of[n.id]),
    )
    section_roots = [r for r in roots if prov_of[r.id].endswith("_0")]
    orphan_roots = [r for r in roots if not prov_of[r.id].endswith("_0")]
    if orphan_roots:
        logger.warning(
            "%s: %d root(s) are not section roots (seam orphans), left as roots: %s",
            book, len(orphan_roots),
            ", ".join(f"{prov_of[r.id]} ({r.name!r})" for r in orphan_roots))

    merges: list[tuple[str, str]] = []
    unresolved: list[str] = []
    for r in section_roots[1:]:
        rp = prov_of[r.id]
        rg = _graph_num(rp)
        cands = [p for p in leaves_by_name.get(r.name, []) if _graph_num(p) < rg]
        if not cands:
            cands = [p for p in inner_by_name.get(r.name, []) if _graph_num(p) < rg]
        if not cands:
            unresolved.append(f"{rp} ({r.name!r})")
            continue
        # Nearest earlier graph wins if a name recurs.
        canon = max(cands, key=_graph_num)
        merges.append((rp, canon))

    if unresolved:
        logger.warning(
            "%s: %d duplicate root(s) had no earlier same-name node, left "
            "unmerged: %s", book, len(unresolved), ", ".join(unresolved))
    logger.info("name-matched %d/%d merges for %s",
                len(merges), len(section_roots) - 1, book)
    return merges


def stitch_nodes(nodes: list[Node], merges: list[tuple[str, str]]) -> list[Node]:
    """Apply the merges and return a fresh, connected, renumbered node list.

    For each ``(duplicate_root, canonical_leaf)`` pair the canonical leaf absorbs
    the duplicate root: it adopts the root's children (which are re-parented to
    it) and the duplicate is removed. Generations are then recomputed absolutely
    from the tree's single root, and ids are reassigned BFS-by-generation,
    right-to-left (eldest-first), preserving the project convention.

    Args:
        nodes: Stage-5 domain nodes (a forest).
        merges: ``(duplicate_root_provenance, canonical_leaf_provenance)`` pairs.

    Returns:
        A new list of :class:`Node` forming one connected tree with absolute
        generations and reassigned ids.
    """
    by_prov = {_provenance(n.notes): n for n in nodes}
    by_id = {n.id: n for n in nodes}
    prov_of = {n.id: _provenance(n.notes) for n in nodes}

    dropped_ids: set[int] = set()
    # Where a dropped node's identity moved to. When a node is folded away it may
    # later be named as another merge's canonical (a re-printed ancestor chain
    # spanning >2 graphs: A<-B and B<-C, so B is a target then a source). Following
    # this map to the surviving node keeps such a chain from attaching children to a
    # dead id (which BFS would later hit, absent from the renumber map).
    folded_into: dict[int, int] = {}

    def _survivor(node: Node) -> Node:
        seen: set[int] = set()
        while node.id in folded_into and node.id not in seen:
            seen.add(node.id)
            node = by_id[folded_into[node.id]]
        return node

    def fold(canon: Node, dup: Node) -> None:
        """``canon`` absorbs ``dup`` (the same person printed twice).

        The canonical node adopts the duplicate's children -- except a child who
        is ALSO already a child of the canonical node by name: a section that
        re-prints a shared ancestor chain (Book 2's 8_10 and 67_68 both open
        克宣 -> 龙润 before diverging) repeats those people too, so such a pair is
        folded recursively instead of duplicated. A fold needs an unambiguous
        name: exactly one child on each side carries it, and the name is at least
        two characters (in a 2-char-name book a lone char is an OCR truncation,
        not an identity; Book 1's 1-char names merge into leaves, so this never
        applies there). Anything ambiguous is adopted as-is -- a visible duplicate
        beats a silent wrong weld.
        """
        canon_names = collections.Counter(by_id[c].name for c in canon.children)
        dup_names = collections.Counter(by_id[c].name for c in dup.children)
        for child_id in list(dup.children):
            child = by_id[child_id]
            twin = None
            if len(child.name) >= 2 and canon_names[child.name] == 1 and dup_names[child.name] == 1:
                twin = next(by_id[c] for c in canon.children if by_id[c].name == child.name)
            if twin is not None:
                logger.info("folding repeated chain: %s %r into %s",
                            prov_of[child.id], child.name, prov_of[twin.id])
                fold(twin, child)
            else:
                canon.children = list(canon.children) + [child_id]
                child.father = canon.id
        # Record the merge in notes while KEEPING the canonical node's OCR tags:
        # prepend the "canonical/duplicate" provenance trace, keep the rest.
        _, _, canon_tags = (canon.notes or "").partition(" | ")
        trace = f"{prov_of[canon.id]}/{prov_of[dup.id]}"
        canon.notes = f"{trace} | {canon_tags}" if canon_tags else trace
        dropped_ids.add(dup.id)
        folded_into[dup.id] = canon.id

    # 1. Merge: fold each duplicate root into its canonical node.
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
        # If this canonical was itself folded away by an earlier merge (a chain
        # A<-B, B<-C), redirect to the node it survives as, so children never
        # attach to a dropped id.
        canon = _survivor(canon)
        if canon.id == dup.id:
            continue  # a chain that folds a node into itself -- nothing to do
        # A valid seam joins the SAME person, so the two nodes' names must agree
        # (a subtree-start page reprints the parent's name). A mismatch means the
        # merge is wrong -- the failure mode that silently mis-connected 7 of Book
        # 1's seams under the old provenance-keyed list. Warn loudly (don't hard-
        # fail: OCR noise could differ a glyph, and blank names can't be checked).
        if dup.name and canon.name and dup.name != canon.name:
            logger.warning(
                "seam name MISMATCH: dup %s %r != canon %s %r -- likely wrong merge",
                dup_prov, dup.name, canon_prov, canon.name)
        fold(canon, dup)

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
        data_dir: Directory holding the Stage-5 output and the stitched output.

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
