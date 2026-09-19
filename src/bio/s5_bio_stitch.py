"""Bio stage 5 (stitch): fold a Book-4 subgraph root into its Book-3 leaf when the
two are the SAME person, identified by an exact son-list match.

This mirrors the graph stitcher (:mod:`src.s7_stitch`): there, a subtree-start page
reprints its parent's name, so each graph root duplicates an earlier leaf and the two
are folded into one node. Here the duplicate spans BOOKS. A person whose lineage runs
off the end of Book 3 reappears at the head of a Book-4 subgraph -- the Book-4 root --
and Book 4 draws that person's children as a graph while Book 3 only *lists* them (in
his biography's son field). So the seam is:

  * a **Book-4 floater root** ``R`` (a root other than the true root 点), whose GRAPH
    children are real B4 nodes in age order (RTL / eldest-first); and
  * a **Book-3 node** ``L`` carrying a linked bio, whose BIO son-list names those same
    children.

``R`` and ``L`` are the same person iff they share a name AND ``L``'s bio son-list
equals ``R``'s graph-children names **exactly, element-wise, in age order**. This is a
deliberately strict, unforgeable signal: a multi-son ordered list matching by chance is
vanishingly unlikely, so no ``exactly-one-candidate`` disambiguation contortions are
needed the way loose per-son matching required. OCR noise that breaks the exact match
leaves the pair unstitched (reported), to be recovered after a human OCR-QA pass -- a
missed merge is a visible floater; a wrong merge silently welds two lineages.

Fold direction: **L (Book 3) is canonical**; R (Book 4) folds into it. L keeps its id
and tree position (it is the parent-generation placement); R's subtree re-parents
beneath L and R's id is dropped. This is additive -- names are never edited, and a pair
is folded only when the son lists match exactly.

Input:  data/tree.jsonl (combined cross-book forest) +
        data/{book}_bio_linked.jsonl (bios keyed by provenance, from s5_link).
Output: data/tree_bio_stitched.jsonl (tree.jsonl with B4 roots folded into B3 leaves) +
        data/bio_stitch_report.json (every merge + every unstitched B4 root, with evidence).

Run:
    PYTHONPATH=. python -m src.bio.s5_bio_stitch            # write outputs
    PYTHONPATH=. python -m src.bio.s5_bio_stitch --dry-run  # report only
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os

logger = logging.getLogger(__name__)

BOOK_TAG = {"b1": "book1", "b2": "book2", "b3": "book3", "b4": "book4"}
TRUE_ROOT_NAME = "点"


def _prov(notes: str) -> str:
    return (notes or "").split(" | ", 1)[0].strip()


def load_combined(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def bio_index(data_dir: str, books: list[str]) -> dict[str, dict]:
    """{'{tag}:{prov}' -> bio} from each book's *_bio_linked.jsonl (same key as tree notes)."""
    tag_of = {v: k for k, v in BOOK_TAG.items()}
    idx = {}
    for book in books:
        p = os.path.join(data_dir, f"{book}_bio_linked.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            n = json.loads(line)
            if "bio" in n:
                idx[f"{tag_of[book]}:{_prov(n['notes'])}"] = n["bio"]
    return idx


def bio_sons(bio: dict) -> tuple[list[str], list[str]]:
    """The bio's son names as (vision, paddle) lists, each in printed (age) order.

    Both readers are kept: OCR noise may leave one reader's list matching the graph
    exactly while the other's is off by a glyph.
    """
    if not bio:
        return [], []
    sons = bio.get("sons", {}) or {}
    vision = [x for x in (sons.get("vision") or []) if x]
    paddle = [x for x in (sons.get("paddle") or []) if x]
    return vision, paddle


def find_merges(tree: list[dict], data_dir: str, books: list[str]):
    """Match each Book-4 floater root to the Book-3 node that is the same person.

    Returns ``(merges, unstitched)`` where each merge is a dict recording the folded
    pair with evidence, and ``unstitched`` lists the B4 roots that found no exact match.
    """
    by_id = {n["id"]: n for n in tree}
    bio = bio_index(data_dir, books)

    # Book-3 candidates: nodes with a linked bio, indexed by name. A person may be
    # named by several nodes, so a name maps to a LIST; the exact son-list match then
    # disambiguates. (list child names in age order = the node's ``children`` order.)
    # The candidate pool is restricted to Book 3: the fold direction is B4-root ->
    # B3-canonical, and ``bio_index`` also loads Book-4 bios, so without this filter a
    # B4 root could fold into another B4 node (off-spec, and a silent wrong weld).
    b3_by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for n in tree:
        if not _prov(n["notes"]).startswith("b3:"):
            continue
        b = bio.get(_prov(n["notes"]))
        if b is not None:
            b3_by_name[n["name"]].append(n)

    # Book-4 floater roots: roots (father == -1) other than the true root 点, that
    # actually have graph children (a childless root has no son list to match).
    roots = [n for n in tree if n["father"] == -1]
    b4_roots = [
        n for n in roots
        if n["name"] != TRUE_ROOT_NAME and n["children"]
    ]

    merges: list[dict] = []
    unstitched: list[dict] = []
    for r in b4_roots:
        kids = [by_id[c]["name"] for c in r["children"] if c in by_id]
        if not kids:  # all children missing from the tree: no son list to match on --
            continue  # matching [] would false-weld any bio with an empty son list.
        cands = []
        for l in b3_by_name.get(r["name"], []):
            vision, paddle = bio_sons(bio[_prov(l["notes"])])
            how = "vision" if vision == kids else "paddle" if paddle == kids else None
            if how:
                cands.append((l, how))
        # Strict: an exact ordered son-list match is unforgeable, so a single hit is
        # decisive. More than one B3 node exact-matching the same list would mean the
        # same person is linked twice -- ambiguous, so skip and report rather than
        # guess which placement is canonical.
        if len(cands) == 1:
            l, how = cands[0]
            merges.append({
                "child_root_prov": _prov(r["notes"]),      # B4 duplicate (folded away)
                "child_root_id": r["id"],
                "canonical_prov": _prov(l["notes"]),        # B3 canonical (survives)
                "canonical_id": l["id"],
                "name": r["name"],
                "sons": kids,
                "matched_reader": how,
            })
        else:
            unstitched.append({
                "root_prov": _prov(r["notes"]), "root_id": r["id"],
                "name": r["name"], "graph_children": kids,
                "candidate_count": len(cands),
            })
    return merges, unstitched


def apply_merges(tree: list[dict], merges: list[dict]) -> list[dict]:
    """Fold each B4 root into its B3 canonical, then recompute absolute generations.

    The canonical (B3) node absorbs the duplicate root's children -- each is re-parented
    to the canonical -- and the duplicate is dropped. Ids are NOT renumbered: the combined
    tree is keyed by ``notes`` provenance downstream (fold + website), and that key must
    stay stable. Generations are recomputed from the true root so the folded B4 subtree
    gets absolute depths.

    The seam is recorded as a ``bio_stitch=canon/dup`` TAG appended after the notes'
    existing ``" | "``-separated tags -- NOT by rewriting the provenance head the way
    :func:`src.s7_stitch.stitch_nodes` does. The head must stay the bare ``{tag}:{prov}``
    because ``s5_fold_combined`` re-keys bios by it afterwards; a rewritten head would drop
    the bio from every merged canonical (exactly the son-listing parent that drove the
    merge). ``s7_stitch`` can rewrite its head because nothing re-keys by it downstream.
    """
    by_id = {n["id"]: n for n in tree}
    dropped: set[int] = set()

    # Canonical (B3) and duplicate (B4-root) id-sets are disjoint by construction (a B4
    # floater root can't also be a B3 bio-carrier). That disjointness is why no
    # chain-following (s7_stitch's ``folded_into``) is needed: a canonical is never itself
    # folded away, so children never attach to a dropped id. Assert it, so a future data
    # shape that broke the assumption fails loudly instead of silently corrupting the tree.
    canon_ids = {m["canonical_id"] for m in merges}
    dup_ids = {m["child_root_id"] for m in merges}
    overlap = canon_ids & dup_ids
    assert not overlap, f"canonical/duplicate id-sets overlap (fold chains): {overlap}"

    for m in merges:
        dup = by_id[m["child_root_id"]]
        canon = by_id[m["canonical_id"]]
        if dup["father"] != -1:
            raise ValueError(
                f"B4 root {m['child_root_prov']} (id {dup['id']}) is not a root "
                f"(father={dup['father']}); merge list is inconsistent")
        # canon adopts dup's children (dup's kids ARE canon's bio-listed sons).
        for cid in list(dup["children"]):
            if cid not in canon["children"]:
                canon["children"].append(cid)
            by_id[cid]["father"] = canon["id"]
        # record the seam as a tag, preserving the provenance head (see docstring).
        trace = f"bio_stitch={m['canonical_prov']}/{m['child_root_prov']}"
        canon["notes"] = f"{(canon.get('notes') or '').rstrip()} | {trace}"
        dropped.add(dup["id"])

    survivors = [n for n in tree if n["id"] not in dropped]

    # Recompute absolute generations from the single true root downward.
    children_of = {n["id"]: list(n["children"]) for n in survivors}
    roots = [n for n in survivors if n["father"] == -1]
    if not any(r["name"] == TRUE_ROOT_NAME for r in roots):
        logger.warning("no true root %r found among %d roots after stitch",
                       TRUE_ROOT_NAME, len(roots))
    gen_of: dict[int, int] = {}
    queue = collections.deque((r["id"], r["generation"]) for r in roots)
    while queue:
        nid, gen = queue.popleft()
        gen_of[nid] = gen
        for cid in children_of.get(nid, []):
            queue.append((cid, gen + 1))
    for n in survivors:
        if n["id"] in gen_of:
            n["generation"] = gen_of[n["id"]]
    return survivors


def bio_stitch(data_dir: str = "data", combined: str = "tree.jsonl",
               books=("book3", "book4"), dry_run: bool = False) -> dict:
    tree = load_combined(os.path.join(data_dir, combined))
    merges, unstitched = find_merges(tree, data_dir, list(books))
    logger.info("bio-stitch: %d exact-son merges, %d B4 roots left unstitched",
                len(merges), len(unstitched))

    if not dry_run:
        survivors = apply_merges(tree, merges)
        out = os.path.join(data_dir, "tree_bio_stitched.jsonl")
        with open(out, "w") as fh:
            for n in survivors:
                fh.write(json.dumps(n, ensure_ascii=False) + "\n")
        logger.info("wrote %d nodes -> %s (dropped %d folded B4 roots)",
                    len(survivors), out, len(tree) - len(survivors))

    rep = os.path.join(data_dir, "bio_stitch_report.json")
    with open(rep, "w") as fh:
        json.dump({"merged": len(merges), "unstitched": len(unstitched),
                   "merges": merges, "unstitched_roots": unstitched},
                  fh, ensure_ascii=False, indent=2)
    return {"merges": merges, "unstitched": unstitched}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--combined", default="tree.jsonl")
    p.add_argument("--books", nargs="+", default=["book3", "book4"])
    p.add_argument("--dry-run", action="store_true", help="report merges, don't write tree")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = bio_stitch(a.data_dir, a.combined, a.books, dry_run=a.dry_run)
    if a.dry_run:
        print(json.dumps({"merged": len(res["merges"]),
                          "sample_merges": res["merges"][:20],
                          "unstitched_sample": res["unstitched"][:10]},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
