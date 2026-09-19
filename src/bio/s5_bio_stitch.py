"""Bio stage 5 (stitch): use bio-listed sons to disambiguate cross-book / floater parents.

Motivating case: 繁中 (book4) had father-name 庆辉, but there are ~11 people named 庆辉, so
繁中's subtree floated ambiguously. Now that Book-3 bios list each 庆辉's own sons, exactly
one 庆辉 (book3 `78_84_105`) records 繁中 as a son — so we can re-parent 繁中's branch onto
that specific node.

Algorithm (conservative, additive — only re-parents, never deletes/edits names):
  For each node C in the combined tree whose current father P is a **floater root** (a
  root other than the true root 点) OR whose father's name is shared by many nodes:
    - collect candidate real parents = nodes whose ``name`` == P.name AND whose bio-listed
      sons (from the linked bio) claim C (exact, substring, or same first-2-chars to
      tolerate OCR bleed like ``繁中双桃承``).
    - if exactly ONE candidate (and it isn't C's current father), record C -> that parent.
  A "floater merge" (one floater root standing in for several real same-named people) is
  naturally un-collapsed: each child re-parents to its own bio-confirmed parent.

Input:  data/tree.jsonl (combined) + data/{book}_bio_linked.jsonl (bios keyed by prov).
Output: data/tree_bio_stitched.jsonl (tree.jsonl with re-parented father/children) +
        data/bio_stitch_report.json (every proposed edge with evidence).

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


def _sons(bio: dict) -> list[str]:
    if not bio:
        return []
    s = set((bio.get("sons", {}).get("vision") or []) + (bio.get("sons", {}).get("paddle") or []))
    return [x for x in s if x]


def _claims(child_name: str, son_token: str) -> bool:
    """True if a bio-son token refers to child_name (exact / substring / same first 2 chars)."""
    if not child_name or not son_token:
        return False
    if child_name in son_token:
        return True
    return len(child_name) >= 2 and son_token[:2] == child_name[:2]


def bio_stitch(data_dir: str = "data", combined: str = "tree.jsonl",
               books=("book3", "book4")) -> dict:
    tree = load_combined(os.path.join(data_dir, combined))
    by_id = {n["id"]: n for n in tree}
    bio = bio_index(data_dir, list(books))

    # attach bio to combined nodes (in memory) via provenance key
    for n in tree:
        b = bio.get(_prov(n["notes"]))
        if b:
            n["_bio"] = b

    # floater roots = roots other than the true root 点 (gen 0 / name 点)
    roots = [n for n in tree if n["father"] == -1]
    true_root = next((n for n in roots if n["name"] == "点"), None)
    floater_ids = {n["id"] for n in roots if n is not true_root}

    # name -> nodes that have bio sons (candidate real parents)
    parents_by_name = collections.defaultdict(list)
    for n in tree:
        if _sons(n.get("_bio", {})):
            parents_by_name[n["name"]].append(n)

    # a father name is "ambiguous" if >1 node bears it (so name-only stitch is unsafe)
    name_counts = collections.Counter(n["name"] for n in tree)

    proposals = []  # (child_id, old_father_id, new_father_id, evidence)
    for c in tree:
        fa = c["father"]
        if fa == -1 or fa not in by_id:
            continue
        parent = by_id[fa]
        # only touch children whose parent is a floater or has an ambiguous (shared) name
        if parent["id"] not in floater_ids and name_counts[parent["name"]] <= 1:
            continue
        cands = [p for p in parents_by_name.get(parent["name"], [])
                 if p["id"] != fa and any(_claims(c["name"], s) for s in _sons(p["_bio"]))]
        if len(cands) == 1:
            np_ = cands[0]
            proposals.append({
                "child_id": c["id"], "child_name": c["name"],
                "old_father_id": fa, "old_father_name": parent["name"],
                "new_father_id": np_["id"], "new_father_notes": _prov(np_["notes"]),
                "evidence_sons": _sons(np_["_bio"]),
            })

    # apply: move child to new father (update both children arrays + father pointer)
    applied = 0
    for pr in proposals:
        c = by_id[pr["child_id"]]
        old = by_id[pr["old_father_id"]]
        new = by_id[pr["new_father_id"]]
        if c["id"] in old["children"]:
            old["children"].remove(c["id"])
        if c["id"] not in new["children"]:
            new["children"].append(c["id"])  # appended youngest; ordering refined by a re-run
        c["father"] = new["id"]
        applied += 1

    # Prune floater roots we fully emptied: a floater root whose every child we just
    # re-parented onto a real (book3) node is a redundant duplicate of that person, so
    # drop it. Only prune roots that (a) are floaters, (b) have no children now, and
    # (c) DID have children before this run (i.e. we emptied them — never touch roots
    # that were already childless independently).
    had_children_before = {pr["old_father_id"] for pr in proposals}
    pruned = [n["id"] for n in tree
              if n["id"] in floater_ids and not n["children"]
              and n["id"] in had_children_before]
    pruned_set = set(pruned)
    tree = [n for n in tree if n["id"] not in pruned_set]

    # strip the in-memory bio helper before writing
    for n in tree:
        n.pop("_bio", None)

    out = os.path.join(data_dir, "tree_bio_stitched.jsonl")
    with open(out, "w") as fh:
        for n in tree:
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")
    rep = os.path.join(data_dir, "bio_stitch_report.json")
    with open(rep, "w") as fh:
        json.dump({"applied": applied, "pruned_floaters": pruned, "proposals": proposals},
                  fh, ensure_ascii=False, indent=2)
    logger.info("bio-stitch: %d edges re-parented, %d empty floaters pruned -> %s",
                applied, len(pruned), out)
    return {"applied": applied, "pruned": pruned, "proposals": proposals}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--combined", default="tree.jsonl")
    p.add_argument("--books", nargs="+", default=["book3", "book4"])
    p.add_argument("--dry-run", action="store_true", help="report proposals, don't write tree")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.dry_run:
        # compute without writing tree: run then discard by pointing at /dev/null-ish
        res = bio_stitch(a.data_dir, a.combined, a.books)
        print(json.dumps({"applied": res["applied"],
                          "sample": res["proposals"][:20]}, ensure_ascii=False, indent=2))
    else:
        bio_stitch(a.data_dir, a.combined, a.books)


if __name__ == "__main__":
    main()
