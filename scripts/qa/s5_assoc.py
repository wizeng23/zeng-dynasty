"""QA: review the graph<->bio association per SUBGRAPH, verify or flag each for later override.

For each subgraph (a ``{start}_{end}`` stem) this pairs the parsed tree nodes with their
linked bios and audits the match. A subgraph is an EXACT match iff, generation by generation:

  * counts line up -- every graph node has a bio and every bio a node (no leftover on
    either side); AND for each linked pair:
  * the bio's own name == the graph node's name; AND
  * the bio's father char == the last char of the graph node's father's name; AND
  * the bio's son list == the graph node's children names (same set/order).

Anything short of that is surfaced as a warning so you can eyeball the subgraph. You then
VERIFY it (looks right) or FLAG it (needs a manual override later); both advance to the next
subgraph -- same loop as the OCR name review (scripts/qa/s6_ocr.py). Decisions persist to
``data/{book}_bio_assoc_review.json`` = ``{stem: "verified" | "flagged"}`` so a re-run resumes
where you left off (``--all`` re-shows already-decided ones; ``--flagged`` shows only flags).

Read-only over the data -- writes only the review-state file. Run:

    PYTHONPATH=. python -m scripts.qa.s5_assoc --book book3
    PYTHONPATH=. python -m scripts.qa.s5_assoc --book book3 --flagged   # revisit flags
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import src.bio.s5_link as L

DATA_DIR = "data"


def _prov(n: dict) -> str:
    return (n.get("notes", "") or "").split(" | ", 1)[0].strip()


def _father_char(bio: dict) -> str | None:
    fc = bio.get("father_char") or {}
    return (fc.get("vision") or fc.get("paddle")) if isinstance(fc, dict) else fc


def _bio_name(bio: dict) -> str | None:
    # s5_link stores the block's name under "name_ocr" (make_bio_field); tolerate "name" too.
    nm = bio.get("name_ocr") or bio.get("name") or {}
    return (nm.get("vision") or nm.get("paddle")) if isinstance(nm, dict) else nm


def _bio_sons(bio: dict) -> list[str]:
    s = bio.get("sons") or {}
    return (s.get("vision") or s.get("paddle") or []) if isinstance(s, dict) else (s or [])


def audit_node(n: dict, by_id: dict) -> list[str]:
    """Per-node problems: name / father-char / sons mismatch (empty == exact)."""
    problems: list[str] = []
    if not n.get("bio"):
        return ["no bio"]
    bio = n["bio"]
    if _bio_name(bio) != n["name"]:
        problems.append(f"name graph={n['name']!r} bio={_bio_name(bio)!r}")
    fa = by_id.get(n["father"])
    fc = _father_char(bio)
    if fa and fa.get("name") and fc and fc[-1] != fa["name"][-1]:
        problems.append(f"father bio={fc!r} vs graph father {fa['name']!r}")
    graph_sons = [by_id[c]["name"] for c in n.get("children", []) if c in by_id]
    if _bio_sons(bio) != graph_sons:
        problems.append(f"sons bio={_bio_sons(bio)} graph={graph_sons}")
    return problems


def audit_subgraph(nodes: list[dict], by_id: dict) -> dict:
    """Audit a whole subgraph. Returns {n_exact, n_total, bad:[(node, problems)]}."""
    bad = []
    for n in nodes:
        ps = audit_node(n, by_id)
        if ps:
            bad.append((n, ps))
    return {"n_exact": len(nodes) - len(bad), "n_total": len(nodes), "bad": bad}


def review_path(book: str) -> str:
    return os.path.join(DATA_DIR, f"{book}_bio_assoc_review.json")


def load_review(book: str) -> dict:
    p = review_path(book)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_review(book: str, review: dict) -> None:
    with open(review_path(book), "w") as f:
        json.dump(review, f, ensure_ascii=False, indent=2, sort_keys=True)


def build(book: str):
    """Return (stems_in_order, {stem: nodes}, by_id)."""
    tree = L.load_tree(os.path.join(DATA_DIR, f"{book}_bio_linked.jsonl"))
    by_id = {n["id"]: n for n in tree}
    tb = collections.defaultdict(list)
    for n in tree:
        s = L._stem_of(n)
        if s:
            tb[s].append(n)
    stems = sorted(tb, key=lambda s: [int(x) for x in s.split("_")])
    return stems, tb, by_id


def run(book: str, show_all: bool, only_flagged: bool):
    stems, tb, by_id = build(book)
    review = load_review(book)

    # audit everything first, so we can show progress and jump to the interesting ones.
    audited = {s: audit_subgraph(tb[s], by_id) for s in stems}
    perfect = [s for s in stems if not audited[s]["bad"]]
    print(f"{book}: {len(stems)} subgraphs -- {len(perfect)} fully exact, "
          f"{len(stems)-len(perfect)} with node-level mismatches\n")

    for stem in stems:
        a = audited[stem]
        decided = review.get(stem)
        if only_flagged and decided != "flagged":
            continue
        if not show_all and not only_flagged:
            if not a["bad"]:
                continue          # skip fully-exact subgraphs unless --all
            if decided:
                continue          # skip already-decided unless --all/--flagged

        print("=" * 72)
        print(f"SUBGRAPH {stem}   {a['n_exact']}/{a['n_total']} nodes exact"
              + (f"   [previously {decided}]" if decided else ""))
        if not a["bad"]:
            print("  ✓ every node's name + father + sons agree with its bio")
        else:
            print(f"  ⚠ {len(a['bad'])} node(s) NOT exact:")
            for n, ps in a["bad"]:
                print(f"     {n['name']:<4} gen{n['generation']} [{_prov(n)}]")
                for p in ps:
                    print(f"        - {p}")

        ans = input("\n  [v]erify  [f]lag  [s]kip  [q]uit > ").strip().lower()
        if ans == "q":
            break
        if ans == "v":
            review[stem] = "verified"; save_review(book, review)
        elif ans == "f":
            review[stem] = "flagged"; save_review(book, review)
        # s / anything else: skip, no state change
    print(f"\nsaved -> {review_path(book)}  "
          f"({sum(v=='verified' for v in review.values())} verified, "
          f"{sum(v=='flagged' for v in review.values())} flagged)")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--all", action="store_true", help="show every subgraph, incl. exact/decided")
    p.add_argument("--flagged", action="store_true", help="show only previously-flagged subgraphs")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    run(a.book, a.all, a.flagged)


if __name__ == "__main__":
    main()
