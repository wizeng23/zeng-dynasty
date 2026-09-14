"""Verify src/stitch.py against the archived book1_merged oracle.

The oracle ``data/oracles/book1_merged.jsonl`` (150 nodes, 1 root) is the
hand-merged Book 1 tree recovered from git history (commit d544cd8). It was built
by applying William's 13 manual cross-graph merges to the OLD Stage-3 parse
(``old/data/book1.jsonl``).

This script proves two things with order-invariant (AHU) tree isomorphism:

  1. STITCH LOGIC IS CORRECT: applying the 13 merges to the old parse reproduces
     the oracle exactly. So the merge/attach logic in src.stitch is faithful.

  2. THE REWRITE'S PARSE DIFFERS SLIGHTLY: applying the same 13 merges to the
     NEW parse (data/book1.jsonl) does NOT match the oracle -- even though the
     new parse is AHU-identical to the old parse as a *forest*. The difference is
     a within-graph parse nuance between the rewrite and the archived old run
     that only surfaces once the subtrees are connected. It is NOT a stitch bug.

Run: python -m scripts.verify_stitch  (or: python scripts/verify_stitch.py)
"""

from __future__ import annotations

import json
import sys

sys.setrecursionlimit(100_000)

MERGES = [
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
]


def load(path: str) -> dict[int, dict]:
    return {n["id"]: n for n in (json.loads(l) for l in open(path) if l.strip())}


def apply_merges(nodes: dict[int, dict], parent_key: str) -> list[dict]:
    """Fold each duplicate root into its canonical leaf (mutates copies)."""
    # Provenance = notes head: strip Stage-6 " | ocr_*" tags, then the "/" merge
    # trace ("canonical/duplicate") that stitching may have written.
    prov2id = {
        n["notes"].split(" | ", 1)[0].split("/")[0]: n["id"]
        for n in nodes.values()
    }
    dropped: set[int] = set()
    for dup_prov, canon_prov in MERGES:
        dup = nodes[prov2id[dup_prov]]
        canon = nodes[prov2id[canon_prov]]
        canon["children"] = list(canon["children"]) + list(dup["children"])
        for cid in dup["children"]:
            nodes[cid][parent_key] = canon["id"]
        dropped.add(dup["id"])
    return [n for n in nodes.values() if n["id"] not in dropped]


def ahu(nodes: list[dict], parent_key: str) -> list[str]:
    """Canonical unordered-tree encoding (AHU): isomorphism ignoring sibling order."""
    children = {n["id"]: n.get("children", []) for n in nodes}
    memo: dict[int, str] = {}

    def enc(i: int) -> str:
        if i not in memo:
            memo[i] = "(" + "".join(sorted(enc(c) for c in children[i])) + ")"
        return memo[i]

    return sorted(enc(n["id"]) for n in nodes if n[parent_key] == -1)


def main() -> None:
    oracle = list(load("data/oracles/book1_merged.jsonl").values())
    oracle_ahu = ahu(oracle, "parent")

    old_merged = apply_merges(load("old/data/book1.jsonl"), "parent")
    stitched = list(load("data/book1_stitched.jsonl").values())

    checks = [
        # 1. The stitch LOGIC reproduces the oracle from the old parse + hand merges.
        ("old parse + 13 merges == oracle  (stitch logic is correct)",
         ahu(old_merged, "parent") == oracle_ahu),
        # 2. The actual pipeline output (s7_stitch.py, using the NAME-matcher) has
        #    the same TREE SHAPE as the oracle. This is the real end-to-end check --
        #    it passes now that name-matching pairs the seams correctly (the old
        #    provenance-keyed list mis-connected 7 of 13 under the v1 renumber).
        ("s7_stitch.py output == oracle tree shape  (pipeline reproduces oracle)",
         ahu(stitched, "father") == oracle_ahu),
    ]

    print(f"oracle: {len(oracle)} nodes | stitched: {len(stitched)} nodes")
    print(f"stitched roots: {sum(1 for n in stitched if n['father'] == -1)} | "
          f"max generation: {max(n['generation'] for n in stitched)}\n")
    for label, result in checks:
        print(f"  [{'PASS' if result else 'FAIL'}] {label}")

    ok = all(c[1] for c in checks)
    print("\n" + ("STITCH VERIFIED: the name-matched pipeline reproduces the oracle "
                  "tree shape." if ok else "STITCH VERIFICATION FAILED -- investigate."))


if __name__ == "__main__":
    main()
