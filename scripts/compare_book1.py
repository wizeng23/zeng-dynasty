"""Compare the rewritten Book 1 parse against two oracles.

Oracles:
  - data/book1_golden.jsonl : 59 nodes, hand-typed, ~100% correct.
      RTL/eldest-first ordering, `father` field. SUBSET (first ~59 nodes).
  - old/data/book1.jsonl    : 163 nodes, old script ~95% correct.
      LTR ordering, `parent` field, empty names. FULL book.

New output:
  - data/book1.jsonl        : 163 nodes, RTL/eldest-first, `father` field.

The comparison is topology-only (parent/child structure), because:
  - IDs are renumbered (new = BFS-by-generation RTL; old = LTR).
  - Sibling order is flipped (new RTL/eldest-first vs old LTR).
  - Names are not populated in either the new output or the old output.

Method: reduce each tree to a canonical, order-independent, ID-independent
"shape signature" per node, computed bottom-up (a Merkle-style hash of the
multiset of child subtree shapes). Two forests match iff their multisets of
per-node shape signatures are equal. This is invariant to sibling order and
to ID renumbering, so it isolates genuine topological differences.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path: str) -> list[dict]:
    with open(os.path.join(REPO, path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def parent_field(rows: list[dict]) -> str:
    """Old rows use `parent`; new/golden use `father`."""
    return "father" if "father" in rows[0] else "parent"


def children_map(rows: list[dict]) -> dict[int, list[int]]:
    return {r["id"]: list(r["children"]) for r in rows}


def roots(rows: list[dict]) -> list[int]:
    pf = parent_field(rows)
    return [r["id"] for r in rows if r[pf] == -1]


def subtree_shape(node: int, kids: dict[int, list[int]], memo: dict[int, str]) -> str:
    """Canonical order-independent, ID-independent signature of the subtree.

    A leaf is "()". An internal node is "(" + sorted child-signatures + ")".
    Sorting the child signatures makes it invariant to sibling order.
    """
    if node in memo:
        return memo[node]
    child_sigs = sorted(subtree_shape(c, kids, memo) for c in kids.get(node, []))
    sig = "(" + "".join(child_sigs) + ")"
    memo[node] = sig
    return sig


def node_shape_multiset(rows: list[dict]) -> Counter:
    """Multiset of per-node subtree signatures across the whole forest."""
    kids = children_map(rows)
    memo: dict[int, str] = {}
    counter: Counter = Counter()
    for r in rows:
        counter[subtree_shape(r["id"], kids, memo)] += 1
    return counter


def degree_multiset(rows: list[dict]) -> Counter:
    """Multiset of child-counts (out-degrees) across all nodes — a coarse invariant."""
    return Counter(len(r["children"]) for r in rows)


def integrity(rows: list[dict]) -> dict:
    """Structural integrity checks: back-reference consistency, cycles, roots."""
    pf = parent_field(rows)
    ids = {r["id"] for r in rows}
    by_id = {r["id"]: r for r in rows}
    kids = children_map(rows)

    dangling_children = []  # child id referenced but not present
    backref_mismatch = []   # child's father/parent != the node claiming it
    for r in rows:
        for c in r["children"]:
            if c not in ids:
                dangling_children.append((r["id"], c))
            elif by_id[c][pf] != r["id"]:
                backref_mismatch.append((r["id"], c, by_id[c][pf]))

    orphan_parent = [r["id"] for r in rows if r[pf] != -1 and r[pf] not in ids]

    # cycle / connectivity: walk down from roots, count reachable
    rts = roots(rows)
    seen: set[int] = set()
    stack = list(rts)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(kids.get(n, []))
    unreachable = ids - seen

    return {
        "n_roots": len(rts),
        "roots": rts,
        "dangling_children": dangling_children,
        "backref_mismatch": backref_mismatch,
        "orphan_parent": orphan_parent,
        "unreachable_from_roots": sorted(unreachable),
    }


def forest_signature(rows: list[dict]) -> str:
    """One signature for the entire forest = sorted root-subtree signatures.

    Wraps all roots under a synthetic super-root so that the *set of trees*
    (and their shapes) must match as a whole, not just per-node multisets.
    """
    kids = children_map(rows)
    memo: dict[int, str] = {}
    root_sigs = sorted(subtree_shape(r, kids, memo) for r in roots(rows))
    return "[" + "".join(root_sigs) + "]"


def compare_multisets(a: Counter, b: Counter) -> dict:
    """Compare two multisets; report overlap and the symmetric difference."""
    keys = set(a) | set(b)
    matched = sum(min(a[k], b[k]) for k in keys)
    total_a = sum(a.values())
    total_b = sum(b.values())
    only_a = Counter({k: a[k] - b[k] for k in keys if a[k] > b[k]})
    only_b = Counter({k: b[k] - a[k] for k in keys if b[k] > a[k]})
    return {
        "matched": matched,
        "total_a": total_a,
        "total_b": total_b,
        "pct_of_a": 100.0 * matched / total_a if total_a else 0.0,
        "pct_of_b": 100.0 * matched / total_b if total_b else 0.0,
        "only_a": only_a,
        "only_b": only_b,
    }


def find_matching_subtree(new_rows: list[dict], golden_rows: list[dict]) -> dict:
    """Golden is the first ~59 nodes = subtree(s) rooted at the book's true root.

    In the new output each Stage-2 graph is a local subtree (local root gen 1).
    Golden's whole tree is a single deep tree rooted at Zeng Dian (id 1).
    We look for a node in the new forest whose subtree signature equals golden's
    forest signature — i.e. golden's shape appears verbatim as a new-output subtree.
    Also report the best-matching new subtree by node-multiset overlap.
    """
    g_sig = forest_signature(golden_rows)  # golden has a single root
    g_multiset = node_shape_multiset(golden_rows)

    new_kids = children_map(new_rows)
    memo: dict[int, str] = {}

    exact_matches = []
    for r in new_rows:
        if subtree_shape(r["id"], new_kids, memo) == g_sig.strip("[]"):
            exact_matches.append(r["id"])

    # Best node-multiset overlap among new subtrees whose size is comparable.
    def subtree_nodes(root: int) -> list[int]:
        out, stack = [], [root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(new_kids.get(n, []))
        return out

    best = None
    for r in new_rows:
        nodes = subtree_nodes(r["id"])
        if not (40 <= len(nodes) <= 80):  # golden is 59 nodes
            continue
        sub_ms = Counter(
            subtree_shape(n, new_kids, memo) for n in nodes
        )
        cmp = compare_multisets(sub_ms, g_multiset)
        if best is None or cmp["matched"] > best[1]["matched"]:
            best = (r["id"], cmp, len(nodes))

    return {
        "golden_forest_sig_matches_new_subtree_at": exact_matches,
        "golden_n_nodes": len(golden_rows),
        "best_size_matched_new_subtree": best,
    }


def golden_generation_check(new_rows: list[dict], golden_rows: list[dict]) -> dict:
    """Compare generation numbering shape golden-vs-new for the matching subtree.

    Golden root is gen 1 (Zeng Dian). If golden's tree appears as a new subtree
    rooted at some node R, then along every path the *relative* generation
    (depth) should agree. We compare the multiset of (out-degree, depth-in-tree)
    to check generations increment identically, independent of absolute offset.
    """
    def depth_degree_ms(rows: list[dict]) -> Counter:
        pf = parent_field(rows)
        by_id = {r["id"]: r for r in rows}
        kids = children_map(rows)
        rts = roots(rows)
        ms: Counter = Counter()
        stack = [(r, 0) for r in rts]
        while stack:
            nid, d = stack.pop()
            ms[(len(by_id[nid]["children"]), d)] += 1
            for c in kids.get(nid, []):
                stack.append((c, d + 1))
        return ms

    g = depth_degree_ms(golden_rows)
    # Compare against the full new forest's (degree, depth) multiset — golden's
    # (degree,depth) pairs should be a sub-multiset of the new forest's.
    n = depth_degree_ms(new_rows)
    missing = Counter({k: g[k] - n[k] for k in g if g[k] > n[k]})
    return {
        "golden_depth_degree_pairs": len(g),
        "pairs_not_covered_by_new_forest": dict(missing),
        "covered": not missing,
    }


def main() -> None:
    new = load("data/book1.jsonl")
    golden = load("data/book1_golden.jsonl")
    old = load("old/data/book1.jsonl")

    print("=" * 72)
    print("BOOK 1 REWRITE — COMPARISON REPORT")
    print("=" * 72)

    # 1. counts
    print("\n[1] NODE COUNTS")
    print(f"  new    (data/book1.jsonl)        : {len(new)}")
    print(f"  golden (data/book1_golden.jsonl) : {len(golden)}  (SUBSET, hand-typed)")
    print(f"  old    (old/data/book1.jsonl)    : {len(old)}  (FULL, ~95% correct)")

    # integrity of each
    print("\n[1b] STRUCTURAL INTEGRITY")
    for name, rows in [("new", new), ("golden", golden), ("old", old)]:
        it = integrity(rows)
        print(f"  {name}: roots={it['n_roots']} "
              f"backref_mismatch={len(it['backref_mismatch'])} "
              f"dangling_children={len(it['dangling_children'])} "
              f"orphan_parent={len(it['orphan_parent'])} "
              f"unreachable={len(it['unreachable_from_roots'])}")
        if it["backref_mismatch"]:
            print(f"      backref_mismatch sample: {it['backref_mismatch'][:5]}")
        if it["dangling_children"]:
            print(f"      dangling sample: {it['dangling_children'][:5]}")

    # 2/3. topology new-vs-old (unordered, ID-independent)
    print("\n[2] TREE TOPOLOGY: new vs old (RTL/LTR-invariant, ID-invariant)")
    new_ms = node_shape_multiset(new)
    old_ms = node_shape_multiset(old)
    cmp = compare_multisets(new_ms, old_ms)
    print(f"  per-node subtree-shape multiset match:")
    print(f"    matched signatures : {cmp['matched']}")
    print(f"    of new ({cmp['total_a']}) : {cmp['pct_of_a']:.1f}%")
    print(f"    of old ({cmp['total_b']}) : {cmp['pct_of_b']:.1f}%")

    # coarse invariant: degree distribution
    nd, od = degree_multiset(new), degree_multiset(old)
    print(f"  out-degree distribution:")
    print(f"    new: {dict(sorted(nd.items()))}")
    print(f"    old: {dict(sorted(od.items()))}")
    print(f"    identical: {nd == od}")

    # whole-forest signature equality
    same_forest = forest_signature(new) == forest_signature(old)
    print(f"  whole-forest signature identical (new==old): {same_forest}")

    if cmp["only_a"] or cmp["only_b"]:
        print("  MISMATCHED subtree shapes (shape -> count):")
        def summarize(sig: str) -> str:
            # depth & node count of a shape string, for human readability
            depth = 0
            maxd = 0
            leaves = 0
            nodes = 0
            for ch in sig:
                if ch == "(":
                    depth += 1
                    nodes += 1
                    maxd = max(maxd, depth)
                elif ch == ")":
                    depth -= 1
            # leaves = count of "()"
            leaves = sig.count("()")
            return f"nodes={nodes} depth={maxd} leaves={leaves}"
        only_a = cmp["only_a"].most_common(12)
        only_b = cmp["only_b"].most_common(12)
        print(f"    only-in-NEW ({sum(cmp['only_a'].values())} nodes, top shapes):")
        for sig, c in only_a:
            s = sig if len(sig) <= 60 else sig[:57] + "..."
            print(f"      x{c}  [{summarize(sig)}]  {s}")
        print(f"    only-in-OLD ({sum(cmp['only_b'].values())} nodes, top shapes):")
        for sig, c in only_b:
            s = sig if len(sig) <= 60 else sig[:57] + "..."
            print(f"      x{c}  [{summarize(sig)}]  {s}")

    # Per-graph structural comparison: does the parent->children partition match?
    # Compare the multiset of "family sizes" (each parent's child count) — a
    # sharper structural fingerprint than the global degree histogram.
    print("\n[2b] FAMILY-SHAPE FINGERPRINT (multiset of each node's child-count)")
    print(f"    new == old family-size multiset: {nd == od}")

    # 4. new vs golden (subset)
    print("\n[3] NEW vs GOLDEN (59-node subset)")
    gm = find_matching_subtree(new, golden)
    print(f"  golden nodes: {gm['golden_n_nodes']}")
    print(f"  golden tree shape found verbatim as new subtree at node id(s): "
          f"{gm['golden_forest_sig_matches_new_subtree_at'] or 'NONE'}")
    best = gm["best_size_matched_new_subtree"]
    if best:
        rid, bcmp, sz = best
        print(f"  best size-comparable new subtree: root id={rid} "
              f"({sz} nodes)")
        print(f"    node-shape overlap vs golden: matched={bcmp['matched']} "
              f"({bcmp['pct_of_b']:.1f}% of golden's {bcmp['total_b']})")

    gc = golden_generation_check(new, golden)
    print(f"  generation numbering (depth,degree) pairs of golden covered by new "
          f"forest: {gc['covered']}")
    if not gc["covered"]:
        print(f"    uncovered pairs (degree,depth)->count: "
              f"{gc['pairs_not_covered_by_new_forest']}")

    # Also directly compare golden vs old on the same subset shape, as a sanity
    # cross-check of golden itself against the (imperfect) old full parse.
    print("\n[3b] SANITY: GOLDEN vs OLD (does old contain golden's shape?)")
    old_ms2 = node_shape_multiset(old)
    g_ms = node_shape_multiset(golden)
    cg = compare_multisets(g_ms, old_ms2)
    print(f"    golden node-shapes matched within old: {cg['matched']}/{cg['total_a']} "
          f"({cg['pct_of_a']:.1f}% of golden)")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
