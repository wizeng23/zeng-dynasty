"""Apply the hand-authored per-subgraph overrides as the book's ground truth.

Reads every ``data/{book}_manual/{stem}.json`` (the editor's output: each graph's
final node list -- box, top/bot endpoints, father/children), and writes:

  * ``books/{book}/4_graphs/{stem}.parse.json`` -- the sidecar the master QA renders
    from, so the QA reflects the overrides. Stale ``{stem}.imaginary.json`` /
    ``.nicks.json`` are removed (a hand-authored graph has no synthetic bridges).
  * ``data/{book}.jsonl`` -- the domain tree, with globally-unique IDs assigned in
    the same RTL / BFS-by-generation order ``s5_build_tree.build_tree`` uses (graphs
    in page order; each graph's nodes eldest-first = rightmost column first), and
    generations inferred by walking down from each root. Names are carried over from
    the existing jsonl by graph-local index where present, else left empty for OCR.
  * ``books/{book}/5_names/{id}.png`` -- the name crop for each node, cut from its
    box (so OCR can re-read them).

CLI: python -m scripts.qa.apply_manual --book book3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os

from src.imaging import get_image, save_image
from src.model import Node
from src.s5_build_tree import _infer_generations

logger = logging.getLogger(__name__)


def _col(n: dict) -> int:
    return (n["box"][0] + n["box"][2]) // 2


def _row(n: dict) -> int:
    return n["top"][0]


def _order_rtl(nodes: list[dict]) -> list[dict]:
    """Eldest-first within each generation row, rows top-to-bottom -- the order
    build_tree assigns IDs in. Rightmost column = eldest (the book reads RTL)."""
    return sorted(nodes, key=lambda n: (_row(n), -_col(n)))


def apply_manual(book: str, books_dir: str = "books", data_dir: str = "data") -> int:
    manual_dir = os.path.join(data_dir, f"{book}_manual")
    graphs_dir = os.path.join(books_dir, book, "4_graphs")
    names_dir = os.path.join(books_dir, book, "5_names")
    os.makedirs(names_dir, exist_ok=True)

    stems = sorted(
        (os.path.splitext(f)[0] for f in os.listdir(manual_dir) if f.endswith(".json")),
        key=lambda s: int(s.split("_")[0]),
    )

    # Carry over existing OCR names by provenance ("{stem}_{local_index}").
    prev_name: dict[str, str] = {}
    jsonl = os.path.join(data_dir, f"{book}.jsonl")
    if os.path.exists(jsonl):
        for line in open(jsonl):
            r = json.loads(line)
            prov = (r.get("notes") or "").split(" | ", 1)[0]
            if r.get("name"):
                prev_name[prov] = r["name"]

    node_idx = 1
    father_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = {}
    records: list[tuple[int, str, dict, str]] = []  # (id, stem, override-node, provenance)

    for stem in stems:
        ov = json.load(open(os.path.join(manual_dir, f"{stem}.json")))
        nodes = ov["nodes"]
        # A node has exactly one father, so ``father`` is authoritative: keep each
        # node only in its father's children list, dropping any stray extra parent
        # (an editor slip where a child was connected twice). Then rebuild every
        # children list from the father map so the two are always consistent.
        by_local = {n["id"]: n for n in nodes}
        fathered = {n["id"]: n.get("father", -1) for n in nodes}
        for n in nodes:
            n["children"] = sorted(
                (m["id"] for m in nodes if fathered.get(m["id"]) == n["id"]),
                key=lambda cid: -_col(by_local[cid]),  # eldest-first (rightmost col)
            )
        ordered = _order_rtl(nodes)
        # local override id -> new global id, in RTL/BFS order
        gid: dict[int, int] = {}
        for local_i, n in enumerate(ordered):
            gid[n["id"]] = node_idx
            records.append((node_idx, stem, n, f"{stem}_{local_i}"))
            node_idx += 1
        for n in ordered:
            nid = gid[n["id"]]
            kids = [gid[c] for c in n["children"] if c in gid]
            children_of[nid] = kids
            for c in kids:
                father_of[c] = nid

        # sidecar for the QA (build children_top from each child's top point)
        by_id = {n["id"]: n for n in nodes}
        sidecar_nodes = []
        for n in nodes:
            sidecar_nodes.append({
                "id": gid[n["id"]],
                "box": n["box"],
                "top": n["top"],
                "bot": n["bot"],
                "empty": not n.get("name"),
                "children_top": [by_id[c]["top"] for c in n["children"] if c in by_id],
            })
        with open(os.path.join(graphs_dir, f"{stem}.parse.json"), "w") as fh:
            json.dump({"nodes": sidecar_nodes, "scrubbed": []}, fh)
        # a hand-authored graph carries no synthetic fills
        for suffix in ("imaginary", "nicks"):
            p = os.path.join(graphs_dir, f"{stem}.{suffix}.json")
            if os.path.exists(p):
                os.remove(p)
        logger.info("%s: %d nodes", stem, len(nodes))

    generation_of = _infer_generations(children_of, father_of)

    with open(jsonl, "w") as out:
        for node_id, stem, n, prov in records:
            # name crop from the box
            graph = get_image(os.path.join(graphs_dir, f"{stem}.png"))
            l, t, r, b = n["box"]
            if r > l and b > t:
                save_image(graph[t:b, l:r], os.path.join(names_dir, f"{node_id}.png"))
            node = Node(
                id=node_id,
                name=prev_name.get(prov, ""),
                name_images=[os.path.join(names_dir, f"{node_id}.png")],
                generation=generation_of.get(node_id, -1),
                father=father_of.get(node_id, -1),
                children=children_of.get(node_id, []),
                notes=prov,
            )
            out.write(json.dumps(dataclasses.asdict(node), ensure_ascii=False) + "\n")

    logger.info(
        "Applied %d graphs -> %d nodes -> %s (+ sidecars, name crops)",
        len(stems), len(records), jsonl,
    )
    return len(records)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", required=True)
    p.add_argument("--books-dir", default="books")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = apply_manual(args.book, args.books_dir, args.data_dir)
    logger.info("Done. %d nodes.", n)


if __name__ == "__main__":
    main()
