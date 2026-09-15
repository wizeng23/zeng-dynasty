"""Stage 5 post-fixes: hand-verified corrections applied on top of the parse.

The parse (:mod:`src.s5_build_tree`) is automatic and occasionally wrong in ways
that are obvious to a human on the QA page but not worth a rule: a line fragment
read as a nameless node, a hang-line whose upper and lower pieces do not line up
(so one person becomes a real name node plus a phantom that took their
children), a name crop that starts inside the glyph because the riser's ink
touches it. This module applies such corrections from ``data/{book}_fixes.json``
-- a ground-truth layer, like ``{book}_overrides.json`` for OCR or
``corners_review.json`` for Stage 1 -- so they are recorded, reviewable, and
re-applied after every Stage 5 run instead of hand-edited into the data.

Nodes are addressed by **provenance** (``"{graph}_{local_index}"``, the first
``" | "`` segment of ``notes``), which is stable across re-runs; ids are not.

Fixes file::

    {
      "delete": ["8_10_9"],                       // phantom with no real children
      "merge":  {"133_133_1": "133_133_0"},       // phantom -> real: real adopts its children
      "recrop": {"69_82_7": [0, 1462, 184, 1770]} // new name box (l, t, r, b) in graph px
    }

Effects: ``data/{book}.jsonl`` (nodes, ``children`` kept eldest-first = rightmost
column first, generations recomputed), the graph parse sidecars
(``{stem}.parse.json``: nodes removed, ``children_top`` and ``box`` updated), and
the name crops (``5_names/{id}.png`` rewritten for recrops, deleted for dropped
nodes). Re-cropped names must be re-OCR'd (``--ocr``).

CLI::

    python -m src.s5_fixes --book book2 [--ocr]
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from src.imaging import get_image, save_image
from src.s5_build_tree import _infer_generations

logger = logging.getLogger(__name__)


def _provenance(notes: str) -> str:
    return (notes or "").split(" | ", 1)[0]


def _stem_of(prov: str) -> str:
    return prov.rsplit("_", 1)[0]


class _Sidecars:
    """Lazy per-graph ``{stem}.parse.json`` access with write-back."""

    def __init__(self, graphs_dir: str):
        self.graphs_dir = graphs_dir
        self.loaded: dict[str, dict] = {}
        self.dirty: set[str] = set()

    def get(self, stem: str) -> dict:
        if stem not in self.loaded:
            self.loaded[stem] = json.load(open(os.path.join(self.graphs_dir, f"{stem}.parse.json")))
        return self.loaded[stem]

    def node(self, stem: str, node_id: int) -> dict | None:
        return next((n for n in self.get(stem)["nodes"] if n["id"] == node_id), None)

    def remove(self, stem: str, node_id: int) -> None:
        side = self.get(stem)
        side["nodes"] = [n for n in side["nodes"] if n["id"] != node_id]
        self.dirty.add(stem)

    def touch(self, stem: str) -> None:
        self.dirty.add(stem)

    def flush(self) -> None:
        for stem in self.dirty:
            with open(os.path.join(self.graphs_dir, f"{stem}.parse.json"), "w") as fh:
                json.dump(self.loaded[stem], fh)


def apply_fixes(book: str, books_dir: str = "books", data_dir: str = "data") -> dict:
    """Apply ``data/{book}_fixes.json`` to the Stage 5 output. Idempotent.

    Returns ``{"deleted": [prov...], "merged": [(phantom, real)...],
    "recropped_ids": [id...]}`` -- the last is what ``--ocr`` re-reads.
    """
    fixes_path = os.path.join(data_dir, f"{book}_fixes.json")
    if not os.path.exists(fixes_path):
        logger.info("no fixes file %s; nothing to do", fixes_path)
        return {"deleted": [], "merged": [], "recropped_ids": []}
    fixes = json.load(open(fixes_path))
    data_path = os.path.join(data_dir, f"{book}.jsonl")
    graphs_dir = os.path.join(books_dir, book, "4_graphs")

    rows = [json.loads(line) for line in open(data_path) if line.strip()]
    by_id = {r["id"]: r for r in rows}
    by_prov = {_provenance(r["notes"]): r for r in rows}
    sidecars = _Sidecars(graphs_dir)
    result: dict = {"deleted": [], "merged": [], "recropped_ids": []}

    def col_of(node: dict) -> int:
        side = sidecars.node(_stem_of(_provenance(node["notes"])), node["id"])
        return side["top"][1] if side else 0

    def sort_children_rtl(node: dict) -> None:
        # Eldest first = rightmost column first (the book reads right-to-left).
        node["children"].sort(key=lambda cid: -col_of(by_id[cid]))
        stem = _stem_of(_provenance(node["notes"]))
        side = sidecars.node(stem, node["id"])
        if side is not None:
            tops = []
            for cid in node["children"]:
                cs = sidecars.node(stem, cid)
                if cs is not None:
                    tops.append(cs["top"])
            side["children_top"] = tops
            sidecars.touch(stem)

    def drop(node: dict) -> None:
        prov = _provenance(node["notes"])
        if node["father"] != -1 and node["father"] in by_id:
            father = by_id[node["father"]]
            father["children"] = [c for c in father["children"] if c != node["id"]]
            sort_children_rtl(father)
        rows.remove(node)
        del by_id[node["id"]]
        del by_prov[prov]
        crop_path = os.path.join(books_dir, book, "5_names", f"{node['id']}.png")
        if os.path.exists(crop_path):
            os.remove(crop_path)
        sidecars.remove(_stem_of(prov), node["id"])

    # --- merge: the real node adopts the phantom's children -------------------
    for phantom_prov, real_prov in (fixes.get("merge") or {}).items():
        phantom, real = by_prov.get(phantom_prov), by_prov.get(real_prov)
        if phantom is None:
            logger.info("merge %s -> %s: phantom already gone (applied earlier)", phantom_prov, real_prov)
            continue
        if real is None:
            raise KeyError(f"merge {phantom_prov} -> {real_prov}: real node not found")
        for cid in phantom["children"]:
            by_id[cid]["father"] = real["id"]
        real["children"] = list(real["children"]) + list(phantom["children"])
        phantom["children"] = []
        sort_children_rtl(real)
        drop(phantom)
        result["merged"].append((phantom_prov, real_prov))
        logger.info("merged phantom %s into %s (%r)", phantom_prov, real_prov, real["name"])

    # --- delete: a phantom with no real children ---------------------------------
    for prov in fixes.get("delete") or []:
        node = by_prov.get(prov)
        if node is None:
            logger.info("delete %s: already gone (applied earlier)", prov)
            continue
        if node["children"]:
            raise ValueError(
                f"delete {prov}: node has children {node['children']}; use 'merge' to "
                f"hand them to the real node first"
            )
        drop(node)
        result["deleted"].append(prov)
        logger.info("deleted phantom %s", prov)

    # --- recrop: explicit name box in graph pixels ---------------------------------
    for prov, box in (fixes.get("recrop") or {}).items():
        node = by_prov.get(prov)
        if node is None:
            raise KeyError(f"recrop {prov}: node not found")
        left, top, right, bottom = box
        stem = _stem_of(prov)
        graph = get_image(os.path.join(graphs_dir, f"{stem}.png"))
        crop = graph[top:bottom, left:right]
        out = os.path.join(books_dir, book, "5_names", f"{node['id']}.png")
        save_image(crop, out)
        side = sidecars.node(stem, node["id"])
        if side is not None:
            side["box"] = [left, top, right, bottom]
            sidecars.touch(stem)
        result["recropped_ids"].append(node["id"])
        logger.info("recropped %s (%r) -> %s box=%s", prov, node["name"], out, box)

    # --- generations + write-back ---------------------------------------------------
    children_of = {r["id"]: list(r["children"]) for r in rows}
    father_of = {r["id"]: r["father"] for r in rows if r["father"] != -1}
    gens = _infer_generations(children_of, father_of)
    for r in rows:
        r["generation"] = gens.get(r["id"], -1)
    with open(data_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    sidecars.flush()
    logger.info("applied fixes to %s: %d merged, %d deleted, %d recropped -> %d nodes",
                book, len(result["merged"]), len(result["deleted"]),
                len(result["recropped_ids"]), len(rows))
    return result


def reocr(book: str, node_ids: list[int], books_dir: str = "books", data_dir: str = "data") -> None:
    """Re-read the given nodes' (re-cropped) name images and fold them into the data."""
    if not node_ids:
        return
    from PIL import Image

    from src import s6_ocr
    from src.s6_ocr_paddle import PaddleEngine

    names_path = os.path.join(data_dir, f"{book}_names.json")
    sidecar = json.load(open(names_path)) if os.path.exists(names_path) else {}
    engine = PaddleEngine()
    for node_id in node_ids:
        img = Image.open(os.path.join(books_dir, book, "5_names", f"{node_id}.png")).convert("L")
        name, conf, boxes = engine.recognize_name_boxed(img)
        rec = {"name": name, "confidence": round(conf, 4), "low_conf": (not name) or conf < 0.90}
        if boxes:
            rec["char_boxes"] = boxes
        old = sidecar.get(str(node_id), {}).get("name")
        sidecar[str(node_id)] = rec
        logger.info("re-OCR node %d: %r -> %r (conf %.3f)", node_id, old, name, conf)
    json.dump(sidecar, open(names_path, "w"), ensure_ascii=False, indent=2)
    s6_ocr.apply_names(book, data_dir=data_dir)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Apply hand-verified Stage 5 fixes.")
    ap.add_argument("--book", required=True)
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ocr", action="store_true", help="re-OCR re-cropped names and apply")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = apply_fixes(args.book, books_dir=args.books_dir, data_dir=args.data_dir)
    if args.ocr:
        reocr(args.book, result["recropped_ids"], books_dir=args.books_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
