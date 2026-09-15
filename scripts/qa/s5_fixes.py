"""Visual QA for the Stage-5 post-fixes layer (``data/{book}_fixes.json``).

The ad hoc ``books/book2/qa/fixes.html`` was a hand-written narrative of one
review round. This is its data-driven successor: it reads the fixes file and
renders one section per fix -- delete / merge / recrop -- showing the affected
node's name crop and, for a recrop, a crop of the graph at the new box so the box
can be eyeballed against the ink. It reads only what already exists after
``src.s5_fixes`` has run (the ``{book}.jsonl`` rows, the ``{stem}.parse.json``
sidecars, the ``5_names/{id}.png`` crops), so run it AFTER applying the fixes.

Usage::

    python -m scripts.qa.s5_fixes --book book2
    # -> books/book2/qa/fixes.html  (serve books/book2/qa on a localhost port)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from src.imaging import get_image

logger = logging.getLogger(__name__)

# A recrop's graph-context view is padded this far around the box so the reviewer
# sees the glyph in its neighbourhood, not just the tight crop.
CONTEXT_PAD = 120


def _provenance(notes: str) -> str:
    return (notes or "").split(" | ")[0]


def _stem_of(prov: str) -> str:
    return prov.rsplit("_", 1)[0]


def _png_data_uri(a: np.ndarray) -> str:
    """A binary ink grid (0 == ink) -> a base64 PNG data URI (self-contained page)."""
    img = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _crop(graph: np.ndarray, box: list[int], pad: int = 0) -> np.ndarray:
    left, top, right, bottom = box
    h, w = graph.shape
    return graph[max(0, top - pad):min(h, bottom + pad),
                 max(0, left - pad):min(w, right + pad)]


def _fig(caption: str, data_uri: str) -> str:
    return (f'<figure><figcaption>{html.escape(caption)}</figcaption>'
            f'<img src="{data_uri}" loading="lazy"></figure>')


def build_page(book: str, books_dir: str = "books", data_dir: str = "data") -> str:
    fixes_path = os.path.join(data_dir, f"{book}_fixes.json")
    if not os.path.exists(fixes_path):
        raise FileNotFoundError(f"no fixes file for {book}: {fixes_path}")
    fixes = json.load(open(fixes_path))

    rows = [json.loads(l) for l in open(os.path.join(data_dir, f"{book}.jsonl"))]
    by_prov = {_provenance(r["notes"]): r for r in rows}
    graphs_dir = os.path.join(books_dir, book, "4_graphs")
    names_dir = os.path.join(books_dir, book, "5_names")

    graph_cache: dict[str, np.ndarray] = {}

    def graph(stem: str) -> np.ndarray:
        if stem not in graph_cache:
            graph_cache[stem] = get_image(os.path.join(graphs_dir, f"{stem}.png"))
        return graph_cache[stem]

    def name_crop_uri(node_id: int) -> str | None:
        p = os.path.join(names_dir, f"{node_id}.png")
        if not os.path.exists(p):
            return None
        img = Image.open(p).convert("L")
        return _png_data_uri((np.asarray(img) > 127).astype(np.uint8))

    sections: list[str] = []

    # --- deletes -------------------------------------------------------------
    for prov in (fixes.get("delete") or []):
        node = by_prov.get(prov)
        gone = node is None
        note = ("node no longer in the parse (deleted as intended)" if gone
                else f"STILL PRESENT as id {node['id']} -- delete did not take?")
        sections.append(
            f'<section><h2>delete {html.escape(prov)}</h2>'
            f'<p>{html.escape(note)}</p></section>')

    # --- merges: phantom -> real. Show the surviving real node's name crop. ---
    for phantom, real in (fixes.get("merge") or {}).items():
        node = by_prov.get(real)
        figs = ""
        if node is not None:
            uri = name_crop_uri(node["id"])
            if uri:
                figs = _fig(f"{real} = {node.get('name', '?')} (id {node['id']}, "
                            f"{len(node.get('children', []))} children)", uri)
        still = phantom in by_prov
        note = (f'{html.escape(phantom)} folded into {html.escape(real)}.'
                + (' <b>WARNING: phantom still present.</b>' if still else ''))
        sections.append(
            f'<section><h2>merge {html.escape(phantom)} &rarr; {html.escape(real)}</h2>'
            f'<p>{note}</p><div class="row">{figs}</div></section>')

    # --- recrops: show the new name crop AND the graph at the box. -----------
    for prov, box in (fixes.get("recrop") or {}).items():
        node = by_prov.get(prov)
        stem = _stem_of(prov)
        figs = ""
        try:
            g = graph(stem)
            figs += _fig(f"graph @ box {box}", _png_data_uri(_crop(g, box)))
            figs += _fig(f"+/-{CONTEXT_PAD}px context",
                         _png_data_uri(_crop(g, box, pad=CONTEXT_PAD)))
        except FileNotFoundError:
            figs += f'<p>graph {html.escape(stem)}.png not found</p>'
        if node is not None:
            uri = name_crop_uri(node["id"])
            if uri:
                figs += _fig(f"5_names crop (id {node['id']}, "
                             f"OCR {node.get('name', '?')})", uri)
        sections.append(
            f'<section><h2>recrop {html.escape(prov)} '
            f'{html.escape(node["name"]) if node else ""}</h2>'
            f'<p>new box (l,t,r,b) = {box}</p><div class="row">{figs}</div></section>')

    style = (
        "body{font-family:system-ui,sans-serif;margin:0;background:#f5f5f4;color:#1c1917}"
        "header{background:#1c1917;color:#fafaf9;padding:12px 20px}"
        "section{padding:16px 20px;border-bottom:1px solid #d6d3d1}"
        "h2{font-size:17px;margin:0 0 6px}p{font-size:13px;color:#44403c;margin:0 0 10px}"
        ".row{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start}figure{margin:0}"
        "figcaption{font-size:12px;color:#57534e;margin-bottom:4px}"
        "img{border:1px solid #a8a29e;background:#fff;display:block;max-height:520px;"
        "width:auto;max-width:100%}")
    n = sum(len(fixes.get(k) or []) for k in ("delete", "merge", "recrop"))
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{book} post-fixes QA</title><style>{style}</style></head>'
        f'<body><header>{book} post-fixes &mdash; {n} fixes from '
        f'data/{book}_fixes.json &middot; '
        f'<a style="color:#93c5fd" href="index.html">back to parse QA</a></header>'
        + "".join(sections) + "</body></html>")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)

    page = build_page(args.book, books_dir=args.books_dir, data_dir=args.data_dir)
    out_dir = os.path.join(args.books_dir, args.book, "qa")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fixes.html")
    with open(out, "w") as fh:
        fh.write(page)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
