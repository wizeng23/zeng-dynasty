"""Per-checkpoint visual artifacts for the v1 (bitonal ADF) pipeline.

Writes a reviewable screenshot at EACH pipeline checkpoint so every stage can be
eyeballed independently, plus a per-page filmstrip that lines the stages up side
by side, and an ``index.html`` to click through. Everything lands under
``data/artifacts/{book}/``:

    data/artifacts/{book}/
      01_raw/{i}.png          raw page rendered from the PDF (pre-clean)
      02_page/{i}.png         deskewed + border-stripped page (Stage 1 output)
      03_graph/{i}.png        cleaned graph: frame + header/label removed (Stage 2)
      04_parse/{i}.png        parse overlay: red name boxes, line endpoints (Stage 3)
      filmstrip/{i}.png       the four stages above, side by side, for one page
      index.html              gallery of the filmstrips

Stages 02/03 are read from the pipeline's own outputs (``pages_{variant}`` /
``graphs_{variant}``); stage 01 is re-rendered from the source PDF; stage 04 is
computed here by running the parser and drawing its detections.

    python -m scripts.qa.all_filmstrip --book book1 --variant bw \
        --pdf books/book1/book1.pdf --first-page 6 --last-page 23
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw

import src.s5_build_tree as bt
import src.s1_extract_pages as ep
from src.imaging import get_image

Image.MAX_IMAGE_PIXELS = None

# Long side (px) each stage thumbnail is scaled to for review.
THUMB = 900


def _rgb(a: np.ndarray) -> Image.Image:
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def _fit(img: Image.Image, long_side: int = THUMB) -> Image.Image:
    s = long_side / max(img.width, img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))


def parse_overlay(a: np.ndarray, config) -> Image.Image:
    """Cleaned graph with red name boxes + endpoint dots for each detected node."""
    try:
        nodes = bt.parse_graph(a, config)
        note = f"{len(nodes)} nodes"
    except Exception as e:  # noqa: BLE001 - keep the image even on parse failure
        nodes, note = [], f"PARSE ERROR: {e}"
    img = _rgb(a)
    dr = ImageDraw.Draw(img)
    half = config.node_max_height // 3
    for n in nodes:
        if n.top is None or n.bot is None:
            continue
        (tr, tc), (br, bc) = n.top, n.bot
        dr.rectangle((tc - half, tr - half, tc + half, tr + half // 2),
                     outline=(220, 30, 30), width=6)
        dr.ellipse((tc - 12, tr - 12, tc + 12, tr + 12), fill=(30, 120, 220))
        dr.ellipse((bc - 12, br - 12, bc + 12, br + 12), fill=(30, 180, 30))
    return img, note


def filmstrip(stages: list[tuple[str, Image.Image]]) -> Image.Image:
    """Lay labelled stage thumbnails left-to-right on one canvas."""
    pad, label_h = 16, 24
    thumbs = [(_fit(im), name) for name, im in stages]
    h = max(t.height for t, _ in thumbs) + label_h
    w = sum(t.width for t, _ in thumbs) + pad * (len(thumbs) + 1)
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    dr = ImageDraw.Draw(canvas)
    x = pad
    for t, name in thumbs:
        canvas.paste(t, (x, label_h))
        dr.text((x, 6), name, fill=(0, 0, 0))
        x += t.width + pad
    return canvas


def build(book, variant, pdf, first_page, last_page, books_dir="books", data_dir="data"):
    root = os.path.join(data_dir, "artifacts", book)
    dirs = {k: os.path.join(root, k) for k in
            ("01_raw", "02_page", "03_graph", "04_parse", "filmstrip")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Canonical bitonal output lives at books/{book}/{pages,graphs}; the grayscale
    # variant is namespaced under gray/.
    sub = "" if variant == "bw" else f"{variant}/"
    pages_dir = os.path.join(books_dir, book, f"{sub}1_pages")
    graphs_dir = os.path.join(books_dir, book, f"{sub}4_graphs")
    config = bt.BOOK_CONFIGS.get(book, bt.BookConfig())

    doc = None
    if pdf:
        import pymupdf
        doc = pymupdf.open(pdf)

    n_pages = len([f for f in os.listdir(pages_dir) if f.endswith(".png")])
    rows = []
    for i in range(n_pages):
        stages = []

        # 01 raw (re-render from PDF, pre-clean) -- optional
        if doc is not None:
            raw = ep.render_page_binary(doc, first_page + i)
            raw_img = _rgb(raw)
            raw_img.save(os.path.join(dirs["01_raw"], f"{i}.png"))
            stages.append(("01 raw", raw_img))

        # 02 deskewed + border-stripped page
        page = get_image(os.path.join(pages_dir, f"{i}.png"))
        page_img = _rgb(page)
        page_img.save(os.path.join(dirs["02_page"], f"{i}.png"))
        stages.append(("02 page", page_img))

        # 03 cleaned graph
        gpath = os.path.join(graphs_dir, f"{i}_{i}.png")
        note = ""
        if os.path.exists(gpath):
            graph = get_image(gpath)
            _rgb(graph).save(os.path.join(dirs["03_graph"], f"{i}.png"))
            stages.append(("03 graph", _rgb(graph)))
            # 04 parse overlay
            ov, note = parse_overlay(graph, config)
            ov.save(os.path.join(dirs["04_parse"], f"{i}.png"))
            stages.append(("04 parse", ov))

        strip = filmstrip(stages)
        strip.save(os.path.join(dirs["filmstrip"], f"{i}.png"))
        rows.append((i, note))
        print(f"page {i}: {note}")

    if doc is not None:
        doc.close()

    html = ["<html><body style='background:#1e1e1e;color:#eee;font-family:sans-serif'>",
            f"<h2>{book} ({variant}) &mdash; per-checkpoint artifacts</h2>",
            "<p>stages: 01 raw &rarr; 02 deskew+strip &rarr; 03 clean graph &rarr; "
            "04 parse (red=name box, blue=line top, green=line bottom)</p>"]
    for i, note in rows:
        html.append(f"<div style='margin:24px 0'><b>page {i}</b> &mdash; {note}<br>"
                    f"<img src='filmstrip/{i}.png' style='max-width:100%;"
                    f"border:1px solid #444'></div>")
    html.append("</body></html>")
    idx = os.path.join(root, "index.html")
    with open(idx, "w") as fh:
        fh.write("\n".join(html))
    print(f"\nwrote artifacts for {len(rows)} pages -> {root}")
    print(f"open: {idx}")
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--variant", default="bw", help="bw or gray")
    ap.add_argument("--pdf", default=None, help="source PDF for the 01_raw stage (optional)")
    ap.add_argument("--first-page", type=int, default=0, help="first content page in the PDF")
    ap.add_argument("--last-page", type=int, default=0)
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    build(args.book, args.variant, args.pdf, args.first_page, args.last_page,
          books_dir=args.books_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
