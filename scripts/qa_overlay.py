"""Visual QA for the parse: overlay detections on the graph images.

Produces two overlays per Book-N graph, plus an HTML index to click through them:

1. **Parse overlay** (``*_parse.png``): the graph image with a RED box around each
   detected name and a RED line for each detected parent->child edge. Overlaid on
   the actual ink, a wrong box or missing/extra edge is obvious at a glance.

2. **Assembly overlay** (``*_pages.png``): the same graph with BLUE vertical bands
   marking where each source page was stitched in, labelled with the page number.
   The book reads right-to-left, so page numbers should ASCEND from right to left;
   an out-of-order or misplaced label means the page merge went wrong.

The graph images (``books/bookN/graphs/{start}_{end}.png``) are already the
pipeline's deskewed pages roughly joined, so this verifies name-detection,
edge-parsing, AND page assembly in one place.

Usage::

    python -m scripts.qa_overlay --book book1
    # -> books/book1/qa/*.png + books/book1/qa/index.html
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Book 2's 36_52 spans 17 pages (~93M px); lift PIL's decompression-bomb guard,
# which these known-safe local scans exceed.
Image.MAX_IMAGE_PIXELS = None

from src import segment as seg
from src import build_tree as bt
from src.imaging import get_image

logger = logging.getLogger(__name__)

RED = (220, 30, 30)
BLUE = (40, 90, 220)


def _to_rgb(a: np.ndarray) -> Image.Image:
    """Binary ink grid (0=ink,1=bg) -> white-background RGB image."""
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def _font(size: int = 22) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def name_box(node: bt.LineNode, width: int) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) name box for a node, matching get_name_image.

    get_name_image crops the band between top and bot, in a +-40px column window
    around the node's column. We draw that same window (in graph coordinates:
    node.top/.bot are (row, col)).
    """
    assert node.top is not None and node.bot is not None
    col = node.top[1]
    top_row = node.top[0]
    bot_row = node.bot[0]
    left = max(0, col - 40)
    right = min(width, col + 40)
    return (left, top_row, right, bot_row)


def draw_parse_overlay(a: np.ndarray, nodes: list[bt.LineNode]) -> Image.Image:
    """Red name boxes + red parent->child edges over the graph image."""
    img = _to_rgb(a)
    draw = ImageDraw.Draw(img)

    # Edges: from each parent's fan-out point (bot) to each child's top, drawn as
    # the book's actual routing — down from the parent to the sibling bar, across
    # to the child's column, then down into the child. The parent's `bot` row is
    # where its line fans out; the children's `top` rows share a common bar just
    # below it, so we route through a bar at the parent's bot row.
    for n in nodes:
        if n.bot is None or not n.children:
            continue
        pr, pc = n.bot  # parent fan-out point (row, col)
        bar_row = pr  # the horizontal sibling bar sits at the parent's bot row
        for child in n.children:
            if child.top is None:
                continue
            cr, cc = child.top  # child's own top (row, col)
            # vertical from parent down to the bar, across to child column, down to child
            draw.line([(pc, pr), (pc, bar_row)], fill=RED, width=3)
            draw.line([(pc, bar_row), (cc, bar_row)], fill=RED, width=3)
            draw.line([(cc, bar_row), (cc, cr)], fill=RED, width=3)

    # Name boxes.
    for n in nodes:
        if n.top is None or n.bot is None:
            continue
        left, top, right, bottom = name_box(n, a.shape[1])
        draw.rectangle([left, top, right, bottom], outline=RED, width=3)

    return img


def assemble_with_seams(
    book: str, start: int, end: int, config: seg.BookConfig, books_dir: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Rebuild a graph the way segment() does, recording each page's x-span.

    Returns the merged grid and a list of (page_index, right_edge_x) seams. Because
    merge_graphs puts the newer page on the LEFT, we track widths as we go and
    report each page's right boundary in final-image coordinates.
    """
    pages_dir = os.path.join(books_dir, book, "pages")
    grids: list[tuple[int, np.ndarray]] = []
    for i in range(start, end + 1):
        a = get_image(os.path.join(pages_dir, f"{i}.png"))
        a = seg.trim_borders(a)
        x = seg.is_tree_start_page(a, config)
        if x != -1:
            a = a[:, :x]
        grids.append((i, seg.shrink_page(a)))

    # Reproduce merge order: start page is the base (rightmost); each later page is
    # stacked to its LEFT. So the final left-to-right page order is end..start.
    merged = grids[0][1]
    graph = merged
    for _, page in grids[1:]:
        graph = seg.merge_graphs(page, graph)

    # Compute each page's right-edge x in the final image. The final width may
    # differ slightly from the sum of page widths (merge trims/pads at seams), so
    # scale the cumulative page widths to the final width for an approximate marker.
    order = [i for i, _ in grids][::-1]  # left-to-right = end..start
    widths = {i: g.shape[1] for i, g in grids}
    total = sum(widths.values())
    final_w = graph.shape[1]
    seams: list[tuple[int, int]] = []
    cum = 0
    for i in order:
        cum += widths[i]
        seams.append((i, int(cum / total * final_w)))
    return graph, seams


def raw_pages_strip(book: str, start: int, end: int, books_dir: str) -> Image.Image:
    """Join the RAW Stage-1 pages (before any trim/shrink) left-to-right.

    This is the "nothing got lost" check: it shows the full deskewed scan of each
    page, *before* trim_borders / shrink_page removed the frame and whitespace. If
    a name or line sits outside the tight box that shrink_page kept, it is still
    visible here, so comparing this strip against the parse overlay reveals any ink
    the cropping discarded. Pages are laid out in the same order as the assembly
    (right-to-left: end..start, so page numbers ascend right->left), with a blue
    boundary + page label per page.
    """
    pages_dir = os.path.join(books_dir, book, "pages")
    order = list(range(start, end + 1))[::-1]  # left-to-right = end..start
    imgs = [
        _to_rgb(get_image(os.path.join(pages_dir, f"{i}.png"))) for i in order
    ]
    h = max(im.height for im in imgs)
    strip = Image.new("RGB", (sum(im.width for im in imgs), h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    font = _font(28)
    x = 0
    for page_i, im in zip(order, imgs):
        strip.paste(im, (x, 0))
        draw.line([(x, 0), (x, h)], fill=BLUE, width=3)
        label = f"p{page_i}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        cx = x + im.width // 2
        draw.rectangle([cx - tw // 2 - 6, 6, cx + tw // 2 + 6, 44], fill=BLUE)
        draw.text((cx - tw // 2, 8), label, fill=(255, 255, 255), font=font)
        x += im.width
    return strip


def draw_assembly_overlay(
    a: np.ndarray, seams: list[tuple[int, int]]
) -> Image.Image:
    """Blue page-boundary lines + page-number labels over the graph image."""
    img = _to_rgb(a)
    draw = ImageDraw.Draw(img)
    font = _font(28)
    prev_x = 0
    for page_i, right_x in seams:
        # boundary line at the right edge of this page's band
        draw.line([(right_x, 0), (right_x, a.shape[0])], fill=BLUE, width=3)
        # page label centered in the band
        cx = (prev_x + right_x) // 2
        label = f"p{page_i}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        draw.rectangle([cx - tw // 2 - 6, 6, cx + tw // 2 + 6, 44], fill=BLUE)
        draw.text((cx - tw // 2, 8), label, fill=(255, 255, 255), font=font)
        prev_x = right_x
    return img


def qa_book(book: str, books_dir: str = "books") -> str:
    """Generate parse + assembly overlays for every graph and an HTML index."""
    seg_cfg = seg.BOOK_CONFIGS[book]
    bt_cfg = bt.BOOK_CONFIGS[book]
    graphs_dir = os.path.join(books_dir, book, "graphs")
    qa_dir = os.path.join(books_dir, book, "qa")
    os.makedirs(qa_dir, exist_ok=True)

    files = sorted(
        (f for f in os.listdir(graphs_dir) if f.endswith(".png")),
        key=lambda f: int(f.split("_")[0]),
    )
    rows: list[tuple[str, str, str, int]] = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        start, end = (int(x) for x in stem.split("_"))
        a = get_image(os.path.join(graphs_dir, fname))

        nodes = bt.parse_graph(a, bt_cfg)
        parse_img = draw_parse_overlay(a, nodes)
        parse_name = f"{stem}_parse.png"
        parse_img.save(os.path.join(qa_dir, parse_name))

        merged, seams = assemble_with_seams(book, start, end, seg_cfg, books_dir)
        pages_img = draw_assembly_overlay(merged, seams)
        pages_name = f"{stem}_pages.png"
        pages_img.save(os.path.join(qa_dir, pages_name))

        raw_img = raw_pages_strip(book, start, end, books_dir)
        raw_name = f"{stem}_raw.png"
        raw_img.save(os.path.join(qa_dir, raw_name))

        rows.append((stem, parse_name, pages_name, raw_name, len(nodes)))
        logger.info("%s: %d nodes, pages %d-%d", stem, len(nodes), start, end)

    index_path = os.path.join(qa_dir, "index.html")
    _write_index(index_path, book, rows)
    logger.info("Wrote %d graph overlays -> %s", len(rows), index_path)
    return index_path


def _write_index(
    path: str, book: str, rows: list[tuple[str, str, str, str, int]]
) -> None:
    total_nodes = sum(r[4] for r in rows)
    cards = "\n".join(
        f"""
    <section class="graph">
      <h2>{stem} <span class="count">{n} nodes</span></h2>
      <div class="pair">
        <figure><figcaption>parse: red = detected names + edges</figcaption>
          <img src="{parse}" loading="lazy"></figure>
        <figure><figcaption>assembly: blue = page seams (numbers ascend right→left)</figcaption>
          <img src="{pages}" loading="lazy"></figure>
        <figure><figcaption>raw pages: full scan before trim/shrink (nothing lost?)</figcaption>
          <img src="{raw}" loading="lazy"></figure>
      </div>
    </section>"""
        for stem, parse, pages, raw, n in rows
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parse QA — {book}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f5f4; color: #1c1917; }}
  header {{ position: sticky; top: 0; background: #1c1917; color: #fafaf9; padding: 12px 20px; }}
  header b {{ color: #f87171; }}
  .graph {{ padding: 16px 20px; border-bottom: 1px solid #d6d3d1; }}
  .graph h2 {{ margin: 0 0 8px; font-size: 18px; }}
  .count {{ color: #78716c; font-weight: normal; font-size: 14px; }}
  .pair {{ display: flex; gap: 24px; overflow-x: auto; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 12px; color: #57534e; margin-bottom: 4px; }}
  img {{ max-height: 78vh; border: 1px solid #a8a29e; background: #fff; }}
</style></head><body>
<header>Parse QA — <b>{book}</b> · {len(rows)} graphs · {total_nodes} nodes ·
  <span style="color:#f87171">red</span> = detected names/edges,
  <span style="color:#93c5fd">blue</span> = page seams</header>
{cards}
</body></html>"""
    open(path, "w").write(html)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Visual QA overlays for the parse.")
    ap.add_argument("--book", default="book1")
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    qa_book(args.book, books_dir=args.books_dir)


if __name__ == "__main__":
    main()
