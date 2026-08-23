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
import json
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
GREEN = (20, 170, 60)


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


ORANGE = (240, 140, 0)


def _is_empty_node(node: bt.LineNode, a: np.ndarray) -> bool:
    """True if the node's name crop is essentially blank — a phantom node.

    These are the seam-break artifacts: a horizontal connector line broken across
    a page seam leaves a tiny endpoint that reads as a node, but no name sits
    there. We flag them so the merge failures are visible at a glance.
    """
    if node.top is None or node.bot is None:
        return True
    return int((1 - bt.get_name_image(node, a)).sum()) < 30


def draw_parse_overlay(
    a: np.ndarray,
    nodes: list[bt.LineNode],
    imaginary: list[list[int]] | None = None,
) -> Image.Image:
    """Overlay the parse: red = names + edges, ORANGE = empty phantoms, GREEN = bridges.

    Empty nodes (blank name crop) are circled in orange — the seam-break artifacts
    where a cross-page connector failed to join, so a child mis-attaches to a
    nameless endpoint instead of its true cross-page parent.

    ``imaginary`` (from the graph's ``{stem}.imaginary.json`` sidecar) lists the
    synthetic connectors the orphan-bridge pass drew, each ``[r0, c0, r1, c1]``.
    They are drawn in GREEN so a human can confirm every invented line joins the
    right two fragments.
    """
    img = _to_rgb(a)
    draw = ImageDraw.Draw(img)

    # Green synthetic bridges first, so red edges/boxes sit on top where they meet.
    for r0, c0, r1, c1 in imaginary or []:
        draw.line([(c0, r0), (c1, r1)], fill=GREEN, width=6)
        draw.ellipse([c1 - 13, r1 - 13, c1 + 13, r1 + 13], outline=GREEN, width=4)
        draw.ellipse([c0 - 13, r0 - 13, c0 + 13, r0 + 13], outline=GREEN, width=4)

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

    # Name boxes (red) for real names; empty phantom nodes circled in orange.
    empty_count = 0
    for n in nodes:
        if n.top is None or n.bot is None:
            continue
        left, top, right, bottom = name_box(n, a.shape[1])
        if _is_empty_node(n, a):
            empty_count += 1
            r = 34
            cx, cy = n.top[1], (n.top[0] + n.bot[0]) // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ORANGE, width=5)
        else:
            draw.rectangle([left, top, right, bottom], outline=RED, width=3)

    parts = []
    if empty_count:
        parts.append(f"{empty_count} empty (orphan) node(s)")
    if imaginary:
        parts.append(f"{len(imaginary)} green bridge(s)")
    if parts:
        draw.text(
            (10, 10),
            "   ".join(parts),
            fill=ORANGE if empty_count else GREEN,
            font=_font(34),
        )
    return img


def stacked_compare(
    book: str, start: int, end: int, config: seg.BookConfig, books_dir: str
) -> Image.Image:
    """Raw pages (top) stacked over the cropped pages (bottom), SAME scale.

    The "nothing lost in the crop" check made easy to read: each page occupies the
    same horizontal column in both rows, at identical scale, so you scan straight
    down one column to compare the full raw scan against what the crop kept. The
    cropped page is RIGHT-aligned in its column (matching the right-to-left reading
    direction, so the content edges line up for a straight-down comparison even
    when a page's crop drifted or shrank) and is naturally narrower/shorter — that
    shrinkage is exactly the whitespace the crop removed; any *name or line*
    missing from the bottom half that's present up top is real lost data.

    Pages run right-to-left (page numbers ascend right→left), matching the graph.
    """
    pages_dir = os.path.join(books_dir, book, "pages")
    order = list(range(start, end + 1))[::-1]  # left-to-right = end..start

    raw_cols: list[Image.Image] = []
    crop_cols: list[Image.Image] = []
    for i in order:
        a = get_image(os.path.join(pages_dir, f"{i}.png"))
        raw_cols.append(_to_rgb(a))
        # Apply the same crop the pipeline does (trim, label-strip, shrink).
        a = seg.trim_borders(a)
        x = seg.is_tree_start_page(a, config)
        if x != -1:
            a = a[:, :x]
        crop_cols.append(_to_rgb(seg.shrink_page(a)))

    col_w = max(im.width for im in raw_cols)  # uniform raw page width
    raw_h = max(im.height for im in raw_cols)
    crop_h = max(im.height for im in crop_cols)

    gap = 20  # blue divider band between the two rows
    W = col_w * len(order)
    H = raw_h + gap + crop_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(30)
    for idx, page_i in enumerate(order):
        x0 = idx * col_w
        # Same scale. Raw is right-aligned in its column and the cropped page is
        # right-aligned to match, so the content's right edge lines up in both
        # rows (right-to-left reading direction) even when a page drifted/shrank.
        canvas.paste(raw_cols[idx], (x0 + col_w - raw_cols[idx].width, 0))
        canvas.paste(crop_cols[idx], (x0 + col_w - crop_cols[idx].width, raw_h + gap))
        draw.line([(x0, 0), (x0, H)], fill=BLUE, width=3)
        label = f"p{page_i}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        cx = x0 + col_w // 2
        draw.rectangle([cx - tw // 2 - 7, 6, cx + tw // 2 + 7, 48], fill=BLUE)
        draw.text((cx - tw // 2, 8), label, fill=(255, 255, 255), font=font)
    draw.rectangle([0, raw_h, W, raw_h + gap], fill=BLUE)
    draw.text((8, raw_h + gap + 2), "▲ raw scan   ▼ kept after crop", fill=(255, 255, 255), font=_font(22))
    return canvas



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
    rows: list[tuple[str, str, str, int]] = []  # (stem, parse, compare, n)
    for fname in files:
        stem = os.path.splitext(fname)[0]
        start, end = (int(x) for x in stem.split("_"))
        a = get_image(os.path.join(graphs_dir, fname))
        a = bt.apply_ignore_regions(a, stem, bt_cfg)

        nodes = bt.parse_graph(a, bt_cfg, graph_stem=stem)
        sidecar = os.path.join(graphs_dir, f"{stem}.imaginary.json")
        imaginary = json.load(open(sidecar)) if os.path.exists(sidecar) else None
        parse_img = draw_parse_overlay(a, nodes, imaginary)
        parse_name = f"{stem}_parse.png"
        parse_img.save(os.path.join(qa_dir, parse_name))

        compare_img = stacked_compare(book, start, end, seg_cfg, books_dir)
        compare_name = f"{stem}_compare.png"
        compare_img.save(os.path.join(qa_dir, compare_name))

        rows.append((stem, parse_name, compare_name, len(nodes)))
        logger.info("%s: %d nodes, pages %d-%d", stem, len(nodes), start, end)

    index_path = os.path.join(qa_dir, "index.html")
    _write_index(index_path, book, rows)
    logger.info("Wrote %d graph overlays -> %s", len(rows), index_path)
    return index_path


def _write_index(
    path: str, book: str, rows: list[tuple[str, str, str, int]]
) -> None:
    total_nodes = sum(r[3] for r in rows)
    cards = "\n".join(
        f"""
    <section class="graph">
      <h2>{stem} <span class="count">{n} nodes</span></h2>
      <figure><figcaption>① raw scan (top) → ② kept after crop (bottom) — same page columns; scan down to confirm nothing lost</figcaption>
        <img class="compare" src="{compare}" loading="lazy"></figure>
      <figure><figcaption>③ parse: red = detected names + edges</figcaption>
        <img class="parse" src="{parse}" loading="lazy"></figure>
    </section>"""
        for stem, parse, compare, n in rows
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
  figure {{ margin: 0 0 16px; overflow-x: auto; }}
  figcaption {{ font-size: 12px; color: #57534e; margin-bottom: 4px; }}
  img {{ border: 1px solid #a8a29e; background: #fff; display: block; }}
  /* Render BOTH rows at the same width (the figure/viewport width) so their
     generation columns and the right-hand root lineage line up straight down for
     comparison. Matching width (not height) is what aligns them: the two images
     cover the same horizontal generation-span, so equal width => same columns.
     Height follows from each image's own aspect ratio and the page scrolls. */
  img.compare, img.parse {{ width: 100%; height: auto; }}
</style></head><body>
<header>Parse QA — <b>{book}</b> · {len(rows)} graphs · {total_nodes} nodes ·
  <span style="color:#f87171">red</span> = detected names/edges,
  <span style="color:#f08c00">orange</span> = empty orphan node,
  <span style="color:#22c55e">green</span> = synthetic orphan-bridge,
  <span style="color:#93c5fd">blue</span> = raw-vs-cropped page columns</header>
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
