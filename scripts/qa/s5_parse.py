"""Visual QA for the parse: overlay detections on the graph images.

Produces two overlays per Book-N graph, plus an HTML index to click through them:

1. **Parse overlay** (``*_parse.png``): the graph image with a RED box around each
   detected name and a RED line for each detected parent->child edge. Overlaid on
   the actual ink, a wrong box or missing/extra edge is obvious at a glance.

2. **Assembly overlay** (``*_pages.png``): the same graph with BLUE vertical bands
   marking where each source page was stitched in, labelled with the page number.
   The book reads right-to-left, so page numbers should ASCEND from right to left;
   an out-of-order or misplaced label means the page merge went wrong.

The graph images (``books/bookN/4_graphs/{start}_{end}.png``) are already the
pipeline's deskewed pages roughly joined, so this verifies name-detection,
edge-parsing, AND page assembly in one place.

Usage::

    python -m scripts.qa.s5_parse --book book1
    # -> books/book1/qa/*.png + books/book1/qa/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Book 2's 36_52 spans 17 pages (~93M px); lift PIL's decompression-bomb guard,
# which these known-safe local scans exceed.
Image.MAX_IMAGE_PIXELS = None

from src import s3_segment as seg
from src.imaging import get_image

logger = logging.getLogger(__name__)

RED = (220, 30, 30)
BLUE = (40, 90, 220)
GREEN = (20, 170, 60)
MAGENTA = (200, 30, 200)  # stray pen marks scrubbed before parsing
CYAN = (0, 170, 200)      # hairline scan-nick fills (not cross-page bridges)

# Browsers refuse to decode an image wider than ~32k px (Chrome: 32,767; Safari is
# stricter), and 36_52's compare strip is 64,719 px. Overlays wider than this are
# downscaled for the page; the on-screen scale is unchanged (display height comes
# from the ORIGINAL pixel height), and every bridge/nick also gets a full-res crop.
MAX_IMG_WIDTH = 16000
# Full-res crop around each fill: this much context beyond the fill's ends/rows.
FILL_CROP_PAD_X = 300
FILL_CROP_PAD_Y = 250
# A fill longer than this gets two end crops instead of one (the anchors matter).
FILL_CROP_MAX_SPAN = 1800
FILL_END_HALF = 600


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


ORANGE = (240, 140, 0)


def draw_parse_overlay(
    a: np.ndarray,
    parse: dict,
    imaginary: list[list[int]] | None = None,
    nicks: list[list[int]] | None = None,
) -> Image.Image:
    """Overlay the parse: red = names + edges, ORANGE = empty phantoms, GREEN = bridges.

    ``parse`` is the Stage-5 sidecar (``{stem}.parse.json``, from
    :func:`src.s5_build_tree.build_parse_sidecar``): per node its name ``box``,
    ``top``/``bot`` endpoints, ``empty`` flag, and ``children_top`` points. The QA
    draws exactly what Stage 5 produced -- no re-parsing, no geometry re-derived
    here (so the box can never drift from the actual crop).

    Empty nodes (blank name crop) are circled in orange -- the seam-break artifacts
    where a cross-page connector failed to join, so a child mis-attaches to a
    nameless endpoint instead of its true cross-page parent.

    ``imaginary`` (from ``{stem}.imaginary.json``) lists the synthetic orphan-bridge
    connectors, each ``[r0, c0, r1, c1]``, drawn GREEN for confirmation. ``nicks``
    (from ``{stem}.nicks.json``) are the hairline scan-nick fills Stage 5 made on
    the way -- a few px of one printed bar the scanner dropped -- drawn CYAN, since
    William's rule keeps green for genuine cross-page connectors only.
    """
    img = _to_rgb(a)
    draw = ImageDraw.Draw(img)
    nodes = parse.get("nodes", [])

    for r0, c0, r1, c1 in nicks or []:
        draw.line([(c0, r0), (c1, r1)], fill=CYAN, width=6)
        draw.ellipse([c0 - 10, r0 - 10, c1 + 10, r1 + 10], outline=CYAN, width=4)

    # Magenta = stray pen marks Stage 5 scrubbed before parsing. Filled semi-boldly
    # + outlined so you see exactly what was removed (the ink is already gone from
    # the graph, so this marks where it was).
    scrubbed = parse.get("scrubbed", [])
    for r0, c0, r1, c1 in scrubbed:
        draw.rectangle([c0 - 6, r0 - 6, c1 + 6, r1 + 6], outline=MAGENTA, width=8)

    # Green synthetic bridges first, so red edges/boxes sit on top where they meet.
    for r0, c0, r1, c1 in imaginary or []:
        draw.line([(c0, r0), (c1, r1)], fill=GREEN, width=6)
        draw.ellipse([c1 - 13, r1 - 13, c1 + 13, r1 + 13], outline=GREEN, width=4)
        draw.ellipse([c0 - 13, r0 - 13, c0 + 13, r0 + 13], outline=GREEN, width=4)

    # Edges: from each parent's fan-out point (bot) down to the sibling bar, across
    # to each child's column, then down into the child. The bar is drawn HALFWAY
    # between the parent's bot row and the children's top rows (rather than flush
    # with the parent) so the fan-out reads clearly. Detected endpoints -- the
    # parent fan-out point and each child top -- are circled for inspection.
    ENDPOINT_R = 22
    for n in nodes:
        pr, pc = n["bot"]
        kids = n["children_top"]
        if kids:
            # Bar sits midway between the parent bot and the shallowest child top.
            child_top = min(cr for cr, _cc in kids)
            bar_row = (pr + child_top) // 2
            draw.line([(pc, pr), (pc, bar_row)], fill=RED, width=8)
            for cr, cc in kids:
                draw.line([(pc, bar_row), (cc, bar_row)], fill=RED, width=8)
                draw.line([(cc, bar_row), (cc, cr)], fill=RED, width=8)
        # Circle only DETECTED endpoints: a parent's fan-out point (it has children
        # hanging off it) and every child-top. A leaf's ``bot`` is not a line end --
        # Stage 5 infers it by scanning down the column for the last ink, and an ADF
        # smear speck below the name drags it 100+ rows past the box (36_52 p37,
        # 毓辇/毓瑛), so circling it just draws a misleading dot under the name.
        if kids:
            draw.ellipse(
                [pc - ENDPOINT_R, pr - ENDPOINT_R, pc + ENDPOINT_R, pr + ENDPOINT_R],
                outline=RED, width=5)
        for cr, cc in kids:
            draw.ellipse(
                [cc - ENDPOINT_R, cr - ENDPOINT_R, cc + ENDPOINT_R, cr + ENDPOINT_R],
                outline=RED, width=5)

    # Name boxes (red) for real names; empty phantom nodes circled in orange.
    empty_count = 0
    for n in nodes:
        if n["empty"]:
            empty_count += 1
            tr, tc = n["top"]
            br = n["bot"][0]
            r = 40
            cy = (tr + br) // 2
            draw.ellipse([tc - r, cy - r, tc + r, cy + r], outline=ORANGE, width=8)
        else:
            left, top, right, bottom = n["box"]
            draw.rectangle([left, top, right, bottom], outline=RED, width=8)

    parts = []
    if empty_count:
        parts.append(f"{empty_count} empty (orphan) node(s)")
    if imaginary:
        parts.append(f"{len(imaginary)} green bridge(s)")
    if nicks:
        parts.append(f"{len(nicks)} cyan nick fill(s)")
    if scrubbed:
        parts.append(f"{len(scrubbed)} scrubbed pen mark(s)")
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
    pages_dir = os.path.join(books_dir, book, "1_pages")
    crops_dir = os.path.join(books_dir, book, "3_crops")
    order = list(range(start, end + 1))[::-1]  # left-to-right = end..start

    raw_cols: list[Image.Image] = []
    crop_cols: list[Image.Image] = []
    for i in order:
        # Row ② shows the ACTUAL Stage-3 crop on disk (books/{book}/3_crops/{i}.png)
        # -- what really feeds the merged graph -- NOT a re-crop. A re-crop that skips
        # whiten_margins hid a real bug: page 83's crop dropped a whole subtree (the
        # top whiten band wiped 纪培's fan-out bar -> shrink_page cut the left), but the
        # re-crop showed it intact. Loading the crop makes the QA tell the truth: a
        # cropped-out subtree shows as missing here, above its missing boxes below.
        # A page with no crop (a biography page, Books 3&4) is skipped in both rows so
        # the columns stay in step.
        crop_path = os.path.join(crops_dir, f"{i}.png")
        if not os.path.exists(crop_path):
            continue
        raw_cols.append(_to_rgb(get_image(os.path.join(pages_dir, f"{i}.png"))))
        crop_cols.append(_to_rgb(get_image(crop_path)))

    raw_h = max(im.height for im in raw_cols)
    crop_h = max(im.height for im in crop_cols)

    # The two bands pack independently so the cropped band carries no horizontal
    # column-whitespace (which would push its tree out of line with the parse
    # row below). The RAW band keeps uniform-width columns (so it reads cleanly
    # against the source and its own blue dividers), while the CROPPED band packs
    # each page at its own width, edge to edge, with a blue divider between pages.
    # Both bands are right-aligned to the composite's right edge, so the content's
    # right edge (the root spine, right-to-left) lines up straight down -- band to
    # band here, and onward to the parse row.
    col_w = max(im.width for im in raw_cols)  # uniform raw page width
    raw_band_w = col_w * len(order)
    div = 10  # blue divider width between cropped pages
    crop_band_w = sum(im.width for im in crop_cols) + div * len(order)
    W = max(raw_band_w, crop_band_w)

    gap = 130  # blue divider band between the two rows (holds the row-legend text)
    H = raw_h + gap + crop_h
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # These labels ride on the full-resolution composite (~11000px tall) but are
    # displayed at ~1 screen height, so a small font renders illegibly. Size the
    # font (and its label box) to the composite so it reads at the display scale.
    LABEL_FONT = 120
    LABEL_PAD = 24        # px of box padding around the text
    font = _font(LABEL_FONT)
    lh = LABEL_FONT + LABEL_PAD   # label-box height

    # Raw band: uniform columns, right-aligned as a block to the composite edge.
    raw_x0 = W - raw_band_w
    for idx, page_i in enumerate(order):
        x0 = raw_x0 + idx * col_w
        canvas.paste(raw_cols[idx], (x0 + col_w - raw_cols[idx].width, 0))
        draw.line([(x0, 0), (x0, raw_h)], fill=BLUE, width=6)
        label = f"p{page_i}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        cx = x0 + col_w // 2
        draw.rectangle(
            [cx - tw // 2 - LABEL_PAD, 6, cx + tw // 2 + LABEL_PAD, 6 + lh],
            fill=BLUE)
        draw.text((cx - tw // 2, 6 + LABEL_PAD // 2), label,
                  fill=(255, 255, 255), font=font)

    # Cropped band: tight-packed pages (no column whitespace) with blue dividers,
    # right-aligned as a block so its right edge matches the raw band's.
    cy = raw_h + gap
    cx = W - crop_band_w
    for idx in range(len(order)):
        draw.line([(cx, cy), (cx, H)], fill=BLUE, width=div)
        cx += div
        canvas.paste(crop_cols[idx], (cx, cy))
        # Page-number label at the top of this cropped column (matches the raw band
        # above, so a page is identifiable in row 2 without counting seams).
        page_i = order[idx]
        label = f"p{page_i}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        lx = cx + crop_cols[idx].width // 2 - tw // 2
        draw.rectangle(
            [lx - LABEL_PAD, cy + 6, lx + tw + LABEL_PAD, cy + 6 + lh], fill=BLUE)
        draw.text((lx, cy + 6 + LABEL_PAD // 2), label,
                  fill=(255, 255, 255), font=font)
        cx += crop_cols[idx].width

    draw.rectangle([0, raw_h, W, raw_h + gap], fill=BLUE)
    draw.text((8, raw_h + gap + 8), "▲ raw scan   ▼ kept after crop",
              fill=(255, 255, 255), font=_font(90))
    return canvas



def _page_seams(
    book: str, start: int, end: int, config: seg.BookConfig, books_dir: str
) -> list[tuple[int, int]]:
    """Page-seam left-x positions in graph coordinates: [(page, left_x), ...].

    Uses the ACTUAL Stage-3 crop widths on disk (books/{book}/3_crops/{p}.png), the
    same crops :mod:`src.s4_merge_pages` stacks left-to-right (order end..start), so
    the cumulative widths give the exact seam x-positions the merged graph uses. A
    page with no crop (a biography page) is skipped, matching the merge.
    """
    crops_dir = os.path.join(books_dir, book, "3_crops")
    order = list(range(start, end + 1))[::-1]
    seams: list[tuple[int, int]] = []
    acc = 0
    for p in order:
        crop_path = os.path.join(crops_dir, f"{p}.png")
        if not os.path.exists(crop_path):
            continue
        seams.append((p, acc))
        acc += get_image(crop_path).shape[1]
    return seams


def _page_of(x: int, seam_pages: list[tuple[int, int]]) -> int:
    """Page number whose column contains x (seam_pages = [(page, left_x), ...])."""
    page = seam_pages[0][0] if seam_pages else -1
    for p, sx in seam_pages:
        if x >= sx:
            page = p
    return page


def _generation_rows(nodes: list[dict]) -> list[int]:
    """Sorted distinct generation-bar y-rows (each node's top row), top-first.

    Clusters node top-rows (a generation's names/bars sit at ~the same y) so a
    bridge's y can be mapped to "gen N" (gen 1 = topmost bar). ``nodes`` are the
    parse-sidecar node dicts.
    """
    tops = sorted({n["top"][0] for n in nodes})
    rows: list[int] = []
    for t in tops:
        if not rows or t - rows[-1] > 60:  # new generation band
            rows.append(t)
    return rows


def _gen_of(y: int, gen_rows: list[int]) -> int:
    """1-indexed generation whose bar-row is nearest y (gen 1 = topmost)."""
    if not gen_rows:
        return -1
    best = min(range(len(gen_rows)), key=lambda i: abs(gen_rows[i] - y))
    return best + 1


def _card_notes(
    imaginary: list[list[int]] | None,
    n_orphans: int = 0,
    seam_pages: list[tuple[int, int]] | None = None,
    gen_rows: list[int] | None = None,
    nicks: list[list[int]] | None = None,
) -> str:
    """Searchable plain-text status line for a card: green bridges + orphans.

    Rendered as text (not only drawn on the image) so the QA page is greppable in a
    browser. Two search keywords, each appearing AT MOST ONCE per card so one Cmd-F
    Enter jumps one card:
      * "green" -- only on cards with synthetic green bridges. Each bridge is
        described by PAGE SPAN and GENERATION (interpretable) plus raw pixels: e.g.
        "gen2 p13->p12 (x3677->4827)".
      * "orphan" -- only on cards that still have >=1 orphan (empty
        node-with-children), an unresolved connectivity issue to look at.
    A clean card says neither word.
    """
    seam_pages = seam_pages or []
    gen_rows = gen_rows or []
    segs = []
    if imaginary:
        parts = []
        for y1, x1, _y2, x2 in imaginary:
            gen = _gen_of(y1, gen_rows)
            p_l, p_r = _page_of(x1, seam_pages), _page_of(x2, seam_pages)
            span = f"p{p_l}->p{p_r}" if p_l != p_r else f"p{p_l}"
            parts.append(f"gen{gen} {span} (x{x1}->{x2})")
        segs.append(f"green: {len(imaginary)} bridge(s) — " + ", ".join(parts))
    else:
        segs.append("no bridges")
    if nicks:
        segs.append(f"nick: {len(nicks)} hairline fill(s)")
    if n_orphans:
        segs.append(f"orphan: {n_orphans} unresolved")
    return " · ".join(segs)


def _fit_width(img: Image.Image) -> Image.Image:
    """Downscale ``img`` to at most MAX_IMG_WIDTH wide (LANCZOS); unchanged if narrower."""
    if img.width <= MAX_IMG_WIDTH:
        return img
    f = img.width / MAX_IMG_WIDTH
    return img.resize((MAX_IMG_WIDTH, max(1, round(img.height / f))), Image.LANCZOS)


def _fill_crops(
    parse_img: Image.Image, fills: list[list[int]], kind: str, stem: str, qa_dir: str,
    seam_pages: list[tuple[int, int]], gen_rows: list[int],
) -> list[tuple[str, str]]:
    """Full-resolution crops of the parse overlay around each fill.

    Returns ``(caption, filename)`` pairs. A short fill gets one crop spanning it;
    a long bridge gets a crop of each END (where it anchors on the seam and lands
    on the next bar -- the parts worth checking), since a 3000px span would not fit.
    """
    out: list[tuple[str, str]] = []
    w, h = parse_img.size
    for i, (r0, c0, r1, c1) in enumerate(fills, 1):
        gen = _gen_of(r0, gen_rows)
        p_l, p_r = _page_of(c0, seam_pages), _page_of(c1, seam_pages)
        span = f"p{p_l}->p{p_r}" if p_l != p_r else f"p{p_l}"
        label = f"{kind} {i}: gen{gen} {span} (x{c0}->{c1}, {c1 - c0}px)"
        top, bot = max(0, min(r0, r1) - FILL_CROP_PAD_Y), min(h, max(r0, r1) + FILL_CROP_PAD_Y)
        if c1 - c0 <= FILL_CROP_MAX_SPAN:
            boxes = [(max(0, c0 - FILL_CROP_PAD_X), min(w, c1 + FILL_CROP_PAD_X), "")]
        else:
            boxes = [(max(0, c0 - FILL_END_HALF), min(w, c0 + FILL_END_HALF), " — left end"),
                     (max(0, c1 - FILL_END_HALF), min(w, c1 + FILL_END_HALF), " — right end")]
        for j, (x0, x1, suffix) in enumerate(boxes):
            name = f"{stem}_{kind}{i}{'ab'[j] if len(boxes) > 1 else ''}.png"
            parse_img.crop((x0, top, x1, bot)).save(os.path.join(qa_dir, name))
            out.append((label + suffix, name))
    return out


def qa_book(
    book: str, books_dir: str = "books", only: set[str] | None = None
) -> str:
    """Generate parse + assembly overlays for every graph and an HTML index.

    ``only``: if given, restrict to these graph stems (e.g. ``{"58_62", "69_82"}``)
    and write a separate ``index_focus.html`` so the full index is left untouched --
    used to iterate on a few problem graphs while the rest are frozen.
    """
    seg_cfg = seg.BOOK_CONFIGS[book]
    graphs_dir = os.path.join(books_dir, book, "4_graphs")
    qa_dir = os.path.join(books_dir, book, "qa")
    os.makedirs(qa_dir, exist_ok=True)

    files = sorted(
        (f for f in os.listdir(graphs_dir) if f.endswith(".png")),
        key=lambda f: int(f.split("_")[0]),
    )
    if only:
        files = [f for f in files if os.path.splitext(f)[0] in only]
    # (stem, parse, compare, n, parse_img_height, compare_img_height, notes)
    rows: list[tuple] = []
    for fname in files:
        stem = os.path.splitext(fname)[0]
        start, end = (int(x) for x in stem.split("_"))
        a = get_image(os.path.join(graphs_dir, fname))

        # Read Stage 5's parse sidecar rather than re-parsing (single source of
        # truth; also fast). Missing sidecar => Stage 5 hasn't been run here.
        parse_path = os.path.join(graphs_dir, f"{stem}.parse.json")
        if not os.path.exists(parse_path):
            logger.warning("%s: no %s.parse.json -- run Stage 5 (s5_build_tree) "
                           "first; skipping", stem, stem)
            continue
        parse = json.load(open(parse_path))
        nodes = parse["nodes"]

        imag_path = os.path.join(graphs_dir, f"{stem}.imaginary.json")
        imaginary = json.load(open(imag_path)) if os.path.exists(imag_path) else None
        nick_path = os.path.join(graphs_dir, f"{stem}.nicks.json")
        nicks = json.load(open(nick_path)) if os.path.exists(nick_path) else None
        parse_img = draw_parse_overlay(a, parse, imaginary, nicks)
        parse_name = f"{stem}_parse.png"
        parse_h = parse_img.height
        _fit_width(parse_img).save(os.path.join(qa_dir, parse_name))

        compare_img = stacked_compare(book, start, end, seg_cfg, books_dir)
        compare_name = f"{stem}_compare.png"
        compare_h = compare_img.height
        _fit_width(compare_img).save(os.path.join(qa_dir, compare_name))

        n_orphans = sum(1 for n in nodes if n["empty"] and n["children_top"])
        # Page-seam x-positions (left->right = end..start) and generation-bar y-rows,
        # so bridge notes can read "p13" and "gen 2" instead of raw pixels.
        seam_pages = _page_seams(book, start, end, seg_cfg, books_dir)
        gen_rows = _generation_rows(nodes)
        notes = _card_notes(imaginary, n_orphans, seam_pages, gen_rows, nicks)
        fills = _fill_crops(parse_img, imaginary or [], "bridge", stem, qa_dir, seam_pages, gen_rows)
        fills += _fill_crops(parse_img, nicks or [], "nick", stem, qa_dir, seam_pages, gen_rows)
        rows.append(
            (stem, parse_name, compare_name, len(nodes), parse_h, compare_h, notes, fills)
        )
        logger.info("%s: %d nodes, pages %d-%d", stem, len(nodes), start, end)

    index_name = "index_focus.html" if only else "index.html"
    index_path = os.path.join(qa_dir, index_name)
    _write_index(index_path, book, rows)
    logger.info("Wrote %d graph overlays -> %s", len(rows), index_path)
    return index_path


def _write_index(path: str, book: str, rows: list[tuple]) -> None:
    total_nodes = sum(r[3] for r in rows)
    # EVERY image -- compare and parse, across all graphs -- is displayed at the
    # SAME source-pixel-to-screen ratio (a fixed pixels-per-vh), so a name glyph is
    # the same on-screen size in the raw page, the crop, and the parse graph, and
    # you can compare stage-to-stage side by side. Display height (vh) is simply
    # ``image_pixel_height / PX_PER_VH``. PX_PER_VH ~= a full-page compare image
    # (~11000px) at ~104vh, the size that reads well; taller graphs get taller (and
    # scroll), shorter ones get shorter, but the *scale* never changes.
    PX_PER_VH = 106.0
    # Cache-buster: a fresh version stamp each regeneration so the browser refetches
    # the overlays instead of showing stale cached PNGs (same filename, new content).
    ver = int(time.time())
    card_parts = []
    for stem, parse, compare, n, parse_ih, compare_ih, notes, fills in rows:
        # Heights come from the ORIGINAL pixel heights, so an overlay that was
        # downscaled to fit the browser's width limit still renders at the same
        # on-screen scale as every other card (just softer).
        compare_h_vh = round(compare_ih / PX_PER_VH, 1)
        parse_h_vh = round(parse_ih / PX_PER_VH, 1)
        fill_html = ""
        if fills:
            thumbs = "\n".join(
                f'<figure class="fill"><figcaption>{cap}</figcaption>'
                f'<img src="{fn}?v={ver}" loading="lazy"></figure>'
                for cap, fn in fills
            )
            fill_html = f"""
      <div class="fills"><p class="fills-title">④ full-resolution crops of each fill (green = seam bridge, cyan = hairline nick) — check both anchors</p>
        {thumbs}</div>"""
        card_parts.append(
            f"""
    <section class="graph">
      <h2>{stem} <span class="count">{n} nodes</span></h2>
      <p class="notes">{notes}</p>
      <figure><figcaption>① raw scan (top) → ② kept after crop (bottom) — same page columns; scan down to confirm nothing lost</figcaption>
        <img class="compare" src="{compare}?v={ver}" style="height:{compare_h_vh}vh" loading="lazy"></figure>
      <figure><figcaption>③ parse: red = detected names + edges (same scale as rows ①②)</figcaption>
        <img class="parse" src="{parse}?v={ver}" style="height:{parse_h_vh}vh" loading="lazy"></figure>{fill_html}
    </section>"""
        )
    cards = "\n".join(card_parts)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Parse QA — {book}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f5f4; color: #1c1917; }}
  header {{ position: sticky; top: 0; background: #1c1917; color: #fafaf9; padding: 12px 20px; }}
  header b {{ color: #f87171; }}
  .graph {{ padding: 16px 20px; border-bottom: 1px solid #d6d3d1; }}
  .graph h2 {{ margin: 0 0 8px; font-size: 18px; }}
  .count {{ color: #78716c; font-weight: normal; font-size: 14px; }}
  /* Searchable text mirror of the green bridges drawn on the image, so the page is
     greppable in-browser (Cmd-F "green" jumps between bridged graphs). */
  .notes {{ margin: 0 0 10px; font-size: 13px; color: #166534; font-family:
    ui-monospace, SFMono-Regular, Menlo, monospace; }}
  /* Right-align both rows: the root spine sits at the right edge of both images
     (the whitespace/page-margin is all on the LEFT), so pinning the images to the
     right lines the trees up straight down. IMPORTANT: use `margin-left:auto` on
     the image itself, NOT `align-items:flex-end` on the flex container — with an
     overflowing item, flex-end alignment pushes the item's LEFT edge off the
     scrollable area, so a wider-than-viewport graph (e.g. 36_52) cannot be
     scrolled back to see its left side. A block figure with overflow-x:auto and
     an auto left margin keeps the right-alignment AND leaves the full width
     scrollable in both directions. */
  figure {{ margin: 0 0 16px; overflow-x: auto; }}
  figcaption {{ font-size: 12px; color: #57534e; margin-bottom: 4px; }}
  /* Both rows keep their natural width (height is set per-card inline). The parse
     overlay and the page-compare derive from the same scans at the same DPI, so
     showing them at the same source-pixel-to-screen scale makes a name glyph (and
     the whole tree) render the same size in both — the per-card parse height is
     compare_height x (parse_img_px / compare_img_px). See _write_index.
     `margin-left:auto` right-aligns a narrower-than-viewport image while keeping a
     wider one fully scrollable. */
  img {{ border: 1px solid #a8a29e; background: #fff; display: block; max-width: none;
    margin-left: auto; }}
  /* Full-res fill crops: natural pixels, wrapped left-to-right, capped in height. */
  .fills {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
  .fills-title {{ flex-basis: 100%; margin: 0; font-size: 12px; color: #57534e; }}
  .fill {{ margin: 0; overflow: visible; }}
  .fill img {{ margin-left: 0; height: 260px; width: auto; }}
</style></head><body>
<header>Parse QA — <b>{book}</b> · {len(rows)} graphs · {total_nodes} nodes ·
  <span style="color:#f87171">red</span> = detected names/edges,
  <span style="color:#f08c00">orange</span> = empty orphan node,
  <span style="color:#22c55e">green</span> = synthetic orphan-bridge,
  <span style="color:#00aac8">cyan</span> = hairline nick fill,
  <span style="color:#c81ec8">magenta</span> = scrubbed pen mark,
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
