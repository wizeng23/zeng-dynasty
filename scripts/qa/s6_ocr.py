"""Local OCR-review tool: a character filmstrip for correcting the OCR readings.

Serves a single-page reviewer at http://localhost:PORT/. Every parsed name across
Book 1 and Book 2 is broken into its individual characters (a 2-char name -> two
cells read LEFT-to-RIGHT), and all characters from all nodes are laid end to end
into one long filmstrip. The top row shows each character's raw crop; the bottom
row is a same-size text box holding the OCR reading, which you correct with your
own IME. The strip stays put and slides: -> moves focus to the next character
(shifting the strip left one cell), <- the reverse.

Colors tell you at a glance: amber = low-confidence OCR, green = you changed it
(an override), neutral = untouched OCR. Enter saves the focused cell + advances;
Escape reverts the focused cell to the OCR reading. Your position is remembered.

Corrections are a SEPARATE ground-truth layer, keyed by ``{provenance}#{charIndex}``
(0-based), written to ``data/{book}_overrides.json``. Re-running OCR never touches
them; ``src.s6_ocr.apply_names`` reassembles each node's name from its per-character
overrides (falling back to the OCR character where there is no override), so an
override always wins.

Character splitting uses the crop's aspect ratio: names are single glyphs stacked
vertically in ~square cells, so a crop's ``round(height / width)`` gives the
character count (verified exact on all 163 Book 1 crops). The crop is cut into
that many equal horizontal bands, refined to the nearest interior whitespace
valley. When the aspect-ratio count disagrees with the OCR character count, the
node is flagged (``count_mismatch``) so you can eyeball it.

Run::

    python -m scripts.qa.s6_ocr              # then open http://localhost:8766/

Nothing here writes into ``{book}.jsonl`` -- run ``src.s6_ocr.apply_names`` afterward
to fold overrides + OCR into the tree.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

BOOKS = ["book1", "book2"]
DATA_DIR = "data"
BOOKS_DIR = "books"


def _page_ranks(book: str) -> dict[str, tuple[int, int]]:
    """Map each node provenance -> (page number, 1-based rank ON that page), cached.

    Reads ``data/{book}_page_ranks.json`` if it is newer than every graph png (and
    this source file); otherwise recomputes via :func:`_compute_page_ranks` and
    rewrites the cache. This keeps tool startup instant after the first run while
    staying correct when graphs are re-segmented.
    """
    cache_path = os.path.join(DATA_DIR, f"{book}_page_ranks.json")
    graphs_dir = os.path.join(BOOKS_DIR, book, "v0", "graphs")
    if not os.path.isdir(graphs_dir):
        return {}
    # Cache is valid iff it is newer than every graph png AND this script.
    newest_src = os.path.getmtime(__file__)
    for f in os.listdir(graphs_dir):
        if f.endswith(".png"):
            newest_src = max(newest_src, os.path.getmtime(os.path.join(graphs_dir, f)))
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= newest_src:
        try:
            raw = json.load(open(cache_path))
            return {k: (v[0], v[1]) for k, v in raw.items()}
        except Exception:
            pass  # corrupt cache -> recompute
    ranks = _compute_page_ranks(book)
    try:
        json.dump(
            {k: [v[0], v[1]] for k, v in ranks.items()},
            open(cache_path, "w"),
        )
    except Exception:
        pass  # non-fatal: just means we recompute next time
    return ranks


def _compute_page_ranks(book: str) -> dict[str, tuple[int, int]]:
    """Parse each graph to map provenance -> (page, 1-based reading-order rank).

    Recovers node x-columns, maps x to a page via the graph's page seams (cumulative
    shrunk-page widths, the same assembly src.s3_segment uses), and ranks nodes within
    each page right-to-left (descending x = eldest first). Returns {} for a book
    whose graphs can't be parsed (the caller falls back to the provenance index).
    """
    import src.v0.segment as seg
    import src.v0.build_tree as bt

    out: dict[str, tuple[int, int]] = {}
    seg_cfg = seg.BOOK_CONFIGS.get(book)
    bt_cfg = bt.BOOK_CONFIGS.get(book, bt.BookConfig())
    graphs_dir = os.path.join(BOOKS_DIR, book, "v0", "graphs")
    pages_dir = os.path.join(BOOKS_DIR, book, "v0", "pages")
    if seg_cfg is None or not os.path.isdir(graphs_dir):
        return out
    from src.imaging import get_image

    for fname in os.listdir(graphs_dir):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]
        try:
            start, end = (int(x) for x in stem.split("_"))
        except ValueError:
            continue
        # Page seams: cumulative widths of the cropped+shrunk pages, left->right
        # order end..start (matching segment's assembly, incl. CROP_KEEP_LEFT).
        order = list(range(end, start - 1, -1))
        seams: list[tuple[int, int]] = []
        acc = 0
        try:
            for p in order:
                seams.append((p, acc))
                a = get_image(os.path.join(pages_dir, f"{p}.png"))
                a = seg.trim_borders(a)
                cut = seg.is_tree_start_page(a, seg_cfg)
                if cut != -1:
                    a = a[:, :cut]
                keep_left = seg.CROP_KEEP_LEFT.get(book, {}).get(p)
                a = seg.shrink_page(a, keep_left=keep_left)
                acc += a.shape[1]
            g = get_image(os.path.join(graphs_dir, fname))
            g = bt.apply_ignore_regions(g, stem, bt_cfg)
            nodes = bt.parse_graph(g, bt_cfg, graph_stem=stem)
        except Exception:
            continue  # skip a graph we can't parse; caller falls back
        # nodes carry (top row, col); provenance index = their position in this list.
        def page_of(x: int) -> int:
            pg = seams[0][0]
            for p, sx in seams:
                if x >= sx:
                    pg = p
            return pg
        per_page: dict[int, list[tuple[int, int]]] = {}
        for li, node in enumerate(nodes):
            if node.top is None:
                continue
            x = node.top[1]
            per_page.setdefault(page_of(x), []).append((x, li))
        for pg, items in per_page.items():
            # reading order on the page = right-to-left (descending x), eldest first
            for rank, (_x, li) in enumerate(sorted(items, key=lambda t: -t[0]), start=1):
                out[f"{stem}_{li}"] = (pg, rank)
    return out


def _provenance(notes: str) -> str:
    """The stable ``{graph}_{localindex}`` key = first ' | '-segment of notes."""
    return notes.split(" | ")[0] if notes else ""


def pinyin_of(text: str) -> str:
    """Tone-marked pinyin for a (short) Chinese string; '' if unavailable.

    Uses pypinyin when present; degrades to empty string so the reviewer still
    works without it.
    """
    if not text:
        return ""
    try:
        from pypinyin import Style, pinyin
    except ImportError:
        return ""
    return " ".join(p[0] for p in pinyin(text, style=Style.TONE))


def _ink_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) bounding box of the crop's actual ink.

    The name crops carry a uniform ~30px whitespace pad on every side (Stage 5's
    NAME_TRIM_PAD). All character-count / split geometry must ignore that pad and
    work on the ink itself; the pad is only for display.
    """
    a = np.asarray(image.convert("L"))
    ink = a < 128
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, image.width, image.height)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def char_count(image: Image.Image) -> int:
    """Estimate the number of stacked characters from the crop's aspect ratio.

    Names are single glyphs in ~square cells stacked vertically, so the count is
    ``round(ink_height / ink_width)``. The ratio is measured on the INK bounding
    box, not the padded crop: the ~30px pad on each side widens the crop and drags
    the ratio down, so a padded 3-char name (e.g. 宪七郎, ink ~136x451 -> 3.32, but
    padded 196x486 -> 2.48) would misround to 2 and mis-split.
    """
    l, t, r, b = _ink_bbox(image)
    iw, ih = r - l, b - t
    if iw <= 0:
        return 1
    return max(1, round(ih / iw))


def split_bands(image: Image.Image, n: int) -> list[tuple[int, int]]:
    """Return ``n`` (top, bottom) row-bands splitting a stacked crop into chars.

    Starts from ``n`` equal bands over the inked rows, then nudges each interior
    boundary to the nearest local whitespace valley so cuts fall between glyphs.
    """
    a = np.asarray(image.convert("L"))
    ink = (a < 128).sum(axis=1)
    rows = [r for r in range(len(ink)) if ink[r] > 0]
    if not rows or n <= 1:
        return [(0, a.shape[0])]
    top, bot = rows[0], rows[-1]
    span = bot - top
    cuts = []
    for k in range(1, n):
        guess = top + round(span * k / n)
        # search +/-15% of a band for the emptiest row near the guess
        window = max(4, span // (n * 3))
        lo, hi = max(top + 1, guess - window), min(bot - 1, guess + window)
        best = min(range(lo, hi + 1), key=lambda r: ink[r]) if hi > lo else guess
        cuts.append(best)
    bounds = [0, *cuts, a.shape[0]]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _overrides_path(book: str) -> str:
    return os.path.join(DATA_DIR, f"{book}_overrides.json")


def _load_overrides(book: str) -> dict[str, str]:
    path = _overrides_path(book)
    return json.load(open(path)) if os.path.exists(path) else {}


def save_override(book: str, key: str, name: str) -> None:
    """Persist one per-character correction ``{provenance}#{charIndex}`` -> char.

    An empty ``name`` removes the override (revert that character to OCR).
    """
    overrides = _load_overrides(book)
    if name:
        overrides[key] = name
    else:
        overrides.pop(key, None)
    with open(_overrides_path(book), "w") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2, sort_keys=True)


def _flags_path(book: str) -> str:
    return os.path.join(DATA_DIR, f"{book}_flags.json")


def _load_flags(book: str) -> set[str]:
    """Return the set of node provenances flagged 'impossible' (unresolvable name).

    Persisted server-side (not just in the browser) because it is a real annotation
    about the data -- names whose correct digital character couldn't be determined
    and should be revisited -- separate from the reviewer's session progress.
    """
    path = _flags_path(book)
    if not os.path.exists(path):
        return set()
    return set(json.load(open(path)))


def save_flag(book: str, prov: str, flagged: bool) -> None:
    """Set or clear the 'impossible' flag for a node (by provenance)."""
    flags = _load_flags(book)
    if flagged:
        flags.add(prov)
    else:
        flags.discard(prov)
    with open(_flags_path(book), "w") as f:
        json.dump(sorted(flags), f, ensure_ascii=False, indent=2)


def build_cells() -> list[dict]:
    """One row per CHARACTER across all books, in book -> id -> char order.

    Each cell: book, id, provenance, char_index, crop_url (band), ocr_char,
    confidence (node-level), low_conf, override, count_mismatch.
    """
    cells: list[dict] = []
    for book in BOOKS:
        jsonl = os.path.join(DATA_DIR, f"{book}.jsonl")
        if not os.path.exists(jsonl):
            continue
        # Skip a book whose v1 name crops haven't been generated yet (Stage 5 not
        # re-run): its cells would all show broken images. It reappears once
        # books/{book}/5_names/ is populated.
        names_dir = os.path.join(BOOKS_DIR, book, "5_names")
        if not (os.path.isdir(names_dir) and os.listdir(names_dir)):
            logger.info("skipping %s: no v1 crops in %s", book, names_dir)
            continue
        sidecar_path = os.path.join(DATA_DIR, f"{book}_names.json")
        sidecar = json.load(open(sidecar_path)) if os.path.exists(sidecar_path) else {}
        overrides = _load_overrides(book)
        flags = _load_flags(book)
        page_ranks = _page_ranks(book)
        nodes = [json.loads(line) for line in open(jsonl) if line.strip()]
        nodes.sort(key=lambda n: n["id"])
        # Per-page name counter: nodes are in id order (= reading order), so numbering
        # 1..n per page as they appear here matches the order names show in the tool.
        page_seen: dict[str, int] = {}
        for n in nodes:
            prov = _provenance(n.get("notes", ""))
            entry = sidecar.get(str(n["id"]), {})
            ocr_name = entry.get("name", n.get("name", ""))
            crop = n["name_images"][0] if n.get("name_images") else ""
            crop_name = os.path.basename(crop) if crop else ""
            # Determine character count from the crop aspect ratio; fall back to
            # the OCR length when there's no crop. Read the v1 crop at the path the
            # jsonl records (books/{book}/5_names/{id}.png) -- NOT the archived v0
            # crops.
            n_by_ratio = None
            if crop and os.path.exists(crop):
                n_by_ratio = char_count(Image.open(crop))
            n_chars = n_by_ratio or max(1, len(ocr_name))
            mismatch = bool(ocr_name) and n_by_ratio is not None and n_by_ratio != len(ocr_name)
            # Page from the parse (_page_ranks gives the correct page per node); the
            # NAME NUMBER is just a running 1..n counter per page in tool order.
            pr = page_ranks.get(prov)
            if pr is not None:
                pages = str(pr[0])
            else:
                m = prov.split("_")
                pages = ""
                if len(m) >= 2 and m[0].isdigit() and m[1].isdigit():
                    pages = m[0] if m[0] == m[1] else f"{m[0]}-{m[1]}"
            if pages:
                page_rank = page_seen.get(pages, 0) + 1
                page_seen[pages] = page_rank
            else:
                page_rank = None
            for ci in range(n_chars):
                key = f"{prov}#{ci}"
                ocr_char = ocr_name[ci] if ci < len(ocr_name) else ""
                cells.append(
                    {
                        "book": book,
                        "id": n["id"],
                        "provenance": prov,
                        "pages": pages,
                        "page_rank": page_rank,
                        "char_index": ci,
                        "n_chars": n_chars,
                        "crop_url": (
                            f"/crop/{book}/{crop_name}?i={ci}&n={n_chars}"
                            if crop_name
                            else ""
                        ),
                        "ocr_char": ocr_char,
                        "confidence": entry.get("confidence"),
                        "low_conf": entry.get("low_conf", False),
                        "override": overrides.get(key, ""),
                        "count_mismatch": mismatch,
                        "flagged": prov in flags,
                    }
                )
    return cells


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Review</title>
<style>
  :root {
    --bg:#f7f5f0; --fg:#1a1a1a; --muted:#6b6b6b; --card:#fff; --line:#e3ddd2;
    --focus:#8a1f1f; --low:#c25a00; --low-bg:#fbe9d6; --ovr:#2f7d32; --ovr-bg:#dcefdc;
    --flag:#8a2be2; --flag-bg:#efe4fb;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#1a1815; --fg:#ececec; --muted:#9a9a9a; --card:#252220; --line:#3a352f;
      --focus:#e07a7a; --low:#e0913a; --low-bg:#3a2a18; --ovr:#7bc47f; --ovr-bg:#1e2e1e;
      --flag:#b57bff; --flag-bg:#2a1e3a; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg);
    color:var(--fg); height:100vh; display:flex; flex-direction:column; overflow:hidden; }
  header { display:flex; justify-content:space-between; align-items:center;
    padding:12px 20px; border-bottom:1px solid var(--line); flex:0 0 auto; }
  .meta { color:var(--muted); font-size:14px; font-variant-numeric:tabular-nums; }
  .legend { display:flex; gap:14px; font-size:12px; align-items:center; }
  .swatch { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px;
    vertical-align:middle; }
  label.filter { font-size:13px; color:var(--muted); cursor:pointer; user-select:none; }
  .stage { flex:1 1 auto; display:flex; flex-direction:column; align-items:center;
    justify-content:center; overflow:hidden; }
  .strip { display:flex; align-items:center; transition:transform .12s ease-out;
    will-change:transform; }
  .cell { flex:0 0 auto; width:var(--cw); margin:0 6px; display:flex; flex-direction:column;
    align-items:center; gap:8px; opacity:.5; transition:opacity .12s, transform .12s; }
  .cell.focus { opacity:1; transform:scale(1.12); }
  .cropbox { width:var(--cw); height:var(--cw); background:#fff; border:2px solid var(--line);
    border-radius:8px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .cropbox img { max-width:100%; max-height:100%; image-rendering:-webkit-optimize-contrast; }
  .txt { width:var(--cw); height:var(--cw); text-align:center; border:2px solid var(--line);
    border-radius:8px; background:var(--card); color:var(--fg);
    font-family:"Noto Serif SC",ui-serif,serif; }
  .txt { font-size:calc(var(--cw) * 0.5); line-height:var(--cw); }
  input.txt { padding:0; }
  input.txt:focus { outline:none; border-color:var(--focus); }
  .cell.low .cropbox, .cell.low .txt { border-color:var(--low); background:var(--low-bg); }
  .cell.low .cropbox { background:#fff; }
  .cell.ovr .txt { border-color:var(--ovr); background:var(--ovr-bg); }
  .cell.mismatch .cropbox { box-shadow:0 0 0 2px var(--low) inset; }
  /* reviewed = you pressed Enter (confirmed) on this cell */
  .cell.reviewed .cropbox::after { content:"✓"; position:absolute; margin:-6px 0 0 -6px;
    color:var(--ovr); font-size:20px; font-weight:700; }
  .cell.reviewed .cropbox { position:relative; }
  .cell.reviewed { opacity:.7; }
  /* flagged = 'impossible' (can't determine the digital character); revisit later.
     Independent of reviewed. Purple frame + ⚑ badge. */
  .cell.flagged .cropbox, .cell.flagged .txt { border-color:var(--flag); }
  .cell.flagged .txt { background:var(--flag-bg); }
  .cell.flagged .cropbox { position:relative; }
  .cell.flagged .cropbox::before { content:"⚑"; position:absolute; top:2px; right:6px;
    color:var(--flag); font-size:22px; font-weight:700; }
  /* Status band = the obvious per-cell state stripe at the very top of the cell.
     Green = reviewed, purple = flagged, both = split. Neutral otherwise. */
  .status { width:var(--cw); height:26px; border-radius:6px; display:flex;
    align-items:center; justify-content:center; font-size:12px; font-weight:700;
    letter-spacing:.03em; color:var(--muted); background:var(--card);
    border:1px solid var(--line); overflow:hidden; }
  .status .half { flex:1; height:100%; display:flex; align-items:center;
    justify-content:center; }
  .cell.reviewed .status { color:#fff; background:var(--ovr); border-color:var(--ovr); }
  .cell.flagged .status { color:#fff; background:var(--flag); border-color:var(--flag); }
  .cell.reviewed.flagged .status { background:none; padding:0; font-size:9.5px;
    letter-spacing:0; }
  .cell.reviewed.flagged .status .rev { background:var(--ovr); color:#fff; }
  .cell.reviewed.flagged .status .flg { background:var(--flag); color:#fff; }
  .status .half:first-child { border-radius:6px 0 0 6px; }
  .status .half:last-child { border-radius:0 6px 6px 0; }
  .tag { font-size:11px; line-height:1.35; color:var(--muted);
    font-variant-numeric:tabular-nums; min-height:74px; text-align:center; }
  .py { font-size:13px; color:var(--muted); height:18px; letter-spacing:.02em;
    font-family:ui-sans-serif,system-ui,sans-serif; }
  .cell.focus .py { color:var(--fg); font-weight:600; }
  .footer { flex:0 0 auto; padding:10px 20px; border-top:1px solid var(--line);
    display:flex; justify-content:space-between; align-items:center; }
  .hint { color:var(--muted); font-size:13px; }
  .saved { color:var(--ovr); font-size:13px; min-width:120px; text-align:right; }
  :root { --cw: 120px; }
</style>
</head>
<body>
  <header>
    <div class="meta" id="progress">…</div>
    <div class="legend">
      <span><span class="swatch" style="background:var(--low)"></span>low-conf</span>
      <span><span class="swatch" style="background:var(--ovr)"></span>edited</span>
      <span><span class="swatch" style="background:var(--low);box-shadow:0 0 0 2px var(--low) inset"></span>count mismatch</span>
      <span><span class="swatch" style="background:var(--ovr)"></span>✓ reviewed</span>
      <span><span class="swatch" style="background:var(--flag)"></span>⚑ flagged</span>
      <label class="filter"><input type="checkbox" id="onlyLow"> only low-conf</label>
      <label class="filter"><input type="checkbox" id="onlyFlag"> only flagged</label>
    </div>
  </header>
  <div class="stage"><div class="strip" id="strip"></div></div>
  <div class="footer">
    <span class="hint">← / → move (no review) · type to edit · Enter = confirm ✓ + next · Esc revert · Alt+F = flag ⚑ (impossible, revisit)</span>
    <span class="saved" id="saved"></span>
  </div>

<script>
let cells = [];
let idx = 0;
let onlyLow = false;
let onlyFlag = false;
const KEY = "ocr_review_char_pos";
const REVIEWED_KEY = "ocr_review_reviewed";  // per-book set of confirmed cell keys
const WINDOW = 8; // cells rendered each side of focus
const $ = (id) => document.getElementById(id);

// A cell is 'reviewed' only once you press Enter (confirm) on it -- cells you
// filter/scroll past without confirming never count. Keyed by book+provenance+char
// so it survives reloads and re-parses.
let reviewed = new Set();
try { reviewed = new Set(JSON.parse(localStorage.getItem(REVIEWED_KEY) || "[]")); } catch {}
const cellKey = (c) => `${c.book}:${c.provenance}#${c.char_index}`;
function markReviewed(c) {
  reviewed.add(cellKey(c));
  try { localStorage.setItem(REVIEWED_KEY, JSON.stringify([...reviewed])); } catch {}
}

function visible() {
  return cells.map((c,i)=>i).filter(i =>
    (!onlyLow || cells[i].low_conf) && (!onlyFlag || cells[i].flagged));
}

function cellClasses(c, focused) {
  let k = "cell";
  if (c.low_conf) k += " low";
  if (c.override) k += " ovr";
  if (c.count_mismatch) k += " mismatch";
  if (reviewed.has(cellKey(c))) k += " reviewed";
  if (c.flagged) k += " flagged";
  if (focused) k += " focus";
  return k;
}

function render() {
  const strip = $("strip");
  strip.innerHTML = "";
  const vis = visible();
  const pos = vis.indexOf(idx);
  const from = Math.max(0, pos - WINDOW), to = Math.min(vis.length - 1, pos + WINDOW);
  for (let p = from; p <= to; p++) {
    const i = vis[p];
    const c = cells[i];
    const focused = (i === idx);
    const el = document.createElement("div");
    el.className = cellClasses(c, focused);
    const cur = c.override || c.ocr_char || "";
    let tag = "&nbsp;";
    if (focused) {
      const bookLbl = c.book === "book1" ? "Book 1" : (c.book === "book2" ? "Book 2" : c.book);
      let pageLbl = c.pages ? ("page " + c.pages) : "page ?";
      if (c.page_rank != null) pageLbl += ", name " + c.page_rank;
      const scoreLbl = (c.confidence == null) ? "score —"
                       : ("score " + Number(c.confidence).toFixed(2));
      const lines = [bookLbl, pageLbl];
      if (c.n_chars > 1) lines.push("char " + (c.char_index+1) + "/" + c.n_chars);
      lines.push(scoreLbl);
      tag = lines.join("<br>");
    }
    const rev = reviewed.has(cellKey(c)), flg = c.flagged;
    let status;
    if (rev && flg) status = `<span class="half rev">✓&nbsp;REVIEWED</span><span class="half flg">⚑&nbsp;FLAGGED</span>`;
    else if (rev) status = "✓ REVIEWED";
    else if (flg) status = "⚑ FLAGGED";
    else status = "unreviewed";
    el.innerHTML =
      `<div class="status">${status}</div>` +
      `<div class="tag">${tag}</div>` +
      `<div class="cropbox">${c.crop_url ? `<img src="${c.crop_url}">` : ""}</div>` +
      (focused
        ? `<input class="txt" id="focusInput" value="${cur.replace(/"/g,'&quot;')}"
             autocomplete="off" autocapitalize="off" spellcheck="false" lang="zh">`
        : `<div class="txt">${cur}</div>`) +
      `<div class="py" data-char="${cur.replace(/"/g,'&quot;')}">&nbsp;</div>`;
    strip.appendChild(el);
  }
  updatePinyin();
  // center the focused cell
  const cw = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--cw"));
  const cellPx = cw + 12;
  const rendered = to - from + 1;
  const focusOffset = (pos - from);
  strip.style.transform = `translateX(${(rendered/2 - focusOffset - 0.5) * cellPx}px)`;
  const revCount = cells.filter(c => reviewed.has(cellKey(c))).length;
  const flagCount = cells.filter(c => c.flagged).length;
  const scope = onlyFlag ? "flagged" : (onlyLow ? "low-conf" : "all");
  $("progress").textContent =
    `${pos+1} / ${vis.length} ${scope}  ·  ${revCount}/${cells.length} reviewed  ·  ${flagCount} flagged`;
  const input = $("focusInput");
  if (input) { input.focus(); input.select(); input.oninput = markDirty; }
  localStorage.setItem(KEY, idx);
}

const pyCache = {};
async function pinyinFor(ch) {
  if (!ch) return "";
  if (pyCache[ch] !== undefined) return pyCache[ch];
  try {
    const r = await fetch("/pinyin?c=" + encodeURIComponent(ch));
    const d = await r.json();
    pyCache[ch] = d.pinyin || "";
  } catch { pyCache[ch] = ""; }
  return pyCache[ch];
}

async function updatePinyin() {
  // Fill every visible cell's pinyin line from its data-char.
  for (const node of document.querySelectorAll(".py")) {
    const ch = node.getAttribute("data-char") || "";
    node.textContent = (await pinyinFor(ch)) || " ";
  }
}

async function refreshFocusPinyin() {
  const el = $("focusInput"); if (!el) return;
  const py = el.closest(".cell").querySelector(".py");
  if (py) { py.setAttribute("data-char", el.value.trim());
    py.textContent = (await pinyinFor(el.value.trim())) || " "; }
}

function markDirty() {
  const el = $("focusInput"); if (!el) return;
  const c = cells[idx];
  el.parentElement.classList.toggle("ovr", el.value.trim() !== (c.ocr_char||""));
  refreshFocusPinyin();
}

async function saveFocus() {
  const el = $("focusInput"); if (!el) return;
  const c = cells[idx];
  const v = el.value.trim();
  const isOverride = v !== (c.ocr_char || "");
  const res = await fetch("/save", { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({book:c.book, key:`${c.provenance}#${c.char_index}`,
      name: isOverride ? v : ""}) });
  c.override = (res.ok && isOverride) ? v : "";
  $("saved").textContent = res.ok ? (isOverride ? "saved ✓" : "reverted ✓") : "save failed";
}

function step(dir) {
  const vis = visible(); if (!vis.length) return;
  let pos = vis.indexOf(idx);
  pos = Math.max(0, Math.min(vis.length-1, pos + dir));
  idx = vis[pos]; render();
}

// Toggle the 'impossible' flag on the focused NODE (all its character cells share
// the flag, keyed by provenance). Independent of reviewed. Persisted server-side.
async function toggleFlag() {
  const c = cells[idx];
  const now = !c.flagged;
  const res = await fetch("/flag", { method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({book:c.book, prov:c.provenance, flagged:now}) });
  if (res.ok) {
    for (const cc of cells) if (cc.book===c.book && cc.provenance===c.provenance) cc.flagged = now;
    $("saved").textContent = now ? "flagged ⚑" : "unflagged";
    render();
  } else { $("saved").textContent = "flag failed"; }
}

document.addEventListener("keydown", (e) => {
  if (e.isComposing) return; // let the IME finish
  // Alt+F toggles the 'impossible' flag on the focused node. Alt so it never
  // collides with typing an 'f' (or a Chinese char) into the name field.
  if ((e.altKey || e.metaKey) && (e.key === "f" || e.key === "F")) {
    e.preventDefault(); toggleFlag(); return;
  }
  // Enter = confirm: save, mark THIS cell reviewed, advance. Arrows just move
  // (and save any pending edit) without marking reviewed -- so scrolling/skipping
  // past a cell never counts it as reviewed.
  if (e.key === "Enter") { e.preventDefault(); const c = cells[idx];
    saveFocus().then(()=>{ markReviewed(c); step(1); }); }
  else if (e.key === "ArrowRight") { e.preventDefault(); saveFocus().then(()=>step(1)); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); saveFocus().then(()=>step(-1)); }
  else if (e.key === "Escape") {
    e.preventDefault();
    const el = $("focusInput"), c = cells[idx];
    if (el) { el.value = c.ocr_char || ""; markDirty();
      fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({book:c.book,key:`${c.provenance}#${c.char_index}`,name:""})});
      c.override=""; $("saved").textContent="reverted ✓"; }
  }
});
$("onlyLow").onchange = (e) => {
  onlyLow = e.target.checked;
  const vis = visible();
  if (vis.length && !vis.includes(idx)) idx = vis[0];
  render();
};
$("onlyFlag").onchange = (e) => {
  onlyFlag = e.target.checked;
  const vis = visible();
  if (vis.length && !vis.includes(idx)) idx = vis[0];
  render();
};

fetch("/data").then(r=>r.json()).then(d=>{
  cells = d;
  const saved = parseInt(localStorage.getItem(KEY) || "0", 10);
  idx = (saved>=0 && saved<cells.length) ? saved : 0;
  render();
});
</script>
</body>
</html>
"""


def _crop_band_png(book: str, fname: str, i: int, n: int) -> bytes:
    """Return PNG bytes of the i-th (of n) character band of a v1 name crop."""
    fpath = os.path.join(BOOKS_DIR, book, "5_names", os.path.basename(fname))
    im = Image.open(fpath).convert("L")
    bands = split_bands(im, n)
    if 0 <= i < len(bands):
        top, bot = bands[i]
        im = im.crop((0, top, im.width, bot))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - match base signature; quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/data":
            self._send(200, json.dumps(build_cells(), ensure_ascii=False))
        elif path.startswith("/crop/"):
            _, _, book, fname = path.split("/", 3)
            qs = dict(
                kv.split("=", 1) for kv in parsed.query.split("&") if "=" in kv
            )
            i, n = int(qs.get("i", 0)), int(qs.get("n", 1))
            try:
                self._send(200, _crop_band_png(book, fname, i, n), "image/png")
            except FileNotFoundError:
                self._send(404, b"not found", "text/plain")
        elif path == "/pinyin":
            from urllib.parse import parse_qs, unquote
            q = parse_qs(parsed.query)
            text = unquote(q.get("c", [""])[0])
            self._send(200, json.dumps({"pinyin": pinyin_of(text)}, ensure_ascii=False))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/save", "/flag"):
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        book = payload.get("book")
        if book not in BOOKS:
            self._send(400, json.dumps({"error": "bad request"}))
            return
        if route == "/flag":
            prov = payload.get("prov")
            if not prov:
                self._send(400, json.dumps({"error": "bad request"}))
                return
            save_flag(book, prov, bool(payload.get("flagged")))
            self._send(200, json.dumps({"ok": True}))
            return
        key = payload.get("key")
        name = payload.get("name", "").strip()
        if not key:
            self._send(400, json.dumps({"error": "bad request"}))
            return
        save_override(book, key, name)
        self._send(200, json.dumps({"ok": True}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = len(build_cells())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("OCR review filmstrip: %d characters. Open http://localhost:%d/", n, args.port)
    logger.info("Corrections -> data/{book}_overrides.json (keyed provenance#charIndex). Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopped.")


if __name__ == "__main__":
    main()
