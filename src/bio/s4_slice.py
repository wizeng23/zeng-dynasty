"""Bio stage 4 (geometric slice): cut a person-block into father / name / column strips,
then OCR each piece separately so reading order + direction are decided by GEOMETRY, not
by a vision model.

Why: sending a whole wide multi-column crop to a vision LLM makes it re-derive the layout,
and it gets it wrong -- reads left-to-right instead of right-to-left, shuffles columns, or
merges several columns into one giant line. The column geometry is something we already
know from the scan, so we slice it ourselves and OCR one column at a time (Paddle does
much better on small single-column crops). We assemble the RTL order.

Per block (see the design discussion):

1. Trim residual generation-band RULE lines off the top/bottom of the crop.
2. Split into a RIGHT region (father header + name) and the BODY (everything to its left).
   The right region is a fixed-width strip on the right edge.
3. In the right strip make ONE horizontal cut below the father header: above = father
   (the horizontal 子之X caption), below = name (all stacked name chars, however many).
4. Slice the BODY into vertical COLUMN STRIPS by whitespace projection (variable-width
   inter-column gaps), right-to-left. Paddle char-box x-clusters cross-check the cuts.
5. Estimate each strip's char count from ink height / per-char pixel size (a QA signal).
6. OCR each piece with Paddle; assemble [father, name, col1, col2, ...] RTL.

Debug mode dumps every slice as a PNG + prints the per-strip OCR so the cuts can be eyeballed.

Run (debug on a few blocks):
    PYTHONPATH=. python -m src.bio.s4_slice --book book3 --ids 139_181_1_2 18_63_0_4 --debug
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SEG_DIR = os.path.join("bio", "3_segment")

# Geometry (pixels at the ~4400-4900px-wide, ~1000-1130px-tall crop resolution). These are
# starting points from real crops; tune on the debug samples.
RIGHT_STRIP_W = 400        # width of the right region holding father header + name
RULE_DARK_FRAC = 0.55      # a row is a "rule line" if >=55% of its pixels are ink
GAP_INK_FRAC = 0.04        # an x-column is a "gap" if its ink <= 4% of the column peak
MIN_GAP_W = 18             # a run of gap-columns >= this wide separates two columns
COL_PAD = 40               # whitespace padding added to EACH side of a column strip
# Every scan is the same resolution, so a printed character column is ~120px wide. Widen each
# detected column to at least this (covers thin extending strokes that ink-projection misses),
# then + COL_PAD each side => ~200px total strip width.
CHAR_W = 120
MIN_COL_W = 30             # ignore true slivers (specks) narrower than this before widening


@dataclass
class Piece:
    kind: str              # "father" | "name" | "col"
    box: list[int]         # [x0,y0,x1,y1] in the (trimmed) crop's coords
    img: Image.Image = field(repr=False, default=None)
    est_chars: int = 0     # ink-height-based char estimate
    text: str = ""         # filled by OCR


# --- ink helpers -------------------------------------------------------------------

def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def _ink(a: np.ndarray) -> np.ndarray:
    """Ink mass in [0,1] per pixel (1 = black)."""
    return (255.0 - a) / 255.0


RULE_SPAN_FRAC = 0.80      # a RULE row's ink must SPAN >= this fraction of the full width
                           # (leftmost..rightmost inked x). The father caption is only ~right
                           # 40% wide, so it is never a rule even when its ink touches the top.
RULE_FILL_FRAC = 0.55      # AND, within that span, >= this fraction of pixels must be inked --
                           # a rule is a near-solid (or dense dashed) bar, whereas a full-width
                           # ROW OF TEXT spans wide but has big gaps between characters.
RULE_BRIDGE = 14           # rule rows within this many px are one (dashed/smeared) rule band


def _rule_rows(ink: np.ndarray, W: int) -> np.ndarray:
    """Boolean per row: True where the ink is a RULE line -- it SPANS nearly the whole width
    AND is densely filled across that span. Full-width span rejects the father caption; the
    fill test rejects a wide row of separate text characters (which has inter-char gaps)."""
    mask = ink > 0.3
    any_ink = mask.any(axis=1)
    left = np.where(any_ink, np.argmax(mask, axis=1), 0)
    right = np.where(any_ink, W - 1 - np.argmax(mask[:, ::-1], axis=1), -1)
    span = np.where(any_ink, (right - left + 1) / float(W), 0.0)
    rowsum = mask.sum(axis=1).astype(float)
    width_px = np.where(any_ink, right - left + 1, 1)
    fill = rowsum / width_px                       # inked fraction WITHIN the ink span
    return (span >= RULE_SPAN_FRAC) & (fill >= RULE_FILL_FRAC)


def trim_rules(img: Image.Image, band_frac: float = 0.18, margin: int = 10) -> tuple[Image.Image, int]:
    """Remove horizontal RULE lines (+ blank margin) from top and bottom.

    A rule line is a horizontal bar spanning nearly the FULL width (RULE_SPAN_FRAC). On ADF
    scans it is often faint/dashed and smeared across several rows, so we bridge nearby rule
    rows (RULE_BRIDGE) into one band and cut past the INNERMOST band in each edge region.
    Because the test is full-width SPAN (not ink density), the ~400px-wide father caption is
    never mistaken for a rule even when its ink touches the top edge. Returns (trimmed, top_off).
    """
    a = _gray(img)
    ink = _ink(a)
    H, W = a.shape
    is_rule = _rule_rows(ink, W)
    band = max(1, int(H * band_frac))

    # TOP: cut below the LOWEST (innermost) rule row in the top band, bridging dashed remnants.
    top = 0
    tr = np.where(is_rule[:band])[0]
    if len(tr):
        low = int(tr[-1])
        r = low
        while r + 1 < H and (is_rule[r + 1] or (r + 1 - low) <= RULE_BRIDGE):
            r += 1
            if is_rule[r]:
                low = r
        top = low + 1 + margin

    # BOTTOM: cut above the HIGHEST (innermost) rule row in the bottom band.
    bot = H
    br = np.where(is_rule[H - band:])[0]
    if len(br):
        high = H - band + int(br[0])
        r = high
        while r - 1 >= 0 and (is_rule[r - 1] or (high - (r - 1)) <= RULE_BRIDGE):
            r -= 1
            if is_rule[r]:
                high = r
        bot = high - margin

    # then skip any remaining blank margin inward to the first/last text row
    rowink = ink.sum(axis=1)
    tthresh = 0.01 * (rowink.max() or 1)
    while top < bot and rowink[top] <= tthresh:
        top += 1
    while bot > top and rowink[bot - 1] <= tthresh:
        bot -= 1
    if bot <= top:
        return img, 0
    return img.crop((0, top, W, bot)), top


# --- father / name (right strip) ---------------------------------------------------

def split_father_name(strip: Image.Image) -> tuple[Piece, Piece]:
    """One horizontal cut in the right strip: above = father header, below = name.

    The father header is the horizontal ink band at the top; below it is a vertical gap,
    then the name (all remaining ink in the strip). We find the first substantial vertical
    gap (blank rows) after the initial ink band and cut there.
    """
    a = _gray(strip)
    ink = _ink(a)
    H, W = a.shape
    rowink = ink.sum(axis=1)
    peak = rowink.max() or 1
    dark = rowink > 0.06 * peak                 # rows that carry text

    # Collect contiguous ink bands (start,end). The father header is the FIRST band that is
    # thick enough to be real text (skip thin rule/speck remnants).
    bands: list[tuple[int, int]] = []
    i = 0
    while i < H:
        if not dark[i]:
            i += 1
            continue
        s = i
        while i < H and dark[i]:
            i += 1
        bands.append((s, i))
    MIN_BAND = max(20, int(0.05 * H))            # a real text band is at least this tall
    real = [b for b in bands if b[1] - b[0] >= MIN_BAND]
    if not real:
        # no clear header -> treat whole strip as name
        return (Piece("father", [0, 0, W, 0], strip.crop((0, 0, W, 0))),
                Piece("name", [0, 0, W, H], strip.crop((0, 0, W, H))))
    fs, fe = real[0]                              # father header band
    # cut = midpoint of the gap after the father band (start of the next band, or fe)
    nxt = next((b[0] for b in real[1:] if b[0] > fe), None)
    cut = (fe + nxt) // 2 if nxt else fe
    father = Piece("father", [0, fs, W, fe], strip.crop((0, fs, W, fe)))
    name = Piece("name", [0, cut, W, H], strip.crop((0, cut, W, H)))
    return father, name


# --- body -> column strips ---------------------------------------------------------

def slice_columns(body: Image.Image) -> list[Piece]:
    """Cut the body into vertical column strips by whitespace projection, RIGHT-TO-LEFT.

    Each raw ink run is then (a) widened symmetrically to at least ``CHAR_W`` (the fixed
    printed-character width; every scan is the same resolution) so a narrow column doesn't clip
    a character's thin extending strokes, and (b) padded by ``COL_PAD`` px on each side. Both
    are clamped to the body's bounds; columns may overlap slightly after padding, which is fine
    (each strip is OCR'd on its own).
    """
    a = _gray(body)
    ink = _ink(a)
    H, W = a.shape
    colink = ink.sum(axis=0)                      # ink mass per x
    peak = colink.max() or 1
    is_gap = colink <= GAP_INK_FRAC * peak

    # find contiguous non-gap runs = columns
    runs: list[tuple[int, int]] = []
    x = 0
    while x < W:
        if is_gap[x]:
            x += 1
            continue
        start = x
        gaprun = 0
        while x < W and not (is_gap[x] and (gaprun := gaprun + 1) >= MIN_GAP_W):
            if not is_gap[x]:
                gaprun = 0
            x += 1
        end = x - gaprun if gaprun else x
        if end - start >= MIN_COL_W:
            runs.append((start, end))
    if not runs:
        return []

    # Widen each run to at least CHAR_W (fixed printed-char width) so thin strokes aren't
    # clipped, then pad COL_PAD px each side (=> ~CHAR_W + 2*COL_PAD total).
    cols: list[tuple[int, int]] = []
    for s, e in runs:
        if e - s < CHAR_W:                        # widen symmetrically around the center
            c = (s + e) / 2.0
            s = c - CHAR_W / 2.0
            e = c + CHAR_W / 2.0
        s -= COL_PAD                              # whitespace padding each side
        e += COL_PAD
        cols.append((max(0, int(round(s))), min(W, int(round(e)))))

    cols.sort(key=lambda c: -c[0])                # right-to-left order
    return [Piece("col", [s, 0, e, H], body.crop((s, 0, e, H))) for s, e in cols]


def estimate_chars(piece: Image.Image) -> int:
    """Rough char count for a vertical column = ink-height / typical glyph height.

    Uses the vertical extent of ink in the strip divided by the median glyph box height
    (approximated as the strip width, since glyphs are ~square). Best-effort QA signal.
    """
    a = _gray(piece)
    ink = _ink(a)
    rows = np.where(ink.sum(axis=1) > 0.02 * (ink.sum(axis=1).max() or 1))[0]
    if len(rows) == 0:
        return 0
    span = rows[-1] - rows[0] + 1
    glyph = max(piece.width * 0.9, 1)            # glyphs ~ column-width tall
    return max(1, round(span / glyph))


# --- assemble ----------------------------------------------------------------------

def _rightmost_ink_x(trimmed: Image.Image) -> int:
    """The rightmost x with real ink (skips right-side scan whitespace/margin). Returns W-1
    if the whole right edge is inked. Used to anchor the father/name region to the text band
    instead of the physical right edge."""
    a = _gray(trimmed)
    ink = _ink(a)
    colink = ink.sum(axis=0)
    peak = colink.max() or 1
    inked = np.where(colink > GAP_INK_FRAC * peak)[0]
    return int(inked[-1]) if len(inked) else trimmed.width - 1


def slice_block(img: Image.Image) -> list[Piece]:
    """Full geometric slice of one person-block crop -> [father, name, col1, col2, ...].

    The right region (father header + name) is anchored to the rightmost INK column, not the
    physical right edge -- so right-side scan whitespace doesn't shift the father/name window
    or leak body columns into it.
    """
    trimmed, _ = trim_rules(img)
    r_edge = _rightmost_ink_x(trimmed) + 1        # one past the last inked column
    right_x0 = max(0, r_edge - RIGHT_STRIP_W)     # 400px window anchored at the ink band
    right = trimmed.crop((right_x0, 0, r_edge, trimmed.height))
    body = trimmed.crop((0, 0, right_x0, trimmed.height))
    father, name = split_father_name(right)
    # father/name boxes come back right-strip-local -> shift to ABSOLUTE trimmed-crop coords so
    # every stored box is in one frame (no downstream translation guesswork). cols are already
    # body-local = absolute (body starts at x=0).
    for p in (father, name):
        p.box = [p.box[0] + right_x0, p.box[1], p.box[2] + right_x0, p.box[3]]
    cols = slice_columns(body)
    pieces = [father, name, *cols]
    for p in pieces:
        p.est_chars = estimate_chars(p.img)
    return pieces


# --- Paddle OCR per piece ----------------------------------------------------------

def ocr_pieces(pieces: list[Piece], engine) -> None:
    """OCR each slice with Paddle.

    Body/name pieces are vertical columns -> sort chars top-to-bottom. The FATHER header is
    printed HORIZONTALLY (子之X), so read it left-to-right by x instead of the vertical
    resort (which would reverse it).
    """
    from src.bio.s4_ocr import _char_boxes
    for p in pieces:
        chars = _char_boxes(engine, np.asarray(p.img.convert("RGB")))
        if not chars:
            p.text = ""
            continue
        if p.kind == "father":
            chars.sort(key=lambda c: c["x0"])            # horizontal, left-to-right
        else:
            chars.sort(key=lambda c: c["y0"])            # vertical column, top-to-bottom
        p.text = "".join(c["ch"] for c in chars)


# Two prompts, chosen per strip by kind. The father header is printed horizontally
# (子之X caption); every other strip (name + body columns) is a vertical column. Each
# names its own single orientation -- no confusing "OR a caption" branch. One added
# clause of genealogy context to steer glyph disambiguation (e.g. 夭 vs 天); kept to one
# sentence since prompt tokens are paid.
_SCAN_NOISE = " Ignore ink smears and very faint characters caused by document scanning."
_STRIP_PROMPT = (
    "This image is ONE vertical column of characters from a biography entry in a Chinese "
    "genealogy. Transcribe EVERY character top-to-bottom in reading order. Output ONLY the "
    "characters, no spaces, no punctuation, no commentary. Use ? for any you truly cannot read."
    + _SCAN_NOISE
)
_FATHER_PROMPT = (
    "This image is a short horizontal caption from a biography entry in a Chinese genealogy. "
    "Transcribe EVERY character left-to-right in reading order. Output ONLY the characters, "
    "no spaces, no punctuation, no commentary. Use ? for any you truly cannot read."
    + _SCAN_NOISE
)


def _prompt_for(kind: str) -> str:
    return _FATHER_PROMPT if kind == "father" else _STRIP_PROMPT


def _gemini_read(gclient, gmodel: str, kind: str, img, retries: int = 4) -> str:
    """One Gemini call for one strip, with backoff on transient errors. Returns the text,
    '?EMPTY' if the strip is unreadable/blank, or '?ERR' only after all retries fail. Kept
    separate so both the overnight runner and the debug path use identical logic."""
    import io
    import time
    from google.genai import types
    if img.width < 4 or img.height < 4:
        return "?EMPTY"
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
    part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
    last = None
    for attempt in range(retries):
        try:
            r = gclient.models.generate_content(
                model=gmodel, contents=[_prompt_for(kind), part])
            return (r.text or "").strip() or "?EMPTY"
        except Exception as e:                            # noqa: BLE001
            last = e
            msg = str(e).lower()
            transient = any(t in msg for t in (
                "429", "rate", "quota", "503", "500", "unavailable", "timeout",
                "deadline", "connection", "reset", "overloaded"))
            if attempt < retries - 1 and transient:
                time.sleep(2 ** attempt + 0.5)            # 1.5, 2.5, 4.5, ... s
                continue
            break
    logger.warning("gemini fail (%s): %s", kind, last)
    return "?ERR"


def gemini_pieces(pieces: list[Piece], model: str = "gemini-3.8-flash") -> None:
    """OCR each slice with a vision model (one call per strip). Because each strip is a
    single column, the model can't shuffle/merge -- it only transcribes glyphs, which is
    its strong suit. Fills p.text."""
    import io
    import os
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for p in pieces:
        if p.img.width < 4 or p.img.height < 4:
            p.text = ""
            continue
        buf = io.BytesIO(); p.img.convert("RGB").save(buf, format="PNG")
        r = client.models.generate_content(
            model=model,
            contents=[_prompt_for(p.kind), types.Part.from_bytes(data=buf.getvalue(),
                                                                 mime_type="image/png")])
        p.text = (r.text or "").strip()


def iter_ids(book: str, ids: list[str], books_dir: str):
    seg = os.path.join(books_dir, book, SEG_DIR)
    for bid in ids:
        path = os.path.join(seg, f"{bid}.png")
        if os.path.exists(path):
            yield bid, Image.open(path).convert("RGB")


def _debug_dump(book: str, bid: str, pieces: list[Piece], out_dir: str) -> None:
    d = os.path.join(out_dir, bid)
    os.makedirs(d, exist_ok=True)
    for i, p in enumerate(pieces):
        p.img.save(os.path.join(d, f"{i:02d}_{p.kind}.png"))


def run(book: str, ids: list[str], books_dir: str = "books", debug: bool = False,
        debug_dir: str = "scratchpad/s4_slice") -> dict:
    engine = None
    from src.bio.s4_ocr import _paddle_engine
    engine = _paddle_engine()
    out = {}
    for bid, img in iter_ids(book, ids, books_dir):
        pieces = slice_block(img)
        ocr_pieces(pieces, engine)
        if debug:
            _debug_dump(book, bid, pieces, debug_dir)
        rec = {
            "id": bid,
            "father": pieces[0].text,
            "name": pieces[1].text,
            "cols": [p.text for p in pieces[2:]],
            "est_chars": [p.est_chars for p in pieces],
            "assembled": "\n".join([pieces[0].text, pieces[1].text,
                                    *[p.text for p in pieces[2:]]]),
        }
        out[bid] = rec
        logger.info("%s: father=%r name=%r cols=%d", bid, rec["father"], rec["name"],
                    len(rec["cols"]))
    return out


# --- persistence: slice every block, save strip PNGs + per-strip Paddle reads -----
#
# Output layout (per book):
#   books/{book}/bio/4_slice/{stem}.jsonl   -- one record per block:
#       {id, box_trimmed, pieces:[{kind, idx, box, est_chars, paddle, gemini, vision}]}
#   books/{book}/bio/4_slice/strips/{id}/{idx}_{kind}.png  -- each strip image (for the
#       vision subagents in step 2, which read image files; also handy for QA).
# Idempotent: a block whose record already has non-empty paddle on all pieces is skipped.

SLICE_DIR = "4_slice"


def _blocks_index(book: str, books_dir: str) -> list[dict]:
    from src.bio.s4_ocr import _load_index
    return _load_index(book, books_dir)


def strip_dir(book: str, bid: str, books_dir: str) -> str:
    return os.path.join(books_dir, book, "bio", SLICE_DIR, "strips", bid)


def slice_and_save(book: str, bid: str, img: Image.Image, books_dir: str,
                   engine=None) -> dict:
    """Slice one block, save each strip PNG, run Paddle per strip, return the record."""
    pieces = slice_block(img)
    sd = strip_dir(book, bid, books_dir)
    os.makedirs(sd, exist_ok=True)
    if engine is not None:
        ocr_pieces(pieces, engine)
    rec = {"id": bid, "pieces": []}
    for i, p in enumerate(pieces):
        p.img.save(os.path.join(sd, f"{i:02d}_{p.kind}.png"))
        rec["pieces"].append({
            "idx": i, "kind": p.kind, "box": p.box, "est_chars": p.est_chars,
            "paddle": p.text, "gemini": "", "vision": "",
        })
    return rec


def paddle_book(book: str, books_dir: str = "books", sections: list[str] | None = None,
                limit: int | None = None) -> dict:
    """Step 1: slice every block + Paddle per strip -> 4_slice/{stem}.jsonl + strip PNGs."""
    from src.bio.s4_ocr import _paddle_engine
    engine = _paddle_engine()
    out_base = os.path.join(books_dir, book, "bio", SLICE_DIR)
    os.makedirs(out_base, exist_ok=True)
    rows = _blocks_index(book, books_dir)
    if sections:
        rows = [r for r in rows if r.get("section") in sections]
    if limit:
        rows = rows[:limit]
    seg = os.path.join(books_dir, book, SEG_DIR)
    by_stem: dict[str, list[dict]] = {}
    n = 0
    for r in rows:
        bid = r["id"]; stem = r["section"]
        img = Image.open(os.path.join(seg, f"{bid}.png")).convert("RGB")
        rec = slice_and_save(book, bid, img, books_dir, engine)
        by_stem.setdefault(stem, []).append(rec)
        n += 1
        if n % 25 == 0:
            logger.info("  ...%d/%d sliced", n, len(rows))
    for stem, recs in by_stem.items():
        with open(os.path.join(out_base, f"{stem}.jsonl"), "w") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("DONE %s: %d blocks sliced across %d sections", book, n, len(by_stem))
    return {"book": book, "blocks": n, "sections": len(by_stem)}


# --- overnight run: Paddle + Gemini per strip, parallel, single writer ------------
#
# One process owns all writes to books/{book}/bio/4_slice/{stem}.jsonl. A ThreadPoolExecutor
# fans out per-BLOCK work (each block: slice -> Paddle every strip -> Gemini every strip),
# and the finished record is appended under a per-stem file lock. Because there is exactly
# ONE writer process, an in-process threading.Lock fully prevents write races (the earlier
# clobber was TWO processes writing the same file -- we never do that here).
#
# Resumable/idempotent: on start we load existing {stem}.jsonl records; a block whose pieces
# already have BOTH non-empty paddle AND a filled gemini (or a recorded gemini failure) is
# skipped. Save-after-every-block => a crash loses at most the in-flight blocks, never a
# completed (paid) Gemini call.
#
# Claude-vision is NOT run here (the Agent tool is driven by the assistant, not a script).
# It runs as a separate pass writing a sidecar {stem}.vision.jsonl -- never this file --
# so the two passes never contend for the same file.

def _load_slice_recs(out_base: str, stem: str) -> dict[str, dict]:
    path = os.path.join(out_base, f"{stem}.jsonl")
    recs: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["id"]] = r
    return recs


def _rec_done(rec: dict) -> bool:
    """A block is done when every piece has a paddle read attempted and a gemini value
    that is non-empty OR explicitly marked failed (so we don't infinitely retry a strip
    Gemini genuinely can't read)."""
    ps = rec.get("pieces") or []
    if not ps:
        return False
    return all(("gemini" in p and p["gemini"] != "") for p in ps)


def _write_stem(out_base: str, stem: str, recs: dict[str, dict]) -> None:
    """Atomic-ish rewrite of one stem file (tmp + rename) under the caller's lock."""
    path = os.path.join(out_base, f"{stem}.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for _id, rec in recs.items():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_book(book: str, books_dir: str = "books", sections: list[str] | None = None,
              limit: int | None = None, workers: int = 8,
              gmodel: str = "gemini-3.8-flash") -> dict:
    """Overnight step 1+2a: slice + Paddle + Gemini for every block, parallel, resumable.
    Writes books/{book}/bio/4_slice/{stem}.jsonl (single writer, per-stem lock)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google import genai
    from src.bio.s4_ocr import _paddle_engine

    engine = _paddle_engine()
    gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out_base = os.path.join(books_dir, book, "bio", SLICE_DIR)
    os.makedirs(out_base, exist_ok=True)
    seg = os.path.join(books_dir, book, SEG_DIR)

    rows = _blocks_index(book, books_dir)
    if sections:
        rows = [r for r in rows if r.get("section") in sections]
    if limit:
        rows = rows[:limit]

    # preload existing records per stem (resume) + a lock per stem
    stems = sorted({r["section"] for r in rows})
    store: dict[str, dict[str, dict]] = {s: _load_slice_recs(out_base, s) for s in stems}
    locks: dict[str, threading.Lock] = {s: threading.Lock() for s in stems}

    todo = [r for r in rows if not _rec_done(store[r["section"]].get(r["id"], {}))]
    logger.info("%s: %d blocks total, %d already done, %d to read (workers=%d, model=%s)",
                book, len(rows), len(rows) - len(todo), len(todo), workers, gmodel)

    engine_lock = threading.Lock()          # Paddle engine is not thread-safe -> serialize it
    done_n = [0]

    def work(r):
        bid = r["id"]; stem = r["section"]
        img = Image.open(os.path.join(seg, f"{bid}.png")).convert("RGB")
        with engine_lock:
            # Paddle inside the lock; Gemini (the slow part) is inside too here for
            # simplicity, but Gemini releases the GIL on I/O so threads still overlap on it.
            # To maximize Gemini overlap we do Paddle-then-release, then Gemini outside:
            pieces = slice_block(img)
            sd = strip_dir(book, bid, books_dir)
            os.makedirs(sd, exist_ok=True)
            ocr_pieces(pieces, engine)
            for i, p in enumerate(pieces):
                p.img.save(os.path.join(sd, f"{i:02d}_{p.kind}.png"))
        # Gemini per strip, OUTSIDE the engine lock so API calls run concurrently.
        # _gemini_read handles backoff/retry on transient errors so an unattended run does
        # not silently fill '?ERR' during a rate-limit blip.
        rec = {"id": bid, "pieces": []}
        for i, p in enumerate(pieces):
            gem = _gemini_read(gclient, gmodel, p.kind, p.img)
            rec["pieces"].append({
                "idx": i, "kind": p.kind, "box": p.box, "est_chars": p.est_chars,
                "paddle": p.text, "gemini": gem, "vision": "",
            })
        # write under this stem's lock (single writer process => race-free)
        with locks[stem]:
            store[stem][bid] = rec
            _write_stem(out_base, stem, store[stem])
        done_n[0] += 1
        if done_n[0] % 10 == 0:
            logger.info("  ...%d/%d blocks read", done_n[0], len(todo))
        return bid

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:                       # noqa: BLE001
                logger.error("block failed: %s", e)

    logger.info("DONE %s: %d blocks across %d sections", book, len(rows), len(stems))
    return {"book": book, "blocks": len(rows), "read": len(todo), "sections": len(stems)}


# --- Claude-vision sidecar: merge subagent reads without touching {stem}.jsonl ----
#
# The vision pass (assistant-driven Agent subagents) writes per-block reads to a SEPARATE
# file, books/{book}/bio/4_slice/{stem}.vision.jsonl, one record per line:
#     {"id": bid, "vision": ["<father text>", "<name text>", "<col0>", ...]}  # by piece idx
# It never opens {stem}.jsonl, so it can run while read_book is still writing that file.
# merge_vision folds the sidecars into the main records' per-piece "vision" field.

def append_vision(book: str, stem: str, bid: str, texts: list[str],
                  books_dir: str = "books") -> None:
    """Append one block's Claude-vision reads to books/{book}/bio/4_slice/{stem}.vision.jsonl.
    Called by the assistant with a subagent's returned per-piece texts. This sidecar file is
    NEVER read/written by read_book, so the two passes cannot race. Deduped by id: a repeat
    id overwrites the earlier line on the next rewrite."""
    out_base = os.path.join(books_dir, book, "bio", SLICE_DIR)
    os.makedirs(out_base, exist_ok=True)
    path = os.path.join(out_base, f"{stem}.vision.jsonl")
    existing: dict[str, list[str]] = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing[r["id"]] = r.get("vision", [])
    existing[bid] = texts
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for _id, v in existing.items():
            fh.write(json.dumps({"id": _id, "vision": v}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def retry_gemini_errs(book: str, books_dir: str = "books",
                      gmodel: str = "gemini-3.8-flash", markers=("?ERR",)) -> dict:
    """Re-run Gemini on strips whose gemini value is in ``markers`` (default just ?ERR;
    pass ("?ERR","?EMPTY") to also retry blanks). Reads each strip's saved PNG from
    4_slice/strips/{id}/{idx}_{kind}.png and rewrites the value in place. Fixes the transient
    network drops that a plain --read-all resume skips (it counts ?ERR as done)."""
    from google import genai
    gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    out_base = os.path.join(books_dir, book, "bio", SLICE_DIR)
    fixed = 0
    for fn in sorted(os.listdir(out_base)):
        if not fn.endswith(".jsonl") or fn.endswith(".vision.jsonl"):
            continue
        stem = fn[: -len(".jsonl")]
        recs = _load_slice_recs(out_base, stem)
        changed = False
        for bid, rec in recs.items():
            for p in rec.get("pieces", []):
                if p.get("gemini") in markers:
                    png = os.path.join(out_base, "strips", bid, f"{p['idx']:02d}_{p['kind']}.png")
                    if not os.path.exists(png):
                        continue
                    img = Image.open(png).convert("RGB")
                    val = _gemini_read(gclient, gmodel, p["kind"], img)
                    if val not in markers:
                        p["gemini"] = val
                        fixed += 1
                        changed = True
        if changed:
            _write_stem(out_base, stem, recs)
    logger.info("retry_gemini_errs %s: %d strips fixed", book, fixed)
    return {"book": book, "fixed": fixed}


def merge_vision(book: str, books_dir: str = "books") -> dict:
    """Fold every {stem}.vision.jsonl into the matching {stem}.jsonl pieces' 'vision'."""
    out_base = os.path.join(books_dir, book, "bio", SLICE_DIR)
    merged = 0
    for fn in sorted(os.listdir(out_base)):
        if not fn.endswith(".vision.jsonl"):
            continue
        stem = fn[: -len(".vision.jsonl")]
        vis: dict[str, list[str]] = {}
        with open(os.path.join(out_base, fn)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    vis[r["id"]] = r.get("vision", [])
        recs = _load_slice_recs(out_base, stem)
        for bid, texts in vis.items():
            rec = recs.get(bid)
            if not rec:
                continue
            for i, p in enumerate(rec.get("pieces", [])):
                if i < len(texts) and texts[i] != "":
                    p["vision"] = texts[i]
                    merged += 1
        # rewrite the main file with vision folded in
        path = os.path.join(out_base, f"{stem}.jsonl")
        with open(path, "w") as ofh:
            for _id, rec in recs.items():
                ofh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("merge_vision %s: %d piece reads folded", book, merged)
    return {"book": book, "pieces_merged": merged}


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--ids", nargs="+", help="debug: slice just these block ids")
    ap.add_argument("--paddle-all", action="store_true",
                    help="step 1: slice every block + Paddle per strip -> 4_slice/")
    ap.add_argument("--read-all", action="store_true",
                    help="overnight: slice + Paddle + Gemini per strip (parallel, resumable)")
    ap.add_argument("--merge-vision", action="store_true",
                    help="fold {stem}.vision.jsonl sidecars into the main records")
    ap.add_argument("--retry-errs", action="store_true",
                    help="re-run Gemini on ?ERR strips (transient net drops) in place")
    ap.add_argument("--sections", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gmodel", default="gemini-3.8-flash")
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")
    if args.read_all:
        res = read_book(args.book, books_dir=args.books_dir, sections=args.sections,
                        limit=args.limit, workers=args.workers, gmodel=args.gmodel)
        print(json.dumps(res, ensure_ascii=False))
        return
    if args.merge_vision:
        print(json.dumps(merge_vision(args.book, books_dir=args.books_dir), ensure_ascii=False))
        return
    if args.retry_errs:
        print(json.dumps(retry_gemini_errs(args.book, books_dir=args.books_dir,
                                           gmodel=args.gmodel), ensure_ascii=False))
        return
    if args.paddle_all:
        paddle_book(args.book, books_dir=args.books_dir, sections=args.sections,
                    limit=args.limit)
        return
    res = run(args.book, args.ids, books_dir=args.books_dir, debug=args.debug)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
