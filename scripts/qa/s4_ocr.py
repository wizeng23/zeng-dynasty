"""QA reviewer for bio stage-4 OCR: compare Paddle vs Claude(vision) per person block,
and hand-verify the final reading.

Serves a single-page reviewer at http://localhost:PORT/. For every 3_segment person
block of a book it shows four columns, side by side, all rendered TOP-TO-BOTTOM /
RIGHT-TO-LEFT (writing-mode: vertical-rl) so each transcription mirrors the scanned crop:

  [ crop image ] [ Paddle ] [ Claude ] [ Verified (editable) ]

The Verified box is prepopulated with the two readers' agreement (line-by-line: where
Paddle and Claude agree, that line is taken; where they differ it's left blank for you to
fill). Editing + Save writes the confirmed text to data/{book}_bio_verified.json keyed by
block id -- the ground-truth OCR layer. Saved blocks are marked done; the list shows
progress and lets you jump to the next unreviewed / disagreeing block.

Run:
    PYTHONPATH=. python -m scripts.qa.s4_ocr --book book3      # then open the printed URL
"""
from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEG_DIR = os.path.join("bio", "3_segment")
OCR_DIR = os.path.join("bio", "4_ocr")
SLICE_DIR = os.path.join("bio", "4_slice")


# --- graph reference (what the tree says a block should be) ------------------------
# For each block, show the tree node at the SAME (subgraph, generation, position) so the
# reviewer can copy the graph's name/father/sons over when the OCR is wrong. Purely
# positional (pos-th node in that gen of that subgraph) -- no fuzzy matching; a position
# with no node just shows blank. Reuses s5_link's subgraph grouping (stem via provenance
# in notes) over data/{book}_stitched.jsonl.

def build_graph_index(book: str, books_dir: str, data_dir: str):
    """(stem,gen)->[nodes RTL/eldest-first] + id->name, from the stitched tree. {} if absent."""
    from collections import defaultdict
    from src.bio.s5_link import _stem_of, section_to_stem
    tree_path = os.path.join(data_dir, f"{book}_stitched.jsonl")
    if not os.path.exists(tree_path):
        return {"tb": {}, "byid": {}, "sec2stem": {}}
    rows = [json.loads(l) for l in open(tree_path) if l.strip()]
    byid = {r["id"]: r for r in rows}
    tb = defaultdict(lambda: defaultdict(list))
    for n in rows:
        tb[_stem_of(n)][n["generation"]].append(n)
    try:
        sec2stem = section_to_stem(book, books_dir)
    except Exception:
        sec2stem = {}
    return {"tb": tb, "byid": byid, "sec2stem": sec2stem}


def _node_ref(byid: dict, n: dict, matched: bool) -> dict:
    father = byid.get(n["father"], {}).get("name") if n.get("father", -1) != -1 else None
    sons = [byid[c]["name"] for c in n.get("children", []) if c in byid]
    return {"name": n.get("name"), "father": father, "sons": sons, "matched": matched}


def _bio_name_candidates(block: dict, verified: str | None) -> list[str]:
    """The block's own-name as best we know it, most-trusted first: the reviewer's Verified
    name (col 1), then each reader's detected name (col at that reader's name_idx)."""
    out: list[str] = []
    if verified:
        cols = [c for c in verified.split("\n")]
        if len(cols) > 1 and cols[1].strip():
            out.append(cols[1].strip())
    fields = block.get("fields") or {}
    for rd in ("vision", "gemini", "mistral", "paddle"):
        cols = block.get(rd) or []
        ni = (fields.get(rd) or {}).get("name_idx")
        if ni is not None and ni < len(cols) and cols[ni].strip():
            out.append(cols[ni].strip())
    # de-dup preserving order
    seen = set(); uniq = []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def graph_ref(idx: dict, block: dict, verified: str | None = None) -> dict | None:
    """The tree node this bio block corresponds to.

    Smart match: if any candidate bio-name (the reviewer's Verified name, else a reader's
    detected name) equals a graph node's VERIFIED name in this block's subgraph+generation,
    show THAT node (``matched: true``) -- robust to segmentation gaps/over-splits. Otherwise
    fall back to the positional node (pos-th in the gen; ``matched: false``). None if no
    graph is loaded.
    """
    tb, byid, sec2stem = idx["tb"], idx["byid"], idx["sec2stem"]
    if not tb:
        return None
    bid = block["id"]
    parts = bid.split("_")
    pos = int(parts[-1])
    sec = "_".join(parts[:-2])
    stem = sec2stem.get(sec, sec)
    gen = block.get("generation")
    nodes = tb.get(stem, {}).get(gen, [])
    if not nodes:
        return {"name": None, "father": None, "sons": None, "matched": False}

    # 1. name-match against the verified graph names in this generation
    cands = _bio_name_candidates(block, verified)
    by_name: dict[str, dict] = {}
    for n in nodes:                       # first occurrence wins (RTL/eldest-first)
        by_name.setdefault((n.get("name") or "").strip(), n)
    for c in cands:
        if c in by_name:
            return _node_ref(byid, by_name[c], matched=True)

    # 2. positional fallback
    if pos < len(nodes):
        return _node_ref(byid, nodes[pos], matched=False)
    return {"name": None, "father": None, "sons": None, "matched": False}


# --- data loading -----------------------------------------------------------------

def load_blocks(book: str, books_dir: str) -> list[dict]:
    """Every block for the book, in section/reading order, with both readers' columns."""
    ocr_dir = os.path.join(books_dir, book, OCR_DIR)
    files = sorted((f for f in os.listdir(ocr_dir) if f.endswith(".jsonl")),
                   key=lambda s: int(s.split("_")[0]))
    blocks = []
    for f in files:
        for line in open(os.path.join(ocr_dir, f)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw = r.get("raw", {})
            paddle = (raw.get("paddle", {}).get("columns")) or []
            vision = _lines((raw.get("vision", {}).get("text")) or "")
            gemini = _lines((raw.get("gemini", {}).get("text")) or "")
            # Mistral: markdown is the flat transcription; strip its leading "# " / "|" noise.
            mtext = (raw.get("mistral", {}).get("markdown")) or ""
            mistral = _lines(re.sub(r"^[#>|\-\s]+", "", mtext, flags=re.M))
            blocks.append({
                "id": r["id"],
                "section": r.get("id", "").rsplit("_", 2)[0],
                "generation": r.get("generation"),
                "paddle": paddle,
                "vision": vision,
                "gemini": gemini,
                "mistral": mistral,
            })
    return blocks


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def slice_ids(book: str, books_dir: str) -> set:
    """Block ids that have slice records (so the QA page knows whether to offer the overlay)."""
    sdir = os.path.join(books_dir, book, SLICE_DIR)
    ids = set()
    if not os.path.isdir(sdir):
        return ids
    for f in os.listdir(sdir):
        if not f.endswith(".jsonl") or f.endswith(".vision.jsonl"):
            continue
        for line in open(os.path.join(sdir, f)):
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def load_slice_reads(book: str, books_dir: str) -> dict:
    """{block_id: {reader: [texts by piece idx]}} from bio/4_slice, for the NEW slice-based
    readings shown above the whole-crop readers. Readers: 'spaddle','sgemini','svision'
    (s = sliced). Paddle/Gemini live in {stem}.jsonl pieces; vision in {stem}.vision.jsonl.
    A reader is omitted for a block if it has no non-empty text (so the UI can skip it).
    ?ERR/?EMPTY are treated as empty. {} if the slice stage hasn't run."""
    sdir = os.path.join(books_dir, book, SLICE_DIR)
    out: dict[str, dict] = {}
    if not os.path.isdir(sdir):
        return out

    def clean(v):
        if v in (None, "?ERR", "?EMPTY"):
            return ""
        # Each strip is ONE column -> flatten any newlines/whitespace the model returned, or
        # the vertical-rl `white-space:pre` slot would render them as multiple L-to-R lines.
        return re.sub(r"\s+", "", v)

    for f in os.listdir(sdir):
        if not f.endswith(".jsonl") or f.endswith(".vision.jsonl"):
            continue
        for line in open(os.path.join(sdir, f)):
            if not line.strip():
                continue
            r = json.loads(line)
            pieces = r.get("pieces", [])
            spaddle = [clean(p.get("paddle")) for p in pieces]
            sgemini = [clean(p.get("gemini")) for p in pieces]
            rd = {}
            if any(spaddle):
                rd["spaddle"] = spaddle
            if any(sgemini):
                rd["sgemini"] = sgemini
            out[r["id"]] = rd
    # fold in vision sidecars
    for f in os.listdir(sdir):
        if not f.endswith(".vision.jsonl"):
            continue
        for line in open(os.path.join(sdir, f)):
            if not line.strip():
                continue
            r = json.loads(line)
            texts = [clean(t) for t in r.get("vision", [])]
            if any(texts):
                out.setdefault(r["id"], {})["svision"] = texts
    return out


def slice_overlay(book: str, books_dir: str, bid: str):
    """Recompute the geometric slice for one block into a SINGLE coordinate frame that
    matches the TRIMMED crop, so the QA page can draw strip outlines that line up exactly.

    The stored 4_slice boxes are in mixed local frames (father/name are right-strip-local;
    cols are body-local; y excludes the trimmed top margin). Rather than reconstruct those
    offsets, we re-run s4_slice.slice_block + trim_rules here (cheap, local) and translate
    every piece box into trimmed-full-crop pixels. Returns (trimmed_png_bytes, meta) where
    meta = {"w","h","pieces":[{"kind","box":[x0,y0,x1,y1]}]}. box coords are in trimmed px.
    """
    import io
    from PIL import Image
    from src.bio import s4_slice as S
    fp = os.path.join(books_dir, book, SEG_DIR, f"{bid}.png")
    if not os.path.exists(fp):
        return None, None
    img = Image.open(fp).convert("RGB")
    trimmed, _top = S.trim_rules(img)      # deterministic; matches the stored boxes' frame
    Wt = trimmed.width
    right_x0 = Wt - S.RIGHT_STRIP_W          # x offset of the right strip in trimmed coords

    def _translate(stored_pieces):
        """Map STORED piece boxes into one trimmed-full-crop frame so the overlay matches the
        strips that were ACTUALLY ocr'd. NEW slices store father/name in ABSOLUTE coords; OLD
        slices stored them right-strip-local (x within [0, RIGHT_STRIP_W]). Detect the old
        format by x1 <= RIGHT_STRIP_W and shift those by right_x0; leave absolute boxes as-is."""
        out = []
        for p in stored_pieces:
            box = p.get("box")
            if not box:
                continue
            x0, y0, x1, y1 = box
            if p["kind"] in ("father", "name") and x1 <= S.RIGHT_STRIP_W + 1:
                x0 += right_x0; x1 += right_x0     # old relative-frame box
            out.append({"kind": p["kind"], "box": [x0, y0, x1, y1]})
        return out

    # Prefer the STORED slice (the boxes the OCR'd strips came from); fall back to a live
    # recompute only when this block has no stored record yet.
    stored = _stored_slice_pieces(book, books_dir, bid)
    if stored:
        pieces = _translate(stored)
    else:
        # live recompute -- slice_block now returns ABSOLUTE trimmed-crop boxes, so just map.
        pieces = [{"kind": p.kind, "box": list(p.box)} for p in S.slice_block(img)]
    buf = io.BytesIO(); trimmed.save(buf, format="PNG")
    return buf.getvalue(), {"w": trimmed.width, "h": trimmed.height, "pieces": pieces}


def _stored_slice_pieces(book: str, books_dir: str, bid: str):
    """The stored pieces (with boxes) for one block from 4_slice/{stem}.jsonl, or [] if none."""
    stem = bid.rsplit("_", 2)[0]
    path = os.path.join(books_dir, book, SLICE_DIR, f"{stem}.jsonl")
    if not os.path.exists(path):
        return []
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if r["id"] == bid:
                return r.get("pieces", [])
    return []


# The sons marker is just the substring 生子 -- the book doesn't always write the count or
# the 名 suffix (生子一 / 生子三 / 生子奇 / 生子一殁 / 生子N名 all occur). Anchoring on 生子
# alone is the reliable signal; strict 生子N名 missed ~47 blocks.
_SONS_START = re.compile("生子")

# Variant-glyph normalization: readers spell the SAME character different ways (Gemini favors
# 歿 for 殁 U+6B81 -- the form the book uses). Normalize to the canonical form EVERYWHERE (safe
# -- same character) so a pure variant difference never counts as a disagreement. Extend freely.
_VARIANT_NORM = {"歿": "殁",    # {variant: canonical}
                 "緒": "绪",    # traditional 緒 (U+7DD2) -> simplified 绪 (U+7EEA), same char
                 "別": "别"}    # traditional/Japanese 別 (U+5225) -> simplified 别 (U+522B), same char


def norm_variants(s: str) -> str:
    if not s:
        return s
    return "".join(_VARIANT_NORM.get(c, c) for c in s)


# Misread PAIRS: two genuinely different characters a reader confuses. Unlike variants, we do
# NOT blanket-normalize (e.g. 究 is real in 研究生) -- we only resolve when Gemini and Claude
# DISAGREE at a char position on exactly this pair, picking the preferred (correct) one. The
# column still counts as a disagreement (stays magenta) so the reviewer confirms.
_MISREAD_PAIRS = {frozenset(("夭", "天")): "夭",     # 天 is the misread of 夭 (die young)
                  frozenset(("究", "宪")): "宪",     # 究 is the misread of the gen-char 宪
                  frozenset(("黄", "黃")): "黄",     # 黃 (U+9EC3) misread of 黄 (U+9EC4) surname
                  frozenset(("桃", "祧")): "祧"}     # Claude misreads 祧 as 桃 in 双祧承嗣 (dual-
                  # inheritance). Safe as a disagreement-only pair: 桃 is a real NAME char but
                  # then both readers AGREE on 桃, so this only fires in the 双祧 context.


def strip_leading_line(g: str, v: str) -> tuple:
    """A residual top rule-line that trim_rules missed gets OCR'd as a leading 一 by one reader.
    If g and v differ ONLY by a leading 一 on one side, drop it from both so they agree and the
    column isn't flagged. Only acts when it's the sole difference -- a legitimate leading 一
    (一九三三年, 一月, where both readers agree) is untouched."""
    if not g or not v or g == v:
        return g, v
    if g.startswith("一") and g[1:] == v:
        return g[1:], v
    if v.startswith("一") and v[1:] == g:
        return g, v[1:]
    return g, v


def resolve_misreads(g: str, v: str) -> str:
    """Return g with each char resolved to the preferred glyph where g and v disagree on a
    known misread pair (position-aligned). Non-disagreements and unequal lengths are untouched."""
    if not g or len(g) != len(v):
        return g
    out = list(g)
    for k in range(len(g)):
        if g[k] != v[k]:
            pref = _MISREAD_PAIRS.get(frozenset((g[k], v[k])))
            if pref:
                out[k] = pref
    return "".join(out)


def _son_score(a: str, b: str) -> float:
    """Conservative similarity for matching a prior son name to a column: shared generation
    char (first char) OR high char overlap. 0 = don't match (won't hijack a date/desc column).
    Mirror of the JS _sonScore in the Fill-sons button."""
    if not a or not b:
        return 0.0
    ca, cb = list(a), list(b)
    gen = 1 if ca[0] == cb[0] else 0
    setb = set(cb)
    overlap = sum(1 for c in ca if c in setb) / max(len(ca), len(cb))
    lensim = 1 - abs(len(ca) - len(cb)) / max(len(ca), len(cb))
    if not gen and overlap < 0.5:
        return 0.0
    return gen * 2 + overlap + lensim * 0.5


def detect_fields(cols: list[str]) -> dict:
    """Best-effort field detection over one reader's columns (column indices).

    Returns ``{father_idx, name_idx, son_idxs}`` -- the columns to emphasize:
    - father header = column 0 (the horizontal ``子之X`` caption)
    - own name      = column 1 (the person's given name)
    - sons          = columns after a ``生子`` marker, up to ``生女`` / a new clause
    All are best-effort; the reviewer can correct them per column in the UI.
    """
    father_idx = 0 if cols else None
    name_idx = 1 if len(cols) > 1 else None
    son_idxs = []
    start = next((i for i, c in enumerate(cols) if _SONS_START.search(c)), None)
    if start is not None:
        for i in range(start + 1, len(cols)):
            c = cols[i]
            if re.search("生女", c) or re.match("[配继殁歿葬享寿卒]", c):
                break
            son_idxs.append(i)
    return {"father_idx": father_idx, "name_idx": name_idx, "son_idxs": son_idxs}


def analyze(block: dict) -> dict:
    """Per-block review aids across all readers.

    - ``prefill``: the Verified box default = Claude's (vision) full read, joined by \\n.
    - ``diff``: per vision column, True where its text appears in no Paddle column.
    - ``fields``: ``{reader: {father_idx, name_idx, son_idxs}}`` for every reader present,
      so each row can emphasize its own detected name/father/sons. The Verified row uses
      the vision fields as its default (the reviewer can reassign per column).
    """
    vision = block.get("vision", [])
    paddle = block.get("paddle", [])
    pset = set(paddle)
    diff = [v not in pset for v in vision]
    fields = {rd: detect_fields(block.get(rd, []))
              for rd in ("vision", "gemini", "mistral", "paddle")}
    return {
        "prefill": "\n".join(vision),
        "diff": diff,
        "fields": fields,
        # convenience: vision fields drive the Verified row's default highlights
        "name_idx": fields["vision"]["name_idx"],
        "father_idx": fields["vision"]["father_idx"],
        "son_idxs": fields["vision"]["son_idxs"],
    }


def verified_path(book: str, data_dir: str) -> str:
    return os.path.join(data_dir, f"{book}_bio_verified.json")


def load_verified(book: str, data_dir: str) -> dict:
    p = verified_path(book, data_dir)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_verified(book: str, data_dir: str, bid: str, text: str) -> None:
    p = verified_path(book, data_dir)
    cur = load_verified(book, data_dir)
    if text.strip():
        cur[bid] = text
    else:
        cur.pop(bid, None)
    with open(p, "w") as fh:
        json.dump(cur, fh, ensure_ascii=False, indent=1)


def fields_path(book: str, data_dir: str) -> str:
    """Reviewer's per-block field-type overrides for Verified columns."""
    return os.path.join(data_dir, f"{book}_bio_fields.json")


def load_fields(book: str, data_dir: str) -> dict:
    p = fields_path(book, data_dir)
    return json.load(open(p)) if os.path.exists(p) else {}


def load_prior_names(book: str, data_dir: str) -> dict:
    """William's earlier NAME-ONLY verification, snapshotted before the full-verify reset.
    {bid: {"father","name","sons":[...]}} extracted from {book}_bio_verified_names.json using
    {book}_bio_fields_names.json overrides (else auto-detect). {} if no snapshot exists."""
    vp = os.path.join(data_dir, f"{book}_bio_verified_names.json")
    fp = os.path.join(data_dir, f"{book}_bio_fields_names.json")
    if not os.path.exists(vp):
        return {}
    verified = json.load(open(vp))
    fields = json.load(open(fp)) if os.path.exists(fp) else {}
    out = {}
    for bid, text in verified.items():
        cols = text.split("\n")
        ov = fields.get(bid) or {}
        det = detect_fields(cols)
        # explicit override wins; "none" suppresses. father/name = single, sons = set.
        def pick(kind, auto_idx):
            # a column is `kind` if overridden to it, or (no override) auto-detected as it
            idxs = [int(k) for k, v in ov.items() if v == kind]
            if idxs:
                return idxs
            if kind == "son":
                return [i for i in (auto_idx or []) if str(i) not in ov]
            return [auto_idx] if (auto_idx is not None and str(auto_idx) not in ov) else []
        f_idx = pick("father", det["father_idx"])
        n_idx = pick("name", det["name_idx"])
        s_idx = pick("son", det["son_idxs"])
        out[bid] = {
            "father": cols[f_idx[0]] if f_idx and f_idx[0] < len(cols) else "",
            "name": cols[n_idx[0]] if n_idx and n_idx[0] < len(cols) else "",
            "sons": [cols[i] for i in s_idx if i < len(cols) and cols[i].strip()],
        }
    return out


def save_fields(book: str, data_dir: str, bid: str, fields: dict) -> None:
    """Persist ``{slot_index: 'father'|'name'|'son'|'none'}`` for one block.

    ``'none'`` is a REAL, persisted override meaning "this column is not a field" -- it
    must survive reload so it overrides auto-detection (e.g. the reviewer clearing a column
    that was wrongly auto-tagged as a son). Only genuinely absent keys are dropped.
    """
    p = fields_path(book, data_dir)
    cur = load_fields(book, data_dir)
    clean = {k: v for k, v in (fields or {}).items() if v}   # v is a non-empty string
    if clean:
        cur[bid] = clean
    else:
        cur.pop(bid, None)
    with open(p, "w") as fh:
        json.dump(cur, fh, ensure_ascii=False, indent=1)


# --- HTML -------------------------------------------------------------------------

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Bio OCR QA — {book}</title>
<style>
  :root {{ --bg:#faf9f7; --fg:#1a1a1a; --muted:#8a8a8a; --line:#ddd; --ok:#0a7f3f;
           --paddle:#1558b0; --vision:#8a5a00; --card:#fff;
           /* field colors: father header (blue underline), own name (green), sons (red) */
           --father:#1558b0; --name:#0a7f3f; --son:#c0392b; }}
  * {{ box-sizing:border-box; }}
  html,body {{ height:100%; }}
  /* Match the website's bio font stack (PingFang SC etc.) -- renders CJK closer to the
     printed characters than bare system-ui, which falls back to a blockier CJK face. */
  :root {{ --font-cjk: ui-sans-serif, system-ui, -apple-system, "Helvetica Neue", Arial,
           "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }}
  body {{ margin:0; font:14px/1.5 var(--font-cjk); background:var(--bg); color:var(--fg);
          display:flex; flex-direction:column; }}
  header {{ background:var(--card); border-bottom:1px solid var(--line);
            padding:8px 16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }}
  header .bid {{ font-weight:600; font-size:15px; }}
  header .prog {{ color:var(--muted); }}
  header .status.done {{ color:var(--ok); }}
  header .status.todo {{ color:#c0392b; }}
  .keys {{ color:var(--muted); font-size:12px; margin-left:auto; }}
  .keys kbd {{ background:var(--bg); border:1px solid var(--line); border-radius:4px;
               padding:0 4px; font:inherit; }}
  .legend {{ color:var(--muted); font-size:12px; }}
  .legend b {{ font-weight:600; }}
  /* Stacked full-width rows: crop -> verified -> claude -> paddle, scroll within block. */
  #stage {{ flex:1; padding:16px; overflow:auto; }}
  .rows {{ display:flex; flex-direction:column; gap:14px; }}
  .cell {{ display:flex; flex-direction:column; gap:4px; }}
  .cell .lab {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
  .cell.paddle .lab {{ color:var(--paddle); }}
  .cell.vision .lab {{ color:var(--vision); }}
  .cell.verified .lab {{ color:var(--ok); }}
  /* Height set per-block in JS so a scanned glyph renders ~= the OCR font size
     (height = FONT_PX * chars-in-tallest-column). Natural aspect, right-aligned (RTL). */
  img.crop {{ max-width:100%; border:1px solid var(--line); border-radius:6px;
              background:#fff; object-fit:contain; object-position:right top;
              align-self:flex-end; display:block; }}
  /* Strip-outline overlay: a positioned wrapper the exact rendered size of the crop, with
     one thin box per slice piece. Kept a 1px HAIRLINE so a slightly-misaligned strip edge
     never sits on top of a glyph; colors match the field palette but are only the border. */
  .cropwrap {{ position:relative; align-self:flex-end; max-width:100%; line-height:0; }}
  .cropwrap .striplayer {{ position:absolute; inset:0; pointer-events:none; }}
  .cropwrap .strip {{ position:absolute; border:1px solid; border-radius:2px;
                      box-sizing:border-box; }}
  .cropwrap .strip.father {{ border-color:rgba(21,88,176,.85); }}   /* blue */
  .cropwrap .strip.name   {{ border-color:rgba(10,127,63,.85); }}   /* green */
  .cropwrap .strip.col    {{ border-color:rgba(192,57,43,.6); }}    /* red, lighter */
  .cropwrap .strip .stag {{ position:absolute; top:-1px; left:-1px; font:9px/1 var(--font-cjk);
                            background:rgba(255,255,255,.8); color:var(--muted); padding:0 2px;
                            border-radius:2px; }}
  .sliceToggle {{ font-size:11px; color:var(--muted); cursor:pointer; user-select:none; }}
  .sliceToggle input {{ vertical-align:middle; margin:0 3px 0 0; }}
  /* vertical, right-to-left: columns run right->left, chars top->bottom (mirrors the scan). */
  .vpanel {{ display:flex; flex-direction:row-reverse; justify-content:flex-start;
             align-items:flex-start; gap:0; padding:8px; border:1px solid var(--line);
             border-radius:6px; background:#fff; overflow-x:auto; }}
  /* OCR glyphs sized to roughly match the scanned characters for column-by-column compare. */
  /* Fixed per-column slot width so Claude & Paddle rows line up column-for-column
     (a missing column shows as an empty .gap slot of the same width). */
  /* Every text row (Claude, Paddle, AND Verified) uses these identical fixed-width slots,
     so the three rows are structurally identical and cannot misalign. Slot width is set
     per-block from the scan's rendered pixel width (--slot). */
  .vcol {{ writing-mode:vertical-rl; text-orientation:upright; white-space:pre;
           /* CJK face first for the pure-Chinese columns -- same as the website's bio text. */
           font-family:"PingFang SC","Hiragino Sans GB","Microsoft YaHei",var(--font-cjk);
           /* text 50% larger than the crop-matching size (FONT_PX=30) for readability;
              the scan crop height still uses FONT_PX, so the image is unchanged. */
           font-size:45px; line-height:1.45; padding:1px 0; border-radius:3px;
           /* Slot width from the scan (--slot), but never narrower than the glyph so the
              larger font can't clip; every row uses this rule so they stay column-aligned. */
           flex:0 0 max(var(--slot,40px), 1.1em); width:max(var(--slot,40px), 1.1em);
           text-align:center; }}
  .vcol.gap {{ background:repeating-linear-gradient(45deg,#f4f4f4,#f4f4f4 4px,#fafafa 4px,#fafafa 8px); }}
  .vcol.diff {{ background:#ffe9d6; }}
  /* FIELD-detection disagreement with Claude: this reader tagged this column as a
     different field (father/name/son/none) than Claude did. Dashed magenta outline,
     distinct from the orange text-diff background. */
  .vcol.fdiff {{ outline:2px dashed #b5179e; outline-offset:-2px; }}     /* Gemini vs Claude */
  .vcol.fdiff-mr {{ outline:2px dashed #0d9488; outline-offset:-2px; }}   /* misread pair, auto-resolved (teal) */
  .vcol.fdiff-lo {{ outline:1px dotted #b8b8b8; outline-offset:-1px; }}   /* Paddle-only (muted) */
  /* Column whose value was auto-filled from William's prior verification (not OCR): green
     tint + a solid green top bar, so it's visibly "mine, verified" vs a raw OCR column. */
  .vcol.prior-filled {{ background:#e6f6ec; box-shadow:inset 0 3px 0 #0a7f3f; }}
  /* Per-CHARACTER diff box: the exact glyph where Gemini and Claude slice reads differ. */
  .cdiff {{ outline:2px solid #b5179e; outline-offset:1px; border-radius:3px;
            background:#fbeaf5; }}
  /* Field emphasis: father = blue underline, own name = green bold, sons = red bold.
     Applied to the glyphs so the same field reads down every reader row + Verified. */
  .vcol.f-father {{ color:var(--father); text-decoration:underline; text-underline-offset:3px;
                    text-decoration-thickness:2px; }}
  /* Father header prints HORIZONTALLY (子之X) in the book -- render it left-to-right,
     top-aligned, so it reads like the scan rather than stacked vertically. It keeps the
     same slot width so the rest of the columns stay aligned. */
  .vcol.horiz {{ writing-mode:horizontal-tb; text-orientation:mixed; white-space:nowrap;
                 align-self:flex-start; overflow:visible;
                 /* size to the horizontal text, not the vertical slot width, so 子之X
                    isn't clipped; it's the rightmost column so growing left is harmless. */
                 flex:0 0 auto; width:auto; min-width:var(--slot,40px); }}
  .vcol.edit.horiz {{ writing-mode:horizontal-tb; }}
  .vcol.f-name {{ color:var(--name); font-weight:800; }}
  .vcol.f-son  {{ color:var(--son);  font-weight:800; }}
  /* Verified columns are the same slot, but editable. */
  .vcol.edit {{ cursor:text; min-height:2em; }}
  .vcol.edit:focus {{ outline:2px solid var(--ok); background:#f0faf3; }}
  /* field markers on the Verified row also show a small colored top border so the
     assignment is visible even before you read the glyph. */
  .vcol.edit.f-father {{ border-top:3px solid var(--father); }}
  .vcol.edit.f-name   {{ border-top:3px solid var(--name); }}
  .vcol.edit.f-son    {{ border-top:3px solid var(--son); }}
  .cell.verified .vpanel {{ min-height:40vh; }}
  /* Graph reference row: horizontal chips (father / name / sons) to copy from. */
  .cell.graph .lab {{ color:#555; }}
  .gline {{ display:flex; gap:18px; flex-wrap:wrap; align-items:baseline;
            padding:8px 10px; border:1px solid var(--line); border-radius:6px;
            background:#f7f9ff; }}
  .gchip {{ display:inline-flex; gap:6px; align-items:baseline; }}
  /* Prior verified box: William's name-only QA, kept separate so full-verify never stomps it.
     A field chip gets a dashed magenta outline when it DIFFERS from the slice readers. */
  .cell.prior .lab {{ color:#7a5c00; }}
  .cell.prior .gline {{ background:#fffdf5; border-color:#e6d9a8; }}
  .gchip.pdiff {{ outline:2px dashed #b5179e; outline-offset:2px; border-radius:4px; }}
  .glab {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .gval {{ font-family:"PingFang SC","Hiragino Sans GB",var(--font-cjk); font-size:26px;
           user-select:text; }}  /* normal selection: drag/double-click to grab one char */
  .gtag {{ font-size:11px; font-weight:700; padding:1px 6px; border-radius:8px; margin-left:8px; }}
  .gtag.ok {{ background:#e6f6ec; color:var(--ok); }}
  .gtag.guess {{ background:#fdeecf; color:#8a5a00; }}
  .cell.gemini .lab {{ color:#6d28d9; }}
  .cell.mistral .lab {{ color:#b45309; }}
  .fieldbar {{ font-size:12px; color:var(--muted); }}
  .fieldbar b {{ font-weight:700; }}
  .splitbtn {{ font:inherit; font-size:12px; padding:1px 8px; border:1px solid var(--line);
               border-radius:10px; background:#eef4ff; color:#1558b0; cursor:pointer; }}
  .splitbtn:hover {{ background:#dce8ff; }}
  /* NEW per-strip slice readings: a lightly-tinted group above the whole-crop readers. */
  .slicegroup {{ display:flex; flex-direction:column; gap:8px; padding:8px;
                 border:1px dashed #9ca3af; border-radius:8px; background:#f6f7fb; }}
  .slicehdr {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em;
               color:#6b7280; font-weight:700; }}
  .cell.sliceread .lab {{ color:#3730a3; }}
  .cell.sliceread.spaddle .lab {{ color:var(--paddle); }}
  .cell.sliceread.sgemini .lab {{ color:#6d28d9; }}
  .cell.sliceread.svision .lab {{ color:var(--vision); }}
</style></head><body>
<header>
  <span class="bid" id="bid">…</span>
  <span class="prog" id="pos"></span>
  <span class="status" id="status"></span>
  <span class="prog" id="prog"></span>
  <span class="legend"><b style="color:var(--father);text-decoration:underline">father</b> · <b style="color:var(--name)">name</b> · <b style="color:var(--son)">son</b> · <span style="outline:2px dashed #b5179e;padding:0 3px">magenta</span>=Gemini↔Claude conflict · <span style="outline:2px dashed #0d9488;padding:0 3px">teal</span>=auto-resolved misread (夭/天,究/宪,黄/黃,桃/祧) · <span style="outline:1px dotted #b8b8b8;padding:0 3px">gray dot</span>=Paddle-only · <span style="background:#e6f6ec;box-shadow:inset 0 3px 0 #0a7f3f;padding:0 3px">green</span>=from your prior verified</span>
  <span class="keys"><kbd>Shift</kbd>+<kbd>←/→</kbd> block · <kbd>Shift</kbd>+<kbd>↑/↓</kbd> to-review · <kbd>e</kbd> edit · <kbd>←/→</kbd> col · <kbd>f</kbd>/<kbd>n</kbd>/<kbd>s</kbd>/<kbd>x</kbd> set father/name/son/none · <kbd>Esc</kbd> stop · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save+next</span>
</header>
<div id="stage"><div class="rows" id="rows"></div></div>
<script>
const BOOK = {book_json};
let BLOCKS = [];
// (crop height comes from the scan's true aspect ratio, set on image load)
let VERIFIED = {{}};
let FIELDS = {{}};       // per-block override of Verified column field types: {{bid: {{slotIdx: 'father'|'name'|'son'|null}}}}
let CUR = 0;
const READERS = ["vision", "gemini", "mistral", "paddle"];
const READER_LABEL = {{vision:"Claude", gemini:"Gemini", mistral:"Mistral", paddle:"Paddle"}};

// The field type for one column of a reader, from that reader's detected fields.
function fieldClassFor(fields, i) {{
  if (!fields) return null;
  if (i === fields.father_idx) return "father";
  if (i === fields.name_idx) return "name";
  if ((fields.son_idxs || []).includes(i)) return "son";
  return null;
}}

function escapeHtml(s) {{ return (s||"").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c])); }}
function colsToText(cols) {{ return cols.join("\\n"); }}
function isDone(b) {{ return VERIFIED[b.id] !== undefined; }}
function isDisagree(b) {{ return colsToText(b.paddle) !== colsToText(b.vision); }}
function needsReview(b) {{ return !isDone(b) || isDisagree(b); }}

// Map Paddle's columns onto Claude's slot order (Claude = canonical). Each Claude slot i
// gets the Paddle column that matches its text (greedy, first unused match); Claude slots
// with no Paddle match stay empty (a gap in the Paddle row). Paddle columns that match no
// Claude slot are appended as extra slots at the end (with a gap in the Claude row) so no
// text is lost. Returns {{ slots: [claudeText...], paddleAt: [paddleText|null...] }} of
// equal length so the two rows line up column-for-column.
function alignToClaude(vision, paddle) {{
  const slots = vision.slice();          // Claude defines slot 0..n-1
  const usedP = new Array(paddle.length).fill(false);
  const paddleAt = slots.map(v => {{
    const j = paddle.findIndex((p, k) => !usedP[k] && p === v);
    if (j >= 0) {{ usedP[j] = true; return paddle[j]; }}
    return null;                          // gap: Paddle has no matching column here
  }});
  // leftover Paddle columns (no Claude match) -> extra trailing slots
  paddle.forEach((p, k) => {{ if (!usedP[k]) {{ slots.push(null); paddleAt.push(p); }} }});
  return {{ slots, paddleAt }};
}}

// Render one text row as fixed-width slots so every row's columns sit at the same x.
// `cells` is an array of {{text, cls[]}} (null text => an empty gap slot).
function slotRow(cells) {{
  const spans = cells.map(c => {{
    if (c.text == null && c.html == null) return `<span class="vcol gap"></span>`;
    // The father header renders horizontally (see .vcol.horiz).
    const cls = c.cls.includes("f-father") ? [...c.cls, "horiz"] : c.cls;
    // c.html (pre-built, e.g. per-char diff markup) wins over plain c.text.
    const inner = (c.html != null) ? c.html : escapeHtml(c.text);
    return `<span class="${{["vcol", ...cls].join(" ")}}">${{inner}}</span>`;
  }});
  return `<div class="vpanel">${{spans.join("") || "—"}}</div>`;
}}

// Variant-glyph normalization (mirror of the backend _VARIANT_NORM): the same character
// spelled differently by a reader (Gemini writes 歿 for 殁). Normalize for BOTH display and
// diffing so a pure variant difference is auto-corrected and never boxed.
const VARIANT_NORM = {{ "歿": "殁", "緒": "绪", "別": "别" }};
function normVar(s) {{ return (s||"").replace(/./g, c => VARIANT_NORM[c] || c); }}

// Per-character diff markup between two column strings (Claude vs Gemini). Aligned by longest
// common subsequence, so an inserted/dropped char (e.g. an extra prefix 一) boxes ONLY that char
// instead of shifting and boxing everything after it. Chars of THIS string outside the LCS are boxed.
function charDiffHtml(mine, other) {{
  const a = [...(mine||"")], b = [...(other||"")];
  const n = a.length, m = b.length;
  const L = Array.from({{length: n + 1}}, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i][j] = a[i] === b[j] ? L[i+1][j+1] + 1 : Math.max(L[i+1][j], L[i][j+1]);
  const keep = new Array(n).fill(false);
  let i = 0, j = 0;
  while (i < n && j < m) {{
    if (a[i] === b[j]) {{ keep[i] = true; i++; j++; }}
    else if (L[i+1][j] >= L[i][j+1]) i++;
    else j++;
  }}
  return a.map((ch, k) => keep[k] ? escapeHtml(ch)
                                  : `<span class="cdiff">${{escapeHtml(ch)}}</span>`).join("");
}}

function renderCurrent() {{
  const b = BLOCKS[CUR];
  // Remember where we are so a page reload returns to this block (per book).
  try {{ localStorage.setItem("qa_cur_" + BOOK, b.id); }} catch (e) {{}}
  const done = isDone(b), disagree = isDisagree(b);
  const verifiedText = done ? VERIFIED[b.id] : b.prefill;
  document.getElementById("bid").textContent = b.id;
  document.getElementById("pos").textContent = `${{CUR + 1}} / ${{BLOCKS.length}} · gen ${{b.generation}}`;
  const st = document.getElementById("status");
  st.textContent = done ? "✓ verified" : (disagree ? "⚠ readers differ" : "· unreviewed");
  st.className = "status " + (done ? "done" : "todo");
  document.getElementById("prog").textContent = `(${{Object.keys(VERIFIED).length}} verified)`;
  // The crop's height follows its TRUE aspect ratio (set on load below), not an OCR
  // column-length estimate -- a reader that merges text into one long column would
  // otherwise inflate the height and leave a blank band under the scan.

  // Every reader is aligned onto Claude's (vision) slot order so columns line up row-to-row.
  // Claude defines the slots; each other reader's columns are placed at the matching slot
  // (gap where absent), extras appended. Field highlights come from each reader's own
  // detected fields, so the same field reads down every row.
  // Build the slot list: Claude (vision) defines slots 0..n; each other reader's columns
  // that match no Claude slot are appended as trailing extra slots (null in `slots`) so no
  // reader's text is ever dropped.
  const vision = b.vision || [];
  let slots = vision.slice();
  for (const rd of READERS) {{
    if (rd === "vision") continue;
    const matched = new Set();
    const vcount = {{}};
    slots.forEach(s => {{ if (s != null) vcount[s] = (vcount[s]||0)+1; }});
    const seen = {{}};
    for (const c of (b[rd] || [])) {{
      seen[c] = (seen[c]||0)+1;
      if (seen[c] > (vcount[c]||0)) slots.push(null);   // extra column -> new trailing slot
    }}
  }}
  // For each reader, greedily place its columns onto the slot texts (gap where absent).
  function readerRow(rd) {{
    const cols = (b[rd] || []).slice();
    const used = new Array(cols.length).fill(false);
    const out = slots.map(s => {{
      if (s == null) return null;
      const j = cols.findIndex((c, k) => !used[k] && c === s);
      if (j >= 0) {{ used[j] = true; return cols[j]; }}
      return null;
    }});
    // drop leftover (unmatched) reader columns into the trailing null slots, in order
    let leftover = cols.filter((c, k) => !used[k]);
    for (let i = 0; i < out.length && leftover.length; i++) {{
      if (slots[i] == null && out[i] == null) out[i] = leftover.shift();
    }}
    return out;
  }}
  const rowText = {{}};
  for (const rd of READERS) rowText[rd] = (rd === "vision") ? slots.slice() : readerRow(rd);

  // Field class per (reader, slot): use that reader's detected fields, indexed by the
  // reader's OWN column position (find where this slot's text sits in the reader's columns).
  function readerFieldCls(rd, slotText) {{
    const f = (b.fields || {{}})[rd];
    if (!f || slotText == null) return null;
    const idx = (b[rd] || []).indexOf(slotText);
    return idx < 0 ? null : fieldClassFor(f, idx);
  }}

  const nSlots = slots.length;
  // Build each reader row's cells, with diff (vs Claude) + field highlight.
  const readerCellRows = {{}};
  // Claude's field-type at each SLOT (the reference the other readers are compared to).
  const claudeFieldAt = slots.map(s => readerFieldCls("vision", s));
  for (const rd of READERS) {{
    readerCellRows[rd] = slots.map((s, i) => {{
      const t = rowText[rd][i];
      if (t == null) return {{ text: null, cls: [] }};
      const cls = [];
      if (rd !== "vision" && t !== s) cls.push("diff");    // text differs from Claude here
      const fc = readerFieldCls(rd, t);
      if (fc) cls.push("f-" + fc);
      // FIELD-detection disagreement: this reader assigns a different field type to this
      // slot than Claude does (e.g. Gemini thinks this column is the father, Claude doesn't).
      // Mark with a dashed outline. (Claude is the reference, so it never flags itself.)
      if (rd !== "vision" && (fc || null) !== (claudeFieldAt[i] || null)) cls.push("fdiff");
      return {{ text: t, cls }};
    }});
  }}

  // Verified: editable columns. For an UNVERIFIED block the default columns come from the
  // PREFILL (the Gemini SLICE reading -- the best reader), NOT the whole-crop Claude `slots`.
  // For a verified block, the saved text is authoritative.
  const savedCols = VERIFIED[b.id] !== undefined ? VERIFIED[b.id].split("\\n") : null;
  const prefillCols = (b.prefill || "").split("\\n");
  const ov = FIELDS[b.id] || {{}};
  const vfields = (b.fields || {{}}).vision || {{}};
  // Auto-detect fields on the DEFAULT verified columns (prefill for unverified) so labels
  // align with what's actually shown: col0=father, col1=name, sons after a 生子 column.
  function detectFieldsOn(cols) {{
    const f = {{father_idx: cols.length ? 0 : null, name_idx: cols.length > 1 ? 1 : null,
               son_idxs: []}};
    const start = cols.findIndex(c => (c||"").includes("生子"));
    if (start >= 0) {{
      for (let i = start + 1; i < cols.length; i++) {{
        if (/生女/.test(cols[i]||"") || /^[配继殁歿葬享寿卒]/.test(cols[i]||"")) break;
        f.son_idxs.push(i);
      }}
    }}
    return f;
  }}
  // Fields backing the Verified row: reviewer overrides win; else auto-detect on the row's
  // own default columns (prefill=slice for unverified, saved for verified) rather than the
  // stale whole-crop vision indices.
  const baseCols = savedCols || prefillCols;
  const autoFields = detectFieldsOn(baseCols);
  function verifiedFieldCls(i) {{
    // A reviewer override wins over auto-detection. "none" means "explicitly not a field"
    // (return null so no color, but it still suppresses the auto-detected field).
    if (i in ov) return ov[i] === "none" ? null : ov[i];
    return fieldClassFor(autoFields, i);
  }}
  // Which FIELD TYPES do Claude & Gemini disagree on (father/name/son)? Compare the TEXT
  // each reader detected for that field; if it differs (or one is missing), the Verified
  // column(s) carrying that field get a dashed outline so the reviewer sees the conflict.
  const gfields = (b.fields || {{}}).gemini || {{}};
  function fieldTexts(rd, fld) {{
    const f=(b.fields||{{}})[rd]||{{}}, cols=b[rd]||[];
    if (fld==="father") return f.father_idx!=null ? [cols[f.father_idx]] : [];
    if (fld==="name")   return f.name_idx!=null ? [cols[f.name_idx]] : [];
    return (f.son_idxs||[]).map(i=>cols[i]);   // son: the set of son columns
  }}
  const disagreeField = {{}};   // {{father:bool,name:bool,son:bool}}
  for (const fld of ["father","name","son"]) {{
    const c=fieldTexts("vision",fld).join("|"), g=fieldTexts("gemini",fld).join("|");
    disagreeField[fld] = (c !== g);
  }}
  // The Verified row must show EVERY saved column, even if the reader-derived `slots` are
  // fewer (readers split columns differently than the reviewer did). Length = max(slots,
  // saved) so a saved edit is never truncated/dropped on reload. Unsaved blocks default to
  // one editable slot per Claude slot (prefill).
  // Default columns = saved (verified) else prefill (=Gemini slice). Length covers all of
  // them so nothing is dropped on reload.
  const nVerified = baseCols.length;
  // Per-column 3-way slice disagreement (gemini vs claude vs paddle): dash any column where
  // they don't all agree. baseCols == the Gemini slice columns for unverified blocks, so
  // sliceDiff[i] aligns to column i. This is the FULL-TEXT verification signal.
  const sliceDiff = (!savedCols && b.sliceDiff) ? b.sliceDiff : [];
  // Columns whose prefill text came from William's PRIOR verified names (father/name/sons),
  // overriding OCR -- shown with a distinct marker so it's clear the value is his, not OCR.
  const priorFilled = new Set((!savedCols && b.priorFilled) ? b.priorFilled : []);
  let verifiedRow = "";
  for (let i = 0; i < nVerified; i++) {{
    const text = baseCols[i] ?? "";
    const fc = verifiedFieldCls(i);
    const cls = ["vcol", "edit"].concat(fc ? ["f-" + fc] : []);
    if (fc === "father") cls.push("horiz");   // father header reads horizontally
    // Disagreement dash: magenta for a genuine Gemini-vs-Claude conflict (fdiff); teal for an
    // auto-resolved misread pair (夭/天, 究/宪 -- likely fine, fdiff-mr); muted gray for
    // Paddle-only (fdiff-lo, the noisy reader).
    if (sliceDiff[i] === "gc") cls.push("fdiff");
    else if (sliceDiff[i] === "misread") cls.push("fdiff-mr");
    else if (sliceDiff[i] === "paddle") cls.push("fdiff-lo");
    if (priorFilled.has(i)) cls.push("prior-filled");   // value from prior verification
    verifiedRow += `<span class="${{cls.join(" ")}}" contenteditable="plaintext-only" data-i="${{i}}">${{escapeHtml(text)}}</span>`;
  }}

  const readerRowsHtml = READERS.map(rd =>
    `<div class="cell ${{rd}}"><span class="lab">${{READER_LABEL[rd]}}</span>${{slotRow(readerCellRows[rd])}}</div>`
  ).join("");

  // NEW: per-strip slice readings (each reader's father/name/col strips, in RTL order as
  // sliced). Shown ABOVE the whole-crop reader rows. A reader with no data is skipped.
  // Field classes come from the slice structure itself: piece 0 = father, piece 1 = name,
  // pieces after a 生子 marker = sons. Rendered with the same fixed-width vertical slots.
  const SLICE_READERS = ["svision", "sgemini", "spaddle"];
  const SLICE_LABEL = {{svision:"Claude ▸ slice", sgemini:"Gemini ▸ slice", spaddle:"Paddle ▸ slice"}};
  const sr = b.sliceReads || {{}};
  function sliceFieldCls(texts, i) {{
    if (i === 0) return "father";
    if (i === 1) return "name";
    // sons: pieces after the first piece containing 生子, until 生女/配/继/殁/葬
    let start = texts.findIndex(t => (t||"").includes("生子"));
    if (start >= 0 && i > start) {{
      for (let k = start + 1; k <= i; k++) {{
        if (/生女|^[配继殁歿葬享寿卒]/.test(texts[k]||"")) return null;
      }}
      return "son";
    }}
    return null;
  }}
  // For Claude/Gemini rows, box the SPECIFIC characters where the two differ (position-wise),
  // so a column-level disagreement points to the exact glyph. Paddle is compared to neither.
  // Normalize variant glyphs (歿->殁) for both display and diffing so a pure variant spelling
  // is auto-corrected and never boxed.
  const flat = (rd) => (sr[rd]||[]).map(t => normVar((t||"").replace(/\\s/g,"")));
  const gCols = flat("sgemini"), vCols = flat("svision");
  const sliceRowsHtml = SLICE_READERS.filter(rd => (sr[rd]||[]).length).map(rd => {{
    const texts = sr[rd];
    const cells = texts.map((t, i) => {{
      const cls = [];
      const fc = sliceFieldCls(texts, i);
      if (fc) cls.push("f-" + fc);
      const norm = t ? normVar((t||"").replace(/\\s/g,"")) : t;
      const cell = {{ text: (norm === "" ? null : norm), cls }};
      // char-diff markup: Gemini vs Claude only (the trusted pair), on normalized text
      if (rd === "sgemini" && norm) cell.html = charDiffHtml(norm, vCols[i]);
      else if (rd === "svision" && norm) cell.html = charDiffHtml(norm, gCols[i]);
      return cell;
    }});
    return `<div class="cell sliceread ${{rd}}"><span class="lab">${{SLICE_LABEL[rd]}}</span>${{slotRow(cells)}}</div>`;
  }}).join("");
  const sliceBlock = sliceRowsHtml
    ? `<div class="slicegroup"><div class="slicehdr">NEW — per-strip slice readings (geometric column slicing)</div>${{sliceRowsHtml}}</div>`
    : "";

  // Graph reference: what the tree says this block's person is (name/father/sons), by
  // position in its subgraph generation. A flat, copyable line -- the truth to copy over
  // when the OCR is wrong. Empty when the graph has no node at this position.
  const g = b.graph;
  let graphHtml = "";
  if (g && (g.name || g.father || (g.sons && g.sons.length))) {{
    const chip = (label, val, color) => val
      ? `<span class="gchip"><span class="glab" style="color:${{color}}">${{label}}</span>` +
        `<span class="gval">${{escapeHtml(val)}}</span></span>` : "";
    const sons = (g.sons || []).map(s => escapeHtml(s)).join("、");
    // matched = found this node by NAME (trustworthy); else positional guess.
    const tag = g.matched
      ? `<span class="gtag ok">✓ name-matched</span>`
      : `<span class="gtag guess">≈ positional guess</span>`;
    graphHtml = `<div class="cell graph"><span class="lab">Graph (tree) — reference to copy from ${{tag}}</span>
      <div class="gline">${{chip("father", g.father, "var(--father)")}}` +
      `${{chip("name", g.name, "var(--name)")}}` +
      `${{sons ? `<span class="gchip"><span class="glab" style="color:var(--son)">sons</span><span class="gval">${{sons}}</span></span>` : ""}}` +
      `</div></div>`;
  }}

  // PRIOR VERIFIED (William's earlier name-only QA, snapshotted). Shows his verified
  // father/name/sons so the full-verify pass never stomps them. Each field is highlighted
  // if it DISAGREES with the slice readers: father/name compared directly to the Gemini/Claude
  // slice father/name; sons compared as a NAME SET (slice columns don't align by index).
  let priorHtml = "";
  const pn = b.priorNames;
  if (pn && (pn.father || pn.name || (pn.sons && pn.sons.length))) {{
    const sf = (r,i) => {{ const a=(b.sliceReads||{{}})[r]||[]; return (a[i]||"").replace(/\\s/g,""); }};
    // slice father/name = piece 0 / 1 of gemini (fallback claude); son set from gemini slice.
    const gcols = ((b.sliceReads||{{}}).sgemini||[]).map(t=>(t||"").replace(/\\s/g,""));
    const vcols = ((b.sliceReads||{{}}).svision||[]).map(t=>(t||"").replace(/\\s/g,""));
    const sliceFather = [gcols[0], vcols[0]].filter(Boolean);
    const sliceName   = [gcols[1], vcols[1]].filter(Boolean);
    const sonSet = (cols) => {{ const s=cols.findIndex(c=>c.includes("生子"));
      const out=[]; if(s>=0){{for(let k=s+1;k<cols.length;k++){{if(/生女|^[配继殁歿葬享寿卒]/.test(cols[k]))break; if(cols[k])out.push(cols[k]);}}}} return out; }};
    const sliceSons = new Set([...sonSet(gcols), ...sonSet(vcols)]);
    // Flag if the prior value differs from ANY present slice reader (surfaces every conflict,
    // even when one reader agrees with the prior and another doesn't).
    const disF = pn.father && sliceFather.length && sliceFather.some(x=>x!==pn.father);
    const disN = pn.name && sliceName.length && sliceName.some(x=>x!==pn.name);
    const priorSonSet = new Set(pn.sons||[]);
    const disS = (pn.sons&&pn.sons.length) && sliceSons.size &&
                 (pn.sons.some(s=>!sliceSons.has(s)) || [...sliceSons].some(s=>!priorSonSet.has(s)));
    const chip = (label,val,color,dis) => val
      ? `<span class="gchip ${{dis?'pdiff':''}}"><span class="glab" style="color:${{color}}">${{label}}</span>`+
        `<span class="gval">${{escapeHtml(val)}}</span></span>` : "";
    const sonsStr = (pn.sons||[]).map(escapeHtml).join("、");
    priorHtml = `<div class="cell prior"><span class="lab">Your prior verified (names) — dashed = differs from slice readers</span>
      <div class="gline">${{chip("father",pn.father,"var(--father)",disF)}}`+
      `${{chip("name",pn.name,"var(--name)",disN)}}`+
      `${{sonsStr?`<span class="gchip ${{disS?'pdiff':''}}"><span class="glab" style="color:var(--son)">sons</span><span class="gval">${{sonsStr}}</span></span>`:""}}`+
      `</div></div>`;
  }}

  const rows = document.getElementById("rows");
  rows.innerHTML = `
    <div class="cell crop"><span class="lab">Original (scan)${{b.hasSlice ? ` · <label class="sliceToggle"><input type="checkbox" id="stripToggle" checked>strip outlines</label>` : ``}}</span>
      <div class="cropwrap" id="cropwrap">
        <img class="crop" id="cropimg" src="${{b.hasSlice ? `/slice/${{b.id}}` : `/img/${{b.id}}`}}">
        <div class="striplayer" id="striplayer"></div>
      </div></div>
    ${{graphHtml}}
    ${{priorHtml}}
    <div class="cell verified"><span class="lab">Verified — click a column (or the empty space left of it to add one) · ←/→ move · f/n/s/x set field · Alt+Enter/Alt+Bksp ins/del col · <button type="button" onclick="splitLongCols()" class="splitbtn">Split &gt;7 (Alt+s)</button> · <button type="button" onclick="fillSons()" class="splitbtn">Fill sons</button> · Ctrl+Enter save</span>
      <div class="vpanel" id="vrow">${{verifiedRow}}</div></div>
    ${{sliceBlock}}
    ${{readerRowsHtml}}`;
  // Once the scan renders, size each column slot to the scan's per-column pixel width
  // (rendered crop width / number of slots) so every text row spans the scan's width and
  // shares its right edge. Panel padding (8px each side) subtracted.
  const img = document.getElementById("cropimg");
  const applySlot = () => {{
    // Height = the scan's TRUE aspect at the rendered width (no blank band). Scale up
    // narrow scans a bit for legibility, but never past the natural size or a screen cap.
    const w = img.getBoundingClientRect().width;
    if (img.naturalWidth > 0 && w > 0) {{
      const natH = w * img.naturalHeight / img.naturalWidth;
      img.style.height = Math.round(Math.min(natH, window.innerHeight * 0.5)) + "px";
    }}
    if (w > 0 && nSlots > 0) rows.style.setProperty("--slot", ((w - 16) / nSlots) + "px");
    drawStrips();
  }};

  // Strip outlines: fetch the piece boxes (trimmed-crop px) once, then position a thin box
  // per piece scaled to the image's CURRENT rendered size. Redrawn on load/resize/toggle.
  let stripMeta = null;
  const layer = document.getElementById("striplayer");
  function drawStrips() {{
    if (!layer) return;
    layer.innerHTML = "";
    const show = document.getElementById("stripToggle");
    if (!b.hasSlice || (show && !show.checked) || !stripMeta) return;
    const rect = img.getBoundingClientRect();
    const rw = rect.width, rh = img.offsetHeight;      // rendered px
    if (!rw || !rh || !stripMeta.w || !stripMeta.h) return;
    const sx = rw / stripMeta.w, sy = rh / stripMeta.h;
    for (const p of stripMeta.pieces) {{
      const [x0, y0, x1, y1] = p.box;
      const d = document.createElement("div");
      d.className = "strip " + p.kind;
      d.style.left   = (x0 * sx) + "px";
      d.style.top    = (y0 * sy) + "px";
      d.style.width  = ((x1 - x0) * sx) + "px";
      d.style.height = ((y1 - y0) * sy) + "px";
      if (p.kind !== "col") {{
        const t = document.createElement("span"); t.className = "stag"; t.textContent = p.kind;
        d.appendChild(t);
      }}
      layer.appendChild(d);
    }}
  }}
  if (b.hasSlice) {{
    fetch(`/slicebox/${{b.id}}`).then(r => r.json()).then(m => {{ stripMeta = m; drawStrips(); }})
      .catch(() => {{}});
    const tg = document.getElementById("stripToggle");
    if (tg) tg.addEventListener("change", drawStrips);
  }}
  if (img.complete && img.naturalWidth) applySlot(); else img.onload = applySlot;
  window.addEventListener("resize", drawStrips);

  // Paste of MULTI-LINE text (e.g. a whole reader's transcription copied in) must land as
  // SEPARATE columns, not a blob in one cell -- otherwise field-labeling hits the whole
  // blob. Split the paste on newlines and distribute across columns from the focused one,
  // then re-render the verified row so each line is its own editable cell immediately.
  const vrow = document.getElementById("vrow");
  // Click in the EMPTY space to the LEFT of the leftmost column -> add a new leftmost column
  // (the RTL-end) and focus it. The panel is row-reverse so columns pack to the right edge and
  // the empty area is on the left; a click that misses every .vcol and lands left of them adds.
  vrow.addEventListener("mousedown", (e) => {{
    if (e.target.closest(".vcol")) return;          // clicked a real column -> normal edit
    const cells = vrow.querySelectorAll(".vcol");
    if (cells.length) {{
      const leftmost = cells[cells.length - 1].getBoundingClientRect();  // last = leftmost (RTL)
      if (e.clientX >= leftmost.left) return;        // click not to the left of it -> ignore
    }}
    e.preventDefault();
    addLeftCol();
  }});
  vrow.addEventListener("paste", (e) => {{
    const raw = (e.clipboardData || window.clipboardData).getData("text");
    if (!raw.includes("\\n")) return;               // single-line paste: let it be normal
    e.preventDefault();
    const lines = raw.split(/\\r?\\n/);
    const cols = currentVerifiedCols();
    const start = Math.max(0, curCol());
    // overwrite from `start`, extending the column list if the paste is longer
    for (let k = 0; k < lines.length; k++) cols[start + k] = lines[k];
    setVerifiedCols(cols);
    focusCol(start + lines.length - 1);
  }});
}}

// The current Verified columns as an array (one entry per editable cell).
function currentVerifiedCols() {{
  return [...document.querySelectorAll("#vrow .vcol.edit")].map(el => el.textContent);
}}

// Replace the Verified row's cells with `cols` (re-rendered in place), preserving field
// highlights from the current block's detection/overrides. Used by multi-line paste.
function setVerifiedCols(cols) {{
  const b = BLOCKS[CUR];
  const ov = FIELDS[b.id] || {{}};
  const det = detectFieldsCols(cols);          // detect on the NEW column text (not stale vision)
  const fieldCls = (i) => {{
    if (i in ov) return ov[i] === "none" ? null : ov[i];
    if (i === det.father_idx) return "father";
    if (i === det.name_idx) return "name";
    if (det.son_idxs.includes(i)) return "son";
    return null;
  }};
  const vrow = document.getElementById("vrow");
  vrow.innerHTML = cols.map((t, i) => {{
    const fc = fieldCls(i);
    const cls = ["vcol", "edit"].concat(fc ? ["f-" + fc] : []);
    if (fc === "father") cls.push("horiz");
    return `<span class="${{cls.join(" ")}}" contenteditable="plaintext-only" data-i="${{i}}">${{escapeHtml(t)}}</span>`;
  }}).join("");
}}

// Auto-detect field type per column on ACTUAL column text (col0=father, col1=name, sons after
// a 生子 column). Same logic as renderCurrent's detectFieldsOn -- kept in one place so every
// path agrees. Returns {{father_idx, name_idx, son_idxs}}.
function detectFieldsCols(cols) {{
  const f = {{father_idx: cols.length ? 0 : null, name_idx: cols.length > 1 ? 1 : null,
             son_idxs: []}};
  const start = cols.findIndex(c => (c||"").includes("生子"));
  if (start >= 0) {{
    for (let i = start + 1; i < cols.length; i++) {{
      if (/生女/.test(cols[i]||"") || /^[配继殁歿葬享寿卒]/.test(cols[i]||"")) break;
      f.son_idxs.push(i);
    }}
  }}
  return f;
}}

// The EFFECTIVE field type of each current Verified column (override else auto-detect on the
// CURRENT column text). Used before a structural edit so labels move with the columns. Detects
// on the live columns -- NOT the stale whole-crop b.fields.vision indices, which don't line up
// with the prefill/Gemini-slice column positions and previously mis-shifted son labels.
function effectiveFields(nCols) {{
  const b = BLOCKS[CUR];
  const ov = FIELDS[b.id] || {{}};
  const cols = currentVerifiedCols();
  const det = detectFieldsCols(cols);
  const out = {{}};
  for (let i = 0; i < nCols; i++) {{
    let t = null;
    if (i in ov) t = ov[i] === "none" ? null : ov[i];
    else if (i === det.father_idx) t = "father";
    else if (i === det.name_idx) t = "name";
    else if (det.son_idxs.includes(i)) t = "son";
    if (t) out[i] = t;
  }}
  return out;
}}

// Persist the block's field overrides (as an index->type map, "none" allowed) and set
// them live so a re-render keeps them.
async function persistFields(map) {{
  const b = BLOCKS[CUR];
  FIELDS[b.id] = map;
  await fetch("/fields", {{ method:"POST", headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{ id: b.id, fields: map }}) }});
}}

// Insert a BLANK column before index `at` (shifts the rest later); or delete column `at`
// (shifts the rest back). Field labels move with their columns. Text edits are saved too.
async function spliceCol(at, mode) {{
  const cols = currentVerifiedCols();
  const eff = effectiveFields(cols.length);
  const shifted = {{}};
  if (mode === "insert") {{
    cols.splice(at, 0, "");
    for (const k in eff) {{ const i = +k; shifted[i >= at ? i + 1 : i] = eff[k]; }}
  }} else {{ // delete
    if (!cols.length) return;
    cols.splice(at, 1);
    for (const k in eff) {{ const i = +k; if (i === at) continue;
                            shifted[i > at ? i - 1 : i] = eff[k]; }}
  }}
  // Make the shifted map AUTHORITATIVE: any column without a label becomes explicit "none".
  // Otherwise verifiedFieldCls falls back to the (now stale) auto-detected vision.son_idxs
  // and a son label re-appears at its OLD index -- i.e. labels don't move with the columns.
  // Persist BEFORE re-rendering so setVerifiedCols reads the NEW map (it renders from FIELDS).
  await persistFields(fullFieldMap(shifted, cols.length));
  setVerifiedCols(cols);
  await saveText(false);                 // persist shifted text WITHOUT advancing the block
  focusCol(mode === "insert" ? at : Math.min(at, cols.length - 1));
}}

// Append a blank column at the LEFTMOST position (= end of the RTL slot order, highest index).
// The scan reads right-to-left, so the far-left is the last column -- clicking blank space
// there has no target, hence this button (and Alt+Shift+Enter). Focuses the new column.
async function addLeftCol() {{
  const n = currentVerifiedCols().length;
  await spliceCol(n, "insert");
  focusCol(n);
}}

// Build a COMPLETE field map over [0,nCols): labeled columns keep their type, every other
// column is pinned to "none" so no stale auto-detection can leak a label back in after a
// structural edit (splice/split). "none" is persisted and overrides detection.
function fullFieldMap(labels, nCols) {{
  const out = {{}};
  for (let i = 0; i < nCols; i++) out[i] = (labels[i] != null ? labels[i] : "none");
  return out;
}}

// Fill the son columns from William's PRIOR verified son names (bio-derived, RTL/eldest
// order, 1-to-1 with the person's actual sons). For each prior son we conservatively fuzzy-
// match a Verified column that sits AFTER the 生子 marker (son columns are always past it;
// this avoids grabbing a date/description column earlier in the bio) and overwrite it; an
// unmatched son is inserted right after 生子 (RTL order). Skipped/description lines between
// sons are left untouched.
function _sonScore(a, b) {{
  // conservative similarity: shared generation char (first char) OR high char overlap.
  if (!a || !b) return 0;
  const ca = [...a], cb = [...b];
  const genMatch = ca[0] === cb[0] ? 1 : 0;           // 昭/宪/庆... generation char
  const setB = new Set(cb);
  const overlap = ca.filter(c => setB.has(c)).length / Math.max(ca.length, cb.length);
  const lenSim = 1 - Math.abs(ca.length - cb.length) / Math.max(ca.length, cb.length);
  // require the gen char to match OR overlap to be strong; else score 0 (won't hijack).
  if (!genMatch && overlap < 0.5) return 0;
  return genMatch * 2 + overlap + lenSim * 0.5;
}}
async function fillSons() {{
  const b = BLOCKS[CUR];
  const priorSons = (b.priorNames && b.priorNames.sons) ? b.priorNames.sons.slice() : [];
  if (!priorSons.length) {{ alert("No prior verified sons for this block."); return; }}
  let cols = currentVerifiedCols();
  const eff = effectiveFields(cols.length);        // current labels {{idx:type}}
  // 生子 marker column (son columns live after it). If absent, we can't safely place -> bail.
  const marker = cols.findIndex(c => (c || "").includes("生子"));
  if (marker < 0) {{ alert("No 生子 column found; add it first, then Fill sons."); return; }}
  // candidate columns = those AFTER the marker, not already father/name.
  const assigned = {{}};    // colIdx -> sonName
  const usedCols = new Set();
  for (const son of priorSons) {{                    // RTL/eldest order preserved
    let best = -1, bestScore = 0;
    for (let i = marker + 1; i < cols.length; i++) {{
      if (usedCols.has(i)) continue;
      if (eff[i] === "father" || eff[i] === "name") continue;
      const s = _sonScore(son, cols[i]);
      if (s > bestScore) {{ bestScore = s; best = i; }}
    }}
    if (best >= 0 && bestScore > 0) {{ assigned[best] = son; usedCols.add(best); }}
    else assigned["INSERT_" + son] = son;           // no match -> insert later
  }}
  // Apply overwrites to matched columns.
  const newLabels = {{}};
  for (const k in eff) newLabels[+k] = eff[k];
  // Clear any inherited/auto-detected `son` label from post-marker columns we did NOT match,
  // so after Fill the son-labeled columns are EXACTLY the prior sons -- a garbage/description
  // column left over (e.g. unmatched OCR junk) won't stay tagged son.
  for (let i = marker + 1; i < cols.length; i++) {{
    if (newLabels[i] === "son" && !usedCols.has(i)) delete newLabels[i];
  }}
  for (const [idx, son] of Object.entries(assigned)) {{
    if (idx.startsWith("INSERT_")) continue;
    cols[+idx] = son; newLabels[+idx] = "son";
  }}
  // Insert unmatched sons right after the marker (RTL order: eldest first == closest to marker).
  let insAt = marker + 1;
  for (const son of priorSons) {{
    if (assigned["INSERT_" + son] === undefined) continue;
    cols.splice(insAt, 0, son);
    // shift labels >= insAt up by one, then tag the new col
    const shifted = {{}};
    for (const k in newLabels) {{ const i = +k; shifted[i >= insAt ? i + 1 : i] = newLabels[k]; }}
    shifted[insAt] = "son";
    for (const k in newLabels) delete newLabels[k];
    Object.assign(newLabels, shifted);
    insAt++;
  }}
  await persistFields(fullFieldMap(newLabels, cols.length));
  setVerifiedCols(cols);
  await saveText(false);
}}

// Split EVERY Verified column longer than MAXCH (=7, the printed page's column height)
// into consecutive MAXCH-char chunks; everything after shifts. A field label stays with
// the FIRST chunk of the column it was on (the name/son sits at the column start), and
// all labels are remapped to their new column indices.
const MAXCH = 7;
async function splitLongCols() {{
  const cols = currentVerifiedCols();
  const eff = effectiveFields(cols.length);
  const out = [];
  const newFields = {{}};
  cols.forEach((c, i) => {{
    const chars = [...c];                         // codepoint-aware (rare glyphs = 1)
    const firstNew = out.length;                  // where this column's first chunk lands
    if (chars.length <= MAXCH) {{
      out.push(c);
    }} else {{
      for (let s = 0; s < chars.length; s += MAXCH) out.push(chars.slice(s, s + MAXCH).join(""));
    }}
    if (eff[i]) newFields[firstNew] = eff[i];     // label follows to the first chunk
  }});
  if (out.length === cols.length) return;         // nothing over-long
  await persistFields(fullFieldMap(newFields, out.length));  // authoritative: no stale leak
  setVerifiedCols(out);                           // render AFTER persist so it reads new map
  await saveText(false);                          // persist without advancing
}}

// Read the Verified row back as \\n-joined columns (slot order == Claude slot order).
function verifiedText() {{
  return [...document.querySelectorAll("#vrow .vcol.edit")]
    .map(el => el.textContent).join("\\n");
}}

function go(delta) {{
  CUR = (CUR + delta + BLOCKS.length) % BLOCKS.length;
  renderCurrent();
}}
function goReview(delta) {{
  for (let step = 1; step <= BLOCKS.length; step++) {{
    const i = (CUR + delta * step + BLOCKS.length * step) % BLOCKS.length;
    if (needsReview(BLOCKS[i])) {{ CUR = i; renderCurrent(); return; }}
  }}
}}
// Focus a Verified column by index (columns render right-to-left; index 0 = rightmost).
function focusCol(i) {{
  const cols = [...document.querySelectorAll("#vrow .vcol.edit")];
  if (!cols.length) return;
  i = Math.max(0, Math.min(cols.length - 1, i));
  cols[i].focus();
  const r = document.createRange(); r.selectNodeContents(cols[i]); r.collapse(false);
  const s = getSelection(); s.removeAllRanges(); s.addRange(r);
}}
function curCol() {{
  const el = document.activeElement;
  return el && el.classList && el.classList.contains("edit") ? +el.dataset.i : -1;
}}

async function saveText(advance) {{
  const b = BLOCKS[CUR];
  const text = verifiedText();
  const r = await fetch("/save", {{ method:"POST", headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{ id: b.id, text }}) }});
  if (r.ok) {{
    VERIFIED[b.id] = text;
    if (advance) {{ if (document.activeElement) document.activeElement.blur(); go(1); }}
  }}
}}
async function save() {{ await saveText(true); }}   // Ctrl+Enter: save + next block

document.addEventListener("keydown", (e) => {{
  if (e.ctrlKey && e.key === "Enter") {{ e.preventDefault(); save(); return; }}
  // Alt+s: split all Verified columns longer than 7 chars (whole-block; works regardless
  // of focus). Use e.code (physical key) because on macOS Alt+letter emits a composed
  // char (Option+s = ß), so e.key would not be "s".
  if (e.altKey && e.code === "KeyS") {{ e.preventDefault(); splitLongCols(); return; }}
  // Alt+Enter / Alt+Backspace: insert a blank column before / delete the focused column
  // (shifts the rest + moves field labels). Only while a Verified column is focused.
  if (e.altKey && curCol() >= 0 && (e.key === "Enter" || e.key === "Backspace")) {{
    e.preventDefault();
    spliceCol(curCol(), e.key === "Enter" ? "insert" : "delete");
    return;
  }}
  // Shift+arrows: navigate BLOCKS (prev/next, and up/down = prev/next to-review)
  if (e.shiftKey) {{
    if (e.key === "ArrowRight") {{ e.preventDefault(); go(1); return; }}
    if (e.key === "ArrowLeft")  {{ e.preventDefault(); go(-1); return; }}
    if (e.key === "ArrowDown")  {{ e.preventDefault(); goReview(1); return; }}
    if (e.key === "ArrowUp")    {{ e.preventDefault(); goReview(-1); return; }}
    return;
  }}
  const editing = curCol() >= 0;
  // Plain ←/→ move between Verified columns (columns are single glyphs stacked vertically,
  // so there's no horizontal text cursor to conflict with). RTL: → goes to the next column
  // to the LEFT (higher index), ← to the right (lower index).
  // Columns render RIGHT-TO-LEFT (index 0 = rightmost), so visual-left = higher index.
  if (editing && e.key === "ArrowLeft")  {{ e.preventDefault(); focusCol(curCol() + 1); return; }}
  if (editing && e.key === "ArrowRight") {{ e.preventDefault(); focusCol(curCol() - 1); return; }}
  // f/n/s/x: (re)assign the focused Verified column's FIELD type (father/name/son/none).
  // Bare single keys (reviewer's choice). Skip while an IME is composing so rare-glyph
  // input isn't disrupted.
  if (editing && !e.isComposing && !e.ctrlKey && !e.metaKey && !e.altKey
      && "fnsx".includes(e.key)) {{
    e.preventDefault();
    // x = "none": an explicit, PERSISTED override that this column is NOT a field, so it
    // overrides auto-detection on reload (e.g. clearing a wrongly auto-tagged son).
    const map = {{f:"father", n:"name", s:"son", x:"none"}};
    setField(curCol(), map[e.key]);
    return;
  }}
  if (!editing && (e.key === "e" || e.key === "Enter")) {{ e.preventDefault(); focusCol(0); }}
  if (e.key === "Escape" && document.activeElement) document.activeElement.blur();
}});

// Override the field type of one Verified slot, persist it, and restyle in place.
async function setField(i, type) {{
  if (i < 0) return;
  const b = BLOCKS[CUR];
  FIELDS[b.id] = FIELDS[b.id] || {{}};
  FIELDS[b.id][i] = type;                       // "none" = explicitly not a field (persisted)
  const el = document.querySelector(`#vrow .vcol.edit[data-i="${{i}}"]`);
  if (el) {{ el.classList.remove("f-father","f-name","f-son");
            if (type) el.classList.add("f-" + type); }}
  await fetch("/fields", {{ method:"POST", headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{ id: b.id, fields: FIELDS[b.id] }}) }});
}}

async function boot() {{
  const [blocks, verified, fields] = await Promise.all([
    fetch("/data").then(r => r.json()),
    fetch("/verified").then(r => r.json()),
    fetch("/fields").then(r => r.json()),
  ]);
  BLOCKS = blocks; VERIFIED = verified; FIELDS = fields || {{}};
  // Restore the last-viewed block for this book (by id, robust to reordering).
  try {{
    const last = localStorage.getItem("qa_cur_" + BOOK);
    if (last) {{ const i = BLOCKS.findIndex(b => b.id === last); if (i >= 0) CUR = i; }}
  }} catch (e) {{}}
  renderCurrent();
}}
boot();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    book = "book3"
    books_dir = "books"
    data_dir = "data"
    blocks: list[dict] = []
    graph_idx: dict = {}   # (stem,gen)->nodes index for the graph reference
    slice_ids: set = set() # block ids that have geometric-slice records (overlay available)
    slice_reads: dict = {} # {id: {reader: [texts]}} NEW per-strip slice readings

    def log_message(self, format, *args):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = PAGE.format(book=self.book, book_json=json.dumps(self.book))
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path == "/data":
            verified = load_verified(self.book, self.data_dir)
            # Load slice data FRESH per request (not cached at startup) so blocks sliced/updated
            # while the server runs -- e.g. a recovery run -- show up without a restart.
            cur_slice_ids = slice_ids(self.book, self.books_dir)
            cur_slice_reads = load_slice_reads(self.book, self.books_dir)
            prior = load_prior_names(self.book, self.data_dir)   # William's name-only QA (snapshot)
            payload = []
            for b in self.blocks:
                ab = dict(b, **analyze(b))          # ab now carries per-reader `fields`
                ab["graph"] = graph_ref(self.graph_idx, ab, verified.get(b["id"]))
                ab["hasSlice"] = b["id"] in cur_slice_ids  # strip-overlay available?
                sr = cur_slice_reads.get(b["id"], {})
                ab["sliceReads"] = sr                # NEW per-strip readings
                ab["priorNames"] = prior.get(b["id"])   # {father,name,sons} or None
                # Verified prefill = the Gemini SLICE reading (best raw OCR). Per-column
                # disagreement LEVEL, so the noisy/weak Paddle reader is de-emphasized:
                #   "gc"     = Gemini vs Claude disagree (HIGH priority -- the pair to trust)
                #   "paddle" = Gemini+Claude AGREE but Paddle differs (LOW priority, muted)
                #   ""       = all present readers agree
                def _flat(v):
                    if v in (None, "?ERR", "?EMPTY"):
                        return ""
                    return norm_variants(re.sub(r"\s+", "", v))   # 歿->殁 (universal variant)
                sg = [_flat(t) for t in sr.get("sgemini", [])]
                sv = [_flat(t) for t in sr.get("svision", [])]
                sp = [_flat(t) for t in sr.get("spaddle", [])]
                # Drop a spurious leading 一 (residual top rule-line OCR'd by one reader) where
                # it's the ONLY difference between Gemini and Claude, so the column agrees.
                n = len(sg)
                for i in range(n):
                    if i < len(sv):
                        sg[i], sv[i] = strip_leading_line(sg[i], sv[i])
                slice_diff = []
                for i in range(n):
                    g = sg[i] if i < len(sg) else ""
                    v = sv[i] if i < len(sv) else ""
                    p = sp[i] if i < len(sp) else ""
                    gc = g and v and g != v            # gemini vs claude both present & differ
                    if gc:
                        # If the ONLY differences are known misread pairs (夭/天, 究/宪) we auto-
                        # resolve them -> flag as "misread" (softer color, likely fine) instead of
                        # "gc" (a genuine unresolved Gemini/Claude conflict needing attention).
                        resolved = (resolve_misreads(g, v) == resolve_misreads(v, g))
                        slice_diff.append("misread" if resolved else "gc")
                    elif p and ((g and p != g) or (v and p != v)):
                        slice_diff.append("paddle")    # only paddle is the odd one out
                    else:
                        slice_diff.append("")
                ab["sliceDiff"] = slice_diff
                if any(sg):
                    # prefill = Gemini slice, with per-char disagreement resolution vs Claude:
                    # where they differ at a position on a known misread pair, pick the preferred
                    # char (夭 over 天, 宪 over 究). Column still flags magenta (real disagreement).
                    cols = [resolve_misreads(sg[i], sv[i] if i < len(sv) else "")
                            for i in range(n)]
                    # Father (col 0) + name (col 1) + SONS from William's PRIOR verified names
                    # (trustworthy) rather than Gemini; body columns stay Gemini slice.
                    # prior_filled = column indices whose text came from prior verification, so
                    # the UI can flag them visually.
                    prior_filled = []
                    pn = prior.get(b["id"])
                    if pn:
                        if pn.get("father") and len(cols) > 0 and cols[0] != pn["father"]:
                            cols[0] = pn["father"]; prior_filled.append(0)
                        elif pn.get("father") and len(cols) > 0:
                            pass
                        if pn.get("name") and len(cols) > 1 and cols[1] != pn["name"]:
                            cols[1] = pn["name"]; prior_filled.append(1)
                        # Sons: fuzzy-match each prior son to a son column AFTER 生子 and override
                        # it only when it DIFFERS from OCR. No-match sons are left as Gemini text.
                        psons = pn.get("sons") or []
                        if psons:
                            marker = next((k for k, c in enumerate(cols) if "生子" in c), -1)
                            if marker >= 0:
                                used = set()
                                for son in psons:
                                    best, best_s = -1, 0.0
                                    for k in range(marker + 1, len(cols)):
                                        if k in used:
                                            continue
                                        s = _son_score(son, cols[k])
                                        if s > best_s:
                                            best_s, best = s, k
                                    if best >= 0 and best_s > 0:
                                        used.add(best)
                                        if cols[best] != son:
                                            cols[best] = son
                                            prior_filled.append(best)
                    ab["priorFilled"] = prior_filled
                    ab["prefill"] = "\n".join(cols)
                payload.append(ab)
            return self._send(200, json.dumps(payload, ensure_ascii=False))
        if self.path == "/verified":
            return self._send(200, json.dumps(load_verified(self.book, self.data_dir),
                                              ensure_ascii=False))
        if self.path == "/fields":
            return self._send(200, json.dumps(load_fields(self.book, self.data_dir),
                                              ensure_ascii=False))
        if self.path.startswith("/img/"):
            bid = self.path[len("/img/"):]
            fp = os.path.join(self.books_dir, self.book, SEG_DIR, f"{bid}.png")
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    return self._send(200, fh.read(), "image/png")
            return self._send(404, b"", "image/png")
        if self.path.startswith("/slice/"):          # trimmed crop that the overlay aligns to
            bid = self.path[len("/slice/"):]
            png, _meta = slice_overlay(self.book, self.books_dir, bid)
            if png is not None:
                return self._send(200, png, "image/png")
            return self._send(404, b"", "image/png")
        if self.path.startswith("/slicebox/"):       # strip outline boxes in trimmed-crop px
            bid = self.path[len("/slicebox/"):]
            _png, meta = slice_overlay(self.book, self.books_dir, bid)
            if meta is not None:
                return self._send(200, json.dumps(meta, ensure_ascii=False))
            return self._send(404, json.dumps({"pieces": []}))
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/save":
            save_verified(self.book, self.data_dir, data["id"], data.get("text", ""))
            return self._send(200, json.dumps({"ok": True}))
        if self.path == "/fields":
            save_fields(self.book, self.data_dir, data["id"], data.get("fields", {}))
            return self._send(200, json.dumps({"ok": True}))
        return self._send(404, "{}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", default="book3")
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args(argv)

    Handler.book = args.book
    Handler.books_dir = args.books_dir
    Handler.data_dir = args.data_dir
    Handler.blocks = load_blocks(args.book, args.books_dir)
    Handler.graph_idx = build_graph_index(args.book, args.books_dir, args.data_dir)
    Handler.slice_ids = slice_ids(args.book, args.books_dir)
    Handler.slice_reads = load_slice_reads(args.book, args.books_dir)
    done = len(load_verified(args.book, args.data_dir))
    print(f"Loaded {len(Handler.blocks)} blocks for {args.book} ({done} already verified).")
    print(f"Open  http://localhost:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
