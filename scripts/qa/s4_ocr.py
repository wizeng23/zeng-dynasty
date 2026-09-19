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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEG_DIR = os.path.join("bio", "3_segment")
OCR_DIR = os.path.join("bio", "4_ocr")


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
            paddle = (r.get("raw", {}).get("paddle", {}).get("columns")) or []
            vtext = (r.get("raw", {}).get("vision", {}).get("text")) or ""
            vision = [ln for ln in vtext.split("\n") if ln.strip()]
            blocks.append({
                "id": r["id"],
                "section": r.get("id", "").rsplit("_", 2)[0],
                "generation": r.get("generation"),
                "paddle": paddle,
                "vision": vision,
            })
    return blocks


import re

_COUNT = "一二三四五六七八九十两"
_SONS_START = re.compile(rf"生子[{_COUNT}]*名")


def analyze(paddle: list[str], vision: list[str]) -> dict:
    """Per-block review aids.

    - ``prefill``: the Verified box default = Claude's (vision) full read, joined by \\n.
    - ``diff``: per column-index, True where paddle[i] != vision[i] (aligned by index) so
      the UI can highlight conflicting columns to draw the eye.
    - ``name_idx``: the vision column index of the person's own name (line 2, after the
      子之X header) so it can be emphasized.
    - ``son_idxs``: vision column indices of the sons -- the columns after a ``生子…名``
      marker up to a ``生女`` / new clause -- the stitch-critical fields to emphasize.
    """
    # Content-aware diff: a vision column is "matched" if the SAME text appears anywhere
    # in the paddle columns (and vice versa). This tolerates the two readers splitting
    # columns slightly differently, so only genuinely different text is flagged --
    # `diff[i]` is per VISION column (the panel the reviewer edits from).
    pset = set(paddle)
    diff = [v not in pset for v in vision]

    # name = the 2nd vision column (line 1 is the 子之X header)
    name_idx = 1 if len(vision) > 1 else None

    # sons = vision columns after 生子…名, until 生女 / a new clause word
    son_idxs = []
    start = next((i for i, c in enumerate(vision) if _SONS_START.search(c)), None)
    if start is not None:
        for i in range(start + 1, len(vision)):
            c = vision[i]
            if re.search("生女", c) or re.match("[配继殁歿葬享寿卒]", c):
                break
            son_idxs.append(i)

    return {
        "prefill": "\n".join(vision),
        "diff": diff,
        "name_idx": name_idx,
        "son_idxs": son_idxs,
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


# --- HTML -------------------------------------------------------------------------

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Bio OCR QA — {book}</title>
<style>
  :root {{ --bg:#faf9f7; --fg:#1a1a1a; --muted:#8a8a8a; --line:#ddd; --ok:#0a7f3f;
           --paddle:#1558b0; --vision:#8a5a00; --card:#fff; }}
  * {{ box-sizing:border-box; }}
  html,body {{ height:100%; }}
  body {{ margin:0; font:14px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg);
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
              align-self:flex-end; }}
  /* vertical, right-to-left: columns run right->left, chars top->bottom (mirrors the scan). */
  .vpanel {{ display:flex; flex-direction:row-reverse; justify-content:flex-start;
             align-items:flex-start; gap:3px; padding:8px; border:1px solid var(--line);
             border-radius:6px; background:#fff; overflow-x:auto; }}
  /* OCR glyphs sized to roughly match the scanned characters for column-by-column compare. */
  /* Fixed per-column slot width so Claude & Paddle rows line up column-for-column
     (a missing column shows as an empty .gap slot of the same width). */
  .vcol {{ writing-mode:vertical-rl; text-orientation:upright; white-space:pre;
           font-size:30px; line-height:1.45; padding:1px 0; border-radius:3px;
           flex:0 0 40px; width:40px; text-align:center; }}
  .vcol.gap {{ background:repeating-linear-gradient(45deg,#f4f4f4,#f4f4f4 4px,#fafafa 4px,#fafafa 8px); }}
  .vcol.diff {{ background:#ffe9d6; }}
  .vcol.son {{ box-shadow: inset 0 0 0 2px #c0392b; }}
  .vcol.name {{ box-shadow: inset 0 0 0 2px var(--ok); }}
  textarea.vtext {{ writing-mode:vertical-rl; text-orientation:upright; white-space:pre;
            font-family:inherit; font-size:30px; line-height:1.45; resize:vertical; width:100%;
            min-height:34vh; color:var(--ok); padding:8px; border:1px solid var(--line);
            border-radius:6px; background:#fff; }}
  textarea.vtext:focus {{ outline:2px solid var(--ok); border-color:var(--ok); }}
</style></head><body>
<header>
  <span class="bid" id="bid">…</span>
  <span class="prog" id="pos"></span>
  <span class="status" id="status"></span>
  <span class="prog" id="prog"></span>
  <span class="legend"><span style="background:#ffe9d6">orange</span>=differ · <b style="color:#c0392b">red</b>=son · <b style="color:var(--ok)">green</b>=name</span>
  <span class="keys"><kbd>Shift</kbd>+<kbd>←/→</kbd> prev/next · <kbd>Shift</kbd>+<kbd>↑/↓</kbd> prev/next to-review · <kbd>e</kbd> edit · <kbd>Esc</kbd> stop · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save+next</span>
</header>
<div id="stage"><div class="rows" id="rows"></div></div>
<script>
let BLOCKS = [];
let VERIFIED = {{}};
let CUR = 0;
const FONT_PX = 30;   // OCR glyph size; the crop is scaled so its glyphs match this

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
    if (c.text == null) return `<span class="vcol gap"></span>`;
    return `<span class="${{["vcol", ...c.cls].join(" ")}}">${{escapeHtml(c.text)}}</span>`;
  }});
  return `<div class="vpanel">${{spans.join("") || "—"}}</div>`;
}}

function renderCurrent() {{
  const b = BLOCKS[CUR];
  const done = isDone(b), disagree = isDisagree(b);
  const verifiedText = done ? VERIFIED[b.id] : b.prefill;
  document.getElementById("bid").textContent = b.id;
  document.getElementById("pos").textContent = `${{CUR + 1}} / ${{BLOCKS.length}} · gen ${{b.generation}}`;
  const st = document.getElementById("status");
  st.textContent = done ? "✓ verified" : (disagree ? "⚠ readers differ" : "· unreviewed");
  st.className = "status " + (done ? "done" : "todo");
  document.getElementById("prog").textContent = `(${{Object.keys(VERIFIED).length}} verified)`;
  // Scale the crop so a scanned glyph ~= FONT_PX: displayed height = FONT_PX * (chars in
  // the tallest OCR column) * line-height, since the scan's columns hold ~the same glyphs.
  const maxChars = Math.max(1, ...b.paddle.map(c => c.length), ...b.vision.map(c => c.length));
  const cropH = Math.round(FONT_PX * 1.45 * maxChars) + 18;  // +padding

  // Align Paddle onto Claude's slots so columns line up row-to-row (gaps where a reader
  // is missing a column). Build the two slotted rows with diff/son/name flags.
  const {{ slots, paddleAt }} = alignToClaude(b.vision, b.paddle);
  const sons = new Set(b.son_idxs || []), nameI = b.name_idx;
  const claudeCells = slots.map((v, i) => {{
    if (v == null) return {{ text: null, cls: [] }};      // slot Paddle-only -> gap in Claude
    const cls = [];
    if (paddleAt[i] !== v) cls.push("diff");              // Paddle here differs / is missing
    if (sons.has(i)) cls.push("son");
    if (i === nameI) cls.push("name");
    return {{ text: v, cls }};
  }});
  const paddleCells = slots.map((v, i) => {{
    const p = paddleAt[i];
    if (p == null) return {{ text: null, cls: [] }};       // gap: Paddle missing this column
    return {{ text: p, cls: p !== v ? ["diff"] : [] }};
  }});

  document.getElementById("rows").innerHTML = `
    <div class="cell crop"><span class="lab">Original (scan)</span>
      <img class="crop" style="height:${{cropH}}px" src="/img/${{b.id}}"></div>
    <div class="cell verified"><span class="lab">Verified — defaults to Claude (e=edit · Ctrl+Enter=save)</span>
      <textarea class="vtext" id="ta">${{escapeHtml(verifiedText)}}</textarea></div>
    <div class="cell vision"><span class="lab">Claude</span>${{slotRow(claudeCells)}}</div>
    <div class="cell paddle"><span class="lab">Paddle</span>${{slotRow(paddleCells)}}</div>`;
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
function focusEdit() {{ const ta = document.getElementById("ta"); if (ta) {{ ta.focus();
    ta.setSelectionRange(ta.value.length, ta.value.length); }} }}

async function save() {{
  const b = BLOCKS[CUR];
  const ta = document.getElementById("ta");
  const r = await fetch("/save", {{ method:"POST", headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{ id: b.id, text: ta.value }}) }});
  if (r.ok) {{ VERIFIED[b.id] = ta.value; ta.blur(); go(1); }}
}}

document.addEventListener("keydown", (e) => {{
  // Ctrl+Enter saves+advances from anywhere (incl. inside the textarea)
  if (e.ctrlKey && e.key === "Enter") {{ e.preventDefault(); save(); return; }}
  // Shift+arrows navigate even while typing (plain arrows stay as the text cursor)
  if (e.shiftKey) {{
    if (e.key === "ArrowRight") {{ e.preventDefault(); go(1); return; }}
    if (e.key === "ArrowLeft")  {{ e.preventDefault(); go(-1); return; }}
    if (e.key === "ArrowDown")  {{ e.preventDefault(); goReview(1); return; }}
    if (e.key === "ArrowUp")    {{ e.preventDefault(); goReview(-1); return; }}
  }}
  const editing = document.activeElement && document.activeElement.id === "ta";
  if (!editing && (e.key === "e" || e.key === "Enter")) {{ e.preventDefault(); focusEdit(); }}
  if (e.key === "Escape") {{ const ta = document.getElementById("ta"); if (ta) ta.blur(); }}
}});

async function boot() {{
  const [blocks, verified] = await Promise.all([
    fetch("/data").then(r => r.json()),
    fetch("/verified").then(r => r.json()),
  ]);
  BLOCKS = blocks; VERIFIED = verified;
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
            payload = [dict(b, **analyze(b["paddle"], b["vision"])) for b in self.blocks]
            return self._send(200, json.dumps(payload, ensure_ascii=False))
        if self.path == "/verified":
            return self._send(200, json.dumps(load_verified(self.book, self.data_dir),
                                              ensure_ascii=False))
        if self.path.startswith("/img/"):
            bid = self.path[len("/img/"):]
            fp = os.path.join(self.books_dir, self.book, SEG_DIR, f"{bid}.png")
            if os.path.exists(fp):
                with open(fp, "rb") as fh:
                    return self._send(200, fh.read(), "image/png")
            return self._send(404, b"", "image/png")
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, "{}")
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        save_verified(self.book, self.data_dir, data["id"], data.get("text", ""))
        return self._send(200, json.dumps({"ok": True}))


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
    done = len(load_verified(args.book, args.data_dir))
    print(f"Loaded {len(Handler.blocks)} blocks for {args.book} ({done} already verified).")
    print(f"Open  http://localhost:{args.port}/")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
