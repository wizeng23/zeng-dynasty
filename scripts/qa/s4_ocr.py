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


def agreement(paddle: list[str], vision: list[str]) -> list[str]:
    """Prepopulate the verified text: take a line where both readers agree, else blank."""
    n = max(len(paddle), len(vision))
    out = []
    for i in range(n):
        p = paddle[i] if i < len(paddle) else ""
        v = vision[i] if i < len(vision) else ""
        out.append(p if p and p == v else "")
    return out


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
  body {{ margin:0; font:14px/1.5 system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
  header {{ position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
            padding:10px 16px; display:flex; gap:16px; align-items:center; z-index:10; }}
  header .prog {{ color:var(--muted); }}
  header button {{ font:inherit; padding:4px 10px; border:1px solid var(--line);
                   border-radius:6px; background:var(--bg); cursor:pointer; }}
  #list {{ padding:16px; display:flex; flex-direction:column; gap:22px; }}
  .block {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
            padding:12px 14px; }}
  .block.done {{ border-color:var(--ok); }}
  .block.disagree .hdr .warn {{ color:#c0392b; }}
  .hdr {{ display:flex; gap:12px; align-items:baseline; margin-bottom:8px; }}
  .hdr .bid {{ font-weight:600; }}
  .hdr .warn {{ color:var(--muted); font-size:12px; }}
  .cols {{ display:flex; gap:14px; align-items:flex-start; overflow-x:auto; }}
  .cell {{ display:flex; flex-direction:column; gap:4px; }}
  .cell .lab {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                color:var(--muted); }}
  .cell.paddle .lab {{ color:var(--paddle); }}
  .cell.vision .lab {{ color:var(--vision); }}
  .cell.verified .lab {{ color:var(--ok); }}
  img.crop {{ max-height:220px; border:1px solid var(--line); border-radius:6px;
              background:#fff; }}
  /* vertical, right-to-left: columns run right->left, chars top->bottom, mirroring the scan */
  .vtext {{ writing-mode:vertical-rl; text-orientation:upright; white-space:pre;
            font-size:19px; line-height:1.35; min-height:160px; max-height:230px;
            padding:6px 8px; border:1px solid var(--line); border-radius:6px;
            background:#fff; overflow:auto; }}
  textarea.vtext {{ font-family:inherit; resize:both; min-width:120px; color:var(--ok); }}
  .saved {{ color:var(--ok); font-size:12px; }}
</style></head><body>
<header>
  <strong>Bio OCR QA · {book}</strong>
  <span class="prog" id="prog"></span>
  <button onclick="jumpNext('todo')">Next unreviewed →</button>
  <button onclick="jumpNext('disagree')">Next disagreement →</button>
  <span class="prog">Each panel reads top→bottom, right→left (like the book). Blue=Paddle · amber=Claude · green=your verified. Ctrl+Enter saves.</span>
</header>
<div id="list"></div>
<script>
const BOOK = {book_json};
let BLOCKS = [];
let VERIFIED = {{}};

function colsToText(cols) {{ return cols.join("\\n"); }}

function render() {{
  const list = document.getElementById("list");
  list.innerHTML = "";
  let done = 0;
  for (const b of BLOCKS) {{
    const isDone = VERIFIED[b.id] !== undefined;
    if (isDone) done++;
    const disagree = colsToText(b.paddle) !== colsToText(b.vision);
    const el = document.createElement("div");
    el.className = "block" + (isDone ? " done" : "") + (disagree ? " disagree" : "");
    el.id = "b_" + b.id;
    el.dataset.todo = isDone ? "0" : "1";
    el.dataset.disagree = disagree ? "1" : "0";
    const verifiedText = VERIFIED[b.id] !== undefined ? VERIFIED[b.id] : colsToText(b.agree);
    el.innerHTML = `
      <div class="hdr">
        <span class="bid">${{b.id}}</span>
        <span class="warn">gen ${{b.generation}} ${{disagree ? "· ⚠ readers differ" : "· readers agree"}}</span>
        <span class="saved" id="saved_${{b.id}}">${{isDone ? "✓ verified" : ""}}</span>
      </div>
      <div class="cols">
        <div class="cell"><span class="lab">crop</span>
          <img class="crop" loading="lazy" src="/img/${{b.id}}"></div>
        <div class="cell paddle"><span class="lab">Paddle</span>
          <div class="vtext">${{escapeHtml(colsToText(b.paddle)) || "—"}}</div></div>
        <div class="cell vision"><span class="lab">Claude</span>
          <div class="vtext">${{escapeHtml(colsToText(b.vision)) || "—"}}</div></div>
        <div class="cell verified"><span class="lab">Verified (edit · Ctrl+Enter)</span>
          <textarea class="vtext" id="ta_${{b.id}}"
            onkeydown="if(event.ctrlKey&&event.key==='Enter'){{save('${{b.id}}');event.preventDefault();}}"
          >${{escapeHtml(verifiedText)}}</textarea></div>
      </div>`;
    list.appendChild(el);
  }}
  document.getElementById("prog").textContent =
    `${{done}} / ${{BLOCKS.length}} verified`;
}}

function escapeHtml(s) {{ return (s||"").replace(/[&<>]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[c])); }}

async function save(id) {{
  const ta = document.getElementById("ta_" + id);
  const r = await fetch("/save", {{ method:"POST",
    headers:{{"Content-Type":"application/json"}},
    body: JSON.stringify({{ id, text: ta.value }}) }});
  if (r.ok) {{
    VERIFIED[id] = ta.value;
    const el = document.getElementById("b_" + id);
    el.classList.add("done"); el.dataset.todo = "0";
    document.getElementById("saved_" + id).textContent = "✓ verified";
    document.getElementById("prog").textContent =
      `${{Object.keys(VERIFIED).length}} / ${{BLOCKS.length}} verified`;
  }}
}}

function jumpNext(kind) {{
  const attr = kind === "disagree" ? "disagree" : "todo";
  const y = window.scrollY;
  for (const el of document.querySelectorAll(".block")) {{
    if (el.dataset[attr] === "1" && el.getBoundingClientRect().top > 10) {{
      el.scrollIntoView({{behavior:"smooth", block:"start"}}); return;
    }}
  }}
  // wrap to first
  const first = [...document.querySelectorAll(".block")].find(el => el.dataset[attr] === "1");
  if (first) first.scrollIntoView({{behavior:"smooth", block:"start"}});
}}

async function boot() {{
  const [blocks, verified] = await Promise.all([
    fetch("/data").then(r => r.json()),
    fetch("/verified").then(r => r.json()),
  ]);
  BLOCKS = blocks; VERIFIED = verified;
  render();
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
            payload = [dict(b, agree=agreement(b["paddle"], b["vision"])) for b in self.blocks]
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
