"""Stage 2 (classify) review: a color-coded page grid, click to fix a label.

Shows every extracted page of a book as a thumbnail in a grid, bordered by its
Stage-2 classification -- **green = tree graph**, **red = biography**. You scan
for a wrong-colored border (a tree page shown red, or a bio page shown green) and
**click the cell to flip** its label. Flips are saved to a human override layer,
``books/{book}/2_classify/page_types_review.json`` (``{"145": "graph"}``), keyed
by page index. Clicking a flipped cell back to its detector label clears the
override. :func:`src.s2_classify_pages.load_bio_pages` honors these overrides, so
a flip actually moves the page on/off the graph path at the next crop/merge.

Thumbnails are served lazily from the already-extracted ``1_pages/{i}.png`` (no
PDF re-render), so a 300-page book loads fast. One server, a tab per book; a
tab's grid loads only when opened.

Port 8762 (clear of 8000 and the other QA servers on 8761/8767).

Run::

    python -m scripts.qa.s2_classify          # then open http://localhost:8762/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from src.imaging import get_image

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)

BOOKS_DIR = "books"
# Only books that were classified (Books 1 & 2 are all-tree, no sidecar).
CLASSIFY_QA_BOOKS = ["book3", "book4"]
# Long side (px) of a grid thumbnail.
THUMB_LONG = 200


# ---- classification state (detector output + human overrides) --------------

def _sidecar_path(book: str) -> str:
    return os.path.join(BOOKS_DIR, book, "2_classify", "page_types.json")


def _review_path(book: str) -> str:
    return os.path.join(BOOKS_DIR, book, "2_classify", "page_types_review.json")


def load_sidecar(book: str) -> dict:
    path = _sidecar_path(book)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_review(book: str) -> dict:
    path = _review_path(book)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_review(book: str, page: int, label: str | None) -> None:
    """Set (or clear, when ``label`` is None) the override for one page."""
    path = _review_path(book)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = load_review(book)
    key = str(page)
    if label is None:
        data.pop(key, None)
    else:
        data[key] = label
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def page_rows(book: str) -> list[dict]:
    """One row per page: detector label, any override, effective label, rule count."""
    sc = load_sidecar(book)
    rev = load_review(book)
    pages = sc.get("pages", {})
    rows = []
    for i in range(sc.get("num_pages", len(pages))):
        entry = pages.get(str(i), {})
        detected = entry.get("type", "graph")
        override = rev.get(str(i))
        rows.append({
            "page": i,
            "detected": detected,
            "override": override,
            "type": override or detected,
            "n_rules": len(entry.get("rule_yfrac", [])),
        })
    return rows


# ---- thumbnail serving -----------------------------------------------------

def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_thumb(book: str, page: int) -> Image.Image:
    a = get_image(os.path.join(BOOKS_DIR, book, "1_pages", f"{page}.png"))
    img = Image.fromarray((a * 255).astype(np.uint8))
    s = THUMB_LONG / max(img.width, img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))


PAGE_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Classify QA</title>
<style>
  body{margin:0;background:#1e1e1e;color:#eee;font-family:system-ui,sans-serif}
  #tabs{position:sticky;top:0;background:#111;padding:8px 12px;border-bottom:1px solid #333;z-index:2}
  #tabs button{background:#2a2a2a;color:#eee;border:1px solid #444;padding:6px 14px;margin-right:6px;cursor:pointer;border-radius:4px}
  #tabs button.active{background:#3a5a8a;border-color:#5a7aba}
  #legend{display:inline-block;margin-left:16px;font-size:13px;color:#aaa}
  .sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin:0 4px 0 12px}
  #stats{margin-left:16px;font-size:13px;color:#9ab}
  #grid{display:flex;flex-wrap:wrap;gap:6px;padding:12px}
  .cell{position:relative;width:110px;cursor:pointer;border:3px solid #444;border-radius:3px;background:#000;overflow:hidden}
  .cell.graph{border-color:#3fb950}
  .cell.bio{border-color:#f05050}
  .cell img{display:block;width:100%;height:auto}
  .cell .lbl{position:absolute;left:0;right:0;bottom:0;font-size:10px;padding:1px 3px;background:rgba(0,0,0,.65);color:#ddd;line-height:1.3}
  .cell.flipped::after{content:"\\270E";position:absolute;top:2px;right:4px;color:#ffcf4d;font-size:14px;text-shadow:0 0 3px #000}
  #toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#2f7d32;color:#fff;padding:8px 16px;border-radius:5px;opacity:0;transition:opacity .2s;pointer-events:none}
  #toast.show{opacity:1}
</style></head><body>
<div id="tabs"></div>
<div id="grid"></div>
<div id="toast"></div>
<script>
const BOOKS=%%TABS%%;
let book=BOOKS[0], rows=[];
const tabs=document.getElementById('tabs'), grid=document.getElementById('grid'), toast=document.getElementById('toast');

function renderTabs(){
  tabs.innerHTML='';
  BOOKS.forEach(b=>{const btn=document.createElement('button');btn.textContent=b;
    if(b===book)btn.className='active';btn.onclick=()=>{book=b;load();};tabs.appendChild(btn);});
  const leg=document.createElement('span');leg.id='legend';
  leg.innerHTML='<span class="sw" style="background:#3fb950"></span>tree'
    +'<span class="sw" style="background:#f05050"></span>bio &nbsp;— click a cell to flip'
    +' <span id="stats"></span>';
  tabs.appendChild(leg);
}
function updateStats(){
  const g=rows.filter(r=>r.type==='graph').length, b=rows.length-g;
  const f=rows.filter(r=>r.override).length;
  document.getElementById('stats').textContent=`${rows.length} pages: ${g} tree / ${b} bio`+(f?` · ${f} overridden`:'');
}
function cellFor(r){
  const d=document.createElement('div');
  d.className='cell '+r.type+(r.override?' flipped':'');
  d.title=`page ${r.page} — detected ${r.detected}${r.override?` → ${r.override}`:''} (rules ${r.n_rules})`;
  const img=document.createElement('img');img.loading='lazy';
  img.src=`/qa/thumb?book=${book}&page=${r.page}`;
  const lbl=document.createElement('div');lbl.className='lbl';
  lbl.textContent=`${r.page} ${r.type}`;
  d.appendChild(img);d.appendChild(lbl);
  d.onclick=()=>flip(r,d);
  return d;
}
async function flip(r,cell){
  const flipped=r.type==='graph'?'bio':'graph';
  // If flipping back to the detector's own label, clear the override.
  const label=(flipped===r.detected)?null:flipped;
  const res=await fetch('/qa/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({book,page:r.page,label})});
  if(!res.ok){showToast('save failed',true);return;}
  r.override=label; r.type=flipped;
  cell.className='cell '+r.type+(r.override?' flipped':'');
  cell.querySelector('.lbl').textContent=`${r.page} ${r.type}`;
  updateStats();
  showToast(`page ${r.page} → ${flipped}`+(label?'':' (cleared override)'));
}
let toastT;
function showToast(msg,err){toast.textContent=msg;toast.style.background=err?'#a33':'#2f7d32';
  toast.classList.add('show');clearTimeout(toastT);toastT=setTimeout(()=>toast.classList.remove('show'),1400);}
async function load(){
  renderTabs();grid.innerHTML='<p style="padding:16px;color:#888">loading…</p>';
  const res=await fetch(`/qa/index?book=${book}`);rows=await res.json();
  grid.innerHTML='';rows.forEach(r=>grid.appendChild(cellFor(r)));updateStats();
}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _book(self, q):
        """Validate ``book`` against the allowlist (guards path traversal)."""
        book = (q.get("book") or [""])[0]
        if book not in CLASSIFY_QA_BOOKS:
            self._send(400, json.dumps({"error": "invalid book"}))
            return None
        return book

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            html = PAGE_HTML.replace("%%TABS%%", json.dumps(CLASSIFY_QA_BOOKS))
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/qa/index":
            book = self._book(q)
            if book is None:
                return
            return self._send(200, json.dumps(page_rows(book)))
        if u.path == "/qa/thumb":
            book = self._book(q)
            if book is None:
                return
            try:
                page = int(q["page"][0])
            except (KeyError, ValueError):
                return self._send(400, json.dumps({"error": "bad page"}))
            path = os.path.join(BOOKS_DIR, book, "1_pages", f"{page}.png")
            if not os.path.exists(path):
                return self._send(404, b"no page", "text/plain")
            return self._send(200, _png_bytes(render_thumb(book, page)), "image/png")
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/qa/save":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        book = payload.get("book")
        if book not in CLASSIFY_QA_BOOKS:
            return self._send(400, json.dumps({"error": "invalid book"}))
        label = payload.get("label")
        if label not in (None, "graph", "bio"):
            return self._send(400, json.dumps({"error": "bad label"}))
        try:
            page = int(payload["page"])
        except (KeyError, ValueError, TypeError):
            return self._send(400, json.dumps({"error": "bad page"}))
        save_review(book, page, label)
        return self._send(200, json.dumps({"ok": True}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8762)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Classify QA at http://localhost:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
