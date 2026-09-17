"""Editable parse QA: fix a subgraph's Stage-5 parse in the browser.

The slow loop -- William describes a broken region, the agent probes pixels and
hand-edits JSON -- is replaced by direct manipulation. Each graph is a canvas: the
pages/crops compare strip (read-only reference) on top, the editable parse below.
You move/resize/add/delete name boxes and connect children to parents; the tool
writes a self-contained per-subgraph ground-truth override to
``data/{book}_manual/{stem}.json`` that REPLACES Stage 5 for that graph.

The override is hand-authored ground truth, so it carries only what the tree IS --
per node a ``box`` [l,t,r,b], branch endpoints ``top``/``bot`` [row,col], ``father``
and eldest-first ``children`` -- not parser scaffolding (no green bridges / nicks /
empty flags). The QA renders a red box per node and a red edge from each child's
``top`` up to its father's ``bot``.

Seeded from the current parse (``src.s5_build_tree`` incl. the vertical riser-gap
fill), then corrected. Mirrors ``scripts/qa/s6_ocr.py``'s server pattern.

CLI:
    python -m scripts.qa.s5_edit --book book3 --port 8787
    # then open http://localhost:8787/
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

from src import s3_segment as seg
from src.imaging import get_image, save_image
import src.s5_build_tree as bt
from scripts.qa.s5_parse import stacked_compare, _fit_width

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

BOOKS_DIR = "books"
DATA_DIR = "data"
BOOK = "book3"  # set by main(); the book being edited


def _manual_dir(book: str) -> str:
    return os.path.join(DATA_DIR, f"{book}_manual")


def _manual_path(book: str, stem: str) -> str:
    return os.path.join(_manual_dir(book), f"{stem}.json")


def _graph_path(book: str, stem: str) -> str:
    return os.path.join(BOOKS_DIR, book, "4_graphs", f"{stem}.png")


def _graph_stems(book: str) -> list[str]:
    d = os.path.join(BOOKS_DIR, book, "4_graphs")
    stems = [os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".png")]
    return sorted(stems, key=lambda s: int(s.split("_")[0]))


def seed_override(book: str, stem: str) -> dict:
    """Parse one graph and return the complete node list (the editable seed).

    Uses the current ``src`` parser (with the riser-gap fill), so re-seeding picks
    up automatic improvements. Only the ground-truth fields are kept.
    """
    cfg = bt.config_for(book)
    a = get_image(_graph_path(book, stem))
    a2, _imag, _nicks = bt.bridge_orphans(a, cfg)
    nodes = bt.parse_graph(a2, cfg, graph_stem=stem)
    for i, n in enumerate(nodes):
        n.id = i
    id_of = {id(n): n.id for n in nodes}
    recs = []
    for n in nodes:
        if n.top is None or n.bot is None:
            continue
        left, top, right, bottom = bt.name_box_coords(n, a2)
        recs.append({
            "id": n.id,
            "box": [int(left), int(top), int(right), int(bottom)],
            "top": [int(n.top[0]), int(n.top[1])],
            "bot": [int(n.bot[0]), int(n.bot[1])],
            "children": [id_of[id(c)] for c in n.children if id(c) in id_of],
            "name": "",
        })
    father = {}
    for r in recs:
        for c in r["children"]:
            father[c] = r["id"]
    for r in recs:
        r["father"] = father.get(r["id"], -1)
    return {"stem": stem, "nodes": recs}


def seed_from_sidecar(book: str, stem: str) -> dict | None:
    """Seed instantly from the on-disk Stage-5 parse sidecar (no re-parse).

    ``{stem}.parse.json`` already holds each node's ``box``/``top``/``bot`` and its
    ``children_top`` points. We turn ``children_top`` back into child ids by matching
    each point to the node whose ``top`` equals it (they are the same recorded
    coordinates). Returns ``None`` if the sidecar is missing, so the caller falls
    back to a live parse. This is what makes opening a big graph instant -- the
    sidecar may be slightly stale vs the newest code, but that is exactly what the
    editor is for (you correct it), and re-parsing a 20k-px graph is minutes.
    """
    sc = os.path.join(BOOKS_DIR, book, "4_graphs", f"{stem}.parse.json")
    if not os.path.exists(sc):
        return None
    src = json.load(open(sc))["nodes"]
    by_top = {tuple(n["top"]): n["id"] for n in src}
    recs = []
    for n in src:
        kids = [by_top[tuple(ct)] for ct in n.get("children_top", []) if tuple(ct) in by_top]
        recs.append({
            "id": n["id"], "box": n["box"], "top": n["top"], "bot": n["bot"],
            "children": kids, "name": "",
        })
    father = {}
    for r in recs:
        for c in r["children"]:
            father[c] = r["id"]
    for r in recs:
        r["father"] = father.get(r["id"], -1)
    return {"stem": stem, "nodes": recs}


def load_override(book: str, stem: str) -> dict:
    """Return the saved override, else seed instantly from the parse sidecar, else
    (no sidecar) fall back to a live parse. The seed is saved so it opens fast next
    time and is the file the editor edits."""
    path = _manual_path(book, stem)
    if os.path.exists(path):
        return json.load(open(path))
    ov = seed_from_sidecar(book, stem) or seed_override(book, stem)
    save_override(book, stem, ov)
    return ov


def save_override(book: str, stem: str, ov: dict) -> None:
    os.makedirs(_manual_dir(book), exist_ok=True)
    ov["stem"] = stem
    with open(_manual_path(book, stem), "w") as fh:
        json.dump(ov, fh, ensure_ascii=False, indent=1)


def _col_of(node: dict) -> int:
    return (node["box"][0] + node["box"][2]) // 2


def normalize(ov: dict) -> dict:
    """Recompute father links from children and keep children eldest-first (RTL).

    The client edits children/father; this is the authority so the two stay
    consistent and the sibling order matches the book (rightmost column = eldest).
    """
    by_id = {n["id"]: n for n in ov["nodes"]}
    for n in ov["nodes"]:
        n["children"] = [c for c in n["children"] if c in by_id]
        n["children"].sort(key=lambda c: -_col_of(by_id[c]))
    father = {}
    for n in ov["nodes"]:
        for c in n["children"]:
            father[c] = n["id"]
    for n in ov["nodes"]:
        n["father"] = father.get(n["id"], -1)
    return ov


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def graph_png(book: str, stem: str) -> tuple[bytes, int, int]:
    """The graph image at FULL resolution, plus (width, height).

    Served full-res so 1 image pixel == 1 SVG coordinate: the browser's CSS
    transform on #wrap (image + overlay together) does all the zoom/pan, which is
    what keeps the overlay locked to the scan. Downscaling here would desync them.
    """
    a = get_image(_graph_path(book, stem))
    h, w = a.shape
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    return _png_bytes(img), w, h


_COMPARE_CACHE: dict[str, Image.Image] = {}


def _compare_img(book: str, stem: str) -> Image.Image:
    """Build (and cache) the pages/crops strip. Cached because it is slow and both
    ``/compare`` and ``/parse`` (for its dimensions) need it."""
    key = f"{book}/{stem}"
    if key not in _COMPARE_CACHE:
        start, end = (int(x) for x in stem.split("_"))
        _COMPARE_CACHE[key] = _fit_width(
            stacked_compare(book, start, end, seg.BOOK_CONFIGS[book], BOOKS_DIR))
    return _COMPARE_CACHE[key]


def _compare_dims(book: str, stem: str) -> tuple[int, int]:
    img = _compare_img(book, stem)
    return img.width, img.height


def compare_png(book: str, stem: str) -> bytes:
    return _png_bytes(_compare_img(book, stem))


def recut_crops(book: str, stem: str, ov: dict) -> None:
    """(Optional) re-cut each node's name crop from the graph for moved/added boxes.

    Writes ``books/{book}/5_names_manual/{stem}_{id}.png`` -- a manual-crop area so
    the real 5_names is not clobbered before Stage 5 consumes the override. OCR of
    these is a later step.
    """
    a = get_image(_graph_path(book, stem))
    out_dir = os.path.join(BOOKS_DIR, book, "5_names_manual")
    os.makedirs(out_dir, exist_ok=True)
    for n in ov["nodes"]:
        l, t, r, b = n["box"]
        if r > l and b > t:
            save_image(a[t:b, l:r], os.path.join(out_dir, f"{stem}_{n['id']}.png"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A002
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
        q = parse_qs(parsed.query)
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/list":
            out = []
            for stem in _graph_stems(BOOK):
                has = os.path.exists(_manual_path(BOOK, stem))
                out.append({"stem": stem, "edited": has})
            self._send(200, json.dumps({"book": BOOK, "graphs": out}, ensure_ascii=False))
        elif path == "/parse":
            stem = q.get("stem", [""])[0]
            ov = normalize(load_override(BOOK, stem))
            a = get_image(_graph_path(BOOK, stem))
            h, w = a.shape
            cw, ch = _compare_dims(BOOK, stem)  # to align the strip to the graph width
            self._send(200, json.dumps(
                {"stem": stem, "nodes": ov["nodes"], "w": int(w), "h": int(h),
                 "cw": int(cw), "ch": int(ch)},
                ensure_ascii=False))
        elif path == "/graph":
            stem = q.get("stem", [""])[0]
            try:
                png, _, _ = graph_png(BOOK, stem)
                self._send(200, png, "image/png")
            except FileNotFoundError:
                self._send(404, b"not found", "text/plain")
        elif path == "/compare":
            stem = q.get("stem", [""])[0]
            try:
                self._send(200, compare_png(BOOK, stem), "image/png")
            except Exception as e:  # compare strip is a nicety; never fatal
                logger.warning("compare %s failed: %s", stem, e)
                self._send(404, b"no compare", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if parsed.path == "/save":
            stem = body["stem"]
            ov = normalize({"stem": stem, "nodes": body["nodes"]})
            save_override(BOOK, stem, ov)
            try:
                recut_crops(BOOK, stem, ov)
            except Exception as e:
                logger.warning("recut crops %s failed: %s", stem, e)
            roots = sum(1 for n in ov["nodes"] if n["father"] == -1)
            self._send(200, json.dumps({"ok": True, "roots": roots,
                                        "nodes": len(ov["nodes"])}))
        elif parsed.path == "/reseed":
            stem = body["stem"]
            # Fast by default (from the on-disk sidecar); live=true re-parses (slow).
            if body.get("live"):
                ov = normalize(seed_override(BOOK, stem))
            else:
                ov = normalize(seed_from_sidecar(BOOK, stem) or seed_override(BOOK, stem))
            save_override(BOOK, stem, ov)
            self._send(200, json.dumps({"ok": True, "nodes": len(ov["nodes"])}))
        else:
            self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html><meta charset=utf-8>
<title>Parse editor</title>
<style>
 body{margin:0;font-family:system-ui;background:#1a1a1a;color:#eee}
 #bar{position:sticky;top:0;background:#000;padding:6px 10px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 select,button{font-size:13px;padding:4px 8px;background:#333;color:#eee;border:1px solid #555;border-radius:4px}
 button:hover{background:#444}
 #status{margin-left:auto;font-size:13px}
 .hint{font-size:12px;color:#9ab}
 /* Dedicated to checking/fixing detected nodes + edges: just the parse graph with
    the editable overlay, in one pan/zoom viewport (always perfectly aligned). */
 #view{position:relative;width:100%;height:88vh;overflow:hidden;background:#111;
       border-top:1px solid #333;cursor:grab}
 #view.panning{cursor:grabbing}
 #wrap{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}
 #g{display:block;position:absolute;left:0;top:0}
 svg{position:absolute;left:0;top:0;overflow:visible}
 .box{fill:rgba(255,40,40,.06);stroke:#e22;stroke-width:2;cursor:move}
 .box.sel{stroke:#0cf;stroke-width:4;fill:rgba(0,200,255,.10)}
 .box.root{stroke:#fa0}
 .edge{stroke:#e22;stroke-width:2;fill:none}
 .dot{fill:#e22;cursor:pointer}
 .hd{fill:#0cf;stroke:#fff;stroke-width:1;cursor:nwse-resize}
</style>
<div id=bar>
 <select id=stem></select>
 <button onclick="mode='connect'">Connect child→parent (c)</button>
 <button onclick="mode='add'">Add node (a)</button>
 <button onclick="delSel()">Delete (Del)</button>
 <button onclick="cleanArtifacts()" title="delete boxes <100px tall &amp; orphaned, or ≥300px wide; flag isolated tall nodes">Clean artifacts (k)</button>
 <button onclick="reseed()">Re-seed</button>
 <button onclick="save()">Save (s)</button>
 <button onclick="fitView()">Fit (f)</button>
 <span class=hint id=modehint>mode: select</span>
 <span id=status></span>
</div>
<div class=hint style="padding:4px 10px">Check &amp; fix detected nodes + edges (pan: drag empty space · zoom: wheel · Fit (f)). Drag box to move, corner to resize. Select a node then 'c' + click a parent to connect. 'a' + click to add. 'k' = clean artifacts (delete tiny-orphan / super-wide boxes, flag isolated nodes).</div>
<div id=view><div id=wrap><img id=g><svg id=ov></svg></div></div>
<script>
// --- state -----------------------------------------------------------------
// The SVG overlay is drawn in GRAPH-IMAGE pixels; a single CSS transform on #wrap
// (translate,scale) maps image px -> screen for the image AND the SVG together, so
// the overlay can never drift out of alignment with the scan. Mouse events convert
// screen -> image px through the same transform.
let stem="", data=null, imgW=0, imgH=0, sel=null, mode="select", nextId=0, drag=null;
let vx=0, vy=0, vz=1;   // viewport transform (screen px translate, scale)
const view=document.getElementById('view'), wrap=document.getElementById('wrap'),
      gimg=document.getElementById('g'), ov=document.getElementById('ov'),
      stemSel=document.getElementById('stem'),
      status=document.getElementById('status'), modehint=document.getElementById('modehint');

function setMode(m){mode=m;modehint.textContent="mode: "+m}
function applyView(){wrap.style.transform=`translate(${vx}px,${vy}px) scale(${vz})`;render();}
const DEFAULT_ZOOM_FRAC = 0.3;  // graphs open at 30% of fit-to-width (less zoomed in)
function fitWidthZoom(){ const vw=view.clientWidth; return imgW ? vw/imgW : 1; }
function fitView(){         // the Fit button / 'f': fit the whole width to the viewport
  vz = fitWidthZoom(); vx = 0; vy = 0; applyView();
}
function defaultView(){     // initial zoom when a graph loads: zoomed out to 30% of fit
  vz = fitWidthZoom() * DEFAULT_ZOOM_FRAC; vx = 0; vy = 0; applyView();
}
async function loadList(){
  const r=await fetch('/list');const j=await r.json();
  stemSel.innerHTML=j.graphs.map(g=>`<option value="${g.stem}">${g.stem}${g.edited?' ✓':''}</option>`).join('');
  loadGraph(j.graphs[0].stem);
}
function layout(){
  gimg.style.width=imgW+'px';gimg.style.height=imgH+'px';
  ov.setAttribute('viewBox',`0 0 ${imgW} ${imgH}`);
  ov.setAttribute('width',imgW);ov.setAttribute('height',imgH);
  ov.style.width=imgW+'px';ov.style.height=imgH+'px';
  wrap.style.width=imgW+'px';wrap.style.height=imgH+'px';
}
async function loadGraph(s){
  stem=s;stemSel.value=s;sel=null;setMode('select');
  const r=await fetch('/parse?stem='+s);data=await r.json();imgW=data.w;imgH=data.h;
  nextId=Math.max(0,...data.nodes.map(n=>n.id))+1;
  gimg.onload=()=>{layout();defaultView();};
  gimg.src='/graph?stem='+s+'&t='+Date.now();
}
function byId(id){return data.nodes.find(n=>n.id===id)}
function render(){
  // stroke/handle sizes are divided by vz so they stay visually constant while zooming
  const k=1/vz, sw=2*k, dr=6*k, hd=11*k;
  let e='';
  for(const n of data.nodes){
    if(n.father!==-1){const p=byId(n.father);if(p){
      const cx=n.top[1],cy=n.top[0],px=p.bot[1],py=p.bot[0],my=(cy+py)/2;
      e+=`<path class=edge style="stroke-width:${sw}" d="M${px} ${py} L${px} ${my} L${cx} ${my} L${cx} ${cy}"/>`;
    }}
  }
  for(const n of data.nodes){
    const[l,t,r,b]=n.box;
    const cls='box'+(n===sel?' sel':'')+(n.father===-1?' root':'');
    e+=`<rect class="${cls}" style="stroke-width:${n===sel?2*sw:sw}" data-id="${n.id}" x="${l}" y="${t}" width="${r-l}" height="${b-t}"/>`;
    // Dots mark REAL branch attachments (and are the handles to drag them): the top
    // dot only if the node has a father (a branch comes down into it), the bot dot
    // only if it has children (branches fan out below). So a leaf shows one dot (its
    // top), a root shows one (its bot), an isolated node none. Box move/resize uses
    // the box + corner handle, not the dots.
    if(n.father!==-1)
      e+=`<circle class=dot cx="${n.top[1]}" cy="${n.top[0]}" r="${dr}" data-end="top" data-id="${n.id}"/>`;
    if(n.children.length)
      e+=`<circle class=dot cx="${n.bot[1]}" cy="${n.bot[0]}" r="${dr}" data-end="bot" data-id="${n.id}"/>`;
  }
  // Resize handle LAST (on top of everything) so a box never intercepts its click.
  // A generous circle at the box's bottom-right corner; radius scales with zoom.
  if(sel){const r=sel.box[2],b=sel.box[3];
    e+=`<circle class=hd cx="${r}" cy="${b}" r="${hd}" data-id="${sel.id}" data-h="1"/>`;}
  ov.innerHTML=e;
  status.textContent=`${data.nodes.length} nodes, ${data.nodes.filter(n=>n.father===-1).length} root(s) · zoom ${(vz*100|0)}%`;
}
// screen (client) coords -> graph image pixels, via the inverse wrap transform.
function toImg(ev){const R=view.getBoundingClientRect();
  return[(ev.clientX-R.left-vx)/vz,(ev.clientY-R.top-vy)/vz];}

view.addEventListener('mousedown',ev=>{
  const el=ev.target;const id=el.dataset&&el.dataset.id?+el.dataset.id:null;const[ox,oy]=toImg(ev);
  if(mode==='add'){
    const n={id:nextId++,box:[ox|0,oy|0,(ox+200)|0,(oy+330)|0],top:[oy|0,(ox+100)|0],bot:[(oy+330)|0,(ox+100)|0],children:[],father:-1,name:""};
    data.nodes.push(n);sel=n;setMode('select');render();return;}
  if(el.dataset&&el.dataset.h){drag={t:'resize',n:byId(id)};ev.preventDefault();return;}
  if(el.dataset&&el.dataset.end){drag={t:'end',n:byId(id),end:el.dataset.end};ev.preventDefault();return;}
  if(id!==null){const n=byId(id);
    if(mode==='connect'&&sel&&sel!==n){sel.father=n.id;if(!n.children.includes(sel.id))n.children.push(sel.id);setMode('select');fixTree();render();return;}
    sel=n;const[l,t]=n.box;drag={t:'move',n,dx:ox-l,dy:oy-t};render();ev.preventDefault();return;}
  // empty space -> pan
  sel=null;drag={t:'pan',sx:ev.clientX,sy:ev.clientY,ox:vx,oy:vy};view.classList.add('panning');render();
});
window.addEventListener('mousemove',ev=>{
  if(!drag)return;
  if(drag.t==='pan'){vx=drag.ox+(ev.clientX-drag.sx);vy=drag.oy+(ev.clientY-drag.sy);applyView();return;}
  const[ox,oy]=toImg(ev);const n=drag.n;
  if(drag.t==='move'){const w=n.box[2]-n.box[0],h=n.box[3]-n.box[1];const nl=(ox-drag.dx)|0,nt=(oy-drag.dy)|0;
    n.box=[nl,nt,nl+w,nt+h];n.top=[n.box[1],(n.box[0]+n.box[2])/2|0];n.bot=[n.box[3],(n.box[0]+n.box[2])/2|0];}
  else if(drag.t==='resize'){n.box[2]=Math.max(n.box[0]+20,ox|0);n.box[3]=Math.max(n.box[1]+20,oy|0);}
  else if(drag.t==='end'){n[drag.end]=[oy|0,ox|0];}
  render();
});
window.addEventListener('mouseup',()=>{if(drag){if(drag.t==='pan')view.classList.remove('panning');drag=null}});
// wheel zoom, anchored on the cursor
view.addEventListener('wheel',ev=>{ev.preventDefault();
  const R=view.getBoundingClientRect();const mx=ev.clientX-R.left,my=ev.clientY-R.top;
  const ix=(mx-vx)/vz,iy=(my-vy)/vz;           // image point under cursor
  const f=ev.deltaY<0?1.05:1/1.05;vz=Math.max(0.02,Math.min(8,vz*f)); // gentle (~3x less)
  vx=mx-ix*vz;vy=my-iy*vz;applyView();
},{passive:false});
function fixTree(){const col=n=>(n.box[0]+n.box[2])/2;
  for(const n of data.nodes)n.children.sort((a,b)=>col(byId(b))-col(byId(a)));}
// Auto-clean this graph's artifact boxes. DELETE: a box <100px tall AND orphaned
// (no father AND no children), OR a box >=300px wide. FLAG (kept, reported): a node
// with no father AND no children that is >=100px tall (a real-sized isolated node
// William should look at). Runs on the in-memory graph; save to persist.
function cleanArtifacts(){
  const H=n=>n.box[3]-n.box[1], W=n=>n.box[2]-n.box[0];
  const orphan=n=>n.father===-1 && n.children.length===0;
  const del=data.nodes.filter(n=> (H(n)<100 && orphan(n)) || W(n)>=300 );
  const flag=data.nodes.filter(n=> orphan(n) && H(n)>=100 && !del.includes(n) );
  if(!del.length && !flag.length){status.textContent='clean: no artifacts found';return;}
  const ids=new Set(del.map(n=>n.id));
  data.nodes=data.nodes.filter(n=>!ids.has(n.id));
  for(const n of data.nodes){n.children=n.children.filter(c=>!ids.has(c));
    if(ids.has(n.father))n.father=-1;}
  if(sel && ids.has(sel.id)) sel=null;
  render();
  const flagMsg = flag.length ? ` · FLAG ${flag.length} isolated node(s) at cols [`+flag.map(n=>((n.box[0]+n.box[2])/2|0)).join(', ')+']' : '';
  status.textContent=`cleaned: deleted ${del.length} artifact(s)${flagMsg} — press s to save`;
  if(flag.length) console.log('Isolated (no father/children, >=100px) nodes to review:',
    flag.map(n=>({id:n.id,box:n.box})));
}
function delSel(){if(!sel)return;const id=sel.id;
  data.nodes=data.nodes.filter(n=>n!==sel);
  for(const n of data.nodes){n.children=n.children.filter(c=>c!==id);if(n.father===id)n.father=-1;}
  sel=null;render();}
async function save(){fixTree();
  const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem,nodes:data.nodes})});
  const j=await r.json();
  // Advance to the NEXT graph in the dropdown (don't jump back to the first). Mark
  // this one edited (✓) in place, then load the next; wrap-around stays put at the end.
  const opt=[...stemSel.options].find(o=>o.value===stem);
  if(opt && !opt.text.includes('✓')) opt.text=stem+' ✓';
  const idx=stemSel.selectedIndex;
  if(idx < stemSel.options.length-1){
    status.textContent=`saved ${stem}: ${j.nodes} nodes, ${j.roots} root(s) → next`;
    loadGraph(stemSel.options[idx+1].value);
  } else {
    status.textContent=`saved ${stem}: ${j.nodes} nodes, ${j.roots} root(s) (last graph)`;
  }}
async function reseed(){if(!confirm('Re-seed from parser? discards manual edits for '+stem))return;
  await fetch('/reseed',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stem})});
  loadGraph(stem);}
stemSel.addEventListener('change',()=>loadGraph(stemSel.value));
window.addEventListener('keydown',ev=>{
  if(ev.target.tagName==='SELECT')return;
  if(ev.key==='c')setMode('connect');else if(ev.key==='a')setMode('add');
  else if(ev.key==='f')fitView();
  else if(ev.key==='k')cleanArtifacts();
  else if(ev.key==='s'){ev.preventDefault();save();}
  else if(ev.key==='Delete'||ev.key==='Backspace')delSel();
  else if(ev.key==='Escape'){sel=null;setMode('select');render();}
});
loadList();
</script>
"""


def main(argv: list[str] | None = None) -> None:
    global BOOK
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--book", default="book3")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args(argv)
    BOOK = args.book
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Parse editor for %s at http://localhost:%d/", BOOK, args.port)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
