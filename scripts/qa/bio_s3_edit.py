"""Editable QA for bio Stage 3: fix a section's person-block boxes in the browser.

Mirrors ``scripts/qa/s5_edit.py`` (server + canvas + SVG overlay) but for biography
sections instead of tree graphs. Each merged bio section is shown as one image with
its 4 generation rules drawn; a red box marks each detected person block. You move /
resize / add / delete boxes per band, and the tool writes a per-section ground-truth
override to ``data/{book}_bio_manual/{stem}.json`` that REPLACES Stage-3 detection.

The merged images are huge (up to ~25786px wide), so the image is served **downscaled**
by ``DISPLAY_SCALE`` for fast loading; the overlay and all editing happen in display
pixels, and boxes are scaled back to full resolution on save.

The header bar shows the live **count gate** -- detected boxes vs tree-oracle node
count for each generation (2..6) -- so you know when a band is complete and correct.

CLI:
    PYTHONPATH=. python -m scripts.qa.bio_s3_edit --book book3 --port 8788
    # then open http://localhost:8788/
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

from src.imaging import get_image
from src.bio import s3_segment as seg

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

BOOKS_DIR = "books"
DATA_DIR = "data"
BOOK = "book3"          # set by main()
DISPLAY_SCALE = 6       # served image is 1/DISPLAY_SCALE; boxes scaled back up on save


def _merged_dir(book: str) -> str:
    return os.path.join(BOOKS_DIR, book, seg.BIO_DIR, seg.MERGED_DIR)


def _segment_dir(book: str) -> str:
    return os.path.join(BOOKS_DIR, book, seg.BIO_DIR, seg.SEGMENT_DIR)


def _qa_input_path(book: str, sec: str) -> str:
    """Pre-QA detections written by s3_segment (the editor's seed input)."""
    return os.path.join(_segment_dir(book), seg.QA_INPUT_DIR, f"{sec}.jsonl")


def _approved_path(book: str, sec: str) -> str:
    """QA-approved full-copy JSONL the editor saves (read by s3_post)."""
    return os.path.join(_segment_dir(book), f"{sec}.jsonl")


def _section_stems(book: str) -> list[str]:
    d = _merged_dir(book)
    return sorted((f[:-4] for f in os.listdir(d) if f.endswith(".png")),
                  key=lambda s: int(s.split("_")[0]))


# --- tree oracle (expected count per generation, per section) ----------------------

_ORACLE: dict[str, list[int]] = {}


def _oracle(book: str) -> dict[str, list[int]]:
    """Map section stem -> expected node count for generations 2..6."""
    if book in _ORACLE:  # cached across requests
        return _ORACLE[book]  # type: ignore[return-value]
    # PRE-STITCH parse (bio sections map 1-1 to pre-stitch graphs; stitch drops/rewrites
    # stems -- see src/bio/s3_segment.py).
    by = seg.tree_counts_by_stem(os.path.join(DATA_DIR, f"{book}.jsonl"))
    sec_to_stem = seg.map_sections_to_stems(_section_stems(book), list(by))
    out = {sec: [by[stem].get(g, 0) for g in seg.BAND_GENERATIONS]
           for sec, stem in sec_to_stem.items()}
    _ORACLE[book] = out  # type: ignore[assignment]
    _SEC_TO_STEM[book] = sec_to_stem  # type: ignore[assignment]
    return out


_SEC_TO_STEM: dict[str, dict[str, str]] = {}


def _section_stem(sec: str) -> str:
    """Tree subgraph stem for a bio section (from the positional section<->stem map)."""
    _oracle(BOOK)  # ensures _SEC_TO_STEM is populated
    return _SEC_TO_STEM.get(BOOK, {}).get(sec, "")


# --- section state load / save -----------------------------------------------------
# The editor works with {section, bands:[{gen,top,bottom}], boxes:[{id,gen,band,x,y,w,h}]}.
# It reads the block JSONL (rows of {id,band,generation,box:[l,t,r,b]}) that s3_segment
# wrote, plus the merged section's rules_y for band boundaries. On save it writes a full
# copy back as JSONL, recomputing each box's provenance id so d stays in RTL order.

def _bands_of(book: str, sec: str) -> list[dict]:
    with open(os.path.join(_merged_dir(book), f"{sec}.json")) as fh:
        meta = json.load(fh)
    a = get_image(os.path.join(_merged_dir(book), f"{sec}.png"))
    h = a.shape[0]
    tops = [0] + meta["rules_y"]
    bots = meta["rules_y"] + [h]
    return [{"gen": seg.BAND_GENERATIONS[i], "top": tops[i], "bottom": bots[i]}
            for i in range(5)]


def _rows_to_boxes(rows: list[dict]) -> list[dict]:
    boxes = []
    for row in rows:
        l, t, r, b = row["box"]
        boxes.append({"id": row["id"], "gen": row["generation"], "band": row["band"],
                      "x": l, "y": t, "w": r - l, "h": b - t})
    return boxes


def load_section(book: str, sec: str) -> dict:
    """Editor state for a section: the approved copy if it exists, else the pre-QA input."""
    path = _approved_path(book, sec) if os.path.exists(_approved_path(book, sec)) \
        else _qa_input_path(book, sec)
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    return {"section": sec, "stem": rows[0]["stem"] if rows else _section_stem(sec),
            "bands": _bands_of(book, sec), "boxes": _rows_to_boxes(rows)}


def save_section(book: str, sec: str, stem: str, bands: list[dict], boxes: list[dict],
                 expected: list[int]) -> list[dict]:
    """Write the approved full-copy JSONL, recomputing each box's {a}_{b}_{c}_{d} id
    (band index c, then d=0 rightmost/eldest increasing leftward). Returns the rows."""
    detected = [sum(1 for bx in boxes if bx["band"] == i) for i in range(5)]
    passed = detected == expected
    rows = []
    for band_idx in range(5):
        band_boxes = sorted((bx for bx in boxes if bx["band"] == band_idx),
                            key=lambda bx: -(bx["x"] + bx["w"]))  # rightmost (eldest) first
        for d, bx in enumerate(band_boxes):
            row = {"section": sec, "stem": stem,
                   "id": f"{sec}_{band_idx}_{d}", "band": band_idx,
                   "generation": seg.BAND_GENERATIONS[band_idx],
                   "box": [bx["x"], bx["y"], bx["x"] + bx["w"], bx["y"] + bx["h"]],
                   "expected_per_gen": expected, "detected_per_gen": detected,
                   "gate_passed": passed}
            rows.append(row)
    os.makedirs(_segment_dir(book), exist_ok=True)
    with open(_approved_path(book, sec), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


# --- image serving (downscaled) ----------------------------------------------------

def section_png(book: str, stem: str) -> tuple[bytes, int, int, int, int]:
    """Downscaled merged section PNG + (disp_w, disp_h, full_w, full_h)."""
    a = get_image(os.path.join(_merged_dir(book), f"{stem}.png"))
    h, w = a.shape
    img = Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")
    dw, dh = max(1, w // DISPLAY_SCALE), max(1, h // DISPLAY_SCALE)
    small = img.resize((dw, dh))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    return buf.getvalue(), dw, dh, w, h


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
        q = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send(200, PAGE.replace("__BOOK__", BOOK), "text/html; charset=utf-8")
        elif parsed.path == "/list":
            out = [{"stem": s, "edited": os.path.exists(_approved_path(BOOK, s))}
                   for s in _section_stems(BOOK)]
            self._send(200, json.dumps({"book": BOOK, "sections": out}))
        elif parsed.path == "/section":
            sec = q.get("stem", [""])[0]
            st = load_section(BOOK, sec)
            _png, dw, dh, fw, fh = section_png(BOOK, sec)
            self._send(200, json.dumps({
                "stem": sec, "scale": DISPLAY_SCALE,
                "dw": dw, "dh": dh, "fw": fw, "fh": fh,
                "bands": st["bands"], "boxes": st["boxes"],
                "expected": _oracle(BOOK).get(sec, [0] * 5),
                "generations": list(seg.BAND_GENERATIONS),
            }))
        elif parsed.path == "/img":
            stem = q.get("stem", [""])[0]
            try:
                png, *_ = section_png(BOOK, stem)
                self._send(200, png, "image/png")
            except FileNotFoundError:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if parsed.path == "/save":
            sec = body["stem"]
            exp = _oracle(BOOK).get(sec, [0] * 5)
            save_section(BOOK, sec, _section_stem(sec), body["bands"], body["boxes"], exp)
            det = [sum(1 for b in body["boxes"] if b["band"] == i) for i in range(5)]
            self._send(200, json.dumps({"ok": True, "detected": det,
                                        "expected": exp, "gate": det == exp}))
        elif parsed.path == "/reseed":
            sec = body["stem"]
            path = _approved_path(BOOK, sec)
            if os.path.exists(path):
                os.remove(path)          # drop the approved copy -> falls back to qa_input
            st = load_section(BOOK, sec)
            self._send(200, json.dumps({"ok": True, "bands": st["bands"], "boxes": st["boxes"]}))
        else:
            self._send(404, b"not found", "text/plain")


PAGE = r"""<!doctype html><meta charset=utf-8>
<title>Bio block editor</title>
<style>
 body{margin:0;font-family:system-ui;background:#1a1a1a;color:#eee}
 #bar{position:sticky;top:0;background:#000;padding:6px 10px;z-index:5;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
 select,button{font-size:13px;padding:4px 8px;background:#333;color:#eee;border:1px solid #555;border-radius:4px}
 button:hover{background:#444}
 #gate{margin-left:auto;font-size:13px;font-family:ui-monospace,monospace}
 .g-ok{color:#3d7}.g-bad{color:#f66}
 #view{position:relative;width:100%;height:86vh;overflow:hidden;background:#111;border-top:1px solid #333;cursor:grab}
 #view.panning{cursor:grabbing}
 #wrap{position:absolute;left:0;top:0;transform-origin:0 0;will-change:transform}
 #g{display:block;position:absolute;left:0;top:0}
 svg{position:absolute;left:0;top:0;overflow:visible}
 .rule{stroke:#48f;stroke-width:1;stroke-dasharray:6 4}
 .box{fill:rgba(255,40,40,.06);stroke:#e22;stroke-width:1.5;cursor:move}
 .box.sel{stroke:#0cf;stroke-width:3;fill:rgba(0,200,255,.12)}
 .hd{fill:#0cf;stroke:#fff;stroke-width:1;cursor:nwse-resize}
 .hint{font-size:12px;color:#9ab;padding:4px 10px}
 .glabel{fill:#8ab;font-size:20px;font-family:ui-monospace,monospace}
</style>
<div id=bar>
 <select id=stem></select>
 <button onclick="setMode('add')">Add box (a)</button>
 <button onclick="delSel()">Delete (Del)</button>
 <button onclick="reseed()">Re-seed</button>
 <button onclick="save()">Save (s)</button>
 <button onclick="fitView()">Fit (f)</button>
 <span class=hint id=modehint>mode: select</span>
 <span id=gate></span>
</div>
<div class=hint>Boxes are person blocks per generation-band (blue dashed = rules). Fills screen height; ←/→ pan ~3/4 screen (starts at right edge = eldest). Drag box to move, corner to resize, 'a'+click to add (snaps to the band you click in), Del to remove. Image is downscaled for speed; boxes save at full resolution. Save advances to the next section.</div>
<div id=view><div id=wrap><img id=g><svg id=ov></svg></div></div>
<script>
const BOOK="__BOOK__";
let stem="",data=null,dw=0,dh=0,scale=6,sel=null,mode="select",nextId=0,drag=null;
let vx=0,vy=0,vz=1;
const view=document.getElementById('view'),wrap=document.getElementById('wrap'),
      gimg=document.getElementById('g'),ov=document.getElementById('ov'),
      stemSel=document.getElementById('stem'),modehint=document.getElementById('modehint'),
      gateEl=document.getElementById('gate');
function setMode(m){mode=m;modehint.textContent="mode: "+m}
function applyView(){wrap.style.transform=`translate(${vx}px,${vy}px) scale(${vz})`;render();}
// Fit the section's full HEIGHT to the viewport (bands stack vertically -> all 5 gens
// visible); pan horizontally through the people. Start scrolled to the right edge
// (eldest/first person is rightmost, RTL).
function fitHeightZoom(){const vh=view.clientHeight;return dh?vh/dh:1;}
function fitView(){vz=fitHeightZoom()*0.98;vy=0;vx=view.clientWidth-dw*vz;applyView();}
// horizontal pan by ~3/4 of a screen width. dir=-1 moves the view left (content right).
function panScreen(dir){vx-=dir*view.clientWidth*0.75;applyView();}
async function loadList(){
  const j=await(await fetch('/list')).json();
  stemSel.innerHTML=j.sections.map(s=>`<option value="${s.stem}">${s.stem}${s.edited?' ✓':''}</option>`).join('');
  // Restore the last-viewed section across reloads (per book); else start at the first.
  let start=j.sections[0].stem;
  try{const last=localStorage.getItem('s3edit_stem_'+BOOK);
      if(last && j.sections.some(s=>s.stem===last)) start=last;}catch(e){}
  loadSection(start);
}
// full-res -> display px
function d(v){return v/scale;}
function layout(){
  gimg.style.width=dw+'px';gimg.style.height=dh+'px';
  ov.setAttribute('viewBox',`0 0 ${dw} ${dh}`);ov.setAttribute('width',dw);ov.setAttribute('height',dh);
  ov.style.width=dw+'px';ov.style.height=dh+'px';wrap.style.width=dw+'px';wrap.style.height=dh+'px';
}
async function loadSection(s){
  stem=s;stemSel.value=s;sel=null;setMode('select');
  try{localStorage.setItem('s3edit_stem_'+BOOK,s);}catch(e){}   // remember for reload
  // Clear the OLD image + boxes IMMEDIATELY so the blank state signals "loading" and the
  // content reappearing signals "done" (the fetch + big image take a while).
  data={bands:[],boxes:[],generations:[],expected:[]};
  ov.innerHTML='';                 // drop old box/rule overlay
  gimg.removeAttribute('src');     // drop old image (canvas goes blank)
  gateEl.innerHTML='<span class=g-bad>loading '+s+'…</span>';
  render();
  data=await(await fetch('/section?stem='+s)).json();
  dw=data.dw;dh=data.dh;scale=data.scale;
  nextId=Math.max(0,...data.boxes.map(b=>+b.id.split('_').pop()||0))+1;
  gimg.onload=()=>{layout();fitView();};
  gimg.src='/img?stem='+s+'&t='+Date.now();
  updateGate();
}
function bandOfY(fy){ // which band index a full-res y falls in
  for(let i=0;i<data.bands.length;i++){if(fy>=data.bands[i].top&&fy<data.bands[i].bottom)return i;}
  return data.bands.length-1;
}
function updateGate(){
  const det=data.bands.map((_,i)=>data.boxes.filter(b=>b.band===i).length);
  const parts=data.generations.map((g,i)=>{
    const ok=det[i]===data.expected[i];
    return `<span class="${ok?'g-ok':'g-bad'}">g${g}:${det[i]}/${data.expected[i]}</span>`;
  });
  const all=det.every((v,i)=>v===data.expected[i]);
  gateEl.innerHTML=(all?'<span class=g-ok>GATE PASS</span> ':'<span class=g-bad>GATE FAIL</span> ')+parts.join(' ');
}
function render(){
  const k=1/vz,sw=1.5*k,hd=8*k;
  let e='';
  for(const bd of data.bands){ // rules between bands
    e+=`<line class=rule x1=0 y1=${d(bd.top)} x2=${dw} y2=${d(bd.top)} style="stroke-width:${sw}"/>`;
    e+=`<text class=glabel x=6 y=${d(bd.top)+22*k} style="font-size:${18*k}px">gen ${bd.gen}</text>`;
  }
  for(const b of data.boxes){
    const cls='box'+(b===sel?' sel':'');
    e+=`<rect class="${cls}" style="stroke-width:${b===sel?2*sw:sw}" data-id="${b.id}" x="${d(b.x)}" y="${d(b.y)}" width="${d(b.w)}" height="${d(b.h)}"/>`;
  }
  if(sel){e+=`<circle class=hd cx="${d(sel.x+sel.w)}" cy="${d(sel.y+sel.h)}" r="${hd}" data-h=1/>`;}
  ov.innerHTML=e;
}
// screen -> display px -> full-res px
function toFull(ev){const R=view.getBoundingClientRect();
  return[((ev.clientX-R.left-vx)/vz)*scale,((ev.clientY-R.top-vy)/vz)*scale];}
view.addEventListener('mousedown',ev=>{
  const el=ev.target,id=el.dataset&&el.dataset.id?el.dataset.id:null;const[fx,fy]=toFull(ev);
  if(mode==='add'){
    const bi=bandOfY(fy),bd=data.bands[bi];
    const b={id:`${stem}_${data.generations[bi]}_new${nextId++}`,gen:data.generations[bi],band:bi,
             x:fx|0,y:bd.top,w:400,h:bd.bottom-bd.top};
    data.boxes.push(b);sel=b;setMode('select');render();updateGate();return;}
  if(el.dataset&&el.dataset.h){drag={t:'resize',b:sel};ev.preventDefault();return;}
  if(id!==null){const b=data.boxes.find(x=>x.id===id);sel=b;drag={t:'move',b,dx:fx-b.x,dy:fy-b.y};render();ev.preventDefault();return;}
  sel=null;drag={t:'pan',sx:ev.clientX,sy:ev.clientY,ox:vx,oy:vy};view.classList.add('panning');render();
});
window.addEventListener('mousemove',ev=>{
  if(!drag)return;
  if(drag.t==='pan'){vx=drag.ox+(ev.clientX-drag.sx);vy=drag.oy+(ev.clientY-drag.sy);applyView();return;}
  const[fx,fy]=toFull(ev),b=drag.b;if(!b)return;
  if(drag.t==='move'){b.x=(fx-drag.dx)|0;b.y=(fy-drag.dy)|0;b.band=bandOfY(b.y+b.h/2);b.gen=data.generations[b.band];}
  else if(drag.t==='resize'){b.w=Math.max(40,(fx-b.x)|0);b.h=Math.max(40,(fy-b.y)|0);}
  render();
});
window.addEventListener('mouseup',()=>{if(drag){if(drag.t==='pan')view.classList.remove('panning');drag=null;updateGate();}});
view.addEventListener('wheel',ev=>{ev.preventDefault();
  const R=view.getBoundingClientRect(),mx=ev.clientX-R.left,my=ev.clientY-R.top;
  const ix=(mx-vx)/vz,iy=(my-vy)/vz,f=ev.deltaY<0?1.05:1/1.05;
  vz=Math.max(0.02,Math.min(12,vz*f));vx=mx-ix*vz;vy=my-iy*vz;applyView();},{passive:false});
function delSel(){if(!sel)return;data.boxes=data.boxes.filter(b=>b!==sel);sel=null;render();updateGate();}
async function save(){
  const j=await(await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem,bands:data.bands,boxes:data.boxes})})).json();
  const opt=[...stemSel.options].find(o=>o.value===stem);
  if(opt&&!opt.text.includes('✓'))opt.text=stem+' ✓';
  const idx=stemSel.selectedIndex;
  if(idx<stemSel.options.length-1)loadSection(stemSel.options[idx+1].value);
  else gateEl.textContent=(j.gate?'saved (gate PASS)':'saved (gate FAIL)')+' — last section';
}
async function reseed(){if(!confirm('Re-seed from detection? discards manual edits for '+stem))return;
  const j=await(await fetch('/reseed',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stem})})).json();
  data.bands=j.bands;data.boxes=j.boxes;sel=null;render();updateGate();}
stemSel.addEventListener('change',()=>loadSection(stemSel.value));
// Jump to the prev/next section in the dropdown (Shift+←/→).
function gotoSection(delta){
  const i=stemSel.selectedIndex, n=stemSel.options.length;
  const j=Math.min(n-1,Math.max(0,i+delta));
  if(j!==i)loadSection(stemSel.options[j].value);
}
window.addEventListener('keydown',ev=>{
  if(ev.target.tagName==='SELECT')return;
  // Shift+←/→: previous / next section.
  if(ev.shiftKey&&ev.key==='ArrowRight'){ev.preventDefault();gotoSection(1);return;}
  if(ev.shiftKey&&ev.key==='ArrowLeft'){ev.preventDefault();gotoSection(-1);return;}
  if(ev.key==='a')setMode('add');else if(ev.key==='s'){ev.preventDefault();save();}
  else if(ev.key==='f')fitView();else if(ev.key==='Delete'||ev.key==='Backspace')delSel();
  else if(ev.key==='ArrowRight'){ev.preventDefault();panScreen(1);}   // reveal content to the right
  else if(ev.key==='ArrowLeft'){ev.preventDefault();panScreen(-1);}   // reveal content to the left
  else if(ev.key==='Escape'){sel=null;setMode('select');render();}
});
loadList();
</script>
"""


def main(argv: list[str] | None = None) -> None:
    global BOOK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="book3")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    BOOK = args.book
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    logger.info("bio block editor for %s at http://localhost:%d/", BOOK, args.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
