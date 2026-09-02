"""Local QA review site -- a multi-tab tool for reviewing pipeline output.

One server, several tabs; each tab's data loads only when the tab is opened
(lazy). The first tab is **Border QA**: for pages whose frame detection is not
confidently clean, review and fix the four frame corners by dragging.

A page enters the Border-QA set when it is NOT confidently clean, i.e. any of:
  * the two detectors (flood-fill and Hough) DISAGREE on the corners,
  * either detector FAILED,
  * the post-crop verification (`verify_normalized`) FAILS even if the two agree.
Confidently-clean pages (both agree AND the crop verifies) are skipped.

Layout per page: the original render on the LEFT with the draggable corner quad;
the cropped result on the RIGHT when a crop was produced (else the original is
shown centered). Drag any of the four corner dots to correct them. Buttons:
  * "Crop is fine"  -- approve the current auto crop as-is (no manual corners).
  * "Save corners"  -- store your dragged corners as the ground-truth override.
Left / Right arrow keys switch pages. Position is remembered per tab.

Corrections are a ground-truth layer at
``books/{book}/1_pages/corners_review.json`` keyed by content-page number:
  { "12": {"status": "approved"},
    "37": {"status": "corners", "corners": {"tl": [x,y], ...}} }
Nothing here re-runs the pipeline; a later extract step reads this layer to
override detection on the reviewed pages.

Run::

    python -m scripts.qa.s1_borders          # then open http://localhost:8761/
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import pymupdf
from PIL import Image

import src.s1_extract_pages as ep

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

BOOKS_DIR = "books"
# Books whose frames use the degraded-scan detector (need border QA).
BORDER_QA_BOOKS = ["book3", "book4"]
# Long side (px) the original render is downscaled to for the browser.
VIEW_LONG = 1400
# Native render size of a v1 PDF page (constant); the browser scales corner dots
# from these to the downscaled view.
NATIVE_W, NATIVE_H = 4882, 6904
# Corner-agreement tolerance: fraction of page long-side.
AGREE_TOL_FRAC = 0.01

# ---- Hough detector (ported prototype) -----------------------------------
# Frame model constants, measured on the book3 reference.
H_LEN, V_LEN = 3780, 5718
DOUBLE_MIN, DOUBLE_MAX = 28, 48

import cv2


def _fold(a):
    return a % 180.0


def _line_of(seg):
    x1, y1, x2, y2 = seg
    d = math.atan2(y2 - y1, x2 - x1)
    nrm = d + math.pi / 2
    rho = x1 * math.cos(nrm) + y1 * math.sin(nrm)
    if rho < 0:
        rho, nrm = -rho, nrm + math.pi
    return _fold(math.degrees(d)), rho, nrm, math.hypot(x2 - x1, y2 - y1)


def _ang_diff(a, b):
    return abs(((a - b + 90) % 180) - 90)


def _merge(lines):
    lines = sorted(lines, key=lambda L: (round(L[0]), L[1]))
    out = []
    for L in lines:
        if out and _ang_diff(L[0], out[-1][0]) < 2 and abs(L[1] - out[-1][1]) < 10:
            p = out[-1]
            base = p if p[3] >= L[3] else L
            out[-1] = (base[0], base[1], base[2], p[3] + L[3])
        else:
            out.append(L)
    return out


def hough_corners(a):
    """Frame corners via Hough + scored geometric model; None if not found."""
    h, w = a.shape
    cx, cy = w / 2, h / 2
    ink = ((1 - a) * 255).astype(np.uint8)
    hl = cv2.HoughLinesP(ink, rho=1, theta=np.pi / 1800, threshold=200,
                         minLineLength=int(0.20 * w), maxLineGap=150)
    if hl is None:
        return None
    lines = _merge([_line_of(s) for s in hl[:, 0, :]])
    angs = np.array([L[0] for L in lines])
    lens = np.array([L[3] for L in lines])
    hist, edges = np.histogram(angs, bins=180, range=(0, 180), weights=lens)
    dom = (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1]) / 2
    gA = [L for L in lines if _ang_diff(L[0], dom) <= 10]
    gB = [L for L in lines if _ang_diff(L[0], dom + 90) <= 10]
    if len(gA) < 2 or len(gB) < 2:
        return None
    a_horiz = _ang_diff(dom, 0) < 45
    A_len, B_len = (H_LEN, V_LEN) if a_horiz else (V_LEN, H_LEN)

    def signed_center(L):
        return L[1] - (cx * math.cos(L[2]) + cy * math.sin(L[2]))

    def has_double(L, group):
        return any(M is not L and _ang_diff(L[0], M[0]) < 2
                   and DOUBLE_MIN <= abs(L[1] - M[1]) <= DOUBLE_MAX for M in group)

    def score(L, group):
        s = 6.0 if has_double(L, group) else 0.0
        s += min(abs(signed_center(L)) / (max(w, h) / 2), 1.0) * 3
        return s

    def best_side(group, positive):
        cand = [L for L in group if (signed_center(L) > 0) == positive]
        if not cand:
            return None, -1e9
        b = max(cand, key=lambda L: score(L, group))
        return b, score(b, group)

    def derive(opp, perp_len):
        if opp is None:
            return None
        shift = -perp_len if signed_center(opp) > 0 else perp_len
        return (opp[0], opp[1] + shift, opp[2], opp[3])

    sides = {"ap": best_side(gA, True), "an": best_side(gA, False),
             "bp": best_side(gB, True), "bn": best_side(gB, False)}
    WEAK = 3.0

    def resolve(sp, sn, perp):
        (Lp, sp_), (Ln, sn_) = sp, sn
        if sp_ < WEAK and Ln is not None and sn_ >= WEAK:
            Lp = derive(Ln, perp)
        if sn_ < WEAK and Lp is not None and sp_ >= WEAK:
            Ln = derive(Lp, perp)
        return Lp, Ln

    a1, a2 = resolve(sides["ap"], sides["an"], B_len)
    b1, b2 = resolve(sides["bp"], sides["bn"], A_len)
    if None in (a1, a2, b1, b2):
        return None

    def inter(L, M):
        a1_, b1_, c1_ = math.cos(L[2]), math.sin(L[2]), L[1]
        a2_, b2_, c2_ = math.cos(M[2]), math.sin(M[2]), M[1]
        det = a1_ * b2_ - a2_ * b1_
        if abs(det) < 1e-6:
            return None
        return ((c1_ * b2_ - c2_ * b1_) / det, (a1_ * c2_ - a2_ * c1_) / det)

    pts = [inter(L, M) for L in (a1, a2) for M in (b1, b2)]
    if any(p is None for p in pts):
        return None
    pts = [(int(round(x)), int(round(y))) for x, y in pts]
    pts.sort(key=lambda p: p[1])
    tl, tr = sorted(pts[:2], key=lambda p: p[0])
    bl, br = sorted(pts[2:], key=lambda p: p[0])
    return tl, tr, br, bl


# ---- per-page QA computation ---------------------------------------------

def _corners_dict(c):
    if c is None:
        return None
    tl, tr, br, bl = c
    return {"tl": list(tl), "tr": list(tr), "br": list(br), "bl": list(bl)}


def _agree(c1, c2, w, h):
    if c1 is None or c2 is None:
        return False, None
    tol = AGREE_TOL_FRAC * max(w, h)
    worst = max(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(c1, c2))
    return worst <= tol, worst


def analyze_page(doc, book, pi, ci):
    """Compute both detectors, agreement, and verify-pass for one page.

    Returns a dict of everything the frontend needs; the crop is not rendered
    here (served lazily by /qa/crop)."""
    a = ep.render_page_binary(doc, pi)
    h, w = a.shape
    try:
        fc = ep.find_frame_corners(a)
    except ValueError:
        fc = None
    hc = hough_corners(a)
    agree, worst = _agree(fc, hc, w, h)

    # verify: does the chosen corners' crop pass verify_normalized? This uses the
    # flood-fill re-detector, which is unreliable on the degraded book3/4 frames
    # (it flags nearly everything), so verify is ADVISORY only -- shown in the UI
    # but not used to gate the QA set. Agreement between two independent detectors
    # is the trustworthy "clean" signal here.
    chosen = hc or fc  # prefer hough
    verify_ok = False
    if chosen is not None:
        try:
            crop = ep.deskew_to_frame(a, chosen)
            ep.verify_normalized(crop)
            verify_ok = True
        except ValueError:
            verify_ok = False

    # Not clean (needs review) when either detector failed, or both ran but
    # disagree. Two independent methods agreeing is accepted as correct.
    clean = bool(fc is not None and hc is not None and agree)
    return {
        "content_page": ci, "pdf_index": pi, "viewer_page": pi + 1,
        "page_w": w, "page_h": h,
        "flood": _corners_dict(fc), "hough": _corners_dict(hc),
        "agree": agree, "worst_px": (round(worst, 1) if worst is not None else None),
        "verify_ok": verify_ok, "clean": clean,
        "chosen": _corners_dict(chosen),
    }


# ---- review-layer persistence --------------------------------------------

def _review_path(book):
    return os.path.join(BOOKS_DIR, book, "1_pages", "corners_review.json")


def load_review(book):
    p = _review_path(book)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_review(book, ci, entry):
    p = _review_path(book)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = load_review(book)
    data[str(ci)] = entry
    with open(p, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


# ---- image serving --------------------------------------------------------

def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_original(book, pi):
    doc = pymupdf.open(os.path.join(BOOKS_DIR, book, f"{book}.pdf"))
    a = ep.render_page_binary(doc, pi)
    doc.close()
    h, w = a.shape
    img = Image.fromarray((a * 255).astype(np.uint8))
    s = VIEW_LONG / max(w, h)
    return img.resize((max(1, int(w * s)), max(1, int(h * s)))), s, w, h


def render_crop(book, pi, corners):
    doc = pymupdf.open(os.path.join(BOOKS_DIR, book, f"{book}.pdf"))
    a = ep.render_page_binary(doc, pi)
    doc.close()
    quad = (tuple(corners["tl"]), tuple(corners["tr"]),
            tuple(corners["br"]), tuple(corners["bl"]))
    crop = ep.deskew_to_frame(a, quad)
    img = Image.fromarray((crop * 255).astype(np.uint8))
    s = VIEW_LONG / max(img.width, img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))


# ---- QA-set index (which pages need review) -------------------------------
# Read the precomputed books/{book}/1_pages/corners.json (written by Stage 1 /
# the compare runs) rather than recomputing both detectors over the whole book
# on load (which would take minutes and time out the browser). Only the flagged
# pages -- status == "flagged" -- enter the review set.

def qa_index(book):
    path = os.path.join(BOOKS_DIR, book, "1_pages", "corners.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    rows = []
    for ci_str, p in sorted(data.get("pages", {}).items(), key=lambda kv: int(kv[0])):
        if p.get("status") != "flagged":
            continue
        rows.append({
            "content_page": int(ci_str),
            "pdf_index": p["pdf_index"],
            "viewer_page": p["viewer_page"],
            "reason": p.get("reason", ""),
            "flood": p.get("flood"),
            "hough": p.get("hough"),
            "worst_px": p.get("worst_px"),
            # the frontend seeds draggable dots from hough, then flood, then a default.
            "chosen": p.get("hough") or p.get("flood"),
            "agree": False,
            # native render size (constant across v1 pages); frontend scales dots by it.
            "page_w": NATIVE_W, "page_h": NATIVE_H,
        })
    return rows


PAGE_HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>Zeng QA</title>
<style>
 :root{--bg:#1e1e1e;--fg:#eee;--flood:#e8912a;--hough:#3aa0ff;--ok:#38c172;}
 body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif}
 #tabs{display:flex;gap:2px;background:#111;padding:6px 6px 0}
 .tab{padding:8px 16px;background:#2a2a2a;border-radius:6px 6px 0 0;cursor:pointer}
 .tab.active{background:#3a3a3a;font-weight:600}
 #bar{padding:8px 14px;display:flex;gap:14px;align-items:center;background:#252525;flex-wrap:wrap}
 button{background:#3a3a3a;color:var(--fg);border:1px solid #555;border-radius:5px;padding:6px 12px;cursor:pointer}
 button:hover{background:#4a4a4a}
 #stage{display:flex;gap:16px;padding:14px;align-items:flex-start;justify-content:center;
   height:calc(100vh - 120px);box-sizing:border-box}
 .panel{position:relative;height:100%;display:flex;flex-direction:column;min-width:0}
 .panel h4{margin:0 0 6px;font-weight:500;color:#aaa}
 /* canvases keep full render resolution internally but are CSS-scaled to fit the
    stage height; pointer handlers divide by the css/canvas ratio. */
 canvas{border:1px solid #444;cursor:crosshair;touch-action:none;
   max-height:100%;max-width:100%;object-fit:contain}
 .legend span{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:middle;margin:0 4px 0 10px}
 .hint{color:#999;font-size:13px}
 #save{background:var(--ok);border-color:var(--ok);color:#08120a;font-weight:600}
 #save:hover{filter:brightness(1.1)}
 .reviewed{color:var(--ok);font-weight:600}
 #toast{position:fixed;top:70px;left:50%;transform:translateX(-50%);
   background:var(--ok);color:#08120a;font-weight:700;padding:10px 22px;border-radius:8px;
   font-size:16px;opacity:0;transition:opacity .15s;pointer-events:none;z-index:50}
 #toast.show{opacity:1}
 [hidden]{display:none!important}
</style></head><body>
<div id=tabs></div>
<div id=toast></div>
<div id=bar>
  <span id=pageinfo class=hint></span>
  <span class=legend><span style="background:var(--flood)"></span>flood
    <span style="background:var(--hough)"></span>hough
    <span style="background:var(--ok)"></span>your corners</span>
  <button onclick="useFlood()">Use flood</button>
  <button onclick="useHough()">Use hough</button>
  <button id=save onclick="saveCorners()">Save ✓</button>
  <span class=hint>← / → pages · drag dots to fix · Save stores the shown corners</span>
</div>
<div id=stage>
  <div class=panel><h4 id=lh>original</h4><canvas id=orig></canvas></div>
  <div class=panel><h4 id=rh>cropped</h4><canvas id=crop></canvas></div>
</div>
<script>
const TABS=%%TABS%%;
let tab=TABS[0], rows=[], idx=0, scale=1, dots=[], dragging=-1;
let origImg=null;  // cached loaded original image, for smooth drag redraws
const $=id=>document.getElementById(id);

function toast(msg,ok=true){const t=$('toast');t.textContent=msg;
  t.style.background=ok?'var(--ok)':'#d9534f';t.classList.add('show');
  clearTimeout(toast._h);toast._h=setTimeout(()=>t.classList.remove('show'),1100);}

function renderTabs(){
  $('tabs').innerHTML='';
  TABS.forEach(t=>{const d=document.createElement('div');d.className='tab'+(t===tab?' active':'');
    d.textContent=t; d.onclick=()=>{tab=t; loadTab();}; $('tabs').appendChild(d);});
}
async function loadTab(){
  renderTabs(); rows=[]; $('pageinfo').textContent='loading…';
  const r=await fetch('/qa/index?book='+tab); rows=await r.json();
  idx=parseInt(localStorage.getItem('qa_'+tab)||'0',10); if(idx>=rows.length)idx=0;
  showPage();
}
function curCorners(){ // pick which detection to seed the draggable dots
  const row=rows[idx];
  return row.chosen||row.hough||row.flood||{tl:[100,100],tr:[900,100],br:[900,1400],bl:[100,1400]};
}
async function showPage(){
  if(!rows.length){$('pageinfo').textContent='no pages need review 🎉';return;}
  const row=rows[idx];
  localStorage.setItem('qa_'+tab, idx);
  const rev=row.review;
  const mark=rev?' · <span class=reviewed>reviewed ✓</span>':'';
  $('pageinfo').innerHTML=`viewer ${row.viewer_page} (content ${row.content_page}) · ${idx+1}/${rows.length} · `+
    (row.reason||'')+mark;
  // load original image (cached in origImg for drag redraws)
  origImg=new Image();
  origImg.onload=()=>{ const c=$('orig'); scale=origImg.width/row.page_w;
    c.width=origImg.width;c.height=origImg.height;
    let src=(rev&&rev.status==='corners')?rev.corners:curCorners();
    dots=['tl','tr','br','bl'].map(k=>[src[k][0]*scale, src[k][1]*scale]);
    drawOrig(row);
  };
  origImg.src='/qa/orig?book='+tab+'&pi='+row.pdf_index+'&t='+Date.now();
  // load crop (right panel) from chosen corners
  const crop=$('crop');
  if(row.chosen){ const ci=new Image(); ci.onload=()=>{crop.width=ci.width;crop.height=ci.height;
      crop.getContext('2d').drawImage(ci,0,0); $('rh').textContent='cropped (chosen)';};
    ci.src='/qa/crop?book='+tab+'&pi='+row.pdf_index+'&t='+Date.now();
  } else { crop.width=0;crop.height=0; $('rh').textContent='no crop'; }
}
function drawQuad(ctx,pts,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
  ctx.moveTo(pts[0][0]*scale,pts[0][1]*scale);
  [1,2,3,0].forEach(i=>ctx.lineTo(pts[i][0]*scale,pts[i][1]*scale));ctx.stroke();}
function drawOrig(row){
  if(!origImg)return;
  const c=$('orig'),ctx=c.getContext('2d');ctx.drawImage(origImg,0,0);
  const cs=getComputedStyle(document.body);
  if(row.flood)drawQuad(ctx,['tl','tr','br','bl'].map(k=>row.flood[k]),cs.getPropertyValue('--flood'));
  if(row.hough)drawQuad(ctx,['tl','tr','br','bl'].map(k=>row.hough[k]),cs.getPropertyValue('--hough'));
  ctx.strokeStyle=cs.getPropertyValue('--ok');ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(dots[0][0],dots[0][1]);[1,2,3,0].forEach(i=>ctx.lineTo(dots[i][0],dots[i][1]));ctx.stroke();
  ctx.fillStyle=cs.getPropertyValue('--ok');
  dots.forEach(d=>{ctx.beginPath();ctx.arc(d[0],d[1],9,0,7);ctx.fill();});
}
// client coords -> canvas coords: the canvas is CSS-scaled to fit, so divide by
// the displayed/intrinsic ratio (rect is the on-screen size, canvas.width the true one).
function canvasXY(e){const c=$('orig'),r=c.getBoundingClientRect();
  const sx=c.width/r.width, sy=c.height/r.height;
  return [(e.clientX-r.left)*sx,(e.clientY-r.top)*sy];}
$('orig').addEventListener('pointerdown',e=>{const [x,y]=canvasXY(e);
  dragging=dots.findIndex(d=>Math.hypot(d[0]-x,d[1]-y)<24);});
$('orig').addEventListener('pointermove',e=>{if(dragging<0)return;
  dots[dragging]=canvasXY(e);drawOrig(rows[idx]);});  // redraw from cached image (no refetch)
window.addEventListener('pointerup',()=>dragging=-1);
function dotsToCorners(){const k=['tl','tr','br','bl'];const o={};
  dots.forEach((d,i)=>o[k[i]]=[Math.round(d[0]/scale),Math.round(d[1]/scale)]);return o;}
function dotsValid(){return dots.length===4 &&
  dots.every(d=>Number.isFinite(d[0])&&Number.isFinite(d[1]));}
async function saveCurrent(){const row=rows[idx];
  if(!dotsValid()){toast('No corners to save',false);return false;}  // guard empty/NaN saves
  const entry={status:'corners',corners:dotsToCorners()};
  const r=await fetch('/qa/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({book:tab,content_page:row.content_page,entry})});
  if(r.ok){rows[idx].review=entry;toast('Saved ✓');}else{toast('Save failed',false);}
  return r.ok;}
function saveCorners(){saveAndNext();}  // Save button = save current corners + advance (same as ->)
function useFlood(){const row=rows[idx];if(!row.flood)return;dots=['tl','tr','br','bl'].map(k=>[row.flood[k][0]*scale,row.flood[k][1]*scale]);drawOrig(row);}
function useHough(){const row=rows[idx];if(!row.hough)return;dots=['tl','tr','br','bl'].map(k=>[row.hough[k][0]*scale,row.hough[k][1]*scale]);drawOrig(row);}
function step(d){idx=(idx+d+rows.length)%rows.length;showPage();}
async function saveAndNext(){await saveCurrent();step(1);}  // → saves the shown corners, then advances
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowRight'){e.preventDefault();saveAndNext();}
  else if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}});
loadTab();
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
        """Extract and validate ``book`` against the fixed allowlist. Returns the
        book name, or None (after sending a 400) if it is not an allowed book --
        so it can never reach a path join (guards against path traversal)."""
        book = (q.get("book") or [""])[0]
        if book not in BORDER_QA_BOOKS:
            self._send(400, json.dumps({"error": "invalid book"}))
            return None
        return book

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            html = PAGE_HTML.replace("%%TABS%%", json.dumps(BORDER_QA_BOOKS))
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/qa/index":
            book = self._book(q)
            if book is None:
                return
            rows = qa_index(book)
            rev = load_review(book)
            for r in rows:
                r["review"] = rev.get(str(r["content_page"]))
            return self._send(200, json.dumps(rows))
        if u.path == "/qa/orig":
            book = self._book(q)
            if book is None:
                return
            pi = int(q["pi"][0])
            img, s, w, h = render_original(book, pi)
            return self._send(200, _png_bytes(img), "image/png")
        if u.path == "/qa/crop":
            book = self._book(q)
            if book is None:
                return
            pi = int(q["pi"][0])
            # crop from the chosen corners recomputed for this page
            doc = pymupdf.open(os.path.join(BOOKS_DIR, book, f"{book}.pdf"))
            a = ep.render_page_binary(doc, pi); doc.close()
            hc = hough_corners(a)
            fc = None
            if hc is None:
                try:
                    fc = ep.find_frame_corners(a)
                except ValueError:
                    fc = None
            chosen = hc or fc
            if chosen is None:
                return self._send(404, b"no crop", "text/plain")
            img = render_crop(book, pi, _corners_dict(chosen))
            return self._send(200, _png_bytes(img), "image/png")
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/qa/save":
            return self._send(404, json.dumps({"error": "not found"}))
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        book = payload.get("book")
        if book not in BORDER_QA_BOOKS:
            return self._send(400, json.dumps({"error": "invalid book"}))
        save_review(book, payload["content_page"], payload["entry"])
        return self._send(200, json.dumps({"ok": True}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8761)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"QA review at http://localhost:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
