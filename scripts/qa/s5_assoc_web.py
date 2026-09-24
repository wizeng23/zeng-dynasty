"""Web QA: review the graph<->bio association one SUBGRAPH at a time; verify or flag.

For every subgraph (a ``{start}_{end}`` stem) the page lists the parsed graph nodes, generation
by generation, next to the bio each one linked to, and audits the pair:

  * **name**   -- bio name == graph node name;
  * **father** -- bio father char (header ``子<order><X>``) == last char of the graph father;
  * **sons**   -- bio son list == the node's children, in age order. Children are read from the
    CROSS-BOOK stitched tree (``data/tree_bio_stitched.jsonl``), so a Book-3 leaf whose sons were
    folded in from Book 4 is checked against them; a leaf whose Book-4 subtree did NOT fold
    shows "graph has no children" (yellow) rather than a hard mismatch.

A subgraph is EXACT when no graph node lacks a bio, no bio is left unlinked, and every linked
pair passes all three checks. Anything else is highlighted. Verify / Flag (with an optional
note) both save and advance to the next subgraph.

Review state -> ``{data-dir}/{book}_bio_assoc_review.json`` =
``{stem: {"status": "verified"|"flagged", "note": str}}``. Each save is a read-modify-write of
ONE stem's entry (never a whole-file rewrite from memory), so concurrent/other edits survive.
Nothing else is ever written.

Data (linked tree, link report, 5_records) is read from this checkout; the scan images (graph
PNGs, name crops, bio block crops -- gitignored) from ``--assets-dir`` (default: the main
checkout).

Run::

    PYTHONPATH=. python -m scripts.qa.s5_assoc_web --port 8770
    # -> http://localhost:8770/   (keys: v verify, f flag, → / s skip, ← back)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import src.bio.s5_link as L

BOOKS = ["book3", "book4"]
TAG = {"book3": "b3", "book4": "b4"}
DATA_DIR = "data"
BOOKS_DIR = "books"
ASSETS_DIR = "/Users/wizeng/Documents/repos/misc/zeng-dynasty"
_LOCK = threading.Lock()


def _prov(n: dict) -> str:
    return (n.get("notes", "") or "").split(" | ", 1)[0].strip()


def _v(d, key):
    """A per-reader field ({vision, paddle}) -> its best string/list."""
    x = d.get(key) if isinstance(d, dict) else None
    if isinstance(x, dict):
        return x.get("vision") or x.get("paddle")
    return x


# --- data ---------------------------------------------------------------------------

def load_book(book: str) -> dict:
    """Everything the page needs for one book, keyed and audited."""
    tree = L.load_tree(os.path.join(DATA_DIR, f"{book}_bio_linked.jsonl"))
    by_id = {n["id"]: n for n in tree}

    # cross-book stitched children, keyed by the combined notes head "{tag}:{prov}"
    comb = [json.loads(l) for l in open(os.path.join(DATA_DIR, "tree_bio_stitched.jsonl"))]
    cby = {n["id"]: n for n in comb}
    comb_kids = {_prov(n): [cby[c]["name"] for c in n["children"] if c in cby] for n in comb}

    # bio records (full father header, raw son cols) keyed by block id
    recs = {}
    for f in glob.glob(os.path.join(BOOKS_DIR, book, "bio", "5_records", "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            recs[r["id"]] = r

    report = json.load(open(os.path.join(DATA_DIR, f"{book}_bio_link_report.json")))
    sec2stem = L.section_to_stem(book, BOOKS_DIR)
    unlinked_bios = collections.defaultdict(list)
    for sec, sd in report["sections"].items():
        for bid in sd.get("unmatched_blocks", []):
            r = recs.get(bid, {})
            unlinked_bios[sec2stem.get(sec)].append({
                "block": bid, "generation": r.get("generation"),
                "name": _v(r, "name"), "father_header": r.get("father_header"),
                "sons": _v(r, "sons") or [], "img": f"/asset/{book}/bio/3_segment/{bid}.png"})

    by_stem = collections.defaultdict(list)
    for n in tree:
        s = L._stem_of(n)
        if s:
            by_stem[s].append(n)
    stems = sorted(by_stem, key=lambda s: [int(x) for x in s.split("_")])

    subgraphs = {}
    for stem in stems:
        rows = []
        for n in sorted(by_stem[stem], key=lambda n: (n["generation"], n["id"])):
            fa = by_id.get(n["father"])
            kids = comb_kids.get(f"{TAG[book]}:{_prov(n)}")
            if kids is None:  # not in the combined tree -> per-book children
                kids = [by_id[c]["name"] for c in n.get("children", []) if c in by_id]
            row = {
                "id": n["id"], "prov": _prov(n), "generation": n["generation"],
                "name": n["name"], "father": fa["name"] if fa else None,
                "children": kids,
                "name_img": (f"/asset/{n['name_images'][0]}" if n.get("name_images") else None),
                "bio": None, "checks": {},
            }
            bio = n.get("bio")
            if bio:
                rec = recs.get(bio.get("block_id"), {})
                b_name = _v(bio, "name_ocr") or _v(bio, "name")
                b_fc = _v(bio, "father_char")
                b_sons = _v(bio, "sons") or []
                row["bio"] = {
                    "block": bio.get("block_id"), "name": b_name, "father_char": b_fc,
                    "father_header": rec.get("father_header"), "sons": b_sons,
                    "sons_raw": rec.get("sons_raw", []), "qa_flags": rec.get("qa_flags", []),
                    "img": f"/asset/{book}/bio/3_segment/{bio.get('block_id')}.png",
                }
                c = row["checks"]
                c["name"] = "ok" if b_name == n["name"] else "bad"
                if not fa or not fa.get("name") or not b_fc:
                    c["father"] = "na"
                else:
                    c["father"] = "ok" if fa["name"].endswith(b_fc) else "bad"   # b_fc may be an IDS
                if b_sons == kids:
                    c["sons"] = "ok"
                elif b_sons and not kids:
                    c["sons"] = "graph_empty"     # sons may live in the next book (unfolded)
                else:
                    c["sons"] = "bad"
            rows.append(row)
        ub = sorted(unlinked_bios.get(stem, []), key=lambda b: b["block"])
        n_nobio = sum(1 for r in rows if not r["bio"])
        n_bad = sum(1 for r in rows if r["bio"] and "bad" in r["checks"].values())
        n_soft = sum(1 for r in rows if r["bio"] and r["checks"].get("sons") == "graph_empty")
        subgraphs[stem] = {
            "stem": stem, "rows": rows, "unlinked_bios": ub,
            "graph_img": f"/asset/{book}/4_graphs/{stem}.png",
            "summary": {"nodes": len(rows), "no_bio": n_nobio, "unlinked_bios": len(ub),
                        "bad": n_bad, "soft": n_soft,
                        "exact": n_nobio == 0 and not ub and n_bad == 0 and n_soft == 0},
        }
    return {"stems": stems, "subgraphs": subgraphs}


def review_path(book: str) -> str:
    return os.path.join(DATA_DIR, f"{book}_bio_assoc_review.json")


def load_review(book: str) -> dict:
    p = review_path(book)
    if not os.path.exists(p):
        return {}
    raw = json.load(open(p))
    # tolerate the terminal tool's {stem: "verified"} shape
    return {k: (v if isinstance(v, dict) else {"status": v, "note": ""}) for k, v in raw.items()}


def save_one(book: str, stem: str, status: str | None, note: str) -> dict:
    """Read-modify-write ONE stem's review entry (status None = clear that stem only)."""
    with _LOCK:
        cur = load_review(book)
        if status is None:
            cur.pop(stem, None)
        else:
            cur[stem] = {"status": status, "note": note}
        tmp = review_path(book) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, review_path(book))
        return cur


# --- page -----------------------------------------------------------------------------

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Bio ↔ graph association QA</title>
<style>
:root{--bg:#fafaf8;--fg:#222;--mut:#777;--ok:#1b7f3b;--bad:#c62828;--soft:#b7791f;--line:#e2e2dc;--card:#fff}
body{margin:0;font:14px/1.45 -apple-system,system-ui,"PingFang SC",sans-serif;background:var(--bg);color:var(--fg)}
header{position:fixed;top:0;left:0;right:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:15px} select,button,input{font:inherit}
button{padding:4px 11px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer}
button.v{border-color:var(--ok);color:var(--ok)} button.f{border-color:var(--bad);color:var(--bad)}
.pill{padding:1px 8px;border-radius:10px;font-size:12px;background:#eee}
.pill.verified{background:#e3f4e8;color:var(--ok)} .pill.flagged{background:#fde8e8;color:var(--bad)}
main{padding:calc(var(--hdr-h,56px) + 12px) 16px 80px;max-width:1500px}
.sum{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 12px}
.sum span{padding:3px 9px;border-radius:6px;background:#fff;border:1px solid var(--line)}
.sum .exact{border-color:var(--ok);color:var(--ok);font-weight:600}
.sum .warn{border-color:var(--bad);color:var(--bad);font-weight:600}
table{border-collapse:collapse;width:100%;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:5px 7px;vertical-align:top;text-align:left}
td{white-space:nowrap} td:last-child{white-space:normal}  /* keep names/son lists on one line; scan column wraps */
th{background:#f3f3ef;font-weight:600;font-size:12px;color:#555;position:sticky;top:var(--hdr-h,56px);z-index:2}
tr.gen td{background:#f7f7f3;font-weight:600;color:#555}
tr.bad{background:#fff5f5} tr.nobio{background:#fffbea} tr.soft{background:#fffdf3}
.ok{color:var(--ok)} .bad{color:var(--bad);font-weight:600} .soft{color:var(--soft)} .na{color:var(--mut)}
.mut{color:var(--mut);font-size:12px} .nm{font-size:17px}
img.nm{height:34px;vertical-align:middle;border:1px solid var(--line);background:#fff}
img.blk{height:150px;border:1px solid var(--line);cursor:zoom-in;background:#fff}
.son{display:inline-block;padding:0 4px;margin:1px;border-radius:4px}
.son.both{background:#e3f4e8} .son.only{background:#fde8e8}
#zoom{display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:20;overflow:auto;cursor:zoom-out}
#zoom img{display:block;margin:20px auto;max-width:96vw}
.ub{margin-top:14px;padding:8px 10px;border:1px solid var(--bad);border-radius:6px;background:#fff}
textarea{width:320px;height:26px;vertical-align:middle}
</style></head><body>
<header>
  <b>Bio ↔ graph association</b>
  <select id="book"></select>
  <select id="filter"><option value="todo">to review (not exact, undecided)</option><option value="all">all subgraphs</option><option value="flagged">flagged only</option><option value="notexact">not exact (incl. decided)</option></select>
  <button id="prev">← prev</button><span id="pos"></span><button id="next">skip →</button>
  <span id="state" class="pill"></span>
  <button class="v" id="vbtn">✓ Verify (v)</button>
  <textarea id="note" placeholder="flag note (optional)"></textarea>
  <button class="f" id="fbtn">⚑ Flag (f)</button>
  <button id="clr" title="remove this subgraph's decision">clear</button>
  <span class="mut" id="counts"></span>
</header>
<main id="main">loading…</main>
<div id="zoom" onclick="this.style.display='none'"><img id="zimg"></div>
<script>
let BOOK=null, DATA=null, REVIEW={}, LIST=[], I=0;
new ResizeObserver(([e])=>document.documentElement.style.setProperty("--hdr-h",
  document.querySelector("header").offsetHeight+"px")).observe(document.querySelector("header"));
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function zoom(src){ $("zimg").src=src; $("zoom").style.display="block"; }
async function loadBook(b){
  BOOK=b; $("main").textContent="loading "+b+"…";
  DATA=await (await fetch("/data?book="+b)).json();
  REVIEW=await (await fetch("/review?book="+b)).json();
  buildList(); I=0; render();
}
function buildList(){
  const f=$("filter").value;
  LIST=DATA.stems.filter(s=>{
    const sm=DATA.subgraphs[s].summary, r=REVIEW[s];
    if(f==="all") return true;
    if(f==="flagged") return r&&r.status==="flagged";
    if(f==="notexact") return !sm.exact;
    return !sm.exact && !r;
  });
  const v=Object.values(REVIEW).filter(r=>r.status==="verified").length, fl=Object.values(REVIEW).filter(r=>r.status==="flagged").length;
  const ex=DATA.stems.filter(s=>DATA.subgraphs[s].summary.exact).length;
  $("counts").textContent=`${DATA.stems.length} subgraphs · ${ex} exact · ${v} verified · ${fl} flagged · ${LIST.length} in view`;
}
function chk(c){ return c==="ok"?'<span class="ok">✓</span>':c==="bad"?'<span class="bad">✗</span>':c==="graph_empty"?'<span class="soft">?</span>':'<span class="na">–</span>'; }
function sonsCell(bio, kids){
  const g=new Set(kids), b=new Set(bio);
  const bs=bio.map(s=>`<span class="son ${g.has(s)?"both":"only"}">${esc(s)}</span>`).join("")||'<span class="mut">none</span>';
  const gs=kids.map(s=>`<span class="son ${b.has(s)?"both":"only"}">${esc(s)}</span>`).join("")||'<span class="mut">none</span>';
  return [bs,gs];
}
function render(){
  if(!LIST.length){ $("main").innerHTML="<p>Nothing in this view. 🎉 Switch the filter to see more.</p>"; $("pos").textContent="0/0"; $("state").textContent=""; return; }
  I=Math.max(0,Math.min(I,LIST.length-1));
  const s=LIST[I], sg=DATA.subgraphs[s], sm=sg.summary, r=REVIEW[s];
  $("pos").textContent=`${I+1}/${LIST.length}`;
  $("state").className="pill "+(r?r.status:""); $("state").textContent=r?(r.status+(r.note?": "+r.note:"")):"undecided";
  $("note").value=r&&r.note?r.note:"";
  let h=`<h2 style="margin:4px 0">Subgraph ${esc(s)} <a class="mut" href="${sg.graph_img}" target="_blank">open graph image ↗</a></h2>`;
  h+=`<div class="sum"><span class="${sm.exact?"exact":"warn"}">${sm.exact?"EXACT MATCH":"NOT EXACT"}</span>
      <span>${sm.nodes} graph nodes</span><span class="${sm.no_bio?"warn":""}">${sm.no_bio} node(s) without a bio</span>
      <span class="${sm.unlinked_bios?"warn":""}">${sm.unlinked_bios} unlinked bio(s)</span>
      <span class="${sm.bad?"warn":""}">${sm.bad} failed check(s)</span>
      <span>${sm.soft} leaf(s) whose sons aren't in the graph</span></div>`;
  h+=`<table><tr><th>graph node</th><th>graph father</th><th>bio name</th><th>bio father</th><th>bio sons</th><th>graph children (cross-book)</th><th>sons</th><th>bio scan (click to zoom)</th></tr>`;
  let g=null;
  for(const row of sg.rows){
    if(row.generation!==g){ g=row.generation; h+=`<tr class="gen"><td colspan="8">generation ${g}</td></tr>`; }
    const b=row.bio, c=row.checks;
    const cls=!b?"nobio":Object.values(c).includes("bad")?"bad":c.sons==="graph_empty"?"soft":"";
    const gimg=row.name_img?`<img class="nm" src="${row.name_img}" loading="lazy"> `:"";
    let bioCell, nameC="", faC="", sonsC="", bS="", gS="";
    if(!b){ bioCell='<span class="bad">no bio linked</span>'; nameC=bioCell; [bS,gS]=sonsCell([],row.children); }
    else{
      bioCell=`<img class="blk" src="${b.img}" loading="lazy" onclick="zoom(this.src)"><div class="mut">${esc(b.block)}${b.qa_flags.length?' · <span class="bad">'+esc(b.qa_flags.join("; "))+'</span>':""}</div>`;
      nameC=`${chk(c.name)} <span class="nm">${esc(b.name)}</span>`;
      faC=`${chk(c.father)} ${esc(b.father_header||"")} <span class="mut">(${esc(b.father_char||"")})</span>`;
      [bS,gS]=sonsCell(b.sons,row.children);
      if(b.sons_raw.length && b.sons_raw.join("|")!==b.sons.join("|")) bS+=`<div class="mut">raw: ${esc(b.sons_raw.join(" | "))}</div>`;
      sonsC=c.sons==="graph_empty"?'<span class="soft" title="graph has no children here (sons may continue in Book 4)">?</span>':chk(c.sons);
    }
    h+=`<tr class="${cls}"><td>${gimg}<span class="nm">${esc(row.name)}</span><div class="mut">${esc(row.prov)} · id ${row.id}</div></td>
        <td>${esc(row.father||"—")}</td><td>${nameC}</td><td>${faC}</td><td>${bS}</td><td>${gS}</td><td>${sonsC}</td><td>${bioCell}</td></tr>`;
  }
  h+="</table>";
  if(sg.unlinked_bios.length){
    h+=`<div class="ub"><b class="bad">Bios with no graph node (${sg.unlinked_bios.length})</b><table><tr><th>gen</th><th>name</th><th>father</th><th>sons</th><th>bio scan</th></tr>`;
    for(const u of sg.unlinked_bios) h+=`<tr><td>${u.generation??""}</td><td class="nm">${esc(u.name)}</td><td>${esc(u.father_header||"")}</td><td>${u.sons.map(esc).join("、")||'<span class="mut">none</span>'}</td><td><img class="blk" src="${u.img}" onclick="zoom(this.src)"><div class="mut">${esc(u.block)}</div></td></tr>`;
    h+="</table></div>";
  }
  $("main").innerHTML=h; window.scrollTo(0,0);
}
async function decide(status){
  const s=LIST[I]; if(!s) return;
  const note=status==="flagged"?$("note").value.trim():"";
  REVIEW=await (await fetch("/review",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({book:BOOK,stem:s,status,note})})).json();
  const f=$("filter").value;
  buildList();
  if(status===null){ I=Math.max(0,LIST.indexOf(s)); render(); return; }
  if(f==="todo"){ render(); }          // decided item drops out of the to-review list
  else { I=LIST.indexOf(s)+1; render(); }
}
$("vbtn").onclick=()=>decide("verified"); $("fbtn").onclick=()=>decide("flagged"); $("clr").onclick=()=>decide(null);
$("next").onclick=()=>{I++;render()}; $("prev").onclick=()=>{I--;render()};
$("filter").onchange=()=>{buildList();I=0;render()};
$("book").onchange=e=>loadBook(e.target.value);
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT") return;
  if(e.key==="v") decide("verified"); else if(e.key==="f") decide("flagged");
  else if(e.key==="ArrowRight"||e.key==="s"){I++;render()} else if(e.key==="ArrowLeft"){I--;render()}
  else if(e.key==="Escape") $("zoom").style.display="none";
});
(async()=>{ const bs=await (await fetch("/books")).json();
  $("book").innerHTML=bs.map(b=>`<option>${b}</option>`).join(""); loadBook(bs[0]); })();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    cache: dict = {}

    def log_message(self, format, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/books":
            return self._send(200, json.dumps(BOOKS))
        if u.path == "/data":
            book = q.get("book", [""])[0]
            if book not in BOOKS:
                return self._send(400, '{"error":"bad book"}')
            if book not in self.cache:
                self.cache[book] = load_book(book)
            return self._send(200, json.dumps(self.cache[book], ensure_ascii=False))
        if u.path == "/review":
            book = q.get("book", [""])[0]
            return self._send(200, json.dumps(load_review(book), ensure_ascii=False))
        if u.path.startswith("/asset/"):
            rel = unquote(u.path[len("/asset/"):])
            if rel.startswith("books/"):          # node name_images are repo-relative
                rel = rel[len("books/"):]
            path = os.path.realpath(os.path.join(ASSETS_DIR, "books", rel))
            root = os.path.realpath(os.path.join(ASSETS_DIR, "books"))
            if not path.startswith(root + os.sep) or not os.path.isfile(path):
                return self._send(404, b"", "text/plain")
            return self._send(200, open(path, "rb").read(), "image/png")
        self._send(404, b"", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/review":
            return self._send(404, b"", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        book, stem = body.get("book"), body.get("stem")
        status = body.get("status")
        if book not in BOOKS or not stem or status not in ("verified", "flagged", None):
            return self._send(400, '{"error":"bad request"}')
        cur = save_one(book, stem, status, (body.get("note") or "").strip())
        self._send(200, json.dumps(cur, ensure_ascii=False))


def main(argv=None):
    global DATA_DIR, BOOKS_DIR, ASSETS_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="where linked data is read AND review state is written")
    ap.add_argument("--books-dir", default=BOOKS_DIR)
    ap.add_argument("--assets-dir", default=ASSETS_DIR,
                    help="checkout holding the gitignored scan images (books/...)")
    a = ap.parse_args(argv)
    DATA_DIR, BOOKS_DIR, ASSETS_DIR = a.data_dir, a.books_dir, a.assets_dir
    print(f"Bio<->graph association QA: http://localhost:{a.port}/  "
          f"(review state -> {os.path.join(DATA_DIR, '{book}_bio_assoc_review.json')})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
