"""Render the QA overlay for a subgraph directly from its manual override file.

Reads data/{book}_manual/{stem}.json (the frozen ground-truth node list) and draws
the SAME overlay the real QA uses, so the corrected result is verified visually.
Nothing is re-parsed: the override is the source of truth.
"""
import sys, os, json, base64

sys.path.insert(0, os.getcwd())
from src.imaging import get_image
from scripts.qa.s5_parse import draw_parse_overlay


def main():
    book, stem = sys.argv[1], sys.argv[2]
    man = json.load(open(f"data/{book}_manual/{stem}.json"))
    by_id = {n["id"]: n for n in man["nodes"]}

    # Build the parse-sidecar shape draw_parse_overlay expects: per node box, top,
    # bot, empty, children_top (each child's top point).
    node_recs = []
    for n in man["nodes"]:
        node_recs.append({
            "id": n["id"],
            "box": n["box"],
            "top": n["top"],
            "bot": n["bot"],
            "empty": n.get("empty", False),
            "children_top": [by_id[c]["top"] for c in n["children"] if c in by_id],
        })
    parse = {"nodes": node_recs, "scrubbed": []}

    a = get_image(f"books/{book}/4_graphs/{stem}.png")
    img = draw_parse_overlay(a, parse, imaginary=man.get("imaginary", []),
                             nicks=man.get("nicks", []))
    out_dir = f"/private/tmp/claude-501/-Users-wizeng-Documents-repos-misc-zeng-dynasty/7d4324ad-2438-4032-9b88-29f56d68b5d0/scratchpad/qa_manual_{book}_{stem}"
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{stem}_parse.png")
    img.save(png)

    roots = [n["id"] for n in man["nodes"] if n["father"] == -1]
    print(f"{stem} (manual): {len(man['nodes'])} nodes, {len(roots)} root(s), "
          f"{len(man.get('imaginary', []))} bridge(s)")

    with open(png, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    html = f"""<!doctype html><meta charset=utf-8><title>QA {book} {stem} MANUAL</title>
<body style="margin:0;background:#eee;font-family:system-ui">
<div style="padding:8px 12px;background:#036;color:#fff">
  {book} / {stem} — MANUAL OVERRIDE: {len(man['nodes'])} nodes, {len(roots)} root(s),
  {len(man.get('imaginary', []))} green bridge(s) — from data/{book}_manual/{stem}.json
</div>
<img src="data:image/png;base64,{b64}" style="width:100%;display:block"></body>"""
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(html)
    print(f"  wrote {out_dir}/index.html")


if __name__ == "__main__":
    main()
