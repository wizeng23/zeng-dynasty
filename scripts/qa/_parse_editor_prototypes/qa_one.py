"""Single-subgraph parse QA: parse ONE graph in memory and render its overlay,
with the pages/crops assembly view stacked ABOVE it (like the big QA page).

Usage: qa_one.py <book> <stem> [out_dir]
Runs bridge_orphans + parse_graph on books/<book>/4_graphs/<stem>.png (using the
current src/ find_lines), draws the parse overlay AND the raw-pages-over-crops
compare strip, and writes a standalone index.html so one fix is verified visually
without re-running the whole book.
"""
import sys, os, base64
import numpy as np

sys.path.insert(0, os.getcwd())
from src.imaging import get_image  # noqa: E402
import src.s5_build_tree as bt  # noqa: E402
from src import s3_segment as seg  # noqa: E402
from scripts.qa.s5_parse import draw_parse_overlay, stacked_compare, _fit_width  # noqa: E402


def main():
    book, stem = sys.argv[1], sys.argv[2]
    start, end = (int(x) for x in stem.split("_"))
    out_dir = sys.argv[3] if len(sys.argv) > 3 else \
        f"/private/tmp/claude-501/-Users-wizeng-Documents-repos-misc-zeng-dynasty/7d4324ad-2438-4032-9b88-29f56d68b5d0/scratchpad/qa_{book}_{stem}"
    os.makedirs(out_dir, exist_ok=True)

    cfg = bt.config_for(book)
    a = get_image(f"books/{book}/4_graphs/{stem}.png")
    a2, imaginary, nicks = bt.bridge_orphans(a, cfg)
    nodes = bt.parse_graph(a2, cfg, graph_stem=stem)
    for i, n in enumerate(nodes):
        n.id = i
    parse = bt.build_parse_sidecar(nodes, a2)

    # Parse overlay (bottom) + compare strip (pages over crops, top) -- same as big QA.
    parse_img = _fit_width(draw_parse_overlay(a2, parse, imaginary=imaginary, nicks=nicks))
    parse_img.save(os.path.join(out_dir, f"{stem}_parse.png"))
    compare_img = _fit_width(stacked_compare(book, start, end, seg.BOOK_CONFIGS[book], "books"))
    compare_img.save(os.path.join(out_dir, f"{stem}_compare.png"))

    roots = [n for n in nodes if not any(n in m.children for m in nodes)]
    print(f"{stem}: {len(nodes)} nodes, {len(roots)} root(s), "
          f"{len(imaginary)} bridge(s), {len(nicks)} nick(s)")

    def b64(name):
        with open(os.path.join(out_dir, name), "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    html = f"""<!doctype html><meta charset=utf-8><title>QA {book} {stem}</title>
<body style="margin:0;background:#eee;font-family:system-ui">
<div style="padding:8px 12px;background:#222;color:#fff">
  {book} / {stem}: {len(nodes)} nodes, {len(roots)} root(s),
  {len(imaginary)} green bridge(s), {len(nicks)} cyan nick(s)
</div>
<div style="padding:6px 12px;background:#444;color:#ddd;font-size:13px">
  ① pages (top) &amp; ② kept-after-crop (bottom) — same scale
</div>
<img src="data:image/png;base64,{b64(f'{stem}_compare.png')}" style="width:100%;display:block">
<div style="padding:6px 12px;background:#444;color:#ddd;font-size:13px">
  parse: red = detected names + edges, orange = phantom, green = bridge, cyan = nick
</div>
<img src="data:image/png;base64,{b64(f'{stem}_parse.png')}" style="width:100%;display:block">
</body>"""
    with open(os.path.join(out_dir, "index.html"), "w") as fh:
        fh.write(html)
    print(f"  wrote {out_dir}/index.html")


if __name__ == "__main__":
    main()
