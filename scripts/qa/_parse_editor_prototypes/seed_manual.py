"""Seed a complete per-subgraph manual override from the (riser-fill'd) parse.

Dumps ONE graph's full final structure -- every node with its name box, top/bot
line-ends, parent, children, and eldest-first order -- to data/{book}_manual/{stem}.json.
That file is self-contained ground truth: Stage 5 can render/emit the graph from it
without re-parsing. Seeded from the best available parse; then hand-corrected.
"""
import sys, os, json
import numpy as np
import cv2

sys.path.insert(0, os.getcwd())
from src.imaging import get_image
import src.s5_build_tree as bt

RISER_MIN, RISER_GAP, RISER_DIL = 60, 8, 5


def _riser_pixels(fg, m):
    fgT = fg.T
    outT = np.zeros_like(fgT)
    for c0 in range(0, fgT.shape[0], bt._GAP_CHUNK):
        outT[c0:c0 + bt._GAP_CHUNK] = bt._run_lengths(fgT[c0:c0 + bt._GAP_CHUNK]) >= m
    return outT.T


def _patched_find_lines(image, threshold=70):
    if image.size == 0:
        return []
    fg = image == 0
    fg = bt._fill_short_gaps(fg, bt.GAP_FILL_H, axis=1, source=bt._bar_pixels(fg))
    rp = _riser_pixels(fg, RISER_MIN); barpx = bt._bar_pixels(fg)
    rp = cv2.dilate(rp.astype(np.uint8), np.ones((1, RISER_DIL), np.uint8)).astype(bool)
    fg = bt._fill_short_gaps(fg, RISER_GAP, axis=0, source=(rp | barpx), anchor=rp)
    nl, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=4)
    res = []
    for l in range(1, nl):
        w = stats[l, cv2.CC_STAT_WIDTH]; h = stats[l, cv2.CC_STAT_HEIGHT]
        if w > threshold + 1 or h > threshold + 1:
            x = stats[l, cv2.CC_STAT_LEFT]; y = stats[l, cv2.CC_STAT_TOP]
            sr, sc = np.where(lab[y:y + h, x:x + w] == l)
            res.append(set(zip((sr + y).tolist(), (sc + x).tolist())))
    return res


def main():
    book, stem = sys.argv[1], sys.argv[2]
    bt.find_lines = _patched_find_lines
    cfg = bt.config_for(book)
    a = get_image(f"books/{book}/4_graphs/{stem}.png")
    a2, imaginary, nicks = bt.bridge_orphans(a, cfg)
    nodes = bt.parse_graph(a2, cfg, graph_stem=stem)
    # local ids by eldest-first order (index within graph)
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
            "empty": bt._is_empty_name(n, a2),
            "children": [id_of[id(c)] for c in n.children if id(c) in id_of],
        })
    # father map
    father = {}
    for r in recs:
        for c in r["children"]:
            father[c] = r["id"]
    for r in recs:
        r["father"] = father.get(r["id"], -1)

    out = {
        "stem": stem,
        "nodes": recs,
        "imaginary": [list(map(int, b)) for b in imaginary],
        "nicks": [list(map(int, b)) for b in nicks],
    }
    out_dir = f"data/{book}_manual"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"seeded {path}: {len(recs)} nodes, "
          f"{sum(1 for r in recs if r['father']==-1)} root(s), "
          f"{len(imaginary)} bridge(s)")


if __name__ == "__main__":
    main()
