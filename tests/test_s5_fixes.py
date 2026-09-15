"""Stage 5 post-fixes: hand-verified corrections applied on top of the parse.

A fixes file (``data/{book}_fixes.json``) lists nodes by PROVENANCE (stable
across re-runs) and is re-applied after every Stage 5 run, like the OCR override
layer. Three operations:

* ``delete``  -- drop a phantom node that has no real children;
* ``merge``   -- fold a phantom into the real node for the same person: the real
  node adopts the phantom's children (kept eldest-first, right-to-left) and the
  phantom is dropped;
* ``recrop``  -- replace a node's name crop with an explicit graph-pixel box.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from src.s5_fixes import apply_fixes


def _node(id: int, prov: str, father: int = -1, children: list[int] | None = None,
          name: str = "") -> dict:
    return {"id": id, "name": name, "name_images": [f"books/tb/5_names/{id}.png"],
            "generation": -1, "father": father, "children": children or [],
            "biography": "", "notes": prov}


def _sidecar_node(id: int, top: tuple[int, int], bot: tuple[int, int],
                  children_top: list[list[int]], box: list[int]) -> dict:
    return {"id": id, "box": box, "top": list(top), "bot": list(bot),
            "empty": False, "children_top": children_top}


def _setup(tmp_path):
    """One graph '1_2': root R(1) -> A(2); phantom P(3) with children B(4), C(5)."""
    books = tmp_path / "books"; data = tmp_path / "data"
    graphs = books / "tb" / "4_graphs"; names = books / "tb" / "5_names"
    graphs.mkdir(parents=True); names.mkdir(parents=True); data.mkdir()
    rows = [
        _node(1, "1_2_0", children=[2], name="根"),
        _node(2, "1_2_1", father=1, name="甲"),
        _node(3, "1_2_2", children=[4, 5]),             # phantom, no name
        _node(4, "1_2_3", father=3, name="乙"),
        _node(5, "1_2_4", father=3, name="丙"),
        _node(6, "1_2_5", name="丁"),                    # stray phantom, no children
    ]
    with open(data / "tb.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    side = {"nodes": [
        _sidecar_node(1, (100, 900), (400, 900), [[800, 900]], [850, 110, 950, 390]),
        _sidecar_node(2, (800, 900), (1100, 900), [], [850, 810, 950, 1090]),
        _sidecar_node(3, (780, 300), (810, 300), [[1200, 200], [1200, 600]], [250, 790, 350, 800]),
        _sidecar_node(4, (1200, 200), (1500, 200), [], [150, 1210, 250, 1490]),
        _sidecar_node(5, (1200, 600), (1500, 600), [], [550, 1210, 650, 1490]),
        _sidecar_node(6, (2000, 50), (2300, 50), [], [0, 2010, 100, 2290]),
    ], "scrubbed": []}
    json.dump(side, open(graphs / "1_2.parse.json", "w"))
    # graph image: white with a black block where node 2's name "really" is
    a = np.ones((2500, 1200), dtype=np.uint8) * 255
    a[790:1100, 860:940] = 0
    Image.fromarray(a).save(graphs / "1_2.png")
    for i in range(1, 7):
        Image.fromarray(np.ones((10, 10), dtype=np.uint8) * 255).save(names / f"{i}.png")
    return books, data


def _load(data, name="tb.jsonl"):
    return {r["id"]: r for r in (json.loads(l) for l in open(data / name) if l.strip())}


def test_merge_gives_phantoms_children_to_real_node_rtl(tmp_path) -> None:
    books, data = _setup(tmp_path)
    fixes = {"merge": {"1_2_2": "1_2_1"}}
    json.dump(fixes, open(data / "tb_fixes.json", "w"))
    apply_fixes("tb", books_dir=str(books), data_dir=str(data))
    rows = _load(data)
    assert 3 not in rows
    assert rows[2]["children"] == [5, 4]          # eldest first = rightmost column first
    assert rows[4]["father"] == 2 and rows[5]["father"] == 2
    assert rows[4]["generation"] == 3
    side = {n["id"]: n for n in json.load(open(books / "tb" / "4_graphs" / "1_2.parse.json"))["nodes"]}
    assert 3 not in side
    assert side[2]["children_top"] == [[1200, 600], [1200, 200]]


def test_delete_drops_a_childless_phantom(tmp_path) -> None:
    books, data = _setup(tmp_path)
    json.dump({"delete": ["1_2_5"]}, open(data / "tb_fixes.json", "w"))
    apply_fixes("tb", books_dir=str(books), data_dir=str(data))
    assert 6 not in _load(data)
    assert not os.path.exists(books / "tb" / "5_names" / "6.png")


def test_delete_refuses_a_node_with_children(tmp_path) -> None:
    books, data = _setup(tmp_path)
    json.dump({"delete": ["1_2_2"]}, open(data / "tb_fixes.json", "w"))
    try:
        apply_fixes("tb", books_dir=str(books), data_dir=str(data))
    except ValueError as e:
        assert "children" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_recrop_rewrites_crop_and_box(tmp_path) -> None:
    books, data = _setup(tmp_path)
    json.dump({"recrop": {"1_2_1": [850, 780, 950, 1100]}}, open(data / "tb_fixes.json", "w"))
    changed = apply_fixes("tb", books_dir=str(books), data_dir=str(data))
    img = Image.open(books / "tb" / "5_names" / "2.png")
    assert img.size == (100, 320)
    assert (np.array(img) == 0).sum() > 0                  # the black block is inside
    side = {n["id"]: n for n in json.load(open(books / "tb" / "4_graphs" / "1_2.parse.json"))["nodes"]}
    assert side[2]["box"] == [850, 780, 950, 1100]
    assert changed["recropped_ids"] == [2]


def test_apply_is_idempotent(tmp_path) -> None:
    books, data = _setup(tmp_path)
    json.dump({"merge": {"1_2_2": "1_2_1"}, "delete": ["1_2_5"]}, open(data / "tb_fixes.json", "w"))
    apply_fixes("tb", books_dir=str(books), data_dir=str(data))
    first = open(data / "tb.jsonl").read()
    apply_fixes("tb", books_dir=str(books), data_dir=str(data))   # already applied: no error
    assert open(data / "tb.jsonl").read() == first
