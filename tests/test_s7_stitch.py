"""Stage 7 stitching: which roots are duplicates, what they match, chain folding.

Book 2 patterns the Book 1 matcher never met:

* seam-broken subtrees leave extra ROOTS inside a graph (provenance index != 0);
  they are not duplicate section roots and must never be name-matched away;
* a section root may duplicate a NON-leaf (4_5's 存学 repeats 0_3's own root);
* sections re-print a shared ancestor chain (8_10 and 67_68 both start
  克宣 -> 龙润), so the repeated chain must fold, not duplicate.
"""

from __future__ import annotations

from src.model import Node
from src.s7_stitch import find_merges, stitch_nodes


def _node(id: int, name: str, prov: str, father: int = -1, children: list[int] | None = None) -> Node:
    return Node(id=id, name=name, name_images=[], generation=-1, father=father,
                children=children or [], biography="", notes=prov)


def _forest_with_orphan() -> list[Node]:
    # graph 0_0: 甲乙 -> 丙丁 (leaf).  graph 1_1: root 丙丁 -> 戊己; plus an
    # orphan root 庚辛 (seam-broken, index 5) whose name collides with a 0_0 leaf.
    return [
        _node(1, "甲乙", "0_0_0", children=[2, 3]),
        _node(2, "丙丁", "0_0_1", father=1),
        _node(3, "庚辛", "0_0_2", father=1),
        _node(4, "丙丁", "1_1_0", children=[5]),
        _node(5, "戊己", "1_1_1", father=4),
        _node(6, "庚辛", "1_1_5", children=[7]),
        _node(7, "壬癸", "1_1_6", father=6),
    ]


def test_only_section_roots_are_duplicates() -> None:
    merges = find_merges(_forest_with_orphan(), "testbook")
    assert merges == [("1_1_0", "0_0_1")]


def test_orphan_root_survives_stitch_as_its_own_root() -> None:
    nodes = _forest_with_orphan()
    out = stitch_nodes(nodes, find_merges(nodes, "testbook"))
    roots = sorted(n.name for n in out if n.father == -1)
    assert roots == ["庚辛", "甲乙"]
    by_name = {n.name: n for n in out}
    assert by_name["戊己"].father == by_name["丙丁"].id


def test_non_leaf_canonical_when_no_leaf_has_the_name() -> None:
    # 0_3: 存学 -> 九舟 -> 如先.   4_5: 存学 -> 九舟 -> 如子.
    nodes = [
        _node(1, "存学", "0_3_0", children=[2]),
        _node(2, "九舟", "0_3_1", father=1, children=[3]),
        _node(3, "如先", "0_3_2", father=2),
        _node(4, "存学", "4_5_0", children=[5]),
        _node(5, "九舟", "4_5_1", father=4, children=[6]),
        _node(6, "如子", "4_5_2", father=5),
    ]
    assert find_merges(nodes, "testbook") == [("4_5_0", "0_3_0")]


def test_repeated_chain_folds_into_one_lineage() -> None:
    nodes = [
        _node(1, "存学", "0_3_0", children=[2]),
        _node(2, "九舟", "0_3_1", father=1, children=[3]),
        _node(3, "如先", "0_3_2", father=2),
        _node(4, "存学", "4_5_0", children=[5]),
        _node(5, "九舟", "4_5_1", father=4, children=[6]),
        _node(6, "如子", "4_5_2", father=5),
    ]
    out = stitch_nodes(nodes, find_merges(nodes, "testbook"))
    names = sorted(n.name for n in out)
    assert names == ["九舟", "如先", "如子", "存学"]          # one 存学, one 九舟
    by_name = {n.name: n for n in out}
    assert by_name["如子"].father == by_name["九舟"].id
    assert by_name["九舟"].father == by_name["存学"].id
    assert [n for n in out if n.father == -1] == [by_name["存学"]]
    assert by_name["如子"].generation == 3


def test_ambiguous_same_name_children_are_not_folded() -> None:
    # canon 万都 already has a child 宏美; the duplicate brings TWO children both
    # OCR'd 宏美 -- folding would weld a stranger's subtree, so adopt them as-is.
    nodes = [
        _node(1, "万都", "0_0_0", children=[2]),
        _node(2, "宏美", "0_0_1", father=1),
        _node(3, "万都", "1_1_0", children=[4, 5]),
        _node(4, "宏美", "1_1_1", father=3),
        _node(5, "宏美", "1_1_2", father=3),
    ]
    out = stitch_nodes(nodes, find_merges(nodes, "testbook"))
    assert sum(1 for n in out if n.name == "宏美") == 3


def test_single_char_names_never_fold() -> None:
    # Book 1 names are one character and its section roots merge into LEAVES,
    # so folding never applies there; a lone char in a 2-char book is an OCR
    # truncation, not an identity.
    nodes = [
        _node(1, "点", "0_0_0", children=[2]),
        _node(2, "参", "0_0_1", father=1),
        _node(3, "参", "1_1_0", children=[4, 5]),
        _node(4, "传", "1_1_1", father=3),
        _node(5, "传", "1_1_2", father=3),
    ]
    out = stitch_nodes(nodes, find_merges(nodes, "testbook"))
    assert sum(1 for n in out if n.name == "传") == 2
