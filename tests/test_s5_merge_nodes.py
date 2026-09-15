"""merge_nodes: fusing a name's upper and lower hang-line pieces into one node.

114_120 毓棋: the piece above the name (col 7119) and the piece below it
(col 7058) are 61px apart -- a misprint -- and merge_max_shift was 60, so the
lower piece became a phantom node holding 毓棋's child. Book 2's names are
~300px apart, so a wider tolerance is safe as long as the nearest candidate
wins.
"""

from __future__ import annotations

from src import s5_build_tree as bt


def test_pieces_offset_by_a_misprint_still_fuse() -> None:
    upper = bt.LineNode(top=(1000, 7119))          # child stub: hangs from the bar above
    lower = bt.LineNode(bot=(1300, 7058))          # parent stub: the line below the name
    lower.children = [bt.LineNode(top=(1800, 7058))]
    nodes = bt.merge_nodes([upper, lower], bt.config_for("book2"))
    assert len(nodes) == 1
    assert nodes[0].top == (1000, 7119) and nodes[0].bot == (1300, 7058)
    assert len(nodes[0].children) == 1


def test_nearest_lower_piece_wins_when_two_are_in_range() -> None:
    upper = bt.LineNode(top=(1000, 7119))
    near = bt.LineNode(bot=(1300, 7100))           # 19px off
    far = bt.LineNode(bot=(1300, 7040))            # 79px off
    nodes = bt.merge_nodes([upper, far, near], bt.config_for("book2"))   # far listed first
    assert len(nodes) == 2
    fused = next(n for n in nodes if n.top is not None)
    assert fused.bot == (1300, 7100)


def test_default_tolerance_is_at_least_90px() -> None:
    assert bt.BookConfig().merge_max_shift >= 90
