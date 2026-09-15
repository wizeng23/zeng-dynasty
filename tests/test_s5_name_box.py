"""Name crop box (Stage 5, ``_tight_ink_box``): an edge must never bisect ink.

The box is found by trimming a +-NAME_HALF_WIDTH window around the node's line
column to the ink inside it. A glyph that reaches the window edge (a wide 毓
centred 50px off its line, 114_120's 毓塘) or a top endpoint that landed inside
the glyph (69_82's 尚澜) leaves ink ON the box edge. William's rule: walk outward
from such an edge until true whitespace, then pad -- capped, so a smear streak or
a neighbour's line cannot drag the box across the page.
"""

from __future__ import annotations

import numpy as np

from src import s5_build_tree as bt

LINE = 7


def _graph() -> np.ndarray:
    return np.ones((1200, 1200), dtype=np.uint8)


def _node(top: tuple[int, int], bot: tuple[int, int]) -> bt.LineNode:
    return bt.LineNode(top=top, bot=bot)


def test_box_follows_a_glyph_past_the_left_window_edge() -> None:
    a = _graph()
    col = 600
    a[300:700, col - 170:col + 60] = 0            # glyph 230 wide, centred 55px LEFT of the line
    node = _node((250, col), (750, col))
    left, top, right, bottom = bt._tight_ink_box(node, a)
    assert left <= col - 170 - bt.NAME_TRIM_PAD + 2, left     # whole glyph + pad, not the window edge
    assert right >= col + 60 + bt.NAME_TRIM_PAD - 2, right


def test_box_walks_up_when_the_top_endpoint_is_inside_the_glyph() -> None:
    a = _graph()
    col = 600
    a[400:700, col - 70:col + 70] = 0            # glyph rows 400..699
    node = _node((430, col), (750, col))          # endpoint reported 30 rows INTO the glyph
    left, top, right, bottom = bt._tight_ink_box(node, a)
    assert top <= 400 - bt.NAME_TRIM_PAD + 2, top


def test_walk_is_capped_so_a_far_blob_is_not_swallowed() -> None:
    a = _graph()
    col = 600
    a[300:700, col - 130:col + 60] = 0            # glyph just past the window edge (10px)
    a[300:700, col - 130 - 300:col - 130 - 100] = 0   # unrelated blob 100px further left, touching nothing
    node = _node((250, col), (750, col))
    left, *_ = bt._tight_ink_box(node, a)
    assert col - 130 - bt.NAME_TRIM_PAD - 2 <= left <= col - 130 - bt.NAME_TRIM_PAD + 2, left


def test_edge_in_whitespace_is_unchanged() -> None:
    a = _graph()
    col = 600
    a[300:700, col - 70:col + 70] = 0
    node = _node((250, col), (750, col))
    left, top, right, bottom = bt._tight_ink_box(node, a)
    assert left == col - 70 - bt.NAME_TRIM_PAD and right == col + 70 + bt.NAME_TRIM_PAD
    # the row trim is smoothed (NAME_SPECK_WIN), so top/bottom sit a few rows outside
    assert 300 - bt.NAME_TRIM_PAD - 8 <= top <= 300 - bt.NAME_TRIM_PAD
    assert 700 + bt.NAME_TRIM_PAD <= bottom <= 700 + bt.NAME_TRIM_PAD + 8


def test_specks_touching_the_edge_do_not_trigger_a_walk() -> None:
    a = _graph()
    col = 600
    a[300:700, col - 70:col + 70] = 0
    rng = np.random.default_rng(0)
    # sparse ADF grain above the box (isolated 2px dots), some landing ON the top edge
    for _ in range(60):
        r = int(rng.integers(300 - 130, 300 - 28)); c = int(rng.integers(col - 60, col + 60))
        a[r:r + 2, c:c + 2] = 0
    _l, top, _r, _b = bt._tight_ink_box(bt.LineNode(top=(250, col), bot=(750, col)), a)
    assert top >= 300 - bt.NAME_TRIM_PAD - 10, top      # stays at the glyph, not the grain


def test_thin_sliver_along_the_edge_does_not_trigger_a_walk() -> None:
    """A 2px smear streak lying along the box edge is not a stroke crossing it."""
    a = _graph()
    col = 600
    a[300:700, col - 70:col + 70] = 0
    edge = col - 70 - bt.NAME_TRIM_PAD
    a[320:680, edge - 1:edge + 1] = 0                   # sliver straddling the pad edge
    left, *_ = bt._tight_ink_box(bt.LineNode(top=(250, col), bot=(750, col)), a)
    assert left >= edge - 1 - bt.NAME_TRIM_PAD, left     # at most the trim's own pad past the sliver
