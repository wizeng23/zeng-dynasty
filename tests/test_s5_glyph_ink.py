"""Glyph-aware ink (Stage 5): a bare line is not part of a name.

Several Stage 5 reads treat ANY ink as name ink. A hang-line running below a
leaf's name (after a phantom took its children), a riser stub, or a line
fragment inside a crop then counts as glyph ink and produces: an inferred leaf
bottom 600 rows down (114_120 毓援/毓棋), a name box that runs down the line, and
a nameless line-fragment node that survives the blank-leaf filter (8_10_9).

Rule: rows whose ink is a single narrow run (<= LINE_RUN_MAX) that persists for
>= LINE_MIN_ROWS consecutive rows are LINE rows -- blank for every name read. A
short narrow stroke (尚's top tick, 点's dot) is shorter than LINE_MIN_ROWS and
stays glyph.
"""

from __future__ import annotations

import numpy as np

from src import s5_build_tree as bt

LINE = 7


def _graph() -> np.ndarray:
    return np.ones((1600, 800), dtype=np.uint8)


def test_line_rows_are_long_narrow_runs_only() -> None:
    band = np.ones((400, 240), dtype=np.uint8)
    band[0:25, 116:125] = 0        # a 25-row tick: narrow but short -> glyph
    band[40:200, 60:180] = 0       # glyph body
    band[200:400, 117:124] = 0     # 200-row hang-line -> line rows
    line = bt._line_only_rows(band, 120)
    assert not line[0:25].any()
    assert not line[40:200].any()
    assert line[200:400].all()


def test_infer_ends_leaf_bottom_stops_at_the_name_not_the_line() -> None:
    a = _graph()
    col = 400
    a[300:600, col - 80:col + 80] = 0            # name
    a[600:1100, col:col + LINE] = 0              # hang-line continuing 500 rows below it
    node = bt.LineNode(top=(280, col))            # leaf: no bot
    bt.infer_ends([node], a)
    assert node.bot is not None
    assert 600 <= node.bot[0] <= 600 + bt.INFER_END_PAD + 2, node.bot


def test_tight_box_bottom_ignores_the_line_below() -> None:
    a = _graph()
    col = 400
    a[300:600, col - 80:col + 80] = 0
    a[600:1100, col:col + LINE] = 0
    node = bt.LineNode(top=(280, col), bot=(1120, col))   # bot already ran down the line
    _l, _t, _r, bottom = bt._tight_ink_box(node, a)
    assert bottom <= 600 + bt.NAME_TRIM_PAD + 8, bottom


def test_tight_box_keeps_a_short_top_tick() -> None:
    a = _graph()
    col = 400
    a[300:325, col - 4:col + 5] = 0              # 尚-style tick, 25 rows x 9 px
    a[340:600, col - 80:col + 80] = 0
    node = bt.LineNode(top=(280, col), bot=(620, col))
    _l, top, _r, _b = bt._tight_ink_box(node, a)
    assert top <= 300, top


def test_line_fragment_node_is_empty() -> None:
    a = _graph()
    col = 400
    a[300:700, col:col + LINE] = 0               # nothing but a 400-row line
    node = bt.LineNode(top=(290, col), bot=(710, col))
    assert bt._is_empty_name(node, a)


def test_parallel_glyph_verticals_are_not_a_line() -> None:
    """门-like: two verticals 60px either side of the node column for 150 rows."""
    band = np.ones((300, 240), dtype=np.uint8)
    band[0:150, 55:64] = 0
    band[0:150, 176:185] = 0
    assert not bt._line_only_rows(band, 120).any()


def test_second_line_in_window_does_not_hide_the_nodes_own_line() -> None:
    """A phantom's hang-line 70px away runs alongside; the node's own line below the
    name (rows 200..400) must still read as line."""
    band = np.ones((400, 240), dtype=np.uint8)
    band[40:200, 60:180] = 0        # glyph
    band[200:400, 117:124] = 0      # own line
    band[0:230, 47:54] = 0          # phantom's line, 70px left, ending at row 230
    line = bt._line_only_rows(band, 120)
    assert line[240:400].all()
    assert not line[40:200].any()


def test_misaligned_line_below_the_name_reaching_the_band_edge_is_line() -> None:
    """114_120 毓援: the hang-line under the name sits 70px LEFT of the node column
    and runs to the bottom of the band."""
    band = np.ones((600, 240), dtype=np.uint8)
    band[40:340, 60:180] = 0          # name
    band[340:600, 47:54] = 0          # line below, off-column, to the band bottom
    line = bt._line_only_rows(band, 120)
    assert line[350:600].all()
    assert not line[40:340].any()


def test_glyph_vertical_ending_inside_the_band_is_not_line() -> None:
    """丁-like: a lone 150-row vertical on the node column that stops well inside."""
    band = np.ones((600, 240), dtype=np.uint8)
    band[100:120, 40:200] = 0         # top bar of 丁
    band[120:270, 117:124] = 0        # its vertical, 150 rows, ends at 270 (band is 600)
    line = bt._line_only_rows(band, 120)
    assert not line.any()            # ends in whitespace: a glyph stroke, not a line
