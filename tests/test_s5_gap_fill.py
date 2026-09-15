"""find_lines: hairline gaps must not split a printed line into two components.

v0 filled short gaps before connected-components (horizontal <= 9px, vertical
<= 3px at v0 scale); v1 never had it. 133_133: a 3px nick at 尚星's riser cut the
bar from 贞杰's riser and orphaned four sons. Only bar-piece ink may bound a fill,
so grain and handwriting are never linked onto a line.
"""

from __future__ import annotations

import numpy as np

from src import s5_build_tree as bt

LINE = 7


def _graph() -> np.ndarray:
    return np.ones((1200, 1200), dtype=np.uint8)


def test_vertical_pinch_is_not_filled_at_pixel_level() -> None:
    """A fully broken hang-line stays two components here: every vertical fill
    tried at v1 resolution either missed the real pinch or welded a glyph's top
    tick to the riser above it. The pinch is repaired structurally in merge_nodes
    (continuation-stub rule, tests/test_s5_merge_nodes.py)."""
    a = _graph()
    a[100:500, 600:600 + LINE] = 0
    a[506:900, 600:600 + LINE] = 0             # 6-row complete break
    assert len(bt.find_lines(a, threshold=200)) == 2


def test_horizontal_nick_in_a_bar_is_one_component() -> None:
    a = _graph()
    a[600:600 + LINE, 100:500] = 0
    a[600:600 + LINE, 520:900] = 0             # 20px nick
    assert len(bt.find_lines(a, threshold=200)) == 1


def test_wide_gaps_stay_separate() -> None:
    a = _graph()
    a[600:600 + LINE, 100:500] = 0
    a[600:600 + LINE, 560:900] = 0             # 60px: a real break, not a nick
    a[100:400, 1000:1000 + LINE] = 0
    a[430:800, 1000:1000 + LINE] = 0           # 30 rows: not a hairline
    assert len(bt.find_lines(a, threshold=200)) == 4


def test_name_glyph_does_not_become_a_line() -> None:
    """Horizontal strokes of a 2-char name 20px apart stay separate blobs, each far
    below the line threshold, even after the vertical fill."""
    a = _graph()
    for k in range(8):
        a[300 + k * 40:300 + k * 40 + 12, 500:660] = 0    # 8 strokes, 28px gaps, 160 wide
    assert bt.find_lines(a, threshold=200) == []
