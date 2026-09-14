"""Endpoint reading for one line component (Stage 5, ``find_line_ends``).

Synthetic components drawn at v1-native scale: lines are ~7px wide, bars are
hundreds of px long, risers a few hundred rows tall.
"""

from __future__ import annotations

import numpy as np

from src.s5_build_tree import find_line_ends

LINE = 7


def _points(mask: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.where(mask)
    return set(zip(rows.tolist(), cols.tolist()))


def _vline(mask: np.ndarray, r0: int, r1: int, c: int) -> None:
    mask[r0:r1 + 1, c:c + LINE] = True


def _hline(mask: np.ndarray, r: int, c0: int, c1: int) -> None:
    mask[r:r + LINE, c0:c1 + LINE] = True


def _near(p: tuple[int, int], row: int, col: int, tol: int = 12) -> bool:
    return abs(p[0] - row) <= tol and abs(p[1] - col) <= tol


def test_plain_fan_out_one_parent_three_children() -> None:
    """A hang-line down to a bar with three risers: the common case."""
    m = np.zeros((900, 1500), dtype=bool)
    _vline(m, 100, 400, 1200)            # parent hang-line
    _hline(m, 400, 200, 1200)            # sibling bar
    for c in (200, 700, 1200):
        _vline(m, 400, 800, c)           # child risers

    tops, bots = find_line_ends(_points(m), threshold=150)

    assert len(tops) == 1 and _near(tops[0], 100, 1200)
    assert len(bots) == 3
    assert all(_near(b, 800 + LINE - 1, c) for b, c in zip(bots, (200, 700, 1200)))


def test_flush_bar_collapses_to_one_parent() -> None:
    """A bar flush with the component top (no hang-line above) is one parent."""
    m = np.zeros((600, 1500), dtype=bool)
    _hline(m, 0, 200, 1200)
    for c in (200, 700, 1200):
        _vline(m, 0, 500, c)

    tops, bots = find_line_ends(_points(m), threshold=150)

    assert len(tops) == 1
    assert len(bots) == 3


def test_stepped_bar_parent_is_the_hang_line_not_the_bar_corner() -> None:
    """Book 2 graph 67_68: 宏羨's hang-line steps LEFT and UP into a raised bar.

    The parent connection (top of the hang-line) sits BELOW the raised bar's
    top edge, and two children hang from the raised bar at a higher row than the
    third child on the through-line. Geometry mirrors the real component
    (rows 3894-4559, cols 381-2264, re-based to 0).
    """
    m = np.zeros((700, 1900), dtype=bool)
    _vline(m, 103, 665, 1876)            # 宏羨 hang-line, through to child 闻诏
    _hline(m, 338, 1237, 1876)           # step bar, left from the hang-line
    _vline(m, 3, 338, 1237)              # step riser up to the raised bar
    _hline(m, 3, 0, 1237)                # raised bar
    _vline(m, 3, 333, 0)                 # riser to 闻评
    _vline(m, 3, 333, 918)               # riser to 闻瑛

    tops, bots = find_line_ends(_points(m), threshold=150)

    assert len(tops) == 1, tops
    assert _near(tops[0], 103, 1876), tops
    assert len(bots) == 3, bots
    assert any(_near(b, 333 + LINE - 1, 0) for b in bots), bots       # 闻评
    assert any(_near(b, 333 + LINE - 1, 918) for b in bots), bots     # 闻瑛
    assert any(_near(b, 665 + LINE - 1, 1876) for b in bots), bots    # 闻诏
