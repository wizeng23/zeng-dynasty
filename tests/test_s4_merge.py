"""Stage 4 seam alignment must ignore ADF smear specks at the page edge.

Book 2 graph 11_17, seam p16|p15: p15's left edge carried two 1-2px smear specks
that were read as line ends, paired with real lines exiting p16, and the median
pair offset shifted the whole page 388 rows off its grid.
"""

from __future__ import annotations

import numpy as np

from src import s4_merge_pages as mp

LINE = 7


def _page(h: int = 1000, w: int = 600) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _ink_rows(a: np.ndarray, col: int) -> list[int]:
    return [int(r) for r in np.where(a[:, col] == 0)[0]]


def test_seam_endpoints_ignore_specks() -> None:
    edge = np.ones((1000, mp._s(5)), dtype=np.uint8)
    edge[100:100 + LINE, :] = 0          # a real line reaching the edge
    edge[700:702, 0:1] = 0               # a 2-row, 1-col smear speck
    edge[850:851, :] = 0                 # a 1-row hairline across the band
    assert mp.seam_endpoints(edge) == [100]


def test_merge_aligns_by_the_real_line_not_the_speck() -> None:
    g1 = _page()
    g1[100:100 + LINE, 300:] = 0         # line exits g1's right edge at row 100
    g1[400:400 + LINE, 300:] = 0         # a second line exits with no partner
    g2 = _page()
    g2[110:110 + LINE, :300] = 0         # the first line enters g2 at row 110
    g2[700:702, 0:2] = 0                 # smear speck on g2's left edge

    merged = mp.merge_graphs(g1, g2)

    seam = g1.shape[1]
    left = [r for r in _ink_rows(merged, seam - 20) if r < 300]
    right = [r for r in _ink_rows(merged, seam + 20) if r < 300]
    assert left == right                 # the line is continuous across the seam
