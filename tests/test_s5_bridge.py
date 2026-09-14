"""Orphan bridging (Stage 5): repair generation bars broken at a page seam.

Synthetic two-page graphs at v1 scale. Ink is 0, background 1. Names are solid
blobs (far more ink than ``BLANK_NAME_MAX_INK``) so a node with a name is never
mistaken for an orphan; a bar with children and no name above it IS an orphan.
"""

from __future__ import annotations

import numpy as np

from src import s5_build_tree as bt

LINE = 7
SEAM = 1200          # left page = cols 0..1199, right page = cols 1200..2399


def _blank(h: int = 1400, w: int = 2400) -> np.ndarray:
    return np.ones((h, w), dtype=np.uint8)


def _vline(a: np.ndarray, r0: int, r1: int, c: int) -> None:
    a[r0:r1 + 1, c:c + LINE] = 0


def _hline(a: np.ndarray, r: int, c0: int, c1: int) -> None:
    a[r:r + LINE, c0:c1 + LINE] = 0


def _name(a: np.ndarray, r0: int, c: int) -> None:
    """A 2-char stacked name: an 80x170 blob centred on column ``c``."""
    a[r0:r0 + 170, c - 40:c + 40] = 0


def _right_page_tree(a: np.ndarray) -> None:
    """Root (named) -> bar at row 500 -> two named children, all on the right page."""
    _name(a, 10, 2003)                       # root's name
    _vline(a, 200, 500, 2000)                # root's line down to the bar
    _hline(a, 500, 1400, 2000)               # right-page piece of the bar
    for c in (1500, 2000):
        _vline(a, 500, 800, c)               # child risers
        _name(a, 830, c + 3)                 # child names


def _left_page_floating_bar(a: np.ndarray, row: int = 505, end: int = 1000) -> None:
    """The same bar's LEFT-page piece: the connector cols end..1400 was lost."""
    _hline(a, row, 300, end)
    for c in (400, 900):
        _vline(a, row, 805, c)
        _name(a, 835, c + 3)


def _orphans(a: np.ndarray) -> list[bt.LineNode]:
    return bt.find_orphans(bt.parse_graph(a, bt.BookConfig()), a)


def test_seam_broken_bar_parses_as_one_orphan() -> None:
    a = _blank()
    _right_page_tree(a)
    _left_page_floating_bar(a)
    orphans = _orphans(a)
    assert len(orphans) == 1
    assert len(orphans[0].children) == 2


def test_bar_true_end_hops_a_nick_but_stops_at_the_page_gap() -> None:
    a = _blank()
    _hline(a, 500, 300, 600)
    _hline(a, 500, 650, 1000)                # 50px nick, well under BAR_TRACE_HOP
    _hline(a, 500, 1400, 2000)               # the far bar, past a real gap
    end = bt.bar_true_end(a, 503, 300)
    assert 1000 <= end <= 1000 + LINE


def test_first_ink_right_lands_on_the_next_bar() -> None:
    a = _blank()
    _hline(a, 500, 300, 1000)
    _hline(a, 495, 1400, 2000)               # drifted 5 rows: within tolerance
    assert bt.first_ink_right(a, 503, 1000 + LINE - 1) == 1400
    b = _blank()
    _hline(b, 500, 300, 1000)
    assert bt.first_ink_right(b, 503, 1000 + LINE - 1) is None


def test_no_candidate_when_reconnection_is_the_graph_edge() -> None:
    a = _blank(w=1500)
    _hline(a, 500, 300, 1000)
    _hline(a, 500, 1450, 1493)               # ink at the very right edge = cross-graph
    assert bt.bridge_candidate(a, 503, 300) is None


def test_no_candidate_for_a_sub_hop_gap() -> None:
    a = _blank()
    _hline(a, 500, 300, 1000)
    _hline(a, 500, 1050, 2000)               # 50px gap: a scan nick, not a page break
    assert bt.bridge_candidate(a, 503, 300) is None


def test_bridge_candidate_spans_seam_gap_to_next_bar() -> None:
    a = _blank()
    _right_page_tree(a)
    _left_page_floating_bar(a)
    cand = bt.bridge_candidate(a, 508, 300)
    assert cand is not None
    row, c0, c1 = cand
    assert 1000 <= c0 <= 1000 + LINE
    assert c1 == 1400


def test_draw_bridge_touches_both_drifted_bars() -> None:
    a = _blank()
    _hline(a, 505, 300, 1000)
    _hline(a, 500, 1400, 2000)
    assert len(bt.find_lines(a, threshold=200)) == 2
    out = bt.draw_bridge(a, 508, 1000 + LINE - 1, 1400)
    assert len(bt.find_lines(out, threshold=200)) == 1     # one traversable bar
    assert a[508, 1200] == 1                                # input untouched


def test_bridge_orphans_reconnects_and_records_the_bridge() -> None:
    a = _blank()
    _right_page_tree(a)
    _left_page_floating_bar(a)
    out, imaginary = bt.bridge_orphans(a, bt.BookConfig())
    assert _orphans(out) == []
    assert len(imaginary) == 1
    r0, c0, r1, c1 = imaginary[0]
    assert r0 == r1 and 1000 <= c0 <= 1000 + LINE and c1 == 1400
    nodes = bt.parse_graph(out, bt.BookConfig())
    root = [n for n in nodes if n.top is not None and n.top[0] < 300]
    assert len(root) == 1 and len(root[0].children) == 4
    assert a[508, 1200] == 1                                # input untouched


def test_bridge_orphans_leaves_cross_graph_orphan_alone() -> None:
    a = _blank(w=1300)
    _left_page_floating_bar(a)                # nothing to the right: parent is off-graph
    out, imaginary = bt.bridge_orphans(a, bt.BookConfig())
    assert imaginary == []
    assert len(_orphans(out)) == 1


def test_bridge_orphans_extends_across_two_page_breaks() -> None:
    a = _blank(w=3600)
    # root + bar on page 3 (cols 2400..)
    _name(a, 10, 3203)
    _vline(a, 200, 500, 3200)
    _hline(a, 500, 2600, 3200)
    _vline(a, 500, 800, 3200)
    _name(a, 830, 3203)
    # page 2 carries a short bare bar fragment (too short to parse as a line):
    # landing on it reconnects nothing, so the bridge must extend past it
    _hline(a, 502, 1500, 1650)
    # page 1: the floating bar with two children
    _left_page_floating_bar(a)
    out, imaginary = bt.bridge_orphans(a, bt.BookConfig())
    assert _orphans(out) == []
    assert len(imaginary) == 1
    assert imaginary[0][3] == 2600                          # extended to the far bar


# --- ADF smear specks (v1 scans) must not count as bar ink -------------------


def _speck(a: np.ndarray, r: int, c: int, rows: int = 2, cols: int = 3) -> None:
    a[r:r + rows, c:c + cols] = 0


def test_bar_true_end_does_not_hop_onto_a_speck() -> None:
    a = _blank()
    _hline(a, 500, 300, 1000)
    _speck(a, 470, 1050)                      # 2-row speck 50px past the bar end
    _hline(a, 500, 1400, 2000)
    end = bt.bar_true_end(a, 503, 300)
    assert 1000 <= end <= 1000 + LINE


def test_first_ink_right_skips_specks() -> None:
    a = _blank()
    _hline(a, 500, 300, 1000)
    _speck(a, 460, 1100)
    _speck(a, 540, 1250, rows=3)
    _hline(a, 500, 1400, 2000)
    assert bt.first_ink_right(a, 503, 1000 + LINE - 1) == 1400


def test_draw_bridge_ignores_a_speck_when_finding_bar_ink() -> None:
    a = _blank()
    _hline(a, 505, 300, 1000)
    _speck(a, 460, 1040)                      # speck just left of the anchor, 48 rows up
    _hline(a, 500, 1400, 2000)
    out = bt.draw_bridge(a, 508, 1050, 1400)  # anchored past the bar end, beside the speck
    filled = np.where((out[:, 1200] == 0))[0]
    assert filled.min() >= 495 and filled.max() <= 515   # thin fill, not up to the speck


# --- acceptance gate: the TARGETED orphan must be resolved, no new orphans -------


def _ln(bot: tuple[int, int], kids: int) -> bt.LineNode:
    n = bt.LineNode(top=(bot[0] - 30, bot[1]), bot=bot)
    n.children = [bt.LineNode(top=(bot[0] + 300, bot[1] + 100 * i)) for i in range(kids)]
    return n


def test_gate_rejects_bridge_that_removes_a_different_orphan() -> None:
    target = _ln((1121, 82), 1)
    other = _ln((1146, 493), 3)
    before = [target, other]
    after = [_ln((1121, 82), 1)]            # count dropped, but the target survived
    assert not bt.bridge_resolves(target, before, after)


def test_gate_rejects_bridge_that_creates_a_new_orphan() -> None:
    target = _ln((1121, 82), 1)
    before = [target, _ln((2079, 7827), 2)]
    after = [_ln((2079, 7827), 2), _ln((3300, 5000), 1)]
    assert not bt.bridge_resolves(target, before, after)


def test_gate_accepts_when_target_gone_and_nothing_new() -> None:
    target = _ln((1121, 82), 1)
    before = [target, _ln((2079, 7827), 2)]
    after = [_ln((2079, 7827), 2)]
    assert bt.bridge_resolves(target, before, after)


# --- a bar that steps vertically at a seam --------------------------------------


def test_trace_follows_the_bar_across_a_vertical_step() -> None:
    a = _blank(w=3000)
    _hline(a, 500, 300, 1000)                 # orphan's piece
    _hline(a, 540, 1100, 2500)                # continues 40 rows lower after a 100px gap
    end, steps = bt.trace_bar(a, 503, 300)
    assert 2500 <= end <= 2500 + LINE
    assert len(steps) == 1
    (x0, x1, y0, y1), = steps
    assert 1000 <= x0 <= 1000 + LINE and x1 == 1100 and abs(y0 - 503) <= LINE and abs(y1 - 543) <= LINE


def test_bridge_orphans_fills_a_seam_step_instead_of_reaching_far() -> None:
    """Book 2 69_82: 尚四's bar piece sits ~35 rows above the page-81 bar across a
    ~100px seam gap. The repair is the short step fill at the seam, not a long
    bridge along the lower bar."""
    a = _blank(w=3000)
    # parent tree on the right: named root, hang-line, bar at row 540 with a child
    _name(a, 10, 2803)
    _vline(a, 200, 540, 2800)
    _hline(a, 540, 1100, 2800)
    _vline(a, 540, 840, 2800)
    _name(a, 870, 2803)
    # orphan piece at row 500: bar 300..1000 with one child
    _hline(a, 500, 300, 1000)
    _vline(a, 500, 800, 400)
    _name(a, 830, 403)
    out, imaginary = bt.bridge_orphans(a, bt.BookConfig())
    assert _orphans(out) == []
    assert len(imaginary) == 1
    r0, c0, r1, c1 = imaginary[0]
    assert 1000 <= c0 <= 1000 + LINE and c1 == 1100      # fills the seam step only


def test_trace_started_at_the_bar_top_edge_records_no_step() -> None:
    a = _blank()
    _hline(a, 500, 300, 1000)                 # bar rows 500..506; parser reports row 500
    end, steps = bt.trace_bar(a, 500, 300)
    assert steps == []
    assert 1000 <= end <= 1000 + LINE
