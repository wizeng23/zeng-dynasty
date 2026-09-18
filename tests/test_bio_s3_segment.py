"""Bio stage 3 segmentation: band split, header-label detection, block ordering.

The header caption is a ~385-400px-wide horizontal ink stripe near a band's top. These
tests build synthetic bands with such stripes (plus narrow prose-width ink columns as
distractors) and check that the detector counts one label per stripe, that bands split
at the rules, and that blocks come out eldest-first (right-to-left).
"""

from __future__ import annotations

import numpy as np

from src.bio import s3_segment as seg


def _band(width: int, height: int = 1100) -> np.ndarray:
    """An empty band ink grid (ink=1, all background=0)."""
    return np.zeros((height, width), dtype=np.uint8)


def _add_label(band: np.ndarray, x: int, y0: int = 95, w: int = 390) -> None:
    """Draw a header-width horizontal ink stripe (with gaps, like real chars)."""
    # three char blocks with whitespace between -> exercises the merge-gap bridging
    for cx in (x, x + 140, x + 280):
        band[y0:y0 + seg.LABEL_WINDOW_H - 20, cx:cx + 100] = 1
    assert w  # documented nominal width


def test_detect_bands_splits_at_rules() -> None:
    bands = seg.detect_bands([1000, 2000, 3000, 4000], 5000)
    assert bands == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
    assert len(bands) == 5  # generations 2..6


def test_find_labels_counts_one_per_stripe() -> None:
    band = _band(6000)
    for x in (500, 2500, 4500):
        _add_label(band, x)
    labels, _y0 = seg.find_labels(band)
    assert len(labels) == 3


def test_find_labels_ignores_narrow_prose_columns() -> None:
    band = _band(4000)
    _add_label(band, 300)                     # one real header (~390px)
    band[95:200, 1500:1560] = 1               # a lone prose column (60px) -- too narrow
    band[95:200, 2500:2600] = 1               # another narrow prose column
    labels, _y0 = seg.find_labels(band)
    assert len(labels) == 1


def test_find_labels_adapts_to_header_offset() -> None:
    # header sits high (y0~30), outside a naive 90px window
    band = _band(3000)
    _add_label(band, 400, y0=30)
    labels, y0 = seg.find_labels(band)
    assert len(labels) == 1
    assert y0 <= 45


def test_blocks_are_eldest_first_right_to_left() -> None:
    labels = [(500, 890), (2500, 2890), (4500, 4890)]  # left edges ascending
    blocks = seg._blocks_from_labels(labels, band_top=0, band_bottom=1100,
                                     generation=6, band_idx=4, stem="2_9", width=6000)
    xs = [b.x for b in blocks]
    assert xs == sorted(xs, reverse=True)              # rightmost (eldest) first
    assert [b.id for b in blocks] == ["2_9_6_0", "2_9_6_1", "2_9_6_2"]
    assert blocks[0].generation == 6


def test_stem_of_notes_strips_local_index_and_alt_provenance() -> None:
    assert seg._stem_of_notes("0_1_7 | ocr_conf=0.9") == "0_1"
    assert seg._stem_of_notes("4_4_0/252_252_0 | x") == "4_4"
    assert seg._stem_of_notes("") == ""


def test_map_sections_to_stems_is_positional() -> None:
    sections = ["2_9", "18_63", "66_77"]
    stems = ["0_1", "10_17", "64_65"]
    mapping = seg.map_sections_to_stems(sections, stems)
    assert mapping == {"2_9": "0_1", "18_63": "10_17", "66_77": "64_65"}
