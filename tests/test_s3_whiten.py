"""Stage 3 bottom whitening anchored to the page's inner border line.

`trim_borders` cuts the page bottom at page-bottom - 90px, which lands ~34px
ABOVE the inner border line, so a whitening measured from the trimmed edge
reaches ~234px above the border. The lowest 3-char names end ~246px above the
inner border (Book 2 p96), leaving 12px. Anchoring the whitening to the detected
inner border and measuring WHITEN_BOTTOM_FROM_BORDER from there is stable across
pages whose border sits differently in the scan.
"""

from __future__ import annotations

import numpy as np

from src import s3_segment as seg



def _page() -> np.ndarray:
    """A 6000x3800 page: outer frame at the very bottom, inner border 40px above it."""
    a = np.ones((6000, 3800), dtype=np.uint8)
    a[5980:6000, :] = 0            # outer frame (bottom)
    a[5940:5946, 120:3680] = 0     # inner border line
    a[0:20, :] = 0                 # outer frame (top)
    a[60:66, 120:3680] = 0         # inner border (top)
    a[:, 0:20] = 0; a[:, 3780:] = 0        # side frames
    return a


def test_inner_border_row_is_found_in_the_untrimmed_page() -> None:
    a = _page()
    assert abs(seg.bottom_inner_border_row(a) - 5940) <= 3


def test_name_230px_above_inner_border_survives_and_page_number_50px_above_is_whitened() -> None:
    a = _page()
    a[5650:5710, 1800:1960] = 0    # 3rd char of a name: bottom 230px above the inner border
    a[5880:5895, 1900:1940] = 0    # a page number 45px above the border
    t = seg.trim_borders(a)
    w = seg.whiten_margins(t, bottom_anchor=seg.bottom_inner_border_row(a) - seg._top_border_cut(a))
    # the trimmed image is aligned to a[top_cut:]; find the name's rows there
    top_cut = seg._top_border_cut(a)
    assert (w[5650 - top_cut:5710 - top_cut, 1800 - seg._s(30):1960] == 0).any()
    assert not (w[5880 - top_cut:5895 - top_cut, :] == 0).any()


def test_margin_from_border_is_at_most_180px() -> None:
    assert seg.WHITEN_BOTTOM_FROM_BORDER <= 180
