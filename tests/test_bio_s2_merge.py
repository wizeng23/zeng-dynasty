"""Bio stage 2: merge a section's crop pages RTL, aligning on the horizontal rules.

The crops (bio stage 1) keep the 4 full-width horizontal rules in place. This stage
joins a section's contiguous pages into one wide image with the FIRST (section-start)
page on the RIGHT, later pages extending left -- so block order matches the tree's
RTL traversal. Pages are aligned by their TOP rule (padded above to a common y); the
BOTTOM rule then serves as the alignment check -- if a page's bottom rule drifts more
than a tolerance from the median, the whole ruled region is inconsistent and the
merge hard-fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bio import s2_merge as m
from src.s2_classify_pages import BIO_RULE_YFRAC

RULE = 8  # rule thickness in px


def _page(h: int, w: int, yfracs: tuple[float, ...] = BIO_RULE_YFRAC,
          content: bool = True) -> np.ndarray:
    """A crop page of height ``h`` with full-width rules at ``yfracs`` of ``h``.

    ``content`` adds a little non-rule ink so a page is not pure rules.
    """
    a = np.ones((h, w), dtype=np.uint8)
    for f in yfracs:
        y = int(f * h)
        a[y:y + RULE, :] = 0
    if content:
        a[h // 2:h // 2 + 40, w // 3:w // 3 + 40] = 0
    return a


def _pad_top(a: np.ndarray, n: int) -> np.ndarray:
    return np.vstack([np.ones((n, a.shape[1]), dtype=np.uint8), a])


def _rule_rows(a: np.ndarray, thresh: float = 0.6) -> list[int]:
    """Cluster centers of high-coverage (full-width rule) rows in ``a``."""
    cov = (1 - a).sum(axis=1) / a.shape[1]
    rows = np.where(cov >= thresh)[0]
    if len(rows) == 0:
        return []
    centers, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > 20:
            centers.append((start + prev) // 2)
            start = r
        prev = r
    centers.append((start + prev) // 2)
    return centers


# --- find_rules ---------------------------------------------------------------

def test_find_rules_returns_four_rules_top_to_bottom() -> None:
    a = _page(5580, 3400)
    rules = m.find_rules(a)
    assert len(rules) == 4
    assert rules == sorted(rules)
    for got, f in zip(rules, BIO_RULE_YFRAC):
        assert abs(got - f * 5580) <= 15


def test_find_rules_detects_faded_low_density_rule_off_canonical() -> None:
    # A faded top rule (density ~0.33, below the old 0.6 threshold) that sits 40px
    # BELOW its canonical y. Density detection misses it, so the canonical fallback
    # would place the top rule at frac*h -- 40px wrong, hiding real misalignment.
    # Run-length detection (using canonical only as a search hint) must find the rule
    # at its TRUE y, so the bottom-rule alignment check can see the drift.
    h, w = 5580, 3400
    a = np.ones((h, w), dtype=np.uint8)
    for i, f in enumerate(BIO_RULE_YFRAC):
        y = int(f * h) + (40 if i == 0 else 0)
        if i == 0:
            a[y:y + RULE, ::3] = 0  # faded: every 3rd column -> low density, long run
        else:
            a[y:y + RULE, :] = 0
    rules = m.find_rules(a)
    assert len(rules) == 4
    true_top = int(BIO_RULE_YFRAC[0] * h) + 40
    assert abs(rules[0] - true_top) <= 15  # found at its REAL y, not snapped to canon


def test_find_rules_snaps_missing_faint_rules_to_canonical() -> None:
    # A sparse page where the top two rules are too faint to detect (like p6/p8 of
    # book3 0_1): only the bottom two exceed the coverage threshold. find_rules must
    # still return 4 y-positions, snapping the missing ones to the canonical fracs.
    h, w = 5580, 3400
    a = _page(h, w, yfracs=(BIO_RULE_YFRAC[2], BIO_RULE_YFRAC[3]))
    rules = m.find_rules(a)
    assert len(rules) == 4
    for got, f in zip(rules, BIO_RULE_YFRAC):
        assert abs(got - f * h) <= 15


# --- trim_sides ---------------------------------------------------------------

def _bio_page(h: int, w: int, text_left: int, text_right: int) -> np.ndarray:
    """A page with the 4 rules and a text block spanning [text_left, text_right).

    Text columns ink far more than the rule-only whitespace columns, mimicking real
    pages (whitespace ~0.004, text ~0.03).
    """
    a = np.ones((h, w), dtype=np.uint8)
    for f in BIO_RULE_YFRAC:
        y = int(f * h)
        a[y:y + RULE, :] = 0                       # rules cross every column
    a[:, text_left:text_right] = 1                 # clear, then draw dense text
    for c in range(text_left, text_right):
        a[100:100 + int(0.03 * h), c] = 0          # ~0.03 density text columns
    return a


def test_trim_sides_removes_whitespace_up_to_the_cap() -> None:
    # Text sits 200px in from each side; the full 80px cap of whitespace comes off
    # (well beyond 80px is still whitespace, so no padding is withheld).
    h, w = 4000, 1000
    a = _bio_page(h, w, text_left=200, text_right=w - 200)
    trimmed = m.trim_sides(a)
    assert trimmed.shape[1] == w - 160  # 80 off each side


def test_trim_sides_keeps_20px_pad_when_stopping_at_content() -> None:
    # Text starts 60px in on the left: reaching content trims 60, but we keep 20px of
    # whitespace before the glyphs -> trim 60-20=40. Right has 200px whitespace -> 80.
    h, w = 4000, 1000
    a = _bio_page(h, w, text_left=60, text_right=w - 200)
    trimmed = m.trim_sides(a)
    assert w - trimmed.shape[1] == 40 + 80
    # 20px of whitespace precedes the first text column in the trimmed image.
    col_dens = (1 - trimmed).sum(axis=0) / trimmed.shape[0]
    text_start = next(i for i, d in enumerate(col_dens) if d > 0.02)
    assert 18 <= text_start <= 22  # ~20px pad kept


def test_trim_sides_trims_nothing_when_content_within_pad() -> None:
    # Text starts only 10px in (< the 20px pad): trimming would leave <20px, so trim 0.
    h, w = 4000, 1000
    a = _bio_page(h, w, text_left=10, text_right=w - 200)
    trimmed = m.trim_sides(a)
    assert w - trimmed.shape[1] == 0 + 80  # left untouched, right full 80


def test_trim_sides_keeps_horizontal_rules() -> None:
    # Trimming columns must not remove the rules (they span full height); they just
    # get shorter. find_rules still sees 4 rules after trimming.
    a = _bio_page(4000, 1000, text_left=200, text_right=800)
    trimmed = m.trim_sides(a)
    assert len(m.find_rules(trimmed)) == 4


# --- merge_section ------------------------------------------------------------

def test_merge_section_places_first_page_on_the_right() -> None:
    # Two pages with distinguishing ink at different columns; after RTL merge the
    # first page's ink must sit to the RIGHT of the second page's ink.
    h = 4000
    p0 = np.ones((h, 1000), dtype=np.uint8)
    for f in BIO_RULE_YFRAC:
        p0[int(f * h):int(f * h) + RULE, :] = 0
    p0[50:60, 20:30] = 0          # marker near p0's left region
    p1 = np.ones((h, 1000), dtype=np.uint8)
    for f in BIO_RULE_YFRAC:
        p1[int(f * h):int(f * h) + RULE, :] = 0
    p1[50:60, 20:30] = 0          # same relative marker

    merged = m.merge_section([p0, p1])

    # Non-rule ink columns: the p0 marker should be right of the p1 marker.
    band = merged[45:65, :]
    ink_cols = np.where((1 - band).sum(axis=0) > 0)[0]
    assert len(ink_cols) >= 2
    p0_marker_col = ink_cols.max()
    p1_marker_col = ink_cols.min()
    assert p0_marker_col > p1_marker_col
    assert p0_marker_col >= merged.shape[1] // 2  # first page in right half


def test_merge_section_aligns_top_rules_to_common_y() -> None:
    # Two pages whose top rules start at different absolute rows (one shifted down by
    # a top margin) but with the SAME rule spacing. After merge the top rule must be a
    # single full-width line at one shared y (top-rule alignment padded them equal).
    p0 = _page(4000, 800)
    p1 = _pad_top(_page(4000, 800), 150)  # same page, pushed 150px down
    merged = m.merge_section([p0, p1])

    rows = _rule_rows(merged)
    top_y = min(rows)
    cov_at_top = (1 - merged[top_y]).sum() / merged.shape[1]
    assert cov_at_top >= 0.9  # rule spans essentially the whole merged width


def test_merge_section_hard_fails_when_bottom_rule_drifts() -> None:
    # A page whose rule spacing is stretched so that, once top-aligned, its bottom
    # rule sits far below the others -> inconsistent ruled region -> hard fail.
    good = _page(4000, 800)
    # Push the bottom rule 90px below its usual spot (past the 50px tolerance but
    # still close enough that find_rules associates it with the bottom slot).
    stretched = _page(4000, 800,
                      yfracs=(0.193, 0.396, 0.597, 0.802 + 90 / 4000))
    with pytest.raises(ValueError, match="bottom rule"):
        m.merge_section([good, good, stretched])


def test_merge_section_accepts_small_bottom_rule_drift() -> None:
    # Real book3 0_1 pages drift <50px; a ~30px bottom-rule difference must pass.
    good = _page(4000, 800)
    slight = _page(4000, 800,
                   yfracs=(0.193, 0.396, 0.597, 0.802 + 30 / 4000))
    merged = m.merge_section([good, slight])
    assert merged.shape[1] == 1600


# --- bio_sections -------------------------------------------------------------

def test_bio_sections_groups_contiguous_pages_per_section() -> None:
    # first_pages marks section starts; a section runs to the page before the next.
    sections = m.bio_sections_from(
        bio_pages={2, 3, 4, 10, 11},
        first_pages={2, 10},
    )
    assert sections == [[2, 3, 4], [10, 11]]


def test_bio_sections_raises_on_gap_within_a_section() -> None:
    # Pages 2,3,5 with only 2 marked as start: 3->5 is a non-contiguous gap, which
    # means a page is missing or misclassified -- never weld across it.
    with pytest.raises(ValueError, match="gap"):
        m.bio_sections_from(bio_pages={2, 3, 5}, first_pages={2})


def test_bio_sections_drops_skipped_page_from_its_section() -> None:
    # Book 4 p214 is deliberately excluded (its branch has no tree graph). It must be
    # dropped from its section run, leaving the rest.
    sections = m.bio_sections_from(
        bio_pages={211, 212, 213, 214},
        first_pages={211},
        skip={214},
    )
    assert sections == [[211, 212, 213]]


def test_bio_sections_skip_in_middle_does_not_trigger_gap() -> None:
    # A deliberate skip in the MIDDLE of a run is an exclusion, not a missing page:
    # dropping it must not raise the contiguity gap error.
    sections = m.bio_sections_from(
        bio_pages={5, 6, 7, 8},
        first_pages={5},
        skip={7},
    )
    assert sections == [[5, 6, 8]]
