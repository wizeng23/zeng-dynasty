# Orphan-bridging in the seam merge — design

**Date:** 2026-08-20
**Status:** implementing (William-specified, then autonomous)

## Problem

Some Book 2 subtree pages are **missing a horizontal connector line** (a real gap
in the printed book). When such a page is stitched, its exposed seam endpoints
don't all have a partner on the neighbouring page — the unmatched ones are
**orphans**. Two consequences:

1. The `merge_graphs` **top-align** step (pair `left_y[0]`↔`right_y[0]`) is
   *unbounded*: when counts differ (e.g. p13 exposes 1 line vs the accumulated
   3), it force-matches the wrong pair and pads by hundreds of px (p13: 841px),
   shifting the whole page far out of place.
2. Orphaned bars are dropped (no connector), so a branch that truly continues
   across the missing-line page is left disconnected → phantom empty nodes and
   an orphaned subtree (e.g. p14's subtree is orphaned because p13 lost its bar).

Ideally **0 orphans** exist; the ones that do are book errors we must repair.

## Fixes

### 1. Cap the alignment shift at ±100px
The top-align pad (segment.py ~L380-388) must not exceed 100px. `find_best_alignment`
already searches only ±100; the top-align is the unbounded one. Clamp it so an
extreme mismatch can't throw the page hundreds of px off.

### 2. Orphan-bridging (trace-right)
In `merge_graphs`, after matched lines are connected and orphans identified,
**repair each orphan** instead of dropping it:

- An orphan bar hangs with **empty space to its right**. Trace its row rightward
  across the concatenated image until it hits ink (another line/branch).
- Draw a horizontal connector bridging the gap — this reconnects the branch that
  continued across the missing-line page.
- The target may be a full page (or more) to the right; the branch on that side
  can look locally complete, but the orphan on the other side proves it is the
  correct connection.

William's worked examples in `11_17`:
- **Top bar**: spans p17→p14, orphaned at p14 (p13 missing its bar). Trace right
  ~a full page → connects to the other orphaned bar.
- **Middle bar**: p16 has an orphan; trace right a bit over a page → connects to a
  branch in p14 (which looks complete from its own side).
- Expected: fixes p16 middle orphan + p14 top & middle orphans.

### Node-shape verification (guardrail)
Per line-segment, three y-coords define a valid branch:
- `min_y` = parent (top of hang-line),
- `bar_y` = horizontal fan-out bar (middle),
- `max_y` = children (bottoms).
Invariant: at `min_y`'s x-column there is a child directly below (a child shares
the parent's column); other children spread left. Flag segments violating this.
An orphan shows as a `bar_y` with empty space to its right.

## QA requirements

- **Green** imaginary lines: record every synthetic segment `merge_graphs` draws
  (orphan-bridges AND the existing tiny seam-connectors) to a sidecar
  `books/bookN/graphs/{stem}.imaginary.json` (list of `[r0,c0,r1,c1]` in final
  graph coords). The QA overlay reads it and draws those lines green.
- **Flag orphans** clearly in the QA overlay (count + markers) for inspection.

## Verification

- Book 2 empty-node count should drop further (currently 22; the ~5 in `11_17`
  are the target).
- Book 1 must stay byte-identical (no orphans there).
- The QA green lines let a human confirm each invented connection is correct.
