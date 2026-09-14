# Bridge / crop fixes to revisit later (Book 2)

> **v1 STATUS (2026-09-14): orphan-bridging is NOT ported to the v1 pipeline.**
> Everything below describes the v0 `src/v0/segment.py` pass (v0 pixel geometry).
> `src/s4_merge_pages.py` explicitly defers bridging; the v1 Book 2 Stage-5 parse
> therefore has 37 empty phantom bars across 12 multi-page graphs (36_52: 10,
> 69_82: 10, 106_113: 3, 11_17: 3, …) and 42 seam-orphan roots that survive into
> `book2_stitched.jsonl` (51 roots). Porting the trace-right rule (seam x from
> cumulative page widths, ×3 scale) is the next pipeline task after OCR review.

Status as of 2026-08-24: **all 45 Book-2 subgraphs are LOCKED** — 0 within-graph
orphans, green bridges match William's hand-verified ground truth, 0 box overlaps,
Book 1 byte-identical. Next pipeline step: William reviews OCR.

The automatic orphan-bridging (trace-right rule, `src/segment.py`) resolved every
graph EXCEPT four, which needed surgical per-graph help. These are recorded here to
revisit and (ideally) fold back into a more general automatic rule so the manual
list can shrink. All manual fixes live in `src/segment.MANUAL_BRIDGES` and
`src/segment.CROP_KEEP_LEFT`; the QA reads them and draws them green.

## The 4 graphs that needed manual help

### 58_62 — faint scan break in a generation bar
- **Wrong:** gen-4 bar had a ~7px break (x1757-1764, y1014-1016) too faint / drifted
  for the auto trace to cross; left a floating (orphan) sub-bar.
- **Fix:** manual rectangle fill `(1013,1755,1018,1766)` — solidify the gap so the
  two runs read as one bar.
- **Revisit:** could the auto rule hop slightly-larger faint gaps when the two runs
  are collinear (same y +-2)? Careful not to cross real riser gaps.

### 69_82 — orphan bar a step above the next bar
- **Wrong:** gen-2 orphan bar ended at (x135,y372); the next bar started at
  (x171,y386) — a small horizontal gap PLUS a ~13px vertical step. Auto rule draws
  only horizontal, so it couldn't bridge the step.
- **Fix:** two manual rects — horizontal stub `(371,133,375,173)` + short vertical
  drop `(371,169,389,174)`.
- **Revisit:** the auto draw_bridge already snaps to bar-ink y at each end; a small
  vertical step within the same page might be handled by extending that snap.

### 121_122 — TWO problems
1. **Crop cut the top generation.** `shrink_page` crops to ink contiguous with the
   densest column; p122's 尚泽/尚沾 (highest gen, x~299-443) sit LEFT of the main tree
   across a blank gap, so they were cropped away — row 2 showed an empty top bar.
   - **Fix:** `CROP_KEEP_LEFT["book2"][122] = 289` pulls p122's left crop out to keep
     them. (A blanket "keep all ink" change would alter 87 pages incl. Book 1 —
     rejected; per-page override instead.)
   - **Revisit:** a general rule could keep any ink group within ~1 page-width of the
     main span, but must not re-admit label/margin remnants on other pages.
2. **Orphan bar after the crop fix.** gen-4 bar (x563) didn't reach its parent.
   - **Fix:** parent-trace manual bridge `(1015,563,1607)` — trace right to the
     parent's column (William's rule).
   - NOTE: 121_122 coords are in the POST-crop-override geometry; if p122's
     CROP_KEEP_LEFT changes, recompute this bridge's x's.

### 126_128 — two orphan bars not reaching their parents
- **Wrong:** gen-3 bar (x445) and gen-5 bar (x2179) each ended before their parent's
  column; the auto rule read them as cross-graph (bar traced toward the edge).
- **Fix:** two parent-trace manual bridges `(693,445,2718)` and `(1003,2179,2623)`
  — trace each right to its parent node's x (nearest named node in the generation
  above, to the right).
- **Revisit:** this is the cleanest candidate to automate — implement William's
  "parent-trace" rule generally: for an orphan, find the nearest named node in the
  generation directly above and to the right, and bridge to its column. If that
  reliably reproduces all 4 manual entries here + 58_62/69_82, the whole manual list
  can be deleted.

## OCR: characters missing dots/strokes (William, 2026-08-25) — REVISIT
Some parsed names are missing small dots/strokes (e.g. a 丶 in a radical). Suspected
cause: `remove_small_islands(max_size=10)` in `src/extract_pages.py:90` (called at
:336) erases every connected ink component <=10px as "scan noise" — but a legitimate
small dot/stroke that is disconnected from the main glyph gets deleted too. So the
character is damaged before it ever reaches OCR.
- **Where:** `src/extract_pages.py` `remove_small_islands`, `max_size=10`.
- **Fix ideas:** lower `max_size`; or only remove specks that are far from any larger
  component (a real dot sits close to its glyph, noise is isolated); or restrict
  removal to the graph-line regions, not the name-crop regions; or run it before,
  not after, name cropping. Must re-verify against the OCR ground truth and the
  parse (don't reintroduce scan-spot false components).
- **Impact:** OCR accuracy + the name crops shown in `scripts/ocr_review.py`.
- NOTE: changing extract_pages re-runs Stage 1 -> would change pages/graphs for BOTH
  books; check the freeze / re-verify everything downstream.

## v1 scan quality: ADF ink smears / graininess (William, 2026-09-01) — CROSS-CUTTING

The new v1 scans were run through an automatic document feeder (ADF), which
**smeared wet/loose ink across the pages** — so regions of a page are grainy or
have faint stray ink dragged from elsewhere on the sheet. This is a property of
the *source scans* (`books/bookN/bookN.pdf`), not any one stage, so **every stage
must be robust to it**:
- **Stage 1 (extract):** smear can create faint stray marks near the page edge /
  above the border that fool the frame detectors (this is the phantom-line-above-
  the-top-border issue on some Book 4 pages — extra whitespace/junk above the true
  outer border). Border detection must tolerate grain and not latch onto smear.
- **Stage 2 (classify):** faint smeared lines could add spurious full-width rows;
  the fixed 4-rule fingerprint (`s2_classify_pages`) held up, but watch for it.
- **Stage 3 (segment/crop):** the fixed-inset `trim_borders` top cut assumes a
  clean border at a fixed offset; smear-induced whitespace above the border breaks
  that assumption and misaligns the graph top across pages -> Stage 4 merge seams
  mismatch. (Active investigation: replace the fixed top trim with a border-band
  detector so the graph top is standardized regardless of smear whitespace.)
- **Stage 5 (build_tree) + OCR:** grain adds small speckle components (interacts
  with the missing-dots `remove_small_islands` issue above) and can degrade name
  crops. Speckle thresholds (e.g. `COL_INK_MIN`) already guard some of this.

No single fix — each stage's thresholds/detectors should be hardened against grain
rather than assuming clean ink. Flag any new anomaly that traces back to a smear.

## Automation opportunity (the big one)
William's **parent-trace rule** ("trace right from the orphan to the x of its
parent = nearest node in the generation above, to the right") resolved 126_128 and
121_122 cleanly and likely generalizes. The current auto rule instead traces to the
next *bar ink*, which fails when the parent's riser is further right than the next
bar fragment. Revisit: replace/augment `_bridge_candidates` with parent-column
targeting (needs reliable generation-band detection — the ad-hoc top-row clustering
used in QA's `_generation_rows` is a starting point but was noisy; use the parse's
structure instead). Verify against `docs/bridge-ground-truth.md` before removing any
manual entry, and re-check the freeze (Book 1 + all other Book 2 graphs).
