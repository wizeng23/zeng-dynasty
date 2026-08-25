# Bridge / crop fixes to revisit later (Book 2)

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
