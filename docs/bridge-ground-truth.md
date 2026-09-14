# Bridge (green-line) ground truth — Book 2

William-verified spec for the orphan-bridging pass (`src/segment.bridge_orphans` /
`_bridge_candidates`). Use this to self-evaluate after any algorithm change: regen
each graph, draw its green lines on the joined row-2 image, and check against the
verdicts below. This is about **precision** — exact endpoints matter.

## THE RULE (William, restated + pixel-verified 2026-08-24)

**Signal:** On the LEFT page there is a **floating subgraph** — a branch whose top
horizontal generation-bar is not connected to the rest of its branch. It is
orphaned **on the bar** (not on a parent *node*). That floating bar's RIGHT edge
sits at the page seam.

**Root cause (precise):** the connection is NOT lost to a seam misprint. It is lost
because the **previous page** (further RIGHT on the graph) is **missing the
horizontal bar** that would connect the floating subgraph to the rest of its
correct branch. That correct branch's horizontal bar **always includes the parent**
(the bar happens to connect to the parent) and **optionally some more children**.
Do NOT call it a "parent bar" — it is the horizontal bar OF THE BRANCH.

The bridge redraws that missing horizontal bar segment: it starts **at the page
seam** (the left edge of the previous / right page) and extends **RIGHTWARD** across
that previous page until it **connects**.

- **LEFT endpoint = the page seam x.** THIS IS THE RELIABLE SIGNAL.
- **RIGHT endpoint = where it "connects" = the first black pixel encountered within
  ~20 y-coordinates of the bridge's y** (tolerating scan drift). In theory that
  pixel is **another horizontal branch line — specifically its LEFTMOST pixel.**
  NOT a reliable signal on its own — it often "looks correct" either way. Do not
  read the right side to decide whether a bridge is needed.

**PIXEL-VERIFIED:** every William-confirmed-correct bridge has its left endpoint
exactly on a page seam (|Δ| = 1px):

| graph | bridge x0 | seam x (page edge) | Δ |
|-------|-----------|--------------------|---|
| 114_120 | 2847 | 2848 (p117) | 1 |
| 22_23   | 934  | 935  (p22)  | 1 |
| 28_30   | 1389 | 1390 (p28)  | 1 |
| 31_35   | 2626 | 2627 (p32)  | 1 |
| 69_82   | 5799 | 5800 (p76)  | 1 |
| 69_82   | 9249 | 9250 (p73)  | 1 |

**Seam x-positions are deterministic** = cumulative shrunk-page widths in the
left-to-right stacking order `segment()` uses (pages prepended LEFT, so graph order
is `[end, end-1, ..., start]`; seam of each page = running sum of widths to its
left; the +100px vertical frame does not shift x). `segment()` does NOT currently
record these — the fix must compute/record them.

NOT a bridge = riser crossing (a child/parent vertical interrupts the bar every
~200px) or scan nick (≤~20px hairline). Scan nicks get horizontal-gap FILL, never
a green bridge. The old "close every run-to-run gap" logic conflated all three —
that is the bug.

## Two distinct repairs — do NOT conflate

1. **Green bridge** — a genuine cross-page connector: the book was *printed*
   with the bar continuing onto the next page, and the page break lost it. Left
   end on the seam, extends right to first ink. This is what `bridge_orphans`
   draws green.
2. **Scan-line fill** — a ~≤20px hairline gap in a horizontal line caused by a
   *bad scan*, NOT a misprint. The bar is a single printed bar; the scan just
   dropped a few px. These should be smoothed/filled (like `bridge_horizontal_gaps`
   / `bridge_vertical_gaps`), NOT drawn as green cross-page bridges.

**Several graphs currently emit green bridges where only scan-line fill is
warranted.** Fixing that is part of the algorithm change.

## v1 results (2026-09-14)

The v1 port (`src/s5_build_tree.bridge_orphans`) reproduces every verdict below
by COUNT: 11_17 4, 22_23 1, 28_30 1, 31_35 1, 36_52 4, 58_62 1 (+1 nick fill),
69_82 2 (+ 尚澜's seam step, v0's manual L-connector), 114_120 1, 121_122 0,
126_128 0. NOTE the coordinates in the verdicts are QA ROW-2 (joined compare
image) pixels, not graph pixels, so they do not rescale to v1 graph coordinates;
compare counts, seam anchoring (every v1 bridge's left end is on a page seam) and
the QA overlay, not x values. Hairline nicks are now recorded separately
(`{stem}.nicks.json`) and never drawn green, as the rule below requires.

## Per-graph verdicts (William, 2026-08-24)

Page seams are the blue verticals in `qa` row 2. Coordinates below are in the
graph-image / imaginary.json coordinate space unless noted.

### 11_17 — NEEDS 4 bridges (currently stores 3, WRONG)
Corrected target (seam-to-first-ink), measured on the joined row-2 image
(6872 wide; seams at x≈230,1383,2536,3689,4842,5995 = p17|p16|p15|p14|p13|p12|p11):
- TOP gen-2:      left p13 seam 3689 -> first ink 4845   (y≈281 in row2 / y≈389 graph)
- LEFT-MID gen-4: left p15 seam 1383 -> first ink 2663   (y≈908 row2 / y≈1010 graph)
- RIGHT-MID gen-4:left p13 seam 3689 -> first ink 5586   (y≈908 row2 / y≈1010 graph)
- BOTTOM gen-6:   left p15 seam 1383 -> first ink 1505   (y≈1543 row2 / y≈1654 graph)
NOTE: coords above are ROW-2 (joined compare) space. The graph-space equivalents
must be recomputed when encoding; the current stored graph-space bridges are the
buggy ones (full-width top 138->6817, fused middle 1073->3676).
William confirmed this 4-bridge set is finally correct.

### 114_120 — CORRECT as-is
Stores 1 bridge: y=1657 x 2847->2980 (span 133). Leave alone.

### 121_122 — SCAN-LINE FILL ONLY, not a misprint
Stores 1 "bridge": y=1017 x 564->567 (span 3). This is a hairline scan gap, not a
cross-page connector. Should be handled by scan-line fill, not a green bridge.

### 126_128 — NO green bridges needed. Book printed correctly.
Currently stores 6 (WRONG): a big y=693 x30->2728 span-2698, plus five tiny
y=1003 spans (2-3px). The tiny ones are ~20px scan gaps needing smoothing; the
big one is spurious. NONE should be green bridges. Only scan-line fill, if
anything.

### 22_23 — CORRECT as-is
Stores 1 bridge: y=1008 x 934->1656 (span 722). Leave alone.

### 28_30 — CORRECT as-is
Stores 1 bridge: y=700 x 1389->1933 (span 544). Leave alone.

### 31_35 — CORRECT as-is
Stores 1 bridge: y=1019 x 2626->2845 (span 219). Leave alone.

### 36_52 — too wide to fully inspect; needs QA scrollbar (see below)
Stores 4 bridges. William's partial read:
- RIGHTMOST long green (y=1013 x 14143->17501, span 3358): OVER-EXTENDS by TWO
  PAGES on the LEFT side, but the net result is the right thing. i.e. correct
  endpoints would move the LEFT end two page-seams to the right; the right end
  is fine. The seam-to-first-ink rule should naturally shorten it.
- The TWO small green bars next to it (y=1341 x12992->13215 span223; y=1658
  x12992->13115 span123): CORRECT.
- The LEFTMOST green bar (y=1640 x8392->8503 span111): William could NOT see it
  (page too wide). VERIFY after scrollbar fix.

### 58_62 — 1 correct, 1 should be scan-fill
Stores 2 bridges:
- RIGHTMOST (y=1326 x3149->3484 span335): CORRECT.
- LEFT one (y=1014 x1756->1764 span8): should NOT be a bridge — just scan-line
  fill (hairline gap), not a misprint.

### 69_82 — CORRECT as-is
Stores 2 bridges: y=1338 x5799->5916 (span117); y=1645 x9249->9385 (span136).
Leave alone.

## Final algorithm (implemented 2026-08-24)

`src/segment.py`:
- `segment()` records **page seams** = cumulative shrunk-page widths in the
  left-to-right stacking order (pages prepend LEFT, so order is
  `[end..start]`); passes them to `bridge_orphans(..., seams=...)`.
- `_seam_at_bar_end(a, row, bar_right, seams)`: first seam at/right of the bar's
  right end (small snap-left), within one page-width. `None` -> not a page-boundary
  orphan.
- `_first_ink_right(a, row, start_x)`: skips the orphan's own ink tail, then the
  gap, returns the first ink of the NEXT branch line (the reconnection). ±20y drift
  tolerance (`BRIDGE_CONNECT_YTOL`).
- `_bridge_candidates`: one candidate per orphan = `[(row, seam, connect)]`, only if
  `connect - seam >= MIN_BRIDGE_SPAN` (60px; rejects seam-adjacent scan nicks).
- `bridge_orphans`: **freezes the orphan list ONCE** (pre-bridge parse). Bridges only
  those orphans. Does NOT re-derive orphans mid-loop -- doing so let a correct bridge
  perturb the re-parse into a false orphan (衍灿) whose bridge wrongly welded two
  subtrees (the "5th bridge" bug). Per-candidate gate: parse must still succeed (no
  ValueError = no two-parent weld).
- Removed dead code: `_gap_has_vertical`, `_empty_count`, `_component_count`,
  `MULTI_GAP_MAX`, `ORPHAN_CROSS_REACH`, full-bar-span candidate.

`scripts/qa_overlay.py`:
- `_card_notes()` renders a **searchable text** line per card listing each green
  bridge's coords/span (Cmd-F "green" jumps between bridged graphs).
- Scroll fix: `figure{overflow-x:auto}` + `img{margin-left:auto}` (was
  `align-items:flex-end`, which made a wider-than-viewport graph's left edge
  unscrollable).

## Summary of required changes
- 11_17: 3 (wrong) -> 4 (correct, seam-to-first-ink).
- 126_128: 6 -> 0 green bridges (scan-fill only).
- 121_122: 1 -> 0 green bridges (scan-fill only).
- 58_62: 2 -> 1 green bridge (drop the span-8 scan-gap one).
- 36_52: rightmost long bridge left-end should retract ~2 pages (seam-to-first-ink);
  verify leftmost bridge after scrollbar fix.
- 114_120, 22_23, 28_30, 31_35, 69_82: already correct — MUST stay unchanged.
- Book 1: no bridges — MUST stay byte-identical.
