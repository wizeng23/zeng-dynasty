# Project history — Zeng family tree (族谱) digitization

A chronological story of the project, for the record. Reconstructed from git
history + the topic logs (`docs/progress.md`, `overnight-worklog.md`,
`bridge-*.md`, `ocr-bakeoff.md`). Newest era last. Append a dated entry after each
significant chunk of work.

Goal: turn a 4-book scanned Chinese zupu into a structured tree + website.
Pipeline: spreads → pages → graph images → tree → OCR → stitch → site.

---

## Era 1 — First pass, hand + Colab (2025-08 → 2025-10)

- **2025-08-10** — Repo created; scans + the book's instruction PDF added.
- **2025-09-28/29** — First hand-typed Book 1 data (`data/book1.jsonl`); small fixes.
- **2025-10-01/03** — Page normalization; clean scan spots; crop each page to just
  its line-graph.
- **2025-10-05/06** — First automated **tree parse**; Book 2 scans added.
- **2025-10-07** — `utils.py`; merge adjacent page-graphs together.
- **2025-10-17** — Finalized Book 1 graph; recovered `book1_merged`; **initial
  website**; Book 2 pages added. (This is the era whose logic now lives in
  `old/book_parser.ipynb` + `old/utils.py`.)
- **2025-10-19** — Improved graph-merging logic.
- **2026-03-01** — README update. (Then a long pause.)

Outcome: a working-but-ad-hoc Colab pipeline; Book 1 mostly done, Book 2 crashed
the parser (2-char stacked names → false double-parents). Per-book fudge factors
were baked inline.

## Era 2 — The rewrite into `src/` (2026-08-17 → 2026-08-18)

Restructured the repo (old pipeline archived to `old/`) and rewrote the pipeline
cleanly into `src/` per `docs/specs/2026-08-18-pipeline-rewrite-design.md`, fixing
the known bugs:
- **RTL / eldest-first** node ordering + BFS-by-generation IDs + `father` field +
  1-indexed generations (the old parse was LTR). Golden Book 1 renumbered to match.
- **Stage 1** (`extract_pages.py`) — bit-exact reproduction of the old page split
  (0.000% pixel diff), both books.
- **Stages 2–3** (`segment.py`, `build_tree.py`) — reproduced Book 1's tree with
  100% topology match to the old parse; then **parsed Book 2**, the book that broke
  the old pipeline (fixed via top-band collapse for stacked names).
- **Website** — Next.js 16 + React 19 + d3-in-React; deployed to GitHub Pages.
- **Stage 3.5 stitching** (`src/stitch.py`) — folds each graph's duplicate root
  into its canonical leaf → one connected lineage. Matching is not automated;
  per-book `BOOK_MERGES` (Book 1's 13 done by hand). Book 2 is a 46-root forest
  until stitched.
- **Stage 3.6 OCR bake-off** — chose **PaddleOCR PP-OCRv5** (local, free): ~90%+ on
  a hand-verified set, beating Google Vision/DocAI/Mistral (all weak on isolated
  glyphs). See `docs/ocr-bakeoff.md`.

## Era 3 — QA tooling + Book 2 defect hunt (2026-08-19)

Built `scripts/qa_overlay.py` (visual parse QA) and used it to quantify Book 2's
page-seam defects:
- Recalibrated the grid-check band (Book 1's 149 warnings were all false positives).
- QA evolved to a 3-row card: raw scan → cropped pages → parse overlay.
- **Quantified the seam-break defect**: 163 empty "phantom" nodes across 19
  multi-page graphs (a connector bar lost at a page seam leaves a nameless orphan).
- Fixed the seam-merge coordinate bug (empties 165 → 89) and bridged short
  horizontal line-breaks in `find_lines` (89 → 22).

## Era 4 — Orphan bridging, first generation (2026-08-20)

- Fixed a y-axis over-shift in `merge_graphs` (was force-top-aligning seam
  endpoints, skewing whole graphs).
- Added **self-verifying orphan-bridging** (draw a connector, re-parse, keep only if
  empties strictly drop): empties 21 → 12 → 10 → 9 via successive candidate shapes.
- **Generation drift** root cause found: `merge_graphs` framed +100px on every merge,
  so content crept down per page. Fix: frame the assembled graph ONCE. (A per-seam
  clamp was tried and reverted — William was right that the true shift is ~10px.)

## Era 5 — OCR at scale + review tool (2026-08-21 → 2026-08-23)

- Fixed 114_120's overlapping-box parse; added `ignore_regions` for grandpa's
  handwritten ink in 8_10 (note 祖祠對坟口/欽秀堂 saved for manual attach to 聞詣).
- **OCR populated** every node's name across Books 1 & 2 with PP-OCRv5 (sidecar
  `data/bookN_names.json`, override layer, low-conf flags).
- Built the **OCR review filmstrip tool** (`scripts/ocr_review.py`): per-character
  crops, pinyin, keyboard nav, persistent overrides.
- Several rounds of QA-overlay alignment fixes (rows 2↔3 lining up).

## Era 6 — Bridging rewrite to the trace-right rule; Book 2 subgraphs LOCKED (2026-08-24 → 2026-08-25)

The big push (full blow-by-blow in `docs/overnight-worklog.md`, A1–A14):
- Rewrote orphan-bridging around William's **trace-right rule** (anchor at the orphan
  bar's true right end, bridge across the page-break gap to the next bar, drawing the
  fill at the real bar-ink y so it forms one traversable line). Iterative gate:
  accept a bridge only if the parse survives and the orphan count strictly drops
  (rejects the "5th-bridge" false orphan + runaway over-hops).
- Verified against William's hand-checked ground truth (`docs/bridge-ground-truth.md`)
  on 10 graphs; 41/45 Book-2 graphs resolved automatically.
- The final **4 graphs** needed surgical per-graph help (`MANUAL_BRIDGES`,
  `CROP_KEEP_LEFT`): 58_62 (faint gap), 69_82 (L-connector), 121_122 (p122 crop
  restored 尚泽/尚沾 + parent-trace bridge), 126_128 (2 parent-trace bridges). Details +
  automation TODO in `docs/bridge-revisit-notes.md`.
- QA gained: searchable "green"/"orphan" text (page-span + generation), row-2 page
  numbers, wide-graph scroll fix.
- **Result: all 45 Book-2 subgraphs LOCKED** — 0 within-graph orphans, 0 box
  overlaps, green bridges match ground truth, Book 1 byte-identical.
- Branch `book2-bridge-trace-right` (commits 0d5ef40 → 32bf348). William to push.

**Next step:** William reviews OCR (via `scripts/qa/ocr.py`), then graph merge
(stitching Book 2 + connecting Book 1), then render the website.

## Era 7 — New scans (v1: cut-spine ADF) + Stage-1 rebuild for all 4 books (2026-08-26 → 2026-09-01)

William re-scanned every book via cut spine + automatic document feeder (ADF): one
page per PDF page, 600 dpi, bitonal (canonical) + grayscale fallback. Books 3 & 4
exist only as these new scans. Naming: `bookN.pdf` = v1 bitonal (parse target),
`bookN_gray.pdf` = v1 grayscale, `bookN_v0.pdf` = the old glass-top scan. The old
glass pipeline is archived in `src/v0/`; `src/` is the canonical v1 pipeline. Work
at **full native resolution** (no downsize) for max OCR fidelity.

**Stage 1 (`src/extract_pages.py`) — rebuilt around finding the printed FRAME.** Each
page is boxed by a closed double-line rectangle; we detect it, then perspective-warp
its 4 corners to a clean rectangle — deskewing, cropping to the frame, and dropping the
scanner margin in one step (v0-style; frame line kept). Detection: seed from each of
the 4 edge midpoints, flood-fill the ink (O(perimeter)), morph-open to kill whiskers,
directional morph-close to bridge small gaps, robust per-side line fit → corners. A
`verify_normalized` re-detect gates the crop. Per-book content page ranges + per-page
corner metadata (`corners.json`) are written every run.

Degraded frames (broken/faded borders, mostly books 3–4) fail flood-fill, so a second
detector — a **Hough + scored geometric model** (double-line 35–40px apart = strongest
signal, outermost line, derive the 4th side from the other 3) — rescues them. A
comparison run flags pages where the two detectors disagree or one fails.

**Human QA layer.** `scripts/qa/` unifies all QA tools (border-corner review server on
8760, OCR filmstrip on 8761, static overlay/artifact generators). William reviewed
every flagged page by dragging corners; saved to `corners_review.json`.
`src/apply_corners.py` does the final generation: per page, corners resolve as review
override → agreed detector → Hough fallback, then warp + write.

**Result: Stage 1 complete for all 4 books** — book1 17, book2 134, book3 292, book4
317 pages. book3 had 12 reviewed pages, book4 46. Stage 2 was also split into two steps
(`src/segment.py` crop → `crops_bw/` + `starts.json`; `src/merge_pages.py` merge →
`graphs_bw/`), done for book1 (14 subtree graphs). Image output is gitignored
(regenerable from PDFs + corners); the corner JSONs are tracked (the review layer is not
reproducible).

**Next step:** crop + merge for books 2/3/4, then `build_tree` on the v1 pipeline.

## Era 8 — biography-page detection + stage renumber (2026-09-01)

Ran the crop stage for books 2/3/4. Book 2 crops + merges cleanly (44 graphs).
Books 3 & 4 exposed a new problem: they **interleave biography pages** (dense
name + birth/death prose) with the tree pages, and the crop stage assumed every
page was a tree — the biographies would have merged into garbage graphs.

**New stage: page classification** (`src/classify_pages.py`). A page is a
biography iff it has exactly 4 full-width horizontal rules at *fixed* y-fractions
≈ 0.19/0.40/0.60/0.80 (the bio cell-grid template) and no other full-width
interior rule; the page's own top/bottom frame border is ignored. Getting here
took several wrong turns — ink-density ratios and a naive "has a ruled grid"
test both failed, because Book 2's **tree** pages share a 4-rule grid too (just
at a different template, ~0.15/0.32/0.48/0.64). William's insight — bio rules are
at *fixed* positions and span *edge to edge*, tree fan-out bars don't — was the
key. Validated: Books 1 & 2 (all-tree) → **0** biographies; Book 3 → 51 graph /
241 bio, Book 4 → 147 graph / 170 bio; **0** conflicts with the independent
subtree-start detector. A frame-edge false-negative (the page border counted as a
spurious 5th rule) was found via a full-book contact-sheet review and fixed.
Writes `books/bookN/pages/page_types.json` (`graph_pages`/`bio_pages`, each bio's
`follows_graph` for later tree↔biography association). The crop stage skips bio
pages when the sidecar exists (no-op for books 1/2 — their output stays
byte-identical, verified by hash).

**Stage renumber (no more `.5`s).** With classification inserted the pipeline is
now 7 whole-integer stages: 1 extract_pages, 2 classify_pages, 3 segment (crop),
4 merge_pages, 5 build_tree, 6 stitch, 7 OCR. Renumbered the `src/` module
docstrings and the live docs (CLAUDE.md, pipeline.md, progress.md, milestones.md,
this file); dated design specs left as historical snapshots.


## Era 9 — Book 1 finished end-to-end; OCR-box QA; name-matched stitch; website (2026-09-13 → 2026-09-14)

Drove **Book 1 all the way through the v1 pipeline** (stages 1→7) to one
connected, OCR'd, publishable lineage. Along the way:

**Stage 3 whitespace/inner-border fix.** `whiten_margins` blanked a fixed ~300px
bottom slab measured from the trimmed edge, which cut into low-hanging 3-char leaf
names (page 13's 信九郎/宪七郎 lost 郎). Reworked to anchor to the **inner border**
(`trim_borders` already leaves the top edge at the top inner border, bottom edge at
the bottom): whiten a fixed 250px below the top / 200px above the bottom (measured
v1-native margins between border and graph), touching no character.

**Stage 5 name-crop rework — reverted to v0's crop-to-ink, scaled for v1.** Long
back-and-forth; the settled rules: crop the node's own segment `[top..bot]`, both
HARD boundaries (never pad past an endpoint into a parent/child riser — fixed 点's
clipped top dot and root/leaf riser stubs); `infer_ends` end-recovery window scaled
to v1 (was v0's 200px unscaled → reached 1 char, truncating 2-char leaves; now 600);
speck-tolerant top/bottom trim (smooth the row profile so a stray ADF smear dot
can't hold the box open — fixed 8/133 empty blocks); and **every** remaining
unscaled v0 pixel literal scaled by `V1_SCALE=3` (insets, sort-band tolerance, and
the blank-crop ink threshold by `V1_SCALE²` since it's an area). Measured the true
v0→v1 page ratio (1300×1950 → ~3781×5725 ≈ **2.93**); kept the clean 3.0.

**Stage 6/7 SWAP.** OCR is now stage 6, stitching stage 7 — stitching matches a
duplicate root to its canonical leaf *by name*, so it needs OCR first. Renamed
`s7_ocr*`→`s6_ocr*` (QA port 8767→8766); fixed a latent import bug (`s6_ocr`
imported nonexistent `src.ocr_paddle`/`ocr_cloud`).

**OCR (stage 6).** PP-OCRv5 populated Book 1 names → `data/book1_names.json` →
`book1.jsonl`. Captured PP-OCRv5's **per-character boxes** (`return_word_box=True`)
into the sidecar. **Rebuilt the OCR-review QA (`scripts/qa/s6_ocr.py`) around one
cell per NAME:** whole-name crop with all char boxes drawn (distinct colors),
vertical top-to-bottom edit box sized to match the crop's glyph scale, a
**coverage-gap flag** ("possible missed char" when ink lands outside all boxes —
caught the 4 real misses: 守云→二, 存义→存, 彭十郎/平十郎→十郎), box normalization for
PP-OCRv5's mis-oriented boxes (transposed x↔y when x1>width; horizontally-tiled →
dropped, fall back to ink-split), char-count baseline = `len(ocr_name)` (not the
aspect-ratio guess that mis-split 彭六郎 at h/w 3.50→4), name-box height sized to
the character count so a hallucinated extra glyph (宣→"宣J") can't be clipped out of
view, reviewed-tracking (Enter=confirm, with one-time key migration from the old
per-char format) + Alt+F impossible-flag (persisted server-side in
`data/book1_flags.json`). William reviewed; 23 human overrides, 2 flags.

**Stitch (stage 7) — the real bug.** Ported v0 stitch → `src/s7_stitch.py`. The
hand-coded `BOOK_MERGES` was authored against the *v0* parse; v1 renumbered graph-8's
leaves, so the same provenance string pointed at a different person and **7 of 13
seams merged the WRONG nodes** (e.g. 吉祥's subtree attached under 泰). `verify_stitch`
had passed only because it compared against an also-v0-derived oracle *by provenance*,
never checking that seams join same-named people. **Fix: now that OCR supplies names,
`find_merges` matches each duplicate root to its earlier same-name leaf** (Book 1:
13/13 unique, no ambiguity). Added a name-mismatch guard in `stitch_nodes` and
updated `verify_stitch` to check the actual `s7_stitch` output's tree shape against
the oracle — **both PASS**; 0 wrong edges vs golden (remaining golden diffs are OCR
character variants: 羨/羡, 浼/免, 申/电). Book 1 → **150 nodes, 1 root (点), 56 gens,
fully connected** → `data/book1_stitched.jsonl`.

**Website.** Reduced the Next.js site (`web-app/`) to a single dataset:
`book1_stitched` (the finished graph), removing golden/parsed/book2; export script
now copies from v1 `5_names/`. Verified it builds and renders all 150 nodes with
OCR names. Deploys to GitHub Pages on push to `main`.

Merged the whole branch into `main` (fast-forward) and pushed. Book 1 **shipped**.

## Era 10 — Book 2 through Stage 7: stepped-bar parse fix, chain-folding stitch (2026-09-14)

**Stage 5 crash (graph 67_68) — root cause verified against the ink, not the handoff.**
The handoff named the mechanism correctly (a *stepped* sibling bar: 宏羨's hang-line
steps left and UP into a raised bar over 闻评/闻瑛, then continues down to 闻诏) but its
proposed fix — "pick the through-line as parent, treat the other top-band endpoints as
children" — would have anchored 闻评 at the bar corner and lost 闻瑛 entirely (col 1300
never appears in the top band). The real break is that `find_line_ends` assumed a
single-level fan-out on BOTH ends: parent at the min row (here it is 103 rows lower)
AND all children within `end_threshold` of the max row (here two hang 330 rows
higher). A synthetic clean-edged stepped bar showed the old code would not even crash
— it silently returns the bar corner as parent; 67_68 crashed only because the bar's
ragged top edge produced multiple min-row specks.

Fix (`src/s5_build_tree.py`): a **stroke read** cross-checks the band read — free ends
of narrow (≤40px) vertical strokes that stay narrow for 100 rows: upward = parent
connection, downward = child connection (bar corners widen at once, ragged specks run
out of ink, so neither passes). The band read stays the fast path and keeps its exact
coordinates where the two agree; the stroke read decides the parent when the band read
is ambiguous or disagrees with one clear hang-line, and adds child risers ending above
the bottom band. Verified on all 685 components of Books 1+2 (old-vs-new harness):
Book 1 byte-identical (sidecars, crops, topology); Book 2 changed exactly two
components — 67_68, and one 11_17 riser whose child 尚嵩 the old read had left as a
disconnected root. `tests/test_s5_line_ends.py` pins plain / flush / stepped bars.

**Stage 5 Book 2:** 44 graphs → **1628 nodes**, 28 sus / 85 grid warnings, 37 empty
phantom nodes in 12 multi-page graphs, 86 roots (44 sections + 42 seam orphans).
The v0 orphan-bridging (trace-right rule) was **never ported to v1** (`s4_merge_pages`
says so explicitly) — that is the seam-orphan count, not a parse regression.

**Stage 6 Book 2:** PP-OCR → `data/book2_names.json`, 1584/1628 names applied (44
blank = phantoms), 314 low-confidence, 0 overrides yet. 65 names came back as a
single char (truncations, e.g. 传 ×24); 宏羨 read as 宏美. Review pending.

**Stage 7 Book 2 — three things the Book 1 matcher never met**, each fixed in
`src/s7_stitch.py` with tests (`tests/test_s7_stitch.py`), Book 1 output byte-identical:
1. Only a graph's own root (`{graph}_0`) is a duplicate section root. Seam-orphan
   roots (index ≠ 0) were being name-matched — 114_120_44 毓棋 would have been welded
   onto a 101_105 leaf by name collision. They now stay roots.
2. A section root may duplicate a NON-leaf: 4_5's 存学 repeats 0_3's own root.
   Leaves still win; inner nodes are the fallback.
3. Sections re-print a shared ancestor chain: 8_10 / 67_68 / 92_93 / 123_124 all open
   克宣 → 龙X → 万X, and 8_10/67_68 share 龙润. Merging only the root duplicated 龙润.
   `stitch_nodes` now **folds** repeated children recursively when the name match is
   unambiguous (one child per side, ≥2 chars). Result: 克宣 → [龙润, 龙滟, 龙淀, 珊龙],
   龙润 → [万邦, 万都], 存学 → 九舟 → 5 sons. 0 seam name mismatches.

Result: `data/book2_stitched.jsonl` — **1591 nodes, 51 roots** = 1 main lineage
(克宣's, reaching gen 16) + 42 seam orphans + 8 unresolved section roots. The 8 are OCR
variant pairs between root and canonical leaf (贞烈/贞列, 贞熊/贞能, 贞杰/贞木, 贞斗/贞升,
贞亮/贞光, 克太, 贞富, 贞年) — the OCR review will resolve them; then re-run
`apply_names` + `s7_stitch`. Regenerated `book1_stitched.jsonl` from the current
`book1.jsonl` (picks up the 宣J→宣 override; topology unchanged).

**Next:** William reviews Book 2 OCR (`scripts/qa/s6_ocr.py` — restart it, it skipped
Book 2 at startup for lacking v1 crops); port orphan-bridging to v1 (`src/v0/segment.py`
→ merge stage; ground truth in `docs/bridge-ground-truth.md` is v0-scaled); then Books 3/4.

**Bridging ported to v1 (later on 2026-09-14).** `s5_build_tree.bridge_orphans`
runs before parsing on multi-page graphs. Getting v0's trace-right rule to work on
the v1 scans took four more fixes, each found by tracing a wrong bridge to its ink:

1. **Stage 4 seam endpoints** (`s4_merge_pages.seam_endpoints`): 1-4 row ADF smear
   specks at page edges were read as line ends; at 11_17's p16|p15 seam two specks
   paired with real lines shifted page 15 down 388 rows (true shift 18). Now a seam
   end needs >= 5 rows and >= 4 inked columns (measured over all 90 Book 2 seams).
2. **Solid ink only** (`SPECK_MIN_ROWS = 4`): the bar trace hopped onto specks past
   the real bar end and the reconnection scan stopped at specks.
3. **Follow the bar through steps** (`trace_bar`): Stage 4 concatenates a page whose
   seam has no matched line end without vertical alignment, so a bar can step 7-60
   rows at a seam (69_82's gen-2 bar steps at 11 of 12 seams). The trace tracks the
   bar's row (edges, not run centres), and every hop/step is a fill candidate tried
   nearest-first -- this closes 69_82's 尚澜 step automatically, the case v0 fixed by
   hand with an L-connector.
4. **Targeted gate** (`bridge_resolves`): the count-only gate accepted a bogus
   34,000px bridge on 69_82 that knocked out an *unrelated* orphan. Orphans are now
   identified by their children's positions; a fill must remove the targeted orphan
   and orphan no new child.

Hairline nicks (<= 60px, no step) are recorded in `{stem}.nicks.json` (cyan in QA);
green (`{stem}.imaginary.json`) is only for cross-page connectors, per William's rule.

**QA review + post-fixes layer (evening).** William reviewed the parse QA and found
six things; each was traced to its cause and fixed through a new hand-verified
layer, `data/book2_fixes.json` applied by `src/s5_fixes.py` (delete / merge /
recrop, by provenance, re-applied after any Stage 5 run):
- p10: a vertical line fragment parsed as a nameless node → delete.
- p80/p82: 衍谟 and 尚澜 crops clipped at the top. Cause: the band read reports every
  child endpoint at the component's MAX row, so a shorter riser's endpoint lands
  inside the glyph below it (尚澜's riser ends at 1442, its neighbour's at 1496) →
  recrop; 尚澜 re-OCRs correctly (was 回澜).
- p106–108 and p133: orphan bars whose children belong to 尚恕 / 贞杰 → merge. (The
  106_113 bar's nick fills DO reach 尚恕's hang-line but also orphaned two other bars
  in the re-parse, so the targeted gate rightly refused them — a bridging limit.)
- p114/p118: hang-line pieces that don't line up produce a phantom holding the realimage.png
  node's children (毓援 → '商科', 毓棋, 毓揄) → merge into the real node.
- p96/p98 (+p110/p117/p118): six 3-char names lost their last char (…子). Cause: the
  Book 2 crops were generated 2026-09-01 with the OLD ~300px bottom whitening; the
  current `whiten_margins` (fixed 2026-09-13) leaves them 12px clear — but only 12px:
  `trim_borders` cuts the bottom at page-bottom−90px, ~34px above the inner border, so
  the 200px whitening actually reaches ~234px above the border. Fix in OCR review now;
  re-run Stage 3+ (or lower `WHITEN_BOTTOM`) later.
Result: 1586 nodes, stitched **1549 nodes, 9 roots** (main lineage + the 8
OCR-mismatched section roots), **0 orphans**. QA: only detected endpoints are circled
(a leaf's inferred `bot` is speck-prone); wide overlays capped at 16k px; full-res
crops per fill.

**Result (v1 Book 2, Stage 5→7, before post-fixes):** 1592 nodes; empty phantom bars **37 -> 2** (both
cross-graph: their bar runs to the graph's right edge), roots **86 -> 50** (44
sections + 6 cross-graph orphans), sus 28 -> 2, grid 85 -> 15. 16 green bridges,
17 nick fills. Per graph vs `docs/bridge-ground-truth.md`: 11_17 4/4, 22_23 1/1,
28_30 1/1, 31_35 1/1, 36_52 4/4 (the over-extended one now seam-to-first-ink),
58_62 1 bridge + 1 nick (= "1 correct, 1 scan-fill"), 69_82 2 + 尚澜's step,
114_120 1/1, 121_122 and 126_128 0 (as ruled). No manual bridges needed.
OCR re-run (285 low-conf, 1584 applied); stitch → `book2_stitched.jsonl` **1555
nodes, 15 roots**: the main lineage (1300 nodes under 存学) + the same 8
OCR-mismatched section roots + 6 cross-graph orphans (106_113_5, 商科, 133_133_1,
毓揄, 毓棋, 8_10_9). Website still shows only Book 1.

## Era 11 — Book-3 prep fixes; whitening anchor, bbox speedup, and two stale diagnoses retired (2026-09-14 → 2026-09-15)

Working through the `docs/handoff.md` item list, all without re-running Book 2
(frozen at `books/book2/frozen_2026-09-14`; every change validated in memory
against those graphs). The theme of the era is that **not every handoff item was a
real bug** — two dissolved on inspection, one would have made things worse — and
the wins came from measuring before cutting.

**Stage 3 whitening anchored to the inner border.** The bottom whitening had been
measured from the trimmed page edge, but `trim_borders` cuts ~34px *above* the
inner border, so the fixed 200px band reached only ~234px above the border — 8px
of clearance from the lowest-hanging 3-char names on p42, which then lost their
last …子 character. Fix (`src/s3_segment.py`): `bottom_inner_border_row` detects
the two-band printed frame at the page bottom (outer line flush with the edge,
inner line ~40px up) and returns the inner one; `whiten_margins(a, bottom_anchor=)`
whitens from `anchor − WHITEN_BOTTOM_FROM_BORDER` (=180) down — a band fixed
*relative to the border*, not the trimmed edge. A 134-page in-memory scan confirmed
it: min clearance **8px → 57px**, every page finds a real 2-band frame (border 56–60px
above the page bottom, no fallbacks), and — the safety property — **no page has
glyph ink inside the whitened band**. The six cut …子 names are safe on the next
re-run. (Book 2's crops are already regenerated with this; downstream is not.)

**`find_lines` component extraction — the real bridging-speed bottleneck.** The
handoff blamed bridging's per-candidate whole-graph re-parse and suggested caching
the labelling. Profiling said otherwise: `parse_graph` was 12.9s on 106_113, and
**11.6s of it was `find_lines`** — specifically `np.where(labels == label)` run
*once per kept component*, each a full pass over the 5924×22998 (136M-element) label
array (~9s across 32 components). Slicing each component out of its own
`connectedComponentsWithStats` bounding box gives **identical pixel sets ~60× faster**:
parse_graph **12.9s → 4.1s**, `bridge_orphans` on 106_113 **78s → 23s**, Book 2
Stage 5 ~50min → ~15min. No caching, no behaviour change, and it speeds *every*
parse — matters most for Book 3's big graphs. The lesson: the slow thing was an
array scan hiding inside the parse, not the re-parsing the handoff pointed at.

**Two stale diagnoses retired by reproduction.** (a) The "106_113 gate refusal" —
supposedly the orphan bar couldn't bridge because doing so orphaned two other bars
— no longer happens: `bridge_orphans` on the raw frozen graph auto-resolves the
sole orphan with a single nick and reproduces the hand-fixed parse **node-for-node**
(100 nodes, 0 unmatched, 0 child-set mismatches). The `find_lines` hairline-nick
refill (Era 10's 1a9917b) welds the broken bar before orphan detection, so the gate
now passes. Consequence: `data/book2_fixes.json` reduces to **just `delete 8_10_9`**
on a re-run (both the 106_113 merge and everything else are now automatic). (b) The
Stage 4 adjacency guard was left as an in-flight commit that *raises* on a
page-number gap rather than silently welding tree pages across missing biography
pages (Books 3/4's 10_138 / 202_247 bogus graphs) — a loud failure is correct here,
because a gap means an upstream start-detection miss, not a wide subtree.

**A regression avoided: the name window stays at ±120.** The handoff wanted
`NAME_HALF_WIDTH` widened to ±160 plus a streak-tolerant column trim. An in-memory
sweep of all frozen Book 2 graphs killed the idea: **173 boxes grow ≥15px at 160**,
a cluster to exactly 320px (the full 2×160 window) as the any-ink trim starts
pulling in the neighbour ~300px away (8_10_8 → 430px, 8_10_9 → 540px), and it
doesn't even fix the wide glyphs — 114_120_54 毓塘 just re-clamps at the wider edge.
Meanwhile the marquee cases (毓援/毓棋/毓塘, 8_10_3, 58_62_68) are **already rescued
by the walk-out** — only ~5 marginal nodes stay clamped. Widening safely would
require the risky streak-trim (a column analog of `_line_only_rows`, must not erase
thin real strokes) for a small payoff on a frozen book. Verdict: leave it at 120;
revisit only if Book 3 shows cut wide glyphs the walk-out misses.

**Small landings.** `scripts/qa/s5_fixes.py` — a data-driven post-fixes QA page
(one section per delete/merge/recrop from `data/{book}_fixes.json`, self-contained
base64), replacing Era 10's ad hoc `books/book2/qa/fixes.html` generator; it
surfaces the 毓塘→毓搪 mis-OCR on its own. And `s5_fixes` merge now moves the real
node's sidecar `bot` to the phantom's, so the QA red edge no longer fans ~60px off
the hang-line (114_120 毓援) — QA-overlay cosmetic only.

Net for the era: five items landed (whitening, bbox speedup, adjacency guard, QA
tool, cosmetic), two were stale (106_113, and the fixes-file pruning that follows),
two were correctly *not* done (window widen = regression; Stage 4 shift carry-over =
bridging already handles it). Nothing required a Book 2 re-run. Tests 66 green;
commits on `main`, William pushes.

---

## Era 12 — Bio geometric parse, stage 1 (border crops) for Books 3 & 4 (2026-09-17)

Started the geometric biography parser (design:
`docs/specs/2026-09-17-bio-geometric-parse-design.md`), its own sub-pipeline
(`src/bio/sN_*.py` → `books/N/bio/N_...`, renumbered from 1), superseding the
text-first bio linker.

**Stage 1 (`s1_crops`)** — crop the printed frame off every bio page. Verified the
book alternates one graph run then its bio section 1:1 (Books 3 & 4 both strict
`GBGB`); derive a section's first page from page *ordering*, not the unreliable
`follows_graph`. Frame geometry is William's **x/y/z model**: ~40px thin rule (x) all
sides, ~220px label band (y) on the thick side (left if even / right if odd), ~730px
右 marker band (z) on a section's first page. Detect each rule by **longest continuous
run** (not ink density — faded/broken rules keep a 2000px+ run while text tops ~200px;
density missed b3 p30/p60 and b4 empty-cell marker bands), cut just inside the
innermost rule, per-side search window capped to that side's max frame width.
`check_frame` sanity-checks each trim vs x/y/z (100px tol) + a no-leftover-line edge
invariant. QA = red kept-region box on the untouched page. **Book 3: 251 pages, Book
4: 183, both 0 flags.**

**Data gap found:** Book 4 p214 (庆炆房系 branch) has a marker band but its tree-graph
page is missing from the parse (graph 210 is 庆炎-only, no 庆炆 graph 206–215; page
numbers contiguous → dropped upstream). Skipped via `SKIP_PAGES`; the missing graph is
an upstream graph-pipeline gap to fix later.

## Era 13 — Bio geometric parse, stage 2 (RTL section merge) for Books 3 & 4 (2026-09-17 → 2026-09-18)

**Stage 2 (`s2_merge`)** joins each bio section's contiguous crop pages into one wide
**right-to-left** image (section-first page rightmost → block order = the tree's
RTL/eldest-first traversal), ready for the generation-band split. Pages are aligned on
their **top** horizontal rule (padded above to a common y); the **bottom** rule is the
check — after alignment every page's bottom rule must sit within `BOTTOM_RULE_TOL` of
the section median or the merge hard-fails. `find_rules` reuses stage 1's run-length
detection with `BIO_RULE_YFRAC` as a search hint (two passes: estimate the top-margin
shift, then locate each rule tightly), recovering faded/sparse-page rules density
missed and tightening the b3 0_1 aligned-bottom spread 32px→8px. Before joining,
`trim_sides` removes up to 80px of whitespace per side (stop at >2× the page's
whitespace baseline), keeping a 20px pad when it stops at content — so a bio split
across a seam no longer faces a doubled ~200px gap — and strips any residual vertical
border-rule sliver stage 1 left at an edge (detected by a run through the top
whitespace band; the p5|p6 seam-stub fix).

**Both books merge, 0 hard-fails: Book 3 = 15 sections, Book 4 = 132.** Book 4 is
structurally different — it interleaves finely (`GBGBGBGBBBB…`), so most sections are a
single bio page and it fills only the top 2 generation-bands (a 4-generation book).
Running it surfaced one genuine per-page scale difference (207_209's p209 extracted
~1.5% taller → 59px bottom-rule drift), so the alignment tolerance was set from the
data to 70px (all b3+b4 sections drift ≤55px; a mis-extracted page would drift
hundreds). Book 4 p214 is excluded from its section via s1's `SKIP_PAGES`. Output →
`books/{3,4}/bio/2_merged/` for stage 3. 18 stage-2 tests.

---

## Era 14 — Bio stage 4: per-person field OCR (ensemble) for Book 3 (2026-09-18)

**Stage 4 (`src/bio/s4_ocr.py`)** reads each per-person block and extracts its fields —
**P0 = sons** (`生子…名 <sons>`, the stitch signal). Two independent readers, reconciled:
- **Paddle PP-OCRv5 detect-then-resort** — run Paddle's detector, discard its reading
  order, re-sort the per-character boxes (`return_word_box`) into RTL columns
  (top-to-bottom within a column). A bake-off showed whole-block Paddle scrambles reading
  order (fatal for sons) and naive vertical-projection column splitting collapses on dense
  blocks where columns touch (ADF smear); detect-then-resort fixes both. Yields char boxes
  for three structural QA checks (inter-column pitch gap, intra-column contiguity, column
  fill ratio) that flag OCR drops with no ground truth.
- **A vision Claude subagent** (`s4_vision_prompt.txt`) transcribes the crop semantically,
  one line per column RTL (line 1 = the *horizontal* father header `子之X`, line 2 = the
  vertical bold name, 3+ = prose). Catches exactly Paddle's blind spots (header + short
  name column).

`reconcile` merges them: sons = union with per-son `agreed` flags; a single-reader son
that is not a real tree child (garbled glyph / trad-simp variant) is demoted to a QA flag
against the tree oracle, keeping the union honest. Block *k* in gen *g* ↔ tree node *k*
(the stage-3 count gate guarantees the mapping); the tree supplies the identity name.

**Validated on section 2_9 (subgraph 0_1 = 传禄):** the union of gen-5 fathers' sons
reconstructs **8 of 9** gen-6 tree names (`庆林 庆鸿 庆亮 庆海 庆荣 庆华 庆财 庆铭`), 0
spurious extras (Paddle-alone: 5/9 with 2 junk extras). The one miss, `庆粮`, is a genuine
**bio↔graph conflict**, not an OCR failure: the tree assigns 庆粮 to father 宪烘, but 宪烘's
biography lists `生子二名 庆财 庆铭` — evidence the graph mis-assigned that edge, exactly the
kind of discrepancy the children-name signal exists to surface for stitching. Output →
`books/book3/bio/4_ocr/{stem}.json` (per-block sons/name/father/daughters + both readers +
`qa_flags` + a `validation` gate); QA overlays → `books/book3/qa/bio_s4/`. A `tight_crop`
shim narrows the still-wide stage-3 blocks (no-op once stage 3 emits tight crops). Design
`docs/specs/2026-09-18-bio-stage4-field-ocr-design.md`, plan
`docs/superpowers/plans/2026-09-18-bio-stage4-field-ocr.md`. Validated by running on real
data (no unit tests, per repo convention).

---

## Era 15 — Bio stage 4 finalized + full-book ensemble OCR (Books 3 & 4) (2026-09-19)

**Scope change: stage 4 is now a PURE per-crop OCR step.** No tree, no generation-based
validation, no node mapping — that moves to a future **stage 5** (father/son cross-check +
stitch into the graph). Stage 4 reads s3_post's finalized combined
`books/{book}/bio/3_segment/blocks.jsonl` (the post-QA index s3_post writes for stage 4) +
the tight `{id}.png` crops, and emits `books/{book}/bio/4_ocr/{stem}.jsonl` — one
**lossless** record per block: BOTH readers' raw output preserved verbatim
(`raw.paddle.columns` + char boxes, `raw.vision.text`) plus best-effort structured fields
kept **per-reader, unmerged** (`sons/name/father_char/daughters/birth`, each
`{paddle, vision}`) + structural `qa_flags`. The branch was rebased on latest `main`
(S3 fully done incl. Book 4); the `tight_crop` shim is now a no-op on the tight crops.

**Ran the full ensemble on both books** (Paddle-only first for a lossless baseline, then a
vision pass layered on): **Book 3 = 620 blocks / 15 sections, Book 4 = 307 blocks / 132
sections.** Paddle ran as two background jobs. The vision pass used a **write-to-file
subagent pattern** (~40 Claude vision subagents, each transcribing a section or chunk and
writing its `{id: transcript}` JSON to disk, replying only "done N") to keep transcripts
out of the orchestrator's context; a helper (`apply_vision.py`) merges the transcripts and
re-runs `ocr_book` with `vision_texts`. Vision fixes Paddle's systematic noise (reversed
2-char reads like 伟庆→庆伟, stray digits 庆鸿3, garbled headers) and reads the horizontal
father header + short bold name that Paddle drops; Paddle confirms the clean reads and
supplies the char-box QA. On the 2_9 re-run with finalized crops the gen-5→gen-6 son set
reconstructs **9/9** (the old 8/9 miss, 宪烘/庆粮, was an artifact of pre-final S3
segmentation). Coverage: Book 3 vision ~617/620 (85_133 chunking dropped 3), Book 4 307/307.

**Known issues (for stage 5 / S3):** `85_133` vision 112/115; `263_271_4_3`/`_4_4` look
like a duplicated crop (S3 over-segmentation). Neither loses data — Paddle covers all 927
blocks. Next: **stage 5** (validate father/son vs the tree, stitch edges into the graph).

---

## Era 16 — Bio stage 5: link bio entries to graph nodes + fold into combined tree (2026-09-19)

**Stage 5 (`src/bio/s5_link.py` + `s5_fold_combined.py`)** associates each stage-4 bio
block with its tree-graph node, per subgraph. Within a (stem, generation) it matches bio
blocks to tree nodes with a cascade: **exact name** (block vision-name == node name) →
**fuzzy name** (≤1 differing char, catches OCR/trad-simp variants) → **sons overlap**
(block sons ∩ node's children names — an independent edge signal) → **positional**
(remaining blocks ↔ nodes in RTL/eldest order, only when counts line up). Each matched node
gets a lossless `bio` field (name_ocr, father_char, sons, daughters, birth, qa_flags, both
raw reads).

**Linked: Book 3 615/620 (99%)** — 490 exact / 78 fuzzy / 18 sons / 29 positional;
**Book 4 301/307 (98%)** — 285/11/1/4. Output `data/{book}_bio_linked.jsonl` +
`_bio_link_report.json`. Then `s5_fold_combined` keys each linked bio by
`{book-tag}:{provenance}` (e.g. `b3:0_1_0`) and folds it onto the renumbered combined
graph → **`data/tree_bio.jsonl`: 875 combined nodes carry a bio** (574 b3 + 301 b4; the
~41 b3 shortfall are nodes cross-stitch pruned from `tree.jsonl`). The 5–6 unmatched
blocks per book are in the sections with the known S3 chunk gap / dup crop (85_133,
18_63, 263_271). Next: use the bio father/sons as evidence to *correct* graph edges
(the 宪烘/庆粮-style conflicts) and to strengthen cross-book stitch.

---

## Pipeline status (snapshot)

Two pipelines: v0 (old glass scans, `src/v0/`) reached OCR for books 1–2; the v1
rebuild (new ADF scans, `src/`) is redoing every stage from clean scans.

**v1 pipeline (current)** — NOTE stages 6/7 swapped: **6 = OCR, 7 = stitch**.

| Book | 1 extract | 2 classify | 3 crop | 4 merge | 5 tree | 6 OCR | 7 stitch |
|------|-----------|------------|--------|---------|--------|-------|----------|
| 1 | done (17) | — all-tree | done (17) | done (14 graphs) | **done (163 nodes)** | **done (23 overrides, 2 flags)** | **done (150 nodes, 1 root, 56 gens)** |
| 2 | done (134, crops from 09-01) | — all-tree | done (134) | done (44 graphs) | **done (1586 nodes; bridged + post-fixes, 0 orphans)** | **done (285 low-conf, review pending)** | **done (1549 nodes, 9 roots: main + 8 OCR-mismatched)** |
| 3 | done (292) | done (51 graph / 241 bio) | done (51) | not started | — | — | — |
| 4 | done (317) | done (147 graph / 170 bio) | done (147) | not started | — | — | — |

Book 1 is **fully done end-to-end** → `data/book1_stitched.jsonl`, published to the
website. **Book 2 runs end-to-end with bridging**; what keeps it from one tree is the
OCR review (section roots whose OCR differs from their canonical leaf) and 6
cross-graph orphans (parent on an adjacent graph).

**v0 pipeline (prior, archived):** books 1–2 reached tree + OCR (book 1 stitched by
hand); superseded by the v1 rebuild above.
