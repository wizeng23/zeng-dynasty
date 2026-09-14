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

---

## Pipeline status (snapshot)

Two pipelines: v0 (old glass scans, `src/v0/`) reached OCR for books 1–2; the v1
rebuild (new ADF scans, `src/`) is redoing every stage from clean scans.

**v1 pipeline (current)** — NOTE stages 6/7 swapped: **6 = OCR, 7 = stitch**.

| Book | 1 extract | 2 classify | 3 crop | 4 merge | 5 tree | 6 OCR | 7 stitch |
|------|-----------|------------|--------|---------|--------|-------|----------|
| 1 | done (17) | — all-tree | done (17) | done (14 graphs) | **done (163 nodes)** | **done (23 overrides, 2 flags)** | **done (150 nodes, 1 root, 56 gens)** |
| 2 | done (134) | — all-tree | done (134) | done (44 graphs) | **BLOCKED at graph 67_68** | — | — |
| 3 | done (292) | done (51 graph / 241 bio) | done (51) | not started | — | — | — |
| 4 | done (317) | done (147 graph / 170 bio) | done (147) | not started | — | — | — |

Book 1 is **fully done end-to-end** → `data/book1_stitched.jsonl`, published to the
website. **Book 2 is blocked at Stage 5** on a single stepped-bar component in graph
`67_68` — see the handoff note below.

**v0 pipeline (prior, archived):** books 1–2 reached tree + OCR (book 1 stitched by
hand); superseded by the v1 rebuild above.


## HANDOFF (2026-09-14) — Book 2 Stage 5 blocker: graph 67_68 stepped bar

**Task:** run Stage 5 (`src/s5_build_tree.py`) for Book 2. It crashes on exactly
**one** component, in **graph `67_68` only** (all 34 earlier graphs parse fine, and
`67_68` is the sole stepped-bar case in the whole book — confirmed by a full scan):

```
ValueError: 67_68.png: expected exactly one parent endpoint, got [(3894, 381), (3894, 2257)]
```

**What it is (verified — render `books/book2/4_graphs/67_68.png`, rows 3700–4650,
cols 250–2450):** a **stepped sibling bar**. Parent **宏羨** (top right); its
hang-line comes down (col ~2257), the bar branches off it and runs high-left over
children **闻评** (col 381) and **闻瑛** (col ~1600), then **steps down** and 宏羨's
line continues to its own child below. The component is rows 3894–4559, cols
381–2264. `find_line_ends` (`s5_build_tree.py:266`) reads the **top band** (within
`end_threshold=150` of `min_x=3894`) and `remove_adjacent` collapses it to TWO
endpoints — col 381 (child 闻评's riser-top, at the bar's high-left corner) and col
2257 (宏羨's real parent hang-line) — because the stepped bar leaves a 639px gap in
the top band (cols 381–1618, then 2257). The existing "fan-out bar flush with the
top" collapse (lines 301–304) assumes a *single-level* continuous bar and so
returns 2, tripping the `len(parents) != 1` assertion.

**Which endpoint is the real parent = col 2257.** Two independent discriminators
both agree:
- **Only col 2257 has ink continuing ABOVE the component** (rows 3600–3893, toward
  宏羨). Cols 381 and 1600 have nothing above — they hang *down* from the bar.
  (This is the truly reliable signal but needs the full image, not just the
  component.)
- **Component-only:** col 2257's vertical line **reaches the component's max row**
  (4559 = max_x — it's the through-line: parent above → continues to a child
  below); col 381 stops early (row 4227, a pure child riser).

**Decision handed to next session (William deferred it):** how to fix
`find_line_ends`. Options weighed:
1. **Through-line pick (component-only):** when the top band yields multiple
   endpoints, choose the parent as the endpoint whose vertical line reaches the
   component bottom; treat the others as children. No image needed. **Recommended
   — simplest, and it's 1 component in the whole book.** Verify the resulting parse
   of 67_68 matches the eyeballed tree (宏羨 parent of 闻评/闻瑛 + its own lower child).
2. **Ink-above (needs image):** thread the graph image into the parse and pick the
   endpoint whose column has ink above the component. Most faithful, but a
   signature change to `find_line_ends`/`parse_graph`.

**After the fix,** finish Book 2: re-run Stage 5 (v1 — the existing `book2.jsonl`
is the STALE v0 parse, name_images path `books/book2/names/`, NOT v1 `5_names/`),
then Stage 6 OCR, then Stage 7 stitch. **Book 2 has no oracle and no hand merges —
it will be the first real test of the automated name-matcher `find_merges` on a
fresh book.** Watch for name collisions (two people sharing a name → the matcher
picks the nearest earlier graph; verify that's right).

**Repo state at handoff:** on `main`, pushed to origin (36 commits incl. all Book 1
work + website). Working tree clean except this doc + any in-progress s5 edits. Env:
`PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python`. The 17-page graph `36_52` makes
full Book-2 s5 runs slow (~minutes) — run in background.
