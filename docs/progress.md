# Progress

Status of each book through the v1 pipeline (7 whole-integer stages; **6 = OCR,
7 = stitch**). See `pipeline.md` for what each stage does, and `history.md` Era 9
+ the HANDOFF note for the latest detail.

| Book | 1 extract | 2 classify | 3 crop | 4 merge | 5 build_tree | 6 OCR | 7 stitch |
|------|-----------|------------|--------|---------|--------------|-------|----------|
| 1 | done | — all-tree | done | done (14 graphs) | **done (163)** | **done (23 ovr, 2 flag)** | **done → book1_stitched.jsonl (150 nodes, 1 root 点, 56 gens)** |
| 2 | done | — all-tree | done | done (44 graphs) | **BLOCKED: graph 67_68 stepped bar** | — | — |
| 3 | done | done (bio/tree split, human-verified) | done | not started | — | — | — |
| 4 | done | done (bio/tree split, human-verified) | done | not started | — | — | — |

**Book 1 is DONE end-to-end** and published to the website. **Book 2 is blocked at
Stage 5** on one stepped-bar component (graph `67_68`) — see `history.md` HANDOFF.

## Comparison oracles for the rewrite

- `data/book1_golden.jsonl` — hand-typed, ~100% correct, **59 nodes** (verified subset only, first ~gen 36). RTL/age-ordered, `father` field. **Primary correctness oracle.**
- `old/data/book1.jsonl` — old script's output, ~95% correct, **163 nodes** (full book 1). LTR-ordered, `parent`, empty names. **Full-pipeline oracle** — expect RTL-vs-LTR ordering diffs plus ~5% real errors.

Note: golden (59) is a subset of the full book (163). Match golden on its 59 for correctness; match old jsonl's 163-node topology (modulo RTL/LTR) for full-pipeline reproduction.

## Autonomous session log (2026-08-18, overnight)

Rewriting the pipeline into `src/` per `specs/2026-08-18-pipeline-rewrite-design.md`.
Objectives: reproduce Book 1 (match golden modulo RTL) → parse Book 2 → website if time.

- [x] `src/extract_pages.py` — stage 1 (bit-exact: 18 pages, 0.000% pixel diff vs old)
- [x] `src/imaging.py` — shared image helpers
- [x] `src/segment.py` — stage 2 (structural label detection, seam-merge) → 14 graphs
- [x] `src/build_tree.py` — stage 3 (connectedComponentsWithStats, RTL/BFS, grid check)
- [x] **Reproduce Book 1 → 100% topology match to old parse, RTL-fixed** ✅ checkpoint
- [ ] Cross-graph stitching (see discrepancy #1 — the one real gap vs golden)
- [x] **Parse Book 2** → 1763 nodes, parses where old code crashed ✅
- [x] **Website (Next.js 16 + React 19 + d3-in-React)** → `web-app/`, builds
      clean, all features verified live ✅ (see Website result below)

_(Updated as work lands. Discrepancies logged below.)_

### Book 1 reproduction result (committed `bf5c03e`)

`scripts/compare_book1.py` — re-runnable topology diff, RTL/ID-invariant.

- **new vs old: 100% topology match** (163/163 subtree signatures, identical
  14-subtree forest, identical out-degree dist `{0:78,1:43,2:29,3:9,4:2,5:1,8:1}`).
  RTL fix applied cleanly: node 2 children `[5,4,3]` (new, eldest-first) vs
  `[3,4,5]` (old, LTR). `father` replaces `parent`. (Correction: the `bf5c03e`
  commit message said "zero grid warnings" — inaccurate; Book 1 actually emits
  149 grid warnings, all drops in 300–332px, just over the 260px band. Topology
  still matches old 100%, so these are a marginally-tight default band, not real
  mis-merges. See discrepancy #11.)
- **new vs golden**: overlap region (graph `0_0`, gens 1–6, 8 nodes) is
  **byte-identical** to golden. Divergence beyond that = stitching gap (below).

### Book 2 result (2026-08-18, not committed — orchestrator commits)

First real generalization test: Book 2 is where the **old** pipeline broke
(`ValueError("More than one parent found")`, 56 components). The rewritten
`src/` pipeline parses it end-to-end. **No oracle exists** (old pipeline never
produced a valid Book 2 jsonl/graphs) — everything below is **structural**
verification, so the flagged items are the safety net.

- **Stage 2 (segment):** 135 pages → **45 graphs**. 44 tree-starts + 91
  continuations. Book 2 needed **only `num_pages=135`** in `BOOK_CONFIGS` —
  every structural label threshold transferred from Book 1 unchanged (label
  span 55–58px, end/width 0.83–0.92, right margin 95–192px — all inside the
  shared bounds, cleanly separated from continuations). Confirms the design
  spec's "label geometry is structural, not per-book" thesis.
- **Stage 2 completeness audit: PASSED.** Ink conserved (graph/post-crop ratio
  0.9973; every graph 1.0000–1.0042, diffs are the small positive seam
  connecting-line pixels). Every page 0–134 assigned to exactly one graph; **no
  orphan or double-covered pages.** One genuine interior seam gap
  (`69_82.png`, cols 136–150) — the *intentional* empty-seam concatenation for
  page 82, logged not silent (see #7).
- **New code (segment.py):** `_concat_top_aligned` helper + empty-seam guard in
  `merge_graphs` (Book 2 has a case Book 1 never hits — a parent line stops
  short of the page border while its child continues on the next page, exposing
  no dangling endpoint; old code crashed `IndexError` on `right_y[0]`).
- **Stage 3 (build_tree):** **1763 nodes**, 1763 name crops (all valid PNGs).
  **169 sus-node warnings**, **227 grid-consistency warnings** (see #8 —
  overwhelmingly benign seam-fragment stubs).
- **New code (build_tree.py):** the fix for where the old pipeline broke —
  `find_line_ends` now collapses the top band (within `end_threshold` of
  `min_x`, mirroring existing bottom-endpoint logic) to one representative when
  Book 2's 2-char stacked names push the fan-out bar flush to a component's top
  and expose two top corners as two parents. Resolves **all 56** Book 2
  fan-out bars → 0 multi-parent; touches **0 of 85** Book 1 components.
  Plus `get_name_image` robustness (guards zero-size crops from ~10px seam
  fragments) and `BOOK_CONFIGS["book2"]` with `gen_row_min/max=280/345`
  (2-char rows drop ~2× Book 1's single-char rows).
- **Structural checks all pass:** 0 child→father / 0 father→child back-ref
  mismatches across all 1763 nodes; every edge `child.gen == father.gen+1` (0
  violations); all roots gen 1; 0 unreachable nodes; out-degree
  `{0:1069,1:262,2:191,3:118,4:65,5:34,6:15,7:8,8:1}` (clean tree shape).
- **Regression guard PASSED (both stages):** `src/segment --book book1` → 14
  graphs, byte-identical pixels vs HEAD (`git diff` empty). `src/build_tree
  --book book1` → 163 nodes, topology byte-identical to HEAD, identical
  name-crop hashes. Book 2 generalization did not leak into Book 1.

### Website result (committed `7fa6cf8`)

Static-website milestone DONE. `web-app/` — Next.js 16 + React 19 + TypeScript +
Tailwind v4 + Biome + next-themes, d3 for layout. Mirrors the mckloset stack.

- **Architecture** (the web-dev concept): d3 computes the layout math
  (`d3.hierarchy`/`d3.tree`), React renders the SVG via JSX. Only d3-zoom touches
  the DOM (bound to the svg ref), writing transform into React state.
- **Three datasets**, switchable: golden (59 nodes, 1 lineage, real Unicode
  names — default) / book1 (163, 14 roots) / book2 (1763, 181 roots). Renders a
  forest honestly when >1 root (virtual super-root, no faked links); shows the
  name-image crop when Unicode name is empty.
- **Features**: click → path-to-root highlight (survives pan/zoom, stops at the
  node's own sub-root); detail panel (father/children eldest-first, ancestor
  chain, biography/notes); RTL eldest-on-the-right; dark mode; keyboard-accessible.
- **Verified live** (Playwright): build + Biome lint clean; screenshots in
  `scratchpad/web-shots/`. Fixed a real ThemeToggle hydration mismatch.
- `public/data` + `public/names` are gitignored generated artifacts
  (`bun run export-data` regenerates from `data/` + `books/*/names/`).
- Known limits: no pan-to-selection; parsed data is fragmented (pending
  stitching); parsed-book names are image crops (pending OCR). Old
  `web/index.html` kept as reference.

### Cross-graph stitching — Stage 6 (`src/stitch.py`)

William DID stitch Book 1 originally (by hand) — recovered `data/book1_merged.jsonl`
(150 nodes, 1 root) from git history (commit `d544cd8`) and archived it as
`data/oracles/book1_merged.jsonl`. The stitch code was never committed (only its
output), and Cell 12 of the old notebook is the literal `# TODO`.

- **What stitching does:** each Stage-2 graph's root (`{graph}_0`) is a *duplicate*
  of a person who appears as a *leaf* in an earlier graph (the subtree-start page
  repeats the parent name). Merging each duplicate into its canonical leaf
  collapses the 14-subtree forest into one connected lineage (163 → 150 nodes),
  then recomputes absolute generations (root=1, max 56) and reassigns BFS/RTL ids.
- **Matching is unsolved automatically.** Pure name-crop pixel-matching recovers
  only ~4/13 merges (graph 8_8's children rank as bad as 39th); no simple
  positional rule. William did the 13 merges by hand. So `src/stitch.py` uses an
  explicit per-book merge list (`BOOK_MERGES`), with `find_merges()` as the seam
  where an automated matcher plugs in later.
- **Verified (`scripts/verify_stitch.py`, AHU tree-isomorphism):**
  `old parse + 13 merges == oracle` ✅ (stitch logic is correct) and
  `stitch.py output == new parse + 13 merges` ✅ (faithful). The one expected
  diff: `new parse + 13 merges != oracle` — a **within-graph parse nuance**
  between the rewrite and the archived old run that only surfaces once connected
  (the new parse is AHU-identical to old as a *forest*). Not a stitch bug — see #12.
- **Website:** added a "Book 1 (stitched)" dataset — renders as **150 people / 1
  lineage**, the connected tree (scanned names, pending OCR). Screenshot
  `scratchpad/web-shots/6-book1-stitched.png`.
- **Next (experiment):** a real automated matcher — combine name-crop similarity
  + reading-order prior (canonical is in a recent graph) + grid column, or OCR
  the ~14 root names first. Then generalize stitching to Book 2 (181 → ~1 tree).

### Parse QA tool + grid-check recalibration (2026-08-20)

`scripts/qa_overlay.py` — visual parse verification. Per graph, overlays on the
graph image: RED name-boxes + RED parent->child edges (verify detection), and
BLUE page-seam lines + page numbers (verify assembly; numbers ascend right->left
per RTL). HTML index at `books/bookN/qa/index.html`. Output gitignored.

**It immediately caught a real issue:** Book 1's 149 grid-check warnings
(discrepancies #8/#11) were **100% false positives**. The grid check bounds the
parent->child generation *drop* (child.top - parent.top ~ one grid row), which is
~300-332px in BOTH books (scanner geometry, not name height). Book 1's old band
[60,260] measured the wrong quantity and flagged every real edge. Fixed: shared
default band [280,345] (commit `ec7ac89`); Book 1 -> 0 warnings, both jsonl
byte-identical. Real mis-merges now stand out.

### Book 2 QA overlays (2026-08-20)

Generated 3-panel QA for all 45 Book 2 graphs (`python -m scripts.qa_overlay
--book book2`; no oracle exists, so visual QA matters most here). Findings from
spot-checking:

- **Parse is fundamentally sound.** 2-char stacked names (克太, 龙滚, 兴渊…) are each
  boxed as one tall unit; every real name is captured. The 17-page `36_52`
  assembly flows unbroken across all 17 seams (root 贞雄 far right, branching
  left, page numbers ascending right→left) — the seam-stitch approach verified at
  max scale.
- **~5 cross-seam connector stubs (contained imperfection).** In graphs like
  `114_120`, a long horizontal connector line broken at a page seam leaves a tiny
  (~10-35px) phantom "joint" node that some children attach to instead of their
  true parent on the adjacent page. The CHILDREN are correctly detected and named;
  only their immediate parent-link routes through a stub. Not lost data — the
  known seam-fragment issue (#7-9), now visually confirmed and localized. A future
  fix would rejoin the broken connector across the seam (like stitching, but
  intra-graph).
- **Grid warnings are the intended signal now:** stub-fragment drops (<280px,
  below floor) + 6 legit tall-gen drops (402px in 114_120, verified benign). The
  band [280,345] no longer fires on normal edges (the Book 1 false-positive fix
  applies to Book 2 too).

### Book 2 seam-break defect — QUANTIFIED (2026-08-20)

The QA overlay (William spotted it on `0_3`/`4_5`) surfaced a systematic Book 2
defect: **163 empty phantom nodes** across the 19 multi-page graphs (0 in
single-page graphs — purely a seam issue). Concentrated in the big graphs:
`69_82` (40), `36_52` (29), `126_128` (22), `106_113` (16), `11_17` (11),
`58_62` (11), `83_87` (10). ~9% of Book 2's 1763 nodes.

**Root cause (confirmed by zooming the seam of `4_5` at col 1061 = page-5 width):**
a horizontal connector line that CROSSES a page seam is not re-aligned by
`merge_graphs`. The merge aligns the dangling *vertical* lines at the seam, but a
long horizontal through-line ends up offset a few px vertically across the
boundary and disconnected. The broken left end leaves a small downward tick that
`find_line_ends` reads as a *parent* endpoint — an empty node (no name sits
there) — and the real children attach to it instead of their true cross-page
parent. Children are always correctly detected/named; only the parent-link is
phantom.

This is the main Book 2 parse defect and the top fix target. Options:
(a) in `merge_graphs`, detect a horizontal line crossing the seam and align on
    it (not just vertical danglers) / bridge the small gap;
(b) post-process: drop empty (ink<30) nodes and re-attach their children to the
    line continuing across the seam.
Book 1 is unaffected (all single-page graphs; 0 empty nodes).

### Seam-merge coordinate bug FIXED (2026-08-20, William-diagnosed)

Root cause (found from William's zoomed seam image showing the connector going UP
from the left line's tip, leaving the two lines unjoined): `merge_graphs` applied
`best_alignment` by padding one side's top (sliding its ink down) and the other's
bottom, but the endpoint bookkeeping was wrong — it shifted `left_y` while g1's
ink stayed put and never moved `right_y`. So both endpoint lists pointed a few px
ABOVE their real lines; the seam fill drew a stray up-stub instead of joining
them. Stage 3 read the stub tip as a parent hang-point → nameless phantom node.

Fix (commit `fc0fdc9`): move each endpoint list by exactly its own side's ink
shift. **Book 2 empty nodes 165 → 89.** Verified on `4_5`: seam now shows a clean
2px bridge joining the lines. Book 1 byte-identical (only nonzero-alignment seams
affected).

**~89 empties remain — a SECOND, distinct cause** (not the alignment bug):
- 2 are in SINGLE-page graphs (`133_133`, `134_134`) — no seam at all. Cause: a
  wide fan-out bar whose parent hang-line rises off the top; `find_line_ends`
  picks the bar's left corner as a spurious top endpoint (a parse edge case in
  the 2-char-name / bare-fan-out-bar handling, not a merge issue).
- The rest are residual seam cases in the big graphs (`69_82`:22, `126_128`:18,
  `36_52`:15) — likely multiple seams per line or larger residuals.
Lower-value long tail; the systematic bug is fixed. Revisit if needed.

### Discrepancies / notes for William

1. **Cross-graph stitching is unimplemented (the one real gap).** The pipeline
   (new *and* old) emits **14 disconnected subtrees**, each capped at depth ≤6,
   one per Stage-2 graph. Golden is **one connected lineage**, gen 1→36. The
   subtree-local roots (gen 1 in each graph) are never linked to their true
   fathers in an earlier graph, and generations are per-graph-local rather than
   absolute. Where new and golden overlap (gens 1–6) they're byte-identical — so
   within-graph parsing is verified correct; only the join across graph seams is
   missing. **This is the top next step** to match golden's single-tree form and
   is what the website will want (absolute generations). New feature, not a bug.
2. **Generation semantics differ by design.** New = per-graph local (root=1 in
   each of 14 graphs, max 6). Golden = absolute (1→36). Decide target before OCR
   / website render.
3. **Names empty in new (and old); golden has them.** New carries `name_images`
   crops only. Reconciling to Unicode names is the Stage-4 OCR job.
4. **`is_tree_start` ratio threshold = 0.75, not spec's 0.80.** Book 1 page 10's
   label ends at end/width 0.777 and IS a real tree-start; 0.80 would misclassify
   it. Verify this holds for Book 2 during segmentation.
5. **Old book1 "36 graphs" was contamination.** `old/book1/graphs/` had 36 files
   because the old book2 cell wrote there too (the copy-paste bug). True Book 1
   output is 14 graphs. Confirmed by re-running the clean old book1 path.

6. **Book 2: 181 disconnected subtrees vs ~45 expected (one per graph).** Same
   root cause as #1 (cross-graph / within-graph stitching unimplemented),
   **amplified** by Book 2's larger multi-page graphs where dangling lines run
   off page bottoms and aren't rejoined. 24 of 45 graphs parse to a single clean
   root (ideal); fragmentation concentrates in the big graphs: **`69_82`**
   (14pp → 31 roots), **`36_52`** (17pp → 24 roots), plus `126_128`,
   `106_113`, `11_17`. This is a **join gap, not a topology error** — every
   node's back-refs and generation deltas are consistent. Not a regression;
   it's discrepancy #1 at Book 2 scale. **Top structural item to eyeball.**
7. **Book 2 `69_82.png` intentional empty-seam fragment (pages 80–82, 贞坚房系).**
   Page 82's node 尚澜 has a right-edge dangling stub whose parent line on
   page 81 stops short of that page's left border (genuine scan gap, left edge =
   0 ink). The new empty-seam guard stacked page 82 top-aligned **without a
   false connecting line**, leaving the 尚深 fragment disconnected for Stage 3's
   connected-components. Correct at Stage 2 (house kept intact as one graph),
   but **Stage 3 must not weld this fragment to a wrong parent** — worth a human
   eyeball since there's no oracle. All three pages correctly grouped; no subtree
   split or dropped.
8. **Book 2 warnings all benign — the 6 suspect ones VERIFIED (eyeballed).**
   All 169 sus-node warnings are height 10–35px broken-line stubs (none
   too-tall); 221/227 grid warnings are short drops (<280px) tied to those
   stubs. The 6 long drops (~402px) are all in `114_120.png` and I visually
   inspected them: they share **one parent** (row ~1869) with **6 evenly-spaced
   sibling children** (row ~2271) — the exact signature of a *correct* fanout,
   not a mis-merge (which would link a misaligned wrong node). The 402px gap is
   simply a legitimate generation row ~100px taller than Book 2's typical
   ~300px, tripping `gen_row_max=345`. The graph renders as a clean regular
   multi-generation tree (贞年 root → 尚敩/尚恵/尚忠 → …). **No topology error.**
   Optional: widen Book 2 `gen_row_max` to ~410 to silence these 6.
9. **Book 2 two-char stacked-name crops — spot-checked OK, not exhaustively
   verified.** `get_name_image` was extended for the 2-char vertical stacks
   (克宣, 龙润 style). Spot-checked 克庄, 行佑, 尚洪, 传绵, 克庄 across 4 graphs —
   all fully capture both characters, tightly trimmed, no clipping. But 1763
   crops were not all eyeballed; the ~10px degenerate seam-fragment crops (now
   guarded from crashing) are surfaced by `verify_nodes` rather than dropped —
   these will produce garbage OCR at Stage 4 and should be reconciled then.
10. **Code/data drift at `bf5c03e` — RESOLVED.** HEAD's committed
   `data/book1.jsonl` carried a `notes` provenance string (`{graph}_{index}`,
   e.g. `13_16_54`) that the committed *code* did not write (it emitted `""`).
   Root cause: `bf5c03e` committed the original workflow's data alongside a later
   code revision that had dropped notes-writing. **Fixed** by restoring
   provenance in `build_tree.py` (`notes = "{graph}_{local_index}"`). Book 1
   `book1.jsonl` now regenerates byte-identical to the committed HEAD; Book 2
   `notes` populated across all 45 source graphs. Provenance is genuinely useful
   for tracing a suspect node back to its source graph during stitching work.

11. **[RESOLVED — see QA tool note above] Book 1: 149 grid-consistency warnings, all 300–332px drops (benign).**
   The default grid band is `gen_row_max=260`; 149 Book 1 edges drop 300–332px,
   just over it. Topology matches old 100%, and the drops cluster tightly (not
   scattered outliers), so this is the default band being marginally tight for
   Book 1's real row geometry, **not** mis-merges. Left as-is rather than
   retuned overnight — a threshold nudge (`gen_row_max` ~340) would silence them
   but is a judgment call for William. The `bf5c03e` message wrongly said "zero".


12. **New rewrite parse vs the merged-oracle parse: a within-graph nuance.**
   `data/book1.jsonl` (rewrite) is AHU-identical to `old/data/book1.jsonl` as a
   *forest*, and `old + 13 merges == book1_merged` oracle exactly. Yet
   `new + 13 merges != oracle` (AHU differs at one spot ~depth 19). Since every
   merge's subtree size/out-degree/depth matches, the difference is a
   within-a-graph node arrangement that only becomes distinguishable once the
   subtrees are connected — a parse nuance between the rewrite and the archived
   old run, NOT a stitch bug (proven by `scripts/verify_stitch.py`). Worth a
   look if we later want the stitched tree to match the old merged artifact
   byte-for-byte, but the rewrite's parse is the one that reproduces old 100%.

## Next up (original plan)

1. Rewrite the pipeline into `src/` scripts, reproducing the Book 1 result, then applying the RTL fix.
2. Verify rewritten Book 1 output against `data/book1_golden.jsonl`.
3. Finish Book 2 (stages 2–3).
4. OCR (stage 7).

## Known bugs carried over from old pipeline (fix in rewrite)

- `merge_graphs` vstack sign bug (fixed in WIP commit `fc8a2b3`, re-apply).
- Book 2 stage-2 wrote merged graphs to `book1/graphs/` (copy-paste bug).
- `sort_nodes` was left-to-right; must be right-to-left / eldest-first.
- `shrink_page` has an unresolved commented-out `find_shrink_start` experiment.
