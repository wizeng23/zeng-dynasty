# Progress

Status of each book through the pipeline stages. See `pipeline.md` for what
each stage does.

| Book | 1: spreads→pages | 2: pages→graphs | 3: graphs→tree | 3.5: OCR | Notes |
|------|------------------|-----------------|----------------|----------|-------|
| 1    | **done (src/)**  | **done (src/)** | **done (src/)** | not started | Reproduced: 100% topology match to old, RTL-fixed. `data/book1.jsonl` |
| 2    | done (cropping)  | **done (src/)** | **done w/ caveats (src/)** | not started | 45 graphs, 1763 nodes. No oracle — verified structurally. Stitching gap amplified (181 subtrees). See Book 2 result. |
| 3    | not started      | —               | —              | —        | Interleaves biography + tree pages |
| 4    | not started      | —               | —              | —        | Same as Book 3 |

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
- [ ] Parse Book 2
- [ ] Website (Next.js + d3-in-React)

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
8. **Book 2 warnings are dominated by benign seam-fragment artifacts, EXCEPT 6.**
   All 169 sus-node warnings are height 10–35px broken-line stubs (none
   too-tall); 221/227 grid warnings are short drops (<280px) tied to those
   stubs. **The 6 exceptions are long drops (~400px), all in `114_120.png`** —
   the only candidates for a genuine wrong-parent mis-merge. **Eyeball
   `books/book2/graphs/114_120.png`** to confirm those 6 aren't real topology
   errors.
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

11. **Book 1: 149 grid-consistency warnings, all 300–332px drops (benign).**
   The default grid band is `gen_row_max=260`; 149 Book 1 edges drop 300–332px,
   just over it. Topology matches old 100%, and the drops cluster tightly (not
   scattered outliers), so this is the default band being marginally tight for
   Book 1's real row geometry, **not** mis-merges. Left as-is rather than
   retuned overnight — a threshold nudge (`gen_row_max` ~340) would silence them
   but is a judgment call for William. The `bf5c03e` message wrongly said "zero".

## Next up (original plan)

1. Rewrite the pipeline into `src/` scripts, reproducing the Book 1 result, then applying the RTL fix.
2. Verify rewritten Book 1 output against `data/book1_golden.jsonl`.
3. Finish Book 2 (stages 2–3).
4. OCR (stage 3.5).

## Known bugs carried over from old pipeline (fix in rewrite)

- `merge_graphs` vstack sign bug (fixed in WIP commit `fc8a2b3`, re-apply).
- Book 2 stage-2 wrote merged graphs to `book1/graphs/` (copy-paste bug).
- `sort_nodes` was left-to-right; must be right-to-left / eldest-first.
- `shrink_page` has an unresolved commented-out `find_shrink_start` experiment.
