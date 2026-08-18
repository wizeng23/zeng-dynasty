# Progress

Status of each book through the pipeline stages. See `pipeline.md` for what
each stage does.

| Book | 1: spreads→pages | 2: pages→graphs | 3: graphs→tree | 3.5: OCR | Notes |
|------|------------------|-----------------|----------------|----------|-------|
| 1    | **done (src/)**  | **done (src/)** | **done (src/)** | not started | Reproduced: 100% topology match to old, RTL-fixed. `data/book1.jsonl` |
| 2    | done (cropping)  | not started     | not started    | not started | Old Cell 6 had a bug writing graphs to book1/ |
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
  `[3,4,5]` (old, LTR). `father` replaces `parent`. Zero verify/grid warnings.
- **new vs golden**: overlap region (graph `0_0`, gens 1–6, 8 nodes) is
  **byte-identical** to golden. Divergence beyond that = stitching gap (below).

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
