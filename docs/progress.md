# Progress

Status of each book through the pipeline stages. See `pipeline.md` for what
each stage does.

| Book | 1: spreads→pages | 2: pages→graphs | 3: graphs→tree | 3.5: OCR | Notes |
|------|------------------|-----------------|----------------|----------|-------|
| 1    | done (old)       | done (old)      | done (old)     | not started | Golden data verified: `data/book1_golden.jsonl` |
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

- [ ] `src/extract_pages.py` — stage 1
- [ ] `src/segment.py` — stage 2 (structural label detection, seam-merge)
- [ ] `src/build_tree.py` — stage 3 (connectedComponentsWithStats, RTL/BFS, grid check)
- [ ] Reproduce Book 1 → verify vs golden
- [ ] Parse Book 2
- [ ] Website (Next.js + d3-in-React)

_(Updated as work lands. Discrepancies logged below.)_

### Discrepancies / notes for William

_(none yet)_

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
