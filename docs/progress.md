# Progress

Status of each book through the pipeline stages. See `pipeline.md` for what
each stage does.

| Book | 1: spreads→pages | 2: pages→graphs | 3: graphs→tree | 3.5: OCR | Notes |
|------|------------------|-----------------|----------------|----------|-------|
| 1    | done (old)       | done (old)      | done (old)     | not started | Golden data verified: `data/book1_golden.jsonl` |
| 2    | done (cropping)  | not started     | not started    | not started | Old Cell 6 had a bug writing graphs to book1/ |
| 3    | not started      | —               | —              | —        | Interleaves biography + tree pages |
| 4    | not started      | —               | —              | —        | Same as Book 3 |

## Current state (post-restructure, 2026-08-17)

- Repo restructured: everything pre-restructure archived in `old/`; survivors
  promoted (raw scans, golden data, Node schema → `src/model.py`, web viewer).
  See `specs/2026-08-17-repo-restructure-design.md`.
- **Golden Book 1 data corrected**: RTL age-order renumber (BFS-by-generation),
  `children` eldest-first, `father` field populated + renamed from `parent`,
  generations 1-indexed.

## Next up

1. Rewrite the pipeline into `src/` scripts (`extract_pages.py`, `segment.py`,
   `build_tree.py`), reproducing the Book 1 result, then applying the RTL fix.
2. Verify rewritten Book 1 output against `data/book1_golden.jsonl`.
3. Finish Book 2 (stages 2–3).
4. OCR (stage 3.5).

## Known bugs carried over from old pipeline (fix in rewrite)

- `merge_graphs` vstack sign bug (fixed in WIP commit `fc8a2b3`, re-apply).
- Book 2 stage-2 wrote merged graphs to `book1/graphs/` (copy-paste bug).
- `sort_nodes` was left-to-right; must be right-to-left / eldest-first.
- `shrink_page` has an unresolved commented-out `find_shrink_start` experiment.
