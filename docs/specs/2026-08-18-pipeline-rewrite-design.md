# Pipeline Rewrite — Design & Algorithm Review

**Date:** 2026-08-18
**Status:** Draft, pending review
**Author:** William Zeng (with Claude Code)
**Follows:** `2026-08-17-repo-restructure-design.md` (restructure now complete)

Rewrite the Colab pipeline (`old/book_parser.ipynb` + `old/utils.py`) into
clean `src/` scripts. This spec records a review of the existing algorithm —
what to keep, what to fix — and the design for the rewrite.

## Verdict on the existing approach: keep it

The core approach is **right for this input** and should be preserved:

- **Deterministic pixel logic over ML.** The books are a fixed font, size,
  and grid layout. Completeness matters (a missed node = a missing ancestor),
  so "every black pixel should be accounted for" beats a probabilistic model
  that's ~95% right. The algorithm is either correct or loudly wrong.
- **Merge pages into one big graph, don't track dangling cross-page refs.**
  A subtree's line-graph runs across many pages (e.g. `36_52.png` spans 17).
  Physically stitching page images at the seam (a few pixels) lets the
  connected-component parser operate on one complete tree with **no special
  node type and no stateful cross-page matching**. This moves all the
  complexity into ~50 lines of `merge_graphs` instead of smearing it across
  the parser. Good tradeoff, not a hack.
- **Grid-packed OCR (planned).** Pack name-crops into a known grid before OCR
  so missing cells are *detectable* — the right defense against ancient
  characters absent from OCR training data / Unicode.

## Page structure (from inspecting real pages)

- **All pages normalize to a fixed 1300×1950 canvas** (same scanner, same
  book format). Trimmed working width ≈ 1150. This is a stable foundation.
- Names sit on a **rigid grid**: generation = vertical row, position =
  horizontal column, connected by straight horizontal/vertical lines.
- A **subtree-start page** has a thin vertical **text label** near the right
  edge (e.g. 侨公房系世系图 = "lineage chart of the 侨 house") plus a black
  tab marker. Continuation pages have no label.
- Book 1 names are mostly single characters; **Book 2 names are often two
  characters stacked vertically** (克宣, 龙润) — affects name-crop + OCR grid.
- Dangling lines run off the bottom/edge of a page onto the next (visible on
  book2/8: 贞荣, 贞富 continue below the page).

## Key finding: the threshold problem is structural, not per-book

The old `is_tree_start_page` hardcodes absolute pixel ranges
(`950 < start < 1050`, `1030 < end < 1100`). Measured label positions on
trimmed pages:

| Page | label start | end | span | end/width |
|------|-------------|-----|------|-----------|
| book1/11 | 896 | 962 | 66 | 0.84 |
| book1/12 | 874 | 931 | 57 | 0.81 |
| book1/13 | 1022 | 1079 | 57 | 0.94 |
| book2/8  | 983 | 1039 | 56 | 0.90 |
| book2/4  | 983 | 1040 | 57 | 0.90 |

The label's **span is book-independent** (~56-66px, one character column).
Its **absolute position varies page-to-page WITHIN Book 1** (874 / 896 / 1022)
— it is *not* a per-book constant. The old range `950 < start < 1050` was too
narrow and only accidentally fit some pages (book1/12 at 874 would fail it).

**Conclusion:** absolute column position is the wrong signal. Detect the label
**structurally**: a thin vertical text column (~one-character span) located in
the **rightmost ~15-20% of the fixed-width page** (end/width ≳ 0.80). This
generalizes across both books with no per-book config — the data proves the
variation is within-book, so per-book config would not even fix it.

Apply the same principle everywhere the old code hardcodes pixel offsets:
prefer invariants (fixed canvas size, proportional regions, relative
geometry) over absolute magic numbers. Keep a per-book config object only as a
fallback for the rare constant that genuinely differs by book.

## Fixes to bake into the rewrite

Ranked by impact as we scale to 4 books / thousands of nodes:

1. **Structural / proportional thresholds** (above) — replaces inline magic
   numbers; the main reason Book 2 broke.
2. **Replace `find_lines` hand-rolled pixel BFS with
   `cv2.connectedComponentsWithStats`** — same output, returns bounding boxes
   for free, C-speed (the `36_52` mega-graph is slow in pure Python), less
   code. (Already used elsewhere in the old notebook.)
3. **Add a grid-consistency check** alongside `verify_nodes`. The real risk in
   the seam-merge is a *mis-merge*: connecting the wrong dangling line to the
   wrong child — a **wrong-but-valid** topology that raises no error.
   `verify_nodes` only catches malformed node geometry. A grid check (children
   are exactly one generation-row below their parent; sibling positions
   consistent) catches mis-merges. Golden data is the Book 1 oracle.
4. **Real logging + delete dead experiments.** Remove `shrink_page`'s
   commented-out `find_shrink_start`, the debug `print`s in
   `is_tree_start_page`, etc. (all preserved in git history).
5. **Carry the `merge_graphs` sign fix** (WIP commit `fc8a2b3`).
6. **Name the two Node types distinctly.** Keep them separate — they are
   different concerns — but rename: parse-time geometric node
   (`old/utils.py` Node: top/bot/children-objects) → `LineNode`/`GraphNode`;
   domain output node (`src/model.py`) → keep as the schema `Node`/`Person`.
7. **Fix `sort_nodes` to RTL / eldest-first + BFS-by-generation ID assignment**
   (matches the corrected golden data — see restructure spec).
8. **Parameterize by book** — fix the old copy-paste bug where Book 2 stage-2
   wrote merged graphs to `book1/graphs/`.

Not urgent but noted: Cell 8's node-merge is O(n²) with a restart-from-scratch
loop; once lines are components with endpoints, matching is a spatial join
(sort/index by position). Revisit if it gets slow.

## Rewrite structure

`src/` modules, each a clear I/O contract with a CLI entry point:

- `model.py` — the domain `Node` schema (done, promoted).
- `extract_pages.py` — stage 1: spreads → normalized pages.
- `segment.py` — stage 2: pages → merged subtree-graph images
  (structural label detection, seam-merge with the sign fix).
- `build_tree.py` — stage 3: graph images → JSONL + name crops
  (connectedComponentsWithStats, RTL/BFS ordering, grid-consistency check).
- `ocr.py` — stage 3.5 (later): grid-packed name images → Unicode.

A `Makefile`/`run.py` with per-book targets: `extract`, `segment`, `build`,
`serve`.

## Rewrite sequencing

1. **Reproduce Book 1 exactly**, module by module, faithful to the working
   result. No premature book-agnostic abstraction.
2. **Verify against `data/book1_golden.jsonl`** (topology + generations).
   This green regression is the gate.
3. **Generalize to Book 2** — introduce the structural-threshold and per-book
   config abstractions *here*, where a second data point makes them real.
   Finish Book 2 stages 2-3.
4. Two data points (Book 1 + Book 2) then inform Books 3-4.

## Overall plan sanity check

- **Algorithm over manual is justified**: hundreds of pages, thousands of
  nodes, ancient characters, OCR unavoidable for biographies anyway.
- **Missing safety net for Books 2-4**: Book 1 has golden data; later books
  need at least a small hand-verified sample per book to catch mis-merges,
  since thousands of nodes can't be eyeballed.
- **Books 3-4 are a different parser** (biographies interleaved with trees) —
  own milestone, not bolted onto this.

## Out of scope

OCR implementation, Books 3-4, dynamic website, daughters/mothers schema —
all tracked in `docs/milestones.md`.
