# Repo Restructure & Pipeline Rewrite — Design Spec

**Date:** 2026-08-17
**Status:** Draft, pending review
**Author:** William Zeng (with Claude Code)

## Goal

Reset the repo to a clean, well-organized, documented state so all future
development runs smoothly through Claude Code. The current state is a
year-old exploration: most logic lives in one Colab-style notebook
(`book_parser.ipynb`) with tuned-by-hand helpers in `utils.py`, plus many
intermediate image artifacts and abandoned experiments.

**Strategy (user's call):** Move *everything* currently in the repo into an
`old/` archive, then **promote** back to the top level only the assets that
have stood the test of time. Rewrite the notebook logic into clean, runnable
scripts. Document as we go. Then move on to Book 2.

Milestones (unchanged): (1) static website rendering all content, once all 4
books are parsed → (2) dynamic website allowing updates.

## Context: the pipeline as it exists today

Reconstructed from `book_parser.ipynb` + `utils.py`. Four stages:

1. **Spreads → pages.** Input `bookN/original/*.png` (two-page camera
   scans). Uses `remove_small_islands`, `get_corners`, `normalize_page` to
   split + deskew each spread into single pages → `bookN/pages/*.png`.
   (Per-book fudge factors for which half-page to skip first.)
2. **Pages → merged tree-graph images.** `trim_borders`,
   `is_tree_start_page`, `shrink_page`, `merge_graphs`. Detects subtree
   boundaries and stitches the line-graph across the pages a subtree spans →
   `bookN/graphs/{start}_{end}.png`.
3. **Graph images → structured tree + name crops.** `find_lines` (BFS
   connected components), `find_line_ends`, node merging, `infer_ends`,
   `sort_nodes`, `get_name_image` → `data/bookN.jsonl` (tree with empty
   `name`, `name_images` pointers), `bookN/names/{id}.png`,
   `bookN/trees/*.json`.
4. **(separate, manual) Spreadsheet → golden JSONL.** Hand-typed data in a
   Google Sheet → `zeng_google_sheet.csv` → `book1_golden.jsonl`. This is
   the *verified* Book 1 data, used to check stage 3's output. OCR (image →
   Unicode name) was never wired up, so the algorithmic JSONL still has
   names as images; the golden data has real names.

### Known issues found (to fix in the rewrite)

- **Traversal order was left-to-right** (`sort_nodes` sorts each generation
  band by ascending x). The book reads **right-to-left with eldest siblings
  on the right**, so the rewrite must sort **RTL / eldest-first**, and assign
  IDs **BFS-by-generation, right-to-left**. *(Golden data already corrected —
  see below.)*
- **`merge_graphs` sign bug** in the else-branch — fixed in WIP commit
  `fc8a2b3`, carry into rewrite.
- **Book 2 Cell 6 writes merged graphs to `book1/graphs/`** (copy-paste
  bug) — that's why `book2/graphs/` is empty. Rewrite parameterizes by book.
- `shrink_page` has an unresolved commented-out `find_shrink_start`
  experiment; debug prints left on in `is_tree_start_page`. Clean up.

## Already done (this session)

Committed on branch `wip-book2-and-golden-fix`:

- **Golden Book 1 data corrected** (commit `53d2a59`): renumbered
  RTL/age-order (BFS-by-generation), `children` arrays eldest-first, `father`
  field populated, `parent`→`father` rename, generations 1-indexed. Applies
  to `data/zeng_google_sheet.csv` and `data/book1_golden.jsonl`.
- Book 2 cropping WIP committed as a checkpoint (commit `fc8a2b3`).

## Schema conventions (canonical)

The `Node` model (currently `data/node.py`) is well-designed and kept, with
one rename:

- `id`, `name`, `name_images`, `generation`, **`father`** (was `parent`),
  `children`, `biography`, `notes`.
- **`father`, not `parent`:** the book contains only men so far. Daughters
  (and mothers) are a later schema evolution.
- **Sibling order right-to-left = eldest-first.** `children` arrays ordered
  eldest→youngest.
- **IDs BFS-by-generation, right-to-left.** Natural because the 4 books are
  split by generation.
- **`generation` 1-indexed** (点 = gen 1), per Chinese zupu (族谱) 世 convention.

## Proposed repo layout

```
zeng-dynasty/
├── CLAUDE.md                 # auto-loaded project context for Claude Code
├── README.md                 # slim human-facing overview (images → docs/)
├── requirements.txt
├── docs/
│   ├── milestones.md         # static site → 4 books → dynamic site
│   ├── progress.md           # per-book status, what's done / next
│   ├── pipeline.md           # how the 4-stage CV/OCR pipeline works
│   └── specs/                # design specs (this file lives here)
├── src/                      # rewritten pipeline (was book_parser.ipynb)
│   ├── model.py              # Node dataclass (promoted from data/node.py, father rename)
│   ├── extract_pages.py      # stage 1: spreads → pages
│   ├── segment.py            # stage 2: pages → merged graph images
│   ├── build_tree.py         # stage 3: graph images → JSONL + name crops
│   └── (ocr.py later)        # stage 3.5: name images → Unicode (not yet built)
├── data/
│   ├── book1_golden.jsonl    # promoted verified Book 1 data
│   └── zeng_google_sheet.csv # sheet export (source of golden)
├── books/
│   ├── book1/                # book1.pdf + regenerated original/ pages/ graphs/ names/ trees/
│   └── book2/                # book2.pdf + ...
├── web/
│   └── index.html            # promoted d3 viewer (parent→father update)
└── old/                      # EVERYTHING from before, moved via git mv (history preserved)
```

**Open layout question for review:** `books/bookN/` nesting vs. keeping
`bookN/` at top level (closer to today). Recommendation: nest under `books/`
to keep the root clean, since `src/`, `data/`, `docs/`, `web/` already live
there.

## What gets promoted from `old/`

| Asset | From | To | Why |
|-------|------|----|-----|
| Raw scans | `book1/book1.pdf`, `book2/book2.pdf`, `bookN/original/` | `books/bookN/` | Irreplaceable source input |
| Golden Book 1 data | `data/book1_golden.jsonl`, `data/zeng_google_sheet.csv` | `data/` | Verified digitized tree (the prize) |
| Node schema | `data/node.py` | `src/model.py` | Clean, documented; rename parent→father |
| Web viewer | `index.html` | `web/index.html` | Working d3 tree renderer |

Everything else stays in `old/` and is *referenced* while rewriting scripts,
not promoted: `book_parser.ipynb`, `utils.py`, all intermediate CV outputs
(`graphs/ cropped/ names/ trees/ pages/`), superseded JSONLs
(`book1.jsonl`, `book1_merged.jsonl`), `graph_visualizer/` (flask
experiments), `parser.py`, `page.py`, `test.py`, loose PNGs, the old README.

## Rewrite plan (the scripts)

Rewrite the notebook into `src/*.py` modules, each with a clear I/O contract
and a CLI entry point so "parse book N" is one command. Faithful
reproduction of the working Book 1 result first, *then* incorporate the RTL
fix and Book 2 support.

- `extract_pages.py` — stage 1, parameterized by book (no per-book skip
  fudges baked in; make them explicit config).
- `segment.py` — stage 2. Carry the `merge_graphs` sign fix; resolve the
  `shrink_page` experiment; write to the correct `books/bookN/graphs/`.
- `build_tree.py` — stage 3. **Fix `sort_nodes` to RTL/eldest-first + BFS
  ID assignment** so algorithmic output matches golden conventions.
- `ocr.py` — stage 3.5, name image → Unicode. Not yet built; candidate
  engines noted in old README (Google Document AI, Mistral OCR). Out of
  scope for the restructure; tracked in `progress.md`.

**Verification:** rerun stage 3 on Book 1 and diff against
`book1_golden.jsonl` (tree topology + generations; names pending OCR). This
is the regression check the golden data exists for.

## Housekeeping

- `.gitignore` + remove committed `.DS_Store` files (in `book1/`, `book2/`).
- Slim the README: it's currently 1.3 MB because base64 images are embedded.
  Move images to `docs/images/` and reference them.
- Large binaries in git (`Zeng Family Tree.pdf`, `book1_old.pdf` ~14 MB):
  decide Git LFS vs. keep-in-`old/`. Recommendation: leave in `old/` for now,
  revisit LFS if the repo gets cloned often.
- A `Makefile` (or `run.py`) with targets: `extract`, `segment`, `build`,
  `serve` (launch the web viewer), parameterized by book.

## Docs to author

- `docs/milestones.md` — the two milestones + sub-goals.
- `docs/progress.md` — per-book, per-stage status table. Book 1: stages 1–3
  done, golden verified, OCR pending. Book 2: stage 1 done (cropping), stages
  2–3 pending.
- `docs/pipeline.md` — the 4-stage explanation above, kept current.
- `CLAUDE.md` — architecture, layout conventions, the per-book-tuning gotcha,
  schema conventions, how to run.

## Out of scope (later)

- OCR (stage 3.5).
- Books 3 & 4 (they interleave biography pages with tree pages — different
  parsing, per old README).
- Dynamic website (milestone 2).
- Daughters/mothers schema evolution.

## Execution order (after approval)

1. `git mv` everything into `old/` (preserves history).
2. Promote the 4 survivor assets to the new layout.
3. Write `CLAUDE.md` + `docs/`.
4. Rewrite `src/` scripts, reproducing Book 1, then RTL fix + Book 2.
5. Housekeeping (.gitignore, README slim, Makefile).
