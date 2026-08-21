# Pipeline

How raw scans become a structured family tree. Four stages (the current
`src/` rewrite reproduces the working Book 1 result from the old
`old/book_parser.ipynb` + `old/utils.py`, then fixes known bugs).

## Stage 1 — Spreads → pages

**In:** `books/bookN/original/*.png` — two-page camera scans (each image is a
left + right page spread).
**Out:** `books/bookN/pages/{i}.png` — single deskewed pages.

Detects the page-border corners on each half of the spread and applies a
perspective normalization. Some per-book fudging exists for which half-page to
skip at the start (Book 1 skips the first spread's right page; Book 2 skips
page 0's right side). The rewrite makes these explicit config, not inline
constants.

Old helpers: `remove_small_islands`, `get_corners`, `normalize_page`.

## Stage 2 — Pages → merged graph images

**In:** `books/bookN/pages/*.png`
**Out:** `books/bookN/graphs/{start}_{end}.png` — one image per subtree,
stitched across the pages it spans.

A subtree's line-graph often runs across several pages. This stage detects
subtree start pages (a label on the upper-right), trims/shrinks each page to
its graph, and merges consecutive pages into a single graph image by aligning
the line endpoints at the page seam.

Old helpers: `trim_borders`, `is_tree_start_page`, `shrink_page`, `merge_graphs`.

**Bugs to fix:** `merge_graphs` vstack sign bug (else-branch); the old Book 2
run wrote merged graphs to `book1/graphs/` (copy-paste); `shrink_page` has an
unresolved commented-out experiment.

## Stage 3 — Graph images → tree + name crops

**In:** `books/bookN/graphs/*.png`
**Out:** `data/bookN.jsonl` (tree, names still as image pointers),
`books/bookN/names/{id}.png`, `books/bookN/trees/*.json`.

Finds the graph's lines as connected components (BFS), identifies each line's
parent/child endpoints, merges line segments into nodes across the graph,
infers missing endpoints, assigns globally-unique IDs, and crops each node's
name image from the page.

**Ordering (bug fix):** nodes must be sorted **right-to-left / eldest-first**
and IDs assigned **BFS-by-generation, right-to-left** — the book reads
right-to-left with older siblings on the right. The old `sort_nodes` was
left-to-right.

Old helpers: `find_lines`, `find_line_ends`, `sort_nodes`, `infer_ends`,
`get_name_image`.

## Stage 3.5 — Cross-graph stitching (`src/stitch.py`)

**In:** `data/bookN.jsonl` (a forest — one subtree per Stage-2 graph)
**Out:** `data/bookN_stitched.jsonl` (one connected tree, absolute generations).

Each graph's root (`{graph}_0`) is a **duplicate** of a person who appears as a
*leaf* in an earlier graph (the subtree-start page repeats the parent name).
Merging each duplicate into its canonical leaf connects the forest into one
lineage, then recomputes absolute generations (root=1) and reassigns BFS/RTL ids.

**Matching is unsolved automatically** (name-crop pixel-matching ≈ 4/13 on Book
1). Merges are an explicit per-book list (`BOOK_MERGES`, Book 1's done by hand);
`find_merges()` is where an automated matcher will plug in. Verified against
`data/oracles/book1_merged.jsonl` via `scripts/verify_stitch.py`.

## Stage 3.6 — OCR

**In:** `books/bookN/names/*.png` + `data/bookN.jsonl` (node ids)
**Out:** `data/bookN_names.json` (sidecar) → merged into `name` on each node.

Convert name-character images to Unicode. Engine: **PaddleOCR PP-OCRv5** —
Chinese-specialized, local, free, and the bake-off winner (96.6% single-char /
100% on common simplified; see `docs/ocr-bakeoff.md`). Its full detect+recognize
pipeline reads multi-character *stacked* names in one pass, so no pre-splitting
is needed for the write path.

Run (after Stage 3, since it reads `bookN.jsonl` for ids and writes names into
it):

```
python -m src.ocr --book bookN --populate
```

This writes a **sidecar** `data/bookN_names.json` (`{id: {name, confidence,
low_conf}}`) and merges it into `data/bookN.jsonl` — setting `name` and appending
`ocr_conf=<score>` (and `ocr_low_conf` when confidence < 0.90) to `notes`. The
sidecar is the source of truth for names, so a Stage-3 re-parse never loses them:
just re-run `--populate` (or `ocr.apply_names`) afterward. `write-all` policy —
every node gets whatever PP-OCRv5 returns; low-confidence glyphs are flagged for
review, not dropped. Nodes with a blank crop keep `name=""` so the website falls
back to the image. For the stitched Book 1, `ocr.apply_names_by_crop` maps the
book1 sidecar onto `book1_stitched.jsonl` by crop id.

Cloud engines (Google Vision, Google Document AI, Mistral) are wired in
`src/ocr_cloud.py` for comparison but lost the bake-off (document/layout engines,
weak on isolated glyphs) — kept as optional second opinions, not the default.

## Golden data (separate, manual — verification)

Book 1 was also hand-typed into a Google Sheet → exported to
`data/zeng_google_sheet.csv` → `data/book1_golden.jsonl`. This verified data
is the ground truth used to check the algorithmic Stage-3 output (tree
topology + generations; names pending OCR).
