# Pipeline

How raw scans become a structured family tree. Seven stages (the current
`src/` rewrite reproduces the working Book 1 result from the old
`old/book_parser.ipynb` + `old/utils.py`, then fixes known bugs).

Stages are numbered as whole integers (no `.5`s). The v1 order:

1. **extract_pages** — scan PDF → upright single pages (`src/s1_extract_pages.py`)
2. **classify_pages** — tag each page tree-graph vs biography (`src/s2_classify_pages.py`)
3. **segment** — crop each tree page to its line-graph (`src/s3_segment.py`)
4. **merge_pages** — stitch a subtree's pages into one graph (`src/s4_merge_pages.py`)
5. **build_tree** — parse graphs → tree JSONL + name crops (`src/s5_build_tree.py`)
6. **OCR** — name crops → Unicode (`src/s6_ocr.py`)
7. **stitch** — connect the per-graph subtrees into one lineage (`src/s7_stitch.py`)

## Stage 1 — Spreads → pages

**In:** `books/bookN/original/*.png` — two-page camera scans (each image is a
left + right page spread).
**Out:** `books/bookN/1_pages/{i}.png` — single deskewed pages.

Detects the page-border corners on each half of the spread and applies a
perspective normalization. Some per-book fudging exists for which half-page to
skip at the start (Book 1 skips the first spread's right page; Book 2 skips
page 0's right side). The rewrite makes these explicit config, not inline
constants.

Old helpers: `remove_small_islands`, `get_corners`, `normalize_page`.

## Stage 2 — Classify pages (tree graph vs biography)

**In:** `books/bookN/1_pages/*.png`
**Out:** `books/bookN/2_classify/page_types.json` — per-page `graph`/`bio` label +
flat `graph_pages` / `bio_pages` lists; each bio page records `follows_graph`
(the nearest preceding tree page) for later graph↔biography association.

Books 3 & 4 interleave **tree** pages (the line-graph) with **biography** pages
(dense name + birth/death prose). Only tree pages belong on the graph path; the
biographies are parsed separately later, but their page numbers are recorded so a
person's tree node can be linked to their biography.

Detection keys on the printed **cell grid**: a biography page is ruled into 5
stacked cells by exactly 4 full-width horizontal dividers at *fixed* vertical
positions (y-fractions ≈ 0.19/0.40/0.60/0.80 on every bio page, both books). A
page is a biography iff it carries exactly those 4 fixed-position full-width rules
and no other full-width interior rule (the page's own top/bottom frame border is
ignored). Tree pages have a different rule count or a different template (Book 2's
ruled tree pages rule at ~0.15/0.32/0.48/0.64). No density heuristic — the fixed
rule fingerprint alone is clean.

Validated: Books 1 & 2 (entirely tree, including their ruled tree pages) → **0**
biographies; Books 3 (51 graph / 241 bio) & 4 (147 graph / 170 bio) match
eyeballed ground truth; **0** conflicts with the Stage-3 subtree-start detector.
Books 1 & 2 have no `page_types.json`, so Stage 3 treats every page as a graph
page — their output is unchanged.

Helper: `src/s2_classify_pages.py` (`is_biography_page`, `classify_pages`,
`load_bio_pages`).

## Stage 3 — Tree pages → per-page crops

**In:** `books/bookN/1_pages/*.png` (tree pages only — biographies skipped via the
Stage-2 `page_types.json` sidecar)
**Out:** `books/bookN/3_crops/{i}.png` (one tree-crop per page) +
`books/bookN/3_crops/starts.json` (which pages start a subtree).

Detects subtree start pages (a `X房系世系图` label on the upper-right), trims the
frame, crops off the label, and shrinks each page to its graph's bounding box.
A page with no label is a continuation of the previous subtree. Biography pages
(from Stage 2) get no crop and no `starts.json` entry.

Helpers: `trim_borders`, `is_tree_start_page`, `shrink_page` (`src/s3_segment.py`).

## Stage 4 — Per-page crops → merged graph images

**In:** `books/bookN/3_crops/*.png` + `starts.json`
**Out:** `books/bookN/4_graphs/{start}_{end}.png` — one image per subtree,
stitched across the pages it spans.

A subtree's line-graph often runs across several pages. This stage merges each
run of continuation pages onto its start page into a single graph image, aligning
the line endpoints at the page seam. Iterates only the pages Stage 3 cropped
(so Book 3/4 biographies are naturally absent).

Helpers: `merge_graphs`, `find_best_orphans`, `matched_shift` (`src/s4_merge_pages.py`).

**Bugs fixed vs old:** `merge_graphs` vstack sign bug; the old Book 2 run wrote
merged graphs to `book1/4_graphs/` (copy-paste).

## Stage 5 — Graph images → tree + name crops

**In:** `books/bookN/4_graphs/*.png`
**Out:** `data/bookN.jsonl` (tree, names still as image pointers),
`books/bookN/5_names/{id}.png`, `books/bookN/trees/*.json`.

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

**Endpoint reading (`find_line_ends`):** the fast *band read* (parent = min row,
children = within `end_threshold` of the max row) assumes a single-level fan-out.
A **stepped** bar (Book 2 graph 67_68: a hang-line that steps up into a raised
sibling bar) breaks both assumptions, so a *stroke read* cross-checks it: free
ends of narrow vertical strokes (upward = parent, downward = child). The stroke
read decides the parent when the band read is ambiguous or disagrees with one
clear hang-line, and adds child risers that end above the bottom band; a bar
flush with the top (no hang-line) keeps the band read. Tests:
`tests/test_s5_line_ends.py`.

## Stage 6 — OCR (`src/s6_ocr.py`)

**In:** `books/bookN/5_names/*.png` + `data/bookN.jsonl` (node ids)
**Out:** `data/bookN_names.json` (sidecar) → merged into `name` on each node.

Convert name-character images to Unicode. Engine: **PaddleOCR PP-OCRv5** —
Chinese-specialized, local, free, and the bake-off winner (96.6% single-char /
100% on common simplified; see `docs/ocr-bakeoff.md`). Its full detect+recognize
pipeline reads multi-character *stacked* names in one pass, so no pre-splitting
is needed for the write path.

**OCR runs before stitching (Stage 7):** stitching matches each graph's duplicate
root to its canonical leaf, and that match is by *name* — so the names must exist
first.

Run (after Stage 5, since it reads `bookN.jsonl` for ids and writes names into
it):

```
python -m src.s6_ocr --book bookN --populate
```

This writes a **sidecar** `data/bookN_names.json` (`{id: {name, confidence,
low_conf}}`) and merges it into `data/bookN.jsonl` — setting `name` and appending
`ocr_conf=<score>` (and `ocr_low_conf` when confidence < 0.90) to `notes`. The
sidecar is the source of truth for names, so a Stage-5 re-parse never loses them:
just re-run `--populate` (or `s6_ocr.apply_names`) afterward. `write-all` policy —
every node gets whatever PP-OCRv5 returns; low-confidence glyphs are flagged for
review, not dropped. Nodes with a blank crop keep `name=""` so the website falls
back to the image. For the stitched Book 1, `s6_ocr.apply_names_by_crop` maps the
book1 sidecar onto `book1_stitched.jsonl` by crop id.

Per-character human corrections live in `data/bookN_overrides.json` (a
ground-truth layer applied on top of the OCR reading). Review tool:
`scripts/qa/s6_ocr.py` (port 8766).

Cloud engines (Google Vision, Google Document AI, Mistral) are wired in
`src/s6_ocr_cloud.py` for comparison but lost the bake-off (document/layout engines,
weak on isolated glyphs) — kept as optional second opinions, not the default.

## Stage 7 — Cross-graph stitching (`src/s7_stitch.py`, TODO; v0 in `src/v0/stitch.py`)

**In:** `data/bookN.jsonl` (a forest — one subtree per Stage-4 graph, names filled in by Stage 6)
**Out:** `data/bookN_stitched.jsonl` (one connected tree, absolute generations).

Each graph's root (`{graph}_0`) is a **duplicate** of a person who appears as a
*leaf* in an earlier graph (the subtree-start page repeats the parent name).
Merging each duplicate into its canonical leaf connects the forest into one
lineage, then recomputes absolute generations (root=1) and reassigns BFS/RTL ids.

**Matching is by NAME** (`find_merges`, since Stage 6 runs first). Only a graph's own
root `{graph}_0` is a duplicate; other roots in a graph are page-seam orphans and
stay roots until bridging repairs the seam. The canonical node is the nearest
earlier same-name **leaf**, else the nearest earlier same-name inner node (Book 2's
4_5 repeats 0_3's own root 存学). Sections that re-print a shared ancestor chain
(克宣 → 龙润 …) are **folded**: a duplicate's child whose name uniquely matches a
child already under the canonical node merges recursively instead of being
duplicated (≥2-char names only; a lone char is an OCR truncation). Tests:
`tests/test_s7_stitch.py`. Historical note — the original v0 approach follows.

**Matching was unsolved automatically in v0** (name-crop pixel-matching ≈ 4/13 on Book
1). Merges are an explicit per-book list (`BOOK_MERGES`, Book 1's done by hand);
`find_merges()` is where an automated matcher will plug in. Verified against
`data/oracles/book1_merged.jsonl` via `scripts/verify_stitch.py`. Not yet ported
to the v1 pipeline — v0 logic lives in `src/v0/stitch.py`.

## Golden data (separate, manual — verification)

Book 1 was also hand-typed into a Google Sheet → exported to
`data/zeng_google_sheet.csv` → `data/book1_golden.jsonl`. This verified data
is the ground truth used to check the algorithmic Stage-5 output (tree
topology + generations; names pending OCR).
