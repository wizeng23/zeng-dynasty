# Zeng Family Tree — Project Guide

Digitizing a 4-book Chinese family-tree (族谱 / zupu) from PDF scans into a
structured tree, and rendering it as a website.

**Milestones:** (1) static website rendering all content once all 4 books are
parsed → (2) dynamic website allowing updates. See `docs/milestones.md`.

## Repo layout

```
src/         canonical v1 parsing pipeline (Python); src/v0/ = archived v0 pipeline
  sN_*.py      one module per stage, prefixed by stage number (s1_extract_pages … s7_stitch)
  model.py / imaging.py   shared (stage-agnostic): Node schema, image I/O
data/        parsed output (book1_golden.jsonl) + sheet export (csv)
books/       per-book assets, namespaced by scan generation. v1 output dirs are
             prefixed by the stage that produces them (no .5s):
  bookN/
    bookN.pdf                canonical scan = v1 bitonal 600dpi
    1_pages/ 3_crops/ 4_graphs/ 5_names/   v1 bitonal pipeline output
      1_pages/corners.json, corners_review.json, page_types.json   corners + bio/tree labels
    gray/  bookN.pdf, bookN_200dpi.pdf, 1_pages/ …   v1 grayscale variant
    v0/    bookN.pdf, original/ pages/ graphs/ names/   archived old glass-scan assets
web/         d3.js tree viewer (index.html)
docs/        milestones.md, progress.md, pipeline.md, history.md, specs/
old/         the entire pre-restructure repo, archived (history preserved via git mv)
```

## Scan versions (v0 / v1)

Two scan generations exist per book, namespaced by folder (no filename suffixes):

- **v1** (CANONICAL, current) — new scans: spines cut off, one page per scan
  through an ADF, 600dpi. `books/bookN/bookN.pdf` = bitonal (the parse target);
  `books/bookN/gray/bookN.pdf` = grayscale variant. All four books have v1 scans.
  The pipeline is `src/` at full native resolution: `s1_extract_pages` (frame-based
  deskew+crop), `s3_segment` (crop to tree), `s4_merge_pages` (stitch multi-page
  subtrees), `s1_apply_corners` (final gen with reviewed corners). Output at
  `books/bookN/{1_pages,3_crops,4_graphs,5_names}/`.
- **v0** (archived) — original glass-top scans (two pages per image), Book 1 & 2
  only. Assets in `books/bookN/v0/` (`bookN.pdf`, `original/`, `pages/`, `graphs/`,
  `names/`); code in `src/v0/`. Superseded by the v1 rebuild.

`books/bookN/bookN.pdf` always means the canonical scan to parse (v1 bitonal).
`books/bookN/gray/bookN_200dpi.pdf` (Book 1) is an early 200dpi ADF test, kept
for reference.

## The pipeline (7 stages)

Scans → structured tree. Whole-integer stage numbers (no `.5`s). See
`docs/pipeline.md` for detail.

1. **extract_pages** (`src/s1_extract_pages.py`) — split + deskew scans into single upright pages → `books/bookN/1_pages/`.
2. **classify_pages** (`src/s2_classify_pages.py`) — tag each page tree-graph vs **biography**. Books 3 & 4 interleave biography-text pages; only tree pages go on the graph path. Writes `books/bookN/2_classify/page_types.json` (`graph_pages`/`bio_pages`, each bio's `follows_graph`). Bio = exactly 4 full-width rules at fixed y-fracs ≈ 0.19/0.40/0.60/0.80. Books 1 & 2 (all-tree) → no sidecar, unchanged.
3. **segment** (`src/s3_segment.py`) — crop each tree page to its line-graph → `books/bookN/3_crops/` + `starts.json` (biographies skipped via the sidecar).
4. **merge_pages** (`src/s4_merge_pages.py`) — stitch a subtree's pages into one graph image → `books/bookN/4_graphs/{start}_{end}.png`.
5. **build_tree** (`src/s5_build_tree.py`) — parse lines into nodes, crop name images → `data/bookN.jsonl`, `books/bookN/5_names/`. Includes **orphan-bridging** (`bridge_orphans`): reconnects generation bars broken across page seams; the synthetic connectors ("green" in QA) are recorded to `books/bookN/4_graphs/{stem}.imaginary.json`. See `docs/bridge-ground-truth.md`.
6. **OCR** (DONE) — PaddleOCR PP-OCRv5 reads each name crop → Unicode. `src/s6_ocr.py` writes sidecar `data/bookN_names.json` + folds into jsonl via `apply_names`. Per-char human overrides live in `data/bookN_overrides.json` (a ground-truth layer). Review tool: `scripts/qa/s6_ocr.py`. **OCR precedes stitching** because stitching matches a graph's duplicate root to its canonical leaf *by name* — it needs the names first.
7. **Stitching** (`src/s7_stitch.py`, TODO — v0 logic in `src/v0/stitch.py`) — fold each graph's duplicate root into its canonical leaf → one connected lineage. Book 1 done in v0 (hand `BOOK_MERGES`); Book 2 auto-matcher TODO.

Spreadsheet path (verification): Book 1 was also hand-typed into a Google Sheet →
`data/zeng_google_sheet.csv` → `data/book1_golden.jsonl`. This **golden** data
is the verified ground truth used to check the algorithmic output.

## Key gotchas

- **Run in the `zeng` conda env, not base.** Base Python lacks the deps
  (paddleocr/cv2/PIL/pypinyin). Use `/opt/miniconda3/envs/zeng/bin/python` with
  `PYTHONPATH=.`, or `conda activate zeng`. Symptom of the wrong env: `ModuleNotFoundError`.
- **Big graphs are slow.** Book 2's `36_52` is 17 pages (~93M px); a full
  `python -m src.s3_segment --book book2` takes minutes. Run it in the background.
- **Publication gate.** The repo (`github.com/wizeng23/zeng-dynasty`) is public;
  agents are BLOCKED from `git push`. Commit locally freely; William pushes.
- **Thresholds are hand-tuned per book's scan geometry.** "Make it work on
  Book N" means re-tuning the segmentation functions, not just re-running.
  The old code baked per-book fudge factors inline — the rewrite makes these
  explicit config.
- **The book reads right-to-left; eldest siblings are on the right.** Node
  ordering, ID assignment, and `children` arrays must be RTL / eldest-first.
  (The old algorithm was left-to-right — a bug the rewrite fixes.)

## Data schema (`src/model.py`)

`id`, `name`, `name_images` (fallback when name unknown), `generation`,
`father`, `children`, `biography`, `notes`.

- **`father`, not `parent`** — the book records only men so far. Daughters/
  mothers are a later schema evolution.
- **`children` ordered eldest-first** (right-to-left).
- **IDs are BFS-by-generation, right-to-left** — natural since the 4 books
  split by generation.
- **`generation` is 1-indexed** — root 点 (Zeng Dian) = gen 1, per zupu 世 convention.

## Working here

- The old logic lives in `old/book_parser.ipynb` + `old/utils.py` — reference
  it when rewriting `src/`, but it is not the source of truth.
- Verify a rewritten Book 1 parse by diffing against `data/book1_golden.jsonl`.
- Design specs go in `docs/specs/YYYY-MM-DD-<topic>-design.md`.
- **QA is gitignored** (`books/*/qa/`): regenerate with `python -m scripts.qa.s5_parse
  --book bookN`, then serve `books/bookN/qa/index.html` on a localhost port to view.

## Docs map (where to look)

- `docs/history.md` — chronological project story (read this to catch up fast).
- `docs/pipeline.md` — per-stage detail; `docs/progress.md` / `milestones.md` — status/goals.
- `docs/bridge-ground-truth.md` — the orphan-bridging rule + hand-verified per-graph truth.
- `docs/bridge-revisit-notes.md` — known pipeline BUGS to fix later (incl. the
  `remove_small_islands` missing-dots bug).
- `docs/future-features.md` — deferred FEATURE ideas (e.g. polyphonic-name support).
- `docs/ocr-bakeoff.md` — why PP-OCRv5; `docs/overnight-worklog.md` — bridging blow-by-blow.

## Current state (2026-09-15)

**Handoff for the next session: `docs/handoff.md`** (rules, validation recipe,
remaining work items in order, OCR review list, Book 3 plan, server ports).

**Book-3-prep fixes landed (2026-09-15, `history.md` Era 11), no Book 2 re-run.**
Stage 3 bottom whitening is now anchored to the *detected* inner border
(`bottom_inner_border_row` + `WHITEN_BOTTOM_FROM_BORDER=180`), so the six …子 names
no longer lose their last char (clearance 8px→57px over 134 pages). `find_lines`
extracts each component from its cv2 bbox (identical output, ~60× faster:
`parse_graph` 12.9s→4.1s). Stage 4 raises on a page-number gap instead of welding
across missing bio pages. On a Book 2 re-run, `data/book2_fixes.json` reduces to
**just `delete 8_10_9`** (the 106_113 merge is now auto-resolved by bridging).
New QA tool `scripts/qa/s5_fixes.py`. `NAME_HALF_WIDTH` stays 120 (widening
regressed). All in-memory-validated against `books/book2/frozen_2026-09-14`.

**Book 1 is DONE through all 7 v1 stages** → `data/book1_stitched.jsonl` (150
nodes, 1 root 点, 56 gens, OCR'd names), published to the website. Stitch seams are
now matched **by name** (`find_merges` in `src/s7_stitch.py`), not the old hardcoded
provenances (which were v0-keyed and mis-connected 7/13 under the v1 renumber).
All on `main`, pushed.

**Book 2 runs through all 7 stages with orphan bridging** (2026-09-14) →
`data/book2_stitched.jsonl`: 1549 nodes, **9 roots** = the main lineage + 8 section
roots whose OCR reading differs from their canonical leaf (贞烈/贞列, 贞熊/贞能,
贞杰/贞木, 贞斗/贞升, 贞亮/贞光, 克太, 贞富, 贞年). 0 orphans. **Post-fixes layer:**
`data/book2_fixes.json` (William's QA review: delete/merge/recrop by provenance) is
applied by `python -m src.s5_fixes --book book2 --ocr` after every Stage 5 run, then
Stage 7. Book 2's *downstream* (4_graphs/5_names) is still from 2026-09-01 crops
where six …子 names lost their last char — OCR override until a Stage 3→7 re-run
(the Stage 3 whitening is fixed in code as of Era 11, so a re-run restores them).
Stage 4 now
rejects ADF smear specks as seam endpoints; Stage 5 has the stroke-end read for
stepped bars and `bridge_orphans` (trace-right rule, follows seam steps, targeted
gate; green `{stem}.imaginary.json`, cyan nicks `{stem}.nicks.json`); Stage 7
merges only `{graph}_0` roots, falls back to non-leaf canonicals, folds re-printed
chains. Next: William reviews Book 2 OCR (`scripts/qa/s6_ocr.py`) and parse QA, then
re-run `s6_ocr.apply_names` + `s7_stitch`. Do NOT re-run Book 1 regression checks
(William: Book 1 is confirmed correct).

Books 3 & 4 are at Stage 2 (classify, human-verified) + Stage 3 (crop).
