# Zeng Family Tree — Project Guide

Digitizing a 4-book Chinese family-tree (族谱 / zupu) from PDF scans into a
structured tree, and rendering it as a website.

**Milestones:** (1) static website rendering all content once all 4 books are
parsed → (2) dynamic website allowing updates. See `docs/milestones.md`.

## Repo layout

```
src/         rewritten parsing pipeline (Python)
  model.py     Node dataclass — the canonical data schema
data/        parsed output (book1_golden.jsonl) + sheet export (csv)
books/       per-book assets: bookN/bookN.pdf (canonical = v1 bitonal 600dpi scan),
             plus bookN_gray.pdf (v1 grayscale), bookN_v0.pdf (old glass-top scan),
             original/ (v0 spreads), pages/, graphs/, names/, trees/
web/         d3.js tree viewer (index.html)
docs/        milestones.md, progress.md, pipeline.md, specs/
old/         the entire pre-restructure repo, archived (history preserved via git mv)
```

## Scan versions (v0 / v1)

Two scan generations exist per book:

- **v0** — the original glass-top scans (book laid open face-down on a scanner,
  two pages per image). Book 1 & 2 only. Lives in `books/bookN/original/*.png`;
  the source PDF is `books/bookN/book1_v0.pdf` / `book2_v0.pdf`. This is what the
  **current `src/` pipeline (Stage 1 `extract_pages`) consumes**.
- **v1** — the new scans: spines cut off, pages through an automatic document
  feeder (ADF), one page per scan, 600dpi. Two renditions:
  `bookN.pdf` = **bitonal** (black/white — the canonical default) and
  `bookN_gray.pdf` = grayscale. Books 3 & 4 are **v1 bitonal only**. The v1
  pipeline lives in `src/v1/` (`extract_pages`, `segment`, `build_tree`),
  full-native resolution; it deskews (pure rotation, no perspective warp),
  strips the printed page frame + header/label text columns, and parses.

`bookN.pdf` always means the canonical scan to parse (now v1 bitonal). Historical
`bookN_gray_200dpi.pdf` (Book 1) is an early 200dpi ADF test, kept for reference.

## The pipeline (4 stages)

Scans → structured tree. See `docs/pipeline.md` for detail.

1. **Spreads → pages** — split + deskew two-page scans (`books/bookN/original/`) into single pages.
2. **Pages → graph images** — detect subtree boundaries, stitch line-graphs across pages → `books/bookN/graphs/{start}_{end}.png`.
3. **Graph images → tree** — parse lines into nodes, crop name images → `data/bookN.jsonl`, `books/bookN/names/`. Includes **orphan-bridging** (`src.segment.bridge_orphans`): reconnects generation bars broken across page seams; the synthetic connectors ("green" in QA) are recorded to `books/bookN/graphs/{stem}.imaginary.json`. See `docs/bridge-ground-truth.md`.
3.5. **Stitching** (`src/stitch.py`) — fold each graph's duplicate root into its canonical leaf → one connected lineage. Book 1 done (hand `BOOK_MERGES`); Book 2 auto-matcher TODO.
4. **OCR** (Stage 3.6, DONE) — PaddleOCR PP-OCRv5 reads each name crop → Unicode. `src/ocr.py` writes sidecar `data/bookN_names.json` + folds into jsonl via `apply_names`. Per-char human overrides live in `data/bookN_overrides.json` (a ground-truth layer). Review tool: `scripts/qa/ocr.py`.

Stage 4 (spreadsheet path): Book 1 was also hand-typed into a Google Sheet →
`data/zeng_google_sheet.csv` → `data/book1_golden.jsonl`. This **golden** data
is the verified ground truth used to check the algorithmic output.

## Key gotchas

- **Run in the `zeng` conda env, not base.** Base Python lacks the deps
  (paddleocr/cv2/PIL/pypinyin). Use `/opt/miniconda3/envs/zeng/bin/python` with
  `PYTHONPATH=.`, or `conda activate zeng`. Symptom of the wrong env: `ModuleNotFoundError`.
- **Big graphs are slow.** Book 2's `36_52` is 17 pages (~93M px); a full
  `python -m src.segment --book book2` takes minutes. Run it in the background.
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
- **QA is gitignored** (`books/*/qa/`): regenerate with `python -m scripts.qa.overlay
  --book bookN`, then serve `books/bookN/qa/index.html` on a localhost port to view.

## Docs map (where to look)

- `docs/history.md` — chronological project story (read this to catch up fast).
- `docs/pipeline.md` — per-stage detail; `docs/progress.md` / `milestones.md` — status/goals.
- `docs/bridge-ground-truth.md` — the orphan-bridging rule + hand-verified per-graph truth.
- `docs/bridge-revisit-notes.md` — known pipeline BUGS to fix later (incl. the
  `remove_small_islands` missing-dots bug).
- `docs/future-features.md` — deferred FEATURE ideas (e.g. polyphonic-name support).
- `docs/ocr-bakeoff.md` — why PP-OCRv5; `docs/overnight-worklog.md` — bridging blow-by-blow.

## Current state (2026-08-25)

Books 1 & 2 fully parsed; **all 45 Book-2 subgraphs LOCKED** (0 within-graph
orphans, bridges match ground truth). OCR populated for both books. Uncommitted-to-
remote work sits on branch `book2-bridge-trace-right` (William to push). **Next
step:** William reviews OCR via `scripts/qa/ocr.py`, then Book 2 stitching, then
website. Remaining orphans in a few graphs are cross-graph (resolved at stitching).
