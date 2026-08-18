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
books/       per-book assets: bookN/book.pdf, original/ (spreads), pages/, graphs/, names/, trees/
web/         d3.js tree viewer (index.html)
docs/        milestones.md, progress.md, pipeline.md, specs/
old/         the entire pre-restructure repo, archived (history preserved via git mv)
```

## The pipeline (4 stages)

Scans → structured tree. See `docs/pipeline.md` for detail.

1. **Spreads → pages** — split + deskew two-page scans (`books/bookN/original/`) into single pages.
2. **Pages → graph images** — detect subtree boundaries, stitch line-graphs across pages → `books/bookN/graphs/{start}_{end}.png`.
3. **Graph images → tree** — parse lines into nodes, crop name images → `data/bookN.jsonl`, `books/bookN/names/`.
4. **OCR** (not yet built) — name image → Unicode character.

Stage 4 (spreadsheet path): Book 1 was also hand-typed into a Google Sheet →
`data/zeng_google_sheet.csv` → `data/book1_golden.jsonl`. This **golden** data
is the verified ground truth used to check the algorithmic output.

## Key gotchas

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
