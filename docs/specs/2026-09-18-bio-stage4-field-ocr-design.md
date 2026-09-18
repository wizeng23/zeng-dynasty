# Bio Stage 4 — per-person field OCR — design

**Status:** design (2026-09-18). Ready to build.
**Branch:** `stage-4-bio-ocr` (off `review-bio-parse-design`, which contains all of
`origin/main` plus bio stages 1–3).
**Author:** William + session.
**Depends on:** bio Stage 3 (`src/bio/s3_segment.py`) — per-person block boxes
(`blocks.json`) + merged section image (`books/{book}/bio/2_merged/{stem}.png`).
**Feeds:** a later merge stage folds Stage 4 fields into the tree nodes and the
cross-book stitch (children-name evidence). Stage 4 itself does **not** mutate
`data/{book}_stitched.jsonl`.

---

## 0. TL;DR

For each per-person **block** that Stage 3 found, OCR its fields and write a
structured record to `books/{book}/bio/4_ocr/{stem}.json`. The **P0 field is the
person's SONS** (`生子…名 <sons>`), the strong stitch signal (a parent's listed sons
are a subset of the next generation's names and say *which* nodes are whose children).
Everything else (own name, dates, spouse, daughters, burial) is captured best-effort.

**Engine: PaddleOCR PP-OCRv5 with detect-then-resort** (reuse `src/s6_ocr_paddle`),
chosen by a real bake-off (§2): run Paddle's detector, then override its reading order
by re-sorting the detected lines into RTL columns geometrically. This fixes the
whole-block reading-order scramble (fatal for son extraction) and is robust to dense
blocks where columns touch. Mistral rejected (tiling hallucination + API cost).

**This session:** prove it on section **2_9** (subgraph `0_1` = 传禄). Validation gate:
the union of all gen-5 blocks' extracted sons must reconstruct the 9 gen-6 tree names.
Stop before node-attachment / stitching.

---

## 1. Inputs (from Stage 3)

Per section stem (e.g. `2_9`):
- `books/{book}/bio/2_merged/{stem}.png` — the merged section image (binary ink grid,
  `0`=ink/`1`=bg, from `src.imaging.get_image`), and `{stem}.json` (`rules_y`,
  `seam_x`, `size`, `pages_reading_order`).
- Stage 3's `blocks.json` — **the source of truth for block geometry**: per block
  `{id, generation, band, x, y, width, height}`, plus `gate_passed`,
  `expected_per_gen`, `detected_per_gen`.

**Stage 4 consumes tight per-person crops** — one person, blank margins removed, like
the canonical example `books/book3/bio/Screenshot 2026-09-18 …png` (786×340, header
`子之禄` + name `纪有` + prose columns). **Producing these tight crops is Stage 3's job**
(coordinated with the in-flight S3 finalization), not Stage 4's. Stage 4 reads the crop
+ `blocks.json` for the block's `generation` (used to pick the oracle generation for
son validation).

The tree oracle `data/{book}_stitched.jsonl` gives each subgraph's node names grouped by
generation (via provenance notes → stem; see Stage 3's `tree_counts_by_stem`), used for
son-name validation.

## 2. OCR engine & method — bake-off result

Prototyped Paddle (whole-block vs column-wise vs Paddle-detect-then-resort) and Mistral
on the canonical tight crop and a stable 宪炳 block (`scratchpad/*bakeoff*.py`,
`detect_resort.py`):

| Method | Canonical (纪有, tight) | 宪炳 (son 庆林) |
|---|---|---|
| Whole-block Paddle | glyphs ✓ but **reading order scrambled** (put last col right after header) | order risk on 2D layout |
| Column-wise Paddle (vertical projection) | **perfect** (11 cols, RTL, conf ~1.0) | **fails** — dense block has no pixel gutters (columns touch / ADF smear) → 1 blob |
| **Paddle-detect-then-resort** | 10 cols, correct RTL prose | **18 cols, `生子一名`→`庆林` adjacent & correct, conf 1.0** |
| Mistral | correct but 戌→成, header merged; **tiling hallucination on wide/sparse crops** | — |

**Decision: PaddleOCR PP-OCRv5 with detect-then-resort.** Run Paddle's full
detect+recognize pipeline (`PaddleOCR(lang="ch")`, reuse `src/s6_ocr_paddle`) to get
per-line boxes + text, then **discard Paddle's reading order** and re-sort the boxes
geometrically into columns (RTL by x-center) then top-to-bottom within a column. This:
- fixes the whole-block **reading-order scramble** (fatal for `生子…名 <sons>` where
  order defines which glyphs are the sons);
- is robust to **dense blocks where columns touch** (vertical-projection splitting
  collapses there; Paddle's detector still separates the lines, and we only override
  ordering);
- keeps each glyph at native resolution; local, free, reproducible.

Mistral stays an **optional** later secondary prose pass; not on the P0 path (tiling
risk + API cost).

## 3. Detect-then-resort (the core algorithm)

Per tight crop:
1. `PaddleOCR(lang="ch").predict(rgb)` → `rec_texts`, `rec_polys`, `rec_scores`.
2. For each detected line take its poly's x-center and top-y.
3. **Cluster into columns by x-center** (tolerance ≈ 0.7× median line width); order
   columns **right-to-left** (descending x-center).
4. Within each column order lines **top-to-bottom** (ascending y); join to column text.
5. Concatenate columns RTL → the person's entry in true reading order.

Edge cases from the bake-off (handled in parse, §4): the **header band** (`子之X` /
`子次X`, horizontal, top row) and the **bold name** need separate handling from the
vertical prose columns — the header/name occasionally split or under-detect, but the
son columns (the P0) read solidly. Keep per-line conf; flag low-conf columns to QA.

## 4. Field parsing (markers, not geometry)

Work over the ordered column texts. Parse by **marker regex** — the bios follow a fixed
clause pattern:

- **Sons (P0):** clause `生子…名` then the following name glyph(s) until `生女` or block
  end. Ordinals `长/次/三/…` may prefix each. Emit `sons: [...]`.
  - `生子一名 X` → one son; `生子二名 X Y` → two; count word is a cross-check on arity.
  - **No-children blocks** (end `配失考`, or no `生子`) → `sons: []` (valid).
- **Own name:** the larger glyph(s) after the header `子之X` / `子次X`. The header
  itself is the *father* char and OCRs poorly (sideways) — skip it for the name.
- **Best-effort fields:** `生于…`(birth) `殁`(death) `葬`(burial) `配…`(spouse)
  `生女… 长/次… 适 <place>`(daughters). Store parsed where clean, else leave in
  `raw_text`.
- **P1 verification (optional):** header father-char, as independent parent evidence.
  Attempt only if the sideways header reads reliably; not required this session.

## 5. Output — `books/{book}/bio/4_ocr/{stem}.json`

```jsonc
{
  "section": "2_9", "stem": "0_1", "engine": "paddle:PP-OCRv5",
  "blocks": [
    { "block_id": "0_1_5_1", "generation": 5, "band": 3, "k": 1,
      "name_ocr": "宪炳", "sons": ["庆鸿", "庆亮"],
      "daughters": ["雪英"], "birth": "…", "death": "…", "spouse": "…",
      "burial": "…", "father_char": null,
      "raw_text": "…", "ocr_conf": 0.xx, "columns": ["…","…"],
      "qa_flags": ["col3: intra-gap dropped char", …] }   // empty = clean
  ],
  "sons_by_gen": { "5": ["庆鸿","庆亮", …] },
  "validation": {
    "gen": 6,
    "sons_union": ["庆林","庆鸿", …],
    "tree_names": ["庆林","庆鸿","庆亮","庆海","庆荣","庆华","庆财","庆铭","庆粮"],
    "missing_from_ocr": [...], "extra_in_ocr": [...], "match": true|false
  }
}
```

## 5a. Structural QA checks (auto, from Paddle char boxes — no extra OCR)

`predict(rgb, return_word_box=True)` returns per-**character** boxes (`text_word` +
`text_word_boxes` `[x0,y0,x1,y1]`, grouped per detected line). From these we compute
three cheap structural checks per block that catch OCR misses *without* ground truth —
validated on the canonical 纪有 and 宪炳 crops (`scratchpad/qa_gaps.py`):

1. **Inter-column pitch.** Column x-centers should be ~1 pitch apart (median pitch ≈
   median char width: canonical 77 vs 66; 宪炳 220 vs 193 — all gaps within ±15%). A
   gap **> 1.6× median pitch** ⇒ a whole column (a person's clause) was likely missed.
   *Confirmed no false positives; would fire on a dropped column (≈2× pitch).*
2. **Intra-column contiguity.** Down a column, consecutive char boxes should touch. A
   **y-gap > 1.2× median char height** between consecutive chars ⇒ a dropped character
   mid-column. *Confirmed no false positives on clean 7-char columns.*
3. **Column fill ratio.** A column whose (char-count × char-height) ≪ its pixel span ⇒
   chars dropped. *This caught the truncated header/name columns the other two missed:
   canonical col0 read `子` alone (should be `子之禄纪有`) and col10 `配` (should be
   `配失考`).* The header/name band is the known weak spot (§3); sons read solidly.

Each check writes a per-column flag into the block record and the QA overlay, so a
reviewer sees *where* an entry is suspect, not just that a count is off.

QA overlay: `books/{book}/qa/bio_s4/{stem}/` — per block, the tight crop with detected
**char** boxes drawn, columns numbered RTL, flagged columns (checks 1–3) outlined red,
extracted sons annotated.

## 6. Validation (no unit tests — repo convention)

Per the repo's "no unit tests" rule, validate by running on real data. Two layers:

- **Structural (self-contained, §5a):** the three char-box checks flag suspect columns
  with no ground truth needed — the assurance the QA "will work" even on sections/books
  we have no oracle for.
- **Semantic gate (ground truth):** run Stage 4 on section 2_9; the union of all gen-5
  blocks' `sons` must equal the 9 gen-6 tree names
  (`庆林 庆鸿 庆亮 庆海 庆荣 庆华 庆财 庆铭 庆粮`). Report `missing_from_ocr` /
  `extra_in_ocr`. This is the ultimate P0 check; the structural flags say *where* to
  look when it fails.
- Eyeball a few blocks' `name_ocr` vs the tree name; eyeball the QA overlay.

A partial match is expected on first pass (some son glyphs are rare / OCR-hard); the
gap list tells us where to iterate. Sons are the must-get-right field — iterate the
column clustering / parse until the gen-5→gen-6 reconstruction is clean on 2_9, then
generalize.

## 7. Module & CLI

`src/bio/s4_ocr.py`, mirroring Stage 3's shape:
```
PYTHONPATH=. python -m src.bio.s4_ocr --book book3
PYTHONPATH=. python -m src.bio.s4_ocr --book book3 --sections 2_9
```
Functions: `detect_columns(rgb, engine) -> list[Column]` (Paddle-detect-then-resort),
`parse_fields(columns) -> dict`, `ocr_section(...)`, `ocr_book(...)`, plus `_save_qa`.
Paddle engine imported lazily (optional dep), like `src/s6_ocr.py` does.

## 8. Out of scope this session

- Node attachment (block k ↔ tree node k) — a later merge stage.
- Stitch-edge creation from son names — later, feeds `s8_cross_stitch`.
- Stage 3 crop finalization (tight cropping, file naming) — done in the S3 work.
- Other books / all sections — generalize after 2_9 (and a couple more) pass.

## 9. Open questions to resolve during build

1. Column-clustering x-center tolerance (bake-off used 0.7× median line width; confirm
   across 2_9's 20 blocks, esp. dense daughter-heavy entries).
2. Son-clause regex robustness: ordinal prefixes (`长/次/三…`), `名` vs `名曰`, count
   word (`生子二名`) as arity cross-check, `生女` not bleeding into sons.
3. Header/name band: how reliably to isolate `子之X`/`子次X` (horizontal, top) and the
   bold name from the prose columns — the bake-off showed these occasionally split or
   under-detect (sons unaffected).
