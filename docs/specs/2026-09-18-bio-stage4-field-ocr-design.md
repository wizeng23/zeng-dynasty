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

**Engine: an ensemble of two independent readers**, chosen by a real bake-off (§2, §2a):
1. **PaddleOCR PP-OCRv5 with detect-then-resort** (reuse `src/s6_ocr_paddle`) — run
   Paddle's detector, then override its reading order by re-sorting the detected lines
   into RTL columns geometrically. Fixes the whole-block reading-order scramble (fatal
   for son extraction), robust to dense touching columns, and yields char-boxes for the
   structural QA checks.
2. **A vision Claude subagent** — transcribes the crop semantically (RTL, one line per
   column, father-header first, name second). Catches Paddle's blind spots (horizontal
   header, short bold name).
Sons (P0) are the reconciled union with per-son agreement flags. Mistral rejected
(tiling hallucination + API cost).

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

## 2a. Ensemble — vision-subagent reader backs Paddle

Paddle is backed by a **second, independent reader**: a vision-capable Claude subagent
(dispatched via the Agent tool with the tight crop) that transcribes the block
semantically. The two readers have **complementary failure modes** — Paddle is strong
on the long vertical prose columns and gives deterministic char-boxes (the QA
gap-checks, §5a), but drops the **horizontal father header** and truncates the **short
bold name column**; the vision reader gets exactly those. Validated on both test crops
(`scratchpad/`, results in the design session):

| Field | Paddle | Vision subagent |
|---|---|---|
| Father header (`子之禄`/`子次棋`) | ✗ dropped `之禄` / partial | **✓ ✓** both crops |
| Bold name (`纪有`/`宪炳`) | ✗ truncated | **✓ ✓** both |
| Sons P0 (`庆林`) | ✓ | **✓** (own line, right after `生子…名`) |
| `配失考` marker | ✗ (`配` only) | **✓** |
| Prose columns | ✓ | ✓ |

**Subagent prompt (validated verbatim):** transcribe one 族谱 person block; vertical
columns read RIGHT-TO-LEFT; output one line per column, RTL; **line 1 = the top-right
horizontal header `子之X`/`子次X` (father marker), line 2 = the person's own
(vertical, bold) name, lines 3+ = the bio prose columns**; `?` for unreadable chars;
transcription only, no translation/commentary.

**Reconciliation (per field):**
- **Father-char** (P1): subagent line 1 (Paddle blind here). Agreement = confirmed.
- **Own name:** subagent line 2, cross-checked against the tree oracle (the identity
  truth) and Paddle's name column; disagreement → `qa_flag`.
- **Sons (P0):** parse `生子…名` from **each** reader independently. Emit the **union**
  for recall, and mark each son with `agreed: true` when **both** readers found it.
  Sons found by only one reader are flagged for review. The sons-union gate (§6) then
  distinguishes ensemble-confirmed from single-source.
- **Prose/dates/spouse/daughters:** keep both reads in `raw_text`; no gate.

**Cost/latency note:** one subagent call per block (~10–16 s, ~37k tokens in the test).
Acceptable for 2_9 (20 blocks); for full-book runs we can gate the subagent to blocks
where Paddle's structural checks (§5a) flag a problem, rather than every block. Decide
when generalizing.

## 3. Detect-then-resort (Paddle's core algorithm)

Per tight crop:
1. `PaddleOCR(lang="ch").predict(rgb)` → `rec_texts`, `rec_polys`, `rec_scores`.
2. For each detected line take its poly's x-center and top-y.
3. **Cluster into columns by x-center** (tolerance ≈ 0.7× median line width); order
   columns **right-to-left** (descending x-center).
4. Within each column order lines **top-to-bottom** (ascending y); join to column text.
5. Concatenate columns RTL → the person's entry in true reading order.

Edge cases from the bake-off (handled in parse, §4). Two elements in the top-right need
care:
- **Father header** (`子之X` / `子次X`) — the *one* HORIZONTAL element on the page (子 之
  X, left-to-right), against otherwise vertical columns. This orientation break is what
  makes Paddle's vertical-tuned detector drop it (it kept only `子`). This is Paddle's
  true blind spot; the vision-ensemble reader (§2a) handles it.
- **Bold name** (`纪有`) — VERTICAL, same orientation as the prose, but the **short
  rightmost column** (often 2 chars) in bold/large type just under the header. Paddle
  truncates or merges it with the header above; not an orientation problem, a
  short-column / header-adjacency one.

The son columns (the P0) read solidly on both engines. Keep per-line conf; flag
low-conf columns to QA.

## 4. Field parsing (markers, not geometry)

Work over the ordered column texts. Parse by **marker regex** — the bios follow a fixed
clause pattern:

- **Sons (P0):** clause `生子…名` then the following name glyph(s) until `生女` or block
  end. Ordinals `长/次/三/…` may prefix each. Emit `sons: [...]`.
  - `生子一名 X` → one son; `生子二名 X Y` → two; count word is a cross-check on arity.
  - **No-children blocks** (end `配失考`, or no `生子`) → `sons: []` (valid).
- **Own name:** line 2 of the ensemble read — the vertical bold column just under the
  header (the ensemble reader delivers it as an explicit line; §2a). The header itself
  is the *father* char, not the person's name — do not use it for the name.
- **Best-effort fields:** `生于…`(birth) `殁`(death) `葬`(burial) `配…`(spouse)
  `生女… 长/次… 适 <place>`(daughters). Store parsed where clean, else leave in
  `raw_text`.
- **Father-char (P1, now free):** line 1 of the ensemble read (the horizontal header
  `子之X`). Independent parent evidence — the vision reader gets it reliably (both test
  crops), so capture it as `father_char` even though it's not required this session.

## 5. Output — `books/{book}/bio/4_ocr/{stem}.json`

```jsonc
{
  "section": "2_9", "stem": "0_1", "engine": "ensemble:paddle+vision",
  "blocks": [
    { "block_id": "0_1_5_1", "generation": 5, "band": 3, "k": 1,
      "name": "宪炳", "father_char": "棋",
      "sons": [ {"name": "庆林", "agreed": true} ],   // agreed = both readers
      "daughters": ["雪英"], "birth": "…", "death": "…", "spouse": "…",
      "burial": "…",
      "paddle": { "columns": ["…"], "sons": ["庆林"], "ocr_conf": 0.xx },
      "vision":  { "lines": ["子次棋","宪炳","…"], "sons": ["庆林"] },
      "qa_flags": ["col3: intra-gap dropped char", "name: paddle≠vision"] } // [] = clean
  ],
  "sons_by_gen": { "5": ["庆林","庆鸿", …] },
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
  look when it fails. Track how many gate members are ensemble-`agreed` (both readers)
  vs single-source — full agreement on the reconstruction is the strong pass.
- Eyeball a few blocks' `name` vs the tree name; eyeball the QA overlay.

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
Functions:
- `paddle_read(rgb, engine) -> {columns, sons, char_boxes}` — detect-then-resort + the
  §5a structural checks.
- `vision_read(crop_path) -> {lines, father_char, name, sons}` — dispatch the vision
  subagent (§2a prompt), parse its RTL lines (line 1 father, line 2 name, 3+ prose).
- `reconcile(paddle, vision, tree_names) -> block_record` — the §2a per-field rules,
  sons = flagged union.
- `ocr_section(...)`, `ocr_book(...)`, `_save_qa`.
Paddle engine imported lazily (optional dep), like `src/s6_ocr.py` does. The vision
reader uses the Agent tool; a `--no-vision` flag runs Paddle-only (for quick iteration
or when the subagent is unavailable).

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
