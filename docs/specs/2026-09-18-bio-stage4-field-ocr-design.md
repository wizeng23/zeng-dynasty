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

**Engine: PaddleOCR PP-OCRv5** (reuse `src/s6_ocr_paddle.PaddleEngine`), chosen by a
real bake-off (§2). Mistral hallucinates on the extreme block aspect ratios; Paddle is
robust, local, free.

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

**Stage 4 re-crops each block from the merged image using `blocks.json` (x/y/w/h),
NOT the Stage-3 PNG files.** The on-disk crop filenames are unstable (currently
band-indexed while `blocks.json` ids are generation-indexed) and Stage 3 is still being
finalized; reading geometry from `blocks.json` keeps Stage 4 decoupled from that churn.

The tree oracle `data/{book}_stitched.jsonl` gives each subgraph's node names grouped by
generation (via provenance notes → stem; see Stage 3's `tree_counts_by_stem`), used for
son-name validation.

## 2. OCR engine — bake-off result

Prototyped both engines on real 2_9 blocks (`scratchpad/bakeoff*.py`):

| Block shape | Paddle PP-OCRv5 | Mistral mistral-ocr-latest |
|---|---|---|
| Normal (宪炳, ~5:1) | sons `庆鸿/庆亮` ✓; name column garbled | sons ✓; prose clean; sideways header hallucinated |
| Wide-sparse (up to 25:1) | reads real content ✓ | **hallucinates tiling** (`…西侧 ×50`, `1. 2. …100.`) ✗ |
| Right-clustered pre-crop (纪有) | — | reads perfectly (`祿之祀有子 … 配失考`) ✓ |

**Decision: PaddleOCR PP-OCRv5** for the P0 path — robust to the extreme aspect ratios
these blocks routinely have (the exact Mistral failure the geometric-parse design doc
warned about), local, free, reproducible. Reuse `src/s6_ocr_paddle.PaddleEngine`
(full detect+recognize pipeline reads multi-column images in reading order).

Mistral remains an **optional** later secondary pass for prose fields on near-square
crops; not on the P0 path.

## 3. Per-block pre-crop (the one fiddly geometry piece)

Stage-3 blocks run from one header's left edge to the next header's left edge, so a
block can be ~25 000px wide with the person's vertical text columns clustered at the
**right** and large blank to the **left** (space reserved for younger siblings printed
lower in the same band). OCR wastes effort / downscales on that.

Pre-crop each block to its **right-side content**:
1. From the block's right edge, walk left over columns; keep the contiguous run,
   ending at the first blank-column run ≥ `CLUSTER_GAP` (tune on real 2_9; ~400px
   start point).
2. Trim to the ink bbox of that column range, add small padding.
3. **Fallback:** if the resulting crop is still very wide (a genuinely wide entry —
   e.g. a generation with a single person whose columns spread the full width), pass
   the whole block to Paddle (it handles wide images; only Mistral tiled on them).

Save the pre-crop box for QA overlay.

## 4. Field parsing (markers, not geometry)

Paddle returns text pieces in reading order (RTL columns). Join them, then parse by
**marker regex** — the bios follow a fixed clause pattern:

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
      "raw_text": "…", "ocr_conf": 0.xx, "precrop": [x0,y0,x1,y1] }
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

QA overlay: `books/{book}/qa/bio_s4/{stem}.png` — merged section downscaled, each
block's pre-crop box drawn, extracted sons annotated.

## 6. Validation (no unit tests — repo convention)

Per the repo's "no unit tests" rule, validate by running on real data:
- Run Stage 4 on section 2_9.
- **Gate:** union of all gen-5 blocks' `sons` == the 9 gen-6 tree names
  (`庆林 庆鸿 庆亮 庆海 庆荣 庆华 庆财 庆铭 庆粮`). Report `missing_from_ocr` /
  `extra_in_ocr`.
- Eyeball a few blocks' `name_ocr` vs the tree name; eyeball the QA overlay.

A partial match is expected on first pass (some son glyphs are rare / OCR-hard); the
gap list tells us where to iterate. Sons are the must-get-right field — iterate the
pre-crop / parse until the gen-5→gen-6 reconstruction is clean on 2_9, then generalize.

## 7. Module & CLI

`src/bio/s4_ocr.py`, mirroring Stage 3's shape:
```
PYTHONPATH=. python -m src.bio.s4_ocr --book book3
PYTHONPATH=. python -m src.bio.s4_ocr --book book3 --sections 2_9
```
Functions: `precrop_block(merged, box) -> Image`, `parse_fields(text_pieces) -> dict`,
`ocr_section(...)`, `ocr_book(...)`, plus `_save_qa`. Paddle engine imported lazily
(optional dep), like `src/s6_ocr.py` does.

## 8. Out of scope this session

- Node attachment (block k ↔ tree node k) — a later merge stage.
- Stitch-edge creation from son names — later, feeds `s8_cross_stitch`.
- Fixing Stage 3 crop-file naming — Stage 3 is being finalized separately.
- Other books / all sections — generalize after 2_9 (and a couple more) pass.

## 9. Open questions to resolve during build

1. `CLUSTER_GAP` value for the right-side pre-crop (tune on 2_9's 20 blocks).
2. Son-clause regex robustness: ordinal prefixes, `名` vs `名曰`, multi-char son names,
   daughters (`生女`) not bleeding into sons.
3. Whether Paddle's reading-order join reliably keeps son glyphs contiguous after the
   `生子…名` marker, or if a son-region sub-crop is needed.
