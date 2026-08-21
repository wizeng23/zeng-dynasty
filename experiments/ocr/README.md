# OCR engine bake-off — experiment artifacts

Frozen scripts + results from choosing the OCR engine for turning name-crop
images (`books/bookN/names/{id}.png`) into Unicode characters. Narrative and
decision live in `docs/ocr-bakeoff.md`; this folder holds the reproducible
scaffolding and the raw per-engine predictions.

## Result (2026-08-21, 233-char split set)

Ground truth: all 163 Book 1 parse crops hand-verified
(`data/oracles/book1_names_truth_full.json`), then stacked multi-char names
split into single-character crops → 233 single-char targets. Priority was
**simplified-Chinese** accuracy (rare/ancient chars are the edge case).

| Engine | Overall (233) | Common simplified (216) | Cost | Runs |
|--------|--------------|-------------------------|------|------|
| **PaddleOCR PP-OCRv5** | **225/233 = 96.6%** | **216/216 = 100%** | free | local |
| Google Vision (DOCUMENT_TEXT_DETECTION) | 196/233 = 84.1% | 86.1% | ~$1.5/1k | cloud |
| Google Document AI (Enterprise OCR) | 190/233 = 81.5% | 84.3% | ~$1.5/1k | cloud |
| Mistral OCR (mistral-ocr-latest) | 20/233 = 8.6% | 9.3% | $4/1k | cloud |

**Winner: PP-OCRv5** — most accurate *and* the only free/local one. The cloud
engines are document/layout systems that expect page context, so they drop or
mis-read isolated glyphs even grid-packed. Full analysis in `docs/ocr-bakeoff.md`.

## Files

- `scripts/make_split_set.py` — builds the 233-char single-char split set from
  the 163-crop truth (vertical gap split of stacked names).
- `scripts/run_bakeoff.py` — runs one engine over the split set and scores it.
  Usage: `PYTHONPATH=. python experiments/ocr/scripts/run_bakeoff.py <engine>`
  (engines: `paddle`, `google:vision`, `google:docai`, `mistral`).
- `results/preds_*.json` — each engine's raw `{split_id: char}` predictions.
- `results/split_truth_233.json` — the split-set ground truth.

## Reproducing

Engines live in `src/ocr.py` (+ `src/ocr_paddle.py`, `src/ocr_cloud.py`).
Cloud engines need creds: `MISTRAL_API_KEY` for Mistral; Google ADC +
`DOCAI_PROJECT`/`DOCAI_LOCATION`/`DOCAI_PROCESSOR_ID` for Doc AI; ADC + enabled
Vision API for Vision. PP-OCRv5 needs only `pip install paddlepaddle paddleocr`.
