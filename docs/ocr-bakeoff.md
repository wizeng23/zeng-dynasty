# OCR engine bake-off (Stage 3.6)

**Date:** 2026-08-18
**Goal:** pick the OCR engine for turning name-crop images
(`books/bookN/names/{id}.png`) into Unicode characters.

## Setup

- **Ground truth:** `data/oracles/book1_names_truth.json` — 40 Book 1 name crops
  hand-labelled (cross-referenced against `data/book1_golden.jsonl` + direct
  reading). The crops are clean, printed, single characters; several are
  **rare/ancient** (点, 㝵, 顼, 玚, 旃) — the known hard case.
- **Harness:** `scripts/eval_ocr.py` scores any engine's `{crop_id: char}`
  predictions — exact-match accuracy, per-char misses, and *skipped* (undetected)
  count. Engines are pluggable (`src/ocr.py`, `OcrEngine` protocol).
- **Grid-packing:** `src/ocr.py` packs crops into one labelled grid per API call
  so a single request covers many crops and undetected cells stay visible.

## Results (40 chars)

| Engine | Exact | Notes |
|--------|-------|-------|
| **PaddleOCR PP-OCRv5** (`paddle`) | **36/40 (90%)** | Chinese-specialized, local, free. Only true miss: 㝵→寻 (×2, genuinely ancient). Other 2 are variant glyphs: 羨→羡, 寳→寶 (same character, different radical form) — effectively ~95%. |
| OpenAI gpt-4o (`openai:gpt-4o`) | 25/40 (62%) | Misses cluster on rare/ancient + simplified↔traditional (㝵→导, 顼→项, 寳→宝). Attempted all (grid-pack works). |
| OpenAI gpt-4.1 (`openai:gpt-4.1`) | 25/40 (62%) | Same as gpt-4o — general vision models plateau here. |

## Decision

**PaddleOCR PP-OCRv5 is the winner** and the default engine:

- Decisively more accurate on this exact task (90% vs 62%), because it is trained
  specifically on Chinese incl. rare/ancient characters — the research consensus
  (PP-OCRv5 was evaluated across Chinese ancient-text scenarios) held up on our
  data.
- Runs **locally, free, reproducible** — no API key, no per-page cost, no data
  leaving the machine.
- The general vision LLMs (gpt-4o / gpt-4.1) are a useful *second opinion* for the
  handful Paddle misses, but not the primary engine.

## Round 2 (2026-08-21): full 233-char split set, +Mistral +Google

Rebuilt the eval to reflect real usage (William: simplified accuracy matters most,
rare chars are the edge case). Hand-verified ground truth for **all 163 Book 1
parse crops** (`data/oracles/book1_names_truth_full.json`), then split the 64
stacked multi-char names into single-character crops (vertical gap split) →
**233 single-char targets**. Ran four engines over the split set. Google auth via
ADC (personal project `wizeng-dev`); Vision + Document AI APIs enabled; Doc AI
Enterprise-OCR processor `zeng-ocr`. Mistral via `mistral-ocr-latest`.

| Engine | Overall (233) | Common-only (216) | Notes |
|--------|--------------|-------------------|-------|
| **PaddleOCR PP-OCRv5** (`paddle`) | **225/233 = 96.6%** | **216/216 = 100%** | Perfect on common simplified chars. 8 misses all rare/ancient/variant (㝵, 羨/羡, 寳/寶, 遜, 迪, 與/舆, 尽/盡). |
| Google Vision (`google:vision`) | 196/233 = 84.1% | 186/216 = 86.1% | Document engine — drops/mis-splits isolated glyphs even grid-packed; misses common chars (点,乐,嘉,安,心). 26 skipped. |
| Google Document AI (`google:docai`) | 190/233 = 81.5% | 182/216 = 84.3% | Enterprise-OCR processor. Same document-engine mismatch + some cell-alignment drift. Slow (per-crop fallback = 233 calls). |
| Mistral OCR (`mistral:mistral-ocr-latest`) | 20/233 = 8.6% | 20/216 = 9.3% | Layout engine, wrong tool. Adds markdown list markers, even hallucinated Arabic on one crop. Per-crop confirmed ~2/10; not an alignment artifact. |

**Confirmed: PP-OCRv5 remains the winner, now decisively — 100% on common
simplified characters.** The document/layout engines (Vision, Doc AI, Mistral)
all underperform because they expect page context, not a grid of isolated glyphs;
they drop or mis-read even easy simplified chars. This matches the research
prediction: only a dedicated crop-in→char-out recognizer (PP-OCRv5's CRNN head)
fits this task. Engines wired in `src/ocr_cloud.py` (`MistralOCREngine`,
`GoogleVisionEngine`, `GoogleDocAIEngine`) + `src/ocr.py get_engine`. Vertical
splitter + 233-char truth built for the eval.

## Not yet tested (future)

- **Mistral OCR** — needs an API key. William flagged it; worth a run for
  completeness, though it is document/layout-oriented rather than single-glyph.
- **Google Document AI** — `gcloud` is configured (project `lema-dev`) but needs a
  processor set up. William's old area; strong but reportedly weaker on ancient
  chars than PP-OCRv5.
- **GLM-OCR (Zhipu)** — tops recent LLM-OCR leaderboards; needs a key.
- Larger eval set (all 59 golden chars, then Book 2's 2-char stacked names, which
  need vertical splitting first).

## Next steps

1. Wire PaddleOCR as the default in a full `python -m src.ocr --book book1` run
   that writes Unicode `name` back into `data/book1_stitched.jsonl` (only when
   confident; keep `name_images` as the fallback).
2. For the ~10% Paddle misses, use a second engine (or manual, per the original
   plan — "Dad's down to help") and normalize traditional/variant forms.
3. Extend to Book 2 (split 2-char vertical stacks into two crops first).
