# Handoff: Book 4 OCR review via model-disagreement flagging

**Copy this whole file into a fresh Claude Code session** in the `zeng-dynasty`
repo. It is self-contained — no prior conversation context needed.

## Goal

Speed up William's Book 4 OCR review. Instead of William eyeballing all ~509 name
crops, YOU (the model) independently read each crop with your own vision, compare
your reading to the automatic PP-OCR reading, and **flag only the nodes where the
two disagree**. Those flagged nodes are the genuinely-uncertain ones. William then
opens the OCR review tool, checks "only flagged", and reviews just that short list.

**Why this works (validated on Book 3, 2026-09-16):** on a 24-node sample, the
model's independent reading matched William's confirmed ground truth 22/24 (92%),
vs 20/24 for raw OCR. Crucially, **every node where the model's reading == OCR's
reading was also correct** — the disagreements are exactly the hard glyphs. So
"OCR and model agree" ≈ trustworthy; flag only the disagreements.

## Environment

- Run everything in the `zeng` conda env: `/opt/miniconda3/envs/zeng/bin/python`
  with `PYTHONPATH=.` (base python lacks paddleocr/cv2/PIL). Symptom of wrong env:
  `ModuleNotFoundError`.
- Book 4 is already parsed + OCR'd (do NOT re-run stages). Inputs on disk:
  - `data/book4.jsonl` — 509 nodes, each with `name` (OCR reading) and `notes`
    (provenance `{graph}_{localindex} | ocr_conf=...`).
  - `data/book4_names.json` — the OCR sidecar `{id: {name, confidence, low_conf}}`.
  - `books/book4/5_names/{id}.png` — the name crop per node (what you read).
- The OCR review tool: `scripts/qa/s6_ocr.py`. Its flag layer is
  `data/book4_flags.json` — a JSON **list of node provenances**
  (e.g. `["0_0_3", "18_18_1"]`). The tool's UI has an **"only flagged"** checkbox
  and `Alt+F` to toggle a flag; a flagged node shows a purple marker.

## Reusability across a regen (important — read before starting)

If Book 4's graphs are ever regenerated, results survive to different degrees:

- **Provenance (`{graph}_{localindex}`) is NOT regen-stable.** localindex is a
  node's position within its graph; if a graph gains or loses a node, every index
  after that point shifts, so `18_18_3` may point to a different person. Anything
  keyed by provenance (flags, William's overrides, your readings) silently
  mis-maps.
- **BUT all 132 Book 4 graphs are single-page** (no seam merges), so they only
  change if a *crop* changes — and the crops are now stable (whitening is off for
  book4, page 287's start is forced). So in practice most graphs won't change on a
  regen, and their provenance-keyed results transfer fine.
- **Make it durable anyway:** ALSO record your reading keyed by a **crop content
  hash** (sha1 of `books/book4/5_names/{id}.png` bytes), not only by provenance.
  Write `data/book4_model_ocr_by_hash.json` = `{crop_sha1: your_reading}`. After a
  regen, a node whose crop is byte-identical keeps its reading regardless of index
  renumbering; a node with a NEW hash (a recovered/changed crop — e.g. a
  newly-found character) simply has no prior reading and gets reviewed fresh. This
  is the key to your reuse question: **unchanged crops reuse automatically, new
  characters are just new work — the old results are never invalidated, only
  augmented.** William's per-character overrides (`book4_overrides.json`) are human
  ground truth and are re-keyable the same way (by crop hash) if provenance drifts.

## What to build / do

### 1. Read every crop and record your reading

For each node in `data/book4.jsonl`, read `books/book4/5_names/{id}.png` with your
vision and record your best Hanzi reading. Names are 1–5 vertically-stacked Hanzi
(mostly 2: a generation char + a given char). Read top-to-bottom.

- Do this in **batches** (e.g. 20–30 crops per pass) — view the images, write your
  reading for each. Be honest about uncertainty; when a glyph is genuinely
  ambiguous (e.g. 火 vs 忄 radical, 吾 vs 有 component), note it — those SHOULD end
  up flagged anyway.
- Save your readings to `data/book4_model_ocr.json` as `{provenance: "your_reading"}`
  so the work is resumable and auditable. Provenance = first `" | "` segment of a
  node's `notes` (there's a `_provenance` helper in `scripts/qa/s6_ocr.py`).

Practical note: 509 crops is a lot of vision calls. Fetch/read them straight from
`books/book4/5_names/{id}.png` (Read tool on the PNG). Don't start a server just to
get crops. Consider doing it graph-by-graph (group by provenance prefix) so you can
stop/resume cleanly.

### 2. Compare and write the flags

For each node, compare your reading (`data/book4_model_ocr.json`) to the OCR reading
(the `name` in `data/book4.jsonl`, or `data/book4_names.json`). Where they **differ**
(after stripping whitespace; compare exact Hanzi strings), add that node's
**provenance** to a set. Write the sorted set to `data/book4_flags.json` (a JSON
list). Mirror the tool's format exactly — look at `data/book3_flags.json` for the
shape (it's a plain list of provenance strings), and use the tool's `save_flag` /
`_flags_path` (`scripts/qa/s6_ocr.py`) if convenient.

Report the counts: total nodes, # agreements (skip these), # disagreements (flagged).
On Book 3's sample the disagreement rate was ~8%, so expect roughly **~40 flagged of
509** — that's William's whole review list.

Optional but nice: also write `data/book4_model_ocr_report.json` listing, per
disagreement, `{provenance, your_reading, ocr_reading}` so William can see both
candidates at a glance.

### 3. Hand back to William

Tell William:
- how many nodes you flagged (the review list size),
- to start the review tool and filter to flagged:
  ```
  PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -m scripts.qa.s6_ocr \
      --books book4 --port 8796
  # open http://localhost:8796/ , tick "only flagged"
  ```
  (Book 1 used 8766, Book 2 8776, Book 3 8786 — 8796 is free for Book 4.)
- His corrections save to `data/book4_overrides.json` (per-character,
  keyed `{provenance}#{charIndex}`), a separate ground-truth layer.

### 4. After William finishes the review

Fold his corrections into the tree, then re-stitch:
```
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c \
  "from src.s6_ocr import apply_names; apply_names('book4')"
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -m src.s7_stitch --book book4
```
(`apply_names` merges OCR + overrides into `data/book4.jsonl`; stitch writes
`data/book4_stitched.jsonl`.)

## Guardrails

- Don't re-run Book 4 stages 3–7 or touch Books 1/2/3 data — this task only READS
  crops and WRITES `data/book4_flags.json` (+ your own model_ocr json).
- The repo is public; agents never `git push` (William pushes). Commit locally only
  if asked.
- Your reading is ~92% accurate, not 100% — never overwrite `name` in the jsonl with
  your reading. You only FLAG; William decides. Where you and OCR agree, trust OCR
  (leave unflagged); where you disagree, flag for William.

## Files this task touches

- WRITES: `data/book4_flags.json` (the flags the review tool reads),
  `data/book4_model_ocr.json` (your readings), optionally
  `data/book4_model_ocr_report.json` (the disagreement list).
- READS: `data/book4.jsonl`, `data/book4_names.json`, `books/book4/5_names/*.png`.
- REFERENCE: `scripts/qa/s6_ocr.py` (`_provenance`, `_flags_path`, `save_flag`,
  the "only flagged" filter), `data/book3_flags.json` (format example).
