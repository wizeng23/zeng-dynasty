# Commands cheat-sheet

Every command that gets real work done, with the offline story called out. Verified
against the code on 2026-09-16. When something here disagrees with the code, the code
wins — re-derive and fix this file.

## Ground rules

- **Always run in the `zeng` conda env.** Base Python lacks paddleocr/cv2/PIL/pypinyin.
  Either `conda activate zeng` first, or prefix each command:
  ```
  PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python ...
  ```
  Symptom of the wrong env: `ModuleNotFoundError`. All examples below assume you've
  either activated `zeng` or are prefixing.
- **`PYTHONPATH=.`** is required for `-m src.*` / `-m scripts.*` imports to resolve.
- Books are named `book1`..`book4`. `src.*` stages take `--book bookN`; the OCR review
  server takes `--books bookN` (plural, comma-separated).
- **Agents can't `git push`** (public repo) — William pushes. Commit locally freely.

## Offline (no-wifi) at a glance

| Task | Offline? | Why |
|------|----------|-----|
| Name OCR (Stage 6, PaddleOCR/PP-OCRv6) | ✅ yes | Local model, cached in `~/.paddlex/official_models/` |
| OCR review server (`scripts.qa.s6_ocr`) | ✅ yes | Local HTTP server, no network |
| Parse stages 1,3,4,5,7 | ✅ yes | Pure local compute |
| Parse QA overlays + viewing | ✅ yes | Writes static HTML/PNG; view via `python -m http.server` |
| Editable parse QA (`scripts.qa.s5_edit`) | ✅ yes | Local HTTP server |
| Model-vision reading of crops (this session's Book 4 task) | ✅ needs Claude | The vision reads are done by the agent, which needs connectivity |
| **Bio OCR (`scripts/run_bio_ocr.py`)** | ❌ **no** | Calls Mistral cloud; needs `MISTRAL_API_KEY` + wifi |
| Website dev / deploy | ✅ dev offline / ❌ deploy needs push | Next.js in `web-app/` |

So on a flight you can: run/re-run name OCR, review it in the browser, re-run any parse
stage, edit parses in the browser, and stitch. You **cannot** run bio OCR (Mistral).

---

## QA servers (one per book, each on its own port)

Convention (kept consistent across sessions): **b1=…66, b2=…76, b3=…86, b4=…96** for
the OCR reviewer. Open the printed `http://localhost:PORT/` in a browser. Ctrl-C stops.

### OCR review — `scripts.qa.s6_ocr`  (local, offline ✅)
Filmstrip of every name crop with the OCR reading; edit to correct. Corrections save to
`data/{book}_overrides.json` (keyed `{provenance}#{charIndex}`), a separate ground-truth
layer. UI has an **"only flagged"** checkbox (reads `data/{book}_flags.json`) and
**Alt+F** to toggle a flag.

```
python -m scripts.qa.s6_ocr --books book1 --port 8766
python -m scripts.qa.s6_ocr --books book2 --port 8776
python -m scripts.qa.s6_ocr --books book3 --port 8786
python -m scripts.qa.s6_ocr --books book4 --port 8796
# default (no args) serves ALL books with v1 crops on :8766
python -m scripts.qa.s6_ocr
```

### Editable parse QA — `scripts.qa.s5_edit`  (local, offline ✅)
Fix a subgraph's Stage-5 parse directly in the browser (move/resize/add/delete name
boxes, connect child→father). Writes a self-contained per-subgraph override to
`data/{book}_manual/{stem}.json` that REPLACES Stage 5 for that graph.

```
python -m scripts.qa.s5_edit --book book3 --port 8787
python -m scripts.qa.s5_edit --book book4 --port 8797
```

### Border / corners QA — `scripts.qa.s1_borders`  (default port 8761)
### Page-classify QA — `scripts.qa.s2_classify`  (default port 8762)
```
python -m scripts.qa.s1_borders --port 8761
python -m scripts.qa.s2_classify --port 8762
```

### Parse-overlay QA — `scripts.qa.s5_parse`  (writes static files, offline ✅)
Does NOT start a server. Writes `books/{book}/qa/*.png` + `books/{book}/qa/index.html`
(QA dir is gitignored — regenerate any time). View it with a plain static server:

```
python -m scripts.qa.s5_parse --book book4
python -m http.server 8000          # from repo root
# open http://localhost:8000/books/book4/qa/index.html
```

Related static-QA generators:
```
python -m scripts.qa.s5_fixes --book book2      # post-fixes review page
python -m scripts.qa.all_filmstrip --book book4 # per-stage filmstrip -> data/artifacts/{book}/
```

---

## Name OCR (Stage 6) — the Book-4 workflow, fully offline ✅

PaddleOCR PP-OCRv6 is a **local** model (cached under `~/.paddlex/official_models/`), so
this all works on a plane. It reads each `books/{book}/5_names/{id}.png` crop.

**1. Populate OCR readings** → writes `data/{book}_names.json`
(`{id: {name, confidence, low_conf}}`). Keyed by node id.
```
python -c "from src.s6_ocr import populate_names; populate_names('book4')"
```
(509 crops ≈ a couple minutes; run in the background for a big book.)

**2. Review** in the browser (see `scripts.qa.s6_ocr` above); corrections →
`data/{book}_overrides.json`.

**3. Fold OCR + overrides into the tree** → updates `data/{book}.jsonl`:
```
python -c "from src.s6_ocr import apply_names; apply_names('book4')"
```

### Model-disagreement flagging (what generated Book 4's review list)
The agent (vision) reads every crop independently, compares to the PP-OCR reading, and
flags only disagreements so the review list is short. Outputs:
- `data/{book}_model_ocr.json` — the model's readings, keyed by provenance (resumable)
- `data/{book}_flags.json` — sorted provenance list the "only flagged" filter reads
- `data/{book}_model_ocr_report.json` — `{provenance, your_reading, ocr_reading}` per disagreement

The comparison/flag-writing step is pure local Python; the **vision reads need the
agent** (so if you're offline without Claude, you can still populate OCR and review it
manually, just not regenerate the model readings). See
`docs/book4-ocr-disagreement-handoff.md`.

---

## Full parse pipeline (per book, all local ✅)

Stages, in order. Each takes `--book bookN`. Big graphs (e.g. Book 2's 17-page `36_52`)
take minutes — run those in the background.

```
python -m src.s1_extract_pages  --book book4     # split+deskew scans -> 1_pages/
python -m src.s2_classify_pages --book book4     # tag tree vs bio pages (books 3&4)
python -m src.s3_segment        --book book4     # crop each tree page -> 3_crops/
python -m src.s4_merge_pages    --book book4     # stitch subtree pages -> 4_graphs/
python -m src.s5_build_tree     --book book4     # parse lines->nodes, crop names -> 5_names/, data/book4.jsonl
# --- Stage 6 OCR here (see above): populate_names -> review -> apply_names ---
python -m src.s5_fixes          --book book2 --ocr  # apply data/book2_fixes.json (Book 2 only, after every Stage 5)
python -m src.s7_stitch         --book book4     # fold duplicate roots -> data/book4_stitched.jsonl
```

> **Book 1 is confirmed correct — don't re-run its regression checks.**
> **Book 2 is frozen (`books/book2/frozen_2026-09-14`) — don't re-run stages 3–7 without
> William's OK.** Test Book 2 changes on a single graph in memory.

Bridging note: Stage 5 does orphan-bridging; synthetic connectors land in
`books/{book}/4_graphs/{stem}.imaginary.json` (green in QA), nicks in `{stem}.nicks.json`
(cyan). See `docs/bridge-ground-truth.md`.

---

## Bio OCR (Stage 8) — needs wifi ❌

Cloud (Mistral `mistral-ocr-latest`). Requires `MISTRAL_API_KEY` in the env and a network
connection — **won't work offline.**
```
MISTRAL_API_KEY=... python scripts/run_bio_ocr.py --book book3
# -> books/{book}/8_bio_ocr/ + data/{book}_bio_ocr.json
```

---

## Verify / diagnostics (all local ✅)

```
python scripts/compare_book1.py       # diff a Book 1 parse vs data/book1_golden.jsonl
python scripts/verify_stitch.py       # check a stitched tree
python scripts/parse_health.py        # defect metrics
python scripts/orphan_probe.py        # bridge diagnostics
python scripts/eval_ocr.py            # OCR accuracy vs oracle
```

---

## Website (`web-app/`, Next.js — dev offline ✅ / deploy needs push ❌)

```
cd web-app
bun run export-data      # regenerate public/data + public/names from data/*.jsonl (gitignored)
bun run dev              # local dev server
bun test                 # vitest unit tests (pinyin/search)
GITHUB_PAGES=true bun run build   # Pages artifact (basePath /zeng-dynasty); plain build has NO basePath
```
Deploy is via GitHub Actions on push — William pushes; the agent can't.
