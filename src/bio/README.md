# Bio sub-pipeline (`src/bio/`)

Parses the **biography pages** of Books 3 & 4 geometrically (the tree-graph pipeline
in `src/` handles the tree pages). Design: `docs/specs/2026-09-17-bio-geometric-parse-design.md`.

Run everything in the `zeng` conda env with `PYTHONPATH=.` (see the repo CLAUDE.md).

## Stages

| Stage | Module | Reads | Writes |
|-------|--------|-------|--------|
| 1 crop | `s1_crops.py` | classified bio pages | `books/{book}/bio/1_crops/{page}.png` |
| 2 merge | `s2_merge.py` | `1_crops/` | `books/{book}/bio/2_merged/{a}_{b}.png` + `.json` |
| 3 segment | `s3_segment.py` → **QA** → `s3_post.py` | `2_merged/` + `data/{book}_stitched.jsonl` | `books/{book}/bio/3_segment/` (see below) |
| 4 field OCR | *(separate, not here)* | `3_segment/` crops + `blocks.jsonl` | children names etc. |

## Stage 3 in detail (segment sections into per-person blocks)

A merged section is 5 stacked **generation-bands** (gens 2–6; gen 1 = branch root, no bio
row), split at the section JSON's `rules_y`. Each person's entry opens with a horizontal
header caption `子之X` — a ~385–400px-wide ink stripe near the band top; detecting those
stripes gives one block per person. A **count gate** checks detected blocks per band ==
tree nodes per generation (`data/{book}_stitched.jsonl` is the oracle).

Stage 3 is three steps — detect, human-QA, then finalize — because the boxes move in QA,
so cropping images before QA would be wasted:

### Step 1 — `s3_segment.py` (pre-QA detection)

```
PYTHONPATH=. python -m src.bio.s3_segment --book book3
```
Writes, per section, **one JSONL** (one block per row, in order) and **no images**:
- `books/{book}/bio/3_segment/qa_input/{a}_{b}.jsonl` — the QA editor's input.
- `books/{book}/qa/bio_s3/{a}_{b}.png` — QA overlay: band rules, detected header boxes,
  and a per-row diff (`gen3: 4/5 MISSING 1`, red when short, green when complete).

Each row: `{section, stem, id, band, generation, box:[l,t,r,b], expected_per_gen,
detected_per_gen, gate_passed}`. Provenance `id = {a}_{b}_{c}_{d}`: `a_b` = section
start/end pages, `c` = band index (0 = top = gen 2 … 4 = bottom = gen 6), `d` = index in
row (0 = rightmost/eldest, increasing leftward — the book is RTL).

### Step 2 — QA editor (`scripts/qa/bio_s3_edit.py`)

```
PYTHONPATH=. python -m scripts.qa.bio_s3_edit --book book3 --port 8788
# open http://localhost:8788/
```
Reads `qa_input/{a}_{b}.jsonl`, shows the merged section (downscaled for speed) with the
boxes and a live count-gate bar. Move/resize/add/delete boxes per band until each band's
count matches the tree. On **Save** it writes a full copy of the corrected state to
`books/{book}/bio/3_segment/{a}_{b}.jsonl` (recomputing every `d` so provenance stays in
RTL order). That approved file — not a diff — is what the post step reads, so adding a
block simply renumbers cleanly.

### Step 3 — `s3_post.py` (finalize)

```
PYTHONPATH=. python -m src.bio.s3_post --book book3
```
For each section with an approved `{a}_{b}.jsonl`, cuts the crop images. Writes:
- `books/{book}/bio/3_segment/{a}_{b}_{c}_{d}.png` — one crop per person.
- `books/{book}/bio/3_segment/blocks.jsonl` — the final combined index stage 4 consumes.

No tree-node association is done: even after QA, a section's bio count is not guaranteed
to equal the subgraph's node count (missing/extra bios), so a positional block→node
mapping is unreliable. Linking a bio to its tree node is left to a later, evidence-based
step (e.g. matching by children names).

## Output layout (`books/{book}/bio/3_segment/`)

```
qa_input/{a}_{b}.jsonl     step 1  (pre-QA detections; never overwritten by QA)
{a}_{b}.jsonl              step 2  (QA-approved full copy)
{a}_{b}_{c}_{d}.png        step 3  (final crops)
blocks.jsonl              step 3  (final combined index for stage 4)
```

Notes: all bio output stays under `books/{book}/bio/` (nothing goes in `data/`). PNGs are
gitignored (`books/*/bio/*/*.png`); JSONL sidecars are tracked. This is a one-off data
pipeline — validate by running on real books and reading the QA overlays / count gate,
not by unit tests.
