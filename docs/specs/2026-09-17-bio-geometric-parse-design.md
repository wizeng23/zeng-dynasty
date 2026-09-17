# Biography geometric parsing — design

**Status:** design / handoff (2026-09-17). Ready for a fresh build session.
**Author of design:** William (+ prior session). Written for a clean context window.
**Supersedes:** the text-first bio linker (`scripts/link_bios.py`, `scripts/run_bio_ocr.py`
whole-page OCR + `biography` prose field). That work was exploratory ("vibe-coded")
and is being **archived**, not extended. Reference it only if useful; build the new
approach from scratch.

---

## 0. TL;DR

Parse Book 3/4 **biography pages** geometrically — the same way the tree **graph**
pipeline works — instead of OCR-ing whole pages and regex-ing the text. Merge a
subgraph's bio pages into one image, split into 5 generation-bands by the printed
horizontal rules, segment each band into per-person **blocks** by whitespace gaps +
the bold/large name font, **gate on block-count == tree-node-count per generation
(hard fail)**, then bounding-box and OCR only the fields we need. The **P0 field is
each person's children's names**, because that is the strong signal that lets us
**stitch** an ambiguous tree (many leaf nodes share a parent's name char; a parent's
listed children names disambiguate the edges).

---

## 1. Why (the problem)

The current approach OCRs each bio page whole (Mistral, `books/{book}/8_bio_ocr/`)
and splits the text on name-header regexes, then matches entries to tree nodes by
name. It works but is weak:

- Whole-strip OCR fails on merged images (extreme aspect ratio → hallucinated
  tiling); per-page OCR fragments a person across page seams; flat pages lose the
  cell/generation structure. Coverage plateaued at ~70% (Book 3) / ~27% (Book 4)
  and the extracted content is not cleanly structured.
- **Stitching is the real blocker.** Matching a node to its parent/children by
  *given name* is ambiguous: many gen-N leaves share the same name character as
  their parent, so name-matching alone can't pick the right edge. **A person's bio
  lists their children's names** — a far stronger, more unique signal. Extracting
  children names reliably is what unlocks stitching (especially cross-book: Book 3's
  leaf 庆-generation → Book 4's 庆-generation roots).

The fix: treat the bio page like the graph — a **structured, count-checkable layout**
— and use the **tree as the ground-truth scaffold** (we already know exactly how many
people are in each generation of each subgraph, and their tree-order).

---

## 2. Background a fresh session needs

### 2.1 The book's structure (verified)

- The book alternates: **one subgraph (tree page(s))** then **several biography
  pages** covering every person in that subgraph.
- A subgraph spans up to **6 generations** (Book 3) / **4** (Book 4). Each bio page
  is ruled by **4 full-width horizontal lines** into **5 stacked cells = generations
  2–6** (generation 1, the subgraph's branch root, has **no** bio row). For Book 4
  (only 4 gens) the deeper rows are simply empty.
- Bio entries are laid out in the **same order as the tree nodes: depth-first,
  eldest-first, RIGHT-TO-LEFT** (the whole repo is RTL; eldest = rightmost). A
  person's row stays blank until the traversal reaches them, so there are large,
  variable whitespace gaps.
- **A person's entry runs horizontally (along X) and can cross a page seam.** This is
  why merging pages of a subgraph into one wide image (per generation-band) makes a
  person's entry contiguous.

### 2.2 The tree is the oracle

`data/{book}_stitched.jsonl` is the parsed tree (one node per line). Fields:
`id, name, name_images, generation, father, children, biography, notes`.
- `generation` is 1-indexed (root = gen 1).
- `notes` starts with **provenance** `"{graph}_{localidx}"`, e.g. `"0_1_7 | ocr_conf=…"`.
  The graph stem is `0_1` (strip the last `_N`; a `/`-joined provenance like
  `4_4_0/252_252_0` means take the first). **This maps each node to the subgraph it
  came from**, and the subgraph's first page index is `int(stem.split("_")[0])`.
- So for any subgraph we can get: its **bio pages** (via Stage 2 `follows_graph`, see
  below), and its **exact node list grouped by generation, in tree order (RTL,
  eldest-first)**. This is the ground truth the geometric parse checks against.

### 2.3 Grouping bio pages → subgraph (reuse this exact logic)

Stage 2 wrote `books/{book}/2_classify/page_types.json` (human-verified). For each
`bio` page it records `follows_graph` = the index of the nearest preceding **tree**
page. Map a bio page to a subgraph stem by: find the stem whose start page (=
`int(stem.split("_")[0])`) is the greatest `<=` that page's `follows_graph`. Group
all bio pages of a stem together; that ordered page run is the **merge unit**. (This
is the grouping the current `link_bios.py` already uses — copy it.)

### 2.4 Reusable primitives (already in the repo)

- `src.imaging.get_image(path) -> np.ndarray` — loads a page as a **binary ink grid**
  (`0` = ink, `1` = background), red-channel thresholded.
- `src.s3_segment.trim_borders(a) -> np.ndarray` — trims the printed frame off all
  four sides. **NOTE:** it leaves the left **book-title band** (武城曾氏重修族谱 +
  page number, ~first 300px of the trimmed width) and the right **branch-label /
  generation-marker band** (传禄房系 vertical label + 派/世 columns, ~last 380px).
  The geometric parse must trim these further — see §4.1.
- `src.s2_classify_pages.BIO_RULE_YFRAC = (0.193, 0.396, 0.597, 0.802)` — the four
  cell-rule y-fractions (of trimmed page height). **Stable across all bio pages**
  (prior work measured rules at fixed positions; empty pages under-detect them, so
  use these canonical fractions, not per-page rule detection).
- `src.s2_classify_pages.load_bio_pages(book)` — the bio-page set (detector +
  human overrides).
- **Reference (do not necessarily reuse):** `src/s4_merge_pages.py` is the tree-graph
  horizontal page merge (seam alignment via matched line-ends). Bio merge is simpler
  (no line-ends to align) but the seam-alignment ideas (`consensus_shift`,
  `_concat_top_aligned`) are a reference if bio rows drift at seams.

### 2.5 Environment / rules (from CLAUDE.md)

- Run in the **`zeng` conda env**: `/opt/miniconda3/envs/zeng/bin/python` with
  `PYTHONPATH=.` (base Python lacks cv2/PIL/numpy/mistralai). Wrong env symptom:
  `ModuleNotFoundError`.
- The repo is **public**; agents never `git push` (commit locally; William pushes).
- **RTL / eldest-first everywhere.** Node order, children arrays, IDs are all
  right-to-left / eldest-first.
- Big images are slow; run long jobs in the background (`nohup … &` + a monitor);
  the harness kills foreground Bash after 10 min.
- Design specs live in `docs/specs/`. QA is gitignored under `books/*/qa/`.

### 2.6 OCR engine

Mistral `mistral-ocr-latest` was chosen for bio text in a 2026-09-15 bake-off
(`docs/ocr-bakeoff.md`-adjacent; memory: bake-off). Env key `MISTRAL_API_KEY` is set.
`$0.004/page`, per-page billed. **But in this new design we OCR small bounded
sub-regions, not whole pages** — so consider whether Mistral (document-oriented) is
even the right tool for tiny name crops; the tree pipeline's name OCR uses **PaddleOCR
PP-OCRv5** (`src/s6_ocr*.py`, 96% on isolated Chinese glyphs incl. rare/ancient),
which is likely better for the bold-name crops. Decide per-field as we get there
(§4.5). Do NOT OCR whole merged strips — that failed (extreme aspect ratio →
hallucinated repetition).

---

## 3. Pipeline shape (mirror the graph stages)

For each **subgraph** (merge unit from §2.3):

```
bio pages (RTL page run)
  → trim ALL borders (incl. title + label bands)                        [§4.1]
  → horizontally merge into one image, RTL (eldest rightmost)           [§4.2]
  → split into 5 generation-bands by the 4 canonical horizontal rules   [§4.3]
  → per band: segment person-BLOCKS (whitespace gaps + bold-name font)  [§4.4]
  → GATE: #blocks(gen g) == #tree-nodes(gen g)?  hard fail + flag       [§4.5]
  → per block: bounding-box + OCR the NAME and FIELDS (children = P0)    [§4.6]
  → attach to tree nodes (block k ↔ tree node k, in RTL order)          [§4.7]
  → children-names → stitch edges                                       [§4.8]
```

Each stage is independently inspectable (save intermediate images / boxes to a QA
dir, like the graph QA). The **count gate (§4.5) is the core verification** — it is
the bio analog of the graph pipeline's structural consistency checks.

---

## 4. Stage details

### 4.1 Trim all borders

`trim_borders(get_image(page))` first (removes the outer frame). Then trim the two
residual bands it leaves:
- **left** ~300px: book title + page number.
- **right** ~380px: the vertical 传禄房系-style branch label + the 派/世 generation
  marker columns.

(These offsets are approximate from prior prototyping on Book 3; **measure/verify per
book** — Book 4's bands may differ. The right band's 派/世 markers are *content* we
may want later for a generation cross-check, but they interfere with block
segmentation, so trim for segmentation and optionally OCR them separately.)
Also blank the full-width horizontal **rule lines** (rows with >~60% ink coverage) so
they don't create false ink at seams.

### 4.2 Horizontal merge (RTL)

Concatenate the subgraph's trimmed bio pages into one wide image so that reading is
**right-to-left with the eldest on the right** — i.e. the **first bio page goes on the
RIGHT**, later pages extend leftward (the prior prototype concatenated left-to-right
in page order, which is mirror-flipped; reverse it). The absolute pixel direction does
not matter as long as **block order in the merged image corresponds 1:1 to the tree's
RTL traversal order** (§2.1) — that correspondence is what §4.7 relies on.
Pad pages to equal height before concat; keep track of page-seam x-positions (useful
for debugging block/seam interactions).

### 4.3 Split into generation-bands

Slice the merged image at the **canonical** `BIO_RULE_YFRAC` positions (scaled to the
merged height) → 5 horizontal bands = generations 2–6. Process each band
independently. (Do not rely on per-page rule detection — empty pages miss rules.)

### 4.4 Segment person-blocks within a band

Within a generation-band, find the individual person **blocks**. Signals:
- **Whitespace gaps** between entries — the primary delimiter. Gap width **varies a
  lot** (from modest to huge). A block = a contiguous run of inked columns bounded by
  blank-column gaps.
- **Bold / large name font** — each person's **name is printed larger and heavier**
  than the surrounding prose. This is a strong secondary signal: detect name glyphs
  (bigger connected components / higher stroke density) to (a) locate block starts and
  (b) disambiguate when a whitespace gap between two people is small/absent.

Open question to resolve empirically (§6): how to split two adjacent blocks with **no
gap** between them — likely by detecting the two bold-name regions and cutting between
them. Save the block bounding boxes for the count gate + QA overlay.

### 4.5 Count gate (verification stage 1) — HARD FAIL

For each generation g of the subgraph: **#detected blocks must equal #tree nodes at
generation g** (from §2.2, the tree is ground truth). If they differ, **hard fail**:
stop, flag `{book} {stem} gen{g}: blocks=X nodes=Y`, and emit a QA overlay so William
can investigate. Do **not** silently split/merge to force a match. (Later we may add a
fixes/override layer like the graph pipeline's `s5_fixes`, but v1 hard-fails.)

### 4.6 Field extraction per block (targeted OCR)

Only after the count gate passes. Within each block, the bios follow a **fixed,
consistent pattern**, so field sub-regions can be located by structure (name at the
top/right in bold; birth/death/burial/spouse clauses; **sons & daughters at the end in
a fixed format**). Bounding-box each needed field and OCR **just that crop**:
- **P0: children's names** (the sons — and daughters — listed at the end). This is the
  stitch signal (§4.8). Nail the exact geometry/pattern of the children region when we
  get here.
- Name (bold region) — for cross-check against the tree node's name.
- Other bio fields (dates, spouse, burial) — lower priority; capture as feasible.

Engine choice per field: bold single/stacked name crops → likely **PaddleOCR PP-OCRv5**
(the tree pipeline's name engine); longer prose fields → Mistral if needed. **Never**
OCR the whole strip.

### 4.7 Attach to tree nodes

Because block order == tree RTL traversal order (§4.2) **and** the count gate passed
(§4.5), block *k* in generation g maps to tree node *k* in generation g (tree order).
Attach the extracted structured fields to that node. The tree name is the oracle for
identity; the OCR name is a cross-check (flag large mismatches).

### 4.8 Stitching via children names (the payoff)

A parent node's OCR'd **children-names** are matched against the given-names of the
candidate child nodes (next generation, and across the Book 3→4 seam where the
庆-generation is otherwise ambiguous). This confirms parent→child **edges** that
name-only matching cannot. Feeds the cross-book stitch
(`src/s8_cross_stitch.py` currently joins books end-to-end; children-name evidence
strengthens the floater placements noted in memory: 传禄/传煦/贞年).

---

## 5. Data / output (proposed — confirm with William)

- New stage dir per book, e.g. `books/{book}/9_bio_blocks/` (next free number after
  the graph pipeline's 1–7 and the archived bio-OCR's ad-hoc `8_bio_ocr`). Per
  subgraph: the merged image, per-generation band images, block bounding boxes
  (JSON), and per-block field crops — all inspectable, like graph QA.
- Structured output `data/{book}_bio.jsonl` or fields folded into
  `{book}_stitched.jsonl`: per node, `{children_names: [...], name_ocr, bio_fields…}`.
- QA overlays under `books/{book}/qa/` (gitignored) showing band splits, block boxes,
  and count-gate pass/fail per generation.

Archive the old text-first artifacts (`scripts/link_bios.py`, `scripts/run_bio_ocr.py`
whole-page path, `data/{book}_linked.jsonl`, `data/{book}_bio_ocr.json`,
`books/{book}/8_bio_ocr/`) — move to an `old/`-style location or delete once this
lands; keep them referenceable during the build.

---

## 6. Open questions to resolve during the build (with William)

1. **No-gap block split** (§4.4): exact method to cut two adjacent blocks when the
   whitespace gap is small/absent — presumably via the two bold-name regions. Prototype
   bold-name detection first.
2. **Bold-name detection**: what threshold reliably separates the large/bold name
   glyphs from prose (stroke width / component size / height)? Measure on real bands.
3. **Border-band widths** (§4.1): confirm the left/right trim per book (Book 4 may
   differ from Book 3's ~300/380px).
4. **Children region geometry** (§4.6): the exact fixed format of the sons/daughters
   list (markers 生子…名 / 长/次/三…; 生女…) and how to box it. Resolve when we reach it.
5. **OCR engine per field** (§4.5/§2.6): Paddle vs Mistral for bold names vs prose.
6. **Count-gate failures** (§4.5): whether to add a manual fixes/override layer (like
   `s5_fixes`) after v1 hard-fail proves the segmentation.

---

## 7. First milestone for the build session

Prove the chain on **one known-good subgraph** (Book 3 `0_1` = 传禄; bio pages 2–9;
tree counts gen2:1, gen3:1, gen4:2, gen5:7, gen6:9). In order:
1. Trim + RTL-merge pages 2–9; save the merged image and the 5 generation-bands.
2. Segment blocks in each band (gaps + bold-name); overlay boxes; **verify the count
   gate passes** for all 5 generations against the tree.
3. Only then move to field boxing (children-names P0).

If the count gate passes on `0_1` and a couple more subgraphs, generalize; where it
hard-fails, investigate with William. Do not build field OCR before the count gate is
solid — it is the foundation.
