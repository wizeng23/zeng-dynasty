# Bio Stage 4 — Field OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Build `src/bio/s4_ocr.py` — read each per-person bio block crop, extract its
fields (P0 = sons) via an ensemble of PaddleOCR (detect-then-resort) + a vision Claude
subagent, and write structured records to `books/{book}/bio/4_ocr/`, validated on
section 2_9 against the tree oracle.

**Architecture:** Two independent readers per block. Paddle's detector returns per-line
+ per-char boxes; we discard Paddle's reading order and re-sort geometrically into RTL
columns (detect-then-resort), and compute structural QA checks from the char boxes. A
vision subagent transcribes the crop semantically (RTL, father-header line 1, name line
2, prose 3+). A reconcile step merges them: sons = flagged union, father from vision,
name cross-checked vs the tree oracle. Output is a per-section JSON sidecar + QA overlay.

**Tech Stack:** Python 3.11 in the `zeng` conda env
(`/opt/miniconda3/envs/zeng/bin/python`, `PYTHONPATH=.`), numpy, PIL,
paddleocr (PP-OCRv5, reused via `src/s6_ocr_paddle`), the Agent tool for the vision
reader, `src.imaging`, `src.bio.s3_segment` primitives.

**Spec:** `docs/specs/2026-09-18-bio-stage4-field-ocr-design.md`

## Global Constraints

- **Env:** run everything with `/opt/miniconda3/envs/zeng/bin/python` and `PYTHONPATH=.`
  (base Python lacks paddleocr/cv2/PIL). Wrong-env symptom: `ModuleNotFoundError`.
- **No unit tests** (repo convention, memory `no-unit-tests`): validate by running on
  real data (section 2_9) + the ground-truth gate + QA overlay. Do NOT add `tests/`.
- **RTL / eldest-first everywhere:** columns read right-to-left; sons listed eldest-first.
- **Commit on `main`-style flow but this is branch `stage-4-bio-ocr`:** commit locally,
  never push (public repo).
- **Do not touch** `data/*_stitched.jsonl`, Book 1/2 assets, or `src/bio/s3_segment.py`
  (S3 finalized separately). Stage 4 only reads S3 output + the oracle.
- **P0 = sons.** Sons must be correct; other fields are best-effort.
- **Ground truth for 2_9 (subgraph 0_1):** gen5 = 宪 names (7), gen6 sons =
  `庆林 庆鸿 庆亮 庆海 庆荣 庆华 庆财 庆铭 庆粮` (9). The gen5 sons-union must reconstruct
  the gen6 set.

---

## File Structure

- Create `src/bio/s4_ocr.py` — the stage. Responsibilities split into functions:
  `tight_crop` (shim), `paddle_read`, `qa_checks`, `vision_read`, `parse_sons`,
  `reconcile`, `ocr_section`, `ocr_book`, `_save_qa`, `main`.
- Create `books/{book}/bio/4_ocr/{stem}.json` — output (gitignored data dir; regenerated).
- Create `books/{book}/qa/bio_s4/{stem}/` — QA overlays (gitignored).
- Reuse: `src/s6_ocr_paddle.PaddleEngine` (Paddle instance + `return_word_box`),
  `src/bio/s3_segment` (`detect_bands`, `find_labels`, `_blocks_from_labels`,
  `tree_counts_by_stem`, `BAND_GENERATIONS`), `src/imaging.get_image`.

**S3 dependency shim:** current S3 emits WIDE blocks (blank margins); the spec assumes
tight crops. Task 1 adds an internal `tight_crop` so Stage 4 runs end-to-end now; it is
a no-op on already-tight crops, so it stays correct once S3 emits tight crops.

---

### Task 1: Block source + tight-crop shim

**Files:**
- Create: `src/bio/s4_ocr.py`

**Interfaces:**
- Consumes: merged image `books/{book}/bio/2_merged/{stem}.png` + `{stem}.json`
  (`rules_y`, `size`, `pages_reading_order`); `src.bio.s3_segment` band/label functions.
- Produces: `iter_blocks(book, sec, books_dir) -> list[BlockCrop]` where
  `BlockCrop = {id: str, generation: int, band: int, k: int, img: PIL.Image}` — the
  tight RGB crop per person, RTL/eldest-first.
- Produces: `tight_crop(gray: np.ndarray) -> np.ndarray` — trim to ink bbox with the
  right-side cluster rule (blank run ≥ `CLUSTER_GAP=400` from the right ends the entry),
  fallback to full width if still huge; pad 20px.

- [ ] **Step 1: Scaffold the module + block iteration**

Create `src/bio/s4_ocr.py`. Reuse S3 to get per-person boxes from the merged image
(don't depend on S3's PNG filenames). Recompute blocks via `detect_bands` + `find_labels`
+ `_blocks_from_labels` per band (identical to `s3_segment.segment_section`), crop each
box from the merged image, apply `tight_crop`.

```python
"""Bio stage 4: per-person field OCR (ensemble Paddle + vision subagent).

Reads each person BLOCK from a merged bio section (books/{book}/bio/2_merged/{stem}.png
+ blocks recomputed via stage 3), tight-crops it, and extracts fields -- P0 = sons --
with two independent readers reconciled. Output: books/{book}/bio/4_ocr/{stem}.json.
See docs/specs/2026-09-18-bio-stage4-field-ocr-design.md.

Run:
    PYTHONPATH=. python -m src.bio.s4_ocr --book book3 --sections 2_9
"""
from __future__ import annotations
import argparse, json, logging, os, re
from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageDraw
from src.imaging import get_image
from src.bio.s3_segment import (
    detect_bands, find_labels, _blocks_from_labels, tree_counts_by_stem,
    map_sections_to_stems, BAND_GENERATIONS,
)

logger = logging.getLogger(__name__)
CLUSTER_GAP = 400   # blank-column run (px) from the right that ends a person's entry
PAD = 20

@dataclass
class BlockCrop:
    id: str; generation: int; band: int; k: int; img: Image.Image

def tight_crop(gray: np.ndarray) -> np.ndarray:
    """Trim a wide block to its right-side content cluster + ink bbox (no-op if tight)."""
    ink = gray < 128
    has = ink.sum(axis=0) > 3
    n = len(has)
    right = n - 1
    while right >= 0 and not has[right]:
        right -= 1
    if right < 0:
        return gray
    left, blank, i = right, 0, right
    while i >= 0:
        if has[i]:
            left, blank = i, 0
        else:
            blank += 1
            if blank >= CLUSTER_GAP:
                break
        i -= 1
    rows = ink[:, left:right + 1].any(axis=1)
    r0 = int(np.argmax(rows)); r1 = len(rows) - int(np.argmax(rows[::-1]))
    c0 = max(0, left - PAD); c1 = min(gray.shape[1], right + 1 + PAD)
    r0 = max(0, r0 - PAD); r1 = min(gray.shape[0], r1 + PAD)
    return gray[r0:r1, c0:c1]

def iter_blocks(book: str, sec: str, books_dir: str = "books") -> list[BlockCrop]:
    merged_dir = os.path.join(books_dir, book, "bio", "2_merged")
    with open(os.path.join(merged_dir, f"{sec}.json")) as fh:
        meta = json.load(fh)
    a = get_image(os.path.join(merged_dir, f"{sec}.png"))   # 0=ink/1=bg
    ink = 1 - a
    w, h = meta["size"]
    stem = f"{meta['pages_reading_order'][0]}_{meta['pages_reading_order'][-1]}"
    bands = detect_bands(meta["rules_y"], h)
    out: list[BlockCrop] = []
    for bi, (top, bot) in enumerate(bands):
        labels, _ = find_labels(ink[top:bot, :])
        blocks = _blocks_from_labels(labels, top, bot, BAND_GENERATIONS[bi], bi, stem, w)
        for k, b in enumerate(blocks):
            gray = (a[b.y:b.y + b.height, b.x:b.x + b.width] * 255).astype("uint8")
            gray = tight_crop(gray)
            out.append(BlockCrop(b.id, b.generation, bi, k,
                                 Image.fromarray(gray).convert("RGB")))
    return out
```

- [ ] **Step 2: Validate block iteration on 2_9 (real data)**

Copy stable inputs to a scratch books dir if the shared dirs are churning (merged 2_9 is
in the main checkout `books/book3/bio/2_merged/`). Run:

```bash
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c "
from src.bio.s4_ocr import iter_blocks
bs = iter_blocks('book3','2_9', books_dir='<stable>/books')
print(len(bs), 'blocks')
from collections import Counter; print(Counter(b.generation for b in bs))
bs[0].img.save('/tmp/b0.png'); print('sizes', [b.img.size for b in bs[:5]])
"
```
Expected: 20 blocks; gen counts `{2:1,3:1,4:2,5:7,6:9}`; the tight crops are far
narrower than the raw 25k-px blocks (e.g. a few hundred–few thousand px wide).

- [ ] **Step 3: Eyeball a tight crop**

Read `/tmp/b0.png` (or a gen-5 block) and confirm it shows exactly one person, blank
margin gone — matching the canonical example.

- [ ] **Step 4: Commit**

```bash
git add src/bio/s4_ocr.py
git commit -m "Bio stage 4: block iteration + tight-crop shim"
```

---

### Task 2: Paddle reader (detect-then-resort) + structural QA checks

**Files:**
- Modify: `src/bio/s4_ocr.py`

**Interfaces:**
- Consumes: `BlockCrop.img`; `src.s6_ocr_paddle.PaddleEngine` (lazy import).
- Produces: `paddle_read(img, engine) -> PaddleResult` where
  `PaddleResult = {columns: list[str], sons: list[str], char_boxes: list[dict],
  qa_flags: list[str]}`. `columns` are RTL-ordered joined column texts;
  `char_boxes` each `{ch,x0,y0,x1,y1,xc,yc,col}`.
- Produces: `_paddle_engine()` (module-level cached PaddleOCR with `return_word_box`).

- [ ] **Step 1: Add the Paddle engine accessor + char-box extraction**

```python
_ENGINE = None
def _paddle_engine():
    global _ENGINE
    if _ENGINE is None:
        from paddleocr import PaddleOCR
        _ENGINE = PaddleOCR(lang="ch")
    return _ENGINE

def _char_boxes(engine, rgb: np.ndarray) -> list[dict]:
    res = engine.predict(rgb, return_word_box=True)
    if not res:
        return []
    d = getattr(res[0], "json", None)
    d = d.get("res", d) if isinstance(d, dict) else res[0]
    wtexts = d.get("text_word") or []
    wboxes = d.get("text_word_boxes") or []
    chars = []
    for line_words, line_boxes in zip(wtexts, wboxes):
        for ch, bx in zip(line_words, line_boxes):
            x0, y0, x1, y1 = (float(v) for v in bx)
            chars.append(dict(ch=ch, x0=x0, y0=y0, x1=x1, y1=y1,
                              xc=(x0 + x1) / 2, yc=(y0 + y1) / 2))
    return chars
```

- [ ] **Step 2: Cluster chars into RTL columns + join**

```python
def _cluster_columns(chars: list[dict], tol_frac: float = 0.7) -> list[list[dict]]:
    if not chars:
        return []
    med_w = float(np.median([c["x1"] - c["x0"] for c in chars])) or 30.0
    tol = max(med_w * tol_frac, 12.0)
    chars = sorted(chars, key=lambda c: -c["xc"])   # RTL
    cols, centers = [], []
    for c in chars:
        if cols and abs(centers[-1] - c["xc"]) <= tol:
            cols[-1].append(c)
            centers[-1] = float(np.mean([x["xc"] for x in cols[-1]]))
        else:
            cols.append([c]); centers.append(c["xc"])
    for col in cols:
        col.sort(key=lambda c: c["y0"])            # top-to-bottom
    for ci, col in enumerate(cols):
        for c in col:
            c["col"] = ci
    return cols
```

- [ ] **Step 3: Structural QA checks (spec §5a)**

```python
def _qa_checks(cols: list[list[dict]]) -> list[str]:
    flags = []
    if not cols:
        return ["no text detected"]
    all_chars = [c for col in cols for c in col]
    med_h = float(np.median([c["y1"] - c["y0"] for c in all_chars])) or 30.0
    centers = [float(np.mean([c["xc"] for c in col])) for col in cols]
    pitches = [abs(centers[i] - centers[i + 1]) for i in range(len(centers) - 1)]
    med_pitch = float(np.median(pitches)) if pitches else 0.0
    for i, p in enumerate(pitches):
        if med_pitch and p > 1.6 * med_pitch:
            flags.append(f"col{i}->col{i+1}: pitch gap {p:.0f} (missed column?)")
    for ci, col in enumerate(cols):
        for j in range(len(col) - 1):
            gap = col[j + 1]["y0"] - col[j]["y1"]
            if gap > 1.2 * med_h:
                flags.append(f"col{ci}: y-gap {gap:.0f} after '{col[j]['ch']}' (dropped char?)")
        span = (col[-1]["y1"] - col[0]["y0"]) if col else 0
        if span > 3 * med_h and len(col) * med_h < 0.5 * span:
            flags.append(f"col{ci}: fill ratio low ({len(col)} chars over {span:.0f}px)")
    return flags
```

- [ ] **Step 4: Assemble `paddle_read` + son parsing (shared parser from Task 4)**

```python
def paddle_read(img: Image.Image, engine) -> dict:
    chars = _char_boxes(engine, np.asarray(img))
    cols = _cluster_columns(chars)
    col_texts = ["".join(c["ch"] for c in col) for col in cols]
    return {"columns": col_texts, "char_boxes": chars,
            "sons": parse_sons(col_texts), "qa_flags": _qa_checks(cols)}
```

- [ ] **Step 5: Validate on the two known crops**

```bash
# regenerate stable crops via iter_blocks; pick the 宪 (son) block and a no-son block
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c "
from src.bio.s4_ocr import iter_blocks, paddle_read, _paddle_engine
bs={b.id:b for b in iter_blocks('book3','2_9', books_dir='<stable>/books')}
eng=_paddle_engine()
for bid,b in bs.items():
    r=paddle_read(b.img, eng)
    print(bid, 'sons', r['sons'], 'flags', r['qa_flags'][:2])
"
```
Expected: at least one gen-5 block yields a 庆-name son; QA flags fire on the
header/name column (fill-ratio) as seen in the design session, not on prose columns.

- [ ] **Step 6: Commit**

```bash
git add src/bio/s4_ocr.py
git commit -m "Bio stage 4: Paddle detect-then-resort reader + structural QA checks"
```

---

### Task 3: Son-clause parser

**Files:**
- Modify: `src/bio/s4_ocr.py`

**Interfaces:**
- Produces: `parse_sons(columns: list[str]) -> list[str]` — from ordered column texts,
  find `生子…名` and return the son given-names (drop the shared generation char if
  present is NOT done here; return names as printed), stopping at `生女`/end.
- Produces: `parse_father(lines) -> str|None`, `parse_daughters`, `parse_dates`
  (best-effort helpers; sons is the gated one).

- [ ] **Step 1: Implement `parse_sons`**

The son clause: a column contains `生子[一二三…]?名` (bore N sons, named), then the
following column(s) hold the son names, until a `生女` column or the block ends. Sons may
carry ordinal prefixes `长/次/三/四/五/幼`. Work on the joined RTL text but keep column
boundaries to know where the son names sit.

```python
_COUNT = "一二三四五六七八九十两"
_ORD = "长次三四五六七八九幼元"
def parse_sons(columns: list[str]) -> list[str]:
    joined = "".join(columns)
    m = re.search(rf"生子[{_COUNT}]*名", joined)
    if not m:
        return []
    tail = joined[m.end():]
    tail = re.split(r"生女", tail, maxsplit=1)[0]   # stop before daughters
    # son names are 1-3 char runs, optionally ordinal-prefixed, separated by nothing;
    # split on ordinals when present, else take contiguous name chars up to a clause word
    tail = re.split(r"[配继殁葬享寿]", tail, maxsplit=1)[0]
    if re.search(rf"[{_ORD}]", tail):
        parts = re.split(rf"(?=[{_ORD}])", tail)
        return [re.sub(rf"^[{_ORD}]", "", p).strip() for p in parts if p.strip()]
    # no ordinals (typical for 1-2 sons): the tail IS the son name(s); one name if short
    tail = tail.strip()
    return [tail] if tail else []
```

- [ ] **Step 2: Validate `parse_sons` on real column outputs**

Feed the Paddle `columns` and the vision `lines` from the 宪 block into `parse_sons`.
Expected: the 宪炳-type block (`生子一名`,`庆林`) → `["庆林"]`; the `生子二名`,`庆鸿`,`庆亮`
block → `["庆鸿","庆亮"]`; a `配失考`/no-`生子` block → `[]`.

```bash
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c "
from src.bio.s4_ocr import parse_sons
print(parse_sons(['子次棋','宪炳','生子一名','庆林','生女五','长春华适杨柳寨']))
print(parse_sons(['子之禄','纪有','生于同治己巳年','配失考']))
"
```
Expected: `['庆林']` then `[]`.

- [ ] **Step 3: Add father/daughter/date helpers (best-effort)**

```python
def parse_father(lines: list[str]) -> str | None:
    if not lines:
        return None
    m = re.match(r"子[之次三四五六七八九幼]?(.)", lines[0])
    return m.group(1) if m else None

def parse_daughters(columns: list[str]) -> list[str]:
    joined = "".join(columns)
    m = re.search(rf"生女[{_COUNT}]*", joined)
    if not m:
        return []
    tail = joined[m.end():]
    return re.findall(rf"[{_ORD}](.{{1,2}}?)适", tail)

def parse_dates(columns: list[str]) -> dict:
    joined = "".join(columns)
    birth = re.search(r"生于([^殁葬配]{2,12})", joined)
    return {"birth": birth.group(1) if birth else None}
```

- [ ] **Step 4: Commit**

```bash
git add src/bio/s4_ocr.py
git commit -m "Bio stage 4: son-clause parser + best-effort father/daughter/date helpers"
```

---

### Task 4: Vision-subagent reader

**Files:**
- Modify: `src/bio/s4_ocr.py`
- Create: `src/bio/s4_vision_prompt.txt` (the validated prompt, one source of truth)

**Interfaces:**
- Produces: `vision_read(crop_path: str) -> dict` = `{lines, father_char, name, sons}`.
  The stage saves each block crop to a temp path, dispatches the vision subagent with the
  prompt, and parses its RTL lines (line 1 father header, line 2 name, 3+ prose).

- [ ] **Step 1: Save the validated prompt**

Write `src/bio/s4_vision_prompt.txt` with the exact validated text (spec §2a): transcribe
one 族谱 person block; vertical columns read RIGHT-TO-LEFT; one line per column, RTL;
line 1 = top-right horizontal header `子之X`/`子次X` (father marker); line 2 = the
person's own vertical bold name; lines 3+ = bio prose columns; `?` for unreadable;
transcription only.

- [ ] **Step 2: Implement `vision_read` (dispatch + parse)**

The dispatch mechanism is the Agent tool (the runner invokes it). `vision_read` takes the
subagent's returned text and parses it:

```python
def parse_vision(text: str) -> dict:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    father = parse_father(lines[:1]) if lines else None
    name = lines[1] if len(lines) > 1 and lines[1] != "?" else None
    sons = parse_sons(lines[2:])
    return {"lines": lines, "father_char": father, "name": name, "sons": sons}
```

`vision_read(crop_path)` is a thin wrapper documenting the dispatch contract: caller
(the `ocr_section` runner, executed by an agent) dispatches the Agent tool with the
prompt + `crop_path`, then calls `parse_vision` on the result. For the batch runner,
Stage 4 writes all block crops to `books/{book}/bio/4_ocr/{stem}/crops/{id}.png` and the
executing agent dispatches one subagent per crop (parallel), collecting `{id: text}`.

- [ ] **Step 3: Validate `parse_vision` on the recorded transcripts**

```bash
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c "
from src.bio.s4_ocr import parse_vision
t='子次棋\n宪炳\n生于公元一九四\n六年二月初八日\n戌时\n配温八秀生于公\n元一九四七年十\n一月二十三辰时\n生子一名\n庆林\n生女五\n长春华适杨柳寨'
print(parse_vision(t))
"
```
Expected: `father_char='棋'`, `name='宪炳'`, `sons=['庆林']`.

- [ ] **Step 4: Commit**

```bash
git add src/bio/s4_ocr.py src/bio/s4_vision_prompt.txt
git commit -m "Bio stage 4: vision-subagent reader (prompt + RTL-line parser)"
```

---

### Task 5: Reconcile + section runner + output

**Files:**
- Modify: `src/bio/s4_ocr.py`

**Interfaces:**
- Consumes: `paddle_read` result, `parse_vision` result, tree oracle names.
- Produces: `reconcile(paddle, vision, tree_name) -> dict` — the block record (spec §5
  schema): `name`, `father_char`, `sons:[{name,agreed}]`, `daughters`, `birth`, `paddle`,
  `vision`, `qa_flags`.
- Produces: `ocr_section(book, sec, books_dir, data_dir, vision_texts) -> dict` — full
  section record incl. `sons_by_gen`, `validation` (vs the next-gen tree names).

- [ ] **Step 1: Implement `reconcile`**

```python
def reconcile(paddle: dict, vision: dict, tree_name: str | None) -> dict:
    p_sons, v_sons = set(paddle["sons"]), set(vision["sons"])
    union = list(dict.fromkeys(paddle["sons"] + vision["sons"]))  # keep order, dedupe
    sons = [{"name": s, "agreed": (s in p_sons and s in v_sons)} for s in union]
    name = vision.get("name") or tree_name
    flags = list(paddle["qa_flags"])
    if vision.get("name") and tree_name and vision["name"] != tree_name:
        flags.append(f"name: vision '{vision['name']}' != tree '{tree_name}'")
    if p_sons != v_sons:
        flags.append(f"sons: paddle {sorted(p_sons)} != vision {sorted(v_sons)}")
    return {"name": name, "father_char": vision.get("father_char"),
            "sons": sons, "daughters": parse_daughters(paddle["columns"]),
            "birth": parse_dates(paddle["columns"]).get("birth"),
            "paddle": {"columns": paddle["columns"], "sons": paddle["sons"]},
            "vision": {"lines": vision["lines"], "sons": vision["sons"]},
            "qa_flags": flags}
```

- [ ] **Step 2: Implement `ocr_section` + tree-oracle names per generation**

Get the subgraph stem for the section (via `map_sections_to_stems` over all merged
sections vs `tree_counts_by_stem`), and the tree node names grouped by generation, in
RTL/tree order, so block k in gen g ↔ tree node k. Build each block record; compute
`sons_by_gen` and `validation` (gen g+1 names == union of gen g sons).

```python
def _tree_names_by_gen(data_dir, book, stem):
    names = {}
    for line in open(os.path.join(data_dir, f"{book}_stitched.jsonl")):
        n = json.loads(line)
        prov = (n.get("notes","") or "").split(" | ",1)[0].split("/",1)[0]
        s = prov.rsplit("_",1)[0] if prov else ""
        if s == stem:
            names.setdefault(n["generation"], []).append(n["name"])
    return names
```

`validation`: for the section's deepest gen G that has sons, check
`set(union of gen (G-1) sons) vs set(tree names at gen G)`; for 2_9 that's gen5 sons vs
gen6 names. Report `missing_from_ocr`, `extra_in_ocr`, `match`, and agreed-count.

- [ ] **Step 3: Wire `ocr_book` + CLI + `_save_qa` + `--no-vision`**

`ocr_book` iterates sections, writes `books/{book}/bio/4_ocr/{sec}.json`, saves crops for
the vision pass, and (unless `--no-vision`) expects a `vision_texts` map (the executing
agent supplies it by dispatching subagents). `_save_qa` writes per-block overlays with
char boxes + flagged columns red. CLI: `--book --sections --books-dir --data-dir
--no-vision --log-level`.

- [ ] **Step 4: Run Paddle-only end-to-end on 2_9**

```bash
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -m src.bio.s4_ocr \
  --book book3 --sections 2_9 --books-dir <stable>/books --data-dir data --no-vision
cat <stable>/books/book3/bio/4_ocr/2_9.json | python -m json.tool | head -60
```
Expected: 20 block records; `validation` compares gen5 sons-union vs the 9 gen6 names;
report the diff (partial match acceptable first pass — that drives iteration).

- [ ] **Step 5: Commit**

```bash
git add src/bio/s4_ocr.py
git commit -m "Bio stage 4: reconcile + section runner + validation + QA overlay"
```

---

### Task 6: Ensemble run on 2_9 + gate

**Files:**
- Modify: `src/bio/s4_ocr.py` (only if the run surfaces parse/reconcile bugs)

**Interfaces:** none new.

- [ ] **Step 1: Generate all 20 block crops for the vision pass**

Run `ocr_book` in a mode that writes `4_ocr/2_9/crops/{id}.png`, then the executing agent
dispatches one vision subagent per crop (parallel, batches of ~5) with
`src/bio/s4_vision_prompt.txt`, collecting `{id: transcript}`.

- [ ] **Step 2: Reconcile with vision + re-run validation**

Feed the `vision_texts` map into `ocr_section`; write the final `2_9.json`.

- [ ] **Step 3: Check the P0 gate**

```bash
PYTHONPATH=. /opt/miniconda3/envs/zeng/bin/python -c "
import json; d=json.load(open('<stable>/books/book3/bio/4_ocr/2_9.json'))
v=d['validation']; print('match', v['match'], 'missing', v['missing_from_ocr'], 'extra', v['extra_in_ocr'])
print('agreed sons:', sum(s['agreed'] for b in d['blocks'] for s in b['sons']))
"
```
Expected target: gen5 sons-union reconstructs the 9 gen6 names
(`庆林 庆鸿 庆亮 庆海 庆荣 庆华 庆财 庆铭 庆粮`). If short, use `qa_flags` + the QA overlay
to find which block/column dropped a son and iterate the parser/clustering.

- [ ] **Step 4: Eyeball the QA overlay + commit results**

Open a couple of `books/book3/qa/bio_s4/2_9/*.png`. Commit any code fixes.

```bash
git add src/bio/s4_ocr.py
git commit -m "Bio stage 4: ensemble run on 2_9 passes the sons-union gate"
```

---

### Task 7: Docs — history + memory

**Files:**
- Modify: `docs/history.md` (add a dated Era entry before the snapshot section)
- Modify: `docs/handoff.md` (note Stage 4 status + remaining: generalize to more sections)

- [ ] **Step 1: Add a short Era entry to `docs/history.md`** (per memory `log-checkpoints-to-history`)

- [ ] **Step 2: Note Stage 4 state + next steps in `docs/handoff.md`**

- [ ] **Step 3: Commit**

```bash
git add docs/history.md docs/handoff.md
git commit -m "docs: log bio stage 4 (field OCR ensemble) checkpoint"
```

---

## Self-Review

**Spec coverage:**
- §1 inputs / tight crops → Task 1. §2 Paddle detect-then-resort → Task 2. §2a ensemble
  (vision reader + reconcile) → Tasks 4, 5. §3 algorithm → Task 2. §4 field parsing →
  Task 3. §5 output schema → Task 5. §5a QA checks → Task 2. §6 validation → Tasks 5, 6.
  §7 module/CLI → Task 5. §8 out-of-scope respected (no node attach/stitch). §9 open
  questions (cluster tol, son regex, header/name) → exercised in Tasks 2/3/6.
- Gap handled: S3 emits wide crops today → Task 1 tight-crop shim (no-op when S3 tightens).

**Placeholder scan:** all code steps contain real code; validation steps use real 2_9
data and the concrete gen6 name set; `<stable>` is the one intentional path token
(the executor's chosen stable books dir — see Task 1 Step 2).

**Type consistency:** `paddle_read`→dict{columns,char_boxes,sons,qa_flags};
`parse_vision`→dict{lines,father_char,name,sons}; `reconcile(paddle,vision,tree_name)`;
`parse_sons(columns)` used identically in Tasks 2/3/4; `BlockCrop` fields consistent.

**Note on test convention:** per the repo's no-unit-tests rule and the user's explicit
instruction, tasks validate by running on real 2_9 data + the ground-truth gate + QA
overlay instead of pytest. This deliberately departs from the writing-plans TDD default.
