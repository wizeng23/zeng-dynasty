# Parse Editor — editable QA UI for Stage-5 parse corrections

**Status:** design / handoff (2026-09-15). Ready for a build session.
**Author of design:** prior session (with William).
**Goal:** replace the slow "William describes a broken region → agent probes pixels →
agent hand-edits JSON" loop with a browser UI where William directly fixes a
subgraph's parse (connect a child, delete a phantom, move a box), and the UI writes
a per-subgraph **manual override** that becomes ground truth for that graph.

This mirrors the OCR review tool (`scripts/qa/s6_ocr.py`) in mechanics: a
`ThreadingHTTPServer` serving one HTML page + JSON endpoints that persist edits.

---

## Why (the problem it solves)

Book 3's large graphs (e.g. `78_84`: 100 nodes) have many small parse errors —
phantom nodes, children detached by hairline/misprint gaps, spurious green bridges.
We already decided (see below) to fix these with a **per-subgraph manual override**
that *replaces* Stage-5's output for that graph. Authoring those overrides by hand
over a chat back-and-forth is too slow across ~10 remaining multi-root graphs.
An editor makes each graph a few minutes of direct manipulation.

## What already exists (build on these, do not reinvent)

1. **The override format + concept — DONE.** A complete, self-contained per-subgraph
   ground-truth file at `data/{book}_manual/{stem}.json`:
   ```json
   {
     "stem": "64_65",
     "nodes": [
       {"id": 0, "box": [l,t,r,b], "top": [row,col], "bot": [row,col],
        "empty": false, "children": [3,2,1], "father": -1},
       ...
     ],
     "imaginary": [[r0,c0,r1,c1], ...],   // green bridges to draw (usually [] after fixing)
     "nicks": [[r0,c0,r1,c1], ...]        // cyan nicks
   }
   ```
   `id` is local to the graph. `children` is eldest-first (RTL: rightmost column
   first). `father` = -1 for a root. `box` is `[left, top, right, bottom]` in graph
   pixels; `top`/`bot` are `[row, col]` line-end points the QA draws edges from.
   Seeded from the best parse, then corrected. `64_65.json` is a worked example
   (already hand-corrected: 34 nodes, 1 root, no bridge).

2. **A seeder — DONE (scratch).** `scratchpad/seed_manual.py <book> <stem>` parses one
   graph with the current `src/find_lines` (which now includes the vertical
   riser-gap fill) and dumps the full node list to `data/{book}_manual/{stem}.json`.
   Fold this into a real `scripts/qa/` entry or a function the server calls.

3. **A read-only single-graph QA — DONE (scratch).** `scratchpad/qa_one.py <book>
   <stem>` renders the compare strip (raw pages over crops) + the parse overlay to a
   standalone HTML. `scratchpad/qa_manual.py <book> <stem>` does the same but renders
   from the override file. The editor replaces these with a live canvas.

4. **The overlay drawing — REUSE.** `scripts/qa/s5_parse.py:draw_parse_overlay(a,
   parse, imaginary, nicks)` draws boxes (red), phantom (orange), bridges (green),
   nicks (cyan), and parent→child edges from `children_top`. `stacked_compare(book,
   start, end, seg_cfg, books_dir)` builds the pages/crops strip. The server can
   render a base PNG once and draw the *editable* layer (boxes, edges) in the
   browser as an SVG/canvas overlay on top of the graph image, so edits are live.

5. **The OCR tool — MIRROR its server pattern.** `scripts/qa/s6_ocr.py`:
   `ThreadingHTTPServer` + `BaseHTTPRequestHandler`; `do_GET` serves `/` (HTML) and
   `/data` (JSON); `do_POST` persists edits to `data/{book}_overrides.json`. Same
   `--book`/`--port` CLI. Copy this skeleton.

6. **Stage-5 consumption — TODO, small.** `src/s5_build_tree.build_tree` must, for
   any graph with a `data/{book}_manual/{stem}.json`, emit those nodes verbatim
   instead of parsing (skip find_lines/bridge for that graph). Also update the big
   QA (`scripts/qa/s5_parse.py`) to render from the override when present. Until
   this lands, the override only drives the editor's own QA.

## Prior decisions (already made with William — honor these)

- **Topology-only overrides.** The editor edits *relationships* (father/children),
  deletes phantoms, and may move a name box; it does NOT need full freehand
  geometry authoring. (Boxes/ends come from the seed; moving a box is a nicety.)
- **Save the override to the subgraph** (keyed by graph stem + local id), not to the
  global renumbered `data/book3.jsonl` ids. Immune to other graphs re-parsing.
- **The override is the source of truth** for its graph; the QA renders exactly what
  the override says (no re-derivation), so what William sees is what ships.
- **`src/find_lines` gap-fill already landed** (this session): symmetric short-gap
  fill heals ≤10px vertical riser breaks bounded to riser ink (won't weld glyphs).
  Re-seeding a graph picks up its benefit. The editor handles the *residual* errors
  the auto-fill can't (misprints, spurious bridges, wrong sibling order).
- **Do NOT re-run/alter Books 1 & 2** (frozen). This tool is for Book 3 (then 4).

## The editor — required capabilities (MVP)

Canvas = the graph PNG (or the compare strip above + parse layer below). On the
parse layer, each node is a draggable/​selectable box with its edges drawn to its
children. Operations, each writing straight to `data/{book}_manual/{stem}.json`:

1. **Delete a phantom node.** Click a box → Delete. Removes the node; its children
   become roots unless reparented (see #2). Orange (empty) boxes are the usual
   targets. (64_65 had a riser-fragment phantom; 78_84 has several.)
2. **Reparent / connect a child.** Select a child, then click its correct parent →
   sets `father` and inserts into the parent's `children` (kept eldest-first by
   column). This is the "connect a child detached by a misprint" case (78_84 p82:
   `庆烛` has no line to `宪林`). If the child has no node at all (dropped entirely),
   see #3.
3. **Add a missing node.** Draw a box on the canvas over a name the parser dropped
   (a misprint gap ate its riser so it never became a component). Assign a new local
   id, set its `box`, a `top`/`bot` (from the box, or click the line-end), and
   parent it. (78_84: `昭溪` was dropped entirely.)
4. **Delete / redraw a spurious bridge.** Green bridges (`imaginary`) that wrongly
   join two subtrees: click to delete the entry. (78_84 had ~4 bad green bridges.)
5. **Move / resize a name box.** Drag box edges to fix a clipped or drifted crop.
   Writes the node's `box`; the crop is re-cut from the graph png on save (mirror
   `s5_fixes.py`'s recrop: `graph[top:bottom, left:right]`). Re-OCR is a later step.
6. **Set sibling order** (optional): reorder a parent's `children`. Default is
   auto-sort by column (RTL/eldest-first); expose a manual override for the rare
   printed-out-of-order case.

Nice-to-haves (not MVP): undo/redo, "seed this graph" button (runs the seeder),
"validate" (no dangling father/child refs, every non-root has a father, 1 root
expected — warn if >1), keyboard nav between nodes like the OCR tool.

## Server endpoints (mirror s6_ocr.py)

- `GET /` → the HTML/JS editor page.
- `GET /graph?stem=64_65` → the base graph PNG (and/or compare strip PNG).
- `GET /parse?stem=64_65` → the current override JSON (seed it on first request if
  no file exists yet, by calling the seeder). Include image dims for canvas scaling.
- `POST /save?stem=64_65` → write the edited node list back to
  `data/{book}_manual/{stem}.json`; re-cut any moved boxes' crops.
- `GET /list` → graphs for `--book`, each with node/root count + whether an override
  exists, so William can see which still need work (like the OCR tool's progress).

CLI: `python -m scripts.qa.s5_edit --book book3 --port 8787` (pick a free port; OCR
uses 8766/8776, parse-QA 8785/8786).

## Validation / done criteria

- Editing `78_84` to 1 root, `昭溪`/`庆烛` connected, phantoms gone, bridges cleared —
  saved and re-openable, with the QA rendering exactly that.
- `src/s5_build_tree.build_tree` emits overridden graphs from their files; a full
  Book 3 build produces a `data/book3.jsonl` reflecting every override; the big QA
  (`scripts.qa.s5_parse --book book3`, port 8785) shows the corrected graphs.

## Known Book-3 graphs still needing correction (as of 2026-09-15)

Multi-root after the gap-fill (in-memory counts): `78_84` (4 roots, ~4 bad bridges,
`昭溪` dropped, `庆烛` detached, phantom under `宪松`), `134_138` (4), `249_250` (2),
`261_262` (2), `284_284` (2), `10_17` (fixes already specified below). `64_65` is
DONE via override (worked example). Others (`0_1`, `182_182`, `202_203`, `272_272`,
`280_280`) parse to 1 root already.

### 10_17 — corrections already decided (apply via override)
- Re-parent the 5 orphaned gen-2 siblings onto `传煦`: children eldest-first =
  `[纪莒, 纪芦, 纪苏, 纪万, 纪芹, 纪佳]` (ids ascending = oldest→youngest); delete the
  phantom no-name root that held the younger five.
- `宪伟` → 4 children eldest-first `[庆海, 庆连编继, 庆龙, 庆湖]`; delete the phantom
  `○`; reconnect `庆龙`'s group (was a separate broken branch).

## Files map

- Override files: `data/{book}_manual/{stem}.json` (worked example: `64_65.json`).
- Scratch prototypes to promote: `scratchpad/seed_manual.py`, `qa_one.py`,
  `qa_manual.py` (in the session scratchpad; copy their logic into `scripts/qa/`).
- Mirror: `scripts/qa/s6_ocr.py` (server), `scripts/qa/s5_parse.py` (overlay draw),
  `src/s5_fixes.py` (recrop-from-graph logic, provenance keying).
- Consumer to add the override hook: `src/s5_build_tree.py:build_tree`.
```
