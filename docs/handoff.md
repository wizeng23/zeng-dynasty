# Handoff — 2026-09-14 (evening), Book 2 fixes for Book 3

Rolling handoff note for the next session. Read `docs/history.md` Era 10 and
`docs/bridge-revisit-notes.md` (bottom sections) for the why; this file is the
what-is-left.

## Bio stage 4 (field OCR) — status (2026-09-18)

`src/bio/s4_ocr.py` is built and validated on Book 3 section 2_9 (Era 14). Branch
`stage-4-bio-ocr`. Ensemble = Paddle detect-then-resort + a vision Claude subagent;
P0 = sons; output `books/{book}/bio/4_ocr/{stem}.json`. On 2_9 the gen-5 sons-union
reconstructs 8/9 gen-6 tree names, 0 extras; the 1 miss (`庆粮`) is a real bio↔graph
conflict (宪烘's bio lists 庆财/庆铭, tree says 庆粮), flagged not hidden.

**Remaining for stage 4:**
- **Vision pass is driven by the executing agent**, not the CLI: `--no-vision` runs
  Paddle-only; for the ensemble, `save_crops(book, sec)` writes crops, an agent
  dispatches one vision subagent per crop (`vision_prompt(crop_path)`), and the
  `{block_id: text}` map is passed to `ocr_section(..., vision_texts=...)`. Consider a
  gating step (only run vision on Paddle-QA-flagged blocks) before full-book runs.
- **Tight-crop dependency:** stage 3 still emits wide blocks; s4's `tight_crop` shim
  handles it (no-op once s3 tightens). Dense blocks (no ≥400px gap) stay wide — Paddle
  handles them, but confirm on other sections.
- **Generalize:** run 66_77 and a couple more Book 3 sections; then Book 4.
- **Investigate the 宪烘/庆粮 bio↔graph conflict** — candidate stitch/graph fix.
- QA overlay generation re-OCRs each block (slow); fine for one section, batch/background
  for a book.

## Hard rules in force

- **Do NOT re-run Book 2 stages 3–7** (`s3_segment`, `s4_merge_pages`,
  `s5_build_tree`, `s6_ocr --populate`, `s7_stitch`) unless William says so. Book 2's
  results carry his hand fixes; a full Stage 5 run is ~50 min. Frozen, restorable
  snapshot: `books/book2/frozen_2026-09-14/` (README inside). Validate code changes
  in memory against it (recipe below). Never write into `books/book2/` or `data/book2*`.
- **Do not run Book 1 regression checks** (William: Book 1 is confirmed correct).
- Repo is public; agents never `git push`. Commit locally.
- Long runs: the harness kills background Bash after 10 min — use `nohup … &` and a
  `Monitor` on the log.

## Validation recipe (used for every fix today)

Parse a frozen graph in memory with its recorded fills drawn on, compare to the
hand-fixed sidecar:

```
F=books/book2/frozen_2026-09-14; g=get_image(f'{F}/4_graphs/{stem}.png')
for suffix in ('imaginary','nicks'): draw each [r0,c0,r1,c1] with bt.draw_bridge(g,r0,c0,c1)
nodes=bt.parse_graph(g, bt.config_for('book2'), graph_stem=stem)
compare len(nodes) and Counter(len(n.children)) with f'{F}/4_graphs/{stem}.parse.json'
```
Expected today (all SAME, 0 orphans): 0_3 49, 22_23 29, 58_62 76, 69_82 190,
114_120 88, 133_133 6. 8_10 = 33 vs frozen 32: the one extra is the line-fragment
phantom `8_10_9` (stays a `delete` in `data/book2_fixes.json`). `parse_graph` does
NOT bridge — hence drawing the recorded fills first. 36_52/69_82 take ~1 min each.

## Landed today (all on `main`, tests green: `PYTHONPATH=. python -m pytest tests -q`)

Stage 4: seam endpoints must be real line ends (ADF specks excluded).
Stage 5: stepped-bar stroke read; orphan bridging port (speck floor, step-following
trace, targeted gate, nick/bridge split, misread-parent bar lookup); glyph-aware ink
(`_row_kinds`/`_line_only_rows`, bare lines are not name ink); walk-out only on a
stroke crossing the box edge; child endpoints at each riser's own bottom;
`merge_max_shift` 60→90 nearest-wins; horizontal hairline refill between bar pieces
in `find_lines`; continuation-stub rule in `merge_nodes` (pinched hang-line);
bridging on single-page graphs too (no-op when nothing is orphaned).
Stage 7: only `{graph}_0` roots are duplicates; non-leaf canonical fallback; chain
folding. Post-fixes layer `src/s5_fixes.py` + `data/book2_fixes.json` (delete /
merge / recrop by provenance, `--ocr`). QA: width cap 16k, per-fill crops, cyan
nicks, only detected endpoints circled, `books/book2/qa/fixes.html` before/after
page (ad hoc script, not yet a tool), OCR server `--books` filter.

On the next Book 2 re-run, `data/book2_fixes.json` should reduce to **just `delete
8_10_9`**. The `merge 106_113_5 -> 106_113_1` is now ALSO redundant — verified
2026-09-14: `bridge_orphans` on the raw frozen 106_113 graph auto-resolves the sole
orphan with a single nick (col 14705), giving 100 nodes that match the hand-fixed
frozen parse node-for-node (0 unmatched, 0 child-set mismatches). The
`find_lines` hairline-nick refill (commit 1a9917b) welds the broken bar before
orphan detection, so the automatic gate now passes. Prune the merge on the next
run.

## Remaining work items, in order

1. ✅ **DONE (2026-09-14)** — **Stage 3 whitening anchored to the inner border.**
   Implemented in `src/s3_segment.py`: `bottom_inner_border_row(page)` detects the
   2-band bottom frame (inner line = 2nd band up), `whiten_margins(a,
   bottom_anchor=)` whitens from `bottom_anchor - WHITEN_BOTTOM_FROM_BORDER` (=180)
   down, wired into `segment()`. 134-page Book 2 in-memory scan: min clearance rose
   from 8px to 57px (p42), NO page has glyph ink in the band, all 134 find a real
   2-band frame (border sits 56–60px above the page bottom). Tests green
   (`test_s3_whiten.py`, xfail removed). Book 2 downstream NOT re-run.
2. ✅ **DONE (2026-09-14, was a stale diagnosis)** — **106_113 gate refusal.** No
   longer refuses: `bridge_orphans` auto-resolves the sole orphan (see the pruning
   note above). No code change needed; the `merge 106_113_5 -> 106_113_1` post-fix is
   now redundant and should be dropped on the next Book 2 re-run.
3. ✅ **DONE (2026-09-14)** — **Bridging speed.** Root cause was not the
   re-parsing itself but `find_lines`' per-component `np.where(labels == label)`,
   a full 100M+-element array scan repeated once per kept component (~9s x32 on
   106_113). Now each component is extracted from its own cv2 bounding box (identical
   pixel sets, ~60x faster): `parse_graph` 12.9s→4.1s, `bridge_orphans` (106_113)
   78s→23s — Book 2 Stage 5 ~50min→~15min, and it scales the same for Book 3's big
   graphs. No caching needed. (If more is wanted later, the re-parse-only-touched-
   region idea still stands, but the bottleneck was the array scan.)
4. ⚠️ **INVESTIGATED, DEFERRED (2026-09-14) — do NOT widen the window.** The
   handoff's ±120→±160 half-width is a **net regression**: an in-memory sweep of
   all frozen Book 2 graphs shows 173 boxes grow ≥15px at 160, a large cluster to
   exactly 320px (the full 2×160 window) — i.e. the any-ink trim starts pulling in
   the neighbour ~300px away (8_10_8 → 430px, 8_10_9 → 540px). 160 doesn't even fix
   the wide glyphs — 114_120_54 毓塘 just re-clamps at the wider edge. Meanwhile the
   marquee cases (114_120_52/53/54 毓援/毓棋/毓塘, 8_10_3, 8_10_30, 58_62_68) are
   ALREADY rescued by the walk-out (`_walk_out_of_ink`) — box escapes the window.
   Only ~5 nodes stay clamped (0_3_1, 4_5_1/5, 58_62_6/17), all edge-column ink of a
   continuing glyph, marginal. A streak-tolerant column trim (a column analog of
   `_line_only_rows`) is the ONLY safe way to widen, and it's the high-risk part
   (must not erase thin real strokes) for a small payoff on a frozen book. Verdict:
   leave `NAME_HALF_WIDTH=120`; revisit ONLY if Book 3 actually shows cut wide
   glyphs the walk-out misses. — Separately, 8_10 聞詣's grandpa note is a **manual
   final-assembly step**, not a Stage 5 fix: it needs hand-measured pixel coords AND
   the note text copied into 聞詣's `notes` (see `docs/final_manual_steps.md`);
   `ignore_regions` only blanks pixels, so it can't do the notes half. Handle at
   Book 2 final assembly.
5. **Stage 4 shift carry-over** (low priority, NOT done) — a page whose seam has no
   matched line end is concatenated top-aligned (`_concat_top_aligned`), producing
   7–60px bar steps that bridging then repairs (69_82 steps at 11 of 12 seams). Carry
   the neighbouring seam's `matched_shift` instead. Left as-is: bridging already
   handles the steps, so this is robustness-only, and touching seam alignment risks
   the merge. Revisit only if a Book 3 graph produces steps bridging can't repair.
6. ✅ **DONE (2026-09-14)** — **QA fixes page.** `scripts/qa/s5_fixes.py`: reads
   `data/{book}_fixes.json` + the applied jsonl/crops, renders one section per
   delete/merge/recrop (name crop + graph-at-box for recrops), self-contained
   base64. `python -m scripts.qa.s5_fixes --book bookN`. Note it OVERWRITES
   `qa/fixes.html`, so it was NOT run against `books/book2/qa/` (that holds
   William's hand-narrated page); it regenerates fresh for Book 3.
7. ✅ **DONE (2026-09-14)** — **Cosmetic bot after merge.** `s5_fixes` merge now
   copies the phantom's sidecar `bot` to the real node's, so the QA red edge no
   longer fans 60px off the hang-line (114_120 毓援). QA-overlay only. (The
   continuation-rule variant of the same cosmetic was not chased — only the merge
   path produced a visible fan.)

## Status: all 7 remaining items resolved (2026-09-14)

Items 1, 2, 3, 6, 7 landed as code/tooling; 4 was investigated and DEFERRED
(widening the window regresses); 5 is left as-is (bridging already handles the
steps). Nothing above requires a Book 2 re-run. What's left is William-side (OCR
review, below) and the Book 3 pipeline run (further below).

## OCR review items for William (Book 2)

- 8 section roots whose OCR differs from the canonical leaf (each reconnects a
  section): 贞烈/贞列, 贞熊/贞能, 贞杰/贞木, 贞斗/贞升, 贞亮/贞光, 克太, 贞富, 贞年.
- 7 three-char …子 names whose last char the OLD crops cut: 36_52_175 添川, 95_99_40
  小生, 95_99_41 新年, 95_99_62 祖底, 106_113_94 细满, 114_120_73 新福, 114_120_82 大满
  (+ 133_133_4 尚旻 lost 10 rows). Ink is back in `3_crops/`; a Stage 3+ re-run
  would restore the crops; until then override in the OCR tool.
- 114_120_55 毓塘 re-OCR'd as 毓搪 (0.90, not flagged) after the widened recrop.
- 65 single-char names are likely truncations (传 ×24).
After corrections: `s6_ocr.apply_names('book2')` then `python -m src.s7_stitch --book
book2` (seconds). Stitch today: 1549 nodes, 9 roots (main + the 8 above), 0 orphans.

## Book 3 plan

Stages 1–2 done and human-verified (41 graph / 251 bio pages). The stale
2026-09-01 Stage 3 crops and Stage 4 graphs (old whitening, pre seam-speck fix)
were DELETED (2026-09-14) — regenerate from scratch after item 1 below.

**Stage 4 merges ADJACENT page numbers ONLY.** A subtree spans a run of
physically consecutive pages; a continuation page must be its predecessor's
number + 1. If `s4_merge_pages` ever tries to merge across a page-number gap,
that is a BUG (not a wide subtree) — it means a page that should have been
flagged as a subtree start in `starts.json` was not. We do NOT silently gate the
attempt: `merge_pages` **raises** on a gap so the latent bug surfaces instead of
being hidden. This bites Books 3 & 4 specifically: their biography pages are
interleaved and absent from `starts.json`, so the old continuation loop silently
welded the next tree page across the missing bio pages (page 17→64), producing
bogus graphs like `10_138` / `202_247` that span huge number gaps. If a
regenerated Book 3 run hits that `ValueError`, fix the upstream start-detection
(Stage 3) — do not loosen the adjacency check.

Re-run Stage 3 (after item 1 above) and Stage 4, then Stage 5 detached (bridging is
slow on 202_247 / 10_138), then `s5_fixes` (create `data/book3_fixes.json`), OCR,
stitch, QA (`scripts.qa.s5_parse --book book3`, serve `books/book3/qa` on 8785).
`BOOK_CONFIGS['book3'].num_pages = 292`. Expect bio-interleaved page ranges in graph
stems.

## Servers (local, started this session)

Parse QA: Book 1 :8765 (William's), Book 2 :8775 (+ `fixes.html`).
OCR review: Book 1 :8766, Book 2 :8776 (`python -m scripts.qa.s6_ocr --books book2 --port 8776`).
