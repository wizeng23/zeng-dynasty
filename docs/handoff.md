# Handoff — 2026-09-14 (evening), Book 2 fixes for Book 3

Rolling handoff note for the next session. Read `docs/history.md` Era 10 and
`docs/bridge-revisit-notes.md` (bottom sections) for the why; this file is the
what-is-left.

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

On the next Book 2 re-run, `data/book2_fixes.json` should reduce to: `delete 8_10_9`
and `merge 106_113_5 -> 106_113_1` (everything else now happens automatically —
verify, then prune the file).

## Remaining work items, in order

1. **Stage 3 whitening anchored to the inner border** — test written and xfail-marked:
   `tests/test_s3_whiten.py` (remove the xfail marker when implementing). Implement in
   `src/s3_segment.py`: `bottom_inner_border_row(page)` = top row of the second
   full-width (>=50% coverage) band from the bottom of the UNTRIMMED page (outer frame
   is the lowest band; inner line ~40px above it; if only one band, use it);
   `whiten_margins(a, bottom_anchor=None)` whitens rows >= `bottom_anchor -
   WHITEN_BOTTOM_FROM_BORDER` when given (anchor in trimmed coords = border row −
   `_top_border_cut(page)`), else the old fixed 200px; `WHITEN_BOTTOM_FROM_BORDER =
   180` (lowest Book 2 glyph ends ~246px above the inner border on p96; today's
   effective reach is ~234px → 12px clearance). Wire it in `segment()`. Validate with
   the clearance scan over all 134 Book 2 pages (`trim_borders` → boundary − lowest
   glyph row with >=25 inked px): min clearance should rise from 8px (p42) to ≥ ~60px
   and NO page may have glyph ink inside the band. The Book 2 crops are already
   regenerated with the current code (`books/book2/3_crops/`) but downstream is NOT.
2. **106_113 gate refusal** — the orphan bar at (2091,15040) has nick candidates
   (17145→17146, 21982→21985, 22001→22009) that DO connect it to 尚恕's hang-line
   (尚恕 then has 6 children) but the re-parse also turns two other bars, at
   (2074,5878) 3 kids and (2091,2612) 4 kids, into orphans, so `bridge_resolves`
   refuses (correctly). Find out why those two lose their parent when the far-right
   bar joins (suspect: the joined component's band read picks a different top /
   the flush-bar collapse changes with the new min row). Until then it is the
   `merge 106_113_5 -> 106_113_1` post-fix.
3. **Bridging speed** — the gate re-parses the whole graph per candidate (Book 2 Stage 5
   ~50 min; 36_52 alone ~25). Cache `find_lines` labelling and re-parse only the
   components touched by a fill, or restrict the re-parse to the orphan's page span.
   Matters for Book 3's 202_247 / 10_138 graphs.
4. **Name window ±120 → ±160 + streak-tolerant column trim** — `_tight_ink_box`
   `NAME_HALF_WIDTH`; the column trim is any-ink, so ADF smear streaks beside a glyph
   add 10–34px (8_10 宏善/贞院, 58_62 传炘). Drop columns whose ink is only a thin
   vertical streak (like the row-side `_line_only_rows`). Also 8_10 闻诣 needs a
   book2 `ignore_regions` entry for grandpa's handwritten note flush against it (see
   `docs/final_manual_steps.md`; the note text goes into 聞詣's `notes`).
5. **Stage 4 shift carry-over** — a page whose seam has no matched line end is
   concatenated top-aligned (`_concat_top_aligned`), producing 7–60px bar steps that
   bridging then repairs (69_82 steps at 11 of 12 seams). Carry the neighbouring
   seam's `matched_shift` instead. Low priority now that steps are handled.
6. **QA fixes page** — `books/book2/qa/fixes.html` was generated by an ad hoc script in
   the session scratchpad; make it a `scripts/qa/` tool driven by `data/{book}_fixes.json`.
7. **Cosmetic**: after a merge, the real node's `bot` keeps the old inferred point, so
   the QA red edge fans out 60px off the hang-line (114_120 毓援). Set `bot` from the
   phantom in `s5_fixes` merge / in the continuation rule.

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
