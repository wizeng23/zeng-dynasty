# Cross-book stitch — design + handoff (2026-09-16, mid-task)

**Goal (William asked):** connect Book 3's 14 subtree roots to their Book 2 leaves →
one connected lineage. If it stitches cleanly, publish Book 3 to the website.
**HARD RULES:** (1) do NOT wipe the `notes` field on existing Book 1+2 stitched
nodes. (2) do NOT wipe notes on the website data either. (3) generations in the
MERGED tree must be ABSOLUTE, not subgraph-relative — William saw generation numbers
swap on the website because merged graphs used per-subgraph gens.

## Key findings (verified this session)

- Files: `data/book1_stitched.jsonl` (150 nodes, 1 root `点`, gens 1–56),
  `data/book2_stitched.jsonl` (1542, 2 roots `存学`/`贞年`, gens 1–16),
  `data/book3_stitched.jsonl` (639, 14 roots all `传*`, gens 1–6).
- **The link is Book 3 root → Book 2 gen-16 LEAF, matched by exact name.** 12/14
  Book 3 roots match a Book 2 gen-16 leaf (all are childless leaves in Book 2 — the
  same "duplicate root repeats as a leaf in the earlier book" pattern within-book
  stitch already handles). The 2 unmatched: `传煦`, `传炯` (both LOW OCR conf — likely
  a char mismatch vs Book 2; leave unmerged or hand-map later).
- Book 2 has 404 `传*` nodes (gens 6 and 16); the relevant ones are the ~377 gen-16
  leaves. Only ~12–14 of them continue into a Book 3 subtree; most stay leaves.
- `notes` format: `"{prov} | ocr_conf=... | ocr_override"`. `_provenance(notes)`
  (in `src/s7_stitch.py`) strips to the bare `{prov}`. Within-book stitch already
  PRESERVES tags and records a merge as `canonProv/dupProv | tags` (e.g.
  `0_3_0/4_5_0 | ocr_conf=...`). REUSE that convention — don't overwrite it.

## Reuse `src/s7_stitch.py:stitch_nodes(nodes, merges)`

It already: folds each `(dup_root_prov, canon_leaf_prov)` — canon adopts the root's
children, recursively folds repeated ancestor chains, records `canon/dup` in notes
KEEPING ocr tags, then recomputes ABSOLUTE generations from the single root and
reassigns BFS/RTL ids. This is exactly the cross-book merge — just across two books'
node lists instead of one graph's.

## Plan (not yet executed)

1. Load book2_stitched + book3_stitched. **Offset Book 3's ids** so they don't
   collide with Book 2's (e.g. `id += 100000`), keep notes verbatim (prefix provenance
   with a book tag if needed to keep them unique, e.g. `b3:0_1_0` — but check this
   doesn't break `_provenance`/`_graph_num`; simpler: rely on id offset + a merges
   list keyed by the ALREADY-UNIQUE (book, prov) — may need a small wrapper).
2. Build the merges list: for each Book 3 root, find the Book 2 gen-16 leaf with the
   same `name`. Skip ambiguous (name appears >1× as a gen-16 leaf — 2 of the 12
   matched had 2 candidates; disambiguate by reading-order/notes or leave unmerged).
   `传煦`,`传炯` have no match → leave as roots (report to William).
3. Concatenate the two node lists, call `stitch_nodes(combined, cross_merges)`. This
   rebases generations absolutely: Book 3 root that merges onto a gen-16 Book 2 leaf
   → Book 3 subtree becomes gens 16,17,18,… (fixes the swap). Book 2's own gens
   stay; Book 1 is a SEPARATE tree (its root `点` is the deep spine — do NOT merge
   Book 2 into Book 1 yet unless William asks; that's a further step).
4. Verify: 0 child↔father mismatches, every edge gen == father.gen+1, all notes on
   pre-existing Book 1/2 nodes UNCHANGED (diff against the input stitched files —
   only Book 3 nodes + the ~12 Book 2 leaves that gained children should change).
5. Write `data/book23_stitched.jsonl` (or extend book2_stitched — but SAFER to write
   a new combined file so book2_stitched stays as-is). Then website: add it as a
   dataset via `web-app` export (`bun run export-data` regenerates public/data +
   public/names from `data/`). Book 3 is NOT yet on the site. Do NOT wipe notes in
   the exported data (the web export currently may or may not carry notes — CHECK).

## Guardrails
- Public repo; agent never `git push` (William pushes). Commit locally only if asked.
- Don't re-run any parse/OCR stage. This is a pure data-join over the stitched jsonls.
- If matching is ambiguous or <12/14 merge cleanly, STOP and report to William before
  writing website data — "if it stitches cleanly" was his condition for publishing.

## State of servers (this session)
Book 3 QA 8785 / editor 8787; Book 4 QA 8795 / editor 8797. Book 3 & 4 OCR review
servers were taken down. Book 3 OCR review DONE (151 corrections, 8 flags remain,
all rare 釒-radical `广*` names). Book 4: graph editing + OCR review still pending;
3 over-cropped pages (36/164/254) fixed via `shrink_full_span_ink`; the 3 fixed
pages need re-OCR after graph edits.
