# TODO: re-run the vision reader on re-cropped bio blocks

**Status:** pending (deferred 2026-09-19 — Claude usage window low).

## Why

`s3_post` used to re-run `tighten_box` on QA-approved boxes and over-shrank wide/sparse
person blocks (dropping whole columns). That was fixed (clamp-only, no re-tighten) and the
crops were regenerated from the approved boxes. The **paid/local readers were re-run** on
the materially-changed blocks (>30px), but the **Claude vision reader was NOT** (it runs as
Agent-tool subagents; deferred to save usage).

So on these blocks, `raw.gemini` / `raw.mistral` / `raw.paddle` reflect the corrected wider
crops, but `raw.vision.text` is **stale** (read from the old, too-tight crops).

## What to do

Re-run the Claude vision subagent on the stale blocks and fold fresh `raw.vision.text`
(+ the parsed vision fields via `src.bio.s4_ocr.parse_vision`) into the `4_ocr` records.

**Block lists (already recorded):**
- Book 3: `data/book3_stale_vision_blocks.json` — **72 blocks**
- Book 4: `data/book4_stale_vision_blocks.json` — **9 blocks**

**How (mirrors the original s4 vision pass):**
1. For each block id, dispatch one vision subagent with `src/bio/s4_vision_prompt.txt`
   (crop at `books/{book}/bio/3_segment/{id}.png`), write-to-file pattern.
2. Fold each transcript into that block's `4_ocr` record: set `raw.vision.text` and the
   structured `name/father_char/sons/daughters/birth` (see how the 3 `85_133_0_*` blocks
   were folded earlier this session).
3. Run subagents in batches; do NOT run a second writer process against the same `4_ocr`
   section files concurrently (a cross-process write race clobbered 10 blocks once — the
   in-process lock in `s4_ocr_ext` does not protect across processes).

**Verify after:** every stale block has non-empty `raw.vision.text`; Book 3 vision back to
620/620, Book 4 to 307/307.

## Notes

- Snapshots of the pre-fix crops exist at `books/book{3,4}/bio/3_segment_snapshot_*` for
  rollback/compare.
- `2_merged` is not committed in this worktree; it was symlinked from the main checkout to
  regenerate crops. Not needed for the vision re-run (which reads `3_segment` crops).
