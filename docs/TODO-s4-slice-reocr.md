# Active task: geometric column-slice re-OCR (s4_slice)

**Status:** spike proven, scaling to full book. Written 2026-09-20, pre-compact handoff.

## The problem this solves

Whole-crop OCR of a bio person-block fails 3 ways because the model re-derives layout:
1. models read left-to-right instead of the book's right-to-left,
2. lines/columns get shuffled out of order,
3. Claude especially merges several columns into one giant line.

Fix: slice each person-block into pieces by GEOMETRY (father header, name, one strip per
body column) and OCR each piece separately. The model then only transcribes a single small
column — it cannot shuffle/merge/reverse. We assemble RTL ourselves.

## Spike result (the 3 WORST blocks in book3, char recall vs verified)

| block | whole-crop Claude | whole-crop Paddle | slice+Paddle | slice+Gemini |
|---|---|---|---|---|
| 85_133_2_7 | 0.46 | 0.86 | 0.79 | **0.94** |
| 139_181_1_2 | 0.19 | 0.82 | 0.75 | **0.97** |
| 18_63_0_4 | 0.34 | 0.71 | 0.76 | **0.99** |

Conclusion: slicing fixes order/direction/merge; per-strip Gemini adds glyph accuracy.
slice+Gemini is decisively best. slice+Paddle ~ ties whole-crop Paddle (order fixed but
Paddle glyph recognition is the ceiling).

## The module: src/bio/s4_slice.py (EXISTS, working prototype, UNCOMMITTED)

Key funcs:
- `slice_block(img) -> [Piece]`  : returns [father, name, col1, col2, ...] RTL. Pipeline:
  - `trim_rules(img)` — strip residual generation-band RULE lines top/bottom. Rules are
    wide dark rows (>=RULE_DARK_FRAC=0.55 of width), can sit behind a blank margin and be
    split into thin sub-bands; we cut below the LAST rule in the top ~18% band, above the
    FIRST in the bottom band, then trim blank margin to first/last text row.
  - right region = rightmost `RIGHT_STRIP_W=400`px = father header + name; body = the rest.
  - `split_father_name(strip)` — ONE horizontal cut: father = first ink band tall enough to
    be real text (>= max(20, 5% of H); skips thin rule/speck remnants); name = everything
    below the gap after it (all stacked name chars).
  - `slice_columns(body)` — whitespace projection: ink-sum per x; x is a gap if ink <=
    GAP_INK_FRAC=0.04 of peak; a gap RUN >= MIN_GAP_W=18 separates columns; ignore columns
    narrower than MIN_COL_W=40. RTL order (rightmost first).
  - `estimate_chars(piece)` — ink vertical span / (~column-width glyph height); QA signal.
- `ocr_pieces(pieces, engine)` — Paddle per strip. FATHER piece read left-to-right by x
  (horizontal caption); all other pieces read top-to-bottom by y (vertical column).
  Uses `_char_boxes` from src.bio.s4_ocr, sorts chars, joins.
- `gemini_pieces(pieces, model="gemini-3.8-flash")` — vision per strip, one call each,
  prompt `_STRIP_PROMPT` (one column, transcribe in reading order, chars only). Winner.
- debug mode dumps each slice PNG to scratchpad/s4_slice/{id}/NN_kind.png + prints OCR.
- CLI: `PYTHONPATH=. python -m src.bio.s4_slice --book book3 --ids <id...> [--debug]`

Geometry constants are px at the crop resolution (~4400-4900w x ~1000-1130h) and are
starting points — tune on debug samples. Father `子` (leftmost header char) sometimes
faint/clipped; Gemini recovers it, Paddle often misreads the tiny header.

## THE PLAN (what to do next, in order)

1. **Finish Paddle on all split columns** — run s4_slice's Paddle path across ALL book3 +
   book4 blocks, store the per-strip results (father/name/cols) somewhere lossless. FREE
   (local). This gives a Paddle-sliced baseline for every block.
2. **Run Gemini AND Claude-vision subagents on all strips** — for every strip of every
   block, get a Gemini read (direct API, gemini-3.8-flash) and a Claude-vision read
   (Agent-tool subagent, the write-to-file pattern). Store all reads per strip losslessly
   (like the existing raw.{paddle,vision,gemini,mistral} ensemble, but now PER STRIP).
   - PAID: Gemini ~15-20 calls/block x ~930 blocks ~= 15-18k calls. Get William's OK on
     cost before the full paid run; test on a sample first. Vision subagents are free but
     a huge fan-out — batch them; do NOT run two writer processes on the same output files
     concurrently (cross-process races clobbered data before — in-process lock only).
   - Use parallelism (ThreadPoolExecutor like s4_ocr_ext) and save-after-every-call.

## Resilience / resume (all stages are crash- and disconnect-safe)

- The two Python runs are LOCAL OS processes (backgrounded) -- they survive assistant
  disconnects; only a host teardown kills them. Logs: scratchpad/s4_slice_resume.log.
- Gemini/Paddle: `--read-all` writes per-block + skips done blocks on restart; `--retry-errs`
  re-reads only ?ERR strips. Paid Gemini work is never lost (saved per block).
- Claude-vision: the COORDINATOR subagent writes each block to {stem}.vision.jsonl as it
  finishes. To resume after any failure, just launch a fresh coordinator -- it recomputes
  remaining work via `scratchpad/vision_helper.py need <book>` (diffs slice files vs sidecars)
  and skips everything already written. The coordinator can also be resumed in place by
  SendMessage to its agent id (context intact). vision_helper.py: `need`/`write`/`writeobj`
  (writeobj validates each array length == piece count before writing).

## FUTURE (William, 2026-09-21): merge Gemini-slice into verified WITHOUT clobbering names
- William's 556/623 verified book3 blocks are mostly the FATHER/SON/NAME columns (not the full
  prose). The Gemini SLICE read is better for the prose/body columns. LATER: design a merge that
  takes Gemini-slice for the body columns but KEEPS William's verified father/name/son columns.
- FOR NOW (done): Verified prefill uses Gemini-slice ONLY for UNVERIFIED blocks
  (verifiedText = done ? VERIFIED[id] : prefill, line ~565). Verified blocks are untouched --
  confirmed: 2_9_0_0 shows saved 子之禄, not Gemini's 子之祿. Do NOT auto-apply slice to verified.

## PAUSED 2026-09-21 (~5h usage limit) -- resume state
- Claude-vision coordinator + all workers KILLED (William paused near 5h limit; will restart).
  STATE SAFE ON DISK: book3 vision = 470 blocks done, 150 remaining. Resume by launching a
  fresh coordinator (same prompt as before) -- it runs `vision_helper.py need book3`, which
  skips the 470 done and returns the 150 left. Nothing lost.
- Book 4 PADDLE-SLICE finished (exit 0): all book4 blocks sliced + Paddled + strip PNGs on
  disk. NO Gemini/vision on book4 yet (awaiting William's slice review in QA, then his go).
- Book 3 Gemini: 620/621 complete (1 ?ERR: 85_133_2_0). Book-3 read-all's last 2 blocks were
  cut off when the chained job was killed but show complete; verify on resume.
- QA page (8767) fixes landed this session (uncommitted): newline-in-strip flatten on read;
  splice/split now persist an AUTHORITATIVE full field map (fullFieldMap) BEFORE re-render so
  son/name labels shift with columns on delete/insert/split; Verified prefill now uses the
  Gemini SLICE reading for blocks that have it (fallback to whole-crop Claude).

## Prompt tweak (2026-09-21): scan-noise instruction
- Added to BOTH _STRIP_PROMPT and _FATHER_PROMPT (Gemini): "Ignore ink smears and very faint
  characters caused by document scanning." (v1 ADF scans have smears models hallucinate into chars.)
- The Claude-VISION subagent prompt is INLINE in the coordinator's Agent calls (not these
  constants) -- when relaunching the vision coordinator (book3 remaining 150, and book4), ADD
  the same sentence to the worker-prompt rules.

## Why the new slicer catches thin columns (e.g. 下 in 273_279_0_0)
- NOT the widening -- it was lowering MIN_COL_W 40->30 (done in the same width-fix commit). A
  thin 1-char column like 下 forms a ~34px ink run: rejected at 40, accepted at 30, then widened
  to CHAR_W+pad. So book3 (OCR'd at MIN_COL_W=40) DROPPED these columns; the new geometry keeps
  them. => Re-slicing + re-OCRing book3 with the new geometry would auto-recover missed thin
  columns (paid; disruptive mid-review -- William's call). Until then, add them by hand in QA.

## Column-width fix (2026-09-21) -- MUST re-slice book4 before its LLM run
- slice_columns now: widen each column to >= COL_MINW_FRAC(0.6) x median column width (so a
  narrow column can't clip a char's thin extending strokes) + COL_PAD(40)px padding each side,
  clamped to body bounds. Fixed the clipped 时 William flagged.
- QA overlay recomputes geometry live so it already shows the new boxes. BUT the strip PNGs on
  disk (book3 AND book4) were cut with the OLD narrow logic. book3 is already OCR'd (leave it;
  re-slice only if re-OCRing). book4 is NOT OCR'd yet -> RE-SLICE book4 (`--paddle-all` again)
  BEFORE running Gemini/vision on it, so the models read the padded strips.

## Book 4 plan (2026-09-21) -- SLICE FIRST, review, THEN LLMs (William's gate)
- Slicing decoupled from LLM reads: run `--paddle-all` (free, local: geometry + Paddle +
  strip PNGs) BEFORE any paid reader. Book 4 paddle-slice running now (standalone process).
- The chained resume job was KILLED so `--read-all book4` does NOT auto-spend Gemini on B4.
- GATE: William reviews Book 4 slice boundaries in the 8767 QA page (strip overlay) BEFORE
  Gemini/Claude-vision are launched on Book 4. Only after his OK:
    1. `--read-all book4` (Gemini) OR a gemini-only fill,
    2. launch a Book 4 vision coordinator (against the already-sliced strips).
- Book 3 leftover: 1 block short of full Gemini (620/621; 85_133_2_0 has a ?ERR). Finish with
  `--book book3 --retry-errs` + `--read-all book3` (skips the 620 done) -- do with the B4 sweep.

## Known data quirk: newlines inside a strip text
- A model sometimes returns a strip's text with embedded newlines (e.g. gemini "五\n宪\n理").
  Each strip is ONE column, so newlines are spurious. QA page flattens on read (load_slice_reads
  `clean()` does re.sub(\s+,'')). STORED jsonl still has them -> when writers are idle, run a
  one-off normalization over 4_slice/*.jsonl + *.vision.jsonl to strip \s from paddle/gemini/
  vision before downstream merge/tree consumes them.

## PENDING when the resume run finishes (2026-09-21)
- Connectivity dropped briefly mid-run -> some strips recorded ?ERR. WHEN the chained run
  reaches ALL DONE, run the error sweep on BOTH books:
    PYTHONPATH=. python -m src.bio.s4_slice --book book3 --retry-errs
    PYTHONPATH=. python -m src.bio.s4_slice --book book4 --retry-errs
  Then re-count ?ERR; repeat retry-errs until 0 (or only genuinely-unreadable strips remain).
- Then `--merge-vision` for both books to fold Claude-vision sidecars into the main records.
- Then commit the 4_slice/*.jsonl reads (currently uncommitted; runs were still writing them).

## Output format decision (RESOLVED 2026-09-20)

Per-strip reads live in a **separate** dir `books/{book}/bio/4_slice/{stem}.jsonl` (one
record per block: `{id, pieces:[{idx, kind, box, est_chars, paddle, gemini, vision}]}`).
Do NOT fold into 4_ocr (William: keep the existing model OCRs intact, no overwrite).
Claude-vision writes a SIDECAR `{stem}.vision.jsonl` (`{id, vision:[texts by idx]}`) that
read_book never touches -> no cross-process write race. `merge_vision` folds sidecars into
the main files' per-piece `vision` field at the end.

## Run status (2026-09-20 ~23:52, overnight)

- Prompt split into `_STRIP_PROMPT` (vertical) + `_FATHER_PROMPT` (horizontal caption),
  chosen per piece by `kind` via `_prompt_for`. One genealogy-context clause each (steers
  e.g. 夭 vs 天); no extra glyph hints (prompt tokens are paid). Same 2 prompts feed Gemini
  AND the Claude-vision subagent.
- `read_book()` = overnight runner: slice + Paddle + Gemini per strip, ThreadPoolExecutor
  (workers=8), ONE writer process, per-stem threading.Lock, atomic tmp+rename, save after
  every block, resumable (`_rec_done` skips finished blocks), `_gemini_read` retry+backoff
  on transient errors. CLI `--read-all`.
- LAUNCHED background: book3 (623 blks, 2 pre-done) -> book4 (307) chained, log at
  `scratchpad/s4_slice_overnight.log`. Smoke test (2 blks) validated: slice+Gemini reads
  clean & in RTL order, beats Paddle on boundary chars + father caption (子之祿 vs 之禄).
  NOTE: Gemini 3.8-flash latency balloons under sustained load (1-3s early -> 30-100s late);
  the full run genuinely needs the night.
- Claude-vision (step 2b): assistant dispatches subagents that read strip PNGs and RETURN
  transcriptions; assistant writes sidecars via `append_vision(book, stem, bid, texts)`.
  Single writer (assistant) => race-free. ALWAYS length-check each returned array == piece
  count before writing (subagents occasionally emit a correction/second array — use the
  final clean one).
  - CONTEXT-EFFICIENT PATTERN (use this): one subagent reads SEVERAL small blocks and
    returns a JSON OBJECT {block_id: [texts...]}. Group ~6 small blocks per subagent. This
    is the only way to cover ~930 blocks without exhausting assistant context. Large blocks
    (>15 pieces) can still go one-per-subagent.
  - Vision is the SECOND-OPINION reader for QA disagreement; Gemini (0.94-0.99, autonomous,
    zero assistant-context cost) is the PRIMARY. Partial vision coverage is fine -- merge_vision
    + QA handle empty `vision` gracefully. If assistant context runs low, STOP vision and let
    Gemini finish everything; resume vision later.
  - PROGRESS (book3, ~00:15): 2_9 fully visioned (20/20). 18_63: 52/112 visioned (all the
    dual-inheritance/承嗣 cases + long multi-son entries done). Vision sweep STOPPED here by
    design (see REALITY CHECK) -- 18_63 alone is 112 blocks; exhaustive vision is infeasible in
    one assistant context. 72 blocks total have Claude-vision second opinions. All OTHER sections
    have Gemini only (which is fine -- Gemini is the primary reader).
  - REMAINING VISION (deferred, resume later per TO-RESUME recipe): 60 blocks of 18_63 still
    unvisioned, plus every other book3 section, plus all of book4. This is a follow-up task, not
    an overnight blocker. Gemini covers all of it.
  - TO RESUME vision cleanly: for a section, `dv={ids in {stem}.vision.jsonl}`;
    `need=[r.id for r in {stem}.jsonl if all pieces have gemini and r.id not in dv]`. Pair/group
    to ~2 big or ~6 small per subagent. ALWAYS validate returned array len == piece count in
    {stem}.jsonl before append_vision (subagents add notes/fences/preambles/second arrays --
    extract just the JSON, use the final clean copy).
  - REALITY CHECK: exhaustive Claude-vision over all ~930 blocks is NOT feasible in one
    assistant context. Gemini (primary, autonomous, 0 assistant-context cost) covers everything.
    Vision = second opinion; partial coverage OK. Prioritize sections under active QA (18_63 has
    the dual-inheritance example). If context runs low: STOP vision, let Gemini finish, run
    `--merge-vision` for both books, commit, and hand off the remaining-vision list.
  - *** RUN HALTED 01:47 2026-09-21: GEMINI PREPAYMENT CREDITS DEPLETED (402 RESOURCE_EXHAUSTED,
    "Your prepayment credits are depleted", ai.studio billing). NOT a bug -- a hard billing wall.
    First 402 at 01:44 (~block 415). I killed the run so it would not burn the rest of book3 +
    all book4 into ?ERR. ***
    STATE AT HALT (book3 only; book4 NEVER STARTED):
      - 461/623 book3 blocks sliced + Paddled (Paddle is local, all good).
      - 418 blocks fully-clean Gemini; 43 blocks have >=1 ?ERR (372 error strips, 6.5%).
      - ~162 book3 blocks + ALL book4 (307) have NO Gemini yet.
      - Vision sidecars (72 blocks: 2_9 + 18_63 partial) unaffected.
    TO RESUME (ONLY after William tops up Gemini credits -- do NOT relaunch before that or it
    just refills ?ERR):
      1. `PYTHONPATH=. python -m src.bio.s4_slice --book book3 --retry-errs`  (re-reads only the
         372 ?ERR strips in place; needs strip PNGs, which exist).
      2. `PYTHONPATH=. python -m src.bio.s4_slice --book book3 --read-all --workers 8` (resume
         skips the 418 done, finishes the ~162 remaining book3 blocks).
      3. `PYTHONPATH=. python -m src.bio.s4_slice --book book4 --read-all --workers 8`.
    CONSIDER a cheaper/pay-as-you-go key or lower workers to avoid another sudden depletion.
  - (historical) Gemini run: steady ~10 blocks/2min, ~0 ?ERR. KNOWN ?ERR strips (transient net drops,
    exhausted retries -- re-run later; note `_rec_done` counts ?ERR as done so a plain --read-all
    resume WON'T retry them; a targeted retry pass must clear the ?ERR value first or filter for
    it): book3 66_77_4_2 piece 3 ("Server disconnected").

## Scoring

365 verified blocks in `data/book3_bio_verified.json` (William's hand-verified ground
truth). Char-recall metric: multiset of CJK chars covered / truth chars (see the spike
scripts). Compare slice+reader assembled text vs verified.

## Guardrails / lessons this session
- Repo is PUBLIC; agents do NOT push. William pushes. Commit locally freely.
- OCR costs money — get explicit approval before paid runs; scope tightly; test on a few.
- macOS: Alt+letter emits composed chars; use e.code for JS shortcuts.
- s3_post was over-tightening QA boxes (fixed: clamp-only). Crops are gitignored.
- zeng conda env: /opt/miniconda3/envs/zeng/bin/python, PYTHONPATH=.
- Running servers: s4 QA 8767 (book3), segment editor 8788 (book4), s6 name QA 8766.
- Env keys present: OPENAI_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY (all printed to an
  earlier transcript — consider rotating; never commit).

## Uncommitted work at handoff
- src/bio/s4_slice.py (NEW, this task).
- scripts/qa/s4_ocr.py QA-tool improvements: 生子-substring son detection; verified dashed
  outline where Claude/Gemini disagree on father/name/son; column split button + Alt+s
  (KeyS); paste-splits-into-columns; Alt+Enter/Alt+Bksp insert/delete column; graph
  name-matching reference; chip text-selection.
- docs/TODO-vision-reocr-stale-crops.md, docs/TODO-duplicate-bio-dual-inheritance.md.
