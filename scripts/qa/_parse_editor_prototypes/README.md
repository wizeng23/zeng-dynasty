# Parse-editor prototypes (handoff)

Working scratch scripts backing `docs/specs/2026-09-15-parse-editor-design.md`.
Promote their logic into a proper `scripts/qa/s5_edit.py` server (mirror
`scripts/qa/s6_ocr.py`). These run against the CURRENT `src/find_lines`.

- `seed_manual.py <book> <stem>` — parse one graph, dump the full node list
  (boxes, ends, father/children) to `data/{book}_manual/{stem}.json`. The seed the
  editor loads on first open.
- `qa_one.py <book> <stem>` — render compare strip (pages/crops) + parse overlay to
  a standalone HTML from a live parse. The read-only view the editor supersedes.
- `qa_manual.py <book> <stem>` — same overlay but rendered FROM the override file,
  proving the override is self-sufficient ground truth.

Worked example already corrected by hand: `data/book3_manual/64_65.json`
(34 nodes, 1 root, no bridge).
