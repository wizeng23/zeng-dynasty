# Milestones

## Milestone 1 — Static website (all 4 books parsed)

Parse all 4 books into structured data and render them as a static website.

- [ ] Book 1 — parsed (stages 1–3 done in old pipeline; golden data verified). OCR pending.
- [ ] Book 2 — cropping (stage 1) done; stages 2–3 pending.
- [ ] Book 3 — not started. Interleaves biography pages with tree pages (needs separate parsing).
- [ ] Book 4 — not started. Same biography/tree interleaving as Book 3.
- [ ] OCR: name images → Unicode characters (stage 7).
- [ ] Static site rendering the combined tree + biographies.

## Milestone 2 — Dynamic website (updates)

Once all 4 books are parsed and the static site works, make it dynamic to
allow updates.

- [ ] Editing / adding nodes.
- [ ] **Daughters and mothers** — the books record only men so far; the schema
      (`father`-only) will need to evolve to represent women.

See `progress.md` for detailed per-stage status, `pipeline.md` for how parsing works.
