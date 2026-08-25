# Future features / ideas (deferred)

Feature ideas to build later, once the core pipeline + website are done. Separate
from `bridge-revisit-notes.md` (which tracks pipeline BUGS to fix). Add dated
entries; move to a spec + implementation when picked up.

---

## Polyphonic characters — multi-pronunciation support (William, 2026-08-25)
Some Chinese characters have **multiple pronunciations** (多音字), so a name can be
read more than one way. Handle this across the website:

1. **Search by any reading.** When search is added to the site, a name should match
   ANY of its characters' possible pinyin readings, not just the one the OCR/pinyin
   library picked. (e.g. searching a name by a secondary reading should still find
   it.)
2. **Node view shows all readings.** When viewing a node, display ALL possible pinyin
   sounds for each character (not just the primary), so the reader sees the
   ambiguity.
3. **Leave room to pin the correct reading.** William may later get confirmation of
   which pronunciation is correct for a specific person's name. The data model +
   UI should allow **definitively naming/pinning** the correct reading for a node
   (a per-node override, like the OCR override layer), while still surfacing the
   alternatives.

Implementation notes (for later):
- pypinyin already exposes multiple readings (`heteronym=True` returns all readings
  per char) — `scripts/ocr_review.py`'s `pinyin_of` currently takes only the first.
- The chosen-reading override wants a small persistent layer keyed by node
  (provenance or id), mirroring `data/{book}_overrides.json` for characters.
- Search index should expand each name to the cartesian set of its chars' readings.
