# TODO / note: duplicate bio entries from dual-inheritance (双桃承嗣 / 兼祧)

**Observed 2026-09-19.** Some people have **two bio entries** because the zupu records a son
who inherits **two branches** (兼祧 / 双祧 / 双桃承嗣) once under each father.

## Canonical example: 庆松 (Book 3, section 18_63, gen 6)

- `18_63_4_11` — header `子次洲` (2nd son of 洲): biological entry. Ends
  `庆松编继一半胞弟宪波双桃承嗣` (…承嗣 = carries on the line).
- `18_63_4_12` — header `子之波` (son of 波): adoptive entry. `庆松 双桃承嗣 …`

Both are the SAME man: same name 庆松, same birth (公元一九三三年乙亥九月初一日),
same wife (胡玉兰), same three sons (繁煌 / 繁灯 / 繁煊). He appears twice on purpose —
once under each father he continues.

## Why it matters (downstream, not stage 4)

- **Stage 4 (OCR) is correct** to transcribe both — they are two real printed entries.
  No fix needed at the OCR/QA layer; do not "dedup" the crops.
- **Stage 5 / tree building** should treat these as ONE person with a dual link, not two
  separate nodes: e.g. a primary father + a recorded secondary father (承嗣/兼祧), or a
  merged node flagged as dual-inheritance. Otherwise 庆松 (and his 3 sons) get duplicated
  in the tree.
- Detection signal: the marker **双桃承嗣 / 承嗣 / 兼祧 / 编继** in the prose, plus two
  blocks with matching name+birth+wife+sons under different `子之X` headers.

## Action (deferred)

When wiring bios into the tree (bio stage 5 / cross-stitch), add a dual-inheritance
merge pass: find bio blocks that share name+birth (or name+sons) across different father
headers and carrying a 承嗣/兼祧 marker, and fold them into a single node with both
father edges recorded. Until then, expect a small number of duplicated people in the tree.
