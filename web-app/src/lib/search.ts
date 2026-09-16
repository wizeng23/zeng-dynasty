// ---------------------------------------------------------------------------
// search.ts — searchable index over people + query matching/ranking
//
// Builds a small in-memory index from the tree's people and matches a query
// against it. A person matches by any of:
//   - Hanzi:               substring of the name (兴, 兴思)
//   - toneless pinyin:     "xingsi" / "xing si"
//   - tone-numbered pinyin: "xing1si1"
// Results are ranked (exact name -> name prefix -> pinyin prefix -> substring)
// and capped. Pure, no DOM — the component (SearchBox) renders the results.
// ---------------------------------------------------------------------------

import { toDisplayPinyin, toSearchPinyin } from "./pinyin";
import type { FamilyNode } from "./tree";

export interface SearchEntry {
  node: FamilyNode;
  name: string; // trimmed Hanzi name
  display: string; // tone-marked reading, e.g. "Xìng Sī"
  toneless: string; // "xingsi"
  toneNumbered: string; // "xing1si1"
}

export interface SearchResult {
  node: FamilyNode;
  display: string; // tone-marked pinyin for showing in the result row
}

export const MAX_RESULTS = 20;

// Build the index once per loaded dataset. Nameless people (image-only crops)
// are skipped: there is nothing to romanize or match on.
export function buildSearchIndex(people: Iterable<FamilyNode>): SearchEntry[] {
  const entries: SearchEntry[] = [];
  for (const node of people) {
    const name = node.name.trim();
    if (!name) continue;
    const { toneless, toneNumbered } = toSearchPinyin(name);
    entries.push({
      node,
      name,
      display: toDisplayPinyin(name),
      toneless,
      toneNumbered,
    });
  }
  return entries;
}

// Normalize a query: trim, lowercase, drop spaces (so "xing si" == "xingsi").
// Hanzi and digits are preserved; only ASCII spacing/case is normalized.
function normalize(query: string): string {
  return query.trim().toLowerCase().replace(/\s+/g, "");
}

// The shared surname 曾 (readings zēng / céng — polyphonic) is not stored on any
// person; every name in the index is a GIVEN name. So if the query leads with the
// surname, drop it before matching against given names. Only strip when something
// follows: a bare "曾" / "zeng" / "ceng2" is a search for the surname alone, which
// has no given name to match, so it stays as-is (and matches nothing).
//
// Pinyin prefixes are matched as a WHOLE syllable: an optional trailing tone
// digit after zeng/ceng belongs to the surname (so "ceng2xingsi" -> "xingsi",
// and the bare "ceng2" is left intact rather than leaving a stray "2").
const SURNAME_HANZI = "曾";
const SURNAME_PINYIN = /^(?:zeng|ceng)[0-9]?/;
function stripSurname(q: string): string {
  // Hanzi 曾 as the first character, with a given name after it.
  if (q.length > 1 && q.startsWith(SURNAME_HANZI)) {
    return q.slice(SURNAME_HANZI.length);
  }
  // Pinyin surname syllable at the start, with a given name after it.
  const m = q.match(SURNAME_PINYIN);
  if (m && q.length > m[0].length) {
    return q.slice(m[0].length);
  }
  return q;
}

// Rank buckets (lower = better). We score each entry by its best-matching field
// and sort ascending, breaking ties by generation then id for a stable order.
const RANK = {
  nameExact: 0,
  namePrefix: 1,
  pinyinExact: 2,
  pinyinPrefix: 3,
  nameSubstring: 4,
  pinyinSubstring: 5,
  none: 99,
} as const;

function scoreEntry(entry: SearchEntry, q: string): number {
  // Hanzi query: match against the name only.
  if (entry.name === q) return RANK.nameExact;
  if (entry.name.startsWith(q)) return RANK.namePrefix;

  // Pinyin query (toneless or tone-numbered): exact, then prefix.
  if (entry.toneless === q || entry.toneNumbered === q) return RANK.pinyinExact;
  if (entry.toneless.startsWith(q) || entry.toneNumbered.startsWith(q)) return RANK.pinyinPrefix;

  // Substrings last.
  if (entry.name.includes(q)) return RANK.nameSubstring;
  if (entry.toneless.includes(q) || entry.toneNumbered.includes(q)) return RANK.pinyinSubstring;

  return RANK.none;
}

// Search the index. Empty query -> no results. Returns at most MAX_RESULTS,
// best match first.
export function searchPeople(index: SearchEntry[], query: string): SearchResult[] {
  const q = stripSurname(normalize(query));
  if (!q) return [];

  const scored: { entry: SearchEntry; score: number }[] = [];
  for (const entry of index) {
    const score = scoreEntry(entry, q);
    if (score !== RANK.none) scored.push({ entry, score });
  }

  scored.sort((a, b) => {
    if (a.score !== b.score) return a.score - b.score;
    if (a.entry.node.generation !== b.entry.node.generation)
      return a.entry.node.generation - b.entry.node.generation;
    return a.entry.node.id - b.entry.node.id;
  });

  return scored
    .slice(0, MAX_RESULTS)
    .map(({ entry }) => ({ node: entry.node, display: entry.display }));
}
