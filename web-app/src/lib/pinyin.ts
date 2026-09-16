// ---------------------------------------------------------------------------
// pinyin.ts — romanization helpers for Hanzi names
//
// The dataset stores names as Hanzi only. These helpers derive pinyin from the
// characters at runtime (via pinyin-pro) for two uses:
//   - display: a readable, tone-marked, per-syllable-capitalized reading
//     (e.g. 兴思 -> "Xìng Sī"), shown next to names in the detail panel.
//   - search:  ASCII forms a user might type — toneless ("xingsi") and
//     tone-numbered ("xing1si1") — used by lib/search.ts to match queries.
// Pure functions, no DOM. Given names only: the shared surname 曾/Zeng is not
// stored per-person, so it is never part of a reading here.
// ---------------------------------------------------------------------------

import { pinyin } from "pinyin-pro";

// Uppercase the first letter of a syllable, leaving tone marks intact.
function capitalize(syllable: string): string {
  if (!syllable) return syllable;
  return syllable[0].toUpperCase() + syllable.slice(1);
}

// A readable reading for display: tone marks, each syllable capitalized,
// space-separated. 兴思 -> "Xìng Sī". Empty in -> empty out.
export function toDisplayPinyin(hanzi: string): string {
  const name = hanzi.trim();
  if (!name) return "";
  const syllables = pinyin(name, { type: "array" }) as string[];
  return syllables.map(capitalize).join(" ");
}

// ASCII forms a searcher might type for a name. Both are lowercased and stripped
// of spaces so the match side can normalize a query the same way:
//   - toneless:     兴思 -> "xingsi"
//   - toneNumbered: 兴思 -> "xing1si1"
export function toSearchPinyin(hanzi: string): {
  toneless: string;
  toneNumbered: string;
} {
  const name = hanzi.trim();
  if (!name) return { toneless: "", toneNumbered: "" };
  const none = (pinyin(name, { toneType: "none", type: "array" }) as string[]).join("");
  const num = (pinyin(name, { toneType: "num", type: "array" }) as string[]).join("");
  return { toneless: none.toLowerCase(), toneNumbered: num.toLowerCase() };
}
