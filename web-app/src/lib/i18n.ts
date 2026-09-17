"use client";

// ---------------------------------------------------------------------------
// i18n.ts — bilingual (English / Chinese) UI strings + a tiny context.
//
// The app's UI chrome can render in English or 中文. Names and pinyin readings
// are data (not translated); everything else — section headers, prompts, stats,
// aria-labels — comes from the `translations` table keyed by language. A React
// context carries the current language so any component can call useLang()
// without prop-drilling. The choice is persisted in localStorage.
// ---------------------------------------------------------------------------

import { createContext, useContext } from "react";

export type Lang = "en" | "zh";

export const DEFAULT_LANG: Lang = "en";
export const LANG_STORAGE_KEY = "zeng-lang";

// Each entry is [English, Chinese]. Functions cover strings with a value
// interpolated (a count, a generation number, an id).
export const translations = {
  title: ["Zeng Family Tree", "曾氏族谱"],
  people: ["people", "人"],
  loading: ["Loading…", "加载中…"],
  failedToLoad: ["Failed to load:", "加载失败："],
  searchPlaceholder: ["Search name, pinyin, or #id…  ( / )", "搜索姓名、拼音或 #编号…  ( / )"],
  searchAriaLabel: ["Search people by name, pinyin, or id", "按姓名、拼音或编号搜索"],
  noMatches: ["No matches.", "无匹配结果。"],
  emptyPanel: [
    "Click a person in the tree to trace their lineage and see their details here.",
    "点击族谱中的人物，查看其世系与详细信息。",
  ],
  father: ["Father", "父亲"],
  children: ["Children", "子嗣"],
  ancestorsToRoot: ["Ancestors to root", "上溯至始祖"],
  biography: ["Biography", "生平"],
  notes: ["Notes", "备注"],
  noFather: ["None recorded — this is a lineage root.", "无记录——此为一世祖。"],
  noChildren: ["No children recorded.", "无子嗣记录。"],
  alreadyRoot: ["This person is already the root.", "此人即为始祖。"],
  lineageRoot: ["lineage root", "一世祖"],
  clearSelection: ["Clear selection", "清除选择"],
  switchToDark: ["Switch to dark mode", "切换到深色模式"],
  switchToLight: ["Switch to light mode", "切换到浅色模式"],
  toggleTheme: ["Toggle color theme", "切换主题颜色"],
  switchToChinese: ["切换到中文", "切换到中文"],
  switchToEnglish: ["Switch to English", "Switch to English"],
  treeAriaLabel: [
    "Family tree diagram — scroll to zoom, drag to pan, click a person to trace their lineage",
    "族谱图——滚动缩放，拖动平移，点击人物查看其世系",
  ],
} as const;

export type TranslationKey = keyof typeof translations;

// Look up a plain string for a language.
export function tr(lang: Lang, key: TranslationKey): string {
  return translations[key][lang === "zh" ? 1 : 0];
}

// Interpolated strings that don't fit the flat table.
export function genLabel(lang: Lang, generation: number): string {
  return lang === "zh" ? `第${generation}世` : `Generation ${generation}`;
}
export function idLabel(lang: Lang, id: number): string {
  return lang === "zh" ? `编号${id}` : `#${id}`;
}
export function childrenTitle(lang: Lang, count: number): string {
  return lang === "zh" ? `子嗣（${count}）` : `Children (${count})`;
}
export function notFoundLabel(lang: Lang, id: number): string {
  return lang === "zh" ? `未找到编号${id}的人物。` : `Person #${id} not found.`;
}

// --- Context so components read the current language without prop-drilling. --
interface LangContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: (key: TranslationKey) => string;
}

export const LangContext = createContext<LangContextValue>({
  lang: DEFAULT_LANG,
  setLang: () => {},
  t: (key) => tr(DEFAULT_LANG, key),
});

export function useLang(): LangContextValue {
  return useContext(LangContext);
}
