"use client";

// LanguageProvider — owns the current UI language, persists it to localStorage,
// and exposes it through LangContext. Wrap the app in this once; components read
// it via useLang().

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  DEFAULT_LANG,
  LANG_STORAGE_KEY,
  type Lang,
  LangContext,
  type TranslationKey,
  tr,
} from "@/lib/i18n";

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState<Lang>(DEFAULT_LANG);

  // Restore the saved choice after mount (localStorage is client-only, and may
  // throw in private mode — tolerate that and keep the default).
  useEffect(() => {
    try {
      const saved = localStorage.getItem(LANG_STORAGE_KEY);
      if (saved === "en" || saved === "zh") setLangState(saved);
    } catch {
      // ignore — default language stands
    }
  }, []);

  const setLang = useCallback((next: Lang) => {
    setLangState(next);
    try {
      localStorage.setItem(LANG_STORAGE_KEY, next);
    } catch {
      // ignore — the choice just won't persist
    }
  }, []);

  const value = useMemo(
    () => ({ lang, setLang, t: (key: TranslationKey) => tr(lang, key) }),
    [lang, setLang],
  );

  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}
