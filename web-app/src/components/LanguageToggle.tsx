"use client";

// LanguageToggle — a compact EN / 中 switch. Shows the language you'd switch TO,
// so it reads as an action. Lives at the top-left of the header.

import { useLang } from "@/lib/i18n";

export function LanguageToggle() {
  const { lang, setLang, t } = useLang();
  const next = lang === "en" ? "zh" : "en";

  return (
    <button
      type="button"
      onClick={() => setLang(next)}
      aria-label={lang === "en" ? t("switchToChinese") : t("switchToEnglish")}
      className="rounded-md border border-border bg-card px-2.5 py-1.5 font-medium text-foreground text-sm transition-colors hover:bg-muted"
    >
      {lang === "en" ? "中文" : "EN"}
    </button>
  );
}
