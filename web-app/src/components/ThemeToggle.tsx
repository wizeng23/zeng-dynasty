"use client";

// A small light/dark toggle for the header, driven by next-themes.
//
// next-themes only knows the resolved theme AFTER the component mounts in the
// browser (before that, the server render can't know the user's OS preference).
// Rendering theme-dependent UI before mount would cause a hydration mismatch, so
// we render a stable placeholder until `mounted` flips true. This is the
// standard next-themes pattern.

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const isDark = resolvedTheme === "dark";

  return (
    <button
      type="button"
      // Toggle to the opposite of whatever is currently showing.
      onClick={() => setTheme(isDark ? "light" : "dark")}
      // Until mounted, the resolved theme is unknown on both server and client,
      // so the label must also stay theme-neutral — deriving it from `isDark`
      // before mount would make the server HTML ("Switch to dark mode") disagree
      // with the client after the theme resolves, a hydration mismatch.
      aria-label={
        !mounted ? "Toggle color theme" : isDark ? "Switch to light mode" : "Switch to dark mode"
      }
      className="rounded-md border border-border bg-card px-2.5 py-1.5 text-foreground text-sm transition-colors hover:bg-muted"
    >
      {/* Until mounted we don't know the theme; render a neutral glyph so the
          button doesn't flip on hydration. */}
      {!mounted ? <span className="opacity-0">·</span> : isDark ? <SunIcon /> : <MoonIcon />}
    </button>
  );
}

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="2" />
      <path
        d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}
