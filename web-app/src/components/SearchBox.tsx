"use client";

// ---------------------------------------------------------------------------
// SearchBox.tsx — find a person by Hanzi name or pinyin.
//
// Sits in the page header. Builds a search index from the loaded tree (memoized
// per dataset) and shows a dropdown of matches as you type. Selecting a result
// calls the shared `onSelect(id)` — the same setter the tree and detail panel
// use — so the person highlights in the tree and opens in the panel.
//
// Keyboard: "/" or Cmd/Ctrl-K focuses the box from anywhere (unless you're
// already typing in a field); Esc clears and blurs; Up/Down move the highlight;
// Enter selects it. Matching + ranking live in lib/search (pure, tested).
// ---------------------------------------------------------------------------

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useLang } from "@/lib/i18n";
import { buildSearchIndex, type SearchResult, searchPeople } from "@/lib/search";
import type { LoadedTree } from "@/lib/tree";

interface SearchBoxProps {
  tree: LoadedTree;
  onSelect: (id: number | null) => void;
}

export function SearchBox({ tree, onSelect }: SearchBoxProps) {
  const { t } = useLang();
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0); // highlighted result index
  const inputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();

  // Index is derived from the tree; rebuild only when the dataset changes.
  const index = useMemo(() => buildSearchIndex(tree.byId.values()), [tree]);

  const results: SearchResult[] = useMemo(() => searchPeople(index, query), [index, query]);

  // Global shortcut: "/" or Cmd/Ctrl-K focuses the search box. Ignore it when
  // the user is already typing somewhere (input/textarea/contenteditable).
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const el = e.target as HTMLElement | null;
      const typing =
        el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
      const cmdK = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k";
      if (cmdK || (e.key === "/" && !typing)) {
        e.preventDefault();
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function choose(result: SearchResult) {
    onSelect(result.node.id);
    setQuery("");
    setOpen(false);
    inputRef.current?.blur();
  }

  function onInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Escape") {
      setQuery("");
      setOpen(false);
      inputRef.current?.blur();
      return;
    }
    if (!results.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((i) => (i + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => (i - 1 + results.length) % results.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const r = results[active];
      if (r) choose(r);
    }
  }

  const showList = open && query.trim().length > 0;

  return (
    <div className="relative w-full sm:w-72">
      <input
        ref={inputRef}
        type="search"
        value={query}
        placeholder={t("searchPlaceholder")}
        aria-label={t("searchAriaLabel")}
        role="combobox"
        aria-expanded={showList}
        aria-controls={listboxId}
        aria-autocomplete="list"
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0); // reset highlight to the top result on each edit
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        // Delay close so a click on a result registers before blur hides it.
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={onInputKeyDown}
        className="w-full rounded-md border border-border bg-card px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary focus:outline-none"
      />

      {showList && (
        <ul
          id={listboxId}
          // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: the combobox/listbox/option pattern is the standard ARIA for a search autocomplete; the input is the combobox and this <ul> is its listbox.
          role="listbox"
          className="absolute right-0 left-0 z-20 mt-1 max-h-80 overflow-y-auto rounded-md border border-border bg-card shadow-lg"
        >
          {results.length === 0 ? (
            <li className="px-3 py-2 text-muted-foreground text-sm">{t("noMatches")}</li>
          ) : (
            results.map((r, i) => (
              // biome-ignore lint/a11y/noNoninteractiveElementToInteractiveRole: listbox options are <li role="option"> per the ARIA combobox pattern; the actual activation lives on the inner <button>.
              // biome-ignore lint/a11y/useFocusableInteractive: focus stays on the combobox input; options are navigated via aria-activedescendant-style highlighting, not by focusing each <li>.
              <li key={r.node.id} role="option" aria-selected={i === active}>
                <button
                  type="button"
                  // onMouseDown (not onClick): fires before the input's blur,
                  // so the selection isn't swallowed by the blur-close.
                  onMouseDown={(e) => {
                    e.preventDefault();
                    choose(r);
                  }}
                  onMouseEnter={() => setActive(i)}
                  className={`flex w-full items-baseline gap-2 px-3 py-2 text-left text-sm transition-colors ${
                    i === active ? "bg-muted" : "hover:bg-muted"
                  }`}
                >
                  <span className="font-medium text-base">{r.node.name}</span>
                  <span className="text-muted-foreground text-xs">{r.display}</span>
                  <span className="ml-auto shrink-0 text-muted-foreground text-xs">
                    gen {r.node.generation} · #{r.node.id}
                  </span>
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
