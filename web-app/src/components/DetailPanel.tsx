"use client";

// ---------------------------------------------------------------------------
// DetailPanel.tsx — the side panel shown when a person is selected.
//
// It's a pure presentational component: it takes the selected id + the loaded
// tree and reads everything it needs (father, children, ancestor chain) from
// the flat `byId` map and the pure helpers in lib/tree. It never touches d3 or
// the SVG. Navigation works by calling `onSelect(id)` — the SAME selection
// setter the tree uses, so clicking a relative here re-highlights the tree.
// ---------------------------------------------------------------------------

import { childrenTitle, genLabel, idLabel, notFoundLabel, useLang } from "@/lib/i18n";
import { toDisplayPinyin } from "@/lib/pinyin";
import {
  ancestryChain,
  type BioEntry,
  bioBest,
  bioBestList,
  type FamilyNode,
  type LoadedTree,
} from "@/lib/tree";

// The shared surname, carried by everyone in the book but never stored per-person
// (the data holds given names only). We show it before each name in a muted style
// so it reads as "surname · given name" without being mistaken for the given name.
const SURNAME_HANZI = "曾";
const SURNAME_PINYIN = "Zēng";

// The muted surname glyph shown just before a Hanzi given name.
function SurnameHanzi() {
  return <span className="mr-0.5 text-muted-foreground opacity-70">{SURNAME_HANZI}</span>;
}

// The muted surname reading shown just before a given-name pinyin reading.
function SurnamePinyin() {
  return <span className="mr-1 text-muted-foreground opacity-70">{SURNAME_PINYIN}</span>;
}

interface DetailPanelProps {
  tree: LoadedTree;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}

export function DetailPanel({ tree, selectedId, onSelect }: DetailPanelProps) {
  const { lang, t } = useLang();

  // Empty state — nothing selected yet.
  if (selectedId === null) {
    return (
      <aside className="flex w-full flex-col border-border border-l bg-card p-5 text-sm sm:w-80">
        <p className="text-muted-foreground">{t("emptyPanel")}</p>
      </aside>
    );
  }

  const person = tree.byId.get(selectedId);
  if (!person) {
    return (
      <aside className="w-full border-border border-l bg-card p-5 text-sm sm:w-80">
        <p className="text-muted-foreground">{notFoundLabel(lang, selectedId)}</p>
      </aside>
    );
  }

  const father = person.father === -1 ? null : (tree.byId.get(person.father) ?? null);

  // children[] is stored eldest-first already, so this list reads oldest -> youngest.
  const children = person.children
    .map((id) => tree.byId.get(id))
    .filter((c): c is FamilyNode => c !== undefined);

  // Ancestor chain is SELF -> ... -> root. Drop `self` (index 0); the rest are
  // the "nearby ancestors" from the immediate father up to the lineage root.
  const ancestors = ancestryChain(person.id, tree.byId).slice(1);
  const isRoot = person.father === -1;

  return (
    <aside className="flex w-full flex-col gap-6 overflow-y-auto border-border border-l bg-card p-5 text-sm sm:w-80">
      {/* Header: the selected person's name, generation, close button. */}
      <div>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            {person.name.trim() ? (
              <>
                {/* Hanzi name, surname muted before the given name. */}
                <div className="font-medium text-2xl">
                  <SurnameHanzi />
                  {person.name}
                </div>
                {/* Pinyin reading, surname muted before the given name. */}
                <div className="text-muted-foreground text-sm">
                  <SurnamePinyin />
                  {toDisplayPinyin(person.name)}
                </div>
              </>
            ) : (
              // No Unicode name yet — fall back to the scanned crop / id.
              <div className="text-lg">
                <PersonLabel person={person} size="lg" />
              </div>
            )}
            <div className="mt-1 text-muted-foreground text-xs uppercase tracking-wide">
              {genLabel(lang, person.generation)}
              {isRoot && ` · ${t("lineageRoot")}`}
            </div>
            <div className="text-muted-foreground text-xs">{idLabel(lang, person.id)}</div>
          </div>
          <button
            type="button"
            onClick={() => onSelect(null)}
            aria-label={t("clearSelection")}
            className="shrink-0 rounded-md px-2 py-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Father. */}
      <Section title={t("father")}>
        {father ? (
          <RelativeButton person={father} onSelect={onSelect} />
        ) : (
          <p className="text-muted-foreground">{t("noFather")}</p>
        )}
      </Section>

      {/* Children (eldest-first). */}
      <Section title={childrenTitle(lang, children.length)}>
        {children.length > 0 ? (
          <ul className="flex flex-col gap-1.5">
            {children.map((child, i) => (
              <li key={child.id} className="flex items-center gap-2">
                {/* Small ordinal hint so the eldest-first order is legible. */}
                <span className="w-4 shrink-0 text-muted-foreground text-xs tabular-nums">
                  {i + 1}
                </span>
                <RelativeButton person={child} onSelect={onSelect} />
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted-foreground">{t("noChildren")}</p>
        )}
      </Section>

      {/* Nearby ancestors: immediate father up to the lineage root. */}
      <Section title={t("ancestorsToRoot")}>
        {ancestors.length > 0 ? (
          <ol className="flex flex-col gap-1.5">
            {ancestors.map((anc) => (
              <li key={anc.id} className="flex items-center gap-2">
                <span className="w-12 shrink-0 text-muted-foreground text-xs">
                  {lang === "zh" ? `第${anc.generation}世` : `gen ${anc.generation}`}
                </span>
                <RelativeButton person={anc} onSelect={onSelect} />
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-muted-foreground">{t("alreadyRoot")}</p>
        )}
      </Section>

      {/* Scanned biography (bio OCR pipeline): birth clause, sons/daughters, and
          the full transcription. Only shown when a bio was linked to this node. */}
      {person.bio && <ScannedBio bio={person.bio} />}

      {/* Biography / notes, only if present. In the golden data biography is
          sometimes a bare Wikipedia URL, so render those as a link. */}
      {person.biography.trim() && (
        <Section title={t("biography")}>
          {isUrl(person.biography.trim()) ? (
            <a
              href={person.biography.trim()}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all text-primary underline underline-offset-2 hover:opacity-80"
            >
              {person.biography.trim()}
            </a>
          ) : (
            <p className="whitespace-pre-wrap text-foreground leading-relaxed">
              {person.biography}
            </p>
          )}
        </Section>
      )}
      {person.notes.trim() && (
        <Section title={t("notes")}>
          <p className="whitespace-pre-wrap text-muted-foreground leading-relaxed">
            {person.notes}
          </p>
        </Section>
      )}
    </aside>
  );
}

// True for a bare http(s) URL (some golden biographies are Wikipedia links).
function isUrl(s: string): boolean {
  return /^https?:\/\/\S+$/.test(s);
}

// The scanned-biography block: what the bio OCR pipeline read off the book for this
// person. Shows the birth clause, the listed sons/daughters, and the full column-by-
// column transcription (vision reading preferred, paddle fallback). Everything is
// best-effort OCR, so it carries a muted "may contain errors" note.
function ScannedBio({ bio }: { bio: BioEntry }) {
  const { t } = useLang();
  const birth = bioBest(bio.birth);
  const sons = bioBestList(bio.sons);
  const daughters = bioBestList(bio.daughters);
  const fullText = bio.raw?.vision?.text?.trim() || (bio.raw?.paddle?.columns ?? []).join("\n");

  // Nothing worth showing (all fields empty) — skip the section entirely.
  if (!birth && sons.length === 0 && daughters.length === 0 && !fullText) return null;

  return (
    <Section title={t("bioScanned")}>
      <div className="flex flex-col gap-2">
        {birth && (
          <div className="flex gap-2">
            <span className="shrink-0 text-muted-foreground text-xs">{t("bioBirth")}</span>
            <span className="text-foreground">{birth}</span>
          </div>
        )}
        {sons.length > 0 && (
          <div className="flex gap-2">
            <span className="shrink-0 text-muted-foreground text-xs">{t("bioSons")}</span>
            <span className="text-foreground">
              {sons.map((s) => `${SURNAME_HANZI}${s}`).join("、")}
            </span>
          </div>
        )}
        {daughters.length > 0 && (
          <div className="flex gap-2">
            <span className="shrink-0 text-muted-foreground text-xs">{t("bioDaughters")}</span>
            <span className="text-foreground">
              {daughters.map((s) => `${SURNAME_HANZI}${s}`).join("、")}
            </span>
          </div>
        )}
        {fullText && (
          <details className="mt-1">
            <summary className="cursor-pointer text-muted-foreground text-xs hover:text-foreground">
              {t("bioFullText")}
            </summary>
            <p className="mt-1.5 whitespace-pre-wrap text-foreground text-sm leading-relaxed">
              {fullText}
            </p>
          </details>
        )}
        <p className="text-muted-foreground text-xs italic">{t("bioOcrNote")}</p>
      </div>
    </Section>
  );
}

// A titled block in the panel.
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 font-semibold text-muted-foreground text-xs uppercase tracking-wide">
        {title}
      </h3>
      {children}
    </div>
  );
}

// A clickable relative (father / child / ancestor). Clicking selects them,
// which re-centers the panel on that person and re-highlights the tree.
function RelativeButton({
  person,
  onSelect,
}: {
  person: FamilyNode;
  onSelect: (id: number | null) => void;
}) {
  const { lang } = useLang();
  return (
    <button
      type="button"
      onClick={() => onSelect(person.id)}
      className="flex min-w-0 flex-1 items-center gap-2 rounded-md border border-border bg-background px-2 py-1.5 text-left transition-colors hover:border-primary hover:bg-muted"
    >
      <span className="flex h-6 shrink-0 items-center whitespace-nowrap">
        <PersonLabel person={person} size="sm" />
      </span>
      {person.name.trim() && (
        <span className="truncate text-muted-foreground text-xs">
          <SurnamePinyin />
          {toDisplayPinyin(person.name)}
        </span>
      )}
      <span className="ml-auto shrink-0 text-muted-foreground text-xs">
        {idLabel(lang, person.id)}
      </span>
    </button>
  );
}

// Render a person's identity: their Unicode name if known, else the scanned
// name-image crop, else just the id. Shared by the header and every relative.
function PersonLabel({ person, size }: { person: FamilyNode; size: "sm" | "lg" }) {
  const hasName = person.name.trim().length > 0;
  const px = size === "lg" ? 28 : 20;

  if (hasName) {
    // Relative cards (sm) get the muted surname prefix; the header avatar (lg) is
    // a compact monogram, so it stays the given name only.
    return (
      <span className={size === "lg" ? "text-lg" : "text-base"}>
        {size === "sm" && <SurnameHanzi />}
        {person.name}
      </span>
    );
  }
  if (person.name_images[0]) {
    // The crops are dark ink on light paper; a subtle rounded frame keeps them
    // readable against both panel backgrounds. A plain <img> (not next/image) is
    // deliberate: these are tiny static local PNGs where optimization adds no value.
    return (
      // biome-ignore lint/performance/noImgElement: tiny static local name-crop PNG; next/image optimization adds no value here.
      <img
        src={person.name_images[0]}
        alt={`Name of person #${person.id}`}
        width={px}
        height={px}
        className="rounded-sm bg-white object-contain"
        style={{ width: px, height: px }}
      />
    );
  }
  return <span className="text-muted-foreground text-xs">#{person.id}</span>;
}
