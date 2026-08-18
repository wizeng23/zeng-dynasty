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

import { ancestryChain, type FamilyNode, type LoadedTree } from "@/lib/tree";

interface DetailPanelProps {
  tree: LoadedTree;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}

export function DetailPanel({ tree, selectedId, onSelect }: DetailPanelProps) {
  // Empty state — nothing selected yet.
  if (selectedId === null) {
    return (
      <aside className="flex w-full flex-col border-border border-l bg-card p-5 text-sm sm:w-80">
        <p className="text-muted-foreground">
          Click a person in the tree to trace their lineage and see their details here.
        </p>
      </aside>
    );
  }

  const person = tree.byId.get(selectedId);
  if (!person) {
    return (
      <aside className="w-full border-border border-l bg-card p-5 text-sm sm:w-80">
        <p className="text-muted-foreground">Person #{selectedId} not found.</p>
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
      {/* Header: the selected person's name (or image), generation, close button. */}
      <div>
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
              <PersonLabel person={person} size="lg" />
            </div>
            <div>
              <div className="text-muted-foreground text-xs uppercase tracking-wide">
                Generation {person.generation}
                {isRoot && " · lineage root"}
              </div>
              <div className="font-medium">#{person.id}</div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => onSelect(null)}
            aria-label="Clear selection"
            className="rounded-md px-2 py-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Father. */}
      <Section title="Father">
        {father ? (
          <RelativeButton person={father} onSelect={onSelect} />
        ) : (
          <p className="text-muted-foreground">None recorded — this is a lineage root.</p>
        )}
      </Section>

      {/* Children (eldest-first). */}
      <Section title={`Children (${children.length})`}>
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
          <p className="text-muted-foreground">No children recorded.</p>
        )}
      </Section>

      {/* Nearby ancestors: immediate father up to the lineage root. */}
      <Section title="Ancestors to root">
        {ancestors.length > 0 ? (
          <ol className="flex flex-col gap-1.5">
            {ancestors.map((anc) => (
              <li key={anc.id} className="flex items-center gap-2">
                <span className="w-8 shrink-0 text-muted-foreground text-xs">
                  gen {anc.generation}
                </span>
                <RelativeButton person={anc} onSelect={onSelect} />
              </li>
            ))}
          </ol>
        ) : (
          <p className="text-muted-foreground">This person is already the root.</p>
        )}
      </Section>

      {/* Biography / notes, only if present. In the golden data biography is
          sometimes a bare Wikipedia URL, so render those as a link. */}
      {person.biography.trim() && (
        <Section title="Biography">
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
        <Section title="Notes">
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
  return (
    <button
      type="button"
      onClick={() => onSelect(person.id)}
      className="flex min-w-0 flex-1 items-center gap-2 rounded-md border border-border bg-background px-2 py-1.5 text-left transition-colors hover:border-primary hover:bg-muted"
    >
      <span className="flex h-6 w-6 shrink-0 items-center justify-center">
        <PersonLabel person={person} size="sm" />
      </span>
      <span className="truncate text-muted-foreground text-xs">#{person.id}</span>
    </button>
  );
}

// Render a person's identity: their Unicode name if known, else the scanned
// name-image crop, else just the id. Shared by the header and every relative.
function PersonLabel({ person, size }: { person: FamilyNode; size: "sm" | "lg" }) {
  const hasName = person.name.trim().length > 0;
  const px = size === "lg" ? 28 : 20;

  if (hasName) {
    return <span className={size === "lg" ? "text-lg" : "text-base"}>{person.name}</span>;
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
