"use client";

// The home page: a dataset switcher over the d3 family-tree view.
//
// This is a Client Component because it holds interactive state (which dataset
// is selected) and fetches the JSONL at runtime. The tree math + rendering live
// in FamilyTree / lib/tree; this file just wires up the controls and stats.

import { useEffect, useState } from "react";
import { DetailPanel } from "@/components/DetailPanel";
import { FamilyTree } from "@/components/FamilyTree";
import { ThemeToggle } from "@/components/ThemeToggle";
import { DATASETS, type DatasetName, type LoadedTree, loadTree } from "@/lib/tree";

export default function Home() {
  const [dataset, setDataset] = useState<DatasetName>("tree");
  const [tree, setTree] = useState<LoadedTree | null>(null);
  const [error, setError] = useState<string | null>(null);

  // The selected person — shared by the tree (for path-to-root highlighting)
  // and the detail panel. null = nothing selected. Lifted here so both children
  // read and write the same value.
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // Reload whenever the selected dataset changes. The `cancelled` flag guards
  // against a race where the user switches datasets before a fetch resolves.
  useEffect(() => {
    let cancelled = false;
    setTree(null);
    setError(null);
    setSelectedId(null); // ids aren't shared across datasets — clear on switch
    loadTree(dataset)
      .then((loaded) => {
        if (!cancelled) setTree(loaded);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  const active = DATASETS.find((d) => d.name === dataset);

  return (
    <main className="flex h-screen flex-col">
      {/* Header: title, dataset switcher, and live stats. */}
      <header className="border-border border-b px-6 py-4">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h1 className="font-semibold text-xl tracking-tight">
              Zeng Family Tree <span className="text-primary">曾氏族谱</span>
            </h1>
            {active && <p className="mt-1 text-muted-foreground text-sm">{active.blurb}</p>}
          </div>

          {/* Stats + theme toggle. */}
          <div className="flex items-center gap-4">
            {tree && (
              <div className="flex gap-4 text-sm">
                <Stat label="people" value={tree.nodeCount.toLocaleString()} />
                <Stat
                  label={tree.rootCount === 1 ? "lineage" : "lineages"}
                  value={tree.rootCount.toLocaleString()}
                />
              </div>
            )}
            <ThemeToggle />
          </div>
        </div>

        {/* Dataset switcher — shown only when there is more than one dataset. */}
        {DATASETS.length > 1 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {DATASETS.map((d) => (
              <button
                key={d.name}
                type="button"
                onClick={() => setDataset(d.name)}
                className={`rounded-md border px-3 py-1.5 font-medium text-sm transition-colors ${
                  d.name === dataset
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-card text-foreground hover:bg-muted"
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>
        )}
      </header>

      {/* Body: the tree viewport on the left, the detail panel on the right.
          On narrow screens they stack (panel below). */}
      <div className="flex min-h-0 flex-1 flex-col sm:flex-row">
        <div className="relative min-h-0 flex-1 bg-background">
          {error && (
            <p className="absolute inset-0 flex items-center justify-center text-primary">
              Failed to load: {error}
            </p>
          )}
          {!error && !tree && (
            <p className="absolute inset-0 flex items-center justify-center text-muted-foreground">
              Loading {active?.label}…
            </p>
          )}
          {!error && tree && (
            <>
              <FamilyTree tree={tree} selectedId={selectedId} onSelect={setSelectedId} />
              <p className="pointer-events-none absolute bottom-3 left-4 text-muted-foreground text-xs">
                Eldest sibling on the right · scroll to zoom, drag to pan, click to trace lineage
              </p>
            </>
          )}
        </div>

        {/* Detail panel — always mounted so its width reserves layout space;
            it renders its own empty state when nothing is selected. */}
        {!error && tree && (
          <DetailPanel tree={tree} selectedId={selectedId} onSelect={setSelectedId} />
        )}
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-right">
      <div className="font-semibold text-foreground tabular-nums">{value}</div>
      <div className="text-muted-foreground text-xs uppercase tracking-wide">{label}</div>
    </div>
  );
}
