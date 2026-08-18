# Zeng Family Tree — Web App

An interactive viewer for the digitized Zeng family tree (曾氏族谱 / *zupu*). It
renders the parsed genealogy as a pan/zoom [d3](https://d3js.org/) tree: click a
person to trace their lineage back to the root and read their details in a side
panel. This is the **static website** milestone — it reads pre-exported data and
has no database, editing, or auth (those come in a later milestone).

## What it shows

Three datasets, switchable in the header:

| Dataset | People | Roots | Names | Notes |
| --- | --- | --- | --- | --- |
| **Golden (Book 1)** | 59 | 1 | real Unicode (点, 参, …) | the showcase — one clean, verified lineage. Default view. |
| **Book 1 (parsed)** | 163 | 14 | scanned image crops | algorithmic parse; a fragmented forest. |
| **Book 2 (parsed)** | 1,763 | 181 | scanned image crops | the largest parse; many disconnected lineages. |

The UI handles both shapes honestly:

- **Forests render as a forest.** When a dataset has more than one root, the app
  synthesizes an invisible virtual root purely so d3 has a single tree to lay
  out — it draws each real root as its own labeled subtree and never invents
  parent/child links between real people.
- **Names fall back to image crops.** Golden data has real Unicode names; the
  parsed books don't (OCR hasn't run yet), so each node renders its scanned
  name-image crop instead.
- **Right-to-left / eldest-first.** The book reads right-to-left, so the eldest
  sibling is drawn on the **right**. Node ordering honors this.

## Stack

- **[Next.js 16](https://nextjs.org/)** (App Router, Turbopack) + **React 19** + **TypeScript**
- **[d3-hierarchy](https://d3js.org/d3-hierarchy)** for tree layout math
- **[Tailwind CSS v4](https://tailwindcss.com/)** (via `@tailwindcss/postcss`)
- **[next-themes](https://github.com/pacocoursey/next-themes)** for light/dark
- **[Biome](https://biomejs.dev/)** for lint + format
- Install with **[bun](https://bun.sh/)**; the app runs on Node.

## Running it

```bash
# 1. Install dependencies
bun install

# 2. Export the parsed data into public/ (copies the 3 JSONL files and the
#    name-crop images from the repo, rewriting image paths to public URLs).
#    Re-run this whenever the upstream data/ or books/*/names/ change.
bun run export-data

# 3. Start the dev server (http://localhost:3000)
bun run dev
```

Other scripts:

```bash
bun run build       # production build
bun run lint        # biome check (no writes)
bun run lint:fix    # biome check --write
```

### Where the data comes from

`bun run export-data` (`scripts/export-web-data.ts`) is the bridge between the
Python parsing pipeline and this app. It:

1. copies `data/book1_golden.jsonl`, `data/book1.jsonl`, `data/book2.jsonl` into
   `public/data/`,
2. rewrites each parsed node's `name_images` paths (e.g.
   `books/book1/names/1.png` → `/names/book1/1.png`) so the browser can fetch
   them, and
3. copies `books/book{1,2}/names/` into `public/names/`.

The exported files under `public/data/` and `public/names/` are what the app
fetches at runtime — the app never reaches outside `public/`.

## Architecture: d3 computes, React renders

The one idea worth internalizing. Responsibilities are split cleanly:

- **`src/lib/tree.ts` — d3 owns the MATH, and touches no DOM.** It fetches a
  JSONL dataset, builds a nested structure, and calls `d3.hierarchy` to compute
  depth / parent links. It also exposes pure ancestry helpers (`ancestryChain`,
  `pathToRootIds`) and a `byId` lookup map. Everything here is plain data.
- **`src/components/FamilyTree.tsx` — React owns the DOM.** It runs `d3.tree` to
  get x/y coordinates, then renders every node and edge as real JSX/SVG that
  React controls. d3 never mutates the DOM here.

Why this matters: highlighting the path-to-root is **derived state**, not a DOM
mutation. The selected person is lifted to the page (`src/app/page.tsx`) and
shared by the tree and the panel. Given a selection, `pathToRootIds` produces the
set of highlighted ids, and the render applies conditional styles (seal-red on
the path, dimmed elsewhere). Because it's derived from props, the highlight
survives pan/zoom automatically — there's no imperative "re-highlight" step.

```
page.tsx           holds selectedId; lays out tree + panel; dataset switcher
├── FamilyTree.tsx  d3.tree layout → JSX/SVG nodes+edges; path highlight; pan/zoom
├── DetailPanel.tsx selected person: father, children, ancestors-to-root, bio/notes
└── lib/tree.ts     loadTree(), d3.hierarchy, ancestry helpers  (no DOM)
```

## Current limitations

- **The parsed books are fragmented.** Book 1 parses into 14 disconnected roots
  and Book 2 into 181 — the real lineages are split across graph images that
  haven't been stitched together yet. The forest view is honest about this; it
  does **not** fake connectivity. Cross-graph stitching is upstream pipeline
  work.
- **Parsed names are images, not text.** Nodes in Book 1/Book 2 show scanned
  name crops because OCR (pipeline stage 4) hasn't run. Once names are
  transcribed to Unicode, they'll render as text like the golden data.
- **No pan-to-selection.** Clicking a node selects and highlights it, but the
  viewport doesn't recenter on it. In a large forest the lit chain can be
  off-screen after a click. A programmatic `zoom.transform` pan-to-selection
  would be a good follow-up.
- **No favicon** — the browser logs a harmless `favicon.ico` 404.
