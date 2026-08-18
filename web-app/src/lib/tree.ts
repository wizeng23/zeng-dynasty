// ---------------------------------------------------------------------------
// tree.ts — data loading + d3.hierarchy construction
//
// This is the "d3 owns the MATH" half of the app. Nothing here touches the DOM.
// It fetches a JSONL dataset, turns the flat list of nodes into a nested
// structure, and hands d3 a single tree to lay out. The React component
// (FamilyTree.tsx) consumes the result and does all the rendering.
// ---------------------------------------------------------------------------

import { type HierarchyNode, hierarchy } from "d3-hierarchy";

// On GitHub Pages the site is served under "/zeng-dynasty", so every absolute
// asset URL ("/data/...", "/names/...") needs that prefix. Next injects the
// base path as NEXT_PUBLIC_BASE_PATH at build time (empty for local dev). Next
// prefixes its OWN assets automatically, but NOT our runtime fetch()/<image>
// URLs — so we prefix those ourselves through this helper.
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export function assetPath(path: string): string {
  return `${BASE_PATH}${path}`;
}

// -- The Node schema, mirrored from src/model.py (the Python source of truth). --
// A person in the genealogy. Names may be a Unicode string (golden data) OR a
// list of image crops (parsed books, where OCR hasn't run yet).
export interface FamilyNode {
  id: number;
  name: string;
  name_images: string[];
  generation: number; // 1-indexed: root 点 = generation 1
  father: number; // -1 means this node is a root (no recorded father)
  children: number[]; // child IDs, ordered eldest-first (right-to-left in the book)
  biography: string;
  notes: string;
}

// The shape d3.hierarchy walks. We keep the raw FamilyNode payload on `data`
// and give d3 a `children` accessor that returns nested FamilyDatum objects.
// The synthetic super-root (see below) is the one node whose `node` is null.
export interface FamilyDatum {
  // The real person, or null for the synthesized virtual root that only exists
  // to join multiple real roots into one tree d3 can lay out.
  node: FamilyNode | null;
  children: FamilyDatum[];
}

export interface LoadedTree {
  // d3's laid-out hierarchy root. When the dataset has >1 real root, this is the
  // virtual super-root; its .children are the real roots.
  root: HierarchyNode<FamilyDatum>;
  nodeCount: number; // real people, excluding the virtual root
  rootCount: number; // number of real roots (separate subtrees / lineages)
  isForest: boolean; // true when rootCount > 1 (a virtual root was synthesized)
  // Flat lookup of the raw people by id — lets the UI resolve father/children
  // for the detail panel without re-walking the hierarchy.
  byId: Map<number, FamilyNode>;
}

export type DatasetName = "book1_golden" | "book1" | "book2";

export const DATASETS: { name: DatasetName; label: string; blurb: string }[] = [
  {
    name: "book1_golden",
    label: "Golden (Book 1)",
    blurb: "One verified lineage with real Unicode names — the showcase dataset.",
  },
  {
    name: "book1",
    label: "Book 1 (parsed)",
    blurb: "Algorithmic parse — a fragmented forest with scanned name images.",
  },
  {
    name: "book2",
    label: "Book 2 (parsed)",
    blurb: "The largest parse: 1,763 people across many disconnected lineages.",
  },
];

// Parse a JSONL blob (one JSON object per line) into FamilyNodes.
function parseJsonl(text: string): FamilyNode[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as FamilyNode);
}

// Turn the flat node list into a nested FamilyDatum tree for d3.
//
// Two things worth understanding here:
//
// 1. FORESTS -> ONE TREE. d3.hierarchy needs exactly one root. The parsed books
//    are fragmented (many father==-1 roots), so when there's more than one we
//    synthesize a virtual super-root ({ node: null }) whose children are the
//    real roots. FamilyTree.tsx hides that node and draws each real root as its
//    own labeled subtree. We never invent parent/child links between real
//    people — the virtual root is purely a layout scaffold.
//
// 2. RIGHT-TO-LEFT / ELDEST-FIRST. The book reads right-to-left, so the eldest
//    child (children[0]) belongs on the RIGHT. d3.tree lays children out
//    left-to-right in array order, so to put the eldest on the right we REVERSE
//    each children array here. After reversing, index 0 renders leftmost =
//    youngest, and the last entry renders rightmost = eldest.
function buildData(nodes: FamilyNode[]): {
  data: FamilyDatum;
  rootCount: number;
  byId: Map<number, FamilyNode>;
} {
  const byId = new Map<number, FamilyNode>();
  for (const n of nodes) byId.set(n.id, n);

  // Recursively wrap a person and their descendants into FamilyDatum objects.
  // `seen` guards against malformed data with cycles so we can't loop forever.
  const seen = new Set<number>();
  function wrap(node: FamilyNode): FamilyDatum {
    seen.add(node.id);
    // Reverse a COPY of children[] so eldest (index 0) ends up rendered on the
    // right. Skip ids we've already visited (cycle guard) or that don't exist.
    const childData = [...node.children]
      .reverse()
      .filter((id) => byId.has(id) && !seen.has(id))
      .map((id) => wrap(byId.get(id) as FamilyNode));
    return { node, children: childData };
  }

  const roots = nodes.filter((n) => n.father === -1);
  const rootData = roots.map((r) => wrap(r));

  if (rootData.length === 1) {
    // Already a single tree — no scaffolding needed.
    return { data: rootData[0], rootCount: 1, byId };
  }

  // Multiple (or zero) roots: wrap them under a virtual super-root.
  return {
    data: { node: null, children: rootData },
    rootCount: rootData.length,
    byId,
  };
}

// Fetch + parse + build a d3.hierarchy for one dataset.
export async function loadTree(name: DatasetName): Promise<LoadedTree> {
  const res = await fetch(assetPath(`/data/${name}.jsonl`));
  if (!res.ok) throw new Error(`Failed to load /data/${name}.jsonl (${res.status})`);
  const nodes = parseJsonl(await res.text());

  // Prefix each name-image crop with the base path too, so the components can
  // use name_images[0] directly as an <image>/<img> src without knowing about
  // deployment. The exported JSONL stores site-absolute paths like
  // "/names/book1/1.png".
  for (const node of nodes) {
    node.name_images = node.name_images.map((p) => (p.startsWith("/") ? assetPath(p) : p));
  }

  const { data, rootCount, byId } = buildData(nodes);

  // hierarchy() reads our `children` accessor and computes depth/height/parent
  // links. It does NOT position anything — d3.tree (in the component) does that.
  const root = hierarchy<FamilyDatum>(data, (d) => d.children);

  return {
    root,
    nodeCount: nodes.length,
    rootCount,
    isForest: rootCount > 1,
    byId,
  };
}

// Convenience: is this datum the synthesized virtual root (not a real person)?
export function isVirtualRoot(d: HierarchyNode<FamilyDatum>): boolean {
  return d.data.node === null;
}

// ---------------------------------------------------------------------------
// Ancestry helpers — pure data walks the UI uses for highlighting + the panel.
//
// The "path to root" is the chain of fathers from a person up to the top of
// THEIR lineage (their sub-root), NOT up to the synthesized virtual root — the
// virtual root isn't a real ancestor, so we never include it. In a single-root
// dataset (golden) this is simply the walk up to the one true root.
// ---------------------------------------------------------------------------

// The clicked person plus every father above them, ordered SELF -> ... -> root.
// Follows the `father` pointers through byId so it works identically whether or
// not a virtual root was synthesized. Cycle-guarded against malformed data.
export function ancestryChain(id: number, byId: Map<number, FamilyNode>): FamilyNode[] {
  const chain: FamilyNode[] = [];
  const seen = new Set<number>();
  let current = byId.get(id);
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    chain.push(current);
    if (current.father === -1) break; // reached this lineage's root
    current = byId.get(current.father);
  }
  return chain;
}

// Just the ids on the path to root, as a Set — the shape the renderer wants for
// an O(1) "is this node highlighted?" check while drawing.
export function pathToRootIds(id: number | null, byId: Map<number, FamilyNode>): Set<number> {
  if (id === null) return new Set();
  return new Set(ancestryChain(id, byId).map((n) => n.id));
}
