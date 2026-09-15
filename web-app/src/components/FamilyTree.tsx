"use client";

// ---------------------------------------------------------------------------
// FamilyTree.tsx — the "React owns the RENDERING" half.
//
// THE KEY PATTERN (this is the concept worth internalizing):
//
//   • d3 owns the MATH.  d3.tree() reads the hierarchy and assigns every node
//     an (x, y). d3-zoom computes the pan/zoom transform matrix. That's it —
//     pure computation, no DOM.
//
//   • React owns the DOM.  We render the SVG ourselves with JSX (<g>, <path>,
//     <text>, <image>) from the coordinates d3 gave us. We do NOT let d3 select
//     nodes and append/mutate elements — that would fight React for control of
//     the DOM. The classic d3 tutorials do `svg.selectAll(...).append(...)`;
//     in React that's an anti-pattern.
//
//   • The ONE exception: d3-zoom. It needs to attach wheel/drag listeners to a
//     real DOM node and read live event state, so we bind it to the <svg> ref
//     inside useEffect. It writes the resulting transform back into React state,
//     and React re-renders the <g> with that transform. So even here, d3 does
//     the math and React does the rendering.
//
// INTERACTION (this file's second job):
//
//   • SELECTION lives in React state, lifted to the page so the side panel and
//     the tree share it. Clicking a node selects it; clicking empty space
//     clears. The "path to root" (the chain of fathers) is highlighted and the
//     rest is dimmed — entirely via conditional CSS classes / props, never by
//     mutating the DOM. Because the highlight is derived from state, it survives
//     panning and zooming for free (a re-render just re-applies the classes).
//
//   • HOVER is local state here (the panel doesn't care about it) and adds a
//     subtle emphasis to whichever node the pointer is over.
// ---------------------------------------------------------------------------

import { tree as d3tree, type HierarchyNode, type HierarchyPointNode } from "d3-hierarchy";
import { select } from "d3-selection";
import { zoom as d3zoom, type ZoomTransform, zoomIdentity } from "d3-zoom";
import { useEffect, useMemo, useRef, useState } from "react";
import { type FamilyDatum, isVirtualRoot, type LoadedTree, pathToRootIds } from "@/lib/tree";

// Layout constants. nodeSize is [horizontalGap, verticalGap] BETWEEN sibling
// slots and BETWEEN generations. Chinese names are tall/narrow; book2 names are
// 2-char vertical stacks, so we give generous vertical room and enough
// horizontal room that single glyphs never collide.
const NODE_DX = 46; // horizontal spacing between sibling nodes
const NODE_DY = 92; // vertical spacing between generations (depth)
// Clear horizontal gap inserted to the LEFT of the main lineage for each
// unattached side-branch root (e.g. 贞年), so it reads as a separate branch
// rather than colliding with the main tree at the same generation row.
const ORPHAN_GAP = 260;
const NODE_R = 16; // node circle radius
const NAME_IMG = 30; // rendered size of a name-image crop

interface FamilyTreeProps {
  tree: LoadedTree;
  // Selection is owned by the page so the detail panel can share it. null = none.
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}

export function FamilyTree({ tree, selectedId, onSelect }: FamilyTreeProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  // The live pan/zoom transform, owned by React state but WRITTEN by d3-zoom.
  const [transform, setTransform] = useState<ZoomTransform>(zoomIdentity);

  // Hover is purely visual and local to the tree — the panel doesn't need it.
  const [hoveredId, setHoveredId] = useState<number | null>(null);

  // --- d3 owns the MATH: compute the layout once per dataset. -------------
  // useMemo so we only re-run d3.tree() when the underlying hierarchy changes,
  // not on every pan/zoom re-render.
  const { nodes, links, centerX, minY } = useMemo(() => {
    // nodeSize gives fixed spacing (vs .size() which stretches to fit a box).
    // Fixed spacing is what keeps names from overlapping regardless of tree size.
    const layout = d3tree<FamilyDatum>().nodeSize([NODE_DX, NODE_DY]);

    // hierarchy() was built in tree.ts; layout() decorates each node with x/y.
    const positioned = layout(tree.root as HierarchyNode<FamilyDatum>);

    // Vertical position from the true GENERATION, not d3's tree depth. The two
    // agree along the main lineage, but an unattached root (贞年, a 贞-generation
    // branch with no father in our data) sits at d3-depth 1 while its real
    // generation is 65 — so we pin every node's y to its generation. This drops
    // 贞年's subtree down to the same rows as the rest of generation 65+, instead
    // of floating it at the top beside 点. The virtual root (no person) keeps its
    // computed y; it isn't drawn.
    for (const n of positioned.descendants()) {
      const gen = n.data.node?.generation;
      if (gen !== undefined) n.y = gen * NODE_DY;
    }

    // Separate each unattached side-branch from the main lineage horizontally.
    // d3 lays every root's subtree in its own x-band, but the bands abut, so a
    // side root pinned to a deep generation (贞年 at gen 65) would butt right up
    // against the main tree's people on that same row. The roots come from
    // tree.ts sorted deepest-first, so the main lineage (点, gen 0) is LAST; the
    // earlier roots are the side-branches. Shift each side-branch subtree left of
    // the main tree's leftmost node, with a fixed gap.
    const rootData = positioned.children ?? [];
    if (rootData.length > 1) {
      const main = rootData[rootData.length - 1]; // 点's tree (added last)
      const mainMinX = Math.min(...main.descendants().map((n) => n.x));
      let cursor = mainMinX; // left edge we pack side-branches up against
      for (let i = rootData.length - 2; i >= 0; i--) {
        const branch = rootData[i];
        const desc = branch.descendants();
        const bMax = Math.max(...desc.map((n) => n.x));
        // Move the branch so its rightmost node sits ORPHAN_GAP left of `cursor`.
        const shift = cursor - ORPHAN_GAP - bMax;
        for (const n of desc) n.x += shift;
        cursor = Math.min(...desc.map((n) => n.x)); // next branch goes further left
      }
    }

    // Real nodes only — drop the synthetic virtual root from what we DRAW.
    // (Its children keep their computed positions; we just don't paint it or
    //  the links leading into it.)
    const drawNodes = positioned.descendants().filter((n) => !isVirtualRoot(n));
    const drawLinks = positioned.links().filter((l) => !isVirtualRoot(l.source)); // skip links from the virtual root

    // Drawing bounds, so the initial view can be centered on the tree.
    const xs = drawNodes.map((n) => n.x);
    const ys = drawNodes.map((n) => n.y);

    return {
      nodes: drawNodes,
      links: drawLinks,
      centerX: (Math.min(...xs) + Math.max(...xs)) / 2, // horizontal midpoint
      minY: Math.min(...ys), // topmost row (the roots)
    };
  }, [tree]);

  // --- The highlight set: ids on the path from the selected node to its root. -
  // Pure derived state (from tree.ts). Because it's derived, panning/zooming —
  // which only change `transform` — never disturb it.
  const pathIds = useMemo(() => pathToRootIds(selectedId, tree.byId), [selectedId, tree.byId]);
  const hasSelection = selectedId !== null;

  // --- d3-zoom: the one place a d3 selection on a ref is idiomatic. --------
  useEffect(() => {
    if (!svgRef.current) return;
    const svg = select(svgRef.current);

    const zoomBehavior = d3zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4]) // min/max zoom
      // On every zoom/pan gesture, d3 hands us a fresh transform. We push it
      // into React state; React re-renders the <g transform=...>. d3 computed
      // it, React applied it.
      .on("zoom", (event) => setTransform(event.transform));

    svg.call(zoomBehavior);

    // Center the tree on first mount. Translate so the tree's horizontal
    // midpoint sits at the viewport center, and its topmost row (the roots)
    // sits a little below the top edge (generations grow downward).
    const svgWidth = svgRef.current.clientWidth;
    const initial = zoomIdentity.translate(svgWidth / 2 - centerX, NODE_DY - minY);
    svg.call(zoomBehavior.transform, initial);

    // Clean up listeners if the dataset changes / component unmounts.
    return () => {
      svg.on(".zoom", null);
    };
    // Re-center when the dataset (and thus centerX/minY) changes.
  }, [centerX, minY]);

  return (
    // Clicking empty canvas clears the selection (a mouse convenience — the
    // detail panel's close button is the keyboard-accessible equivalent, so the
    // canvas needs no keyboard handler of its own). Node clicks stopPropagation
    // so they don't bubble up here.
    // biome-ignore lint/a11y/useKeyWithClickEvents: same — the canvas background is a mouse-only deselect convenience, not a keyboard control.
    <svg
      ref={svgRef}
      role="img"
      aria-label="Family tree diagram — scroll to zoom, drag to pan, click a person to trace their lineage"
      className="h-full w-full cursor-grab touch-none select-none active:cursor-grabbing"
      onClick={() => onSelect(null)}
      // The outer <svg> is the fixed viewport; the inner <g> is what we pan/zoom.
    >
      <title>Zeng family tree</title>
      {/* React renders the transform d3-zoom computed. */}
      <g transform={transform.toString()}>
        {/* Links first so nodes paint on top of them. */}
        {links.map((link) => {
          // A link is "on the path" when BOTH endpoints are in the chain — i.e.
          // it's a father->child edge along the lineage to root.
          const sourceId = link.source.data.node?.id;
          const targetId = link.target.data.node?.id;
          const onPath =
            sourceId !== undefined &&
            targetId !== undefined &&
            pathIds.has(sourceId) &&
            pathIds.has(targetId);
          return (
            <path
              key={`${sourceId ?? "v"}-${targetId}`}
              d={elbowPath(link.source, link.target)}
              fill="none"
              // Highlighted edges use the seal-red accent and a thicker stroke;
              // when there's a selection, off-path edges dim back.
              stroke={onPath ? "var(--primary)" : "var(--border)"}
              strokeWidth={onPath ? 2.5 : 1.5}
              className="transition-opacity duration-200"
              opacity={hasSelection && !onPath ? 0.15 : 1}
            />
          );
        })}

        {/* Nodes. */}
        {nodes.map((node) => {
          const id = node.data.node?.id;
          const onPath = id !== undefined && pathIds.has(id);
          return (
            <NodeMark
              key={id}
              node={node}
              isSelected={id === selectedId}
              onPath={onPath}
              dimmed={hasSelection && !onPath}
              isHovered={id === hoveredId}
              onSelect={onSelect}
              onHover={setHoveredId}
            />
          );
        })}
      </g>
    </svg>
  );
}

interface NodeMarkProps {
  node: HierarchyPointNode<FamilyDatum>;
  isSelected: boolean; // the clicked node itself
  onPath: boolean; // on the highlighted path to root
  dimmed: boolean; // there's a selection and this node isn't on the path
  isHovered: boolean;
  onSelect: (id: number | null) => void;
  onHover: (id: number | null) => void;
}

// One person: a circle plus either their Unicode name or a name-image crop.
function NodeMark({
  node,
  isSelected,
  onPath,
  dimmed,
  isHovered,
  onSelect,
  onHover,
}: NodeMarkProps) {
  const person = node.data.node;
  if (!person) return null; // virtual root — never drawn

  const hasName = person.name.trim().length > 0;
  const isRoot = person.father === -1;

  // Fill/stroke priority: selected node is the strongest, then path/root accent,
  // then plain. Selected + on-path both read as seal red so the chain is one color.
  const accented = isSelected || onPath || isRoot;
  const strokeWidth = isSelected ? 3 : isHovered ? 2.5 : 1.5;

  // A readable label for assistive tech: the Unicode name if we have one, else
  // the id (the image crop has no text we can announce).
  const ariaLabel = hasName ? `Person ${person.name}` : `Person #${person.id}`;

  return (
    // biome-ignore lint/a11y/useSemanticElements: an SVG <g> can't be a real <button> — buttons aren't valid SVG children. role="button" + keyboard handlers below give it equivalent semantics.
    <g
      transform={`translate(${node.x},${node.y})`}
      className="cursor-pointer transition-opacity duration-200 focus:outline-none"
      opacity={dimmed ? 0.2 : 1}
      // Focusable + keyboard-activatable so the tree isn't mouse-only.
      role="button"
      tabIndex={0}
      aria-label={ariaLabel}
      aria-pressed={isSelected}
      onClick={(e) => {
        // Don't let the click reach the <svg>, which would clear the selection.
        e.stopPropagation();
        onSelect(person.id);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          e.stopPropagation();
          onSelect(person.id);
        }
      }}
      onMouseEnter={() => onHover(person.id)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(person.id)}
      onBlur={() => onHover(null)}
    >
      {/* Selection ring: a faint halo behind the node so the selected person
          pops even inside a dense forest. */}
      {isSelected && (
        <circle
          r={NODE_R + 5}
          fill="none"
          stroke="var(--primary)"
          strokeWidth={1.5}
          opacity={0.5}
        />
      )}
      <circle
        r={NODE_R}
        // Selected / on-path / root all take the seal-red accent fill; everyone
        // else is a plain card circle.
        fill={accented ? "var(--primary)" : "var(--card)"}
        stroke={accented ? "var(--primary)" : "var(--border)"}
        strokeWidth={strokeWidth}
        className="transition-all duration-150"
      />
      {hasName ? (
        // Unicode name (golden data): render as centered text.
        <text
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={18}
          fill={accented ? "var(--primary-foreground)" : "var(--foreground)"}
          style={{ fontFamily: "var(--font-sans)" }}
        >
          {person.name}
        </text>
      ) : person.name_images[0] ? (
        // No Unicode name yet: show the scanned crop, centered on the node.
        <image
          href={person.name_images[0]}
          x={-NAME_IMG / 2}
          y={-NAME_IMG / 2}
          width={NAME_IMG}
          height={NAME_IMG}
          preserveAspectRatio="xMidYMid meet"
        />
      ) : (
        // Neither name nor image — show the id so nothing is silently blank.
        <text
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={10}
          fill="var(--muted-foreground)"
        >
          {person.id}
        </text>
      )}
    </g>
  );
}

// A right-angle ("elbow") connector from a parent down to a child: drop halfway,
// go across, then drop the rest. Reads more like a printed genealogy than a
// diagonal line would.
function elbowPath(
  source: HierarchyPointNode<FamilyDatum>,
  target: HierarchyPointNode<FamilyDatum>,
): string {
  const midY = (source.y + target.y) / 2;
  return `M${source.x},${source.y} V${midY} H${target.x} V${target.y}`;
}
