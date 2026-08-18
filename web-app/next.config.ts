import type { NextConfig } from "next";

// GitHub Pages serves this repo at https://wizeng23.github.io/zeng-dynasty/,
// so the site lives under the "/zeng-dynasty" base path, not the domain root.
// `basePath` makes Next emit that prefix on its own routes/assets; we expose it
// to the client as NEXT_PUBLIC_BASE_PATH so our own `fetch("/data/...")` calls
// can prefix it too (Next does NOT rewrite fetch URLs automatically).
// A local `bun run dev` / `bun run build` uses no prefix; the Pages build sets
// GITHUB_PAGES=true (see the deploy workflow) to turn it on.
const isGithubPages = process.env.GITHUB_PAGES === "true";
const basePath = isGithubPages ? "/zeng-dynasty" : "";

const nextConfig: NextConfig = {
  // Emit a fully static site (plain HTML/JS/CSS) into `out/` — no Node server,
  // which is what GitHub Pages can host.
  output: "export",
  basePath,
  // Static export can't run Next's on-the-fly image optimizer.
  images: { unoptimized: true },
  // Make the base path readable at runtime for our data/image fetches.
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  // Pages serves each route as a folder with an index.html.
  trailingSlash: true,
};

export default nextConfig;
