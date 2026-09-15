/**
 * Snapshot the parsed tree data + name-crop images into web-app/public so the
 * static site can fetch them at runtime.
 *
 * What it does:
 *   1. Copies data/book1_stitched.jsonl into public/data/.
 *   2. Rewrites each node's `name_images` path from the repo path
 *      ("books/book1/5_names/1.png") to the public URL the browser can load
 *      ("/names/book1/1.png").
 *   3. Copies books/book1/5_names/ -> public/names/book1/.
 *
 * Run from web-app/:  bun run export-data   (or: npx tsx scripts/export-web-data.ts)
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
// scripts/ lives inside web-app/, and the repo root is two levels up.
const WEB_APP = join(__dirname, "..");
const REPO_ROOT = join(WEB_APP, "..");

const PUBLIC_DATA = join(WEB_APP, "public", "data");
const PUBLIC_NAMES = join(WEB_APP, "public", "names");

// The combined genealogy: Books 1 and 2 spliced into one tree at the shared 存学
// node, generations numbered to the official 字辈. Nodes carry Unicode names with
// the name-crop images as a fallback. name_images point at
// "books/book{1,2}/5_names/*.png", rewritten to "/names/book{1,2}/*.png" (the per-
// node book is inferred from that path, so both books' crops are served).
const DATASET = { jsonl: "data/tree_combined.jsonl" } as const;
const NAME_BOOKS = ["book1", "book2"] as const;

function copyJsonl(srcRel: string, destName: string): number {
  const srcPath = join(REPO_ROOT, srcRel);
  const destPath = join(PUBLIC_DATA, destName);

  const lines = readFileSync(srcPath, "utf8").split("\n").filter(Boolean);

  const rewritten = lines.map((line) => {
    const node = JSON.parse(line) as { name_images?: string[] };
    // Rewrite "books/bookN/5_names/1.png" -> "/names/bookN/1.png". The book is
    // taken from the source path so a combined tree's book1 + book2 crops both
    // resolve correctly.
    if (Array.isArray(node.name_images)) {
      node.name_images = node.name_images.map((p) => {
        const parts = p.split("/");
        const base = parts.pop() as string;
        const book = parts.find((s) => /^book\d+$/.test(s)) ?? "book1";
        return `/names/${book}/${base}`;
      });
    }
    return JSON.stringify(node);
  });

  writeFileSync(destPath, `${rewritten.join("\n")}\n`);
  return rewritten.length;
}

function copyNames(book: string): number {
  // v1 pipeline writes name crops to 5_names/ (Stage 5).
  const srcDir = join(REPO_ROOT, "books", book, "5_names");
  const destDir = join(PUBLIC_NAMES, book);
  if (!existsSync(srcDir)) {
    console.warn(`  (no 5_names dir for ${book}, skipping)`);
    return 0;
  }
  cpSync(srcDir, destDir, { recursive: true });
  return readdirSync(destDir).filter((f) => f.endsWith(".png")).length;
}

function main() {
  mkdirSync(PUBLIC_DATA, { recursive: true });
  mkdirSync(PUBLIC_NAMES, { recursive: true });

  console.log("Copying JSONL data -> public/data/");
  const n = copyJsonl(DATASET.jsonl, "tree.jsonl");
  console.log(`  ${DATASET.jsonl}  (${n} nodes)  [rewrote image paths]`);

  console.log("Copying name-crop images -> public/names/");
  for (const book of NAME_BOOKS) {
    const c = copyNames(book);
    console.log(`  ${book}: ${c} images`);
  }

  console.log("Done.");
}

main();
