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

// The single showcase dataset: Book 1, fully parsed -> OCR'd -> cross-graph
// stitched into one connected lineage (src/s7_stitch.py). It carries real Unicode
// names (Stage 6 OCR) AND the v1 name-crop images as a fallback. Its name_images
// point at "books/book1/5_names/*.png", rewritten to "/names/book1/*.png".
const BOOKS = [
  { jsonl: "data/book1_stitched.jsonl", book: "book1" },
] as const;

function copyJsonl(srcRel: string, book: string | null): number {
  const srcPath = join(REPO_ROOT, srcRel);
  const fileName = srcRel.split("/").pop() as string;
  const destPath = join(PUBLIC_DATA, fileName);

  const lines = readFileSync(srcPath, "utf8").split("\n").filter(Boolean);

  const rewritten = lines.map((line) => {
    const node = JSON.parse(line) as { name_images?: string[] };
    // Rewrite "books/book1/names/1.png" -> "/names/book1/1.png" for parsed books.
    if (book && Array.isArray(node.name_images)) {
      node.name_images = node.name_images.map((p) => {
        const base = p.split("/").pop() as string;
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
  for (const { jsonl, book } of BOOKS) {
    const n = copyJsonl(jsonl, book);
    console.log(`  ${jsonl}  (${n} nodes)${book ? "  [rewrote image paths]" : "  [golden]"}`);
  }

  console.log("Copying name-crop images -> public/names/");
  for (const book of ["book1"]) {
    const n = copyNames(book);
    console.log(`  ${book}: ${n} images`);
  }

  console.log("Done.");
}

main();
