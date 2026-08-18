/**
 * Snapshot the parsed tree data + name-crop images into web-app/public so the
 * static site can fetch them at runtime.
 *
 * What it does:
 *   1. Copies the 3 JSONL files into public/data/.
 *   2. For the two parsed (non-golden) books, rewrites each node's
 *      `name_images` paths from the repo path ("books/book1/names/1.png")
 *      to the public URL the browser can load ("/names/book1/1.png").
 *      Golden data is copied verbatim — it has real Unicode names, no images.
 *   3. Copies books/book{1,2}/names/ -> public/names/book{1,2}/.
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

// Each source book: its JSONL and (for parsed books) its name-image folder.
const BOOKS = [
  { jsonl: "data/book1_golden.jsonl", book: null }, // golden: real names, no image rewrite
  { jsonl: "data/book1.jsonl", book: "book1" },
  { jsonl: "data/book2.jsonl", book: "book2" },
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
  const srcDir = join(REPO_ROOT, "books", book, "names");
  const destDir = join(PUBLIC_NAMES, book);
  if (!existsSync(srcDir)) {
    console.warn(`  (no names dir for ${book}, skipping)`);
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
  for (const book of ["book1", "book2"]) {
    const n = copyNames(book);
    console.log(`  ${book}: ${n} images`);
  }

  console.log("Done.");
}

main();
