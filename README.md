# Zeng Family Tree

Digitizing a 4-book Chinese family-tree (族谱 / zupu) from PDF scans into a
structured tree, and rendering it as a website.

## Status

- **Book 1:** parsed; verified golden data in `data/book1_golden.jsonl`. OCR pending.
- **Book 2:** page cropping done; tree parsing in progress.
- **Books 3–4:** not started.

Two milestones: a static website once all 4 books are parsed, then a dynamic
website allowing updates. See [`docs/milestones.md`](docs/milestones.md).

## Layout

| Path | What |
|------|------|
| `src/` | Parsing pipeline (Python). `model.py` is the data schema. |
| `data/` | Parsed output + Google Sheet export. |
| `books/` | Per-book scans and intermediate artifacts. |
| `web/` | d3.js tree viewer. |
| `docs/` | Milestones, progress, pipeline docs, design specs. |
| `old/` | The pre-restructure repo, archived. |

## How it works

Scans → structured tree in four stages (spreads→pages→graph images→tree,
then OCR). See [`docs/pipeline.md`](docs/pipeline.md).

## Viewing the tree

Serve the repo root and open `web/index.html` (it fetches
`data/book1_golden.jsonl`):

```
python3 -m http.server
# then open http://localhost:8000/web/index.html
```
