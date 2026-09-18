"""Bio stage 3 (post-QA): cut the final person-block crops from QA-approved boxes.

Runs AFTER the QA editor. It reads each QA-approved per-section JSONL
``books/{book}/bio/3_segment/{a}_{b}.jsonl`` (a full copy of the block state the editor
saved -- see ``scripts/qa/bio_s3_edit.py``), cuts one crop per block from the merged
section image, and writes:

- ``books/{book}/bio/3_segment/{a}_{b}_{c}_{d}.png`` -- one crop per person.
- ``books/{book}/bio/3_segment/blocks.jsonl`` -- the final combined index (every block of
  every section, one row each, in section/band/d order) that stage 4 consumes.

Provenance ``{a}_{b}_{c}_{d}``: ``a_b`` = section pages, ``c`` = band index (0 = top = gen
2 ... 4 = bottom = gen 6), ``d`` = index in row (0 = rightmost/eldest, increasing left).

Only sections whose QA JSONL exists are processed; edit a section in the QA tool first.

Run:
    PYTHONPATH=. python -m src.bio.s3_post --book book3
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from src.imaging import get_image, save_image
from src.bio import s3_segment as seg

logger = logging.getLogger(__name__)


def _segment_dir(books_dir: str, book: str) -> str:
    return os.path.join(books_dir, book, seg.BIO_DIR, seg.SEGMENT_DIR)


def _read_section_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def post_book(book: str, sections: list[str] | None = None, books_dir: str = "books",
              data_dir: str = "data") -> int:
    """Cut crops + assign a tree node to each bio + write the combined final JSONL.

    For a section whose count gate passes, block d maps 1-1 to the d-th tree node of its
    generation in DFS eldest-first order (the alignment is only valid once QA has found
    every block, which is why node assignment lives here, not in s3_segment).
    """
    seg_dir = _segment_dir(books_dir, book)
    merged_dir = os.path.join(books_dir, book, seg.BIO_DIR, seg.MERGED_DIR)
    nodes_by_stem_gen = seg.tree_nodes_by_stem_gen(
        os.path.join(data_dir, f"{book}_stitched.jsonl"))

    approved = sorted(
        (f[:-6] for f in os.listdir(seg_dir) if f.endswith(".jsonl") and f != "blocks.jsonl"),
        key=lambda s: int(s.split("_")[0]),
    )
    todo = sections or approved
    combined: list[dict] = []
    n_crops = 0
    for sec in todo:
        rows = _read_section_jsonl(os.path.join(seg_dir, f"{sec}.jsonl"))
        rows.sort(key=lambda r: (r["band"], int(r["id"].rsplit("_", 1)[1])))
        stem = rows[0]["stem"] if rows else ""
        gate = all(r.get("gate_passed") for r in rows)
        per_gen = nodes_by_stem_gen.get(stem, {}) if gate else {}

        a = get_image(os.path.join(merged_dir, f"{sec}.png"))
        # d-index within each band -> the d-th DFS node of that generation
        band_d: dict[int, int] = {}
        for row in rows:
            l, t, r_, b = row["box"]
            save_image(a[t:b, l:r_], os.path.join(seg_dir, f"{row['id']}.png"))
            n_crops += 1
            d = band_d.get(row["band"], 0)
            band_d[row["band"]] = d + 1
            gen_nodes = per_gen.get(row["generation"], [])
            node = gen_nodes[d] if gate and d < len(gen_nodes) else None
            row["node_id"] = node["id"] if node else None
            row["node_name"] = node.get("name") if node else None
            combined.append(row)
        logger.info("%s: %d crops, node-mapped=%s", sec, len(rows), gate)

    with open(os.path.join(seg_dir, "blocks.jsonl"), "w") as fh:
        for row in combined:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %d crops across %d sections + blocks.jsonl (%d rows)",
                n_crops, len(todo), len(combined))
    return n_crops


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", required=True)
    parser.add_argument("--sections", nargs="+", default=None)
    parser.add_argument("--books-dir", default="books")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    post_book(args.book, sections=args.sections, books_dir=args.books_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
