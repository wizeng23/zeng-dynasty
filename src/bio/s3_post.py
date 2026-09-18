"""Bio stage 3 (post-QA): cut the final person-block crops from QA-approved boxes.

Runs AFTER the QA editor. It reads each QA-approved per-section JSONL
``books/{book}/bio/3_segment/{a}_{b}.jsonl`` (a full copy of the block state the editor
saved -- see ``scripts/qa/bio_s3_edit.py``), cuts one crop per block from the merged
section image, and writes:

- ``books/{book}/bio/3_segment/{a}_{b}_{c}_{d}.png`` -- one crop per person.
- ``books/{book}/bio/3_segment/blocks.jsonl`` -- the final combined index (every block of
  every section, one row each, in section/band/d order) that stage 4 consumes.

No tree-node association is done here: even after QA the number of bios in a section is
not guaranteed to equal the subgraph's node count, so a positional block->node mapping is
unreliable. Linking a bio to its tree node is left to a later, evidence-based step.

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


def post_book(book: str, sections: list[str] | None = None, books_dir: str = "books") -> int:
    """Cut the final crops from QA-approved boxes + write the combined final JSONL.

    No tree-node association: even after QA the bio count in a section is not guaranteed
    to equal the graph's node count for that subgraph (missing/extra bios), so a
    positional block->node mapping is not reliable. We just emit the crops + boxes; any
    bio<->node linking is left to a later, evidence-based step (e.g. children names).
    """
    seg_dir = _segment_dir(books_dir, book)
    merged_dir = os.path.join(books_dir, book, seg.BIO_DIR, seg.MERGED_DIR)

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
        a = get_image(os.path.join(merged_dir, f"{sec}.png"))
        H, W = a.shape
        ink = 1 - a
        for row in rows:
            # re-tighten to text (catches QA-added/moved boxes) and clamp to the image
            # bounds (a QA box may extend a few px past the edge -> otherwise a bad crop).
            l, t, r_, b = seg.tighten_box(ink, row["box"])
            l, t = max(0, l), max(0, t)
            r_, b = min(W, r_), min(H, b)
            row["box"] = [l, t, r_, b]
            save_image(a[t:b, l:r_], os.path.join(seg_dir, f"{row['id']}.png"))
            n_crops += 1
            combined.append(row)
        logger.info("%s: %d crops", sec, len(rows))

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
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    post_book(args.book, sections=args.sections, books_dir=args.books_dir)


if __name__ == "__main__":
    main()
