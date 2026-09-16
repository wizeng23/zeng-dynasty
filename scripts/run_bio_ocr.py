"""Stage 8 (bio track): OCR every biography page with Mistral OCR.

Books 3 & 4 interleave biography pages (dense vertical name+prose columns) with
tree pages. Stage 2 tags them (``bio_pages`` in ``2_classify/page_types.json``);
this reads each bio page's PNG, trims the printed border with the same
:func:`trim_borders` the tree path uses, and OCRs it with Mistral
``mistral-ocr-latest``. Mistral reads the vertical top-to-bottom / right-to-left
columns natively (chosen in the 2026-09-15 OCR bake-off: ~0.98 char recall,
$0.004/page, ~1s -- see docs).

A single ``ocr.process`` call with ``document_annotation_format`` returns BOTH the
plain ``markdown`` transcription (the source of truth) AND a best-effort structured
``persons[]`` split -- billed as one page. The structured split mis-segments
cross-page continuations (a person whose entry spills onto the next page), so the
markdown is authoritative; ``persons[]`` is kept as a convenience for later
biography<->tree linking.

Output (permanent, committed):
  books/{book}/8_bio_ocr/{page}.json   per-page: markdown + persons + meta
  data/{book}_bio_ocr.json             rollup {page: markdown} written at the end

Idempotent: a page whose per-page JSON already exists (non-empty markdown) is
SKIPPED, so a re-run only fills gaps -- never re-bills, and resumes cleanly after a
throttle/crash. Retries 429/5xx with exponential backoff; a page that still fails
is logged and the run continues.

CLI:
    PYTHONPATH=. python -m scripts.run_bio_ocr --book book3
    PYTHONPATH=. python -m scripts.run_bio_ocr --book book3 --pages 2,3,7   # subset
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
from datetime import datetime, timezone

from PIL import Image
from pydantic import BaseModel, Field

from src.imaging import get_image
from src.s2_classify_pages import load_bio_pages
from src.s3_segment import trim_borders

logger = logging.getLogger(__name__)

MODEL = "mistral-ocr-latest"
OUT_SUBDIR = "8_bio_ocr"
PRICE_PER_PAGE = 4.00 / 1000  # $4 / 1000 pages (mistral-ocr-latest, Sep 2026)

# Throttle: Mistral OCR is ~1s/page; pace conservatively to avoid QPS limits.
DELAY_BETWEEN = 0.4        # seconds between successful calls
CHUNK = 25                 # pages per progress-flush chunk
MAX_RETRIES = 4            # on 429/5xx
BACKOFF_BASE = 2.0         # 2, 4, 8, 16s


class _Person(BaseModel):
    name: str = Field(description="the person's given name characters (empty if none)")
    pai: str = Field(description="第..派 generation marker if present, else empty")
    shi: str = Field(description="第..世 generation marker if present, else empty")
    full_text: str = Field(
        description="all Chinese text for this person, read top-to-bottom within a "
        "column and right-to-left across columns")


class _Page(BaseModel):
    persons: list[_Person]


def _page_png_b64(book: str, page: int, books_dir: str) -> tuple[str, int]:
    """Trim borders off the page PNG and return (base64 png, byte size)."""
    path = os.path.join(books_dir, book, "1_pages", f"{page}.png")
    a = trim_borders(get_image(path))
    buf = io.BytesIO()
    Image.fromarray((a * 255).astype("uint8")).convert("L").save(buf, format="PNG")
    raw = buf.getvalue()
    return base64.b64encode(raw).decode(), len(raw)


def _ocr_one(client, resp_format, b64: str):
    """One Mistral OCR call with retry/backoff. Returns the raw response dump."""
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.ocr.process(
                model=MODEL,
                document={"type": "image_url",
                          "image_url": f"data:image/png;base64,{b64}"},
                document_annotation_format=resp_format,
                include_image_base64=False,
            )
            return resp.model_dump()
        except Exception as e:  # noqa: BLE001 -- retry transient, re-raise terminal
            msg = str(e)
            transient = any(s in msg for s in ("429", "500", "502", "503", "504",
                                               "rate", "timeout", "Timeout"))
            last = e
            if attempt < MAX_RETRIES and transient:
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning("  transient error (attempt %d): %s; retry in %.0fs",
                               attempt + 1, msg[:120], wait)
                time.sleep(wait)
                continue
            raise
    raise last  # pragma: no cover


def run(book: str, books_dir: str = "books", data_dir: str = "data",
        pages: list[int] | None = None) -> dict:
    from mistralai import Mistral
    from mistralai.extra import response_format_from_pydantic_model

    out_dir = os.path.join(books_dir, book, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    if pages is None:
        bio = sorted(load_bio_pages(book, books_dir=books_dir))
    else:
        bio = sorted(pages)
    logger.info("Stage 8 bio OCR: %s, %d bio pages -> %s", book, len(bio), out_dir)

    # follows_graph per bio page (nearest preceding tree page), for later linking.
    pt_path = os.path.join(books_dir, book, "2_classify", "page_types.json")
    follows = {}
    if os.path.exists(pt_path):
        pt = json.load(open(pt_path))
        for k, v in pt.get("pages", {}).items():
            if v.get("type") == "bio":
                follows[int(k)] = v.get("follows_graph")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    resp_format = response_format_from_pydantic_model(_Page)

    done = skipped = failed = 0
    failed_pages: list[int] = []
    t0 = time.time()
    for i, page in enumerate(bio):
        pj = os.path.join(out_dir, f"{page}.json")
        if os.path.exists(pj):
            try:
                existing = json.load(open(pj))
                if existing.get("markdown"):
                    skipped += 1
                    continue
            except Exception:
                pass  # corrupt/empty -> re-OCR

        try:
            b64, nbytes = _page_png_b64(book, page, books_dir)
            d = _ocr_one(client, resp_format, b64)
            md = "\n".join(p.get("markdown", "") for p in d.get("pages", []))
            da = d.get("document_annotation")
            persons = None
            if da:
                try:
                    persons = (json.loads(da) if isinstance(da, str) else da).get("persons")
                except Exception:
                    persons = None
            rec = {
                "book": book, "page": page,
                "follows_graph": follows.get(page),
                "markdown": md,
                "persons": persons,
                "chars": len(md),
                "mistral_model": MODEL,
                "doc_size_bytes": nbytes,
                "ocr_ts": datetime.now(timezone.utc).isoformat(),
            }
            json.dump(rec, open(pj, "w"), ensure_ascii=False, indent=2)
            done += 1
        except Exception as e:  # noqa: BLE001
            logger.error("  page %d FAILED terminally: %s", page, str(e)[:160])
            failed += 1
            failed_pages.append(page)
            time.sleep(DELAY_BETWEEN)
            continue

        time.sleep(DELAY_BETWEEN)
        if (i + 1) % CHUNK == 0 or (i + 1) == len(bio):
            cost = (done) * PRICE_PER_PAGE
            logger.info("  ...%d/%d (done=%d skip=%d fail=%d) ~$%.3f, %.0fs",
                        i + 1, len(bio), done, skipped, failed, cost, time.time() - t0)

    # Rollup {page: markdown} from every per-page file present.
    rollup = {}
    for page in bio:
        pj = os.path.join(out_dir, f"{page}.json")
        if os.path.exists(pj):
            try:
                rollup[str(page)] = json.load(open(pj)).get("markdown", "")
            except Exception:
                pass
    rollup_path = os.path.join(data_dir, f"{book}_bio_ocr.json")
    json.dump(rollup, open(rollup_path, "w"), ensure_ascii=False, indent=2)

    logger.info("DONE: done=%d skipped=%d failed=%d | rollup=%d pages -> %s",
                done, skipped, failed, len(rollup), rollup_path)
    if failed_pages:
        logger.warning("FAILED pages (re-run to retry, idempotent): %s", failed_pages)
    return {"done": done, "skipped": skipped, "failed": failed,
            "failed_pages": failed_pages, "rollup": rollup_path,
            "cost_usd": round(done * PRICE_PER_PAGE, 4)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pages", default=None,
                    help="comma-separated subset, e.g. 2,3,7 (default: all bio pages)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s")
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    run(args.book, books_dir=args.books_dir, data_dir=args.data_dir, pages=pages)


if __name__ == "__main__":
    main()
