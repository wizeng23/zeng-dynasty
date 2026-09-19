"""Bio stage 4 (ext): fold two more per-crop readers into the 4_ocr records.

Adds **Gemini 3.8 Flash** and **Mistral OCR** as extra ensemble readers alongside the
existing Paddle + Claude-vision reads (see :mod:`src.bio.s4_ocr`). Both run per tight
person-crop (``3_segment/{id}.png``) -- the SAME granularity as the s4 ensemble -- and
their full output is folded losslessly into each block's existing
``books/{book}/bio/4_ocr/{stem}.jsonl`` record under ``raw.gemini`` / ``raw.mistral``.

Readers:

* **Gemini** (``gemini-3.8-flash``): one ``generate_content`` per crop with the shared
  s4 transcription prompt (RTL columns, one line per column). Stored flat as
  ``raw.gemini.text`` (Gemini vision returns no char boxes). Bake-off recall ~0.996.
* **Mistral** (``mistral-ocr-latest``): one ``ocr.process`` per crop with
  ``document_annotation_format`` -- the same call the shipped page-level Stage 8
  (:mod:`scripts.run_bio_ocr`) makes. Saves ALL result forms:
  ``raw.mistral.markdown`` (flat, source of truth), ``raw.mistral.persons`` (structured
  split), and ``raw.mistral.pages`` (the full response dump, incl. any per-block
  bounding-box / layout data Mistral returns). Bake-off recall ~0.976, ~$0.004/crop.

Idempotent + resumable: a block that already has a non-empty read for a given reader is
SKIPPED (never re-billed). Transient 429/5xx are retried with exponential backoff; a
block that still fails is left without that reader and the run continues.

Run:
    PYTHONPATH=. python -m src.bio.s4_ocr_ext --book book3 --readers gemini mistral --limit 50
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BIO_DIR = "bio"
SEGMENT_DIR = "3_segment"
OUT_DIR = "4_ocr"

GEMINI_MODEL = "gemini-3.8-flash"
MISTRAL_MODEL = "mistral-ocr-latest"

MAX_RETRIES = 4
BACKOFF_BASE = 2.0
REQUEST_TIMEOUT = 90.0   # hard per-call ceiling so a hung connection can't stall the run
WORKERS = 8              # concurrent (block, reader) API calls

# Shared transcription prompt (inline-image variant of s4_vision_prompt.txt: no "Read the
# image with the Read tool" tail, since these readers receive the image inline).
PROMPT = (
    "You are transcribing one person's entry from a Chinese genealogy book (族谱).\n"
    "The image is a single person's biographical block, printed in traditional vertical "
    "columns read RIGHT-TO-LEFT (rightmost column first).\n\n"
    "Transcribe ALL visible Chinese characters. Output rules:\n"
    "- One line per vertical column, in right-to-left order (rightmost column = first line).\n"
    "- Within each column, read top to bottom.\n"
    "- Line 1 MUST be the small header caption in the TOP-RIGHT (format 子之X or 子次X or "
    "子三X…, one character X = the father's name), printed HORIZONTALLY. If absent, write ?.\n"
    "- Line 2 MUST be the person's own name (the larger/bolder character(s) in the rightmost "
    "vertical column). If unsure, write ?.\n"
    "- Lines 3+ : biography prose columns, one column per line, right-to-left.\n"
    "- Output ONLY the transcription lines. No translation, commentary, or pinyin. "
    "Use ? for any character you genuinely cannot read.\n"
)


# --- structured schema for Mistral document_annotation (same shape as Stage 8) ------

class _Person(BaseModel):
    name: str = Field(description="the person's given name characters (empty if none)")
    pai: str = Field(description="第..派 generation marker if present, else empty")
    shi: str = Field(description="第..世 generation marker if present, else empty")
    full_text: str = Field(
        description="all Chinese text for this person, read top-to-bottom within a "
        "column and right-to-left across columns")


class _Block(BaseModel):
    persons: list[_Person]


# --- crop IO -----------------------------------------------------------------------

def _crop_png_b64(book: str, bid: str, books_dir: str) -> str:
    path = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR, f"{bid}.png")
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _is_transient(msg: str) -> bool:
    return any(s in msg for s in ("429", "500", "502", "503", "504",
                                  "rate", "timeout", "Timeout", "UNAVAILABLE",
                                  "RESOURCE_EXHAUSTED",
                                  # connection-level blips (not HTTP status)
                                  "Connection reset", "reset by peer", "ConnectionError",
                                  "RemoteProtocol", "ConnectTimeout", "ReadError",
                                  "peer closed", "EOF"))


# --- Gemini reader -----------------------------------------------------------------

def read_gemini(book: str, bid: str, books_dir: str, client) -> dict:
    """Transcribe one crop with Gemini; returns ``{"text": ...}`` (lossless)."""
    from google.genai import types

    path = os.path.join(books_dir, book, BIO_DIR, SEGMENT_DIR, f"{bid}.png")
    with open(path, "rb") as fh:
        data = fh.read()
    part = types.Part.from_bytes(data=data, mime_type="image/png")
    cfg = types.GenerateContentConfig(http_options=types.HttpOptions(
        timeout=int(REQUEST_TIMEOUT * 1000)))  # google-genai timeout is in ms
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = client.models.generate_content(model=GEMINI_MODEL,
                                                contents=[PROMPT, part], config=cfg)
            return {"text": r.text or "", "model": GEMINI_MODEL,
                    "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < MAX_RETRIES and _is_transient(str(e)):
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise
    raise last


# --- Mistral reader ----------------------------------------------------------------

def read_mistral(book: str, bid: str, books_dir: str, client, resp_format) -> dict:
    """OCR one crop with Mistral; returns markdown + persons + full page dump (lossless)."""
    b64 = _crop_png_b64(book, bid, books_dir)
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.ocr.process(
                model=MISTRAL_MODEL,
                document={"type": "image_url",
                          "image_url": f"data:image/png;base64,{b64}"},
                document_annotation_format=resp_format,
                include_image_base64=False,
            )
            d = resp.model_dump()
            md = "\n".join(p.get("markdown", "") for p in d.get("pages", []))
            da = d.get("document_annotation")
            persons = None
            if da:
                try:
                    persons = (json.loads(da) if isinstance(da, str) else da).get("persons")
                except Exception:
                    persons = None
            return {
                "markdown": md,
                "persons": persons,
                "pages": d.get("pages", []),        # full dump incl. any bbox/layout
                "model": MISTRAL_MODEL,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < MAX_RETRIES and _is_transient(str(e)):
                time.sleep(BACKOFF_BASE ** (attempt + 1))
                continue
            raise
    raise last


# --- fold into 4_ocr records -------------------------------------------------------

def _has_read(rec: dict, reader: str) -> bool:
    raw = (rec.get("raw") or {}).get(reader)
    if not raw:
        return False
    if reader == "gemini":
        return bool((raw.get("text") or "").strip())
    if reader == "mistral":
        return bool((raw.get("markdown") or "").strip())
    return False


def run(book: str, readers: list[str], books_dir: str = "books",
        sections: list[str] | None = None, limit: int | None = None,
        workers: int = WORKERS) -> dict:
    """Fold the requested readers into 4_ocr records, in block-index order."""
    out_base = os.path.join(books_dir, book, BIO_DIR, OUT_DIR)
    # load every 4_ocr section file -> {stem: [records...]} preserving order
    by_stem: dict[str, list[dict]] = {}
    for fn in sorted(os.listdir(out_base)):
        if not fn.endswith(".jsonl"):
            continue
        stem = fn[:-6]
        if sections and stem not in sections:
            continue
        with open(os.path.join(out_base, fn)) as fh:
            by_stem[stem] = [json.loads(l) for l in fh if l.strip()]

    # flat ordered list of (stem, record) for limit slicing
    flat = [(stem, rec) for stem in sorted(by_stem, key=lambda s: int(s.split("_")[0]))
            for rec in by_stem[stem]]
    if limit:
        flat = flat[:limit]
    logger.info("s4_ocr_ext: %s readers=%s over %d blocks", book, readers, len(flat))

    # clients (lazy)
    gclient = mclient = resp_format = None
    if "gemini" in readers:
        from google import genai
        gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if "mistral" in readers:
        from mistralai import Mistral
        from mistralai.extra import response_format_from_pydantic_model
        mclient = Mistral(api_key=os.environ["MISTRAL_API_KEY"],
                          timeout_ms=int(REQUEST_TIMEOUT * 1000))
        resp_format = response_format_from_pydantic_model(_Block)

    # One lock guards ALL section-file writes AND the shared counters/records, so the
    # worker threads can't corrupt a file or race a counter. A read that mutates its own
    # `rec["raw"]` in place is fine (each rec is touched by <=2 tasks, different keys),
    # but the flush + counter update happen under the lock.
    io_lock = threading.Lock()

    def flush(stem: str) -> None:
        with open(os.path.join(out_base, f"{stem}.jsonl"), "w") as fh:
            for rec in by_stem[stem]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = {r: {"done": 0, "skip": 0, "fail": 0} for r in readers}
    touched_stems: set[str] = set()

    # Build the task list: one (stem, rec, reader) per reader still missing on each block.
    tasks = []
    for stem, rec in flat:
        rec.setdefault("raw", {})
        for reader in readers:
            if _has_read(rec, reader):
                counts[reader]["skip"] += 1
            else:
                tasks.append((stem, rec, reader))
    total = len(tasks)
    logger.info("  %d reader-calls to run (workers=%d)", total, workers)

    def work(task):
        stem, rec, reader = task
        if reader == "gemini":
            result = read_gemini(book, rec["id"], books_dir, gclient)
        else:
            result = read_mistral(book, rec["id"], books_dir, mclient, resp_format)
        # Mutate + persist under the lock so file writes never interleave. Every paid call
        # is flushed the instant it returns, so none is lost to a later crash (idempotent).
        with io_lock:
            rec["raw"][reader] = result
            counts[reader]["done"] += 1
            touched_stems.add(stem)
            flush(stem)
        return task

    done_n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            stem, rec, reader = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                with io_lock:
                    counts[reader]["fail"] += 1
                logger.error("  %s %s FAILED: %s", reader, rec["id"], str(e)[:140])
            done_n += 1
            if done_n % 20 == 0 or done_n == total:
                logger.info("  ...%d/%d %s", done_n, total, counts)

    logger.info("DONE %s: %s | wrote %d section files", book, counts, len(touched_stems))
    return {"book": book, "counts": counts, "sections": sorted(touched_stems)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True)
    ap.add_argument("--readers", nargs="+", default=["gemini", "mistral"],
                    choices=["gemini", "mistral"])
    ap.add_argument("--books-dir", default="books")
    ap.add_argument("--sections", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N blocks (smoke test)")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="concurrent (block, reader) API calls")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s: %(message)s")
    run(args.book, args.readers, books_dir=args.books_dir,
        sections=args.sections, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
