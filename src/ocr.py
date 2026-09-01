"""Stage 7 -- OCR: turn name-crop images into Unicode characters.

Each parsed node carries a name-image crop (``books/bookN/names/{id}.png``) rather
than a Unicode name until this stage runs. The characters are printed, single, and
on a clean grid, but some are *ancient* -- not in modern Chinese -- which is the
known hard case (a plain OCR may silently drop them).

Two design ideas from the project's original plan:

- **Grid-packing.** Instead of one API call per tiny crop, pack many crops into a
  labelled grid sheet and OCR the sheet in one call. We know each cell's crop id,
  so an *undetected* character is detectable (a missing cell), not a silent gap --
  the same "every character is accounted for" principle the parser uses.
- **Pluggable engines.** OCR quality moves fast; the engine is a strategy. Each
  engine implements :class:`OcrEngine` (``name`` + ``recognize(crops) -> {id: char}``)
  so we can bake several off against the ground truth (``scripts/eval_ocr.py``) and
  pick the best per book.

This module provides the engine interface, the grid-pack helper, and the engines
that need no extra local dependencies (an OpenAI vision engine). Heavier local
engines (PaddleOCR, EasyOCR) live behind optional imports so the module loads
without them.

CLI::

    python -m src.ocr --book book1 --engine openai --ids 1-40 --out preds.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crop loading + grid packing
# ---------------------------------------------------------------------------
@dataclass
class Crop:
    """One name-crop to recognize."""

    id: int
    image: Image.Image  # grayscale ("L") PIL image


def load_crops(book: str, ids: list[int], books_dir: str = "books") -> list[Crop]:
    """Load the name-crop PNGs for the given ids."""
    crops = []
    for i in ids:
        path = os.path.join(books_dir, book, "names", f"{i}.png")
        if not os.path.exists(path):
            logger.warning("missing crop %s", path)
            continue
        crops.append(Crop(id=i, image=Image.open(path).convert("L")))
    return crops


def split_characters(
    image: Image.Image, ncols: int | None = None, min_gap: int = 6
) -> list[Image.Image]:
    """Split a stacked-name crop into per-character images, top to bottom.

    A name of two or three characters is written vertically stacked in one crop.
    OCR engines here read one glyph at a time, so we cut the crop at the widest
    interior horizontal whitespace bands. Characters sit far apart vertically
    while a glyph's own internal gaps are small, giving a clean valley to cut on.

    Args:
        image: The name crop (any mode; read as grayscale).
        ncols: Expected character count. When given, we make exactly
            ``ncols - 1`` cuts at the widest interior gaps -- robust when the
            count is known from other data. When ``None``, we cut at every
            interior gap of at least ``min_gap`` rows (best-effort).
        min_gap: Minimum interior gap height (rows) to treat as a character
            boundary when ``ncols`` is not given.

    Returns:
        The per-character sub-images in reading (top-to-bottom) order. A crop
        with no detectable ink returns ``[image]`` unchanged.
    """
    a = np.asarray(image.convert("L"))
    ink_per_row = (a < 128).sum(axis=1)
    ink_rows = [r for r in range(len(ink_per_row)) if ink_per_row[r] > 0]
    if not ink_rows:
        return [image]
    top, bot = ink_rows[0], ink_rows[-1]
    a2 = a[top : bot + 1]
    ink2 = (a2 < 128).sum(axis=1)

    gap_rows = [r for r in range(len(ink2)) if ink2[r] == 0]
    runs: list[tuple[int, int]] = []
    if gap_rows:
        s = p = gap_rows[0]
        for r in gap_rows[1:]:
            if r != p + 1:
                runs.append((s, p))
                s = r
            p = r
        runs.append((s, p))

    if ncols is not None:
        widest = sorted(runs, key=lambda run: -(run[1] - run[0]))[: ncols - 1]
        cuts = sorted((run[0] + run[1]) // 2 for run in widest)
    else:
        cuts = sorted(
            (run[0] + run[1]) // 2 for run in runs if run[1] - run[0] + 1 >= min_gap
        )

    bounds = [0, *cuts, a2.shape[0]]
    return [Image.fromarray(a2[bounds[k] : bounds[k + 1]]) for k in range(len(bounds) - 1)]


def pack_grid(
    crops: list[Crop],
    cols: int = 10,
    cell: int = 96,
    label: bool = True,
) -> tuple[Image.Image, list[int]]:
    """Pack crops into one labelled grid sheet.

    Each crop is scaled to fit a ``cell``-sized square on a white background, laid
    out row-major, optionally with its **1-based grid position** printed in the
    corner (so the model can report "cell 7 = X" and we map position -> crop id).

    Returns the sheet image and a row-major list of crop ids per cell.
    """
    pad = 8
    inner = cell - 20  # leave room for the position label
    rows = (len(crops) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    sheet = Image.new("L", (W, H), color=255)
    draw = ImageDraw.Draw(sheet)
    order: list[int] = []
    for idx, crop in enumerate(crops):
        r, c = divmod(idx, cols)
        a = np.asarray(crop.image)
        h, w = a.shape
        s = min(inner / h, inner / w)
        resized = crop.image.resize((max(1, int(w * s)), max(1, int(h * s))))
        cx = pad + c * (cell + pad)
        cy = pad + r * (cell + pad)
        sheet.paste(
            resized,
            (cx + (cell - resized.width) // 2, cy + 16 + (inner - resized.height) // 2),
        )
        if label:
            draw.text((cx + 2, cy + 2), str(idx + 1), fill=0)
        order.append(crop.id)
    return sheet, order


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------
class OcrEngine(Protocol):
    name: str

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        """Return {crop_id: predicted_char} for the given crops."""
        ...


class OpenAIVisionEngine:
    """OCR via an OpenAI vision model, using grid-packed sheets.

    Sends one labelled grid per batch and asks for a per-cell character, so a
    single call covers many crops and undetected cells are visible in the reply.
    """

    def __init__(self, model: str = "gpt-4o", batch: int = 40):
        self.name = f"openai:{model}"
        self.model = model
        self.batch = batch

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        from openai import OpenAI

        client = OpenAI()
        out: dict[int, str] = {}
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            sheet, order = pack_grid(chunk)
            prompt = (
                "This is a grid of individual Chinese characters from a printed "
                "genealogy (some are rare/ancient forms). Each cell is numbered in "
                "its top-left corner. Return STRICT JSON mapping each cell number "
                "(as a string) to the single Unicode character in it, e.g. "
                '{"1":"点","2":"参"}. Exactly one character per cell. If a cell is '
                'truly unreadable use "".'
            )
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_to_data_url(sheet)},
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            for cell_str, char in data.items():
                try:
                    pos = int(cell_str) - 1
                except ValueError:
                    continue
                if 0 <= pos < len(order):
                    out[order[pos]] = char
            logger.info("  batch %d-%d: %d chars", start, start + len(chunk), len(data))
        return out


def get_engine(name: str) -> OcrEngine:
    """Resolve an engine by name. Heavy engines import lazily."""
    if name.startswith("openai"):
        model = name.split(":", 1)[1] if ":" in name else "gpt-4o"
        return OpenAIVisionEngine(model=model)
    if name == "paddle":
        from src.ocr_paddle import PaddleEngine  # optional dependency

        return PaddleEngine()
    if name in ("mistral", "google:vision", "google:docai"):
        from src import ocr_cloud  # optional dependencies (SDKs + auth)

        if name == "mistral":
            return ocr_cloud.MistralOCREngine()
        if name == "google:vision":
            return ocr_cloud.GoogleVisionEngine()
        return ocr_cloud.GoogleDocAIEngine()
    raise ValueError(f"unknown OCR engine: {name!r}")


def _parse_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def populate_names(
    book: str,
    books_dir: str = "books",
    data_dir: str = "data",
    low_conf: float = 0.90,
) -> dict[str, dict]:
    """OCR every parsed node's name crop and write a names sidecar for the book.

    Reads ``{data_dir}/{book}.jsonl`` for the node ids, OCRs each crop with
    PP-OCRv5's full pipeline (which reads multi-character stacked names in one
    pass), and writes ``{data_dir}/{book}_names.json``::

        {"1": {"name": "点", "confidence": 0.999, "low_conf": false}, ...}

    Every node gets an entry (``write all``); ``low_conf`` marks names whose
    weakest-character confidence is below ``low_conf`` so they can be reviewed.
    The sidecar is applied on top of the parse (see :func:`apply_names`), so a
    re-parse never loses the names -- re-run this after re-parsing.

    Returns the sidecar dict it wrote.
    """
    from src.ocr_paddle import PaddleEngine

    jsonl = os.path.join(data_dir, f"{book}.jsonl")
    ids = [json.loads(line)["id"] for line in open(jsonl) if line.strip()]
    engine = PaddleEngine()
    logger.info("Populating %d names for %s", len(ids), book)

    sidecar: dict[str, dict] = {}
    n_low = 0
    for k, node_id in enumerate(ids):
        path = os.path.join(books_dir, book, "names", f"{node_id}.png")
        if not os.path.exists(path):
            logger.warning("missing crop %s", path)
            continue
        name, conf = engine.recognize_name(Image.open(path).convert("L"))
        is_low = (not name) or conf < low_conf
        n_low += int(is_low)
        sidecar[str(node_id)] = {
            "name": name,
            "confidence": round(conf, 4),
            "low_conf": is_low,
        }
        if (k + 1) % 100 == 0:
            logger.info("  ...%d/%d", k + 1, len(ids))

    out_path = os.path.join(data_dir, f"{book}_names.json")
    json.dump(sidecar, open(out_path, "w"), ensure_ascii=False, indent=2)
    logger.info(
        "Wrote %d names -> %s (%d low-confidence < %.2f)",
        len(sidecar),
        out_path,
        n_low,
        low_conf,
    )
    return sidecar


def _provenance_of(notes: str) -> str:
    """The stable ``{graph}_{localindex}`` key: the first ' | '-segment of notes."""
    return notes.split(" | ")[0] if notes else ""


def _strip_ocr_tags(notes: str) -> str:
    """Drop any prior ``ocr_conf=``/``ocr_low_conf``/``ocr_override`` tags from notes.

    Keeps the leading provenance and any other human notes, so :func:`apply_names`
    is idempotent -- re-running never stacks duplicate ocr tags.
    """
    segs = [s for s in notes.split(" | ") if s]
    kept = [s for s in segs if not s.startswith(("ocr_conf=", "ocr_low_conf", "ocr_override"))]
    return " | ".join(kept)


def _resolve_name(prov: str, ocr_name: str, overrides: dict[str, str]) -> tuple[str, bool]:
    """Reassemble a node's name from per-character overrides over the OCR name.

    Overrides are keyed ``{provenance}#{charIndex}`` (0-based) by the filmstrip
    review tool. For each character position we take the override if present, else
    the OCR character. The name length is the max of the OCR length and the
    highest overridden index, so an override can also *extend* a name (e.g. when
    OCR under-read a stacked glyph). Returns ``(name, had_override)``.
    """
    max_idx = len(ocr_name) - 1
    char_overrides: dict[int, str] = {}
    for key, val in overrides.items():
        if "#" not in key:
            continue
        p, _, ci = key.rpartition("#")
        if p != prov or not ci.isdigit():
            continue
        i = int(ci)
        char_overrides[i] = val
        max_idx = max(max_idx, i)
    if not char_overrides:
        return ocr_name, False
    chars = []
    for i in range(max_idx + 1):
        if i in char_overrides:
            chars.append(char_overrides[i])
        elif i < len(ocr_name):
            chars.append(ocr_name[i])
    return "".join(chars), True


def apply_names(book: str, data_dir: str = "data") -> int:
    """Merge OCR names + manual per-character overrides into ``{book}.jsonl``.

    Precedence: manual overrides in ``{book}_overrides.json`` (the human
    ground-truth layer written by ``scripts/qa/ocr.py``, keyed
    ``{provenance}#{charIndex}``) win, character by character, over the OCR reading
    in ``{book}_names.json``. Sets each node's ``name`` and rewrites the ``ocr_*``
    tags in ``notes`` (idempotent): ``ocr_conf=<score>`` always, ``ocr_low_conf``
    when confidence is low, ``ocr_override`` when any character came from the
    override layer. A node with neither an override nor an OCR name keeps
    ``name=""`` so the website falls back to its crop image. Returns the count of
    non-empty names applied.
    """
    names_path = os.path.join(data_dir, f"{book}_names.json")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"no names sidecar: {names_path}; run populate_names")
    sidecar = json.load(open(names_path))
    ov_path = os.path.join(data_dir, f"{book}_overrides.json")
    overrides = json.load(open(ov_path)) if os.path.exists(ov_path) else {}

    jsonl = os.path.join(data_dir, f"{book}.jsonl")
    lines = [json.loads(line) for line in open(jsonl) if line.strip()]
    applied = 0
    n_override_nodes = 0
    for node in lines:
        prov = _provenance_of(node.get("notes", ""))
        entry = sidecar.get(str(node["id"]), {})
        ocr_name = entry.get("name", "")
        name, had_override = _resolve_name(prov, ocr_name, overrides)
        node["name"] = name
        n_override_nodes += int(had_override)

        base = _strip_ocr_tags(node.get("notes", ""))
        tags = []
        if entry:
            tags.append(f"ocr_conf={entry['confidence']}")
            if entry.get("low_conf"):
                tags.append("ocr_low_conf")
        if had_override:
            tags.append("ocr_override")
        node["notes"] = " | ".join([base, *tags]) if base else " | ".join(tags)
        if name:
            applied += 1
    with open(jsonl, "w") as f:
        for node in lines:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    logger.info(
        "Applied %d names -> %s (%d nodes with overrides)", applied, jsonl, n_override_nodes
    )
    return applied


def apply_names_by_crop(
    target_jsonl: str, names_book: str, data_dir: str = "data"
) -> int:
    """Apply a book's names sidecar to a jsonl keyed by crop-id-in-``name_images``.

    Some derived files (e.g. ``book1_stitched.jsonl``) renumber node ``id`` but
    keep each node's original crop path in ``name_images`` (``.../names/13.png``).
    This applies ``{names_book}_names.json`` by reading the crop id from the first
    ``name_images`` entry, so the stitched dataset shows the same OCR'd names.
    Returns the count of names applied.
    """
    sidecar = json.load(open(os.path.join(data_dir, f"{names_book}_names.json")))
    lines = [json.loads(line) for line in open(target_jsonl) if line.strip()]
    applied = 0
    for node in lines:
        imgs = node.get("name_images") or []
        if not imgs:
            continue
        crop_id = os.path.splitext(os.path.basename(imgs[0]))[0]
        entry = sidecar.get(crop_id)
        if not entry:
            continue
        node["name"] = entry["name"]
        prov = node.get("notes", "")
        tags = [f"ocr_conf={entry['confidence']}"]
        if entry["low_conf"]:
            tags.append("ocr_low_conf")
        node["notes"] = " | ".join([prov, *tags]) if prov else " | ".join(tags)
        if entry["name"]:
            applied += 1
    with open(target_jsonl, "w") as f:
        for node in lines:
            f.write(json.dumps(node, ensure_ascii=False) + "\n")
    logger.info("Applied %d names -> %s", applied, target_jsonl)
    return applied


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="OCR name crops to Unicode.")
    ap.add_argument("--book", default="book1")
    ap.add_argument("--engine", default="openai:gpt-4o")
    ap.add_argument("--ids", default="1-40", help="e.g. 1-40 or 1,2,5-9")
    ap.add_argument("--out", default="scratchpad/ocr_preds.json")
    ap.add_argument(
        "--populate",
        action="store_true",
        help="OCR every node's name with PP-OCRv5 and write {book}_names.json, "
        "then apply the names into {book}.jsonl (ignores --engine/--ids/--out).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.populate:
        populate_names(args.book)
        apply_names(args.book)
        return

    crops = load_crops(args.book, _parse_ids(args.ids))
    engine = get_engine(args.engine)
    logger.info("Running %s on %d crops", engine.name, len(crops))
    preds = engine.recognize(crops)
    json.dump(
        {"engine": engine.name, "labels": {str(k): v for k, v in preds.items()}},
        open(args.out, "w"),
        ensure_ascii=False,
        indent=2,
    )
    logger.info("Wrote %d predictions -> %s", len(preds), args.out)


if __name__ == "__main__":
    main()
