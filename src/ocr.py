"""Stage 3.6 -- OCR: turn name-crop images into Unicode characters.

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


def pack_grid(
    crops: list[Crop],
    cols: int = 10,
    cell: int = 96,
    label: bool = True,
) -> tuple[Image.Image, list[list[int]]]:
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="OCR name crops to Unicode.")
    ap.add_argument("--book", default="book1")
    ap.add_argument("--engine", default="openai:gpt-4o")
    ap.add_argument("--ids", default="1-40", help="e.g. 1-40 or 1,2,5-9")
    ap.add_argument("--out", default="scratchpad/ocr_preds.json")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

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
