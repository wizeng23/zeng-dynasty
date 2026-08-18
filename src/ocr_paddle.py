"""PaddleOCR (PP-OCRv5) engine -- optional local dependency.

PP-OCRv5 is Chinese-specialized and explicitly trained on rare/ancient characters,
which makes it a strong candidate for this genealogy. It runs fully local (no API),
so it is reproducible and free, at the cost of a heavy install (paddlepaddle).

Kept in a separate module so ``src.ocr`` imports without paddle present.
"""

from __future__ import annotations

import logging

import numpy as np

from src.ocr import Crop

logger = logging.getLogger(__name__)


class PaddleEngine:
    """OCR each crop individually with PP-OCRv5 recognition.

    The crops are already single tightly-cropped characters, so we run the
    recognizer directly (text-line recognition on a one-character image) rather
    than the full detect+recognize pipeline.
    """

    def __init__(self, lang: str = "ch"):
        self.name = "paddle:PP-OCRv5"
        from paddleocr import PaddleOCR

        # Recognition-only: detection/orientation add nothing for pre-cropped
        # single glyphs and can mis-split them.
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        out: dict[int, str] = {}
        for crop in crops:
            # PP-OCRv5 recognizes darker ink on light ground; feed RGB.
            arr = np.asarray(crop.image.convert("RGB"))
            result = self._ocr.predict(arr)
            text = _first_text(result)
            if text:
                out[crop.id] = text[0]  # first (should be only) character
            else:
                logger.info("  paddle: no text for crop %d", crop.id)
        return out


def _first_text(result) -> str:
    """Pull the recognized string out of a PaddleOCR 3.x predict() result."""
    if not result:
        return ""
    res = result[0]
    # PaddleOCR 3.x returns objects exposing a dict with "rec_texts".
    data = getattr(res, "json", None)
    if isinstance(data, dict):
        texts = data.get("res", {}).get("rec_texts") or data.get("rec_texts")
        if texts:
            return "".join(texts)
    if isinstance(res, dict):
        texts = res.get("rec_texts")
        if texts:
            return "".join(texts)
    return ""
