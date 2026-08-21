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
        # single glyphs and can mis-split them. Used by the bake-off harness.
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        # The full detect+recognize pipeline is used for multi-character stacked
        # names: it locates and reads each glyph in the crop and returns them in
        # reading order, so we do not need to pre-split the crop ourselves.
        self._full = None
        self._lang = lang

    def _full_ocr(self):
        if self._full is None:
            from paddleocr import PaddleOCR

            self._full = PaddleOCR(lang=self._lang)
        return self._full

    def recognize_name(self, image) -> tuple[str, float]:
        """Read a whole name crop (1+ stacked chars) -> (name, min_confidence).

        Runs the full detect+recognize pipeline and joins the recognized text
        pieces in reading order. ``min_confidence`` is the lowest per-piece score
        (the weakest character governs how much to trust the whole name). Returns
        ``("", 0.0)`` when nothing is recognized.
        """
        arr = np.asarray(image.convert("RGB"))
        result = self._full_ocr().predict(arr)
        if not result:
            return "", 0.0
        res = result[0]
        data = getattr(res, "json", None)
        d = data.get("res", data) if isinstance(data, dict) else res
        if not isinstance(d, dict):
            return "", 0.0
        texts = d.get("rec_texts") or []
        scores = d.get("rec_scores") or []
        name = "".join(texts)
        conf = min((float(s) for s in scores), default=0.0)
        return name, conf

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        out: dict[int, str] = {}
        for crop in crops:
            text, _score = self._recognize_one(crop.image)
            if text:
                out[crop.id] = text[0]  # first (should be only) character
            else:
                logger.info("  paddle: no text for crop %d", crop.id)
        return out

    def recognize_scored(self, crops: list[Crop]) -> dict[int, tuple[str, float]]:
        """Like :meth:`recognize` but also returns PP-OCRv5's confidence score.

        Returns ``{crop_id: (char, score)}`` for every crop that produced text;
        crops with no recognized text are omitted. ``score`` is the recognizer's
        own probability in ``[0, 1]`` -- low values flag rare/ambiguous glyphs
        worth review.
        """
        out: dict[int, tuple[str, float]] = {}
        for crop in crops:
            text, score = self._recognize_one(crop.image)
            if text:
                out[crop.id] = (text[0], score)
            else:
                logger.info("  paddle: no text for crop %d", crop.id)
        return out

    def _recognize_one(self, image) -> tuple[str, float]:
        # PP-OCRv5 recognizes darker ink on light ground; feed RGB.
        arr = np.asarray(image.convert("RGB"))
        result = self._ocr.predict(arr)
        return _first_text_score(result)


def _first_text_score(result) -> tuple[str, float]:
    """Pull the recognized string + confidence out of a PaddleOCR 3.x result."""
    if not result:
        return "", 0.0
    res = result[0]
    data = getattr(res, "json", None)
    d = data.get("res", data) if isinstance(data, dict) else res
    if isinstance(d, dict):
        texts = d.get("rec_texts")
        scores = d.get("rec_scores")
        if texts:
            score = float(scores[0]) if scores else 0.0
            return "".join(texts), score
    return "", 0.0
