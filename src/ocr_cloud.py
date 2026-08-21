"""Cloud OCR engines -- Mistral OCR, Google Cloud Vision, Google Document AI.

These are all document/full-page OCR services rather than single-glyph
classifiers, so on a bare one-character crop they tend to return nothing or
hallucinate surrounding structure. We therefore reuse :func:`src.ocr.pack_grid`
to lay many crops onto one labelled sheet and OCR the sheet in a single call --
the dense-text context these engines expect -- then map each recognized token
back to its cell by reading order.

Each engine here follows the :class:`src.ocr.OcrEngine` protocol (``name`` +
``recognize(crops) -> {id: char}``) and is imported lazily by
:func:`src.ocr.get_engine` so ``src.ocr`` loads without these SDKs present.

Auth:
* Mistral -- ``MISTRAL_API_KEY`` in the environment.
* Google  -- Application Default Credentials (``gcloud auth application-default
  login``); the Vision and Document AI APIs must be enabled on the ADC project.
  Document AI also needs a processor id (``DOCAI_PROCESSOR_ID``) and location
  (``DOCAI_LOCATION``, default ``us``) plus project (``DOCAI_PROJECT``).
"""

from __future__ import annotations

import io
import logging
import os
import re

from PIL import Image

from src.ocr import Crop, pack_grid

logger = logging.getLogger(__name__)

# A "character" for our purpose: a CJK ideograph (incl. extension blocks) or a
# CJK-compatibility form. Used to strip markdown/latin noise the document
# engines add around the glyphs.
_CJK = re.compile(
    r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ffff]"
)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _cjk_chars(text: str) -> list[str]:
    """Every CJK character in ``text``, in order (drops list markers, spaces)."""
    return _CJK.findall(text)


class MistralOCREngine:
    """Mistral OCR (``mistral-ocr-latest``) over grid-packed sheets.

    Mistral OCR returns markdown per page. We pack a grid, OCR it, then take the
    CJK characters from the returned markdown in reading order and align them to
    the grid cells. Because a document engine may merge/split cells, alignment is
    best-effort: if the CJK-count matches the cell-count we map 1:1, otherwise we
    fall back to one call per crop for that batch.
    """

    def __init__(self, model: str = "mistral-ocr-latest", batch: int = 40):
        self.name = f"mistral:{model}"
        self.model = model
        self.batch = batch
        from mistralai import Mistral

        self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def _ocr_image(self, img: Image.Image) -> str:
        import base64

        b64 = base64.b64encode(_png_bytes(img)).decode()
        resp = self._client.ocr.process(
            model=self.model,
            document={"type": "image_url", "image_url": f"data:image/png;base64,{b64}"},
        )
        return "\n".join(p.markdown for p in resp.pages) if resp.pages else ""

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        out: dict[int, str] = {}
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            sheet, order = pack_grid(chunk, label=False)
            chars = _cjk_chars(self._ocr_image(sheet))
            if len(chars) == len(order):
                out.update(dict(zip(order, chars)))
            else:
                logger.info(
                    "  mistral: grid count %d != cells %d; per-crop fallback",
                    len(chars),
                    len(order),
                )
                for crop in chunk:
                    c = _cjk_chars(self._ocr_image(crop.image))
                    if c:
                        out[crop.id] = c[0]
        return out


class GoogleVisionEngine:
    """Google Cloud Vision DOCUMENT_TEXT_DETECTION over grid-packed sheets."""

    def __init__(self, batch: int = 40):
        self.name = "google:vision"
        self.batch = batch
        from google.cloud import vision

        self._vision = vision
        self._client = vision.ImageAnnotatorClient()

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        out: dict[int, str] = {}
        ctx = self._vision.ImageContext(language_hints=["zh-Hans", "zh-Hant"])
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            sheet, order = pack_grid(chunk, label=False)
            image = self._vision.Image(content=_png_bytes(sheet))
            resp = self._client.document_text_detection(image=image, image_context=ctx)
            if resp.error.message:
                raise RuntimeError(f"Vision API: {resp.error.message}")
            chars = _cjk_chars(resp.full_text_annotation.text)
            if len(chars) == len(order):
                out.update(dict(zip(order, chars)))
            else:
                logger.info(
                    "  vision: grid count %d != cells %d; per-crop fallback",
                    len(chars),
                    len(order),
                )
                for crop in chunk:
                    im = self._vision.Image(content=_png_bytes(crop.image))
                    r = self._client.document_text_detection(image=im, image_context=ctx)
                    c = _cjk_chars(r.full_text_annotation.text)
                    if c:
                        out[crop.id] = c[0]
        return out


class GoogleDocAIEngine:
    """Google Document AI Enterprise OCR processor over grid-packed sheets.

    Needs a created OCR processor. Reads ``DOCAI_PROJECT`` (default: ADC project),
    ``DOCAI_LOCATION`` (default ``us``), ``DOCAI_PROCESSOR_ID``.
    """

    def __init__(self, batch: int = 40):
        self.name = "google:docai"
        self.batch = batch
        from google.cloud import documentai_v1 as documentai

        self._da = documentai
        self._client = documentai.DocumentProcessorServiceClient()
        project = os.environ.get("DOCAI_PROJECT")
        location = os.environ.get("DOCAI_LOCATION", "us")
        proc_id = os.environ["DOCAI_PROCESSOR_ID"]
        if not project:
            import google.auth

            _, project = google.auth.default()
        self._name = self._client.processor_path(project, location, proc_id)

    def _ocr_image(self, img: Image.Image) -> str:
        raw = self._da.RawDocument(content=_png_bytes(img), mime_type="image/png")
        opts = self._da.ProcessOptions(
            ocr_config=self._da.OcrConfig(
                hints=self._da.OcrConfig.Hints(language_hints=["zh-Hans", "zh-Hant"])
            )
        )
        req = self._da.ProcessRequest(
            name=self._name, raw_document=raw, process_options=opts
        )
        return self._client.process_document(request=req).document.text

    def recognize(self, crops: list[Crop]) -> dict[int, str]:
        out: dict[int, str] = {}
        for start in range(0, len(crops), self.batch):
            chunk = crops[start : start + self.batch]
            sheet, order = pack_grid(chunk, label=False)
            chars = _cjk_chars(self._ocr_image(sheet))
            if len(chars) == len(order):
                out.update(dict(zip(order, chars)))
            else:
                logger.info(
                    "  docai: grid count %d != cells %d; per-crop fallback",
                    len(chars),
                    len(order),
                )
                for crop in chunk:
                    c = _cjk_chars(self._ocr_image(crop.image))
                    if c:
                        out[crop.id] = c[0]
        return out
