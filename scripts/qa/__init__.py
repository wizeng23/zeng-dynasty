"""QA tooling for the Zeng zupu pipeline — review + verify each stage's output.

All QA tools live here and run as ``python -m scripts.qa.<tool>``:

Interactive review servers (open in a browser):
  * ``borders``  — frame/corner QA for Stage 1 (extract_pages). Reviews pages
                   whose two frame detectors disagree or failed; drag corners to
                   fix.  Port 8760.  →  python -m scripts.qa.borders
  * ``ocr``      — per-character name filmstrip for correcting OCR readings.
                   Port 8761.  →  python -m scripts.qa.ocr

Static-artifact generators (write images + an HTML index to click through):
  * ``overlay``  — parse overlays per subtree graph (name boxes, line endpoints).
                   →  python -m scripts.qa.overlay --book bookN
  * ``artifacts``— per-stage checkpoint screenshots (raw → page → graph → parse).
                   →  python -m scripts.qa.artifacts --book bookN ...

The two servers use adjacent ports (8760 / 8761), both clear of 8000, so they
can run at the same time and never collide with local dev work on 8000.
"""
