"""QA tooling for the Zeng zupu pipeline — review + verify each stage's output.

Tools are named by the pipeline stage they review (``sN_``), matching the ``sN_``
modules in ``src/``. A tool that spans stages uses an ``all_`` prefix instead.
Interactive servers use ports whose last digit echoes the stage. Run each as
``python -m scripts.qa.<tool>``:

Interactive review servers (open in a browser):
  * ``s1_borders``  — Stage 1 (extract): frame/corner QA. Reviews pages whose two
                      frame detectors disagree or failed; drag corners to fix.
                      Port 8761.  →  python -m scripts.qa.s1_borders
  * ``s2_classify`` — Stage 2 (classify): page grid colored tree/bio; click a cell
                      to flip a wrong label (saved as an override crop/merge honor).
                      Port 8762.  →  python -m scripts.qa.s2_classify
  * ``s7_ocr``      — Stage 7 (OCR): per-character name filmstrip for correcting
                      OCR readings.  Port 8767.  →  python -m scripts.qa.s7_ocr

Static-artifact generators (write images + an HTML index to click through):
  * ``s5_parse``      — Stage 5 (build_tree): parse overlays per subtree graph
                        (red name boxes + edges, green orphan bridges, blue page
                        seams). The main parse/bridging review tool.
                        →  python -m scripts.qa.s5_parse --book bookN
  * ``all_filmstrip`` — spans Stages 1–5: per-page filmstrip (raw → page → graph →
                        parse, side by side). Cross-stage page tracer.
                        →  python -m scripts.qa.all_filmstrip --book bookN ...

The three servers use ports 8761 / 8762 / 8767 (last digit = stage), all clear of
8000, so they can run at the same time and never collide with local dev on 8000.
"""
