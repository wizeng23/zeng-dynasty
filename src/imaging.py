"""Shared image I/O and low-level pixel helpers for the parsing pipeline.

Images are represented as 2D ``numpy.uint8`` arrays of a *binary* grid where
``0`` == an ink (black) pixel and ``1`` == background (white). This inverted
convention is inherited from the original pipeline: it makes "count the ink in
a row/column" a simple ``sum(1 - a)`` and lets padding be plain arrays of ones.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Directions used by the binary-ink convention.
INK = 0
BACKGROUND = 1


def get_image(filepath: str) -> np.ndarray:
    """Load an image as a binary ink grid.

    The scans use red guide-lines and grey noise; only sufficiently dark pixels
    in the red channel are treated as ink. Returns a ``uint8`` array where
    ``0`` == ink and ``1`` == background.
    """
    image = Image.open(filepath).convert("RGB")
    data = np.asarray(image)
    # Threshold on the red channel: this filters out red guide-marks and keeps
    # only dark ink. Values > 150 are background (1), <= 150 are ink (0).
    if len(data.shape) == 3:
        return (data[:, :, 0] > 150).astype(np.uint8)
    return (data > 150).astype(np.uint8)


def save_image(a: np.ndarray, filepath: str) -> None:
    """Save a binary ink grid to ``filepath`` as an 8-bit grayscale PNG."""
    img = Image.fromarray(a * 255, mode="L")
    img.save(filepath)


def show_image(a: np.ndarray) -> None:
    """Display a binary ink grid in the default image viewer (debug helper)."""
    out = Image.fromarray(np.uint8(a * 255))
    out.show()


def pad_image(a: np.ndarray, directions: str, padding: int = 20) -> np.ndarray:
    """Pad the given sides of an image with background (white) pixels.

    Args:
        a: Binary ink grid.
        directions: Any combination of the characters ``u`` (top), ``d``
            (bottom), ``l`` (left), ``r`` (right).
        padding: Number of pixels to add on each selected side.

    Returns:
        A new padded array.
    """
    if "u" in directions:
        a = np.vstack([a, np.ones((padding, a.shape[1])).astype(np.uint8)])
    if "d" in directions:
        a = np.vstack([np.ones((padding, a.shape[1])).astype(np.uint8), a])
    if "l" in directions:
        a = np.hstack([np.ones((a.shape[0], padding)).astype(np.uint8), a])
    if "r" in directions:
        a = np.hstack([a, np.ones((a.shape[0], padding)).astype(np.uint8)])
    return a
