"""Clean exports of an extracted (black ink on white) signature crop:
a transparent-background PNG and an SVG traced from the ink outlines."""
from __future__ import annotations

import cv2
import numpy as np


def _ink_mask(signature_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(signature_bgr, cv2.COLOR_BGR2GRAY)
    return (gray < 128).astype(np.uint8)


def to_transparent_png(signature_bgr: np.ndarray) -> np.ndarray:
    """BGRA image: black ink, fully transparent everywhere else. The
    original anti-aliasing is kept as partial transparency."""
    gray = cv2.cvtColor(signature_bgr, cv2.COLOR_BGR2GRAY)
    out = np.zeros((*gray.shape, 4), dtype=np.uint8)
    out[:, :, 3] = 255 - gray  # darker pixel -> more opaque ink
    return out


def to_svg(signature_bgr: np.ndarray, epsilon: float = 0.6, color: str = "#000000") -> str:
    """SVG of the ink outlines (holes handled with even-odd fill), scaled by
    the image size so it can be resized freely without going blocky."""
    mask = _ink_mask(signature_bgr)
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    paths = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue  # speck
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        paths.append("M" + " L".join(f"{x},{y}" for x, y in points) + " Z")
    d = " ".join(paths)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'<path d="{d}" fill="{color}" fill-rule="evenodd"/></svg>\n'
    )
