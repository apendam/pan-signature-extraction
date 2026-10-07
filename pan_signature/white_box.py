"""Find the white signature box printed on newer-format PAN cards (Type 1).

These cards have no "Signature" caption: the holder signs inside a plain
white rectangle next to the Date of Birth, above the "PAN Application
Digitally Signed" footer. When such a box is present it is the signature
region and nothing else should be considered.

Works on the (already upright) card image in pixel coordinates; the search
is confined to a window so bright glare elsewhere is not mistaken for it.
"""
from __future__ import annotations

import cv2
import numpy as np

from .signature_locator import BBox

# The card itself is often bright too (glare, washed-out phone photos), so
# "white box" is judged relative to the window around it: a region clearly
# less saturated than the card but not darker than it. The right saturation
# cut-off varies per photo, so levels are tried from strict to loose and the
# first one that yields a box-shaped blob wins (the tightest fit).
_SATURATION_LEVELS = range(12, 52, 4)
_MIN_AREA_FRACTION = 0.004
_MAX_AREA_FRACTION = 0.08
_MIN_ASPECT, _MAX_ASPECT = 1.5, 8.0
_MIN_HULL_FILL = 0.8  # blob area / its bounding box, using the convex hull
_INSET_FRACTION = 0.04
_GROW_STEPS = 6  # extra (looser) saturation levels tried after the first hit
# A real white box is much less saturated than the card around it. Candidates
# that are not clearly more colourless than a ring just outside them (e.g. a
# glare patch or bright card art) are rejected.
_RING_FRACTION = 0.15
_MAX_SATURATION_RATIO = 0.65
_MIN_SATURATION_DROP = 10


def _best_box(mask, image_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if not (_MIN_AREA_FRACTION * image_area <= area <= _MAX_AREA_FRACTION * image_area):
            continue
        if not (_MIN_ASPECT <= w / max(h, 1) <= _MAX_ASPECT):
            continue
        if cv2.contourArea(cv2.convexHull(contour)) / area < _MIN_HULL_FILL:
            continue
        if area > best_area:
            best, best_area = (x, y, w, h), area
    return best


def _stands_out(saturation: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    pad = max(3, round(_RING_FRACTION * min(w, h)))
    height, width = saturation.shape
    ex0, ey0 = max(0, x - pad), max(0, y - pad)
    ex1, ey1 = min(width, x + w + pad), min(height, y + h + pad)
    ring = np.ones((ey1 - ey0, ex1 - ex0), dtype=bool)
    ring[y - ey0 : y - ey0 + h, x - ex0 : x - ex0 + w] = False
    if ring.sum() < 50:
        return True  # box fills the window; nothing to compare against
    inner = float(np.median(saturation[y : y + h, x : x + w]))
    outer = float(np.median(saturation[ey0:ey1, ex0:ex1][ring]))
    return inner <= _MAX_SATURATION_RATIO * outer and outer - inner >= _MIN_SATURATION_DROP


def find_white_box(image_bgr: np.ndarray, window: BBox) -> BBox | None:
    """Tightest bright, colourless, wide rectangle inside `window`, inset a
    little so the box's own border is not part of the crop."""
    window = window.clip(image_bgr.shape[1], image_bgr.shape[0])
    sub = image_bgr[window.y0 : window.y1, window.x0 : window.x1]
    if sub.size == 0:
        return None

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    value = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    not_darker = value >= np.median(value)
    closer = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    image_area = image_bgr.shape[0] * image_bgr.shape[1]

    def box_at(level):
        mask = ((saturation <= level) & not_darker).astype(np.uint8) * 255
        # Handwriting inside the box leaves holes; close them so the box is one blob.
        return _best_box(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closer), image_area)

    best, found_at = None, None
    for level in _SATURATION_LEVELS:
        best = box_at(level)
        if best is not None:
            found_at = level
            break
    if best is None:
        return None

    # The strictest hit tends to be only the box's brightest core. Keep
    # loosening while the blob stays box-shaped and still contains the
    # previous one, so the whole box (and all the handwriting) is covered.
    for level in range(found_at + 2, found_at + 2 + _GROW_STEPS * 2, 2):
        grown = box_at(level)
        if grown is None:
            break
        gx, gy, gw, gh = grown
        bx, by, bw, bh = best
        if gx <= bx and gy <= by and gx + gw >= bx + bw and gy + gh >= by + bh:
            best = grown
        else:
            break
    x, y, w, h = best
    if not _stands_out(saturation, x, y, w, h):
        return None
    inset_x, inset_y = round(_INSET_FRACTION * w), round(_INSET_FRACTION * h)
    return BBox(
        x0=window.x0 + x + inset_x,
        y0=window.y0 + y + inset_y,
        x1=window.x0 + x + w - inset_x,
        y1=window.y0 + y + h - inset_y,
    )
