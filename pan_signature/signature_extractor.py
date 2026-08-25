"""OpenCV refinement: shrink a coarse ROI down to the actual ink strokes.

The ROI from `signature_locator.py` / `mistral_signature_locator.py` is
deliberately generous (a caption anchor's own bounding box, or a fixed
page fraction). This step thresholds it, optionally strips out printed
blue caption text, groups the remaining dark pixels into connected
components, and tightens the crop to the union of components that look
like ink rather than page noise.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import ExtractorConfig
from .signature_locator import BBox

# Confirmed against a real Indian PAN card: the printed field caption
# ("हस्ताक्षर / Signature") sitting right next to the signature is blue,
# while the signature itself is black. These HSV bounds (OpenCV's 0-180
# hue scale) are a generic "blue ink" range, not tuned to one sample.
_BLUE_INK_HSV_LOWER = np.array([90, 60, 40])
_BLUE_INK_HSV_UPPER = np.array([140, 255, 255])


def refine_and_crop(
    image_bgr: np.ndarray,
    roi: BBox,
    config: ExtractorConfig = ExtractorConfig(),
) -> tuple[np.ndarray, BBox]:
    """Crop `image_bgr` to `roi`, then tighten the crop to the detected
    ink strokes inside it.

    Returns (cropped_image, bbox_in_original_image). If no ink-like
    component is found, returns the full ROI crop unchanged so the
    pipeline degrades gracefully instead of failing.
    """
    crop = image_bgr[roi.y0 : roi.y1, roi.x0 : roi.x1]
    if crop.size == 0:
        return crop, roi

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if config.exclude_blue_ink:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, _BLUE_INK_HSV_LOWER, _BLUE_INK_HSV_UPPER)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(blue_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = config.min_component_area_fraction * crop.shape[0] * crop.shape[1]
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]

    if not boxes:
        return crop, roi

    xs0 = min(b[0] for b in boxes)
    ys0 = min(b[1] for b in boxes)
    xs1 = max(b[0] + b[2] for b in boxes)
    ys1 = max(b[1] + b[3] for b in boxes)

    pad = config.padding_px
    xs0 = max(0, xs0 - pad)
    ys0 = max(0, ys0 - pad)
    xs1 = min(crop.shape[1], xs1 + pad)
    ys1 = min(crop.shape[0], ys1 + pad)

    refined = crop[ys0:ys1, xs0:xs1]
    refined_bbox = BBox(
        x0=roi.x0 + xs0, y0=roi.y0 + ys0, x1=roi.x0 + xs1, y1=roi.y0 + ys1
    )
    return refined, refined_bbox
