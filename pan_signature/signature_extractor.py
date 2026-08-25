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


def _ink_mask(crop_bgr: np.ndarray, config: ExtractorConfig) -> np.ndarray:
    """Foreground mask of ink-like pixels: dark strokes, optionally minus
    blue printed captions. Shared by `refine_and_crop` (to find the tight
    bounding box) and `whiten_background` (to know what to keep)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if config.exclude_blue_ink:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, _BLUE_INK_HSV_LOWER, _BLUE_INK_HSV_UPPER)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(blue_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


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

    mask = _ink_mask(crop, config)
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


def whiten_background(
    crop_bgr: np.ndarray, config: ExtractorConfig = ExtractorConfig()
) -> np.ndarray:
    """Replace everything outside the detected ink strokes with solid
    white, so the output is just the signature rather than the card's
    textured background. No-op on an empty crop."""
    if crop_bgr.size == 0:
        return crop_bgr
    mask = _ink_mask(crop_bgr, config)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    white = np.full_like(crop_bgr, 255)
    return np.where(mask_3ch > 0, crop_bgr, white)


def orient_horizontal(
    image_bgr: np.ndarray, direction: str = "counterclockwise"
) -> np.ndarray:
    """Rotate a taller-than-wide image 90 degrees so width > height (a
    signature reads naturally landscape). No-op if already landscape/
    square, or on an empty image.

    `direction` picks which way to rotate -- confirmed "counterclockwise"
    against a real PAN card scanned in portrait orientation (verified the
    signature reads left-to-right afterward, not backwards); if your
    source images are rotated the other way, pass "clockwise" instead.
    """
    if image_bgr.size == 0 or image_bgr.shape[0] <= image_bgr.shape[1]:
        return image_bgr
    code = (
        cv2.ROTATE_90_CLOCKWISE
        if direction == "clockwise"
        else cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    return cv2.rotate(image_bgr, code)
