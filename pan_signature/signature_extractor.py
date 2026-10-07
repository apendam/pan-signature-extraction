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
    if config.local_contrast_ink:
        # Kernel must be wider than a pen stroke but narrower than the
        # background patches we want to ignore.
        k = max(5, int(0.25 * min(gray.shape)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        blurred = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
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


def _skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton (Lantuejoul): the centre lines of the strokes."""
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(mask)
    work = mask.copy()
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, element)
        skeleton |= cv2.subtract(work, cv2.dilate(eroded, element))
        work = eroded
    return skeleton


def _fill_small_holes(mask: np.ndarray, max_area: float, max_width: float) -> np.ndarray:
    """Fill interior holes that are tiny (area) or narrow slits (minimum
    width, however long); real letter loops are both larger and wider."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    out = mask.copy()
    for contour, info in zip(contours, hierarchy[0]):
        if info[3] == -1:  # an outer outline, not a hole
            continue
        narrow = min(cv2.minAreaRect(contour)[1]) < max_width
        if narrow or cv2.contourArea(contour) < max_area:
            cv2.drawContours(out, [contour], -1, 255, -1)
    return out


def _thin_strokes(mask: np.ndarray, amount_px: float) -> np.ndarray:
    """Shave `amount_px` off each side of every stroke. Strokes thinner than
    that would vanish, so their skeleton is added back to keep them
    connected; edges are then re-smoothed."""
    k = max(1, round(amount_px))
    eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    centre = cv2.dilate(_skeleton(mask), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    thinned = eroded | centre
    return ((cv2.GaussianBlur(thinned, (0, 0), sigmaX=0.8) > 120) * 255).astype(np.uint8)


def binarize(
    crop_bgr: np.ndarray, config: ExtractorConfig = ExtractorConfig()
) -> np.ndarray:
    """Pure black ink on a pure white background (still a 3-channel BGR
    image so it can be written/rotated like any other crop).

    Works on a black-hat (local-contrast) response of the crop:
      1. Optional light denoise, then upscale small crops (so thin strokes
         on low-resolution cards stay separate; the output can therefore
         be larger than the input crop).
      2. A coarse Otsu cut finds everything ink-like; a second cut inside
         that foreground picks the *strong* ink (the darkest strokes).
      3. Hysteresis: keep weak ink pixels only where they are connected to
         strong ink, which bridges broken strokes without admitting
         lighter blue fringe or card texture that touches no real stroke.
      4. Close 1px gaps, smooth edges, fill hairline holes inside strokes,
         thin the strokes for crisper lines, drop tiny isolated specks.
    No-op on an empty crop.
    """
    if crop_bgr.size == 0:
        return crop_bgr

    if config.binarize_denoise:
        crop_bgr = cv2.fastNlMeansDenoisingColored(crop_bgr, None, 3, 3, 5, 15)
    long_side = max(crop_bgr.shape[:2])
    scale = int(np.clip(round(config.binarize_target_long_side / long_side), 1, 8))
    if scale > 1:
        crop_bgr = cv2.resize(
            crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
        )

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(5, int(0.25 * min(gray.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    response = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

    coarse_cut, _ = cv2.threshold(response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = response > coarse_cut
    if config.exclude_blue_ink:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        ink &= cv2.inRange(hsv, _BLUE_INK_HSV_LOWER, _BLUE_INK_HSV_UPPER) == 0

    out = np.full_like(crop_bgr, 255)
    if ink.sum() < 10:
        out[ink] = 0
        return out

    strong_cut, _ = cv2.threshold(
        response[ink].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    span = strong_cut - coarse_cut
    strong = ink & (response > coarse_cut + config.binarize_strictness * span)
    weak = ink & (response > coarse_cut + config.binarize_weak_cutoff * span)

    # Hysteresis: weak components that contain at least one strong pixel.
    n_labels, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n_labels, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    mask = (keep[labels] * 255).astype(np.uint8)

    if config.binarize_smooth:
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        sigma = 0.6 * scale / 2 + 0.4
        mask = ((cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma) > 110) * 255).astype(np.uint8)

    if config.binarize_fill_holes_px > 0:
        mask = _fill_small_holes(
            mask, config.binarize_fill_holes_px * scale * scale, config.binarize_fill_slit_px * scale
        )
    if config.binarize_thin_px > 0:
        mask = _thin_strokes(mask, config.binarize_thin_px * scale)

    if config.binarize_min_speck_px > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = config.binarize_min_speck_px * scale * scale
        small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
        mask[np.isin(labels, small[small > 0])] = 0

    out[mask > 0] = 0
    return out


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
