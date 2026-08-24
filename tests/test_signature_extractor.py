import cv2
import numpy as np

from pan_signature.signature_extractor import refine_and_crop
from pan_signature.signature_locator import BBox


def _blank_page(width=400, height=300):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_refines_roi_to_ink_strokes():
    image = _blank_page()
    # Draw a "signature"-like stroke well inside a generous ROI.
    cv2.line(image, (150, 180), (250, 160), (0, 0, 0), 3)
    cv2.line(image, (170, 190), (230, 150), (0, 0, 0), 3)

    roi = BBox(x0=50, y0=100, x1=350, y1=250)
    refined_crop, refined_bbox = refine_and_crop(image, roi)

    # The refined box should be strictly smaller than the coarse ROI...
    assert (refined_bbox.x1 - refined_bbox.x0) < (roi.x1 - roi.x0)
    assert (refined_bbox.y1 - refined_bbox.y0) < (roi.y1 - roi.y0)
    # ...and still contain the strokes we drew.
    assert refined_bbox.x0 <= 150 and refined_bbox.x1 >= 250
    assert refined_crop.size > 0


def test_returns_full_roi_when_no_ink_found():
    image = _blank_page()
    roi = BBox(x0=50, y0=100, x1=150, y1=150)

    refined_crop, refined_bbox = refine_and_crop(image, roi)

    assert refined_bbox == roi
    assert refined_crop.shape[0] == roi.y1 - roi.y0
    assert refined_crop.shape[1] == roi.x1 - roi.x0


def test_empty_roi_is_handled_gracefully():
    image = _blank_page()
    roi = BBox(x0=100, y0=100, x1=100, y1=200)  # zero width

    refined_crop, refined_bbox = refine_and_crop(image, roi)

    assert refined_crop.size == 0
    assert refined_bbox == roi
