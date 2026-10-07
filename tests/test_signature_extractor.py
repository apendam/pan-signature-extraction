from dataclasses import replace

import cv2
import numpy as np

from pan_signature.config import ExtractorConfig
from pan_signature.signature_extractor import (
    binarize,
    orient_horizontal,
    refine_and_crop,
    whiten_background,
)
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


def test_excludes_blue_printed_caption_next_to_black_signature():
    # Mirrors a real Indian PAN card: a blue printed caption sits right
    # next to the black signature within the same coarse ROI.
    image = _blank_page()
    cv2.line(image, (60, 150), (140, 130), (0, 0, 0), 3)  # black "signature"
    cv2.line(image, (260, 150), (340, 130), (255, 0, 0), 3)  # blue "caption" (BGR)

    roi = BBox(x0=0, y0=100, x1=400, y1=200)
    refined_crop, refined_bbox = refine_and_crop(image, roi)

    assert refined_bbox.x1 <= 200  # blue stroke (x>=260) excluded
    assert refined_bbox.x0 <= 60 and refined_bbox.x1 >= 140  # black stroke kept
    assert refined_crop.size > 0


def test_keeps_blue_ink_when_exclusion_is_disabled():
    image = _blank_page()
    cv2.line(image, (60, 150), (140, 130), (0, 0, 0), 3)
    cv2.line(image, (260, 150), (340, 130), (255, 0, 0), 3)

    roi = BBox(x0=0, y0=100, x1=400, y1=200)
    config = ExtractorConfig(exclude_blue_ink=False)
    refined_crop, refined_bbox = refine_and_crop(image, roi, config)

    assert refined_bbox.x1 > 200  # blue stroke now included


def test_whiten_background_clears_everything_but_the_ink():
    # A textured (non-white) background, like a card photo, with a
    # signature-like stroke on it.
    image = np.full((100, 200, 3), 180, dtype=np.uint8)
    cv2.line(image, (40, 50), (160, 50), (0, 0, 0), 5)

    whitened = whiten_background(image)

    assert tuple(whitened[10, 10]) == (255, 255, 255)  # far from the stroke
    assert tuple(whitened[50, 100])[0] < 128  # on the stroke, still dark


def test_whiten_background_handles_empty_image():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert whiten_background(empty).size == 0


def test_orient_horizontal_rotates_portrait_to_landscape():
    portrait = np.zeros((100, 40, 3), dtype=np.uint8)
    portrait[10, 5] = (1, 2, 3)  # a marker pixel to check rotation direction

    landscape = orient_horizontal(portrait, direction="counterclockwise")

    assert landscape.shape[0] < landscape.shape[1]  # now width > height
    assert np.array_equal(
        landscape, cv2.rotate(portrait, cv2.ROTATE_90_COUNTERCLOCKWISE)
    )


def test_orient_horizontal_direction_matters():
    portrait = np.zeros((100, 40, 3), dtype=np.uint8)
    portrait[10, 5] = (1, 2, 3)

    cw = orient_horizontal(portrait, direction="clockwise")
    ccw = orient_horizontal(portrait, direction="counterclockwise")

    assert not np.array_equal(cw, ccw)


def test_orient_horizontal_is_noop_for_landscape_or_square():
    landscape = np.zeros((40, 100, 3), dtype=np.uint8)
    square = np.zeros((50, 50, 3), dtype=np.uint8)

    assert np.array_equal(orient_horizontal(landscape), landscape)
    assert np.array_equal(orient_horizontal(square), square)


def test_binarize_outputs_only_pure_black_and_white():
    crop = np.full((60, 120, 3), (200, 180, 120), dtype=np.uint8)  # tinted card
    cv2.line(crop, (10, 40), (100, 20), (120, 40, 20), 3)  # blue-pen stroke

    out = binarize(crop, ExtractorConfig(exclude_blue_ink=False))

    assert out.ndim == 3 and out.shape[2] == 3
    assert set(np.unique(out)) <= {0, 255}
    assert (out == 0).any() and (out == 255).any()


def test_binarize_keeps_dark_stroke_and_drops_faint_halo():
    crop = np.full((60, 200, 3), 230, dtype=np.uint8)
    cv2.line(crop, (10, 30), (190, 30), (200, 190, 150), 9)  # faint wide halo
    cv2.line(crop, (10, 30), (190, 30), (20, 20, 20), 2)  # dark core stroke

    out = binarize(crop, ExtractorConfig(exclude_blue_ink=False))

    scale = out.shape[0] // crop.shape[0]
    black_rows = np.where((out[:, :, 0] == 0).any(axis=1))[0]
    # Black pixels hug the thin core (a few px tall), not the 9px halo.
    assert (black_rows.max() - black_rows.min()) / scale < 6


def test_binarize_bridges_a_faint_gap_in_a_stroke():
    # A pen stroke whose middle is faint (what a dashed/broken line looks
    # like on a low-res card) should come out as ONE connected stroke.
    crop = np.full((50, 200, 3), 235, dtype=np.uint8)
    cv2.line(crop, (10, 25), (80, 25), (20, 20, 20), 3)
    cv2.line(crop, (80, 25), (120, 25), (114, 114, 114), 3)  # ink-like but weaker middle
    cv2.line(crop, (120, 25), (190, 25), (20, 20, 20), 3)

    def strokes(config):
        out = binarize(crop, config)
        n, _ = cv2.connectedComponents((out[:, :, 0] == 0).astype(np.uint8))
        return n - 1

    base = ExtractorConfig(exclude_blue_ink=False, binarize_smooth=False, binarize_thin_px=0)
    # Without hysteresis (weak cut-off == strong cut-off) the weak middle
    # is dropped and the stroke breaks in two; with it, the stroke holds.
    assert strokes(replace(base, binarize_weak_cutoff=base.binarize_strictness)) == 2
    assert strokes(base) == 1


def test_binarize_drops_isolated_specks():
    crop = np.full((50, 200, 3), 235, dtype=np.uint8)
    cv2.line(crop, (10, 25), (190, 25), (20, 20, 20), 3)
    crop[5, 5] = (20, 20, 20)  # one-pixel dust speck far from the stroke

    out = binarize(crop, ExtractorConfig(exclude_blue_ink=False))

    scale = out.shape[0] // crop.shape[0]
    assert (out[: 12 * scale, : 12 * scale] == 255).all()


def _line_thickness(out):
    ink = out[:, :, 0] == 0
    return ink.sum(axis=0)[ink.sum(axis=0) > 0].mean()  # mean ink pixels per ink column


def test_thinning_makes_strokes_thinner_but_keeps_them_connected():
    crop = np.full((60, 200, 3), 235, dtype=np.uint8)
    cv2.line(crop, (10, 30), (190, 30), (20, 20, 20), 5)

    base = ExtractorConfig(exclude_blue_ink=False, binarize_thin_px=0)
    thin = ExtractorConfig(exclude_blue_ink=False, binarize_thin_px=0.8)

    thick_out, thin_out = binarize(crop, base), binarize(crop, thin)
    n, _ = cv2.connectedComponents((thin_out[:, :, 0] == 0).astype(np.uint8))

    assert _line_thickness(thin_out) < 0.8 * _line_thickness(thick_out)
    assert n - 1 == 1  # still one unbroken stroke


def test_hollow_centre_of_a_marker_stroke_is_filled_but_a_real_loop_is_kept():
    crop = np.full((120, 260, 3), 235, dtype=np.uint8)
    cv2.line(crop, (10, 30), (120, 30), (20, 20, 20), 9)  # thick stroke...
    cv2.line(crop, (10, 30), (120, 30), (200, 200, 200), 1)  # ...with a pale centre line
    cv2.circle(crop, (200, 70), 22, (20, 20, 20), 3)  # a genuine open loop (letter 'o')

    out = binarize(crop, ExtractorConfig(exclude_blue_ink=False))

    scale = out.shape[0] // crop.shape[0]
    stroke = out[: 50 * scale, : 130 * scale, 0] == 0
    loop_inside = out[70 * scale, 200 * scale, 0]
    assert stroke[30 * scale, 60 * scale]  # the pale centre line did not leave a hole
    assert loop_inside == 255  # the loop's interior is still open
