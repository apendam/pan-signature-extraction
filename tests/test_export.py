import cv2
import numpy as np

from pan_signature.export import to_svg, to_transparent_png


def _signature():
    img = np.full((60, 200, 3), 255, dtype=np.uint8)
    cv2.line(img, (20, 30), (180, 30), (0, 0, 0), 5)
    cv2.circle(img, (100, 30), 12, (0, 0, 0), 3)  # a loop => a hole inside the ink
    return img


def test_transparent_png_has_ink_opaque_and_background_transparent():
    out = to_transparent_png(_signature())

    assert out.shape == (60, 200, 4)
    assert out[30, 20, 3] == 255  # on the stroke
    assert out[5, 5, 3] == 0  # empty corner
    assert (out[:, :, :3] == 0).all()  # ink colour is black everywhere


def test_svg_traces_ink_with_even_odd_holes_and_matches_size():
    svg = to_svg(_signature())

    assert 'viewBox="0 0 200 60"' in svg
    assert 'fill-rule="evenodd"' in svg
    assert svg.count("M") >= 2  # outer outline plus the loop's inner hole
