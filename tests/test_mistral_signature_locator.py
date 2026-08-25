from types import SimpleNamespace

import pytest

from pan_signature.mistral_signature_locator import locate_signature_region_mistral


def _block(block_type, top_left_x, top_left_y, bottom_right_x, bottom_right_y, content=""):
    return SimpleNamespace(
        type=block_type,
        top_left_x=top_left_x,
        top_left_y=top_left_y,
        bottom_right_x=bottom_right_x,
        bottom_right_y=bottom_right_y,
        content=content,
    )


def _page(blocks):
    return SimpleNamespace(blocks=blocks)


def _response(pages):
    return SimpleNamespace(pages=pages)


def test_locates_signature_block_and_returns_its_bbox():
    text = _block("text", 0, 0, 200, 40, content="PERMANENT ACCOUNT NUMBER")
    signature = _block("signature", 100, 500, 300, 560, content="")
    response = _response([_page([text, signature])])

    bbox = locate_signature_region_mistral(response, image_width=1000, image_height=600)

    assert bbox.as_tuple() == (100, 500, 300, 560)


def test_raises_runtime_error_when_no_signature_block_present():
    text = _block("text", 0, 0, 200, 40, content="PERMANENT ACCOUNT NUMBER")
    response = _response([_page([text])])

    with pytest.raises(RuntimeError, match="No signature block found"):
        locate_signature_region_mistral(response, image_width=1000, image_height=600)


def test_treats_bbox_as_absolute_pixels_when_a_coordinate_exceeds_one():
    # If these were (mis)treated as normalized fractions, scaling by the
    # image size and clipping would produce a very different box.
    signature = _block("signature", 50, 60, 70, 80)
    response = _response([_page([signature])])

    bbox = locate_signature_region_mistral(response, image_width=1000, image_height=1000)

    assert bbox.as_tuple() == (50, 60, 70, 80)


def test_treats_bbox_as_normalized_fractions_when_no_coordinate_exceeds_one():
    signature = _block("signature", 0.05, 0.6, 0.25, 0.8)
    response = _response([_page([signature])])

    bbox = locate_signature_region_mistral(response, image_width=1000, image_height=500)

    assert bbox.as_tuple() == (
        round(0.05 * 1000),
        round(0.6 * 500),
        round(0.25 * 1000),
        round(0.8 * 500),
    )
