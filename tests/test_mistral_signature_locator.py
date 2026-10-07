from types import SimpleNamespace

from pan_signature.config import LocatorConfig
from pan_signature.mistral_signature_locator import (
    locate_signature_region_mistral,
    locate_signature_region_mistral_with_source,
)


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


def test_falls_back_to_content_keyword_when_no_block_is_typed_signature():
    # Confirmed on a real PAN card: Mistral merges the signature ink and
    # its printed caption into one "text" block instead of a "signature"
    # typed block. The locator should still find it via block content.
    text = _block("text", 0, 0, 200, 40, content="PERMANENT ACCOUNT NUMBER")
    signature_and_caption = _block(
        "text", 23, 1235, 340, 1985, content="Abhyd. P\nहस्ताक्षर / Signature"
    )
    response = _response([_page([text, signature_and_caption])])

    bbox = locate_signature_region_mistral(response, image_width=2000, image_height=3300)

    # Same block, minus its caption line. The block is 750px tall for two
    # lines, but a printed caption is nowhere near 375px, so the cut is
    # capped at 4% of the image height (132px).
    assert bbox.as_tuple() == (23, 1235, 340, 1985 - 132)


def test_falls_back_to_default_roi_when_nothing_matches():
    text = _block("text", 0, 0, 200, 40, content="PERMANENT ACCOUNT NUMBER")
    response = _response([_page([text])])

    bbox = locate_signature_region_mistral(response, image_width=1000, image_height=1000)

    config = LocatorConfig()
    expected_x0 = round(config.default_roi_fraction[0] * 1000)
    expected_y0 = round(config.default_roi_fraction[1] * 1000)
    assert bbox.x0 == expected_x0
    assert bbox.y0 == expected_y0


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


def test_caption_only_block_searches_above_the_caption():
    # Seen on real cards: "Signature" is its own block and the handwriting
    # is printed above it, so the ROI must end at the caption's top edge.
    caption = _block("text", 175, 309, 280, 325, content="हस्ताक्षर / Signature")
    response = _response([_page([caption])])

    bbox, source = locate_signature_region_mistral_with_source(
        response, image_width=532, image_height=335
    )

    assert source == "above_caption"
    assert bbox.y1 == 309  # caption itself excluded
    assert bbox.y0 == 309 - 3 * 16  # three caption-heights up
    assert bbox.x0 < 175 and bbox.x1 > 280  # widened for wide handwriting


def test_right_of_dob_band_used_when_caption_is_missing():
    dob = _block("text", 20, 242, 178, 273, content="जन्म की तारीख / Date of Birth\n13/06/1986")
    photo = _block("image", 352, 188, 452, 289)
    response = _response([_page([dob, photo])])

    bbox, source = locate_signature_region_mistral_with_source(
        response, image_width=474, image_height=302
    )

    assert source == "right_of_dob"
    assert bbox.x0 == 178  # starts where the DOB text ends
    assert bbox.x1 == 352  # stops at the photo


def test_reports_default_source_when_nothing_matches():
    text = _block("text", 0, 0, 200, 40, content="PERMANENT ACCOUNT NUMBER")
    response = _response([_page([text])])

    _, source = locate_signature_region_mistral_with_source(
        response, image_width=1000, image_height=1000
    )

    assert source == "default"
