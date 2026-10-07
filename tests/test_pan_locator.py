import numpy as np

from pan_signature import pan_locator as L


def _block(kind, x0, y0, x1, y1, content=""):
    return {"type": kind, "content": content, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _type1_card():
    """Blue-ish card with a white box right of Date of Birth and the
    'Digitally Signed' footer beyond it (the newer, caption-less layout)."""
    image = np.full((600, 1040, 3), (190, 120, 60), dtype=np.uint8)  # box ~3% of the frame
    image[190:270, 170:400] = (245, 245, 245)  # the white box
    dob = _block("text", 20, 190, 150, 270, "जन्म की तारीख / Date of Birth\n01/01/1990")
    footer = _block("text", 420, 200, 510, 240, "PAN Application Digitally Signed")
    return image, [dob, footer]


def test_type1_uses_the_handwriting_block_inside_the_box():
    image, blocks = _type1_card()
    blocks.append(_block("text", 230, 205, 330, 255, "Pappu"))

    located = L.locate(image, blocks)

    assert located.pan_type == L.TYPE1
    roi = located.roi
    # Around the handwriting, not the whole box...
    assert roi.x0 > 170 and roi.x1 < 400
    # ...covering it fully, and never leaving the white box.
    assert roi.x0 <= 230 and roi.x1 >= 330 and roi.y0 <= 205 and roi.y1 >= 255
    assert 170 <= roi.x0 and roi.x1 <= 400 and 190 <= roi.y0 and roi.y1 <= 270


def test_type1_prefers_a_signature_typed_block():
    image, blocks = _type1_card()
    blocks.append(_block("text", 172, 195, 190, 205, "ab"))  # stray text bit left of it
    blocks.append(_block("signature", 240, 205, 340, 255))

    roi = L.locate(image, blocks).roi

    assert roi.x0 <= 240 and roi.x1 >= 340
    assert roi.x0 > 190  # not stretched to cover the stray text


def test_type1_band_is_used_when_the_box_cannot_be_segmented():
    # No colour contrast anywhere: the white box cannot be found, but the
    # band between Date of Birth and the footer is still known.
    image = np.full((300, 520, 3), 230, dtype=np.uint8)
    dob = _block("text", 20, 190, 150, 270, "जन्म की तारीख / Date of Birth\n01/01/1990")
    footer = _block("text", 420, 200, 510, 240, "PAN Application Digitally Signed")
    hand = _block("text", 230, 205, 330, 255, "Pappu")

    located = L.locate(image, [dob, footer, hand])

    assert located.pan_type == L.TYPE1_BAND
    assert located.roi.x0 <= 230 and located.roi.x1 >= 330
    assert located.roi.x1 <= 420  # stays left of the footer


def test_typed_block_that_is_really_the_caption_is_cut_by_position():
    image = np.full((400, 600, 3), 200, dtype=np.uint8)
    pad = [_block("text", 20, 20 + i * 30, 200, 40 + i * 30, f"printed line {i}") for i in range(4)]
    spanning = _block("signature", 100, 300, 300, 380, "हस्ताक्षर / Signature")

    located = L.locate(image, pad + [spanning])

    assert located.pan_type == L.TYPE3
    assert located.roi.y1 < 380  # the caption line is not inside the region
