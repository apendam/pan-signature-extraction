from types import SimpleNamespace

from pan_signature.config import LocatorConfig
from pan_signature.signature_locator import locate_signature_region


def _vertex(x, y):
    return SimpleNamespace(x=x, y=y)


def _token(text, document_text, x0, y0, x1, y1):
    start_index = document_text.index(text)
    end_index = start_index + len(text)
    text_segment = SimpleNamespace(start_index=start_index, end_index=end_index)
    bounding_poly = SimpleNamespace(
        normalized_vertices=[
            _vertex(x0, y0),
            _vertex(x1, y0),
            _vertex(x1, y1),
            _vertex(x0, y1),
        ]
    )
    layout = SimpleNamespace(
        text_anchor=SimpleNamespace(text_segments=[text_segment]),
        bounding_poly=bounding_poly,
    )
    return SimpleNamespace(layout=layout)


def _document(text, tokens):
    page = SimpleNamespace(tokens=tokens)
    return SimpleNamespace(text=text, pages=[page])


def test_uses_default_roi_when_no_anchor_found():
    document = _document("PERMANENT ACCOUNT NUMBER CARD", [])
    bbox = locate_signature_region(document, image_width=1000, image_height=600)

    config = LocatorConfig()
    expected_x0 = round(config.default_roi_fraction[0] * 1000)
    expected_y0 = round(config.default_roi_fraction[1] * 600)
    assert bbox.x0 == expected_x0
    assert bbox.y0 == expected_y0


def test_anchors_above_a_signature_caption_token():
    text = "Signature"
    tokens = [_token("Signature", text, x0=0.1, y0=0.9, x1=0.3, y1=0.94)]
    document = _document(text, tokens)

    bbox = locate_signature_region(document, image_width=1000, image_height=1000)

    # ROI should sit above the caption (smaller y) and be clipped to the image.
    assert bbox.y1 == round(0.9 * 1000)
    assert bbox.y0 < bbox.y1
    assert 0 <= bbox.x0 < bbox.x1 <= 1000


def test_picks_lowest_token_when_multiple_match():
    text = "Signature Signature"
    first = "Signature"
    second_start = text.index("Signature", 1)
    tokens = [
        _token(first, text, x0=0.1, y0=0.2, x1=0.3, y1=0.24),
    ]
    # Second occurrence needs its own segment since both share the word;
    # build it manually to control the bounding box independently.
    second_segment = SimpleNamespace(
        start_index=second_start, end_index=second_start + len("Signature")
    )
    second_layout = SimpleNamespace(
        text_anchor=SimpleNamespace(text_segments=[second_segment]),
        bounding_poly=SimpleNamespace(
            normalized_vertices=[
                _vertex(0.1, 0.9),
                _vertex(0.3, 0.9),
                _vertex(0.3, 0.94),
                _vertex(0.1, 0.94),
            ]
        ),
    )
    tokens.append(SimpleNamespace(layout=second_layout))
    document = _document(text, tokens)

    bbox = locate_signature_region(document, image_width=1000, image_height=1000)

    # Should anchor on the lower (y=0.9) token, not the higher one (y=0.2).
    assert bbox.y1 == round(0.9 * 1000)
