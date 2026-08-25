"""Turn a Mistral OCR response into a pixel region of interest (ROI) for
the signature.

Empirical finding (tested against a real Indian PAN card, using both
`model="mistral-ocr-latest"` and the explicit `"mistral-ocr-4-1"`): no
block ever came back with `type == "signature"`. Instead, the signature's
ink and the printed caption next to it ("हस्ताक्षर / Signature") were
merged into one ordinary `"text"` block. So the "no anchor heuristic
needed" idea (Mistral hands you a signature box directly) does not hold
up in practice for this document type -- the docs describe a `type`
field with a `"signature"` value, and the installed SDK's response models
agree on the field names, but that classification apparently doesn't
trigger for a compact, ID-card-style signature area.

This module therefore mirrors `signature_locator.py`'s tiered, degrade-
gracefully approach instead of assuming a dedicated block exists:

1. If a block's type discriminator does say "signature" (kept in case a
   future model version or a different document layout does emit it),
   use it.
2. Otherwise, look for a block whose *content* text contains one of
   `LocatorConfig.anchor_keywords` (e.g. "Signature" / "हस्ताक्षर") and
   use that block's own bounding box as the ROI -- confirmed on a real
   sample to tightly bound the signature (plus its caption, which
   `refine_and_crop()`'s blue-ink exclusion then strips back out, since
   the caption is printed in blue and the signature is black).
3. Otherwise, fall back to `LocatorConfig.default_roi_fraction`, exactly
   as the Document AI path does.

This module only reads plain attributes/keys (duck-typed) so it works
against the real `mistralai` SDK response object or against a
lightweight dict/SimpleNamespace stand-in with the same shape in tests.
"""
from __future__ import annotations

from .config import LocatorConfig
from .signature_locator import BBox

# "type" is the field name confirmed by both the Mistral docs and the
# installed SDK's response models; the others are kept as a defensive
# fallback in case of schema drift or a hand-rolled dict response.
_TYPE_FIELD_NAMES = ("type", "block_type", "category")

_BBOX_FIELD_NAMES = ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")


def _get(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None


def _pages_of(ocr_response):
    return _get(ocr_response, "pages") or []


def _blocks_of(page):
    return _get(page, "blocks") or []


def _all_blocks(ocr_response):
    for page in _pages_of(ocr_response):
        for block in _blocks_of(page):
            yield block


def _block_type_value(block):
    for name in _TYPE_FIELD_NAMES:
        value = _get(block, name)
        if isinstance(value, str) and value:
            return value
    return None


def _is_signature_type(value: str) -> bool:
    return "signature" in value.strip().lower()


def _block_content(block) -> str:
    return _get(block, "content") or ""


def _contains_keyword(text: str, keywords) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _lowest(blocks):
    """Prefer the bottom-most match (largest top_left_y), same tie-break
    as the Document AI anchor locator, in case more than one candidate
    matches."""
    return max(blocks, key=lambda b: _get(b, "top_left_y") or 0)


def _find_signature_block(ocr_response, keywords):
    typed_matches = [
        block
        for block in _all_blocks(ocr_response)
        if (value := _block_type_value(block)) is not None and _is_signature_type(value)
    ]
    if typed_matches:
        return _lowest(typed_matches)

    content_matches = [
        block
        for block in _all_blocks(ocr_response)
        if _contains_keyword(_block_content(block), keywords)
    ]
    if content_matches:
        return _lowest(content_matches)

    return None


def _bbox_from_block(block, image_width: int, image_height: int) -> BBox:
    x0, y0, x1, y1 = (_get(block, name) for name in _BBOX_FIELD_NAMES)
    if None in (x0, y0, x1, y1):
        raise RuntimeError(
            "Signature block is missing one or more expected bounding-box "
            f"fields {_BBOX_FIELD_NAMES}."
        )

    # Absolute pixel coordinates vs. normalized 0..1 fractions: a
    # normalized fraction can never exceed 1.0, so any coordinate above
    # that threshold means the box is already in pixel space.
    if max(x0, y0, x1, y1) > 1.0:
        px0, py0, px1, py1 = x0, y0, x1, y1
    else:
        px0, py0, px1, py1 = (
            x0 * image_width,
            y0 * image_height,
            x1 * image_width,
            y1 * image_height,
        )

    return BBox(
        x0=round(px0), y0=round(py0), x1=round(px1), y1=round(py1)
    ).clip(image_width, image_height)


def _default_roi(image_width: int, image_height: int, config: LocatorConfig) -> BBox:
    roi_x0, roi_y0, roi_x1, roi_y1 = config.default_roi_fraction
    return BBox(
        x0=round(roi_x0 * image_width),
        y0=round(roi_y0 * image_height),
        x1=round(roi_x1 * image_width),
        y1=round(roi_y1 * image_height),
    ).clip(image_width, image_height)


def locate_signature_region_mistral(
    ocr_response,
    image_width: int,
    image_height: int,
    config: LocatorConfig = LocatorConfig(),
) -> BBox:
    block = _find_signature_block(ocr_response, config.anchor_keywords)
    if block is None:
        return _default_roi(image_width, image_height, config)
    return _bbox_from_block(block, image_width, image_height)
