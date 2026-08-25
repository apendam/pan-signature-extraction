"""Turn a Mistral OCR response into a pixel region of interest (ROI) for
the signature.

Unlike Document AI's presence-only signature field (see `docai_client.py`
and `signature_locator.py`), Mistral's OCR block-classification feature
(`include_blocks=True`, the default -- documented under "OCR 4" at
docs.mistral.ai/studio/document-processing/basic_ocr) can label a content
block's `type` field with the literal value `"signature"` directly,
carrying its own bounding box on four flat fields: `top_left_x`,
`top_left_y`, `bottom_right_x`, `bottom_right_y`. If that holds, no
anchor-text heuristic is needed on this path.

Schema assumption and confidence
---------------------------------
The field names above match both the official docs and the response
models shipped in the `mistralai` SDK actually installed for this project
(`OCRSignatureBlock` etc. in `mistralai.client.models`) -- so the field
*names* are well supported. What is **not** confirmed is a real, populated
example: no API response containing an actual signature-typed block could
be found in Mistral's docs or in third-party write-ups at the time this
was written. The SDK types `top_left_x` and friends as plain integers,
which suggests they are absolute pixel coordinates rather than normalized
0..1 fractions, but that is inferred from the declared schema, not from
a live call. **Re-run this against one real OCR call on a document with
a visible signature before relying on it in production.**

Because of that residual uncertainty this module (a) checks a short list
of plausible field-name variants for the block-type discriminator instead
of hard-coding a single guess, and raises a specific error naming what it
checked if nothing matches, rather than silently returning the wrong
region, and (b) accepts either an absolute-pixel or a normalized (0..1)
bounding box, deciding which by checking whether any coordinate exceeds
1.0 (a normalized fraction cannot).

This module only reads plain attributes/keys (duck-typed) so it works
against the real `mistralai` SDK response object or against a
lightweight dict/SimpleNamespace stand-in with the same shape in tests.
"""
from __future__ import annotations

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


def _block_type_value(block):
    """Return the block's type-discriminator string, checking each
    plausible field name in turn, or None if none of them are present."""
    for name in _TYPE_FIELD_NAMES:
        value = _get(block, name)
        if isinstance(value, str) and value:
            return value
    return None


def _is_signature_type(value: str) -> bool:
    return "signature" in value.strip().lower()


def _find_signature_block(ocr_response):
    for page in _pages_of(ocr_response):
        for block in _blocks_of(page):
            value = _block_type_value(block)
            if value is not None and _is_signature_type(value):
                return block
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


def locate_signature_region_mistral(
    ocr_response, image_width: int, image_height: int
) -> BBox:
    block = _find_signature_block(ocr_response)
    if block is None:
        raise RuntimeError(
            "No signature block found in the Mistral OCR response. Checked "
            f"each block's {_TYPE_FIELD_NAMES} field(s) for a value "
            "containing 'signature'; none matched. Make sure the OCR call "
            "was made with include_blocks=True and that the document has "
            "a visible/legible signature region."
        )
    return _bbox_from_block(block, image_width, image_height)
