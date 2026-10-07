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
   `LocatorConfig.anchor_keywords` (e.g. "Signature" / "हस्ताक्षर").
   On the PAN layouts tested the handwriting sits ABOVE that caption, so
   the ROI is built upward from the caption (caption-only block), or is
   the merged block minus its caption line. Either way the caption is
   excluded by position, not by ink colour, so blue-pen signatures work.
3. Otherwise, look for the "Date of Birth" block and take the band to its
   right (up to the card photo) -- for when OCR garbles the caption.
4. Otherwise, fall back to `LocatorConfig.default_roi_fraction`, exactly
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


def _content_lines(block) -> list[str]:
    return [line for line in _block_content(block).splitlines() if line.strip()]


def _above_caption_roi(
    block, image_width: int, image_height: int, config: LocatorConfig
) -> BBox:
    """ROI for the handwriting printed ABOVE a matched caption block.

    The caption itself is left out of the ROI by position (not by ink
    colour), so signatures written in blue pen survive. Two shapes of
    caption block occur in practice:
      * caption-only ("Signature" alone): search upward from its top edge;
      * merged (Mistral OCR'd the handwriting into the same block, e.g.
        "<scrawl>\nहस्ताक्षर / Signature"): the block already spans the
        handwriting, so keep it and drop only its last (caption) line.
    """
    box = _bbox_from_block(block, image_width, image_height)
    lines = _content_lines(block)
    line_h = (box.y1 - box.y0) / max(1, len(lines))
    caption_only = all(_contains_keyword(l, config.anchor_keywords) for l in lines)

    if caption_only:
        width = box.x1 - box.x0
        return BBox(
            x0=round(box.x0 - config.caption_pad_left * width),
            y0=round(box.y0 - config.caption_search_lines_above * line_h),
            x1=round(box.x1 + config.caption_pad_right * width),
            y1=box.y0,
        ).clip(image_width, image_height)
    # Merged block: lines are not equal height (the handwriting is taller
    # than the printed caption), so cap the caption cut at a plausible
    # caption height instead of trusting h / n_lines.
    caption_h = min(line_h, config.max_caption_height_fraction * image_height)
    return BBox(
        x0=box.x0, y0=box.y0, x1=box.x1, y1=round(box.y1 - caption_h)
    ).clip(image_width, image_height)


def _right_of_dob_roi(
    ocr_response, image_width: int, image_height: int, config: LocatorConfig
) -> BBox | None:
    """Fallback when no caption block matched: on newer PAN layouts the
    signature sits right of the Date-of-Birth block, in the same band,
    up to the card photo (the next image block to its right)."""
    dob_blocks = [
        b for b in _all_blocks(ocr_response)
        if _contains_keyword(_block_content(b), config.dob_keywords)
    ]
    if not dob_blocks:
        return None
    dob = _bbox_from_block(_lowest(dob_blocks), image_width, image_height)
    band_h = dob.y1 - dob.y0
    y0, y1 = round(dob.y0 - 0.5 * band_h), round(dob.y1 + 1.0 * band_h)

    x1 = image_width
    for block in _all_blocks(ocr_response):
        if (_block_type_value(block) or "") not in ("image", "footer"):
            continue
        other = _bbox_from_block(block, image_width, image_height)
        if other.x0 >= dob.x1 and other.y0 < y1 and other.y1 > y0:
            x1 = min(x1, other.x0)
    if x1 <= dob.x1:
        return None
    return BBox(x0=dob.x1, y0=y0, x1=x1, y1=y1).clip(image_width, image_height)


def locate_signature_region_mistral_with_source(
    ocr_response,
    image_width: int,
    image_height: int,
    config: LocatorConfig = LocatorConfig(),
) -> tuple[BBox, str]:
    """Like `locate_signature_region_mistral`, but also reports which tier
    produced the ROI: "typed", "above_caption", "right_of_dob" or
    "default". Only "default" can still contain a printed caption, so
    only that tier needs ink-colour caption stripping downstream."""
    block = _find_signature_block(ocr_response, config.anchor_keywords)
    if block is not None:
        if (value := _block_type_value(block)) is not None and _is_signature_type(value):
            return _bbox_from_block(block, image_width, image_height), "typed"
        return _above_caption_roi(block, image_width, image_height, config), "above_caption"

    roi = _right_of_dob_roi(ocr_response, image_width, image_height, config)
    if roi is not None:
        return roi, "right_of_dob"
    return _default_roi(image_width, image_height, config), "default"


def locate_signature_region_mistral(
    ocr_response,
    image_width: int,
    image_height: int,
    config: LocatorConfig = LocatorConfig(),
) -> BBox:
    return locate_signature_region_mistral_with_source(
        ocr_response, image_width, image_height, config
    )[0]
