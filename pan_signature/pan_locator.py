"""Type-aware signature locator for an *upright* PAN card.

Three layouts occur on real cards (confirmed on a 200-image sample):

1. ``type1_white_box`` -- newer cards: the holder signs inside a plain white
   box, and there is no "Signature" caption. If a white box is present it
   is the signature region and nothing else is considered.
2. ``type2_english_caption`` -- the signature sits above a caption that
   reads just "Signature".
3. ``type3_bilingual_caption`` -- the signature sits above a caption that
   reads "हस्ताक्षर / Signature".

For 2 and 3 the caption is excluded from the region *by position*, so the
word "Signature" never lands in the output, and blue-pen signatures are not
lost to colour filtering. Two fallbacks cover cards where OCR dropped or
garbled the caption: Mistral's own ``signature``-typed block, then the band
right of the Date-of-Birth block. Everything works on plain block dicts
({"type","content","x0","y0","x1","y1"}) in pixel coordinates.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import numpy as np

from .config import LocatorConfig
from .signature_locator import BBox
from .white_box import find_white_box

TYPE1 = "type1_white_box"
TYPE1_BAND = "type1_white_box_band"  # box not detected; band between DOB and footer
TYPE2 = "type2_english_caption"
TYPE3 = "type3_bilingual_caption"
TYPED = "typed_signature_block"
RIGHT_OF_DOB = "right_of_dob"
DEFAULT = "default"

_CAPTION_EN = re.compile(r"\bsignature\b", re.I)
_CAPTION_HI = "हस्ताक्षर"
_DOB = re.compile(r"date\s*of\s*birth|जन्म", re.I)
_FOOTER = re.compile(r"digitally\s+signed", re.I)
_SERIAL = re.compile(r"^\W*\d[\d\s/.\-]{5,}\W*$")  # printed dates / serial numbers


@dataclass(frozen=True)
class Located:
    roi: BBox
    pan_type: str


def _content(block) -> str:
    return block.get("content") or ""


def _is_caption(block) -> bool:
    text = _content(block)
    return bool(_CAPTION_EN.search(text)) or _CAPTION_HI in text


def _is_signature_typed(block) -> bool:
    return "signature" in (block.get("type") or "").lower()


def _w(block) -> float:
    return block["x1"] - block["x0"]


def _h(block) -> float:
    return block["y1"] - block["y0"]


def _clip(x0, y0, x1, y1, width: int, height: int) -> BBox:
    return BBox(round(x0), round(y0), round(x1), round(y1)).clip(width, height)


def _reference_line_height(blocks, exclude, height: int) -> float:
    """Typical height of one printed text line on this card, taken from
    single-line text blocks. Used to tell a caption-only block from one
    where OCR merged the handwriting into the caption block."""
    heights = [
        _h(b)
        for b in blocks
        if b is not exclude
        and b.get("type") in ("text", "title")
        and "\n" not in _content(b).strip()
        and 3 < len(_content(b).strip()) < 40
        and _w(b) > _h(b)
    ]
    return statistics.median(heights) if heights else 0.035 * height


def _above_caption_roi(caption, blocks, width, height, config: LocatorConfig) -> BBox:
    cap_w, cap_h = _w(caption), _h(caption)
    ref_h = _reference_line_height(blocks, caption, height)
    merged = cap_h > 1.6 * ref_h
    if merged:
        # OCR merged the handwriting into the caption block: it already
        # spans the signature, so keep the block and drop its caption line.
        top = caption["y0"] - 0.5 * ref_h
        bottom = caption["y1"] - min(ref_h, 0.5 * cap_h)
    else:
        top = caption["y0"] - config.caption_search_lines_above * cap_h
        bottom = caption["y0"]

    # Printed text immediately left/right of the caption (e.g. the Date of
    # Birth block) bounds the widened region. Blocks that overlap the
    # caption's own span are handwriting that OCR read as text/image, not
    # neighbours: the region is widened to cover them in full, since a
    # signature is often much wider than its caption.
    left = caption["x0"] - config.caption_pad_left * cap_w
    right = caption["x1"] + config.caption_pad_right * cap_w
    bound_left, bound_right = -np.inf, np.inf
    for other in blocks:
        if other is caption or other.get("type") not in ("text", "title", "image"):
            continue
        near_band = other["y1"] > top and other["y0"] < bottom
        if not near_band:
            continue
        overlaps_caption = other["x0"] < caption["x1"] and other["x1"] > caption["x0"]
        if overlaps_caption:
            if other.get("type") in ("text", "image") and not _FOOTER.search(_content(other)):
                left, right = min(left, other["x0"]), max(right, other["x1"])
        elif other.get("type") in ("text", "title") and len(_content(other).strip()) > 3:
            if other["x1"] <= caption["x0"]:
                bound_left = max(bound_left, other["x1"])
            elif other["x0"] >= caption["x1"]:
                bound_right = min(bound_right, other["x0"])
    left, right = max(left, bound_left), min(right, bound_right)
    return _clip(left, top, right, bottom, width, height)


def _dob_block(blocks):
    matches = [b for b in blocks if _DOB.search(_content(b))]
    return max(matches, key=lambda b: b["y0"]) if matches else None


def _dob_footer_window(blocks, width, height):
    """Band right of the Date-of-Birth block, up to the "Digitally Signed"
    footer when it sits to the right (where the Type 1 white box lives).
    Returns (window, bounded_by_footer)."""
    dob = _dob_block(blocks)
    if dob is None:
        return _clip(0.2 * width, 0.55 * height, 0.95 * width, height, width, height), False
    footer = next((b for b in blocks if _FOOTER.search(_content(b))), None)
    bounded = bool(footer and footer["x0"] > dob["x1"])
    x1 = footer["x0"] if bounded else dob["x1"] + 4 * _w(dob)
    top, bottom = dob["y0"] - 0.25 * _h(dob), dob["y1"] + 0.5 * _h(dob)
    for other in blocks:
        # Printed lines just above / below the band (father's name, footer)
        # are not part of the signature box.
        if other is dob or other.get("type") not in ("text", "title"):
            continue
        overlaps_x = other["x0"] < x1 and other["x1"] > dob["x1"]
        if not overlaps_x:
            continue
        if _FOOTER.search(_content(other)) and other["y0"] > dob["y0"]:
            bottom = min(bottom, other["y0"] - 2)
        elif other["y1"] <= dob["y0"] + 0.1 * _h(dob):
            top = max(top, other["y1"] + 2)
    window = _clip(dob["x1"] - 0.05 * _w(dob), top, x1, bottom, width, height)
    return window, bounded


def _white_box_roi(image_bgr: np.ndarray, blocks, has_footer: bool) -> BBox | None:
    height, width = image_bgr.shape[:2]
    window, _ = _dob_footer_window(blocks, width, height)
    return find_white_box(image_bgr, window)


def _band_roi(blocks, width, height) -> BBox | None:
    """Type 1 fallback when the white box itself can't be segmented (pale or
    washed-out photo): the box always sits between Date of Birth and the
    footer, so use that band, tightened to any OCR block inside it."""
    window, bounded = _dob_footer_window(blocks, width, height)
    if not bounded or _dob_block(blocks) is None:
        return None
    inside = [
        b
        for b in blocks
        if b.get("type") in ("text", "image")
        and not _FOOTER.search(_content(b))
        and b["x0"] >= window.x0 - 5
        and b["x1"] <= window.x1 + 5
        and b["y0"] >= window.y0 - 5
        and b["y1"] <= window.y1 + 5
    ]
    if not inside:
        return window
    x0, y0 = min(b["x0"] for b in inside), min(b["y0"] for b in inside)
    x1, y1 = max(b["x1"] for b in inside), max(b["y1"] for b in inside)
    pad_x, pad_y = 0.15 * (x1 - x0), 0.25 * (y1 - y0)
    return _clip(
        max(window.x0, x0 - pad_x), max(window.y0, y0 - pad_y),
        min(window.x1, x1 + pad_x), min(window.y1, y1 + pad_y), width, height,
    )


def _intersect(a: BBox, b: BBox) -> BBox | None:
    box = BBox(max(a.x0, b.x0), max(a.y0, b.y0), min(a.x1, b.x1), min(a.y1, b.y1))
    return box if box.x1 > box.x0 and box.y1 > box.y0 else None


def _area(b: BBox) -> int:
    return max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0)


def _handwriting_in(window: BBox, box: BBox | None, blocks, width, height) -> BBox | None:
    """Region of the handwriting in a Type 1 box, from Mistral's blocks.

    Blocks are looked for inside `window` (the band between Date of Birth
    and the footer, which is stable), not inside the detected `box`, which
    can come out partial on pale photos. Mistral's own `signature`-typed
    block is used when it exists; otherwise the text/image block(s) in the
    window (usually the signature read as a name, or as an image). The union
    is padded generously, since OCR block boxes are often tighter than the
    ink, then limited to the window and, when the detected box really does
    contain the handwriting, to that box so nothing outside it leaks in.
    """
    def clipped(b):
        return _clip(b["x0"], b["y0"], b["x1"], b["y1"], width, height)

    typed = [
        b for b in blocks
        if _is_signature_typed(b) and not _is_caption(b) and _intersect(window, clipped(b))
    ]
    if typed:
        chosen = [max(typed, key=lambda b: _w(b) * _h(b))]
    else:
        chosen = []
        for b in blocks:
            if b.get("type") not in ("text", "image") or _FOOTER.search(_content(b)):
                continue
            if _DOB.search(_content(b)):
                continue
            c = clipped(b)
            inter = _intersect(window, c)
            if _area(c) > 0 and inter and _area(inter) >= 0.6 * _area(c):
                chosen.append(b)
    if not chosen:
        return None

    x0, y0 = min(b["x0"] for b in chosen), min(b["y0"] for b in chosen)
    x1, y1 = max(b["x1"] for b in chosen), max(b["y1"] for b in chosen)
    ink = _clip(x0, y0, x1, y1, width, height)
    pad_x, pad_y = 0.4 * (x1 - x0), 0.35 * (y1 - y0)
    roi = _intersect(_clip(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, width, height), window)
    if roi is None:
        return None
    if box is not None:
        inside = _intersect(box, ink)
        if inside is not None and _area(inside) >= 0.9 * _area(ink):
            roi = _intersect(roi, box) or roi
    return roi


def _right_of_dob_roi(blocks, width, height) -> BBox | None:
    """Fallback when OCR lost the caption. On the layouts that occur, the
    signature sits right of the Date-of-Birth block and left of the photo."""
    dob = _dob_block(blocks)
    if dob is None:
        return None
    ref_h = _reference_line_height(blocks, dob, height)

    # OCR usually still reports *something* there (even if garbled), merging
    # the handwriting with the caption beneath it: treat it like a merged
    # caption block and drop the caption line by position.
    candidates = [
        b
        for b in blocks
        if b is not dob
        and b.get("type") == "text"
        and not _FOOTER.search(_content(b))
        and not _SERIAL.match(_content(b).strip())
        and _h(b) <= 2 * _w(b)  # not a vertical printed date at the card edge
        and b["x0"] >= dob["x1"] - 0.05 * _w(dob)
        and min(b["y1"], dob["y1"]) - max(b["y0"], dob["y0"]) > 0.3 * min(_h(b), _h(dob))
    ]
    if candidates:
        b = max(candidates, key=_w)
        cut = min(0.7 * ref_h, 0.4 * _h(b))
        return _clip(b["x0"], b["y0"] - 0.3 * ref_h, b["x1"], b["y1"] - cut, width, height)

    band_h = _h(dob)
    y0, y1 = dob["y0"] - 0.5 * band_h, dob["y1"] + 0.25 * band_h
    for other in blocks:
        if _FOOTER.search(_content(other)) and other["y0"] > dob["y0"]:
            y1 = min(y1, other["y0"] - 2)
    x1 = float(width)
    for other in blocks:
        if (other.get("type") or "") in ("image", "footer"):
            if other["x0"] >= dob["x1"] and other["y0"] < y1 and other["y1"] > y0:
                x1 = min(x1, other["x0"])
    if x1 <= dob["x1"]:
        return None
    return _clip(dob["x1"], y0, x1, y1, width, height)


def locate(
    image_bgr: np.ndarray, blocks, config: LocatorConfig = LocatorConfig()
) -> Located:
    height, width = image_bgr.shape[:2]
    captions = [b for b in blocks if _is_caption(b)]
    has_footer = any(_FOOTER.search(_content(b)) for b in blocks)

    # Type 1: a white box wins over everything else. Only looked for on
    # cards with no caption (or the newer-format footer), so glare on an
    # older card is not mistaken for one.
    if not captions or has_footer:
        box = _white_box_roi(image_bgr, blocks, has_footer)
        if box is not None:
            window, _ = _dob_footer_window(blocks, width, height)
            return Located(_handwriting_in(window, box, blocks, width, height) or box, TYPE1)
        if has_footer and not captions:
            window, bounded = _dob_footer_window(blocks, width, height)
            band = _band_roi(blocks, width, height)
            if band is not None:
                return Located(_handwriting_in(window, None, blocks, width, height) or band, TYPE1_BAND)

    # A typed block whose text is the caption itself spans handwriting AND
    # caption; it is handled as a merged caption block below, so the
    # caption is cut off by position instead of ending up in the output.
    typed = [b for b in blocks if _is_signature_typed(b) and not _is_caption(b)]
    if typed:
        b = max(typed, key=lambda b: b["y0"])
        return Located(_clip(b["x0"], b["y0"], b["x1"], b["y1"], width, height), TYPED)

    if captions:
        caption = max(captions, key=lambda b: b["y0"])
        pan_type = TYPE3 if _CAPTION_HI in _content(caption) else TYPE2
        return Located(_above_caption_roi(caption, blocks, width, height, config), pan_type)

    roi = _right_of_dob_roi(blocks, width, height)
    if roi is not None:
        return Located(roi, RIGHT_OF_DOB)

    x0, y0, x1, y1 = config.default_roi_fraction
    return Located(_clip(x0 * width, y0 * height, x1 * width, y1 * height, width, height), DEFAULT)
