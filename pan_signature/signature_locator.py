"""Turn a Document AI OCR result into a pixel region of interest (ROI)
that should contain the signature.

Document AI's own "signature" entity type is presence-only (see
`docai_client.py`), so instead we anchor on ordinary OCR text tokens: if a
caption like "Signature" was recognized on the page, we build the ROI
relative to it; otherwise we fall back to a fixed fraction of the page
that you calibrate against your own document samples.

This module only reads plain attributes (`.pages`, `.tokens`,
`.layout.text_anchor`, `.layout.bounding_poly.normalized_vertices`,
`.text`) so it works against the real `documentai_v1.Document` proto or
against a lightweight stand-in with the same shape in tests.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import LocatorConfig


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            x0=max(0, min(self.x0, width)),
            y0=max(0, min(self.y0, height)),
            x1=max(0, min(self.x1, width)),
            y1=max(0, min(self.y1, height)),
        )

    def as_tuple(self):
        return (self.x0, self.y0, self.x1, self.y1)


def _token_text(document_text: str, token) -> str:
    segments = token.layout.text_anchor.text_segments
    parts = [
        document_text[int(seg.start_index or 0) : int(seg.end_index)]
        for seg in segments
    ]
    return "".join(parts)


def _normalized_bbox(token):
    vertices = token.layout.bounding_poly.normalized_vertices
    xs = [getattr(v, "x", 0.0) or 0.0 for v in vertices]
    ys = [getattr(v, "y", 0.0) or 0.0 for v in vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _find_anchor_bbox(document, keywords):
    """Return the normalized bbox of the lowest-on-page token whose text
    contains one of `keywords`, or None if no token matches."""
    best = None
    for page in document.pages:
        for token in page.tokens:
            text = _token_text(document.text, token).strip().lower()
            if not text:
                continue
            if any(keyword in text for keyword in keywords):
                bbox = _normalized_bbox(token)
                if best is None or bbox[1] > best[1]:
                    best = bbox
    return best


def locate_signature_region(
    document,
    image_width: int,
    image_height: int,
    config: LocatorConfig = LocatorConfig(),
) -> BBox:
    anchor = _find_anchor_bbox(document, config.anchor_keywords)
    if anchor is not None:
        nx0, ny0, nx1, ny1 = anchor
        roi_y1 = ny0
        roi_y0 = max(0.0, ny0 - config.anchor_margin_below)
        roi_x0 = max(0.0, nx0 - 0.05)
        roi_x1 = min(1.0, nx0 + config.anchor_margin_right)
    else:
        roi_x0, roi_y0, roi_x1, roi_y1 = config.default_roi_fraction

    return BBox(
        x0=round(roi_x0 * image_width),
        y0=round(roi_y0 * image_height),
        x1=round(roi_x1 * image_width),
        y1=round(roi_y1 * image_height),
    ).clip(image_width, image_height)
