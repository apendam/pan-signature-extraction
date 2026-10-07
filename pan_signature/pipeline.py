"""End-to-end: image on disk -> OCR (Document AI or Mistral) -> ROI -> OpenCV crop."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .config import (
    DocAIConfig,
    ExtractorConfig,
    LocatorConfig,
    MistralConfig,
    docai_config_from_env,
    mistral_config_from_env,
)
from .docai_client import DocumentAIClient
from .mistral_client import MistralOCRClient
from .mistral_signature_locator import locate_signature_region_mistral_with_source
from . import pan_locator
from .mistral_signature_locator import _get as _field
from .orientation import (
    detect_rotation,
    rotate_blocks,
    rotate_image,
    rotated_size,
)
from .signature_extractor import (
    binarize,
    orient_horizontal,
    refine_and_crop,
    whiten_background,
)
from .signature_locator import BBox, locate_signature_region

_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _mime_type_for(path: Path) -> str:
    try:
        return _MIME_TYPES[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported image extension: {path.suffix}") from exc


def extract_signature(
    image_path: str,
    provider: str = "docai",
    docai_config: DocAIConfig | None = None,
    mistral_config: MistralConfig | None = None,
    locator_config: LocatorConfig = LocatorConfig(),
    extractor_config: ExtractorConfig = ExtractorConfig(),
    debug_out_path: str | None = None,
) -> tuple[np.ndarray, BBox]:
    """Returns (signature_image, bbox_in_original_image). `bbox_in_original_image`
    is in the source image's coordinate frame regardless of any whitening/
    rotation `extractor_config` applies to `signature_image` itself."""
    path = Path(image_path)
    image_bytes = path.read_bytes()
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image_bgr.shape[:2]

    if provider == "docai":
        client = DocumentAIClient(docai_config or docai_config_from_env())
        document = client.process_image_bytes(
            image_bytes, mime_type=_mime_type_for(path)
        )
        roi = locate_signature_region(document, width, height, locator_config)
    elif provider == "mistral":
        client = MistralOCRClient(mistral_config or mistral_config_from_env())
        ocr_response = client.process_image_bytes(
            image_bytes, mime_type=_mime_type_for(path)
        )
        roi, source = locate_signature_region_mistral_with_source(
            ocr_response, width, height, locator_config
        )
        if source != "default":
            # The caption is already excluded by position; stripping blue
            # ink here would also erase blue-pen signatures.
            extractor_config = replace(
                extractor_config, exclude_blue_ink=False, local_contrast_ink=True
            )
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'docai' or 'mistral'.")

    signature_crop, refined_bbox = refine_and_crop(image_bgr, roi, extractor_config)

    if debug_out_path:
        debug_image = image_bgr.copy()
        cv2.rectangle(debug_image, (roi.x0, roi.y0), (roi.x1, roi.y1), (0, 165, 255), 2)
        cv2.rectangle(
            debug_image,
            (refined_bbox.x0, refined_bbox.y0),
            (refined_bbox.x1, refined_bbox.y1),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(debug_out_path, debug_image)

    if extractor_config.black_and_white:
        signature_crop = binarize(signature_crop, extractor_config)
    elif extractor_config.whiten_background:
        signature_crop = whiten_background(signature_crop, extractor_config)
    if extractor_config.orient_horizontal:
        signature_crop = orient_horizontal(
            signature_crop, extractor_config.rotate_direction
        )

    return signature_crop, refined_bbox


# --------------------------------------------------------------------------
# Type-aware path (Mistral): upright -> locate by PAN type -> crop -> B&W -> QC
# --------------------------------------------------------------------------

OK = "ok"
NOT_READABLE = "not_readable"

# Quality gates on the final black-and-white crop / region.
_MAX_ROI_AREA_FRACTION = 0.25  # region bigger than this => the locator got lost
_MIN_INK_FRACTION = 0.003  # below this the box is blank / nothing legible
_MAX_INK_FRACTION = 0.35  # above this it is texture or text, not a signature


@dataclass
class SignatureResult:
    status: str  # OK or NOT_READABLE
    pan_type: str  # see pan_locator (TYPE1/TYPE2/TYPE3/...), or "unknown"
    rotation: int | None  # clockwise degrees applied to make the card upright
    reason: str  # why it was flagged; "" when OK
    crop: np.ndarray | None = None
    bbox: BBox | None = None  # in the upright (rotated) image's coordinates
    roi: BBox | None = None


def normalize_blocks(ocr_response, width: int, height: int) -> list[dict]:
    """OCR blocks as plain dicts in the image's pixel coordinates."""
    pages = _field(ocr_response, "pages") or []
    if not pages:
        return []
    page = pages[0]
    dims = _field(page, "dimensions")
    ocr_w = _field(dims, "width") if dims is not None else None
    ocr_h = _field(dims, "height") if dims is not None else None
    sx = width / ocr_w if ocr_w else 1.0
    sy = height / ocr_h if ocr_h else 1.0

    blocks = []
    for raw in _field(page, "blocks") or []:
        coords = [
            _field(raw, n)
            for n in ("top_left_x", "top_left_y", "bottom_right_x", "bottom_right_y")
        ]
        if None in coords:
            continue
        if max(coords) <= 1.0:  # normalized fractions
            x0, y0, x1, y1 = (coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height)
        else:
            x0, y0, x1, y1 = coords[0] * sx, coords[1] * sy, coords[2] * sx, coords[3] * sy
        blocks.append(
            {
                "type": _field(raw, "type") or "",
                "content": _field(raw, "content") or "",
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            }
        )
    return blocks


def _flag(pan_type, rotation, reason, roi=None) -> SignatureResult:
    return SignatureResult(NOT_READABLE, pan_type, rotation, reason, roi=roi)


def _call_ocr(client, image_bgr, path: Path):
    """Send the image; if the API rejects the file (e.g. a truncated JPEG),
    retry once with a clean re-encode of what OpenCV could decode."""
    try:
        return client.process_image_bytes(path.read_bytes(), mime_type=_mime_type_for(path))
    except Exception as exc:
        if "400" not in str(exc):
            raise
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise
        return client.process_image_bytes(encoded.tobytes(), mime_type="image/jpeg")


def extract_signature_report(
    image_path: str,
    ocr_response=None,
    mistral_config: MistralConfig | None = None,
    locator_config: LocatorConfig = LocatorConfig(),
    extractor_config: ExtractorConfig = ExtractorConfig(),
    debug_out_path: str | None = None,
) -> SignatureResult:
    """Full type-aware extraction for one PAN image via Mistral OCR.

    Pass `ocr_response` (a response object or its dict form) to reuse a
    cached OCR result instead of calling the API. Never raises for a bad
    card: returns status NOT_READABLE with a reason so callers can report
    it instead of saving a wrong crop.
    """
    path = Path(image_path)
    image = cv2.imread(str(path))
    if image is None:
        return _flag("unknown", None, "image could not be decoded")

    if ocr_response is None:
        client = MistralOCRClient(mistral_config or mistral_config_from_env())
        ocr_response = _call_ocr(client, image, path)

    height, width = image.shape[:2]
    blocks = normalize_blocks(ocr_response, width, height)
    if len(blocks) <= 2:
        return _flag("unknown", None, "OCR returned too little structure to locate anything")

    rotation = detect_rotation(blocks, width, height)
    if rotation is None:
        texts = [b for b in blocks if b["type"] in ("text", "title") and len(b["content"].strip()) > 3]
        tall = sum(1 for b in texts if (b["y1"] - b["y0"]) > 1.5 * (b["x1"] - b["x0"]))
        if texts and tall / len(texts) > 0.5:
            return _flag("unknown", None, "card is sideways and its orientation could not be determined")
        rotation = 0

    upright = rotate_image(image, rotation)
    blocks = rotate_blocks(blocks, rotation, width, height)

    located = pan_locator.locate(upright, blocks, locator_config)
    roi = located.roi
    uh, uw = upright.shape[:2]
    if (roi.x1 - roi.x0) * (roi.y1 - roi.y0) > _MAX_ROI_AREA_FRACTION * uw * uh:
        return _flag(located.pan_type, rotation, "could not isolate a signature region", roi)
    if located.pan_type == pan_locator.DEFAULT:
        return _flag(located.pan_type, rotation, "no caption, white box or signature block found", roi)

    # The ROI already excludes the caption by position, so colour filtering
    # (which would erase blue-pen signatures) is turned off; ink is picked
    # out by local contrast instead.
    config = replace(extractor_config, exclude_blue_ink=False, local_contrast_ink=True)
    crop, bbox = refine_and_crop(upright, roi, config)

    if debug_out_path:
        debug = upright.copy()
        cv2.rectangle(debug, (roi.x0, roi.y0), (roi.x1, roi.y1), (0, 165, 255), 2)
        cv2.rectangle(debug, (bbox.x0, bbox.y0), (bbox.x1, bbox.y1), (0, 255, 0), 2)
        cv2.imwrite(debug_out_path, debug)

    final = binarize(crop, config) if config.black_and_white else crop
    ink = float((final == 0).all(axis=2).mean()) if config.black_and_white else 1.0
    if config.black_and_white and ink < _MIN_INK_FRACTION:
        return _flag(located.pan_type, rotation, "no legible signature (blank or too faint)", roi)
    if config.black_and_white and ink > _MAX_INK_FRACTION:
        return _flag(located.pan_type, rotation, "region is mostly texture/text, not a signature", roi)

    if config.orient_horizontal:
        final = orient_horizontal(final, config.rotate_direction)
    return SignatureResult(OK, located.pan_type, rotation, "", final, bbox, roi)
