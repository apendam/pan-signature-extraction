"""End-to-end: image on disk -> Document AI OCR -> ROI -> OpenCV crop."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import DocAIConfig, ExtractorConfig, LocatorConfig, docai_config_from_env
from .docai_client import DocumentAIClient
from .signature_extractor import refine_and_crop
from .signature_locator import BBox, locate_signature_region

_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def _mime_type_for(path: Path) -> str:
    try:
        return _MIME_TYPES[path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported image extension: {path.suffix}") from exc


def extract_signature(
    image_path: str,
    docai_config: DocAIConfig | None = None,
    locator_config: LocatorConfig = LocatorConfig(),
    extractor_config: ExtractorConfig = ExtractorConfig(),
    debug_out_path: str | None = None,
) -> tuple[np.ndarray, BBox]:
    path = Path(image_path)
    image_bytes = path.read_bytes()
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    height, width = image_bgr.shape[:2]

    client = DocumentAIClient(docai_config or docai_config_from_env())
    document = client.process_image_bytes(image_bytes, mime_type=_mime_type_for(path))

    roi = locate_signature_region(document, width, height, locator_config)
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

    return signature_crop, refined_bbox
