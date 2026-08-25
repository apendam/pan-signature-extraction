"""Thin wrapper around the Mistral OCR API's ``ocr.process`` call.

Note on scope: unlike Document AI, whose "signature" field only reports
presence with no bounding box (see `docai_client.py`), Mistral's OCR block
classification (`include_blocks=True`, the default) can tag a content
block's `type` as `"signature"` directly, with its own bounding box --
so no OCR-anchor heuristic is needed on this path. See
`mistral_signature_locator.py` for the caveat on how firmly that response
schema is confirmed and how the bounding box is interpreted.
"""
from __future__ import annotations

import base64

from mistralai.client import Mistral
from mistralai.client.models import OCRResponse

from .config import MistralConfig


class MistralOCRClient:
    def __init__(self, config: MistralConfig):
        self._config = config
        self._client = Mistral(api_key=config.api_key)

    def process_image_bytes(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> OCRResponse:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        document = {
            "type": "image_url",
            "image_url": f"data:{mime_type};base64,{encoded}",
        }
        return self._client.ocr.process(
            model=self._config.model, document=document, include_blocks=True
        )
