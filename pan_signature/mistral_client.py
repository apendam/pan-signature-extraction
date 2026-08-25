"""Thin wrapper around the Mistral OCR API's ``ocr.process`` call.

Note on scope: Mistral's OCR block classification (`include_blocks=True`,
the default) is documented as being able to tag a content block's `type`
as `"signature"` directly, with its own bounding box. In practice, tested
against a real Indian PAN card with both `model="mistral-ocr-latest"` and
the explicit `"mistral-ocr-4-1"`, no block ever came back typed that way
-- the signature and a printed caption next to it were merged into one
ordinary `"text"` block instead. So this path still needs an anchor-style
fallback, same spirit as Document AI's OCR-anchor heuristic (see
`docai_client.py`). See `mistral_signature_locator.py` for exactly how
that's handled.
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
