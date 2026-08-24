"""Thin wrapper around the Document AI ``process_document`` RPC.

Note on scope: Document AI's built-in "signature" entity/visual-element
type only reports *whether* a signature is present (a "Detected" flag with
a confidence score) -- it does not return a bounding box, because the
model treats it as a derived, presence-only field. To get real
coordinates we run a plain OCR/Form Parser processor, keep the text
tokens' bounding boxes, and use them as anchors in
`signature_locator.py`. See the README for details.
"""
from __future__ import annotations

from google.cloud import documentai_v1 as documentai

from .config import DocAIConfig


class DocumentAIClient:
    def __init__(self, config: DocAIConfig):
        self._config = config
        opts = {"api_endpoint": f"{config.location}-documentai.googleapis.com"}
        self._client = documentai.DocumentProcessorServiceClient(client_options=opts)
        self._processor_name = self._client.processor_path(
            config.project_id, config.location, config.processor_id
        )

    def process_image_bytes(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> documentai.Document:
        raw_document = documentai.RawDocument(content=image_bytes, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=self._processor_name, raw_document=raw_document
        )
        result = self._client.process_document(request=request)
        return result.document
