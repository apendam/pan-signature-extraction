"""Runtime configuration, loaded from environment variables.

Document AI credentials are never hard-coded; they come from Application
Default Credentials (``gcloud auth application-default login`` or a
service-account key referenced by ``GOOGLE_APPLICATION_CREDENTIALS``).
"""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocAIConfig:
    project_id: str
    location: str
    processor_id: str


@dataclass(frozen=True)
class LocatorConfig:
    # Text tokens that, if found on the page, anchor the signature ROI.
    # "हस्ताक्षर" (Hindi for "signature") is printed next to the English
    # caption on real Indian PAN cards -- confirmed against a real sample.
    anchor_keywords: tuple = ("signature", "sign", "हस्ताक्षर")
    # Fallback ROI as (x0, y0, x1, y1) fractions of the page, used when no
    # anchor text is found. Tuned for the bottom-left signature box common
    # on Indian PAN cards; recalibrate against your own samples with
    # `scripts/extract_signature.py --debug`.
    default_roi_fraction: tuple = (0.03, 0.55, 0.55, 0.95)
    # How far below/right of an anchor token the ROI extends, as fractions
    # of page height/width.
    anchor_margin_below: float = 0.22
    anchor_margin_right: float = 0.35


@dataclass(frozen=True)
class ExtractorConfig:
    min_component_area_fraction: float = 0.0006
    padding_px: int = 6
    # Real Indian PAN cards print field captions (e.g. "Signature") in
    # blue and handwritten/entered values in black -- excluding blue ink
    # before thresholding keeps a caption sitting right next to the
    # signature out of the crop. Confirmed against a real sample; harmless
    # no-op on grayscale scans since desaturated pixels never match.
    exclude_blue_ink: bool = True


@dataclass(frozen=True)
class MistralConfig:
    api_key: str
    model: str


def docai_config_from_env() -> DocAIConfig:
    missing = [
        name
        for name in ("GOOGLE_CLOUD_PROJECT", "DOCAI_PROCESSOR_ID")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. See .env.example."
        )
    return DocAIConfig(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("DOCAI_LOCATION", "us"),
        processor_id=os.environ["DOCAI_PROCESSOR_ID"],
    )


def mistral_config_from_env() -> MistralConfig:
    missing = [name for name in ("MISTRAL_API_KEY",) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. See .env.example."
        )
    return MistralConfig(
        api_key=os.environ["MISTRAL_API_KEY"],
        model=os.environ.get("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
    )
