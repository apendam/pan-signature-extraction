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
    anchor_keywords: tuple = ("signature", "sign")
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
