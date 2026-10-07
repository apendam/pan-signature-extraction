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
    # Mistral path: PAN cards print the signature ABOVE its "Signature"
    # caption, so when the caption is its own OCR block the ROI is built
    # upward from the caption's top edge. Heights are in multiples of the
    # caption's line height; widths are fractions of the caption's width
    # (handwriting is often wider than the printed caption).
    caption_search_lines_above: float = 3.0
    caption_pad_left: float = 0.6
    caption_pad_right: float = 0.6
    # Upper bound on the printed caption's height (fraction of image
    # height) when trimming it off a merged signature+caption block.
    max_caption_height_fraction: float = 0.04
    # Keywords for the "Date of Birth" block, used as a fallback anchor
    # (the signature sits to its right on newer PAN layouts) when OCR
    # garbles or drops the signature caption.
    dob_keywords: tuple = ("date of birth", "जन्म")


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
    # Replace everything outside the detected ink strokes with solid
    # white, so the output isn't the original card's textured background.
    whiten_background: bool = True
    # Rotate a taller-than-wide crop 90 degrees so width > height (a
    # signature reads naturally landscape). No-op if already landscape.
    orient_horizontal: bool = True
    # Final output as pure black ink on a pure white background (instead
    # of keeping the ink's original color). Supersedes whiten_background.
    black_and_white: bool = True
    # How the black-and-white step keeps only the darkest, highest-contrast
    # strokes: a second threshold is taken between the coarse ink cut-off
    # (0.0) and the stricter "strong ink" cut-off (1.0). Lower keeps more
    # faint ink (and more caption/background); higher drops faint strokes.
    binarize_strictness: float = 0.6
    # Small crops are upscaled (cubic) before thresholding so thin strokes
    # on low-resolution cards stay separate instead of merging into
    # blobs. The scale is chosen to bring the long side near this many
    # pixels, between 1x and 8x.
    binarize_target_long_side: int = 1600
    # Hysteresis: besides the strong ink pixels (binarize_strictness), keep
    # weaker ink pixels that are connected to one of them. This repairs
    # broken/dashed strokes (the faint middle of a pen stroke) without
    # re-admitting isolated background. Same 0..1 scale as strictness;
    # must be <= binarize_strictness. Lower = bolder, more connected.
    binarize_weak_cutoff: float = 0.35
    # Light non-local-means denoise of the color crop before thresholding.
    binarize_denoise: bool = True
    # Close 1px gaps and smooth jagged edges of the final mask.
    binarize_smooth: bool = True
    # Thin the strokes by this many pixels per side (at the original crop's
    # resolution) for crisper, finer lines; thin strokes are kept connected
    # via their skeleton. 0 disables. Larger values give finer lines.
    binarize_thin_px: float = 0.5
    # Interior gaps smaller than this (original-resolution px^2) are filled
    # before thinning: marker ink often has a lighter centre that would
    # otherwise turn into a hollow outline. Real letter loops are larger.
    binarize_fill_holes_px: int = 60
    # ...and so are narrow slits (min width below this, in original-resolution
    # px), whatever their length.
    binarize_fill_slit_px: float = 3.0
    # Drop isolated specks smaller than this many pixels (measured at the
    # original crop's resolution, before upscaling).
    binarize_min_speck_px: int = 6
    # Pick out ink by local contrast (black-hat transform) instead of one
    # global Otsu threshold: thin dark/colored strokes stand out while
    # smooth card-background patches (holograms, gradients) do not. Used
    # when the ROI is already caption-free, so blue-pen signatures are kept.
    local_contrast_ink: bool = False
    # Which way to rotate when orient_horizontal kicks in. Confirmed
    # against a real sample scanned in portrait orientation (verified the
    # signature reads left-to-right afterward, not backwards); if your
    # source images are rotated the other way, flip this to "clockwise".
    rotate_direction: str = "counterclockwise"


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
