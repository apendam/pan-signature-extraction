#!/usr/bin/env python3
"""CLI: extract the signature crop from a PAN image.

Usage:
    python scripts/extract_signature.py --image sample.jpg --out signature.png
    python scripts/extract_signature.py --image sample.jpg --out signature.png --debug debug.jpg

Requires GOOGLE_CLOUD_PROJECT and DOCAI_PROCESSOR_ID (and optionally
DOCAI_LOCATION) in the environment -- see .env.example.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from pan_signature.pipeline import extract_signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to the PAN image")
    parser.add_argument("--out", required=True, help="Where to save the cropped signature")
    parser.add_argument(
        "--debug",
        default=None,
        help="Optional path to save a copy of the input with the ROI (orange) "
        "and refined crop (green) boxes drawn on it, for calibration",
    )
    args = parser.parse_args()

    crop, bbox = extract_signature(args.image, debug_out_path=args.debug)
    cv2.imwrite(args.out, crop)
    print(f"Saved signature crop to {args.out} (bbox={bbox.as_tuple()})")
    if args.debug:
        print(f"Saved debug overlay to {args.debug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
