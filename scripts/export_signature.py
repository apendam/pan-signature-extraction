#!/usr/bin/env python3
"""Export extracted signatures (the *.sig.png files) as a transparent PNG
and an SVG.

Usage:
    python scripts/export_signature.py output/all200/234917.sig.png --out-dir output/exports
    python scripts/export_signature.py output/all200 --out-dir output/exports   # every *.sig.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from pan_signature.export import to_svg, to_transparent_png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a *.sig.png file, or a folder of them")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    files = sorted(source.glob("*.sig.png")) if source.is_dir() else [source]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in files:
        stem = path.name.removesuffix(".sig.png").removesuffix(".png")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"{path.name}: unreadable, skipped")
            continue
        cv2.imwrite(str(out / f"{stem}.signature.png"), to_transparent_png(image))
        (out / f"{stem}.signature.svg").write_text(to_svg(image))
        print(f"{stem}: wrote {stem}.signature.png and {stem}.signature.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
