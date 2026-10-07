#!/usr/bin/env python3
"""Batch: extract signatures from many PAN images (Mistral OCR).

Usage:
    python scripts/batch_extract.py --images path/to/dir_or_file [more ...] \\
        --out-dir output/run1 [--ocr-cache-dir ocr_cache] [--limit 20]

For every image this writes ``<out-dir>/<stem>.sig.png`` (only when the
signature was found), ``<out-dir>/<stem>.debug.jpg`` (search region in
orange, final crop in green, on the upright card) and a single
``<out-dir>/report.csv`` with the PAN type, rotation, status and, for
cards that could not be read, the reason.

``--ocr-cache-dir`` stores each OCR response as JSON so re-runs (e.g. while
tuning thresholds) cost no API calls. Those files contain the text on the
cards: keep the directory out of version control.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from pan_signature.config import mistral_config_from_env
from pan_signature.mistral_client import MistralOCRClient
from pan_signature.pipeline import NOT_READABLE, OK, _call_ocr, extract_signature_report

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _collect(paths):
    files = []
    for raw in paths:
        p = Path(raw)
        files.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in _IMAGE_SUFFIXES) if p.is_dir() else [p])
    return files


def _ocr(path, client, cache_dir):
    cache_file = cache_dir / f"{path.stem}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())
    image = cv2.imread(str(path))
    for attempt in range(6):
        try:
            response = _call_ocr(client, image, path)
            break
        except Exception as exc:
            if "429" in str(exc):
                time.sleep(4 * (attempt + 1))
                continue
            raise
    else:
        raise RuntimeError("rate limit (429) persisted")
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if cache_file:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, default=str))
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ocr-cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.ocr_cache_dir) if args.ocr_cache_dir else None
    files = _collect(args.images)[: args.limit]
    client = None  # created lazily: a fully cached run needs no API key

    rows = []
    for path in files:
        try:
            cache_hit = cache_dir and (cache_dir / f"{path.stem}.json").exists()
            if not cache_hit and client is None:
                client = MistralOCRClient(mistral_config_from_env())
            ocr = _ocr(path, client, cache_dir)
            result = extract_signature_report(
                str(path), ocr_response=ocr, debug_out_path=str(out_dir / f"{path.stem}.debug.jpg")
            )
        except Exception as exc:  # keep going; report the card instead of aborting the batch
            rows.append({"image": path.name, "status": NOT_READABLE, "pan_type": "unknown", "rotation": "", "reason": f"error: {str(exc)[:100]}"})
            print(f"{path.name}: ERROR {exc}")
            continue
        if result.status == OK:
            cv2.imwrite(str(out_dir / f"{path.stem}.sig.png"), result.crop)
        rows.append({
            "image": path.name, "status": result.status, "pan_type": result.pan_type,
            "rotation": "" if result.rotation is None else result.rotation, "reason": result.reason,
        })
        print(f"{path.name}: {result.status} {result.pan_type} rot={result.rotation} {result.reason}")

    with (out_dir / "report.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "status", "pan_type", "rotation", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(r["status"] == OK for r in rows)
    print(f"\n{ok}/{len(rows)} extracted; report: {out_dir / 'report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
