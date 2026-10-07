"""Put a sideways / upside-down PAN card upright using OCR block geometry.

About a quarter of real-world PAN photos are taken rotated 90 degrees, and
every locator in this package assumes "above the caption" means "above in
the image". Rather than spend extra OCR calls on trial rotations, this
module uses two blocks that every PAN card has at a fixed relative spot:
the "INCOME TAX DEPARTMENT" title (top-left) and the "GOVT. OF INDIA" title
(top-right). The rotation that puts the first to the left of the second, on
roughly the same row, is the upright one.

Blocks are plain dicts: {"type", "content", "x0", "y0", "x1", "y1"}.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

_TITLE_RE = re.compile(r"income\s*tax|आयकर", re.I)
_GOVT_RE = re.compile(r"govt|सरकार", re.I)

# Clockwise degrees applied to the *image* to make the card upright.
ROTATIONS = (0, 90, 180, 270)


def box_of(block) -> tuple[float, float, float, float]:
    return block["x0"], block["y0"], block["x1"], block["y1"]


def rotate_box(box, rotation: int, width: int, height: int):
    """Map a box from original-image coordinates to the coordinates of the
    image after `rotation` degrees clockwise."""
    x0, y0, x1, y1 = box
    if rotation == 0:
        return x0, y0, x1, y1
    if rotation == 90:
        return height - y1, x0, height - y0, x1
    if rotation == 180:
        return width - x1, height - y1, width - x0, height - y0
    if rotation == 270:
        return y0, width - x1, y1, width - x0
    raise ValueError(f"rotation must be one of {ROTATIONS}, got {rotation}")


def rotated_size(width: int, height: int, rotation: int) -> tuple[int, int]:
    return (height, width) if rotation in (90, 270) else (width, height)


def rotate_image(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[rotation]
    return cv2.rotate(image, code)


def rotate_blocks(blocks, rotation: int, width: int, height: int):
    out = []
    for block in blocks:
        x0, y0, x1, y1 = rotate_box(box_of(block), rotation, width, height)
        out.append({**block, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return out


def _largest_match(blocks, pattern):
    matches = [b for b in blocks if pattern.search(b.get("content") or "")]
    if not matches:
        return None
    return max(matches, key=lambda b: (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))


def detect_rotation(blocks, width: int, height: int) -> int | None:
    """Clockwise rotation (0/90/180/270) that makes the card upright, or
    None when it can't be told (title blocks missing or merged into one)."""
    title = _largest_match(blocks, _TITLE_RE)
    govt = _largest_match(blocks, _GOVT_RE)
    if title is None or govt is None or title is govt:
        return None

    best_rotation, best_score = None, 0.0
    for rotation in ROTATIONS:
        tx0, ty0, tx1, ty1 = rotate_box(box_of(title), rotation, width, height)
        gx0, gy0, gx1, gy1 = rotate_box(box_of(govt), rotation, width, height)
        # Govt title should sit to the right of the department title and on
        # about the same row; penalise vertical offset heavily.
        score = ((gx0 + gx1) - (tx0 + tx1)) / 2 - 3 * abs((gy0 + gy1) - (ty0 + ty1)) / 2
        if score > best_score:
            best_rotation, best_score = rotation, score
    return best_rotation
