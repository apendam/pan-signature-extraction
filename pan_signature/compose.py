"""Side-by-side review image: the PAN card on the left, the extracted
signature on the right (or the reason it could not be read)."""
from __future__ import annotations

import cv2
import numpy as np

_PANEL_HEIGHT = 560
_SIGNATURE_PANEL_WIDTH = 760
_MARGIN = 24


def _fit(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    scale = min(max_w / image.shape[1], max_h / image.shape[0])
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=interpolation,
    )


def make_pair(
    card_bgr: np.ndarray,
    signature_bgr: np.ndarray | None,
    caption: str,
    unreadable_reason: str = "",
) -> np.ndarray:
    """Card on the left, signature on the right, `caption` along the top.
    When `signature_bgr` is None the right panel states `unreadable_reason`."""
    card = _fit(card_bgr, 10_000, _PANEL_HEIGHT)
    body_h = _PANEL_HEIGHT + 2 * _MARGIN
    top = 44
    canvas = np.full(
        (top + body_h, card.shape[1] + _SIGNATURE_PANEL_WIDTH + 3 * _MARGIN, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(canvas, caption, (_MARGIN, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    canvas[top + _MARGIN : top + _MARGIN + card.shape[0], _MARGIN : _MARGIN + card.shape[1]] = card

    x0 = card.shape[1] + 2 * _MARGIN
    cv2.line(canvas, (x0 - _MARGIN // 2, top), (x0 - _MARGIN // 2, canvas.shape[0] - 8), (200, 200, 200), 1)
    if signature_bgr is not None:
        sig = _fit(signature_bgr, _SIGNATURE_PANEL_WIDTH, _PANEL_HEIGHT)
        y = top + _MARGIN + (_PANEL_HEIGHT - sig.shape[0]) // 2
        canvas[y : y + sig.shape[0], x0 : x0 + sig.shape[1]] = sig
    else:
        y = top + body_h // 2
        cv2.putText(canvas, "NOT READABLE", (x0, y - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2)
        cv2.putText(canvas, unreadable_reason[:70], (x0, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    return canvas
