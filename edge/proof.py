from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def scale_roi_px(
    roi_px: tuple[int, int, int, int],
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map an AI-frame pixel ROI onto a (usually larger) evidence still."""
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return roi_px
    sx = dst_w / src_w
    sy = dst_h / src_h
    x1, y1, x2, y2 = roi_px
    nx1 = int(round(x1 * sx))
    ny1 = int(round(y1 * sy))
    nx2 = int(round(x2 * sx))
    ny2 = int(round(y2 * sy))
    nx1 = max(0, min(max(0, dst_w - 1), nx1))
    ny1 = max(0, min(max(0, dst_h - 1), ny1))
    nx2 = max(nx1 + 1, min(dst_w, nx2))
    ny2 = max(ny1 + 1, min(dst_h, ny2))
    return nx1, ny1, nx2, ny2


def save_proof(
    frame: np.ndarray,
    roi_px: tuple[int, int, int, int],
    stamp: datetime,
    proofs_root: Path,
    kind: str = "abandoned",
) -> Path:
    annotated = frame.copy()
    x1, y1, x2, y2 = roi_px
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 80, 255), 2)
    label = stamp.strftime("%Y-%m-%d %H:%M:%S")
    title = f"{kind.upper()}  {label}"
    cv2.rectangle(annotated, (8, 8), (8 + 18 * len(title), 42), (10, 10, 10), -1)
    cv2.putText(
        annotated,
        title,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    day_dir = proofs_root / stamp.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{kind}_{stamp.strftime('%H%M%S')}.jpg"
    if path.exists():
        path = day_dir / f"{kind}_{stamp.strftime('%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), annotated)
    return path
