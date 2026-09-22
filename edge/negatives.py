"""Bank crops of tracks killed as inanimate, for later training.

The garage is full of objects nobody has seen the pose model hallucinate on
yet, and one demo video cannot cover them. Every track the liveness sweep kills
is a confirmed false positive with a free label, so writing the crop turns live
operation into a growing hard-negative set for a future fine-tune.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2

MAX_PER_DAY = 500
_MIN_CROP_PX = 16


def negatives_root(data_dir: Path) -> Path:
    return Path(data_dir) / "hard_negatives"


def _day_dir(data_dir: Path, stamp: datetime) -> Path:
    return negatives_root(data_dir) / stamp.strftime("%Y-%m-%d")


def bank_hard_negatives(
    frame,
    tracks: list,
    data_dir: Path,
    *,
    stamp: datetime | None = None,
    max_per_day: int = MAX_PER_DAY,
) -> list[Path]:
    """Write one JPEG + JSON sidecar per killed track. Never raises."""
    if frame is None or not tracks:
        return []
    stamp = stamp or datetime.now()
    written: list[Path] = []
    try:
        day = _day_dir(data_dir, stamp)
        day.mkdir(parents=True, exist_ok=True)
        if len(list(day.glob("*.jpg"))) >= max_per_day:
            return []
        h, w = frame.shape[:2]
        for trk in tracks:
            x1, y1, x2, y2 = (int(v) for v in trk.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
                continue
            base = f"{stamp.strftime('%H%M%S')}_t{trk.track_id}"
            jpg = day / f"{base}.jpg"
            if not cv2.imwrite(str(jpg), frame[y1:y2, x1:x2]):
                continue
            meta = {
                "track_id": trk.track_id,
                "captured_at": stamp.isoformat(timespec="seconds"),
                "bbox": [x1, y1, x2, y2],
                "conf": round(float(trk.conf), 4),
                "hits": int(trk.hits),
                "motion": round(float(trk.motion), 4),
                "jitter": None if trk.jitter is None else round(float(trk.jitter), 5),
                "liveness": None if trk.liveness is None else round(float(trk.liveness), 4),
                "keypoints": [[round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 3)] for p in (trk.keypoints or [])],
                "reason": "inanimate",
            }
            (day / f"{base}.json").write_text(json.dumps(meta, indent=2))
            written.append(jpg)
    except Exception as exc:
        print(f"[Negatives] Skipped banking ({exc})")
    return written
