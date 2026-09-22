from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VEHICLE_COCO_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class VehicleDetection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    vehicle_type: str  # 'car', 'truck', 'bus', 'motorcycle'
    vehicle_id: str = ""
    color_name: str = "Vehicle"
    active_technician: str | None = None
    wrench_seconds: float = 0.0

    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def roi(self, frame_w: int, frame_h: int) -> list[float]:
        w = max(frame_w, 1)
        h = max(frame_h, 1)
        return [
            round(self.x1 / w, 4),
            round(self.y1 / h, 4),
            round((self.x2 - self.x1) / w, 4),
            round((self.y2 - self.y1) / h, 4),
        ]

    def as_dict(self, frame_w: int, frame_h: int) -> dict[str, Any]:
        return {
            "id": self.vehicle_id,
            "type": self.vehicle_type,
            "label": f"{self.vehicle_type.capitalize()} #{self.vehicle_id[-4:] if len(self.vehicle_id) >= 4 else self.vehicle_id}",
            "conf": round(self.conf, 2),
            "roi": self.roi(frame_w, frame_h),
            "wrench_seconds": round(self.wrench_seconds, 1),
            "technician": self.active_technician,
        }


@dataclass
class VehicleTrack:
    vehicle_id: str
    vehicle_type: str
    last_roi: list[float]
    first_seen_t: float
    last_seen_t: float
    is_present: bool = True
    job_id: str = ""
    accumulated_wrench_seconds: float = 0.0
    primary_technician: str | None = None


class VehicleTracker:
    """Tracks active vehicles and detects when a vehicle leaves/departs the garage."""

    def __init__(self, departure_grace_seconds: float = 15.0) -> None:
        self.departure_grace_seconds = float(departure_grace_seconds)
        self.tracks: dict[str, VehicleTrack] = {}

    def update(
        self,
        detections: list[VehicleDetection],
        now: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[list[VehicleTrack], list[str]]:
        """
        Update vehicle tracks and detect departures.
        Returns (active_tracks, newly_departed_vehicle_ids).
        """
        seen_ids = set()
        for v in detections:
            v_id = v.vehicle_id
            seen_ids.add(v_id)
            roi = v.roi(frame_w, frame_h)
            if v_id in self.tracks:
                track = self.tracks[v_id]
                track.last_seen_t = now
                track.last_roi = roi
                track.is_present = True
            else:
                self.tracks[v_id] = VehicleTrack(
                    vehicle_id=v_id,
                    vehicle_type=v.vehicle_type,
                    last_roi=roi,
                    first_seen_t=now,
                    last_seen_t=now,
                    is_present=True,
                )

        departed_ids: list[str] = []
        for v_id, track in list(self.tracks.items()):
            if v_id not in seen_ids and track.is_present:
                if (now - track.last_seen_t) >= self.departure_grace_seconds:
                    track.is_present = False
                    departed_ids.append(v_id)

        active_tracks = [t for t in self.tracks.values() if t.is_present]
        return active_tracks, departed_ids


def extract_vehicle_detections(
    result: Any,
    frame_w: int,
    frame_h: int,
    conf_min: float = 0.20,
) -> list[VehicleDetection]:
    """Extract vehicle bounding boxes from standard YOLO inference result."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    data = getattr(boxes, "data", None)
    if data is None:
        return []

    rows = data.cpu().numpy() if hasattr(data, "cpu") else data
    vehicles: list[VehicleDetection] = []
    idx = 1
    for row in rows:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            int(row[5]),
        )
        if conf < conf_min:
            continue
        vtype = VEHICLE_COCO_CLASSES.get(cls_id)
        if not vtype:
            continue

        # Filter out tiny artifacts (< 3% of screen width or height)
        bw = x2 - x1
        bh = y2 - y1
        if bw < frame_w * 0.03 or bh < frame_h * 0.03:
            continue

        v_id = f"auto_{vtype}_{idx}"
        idx += 1
        vehicles.append(
            VehicleDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=conf,
                vehicle_type=vtype,
                vehicle_id=v_id,
            )
        )
    return vehicles
