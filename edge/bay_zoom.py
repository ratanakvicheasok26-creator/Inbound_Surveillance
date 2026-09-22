"""SAHI-style zoom: run YOLO-pose on empty vehicle-bay crops.

Full-frame SAHI tiles the whole shop. This only zooms bays that currently have
no accepted person, remaps boxes and keypoints back to frame coordinates, then
applies the same kinematic gates as a normal pass.
"""

from __future__ import annotations

from occupancy import clamp_roi, detection_in_bay, roi_to_pixels
from person import SHORTCUT_MIN_CONF, Detection, extract_keypoints, is_human_pose

DEFAULT_BAY_ZOOM_PAD = 0.08
DEFAULT_MERGE_IOU = 0.5
_MIN_CROP_PX = 8
# Zoom inflates engine-bay FPs; never run the crop pass below this floor.
BAY_ZOOM_MIN_CONF = min(SHORTCUT_MIN_CONF, 0.45)


def occupancy_hints(snapshots) -> dict[str, dict]:
    """Map bay_id -> session_open / vehicle_present from the last occupancy tick."""
    hints: dict[str, dict] = {}
    for snap in snapshots or []:
        bay_id = getattr(snap, "bay_id", None)
        if not bay_id:
            continue
        hints[str(bay_id)] = {
            "session_open": bool(getattr(snap, "session_open", False)),
            "vehicle_present": bool(getattr(snap, "vehicle_present", False)),
        }
    return hints


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 1e-6) * max(ay2 - ay1, 1e-6)
    area_b = max(bx2 - bx1, 1e-6) * max(by2 - by1, 1e-6)
    return inter / (area_a + area_b - inter + 1e-6)


def box_containment(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Fraction of the smaller box that is inside the larger box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 1e-6) * max(ay2 - ay1, 1e-6)
    area_b = max(bx2 - bx1, 1e-6) * max(by2 - by1, 1e-6)
    return inter / min(area_a, area_b)


def bay_crop_xyxy(
    bay: dict,
    frame_w: int,
    frame_h: int,
    pad: float = DEFAULT_BAY_ZOOM_PAD,
) -> tuple[int, int, int, int]:
    """Axis-aligned crop of a bay polygon (or ROI) with fractional padding."""
    xs: list[float] = []
    ys: list[float] = []
    polygon = bay.get("polygon") if isinstance(bay, dict) else None
    if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
        for pt in polygon:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    xs.append(float(pt[0]) * frame_w)
                    ys.append(float(pt[1]) * frame_h)
                except (TypeError, ValueError):
                    continue
    if len(xs) < 2:
        roi = list(bay.get("roi") or [0.30, 0.20, 0.40, 0.60]) if isinstance(bay, dict) else [0.30, 0.20, 0.40, 0.60]
        x1, y1, x2, y2 = roi_to_pixels(frame_w, frame_h, clamp_roi(roi))
        xs, ys = [float(x1), float(x2)], [float(y1), float(y2)]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    pad_x = max(0.0, float(pad)) * max(x2 - x1, 1.0)
    pad_y = pad * max(y2 - y1, 1.0)
    x1i = int(max(0, x1 - pad_x))
    y1i = int(max(0, y1 - pad_y))
    x2i = int(min(frame_w, x2 + pad_x))
    y2i = int(min(frame_h, y2 + pad_y))
    if x2i - x1i < _MIN_CROP_PX or y2i - y1i < _MIN_CROP_PX:
        return 0, 0, 0, 0
    return x1i, y1i, x2i, y2i


def remap_keypoints(
    keypoints: list,
    crop_xyxy: tuple[int, int, int, int],
) -> list[tuple[float, float, float]]:
    ox, oy = float(crop_xyxy[0]), float(crop_xyxy[1])
    out: list[tuple[float, float, float]] = []
    for pt in keypoints or []:
        if pt is None or len(pt) < 2:
            out.append((0.0, 0.0, 0.0))
            continue
        conf = float(pt[2]) if len(pt) >= 3 else 1.0
        out.append((float(pt[0]) + ox, float(pt[1]) + oy, conf))
    return out


def remap_detection(det: Detection, crop_xyxy: tuple[int, int, int, int]) -> Detection:
    ox, oy = float(crop_xyxy[0]), float(crop_xyxy[1])
    out = Detection(
        det.x1 + ox,
        det.y1 + oy,
        det.x2 + ox,
        det.y2 + oy,
        det.conf,
        remap_keypoints(det.keypoints, crop_xyxy),
    )
    out.accepted = det.accepted
    out.track_id = det.track_id
    out.identity = det.identity
    out.identity_conf = det.identity_conf
    out.is_staff = det.is_staff
    return out


def suppress_nested_boxes(
    dets: list[Detection],
    containment_threshold: float = 0.75,
    iou_threshold: float = 0.40,
) -> list[Detection]:
    """Suppress nested duplicate boxes (e.g. torso inside full body)."""
    if len(dets) <= 1:
        return list(dets)
    sorted_dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: list[Detection] = []
    for det in sorted_dets:
        is_nested = False
        for existing in kept:
            iou = box_iou(det.box(), existing.box())
            cont = box_containment(det.box(), existing.box())
            if cont >= containment_threshold or iou >= iou_threshold:
                is_nested = True
                break
        if not is_nested:
            kept.append(det)
    return kept


def merge_detections(
    full_accepted: list[Detection],
    full_rejected: list[Detection],
    zoom_accepted: list[Detection],
    zoom_rejected: list[Detection],
    iou_threshold: float = DEFAULT_MERGE_IOU,
    containment_threshold: float = 0.70,
) -> tuple[list[Detection], list[Detection]]:
    """Merge full-frame detections and zoom detections.
    
    If a zoom detection overlaps an existing full-frame detection, prefer the zoom
    detection if it has higher confidence or richer keypoints (since zoom crops offer
    higher spatial resolution). Add non-overlapping zoom boxes.
    """
    merged_accepted = list(full_accepted)
    for det in zoom_accepted:
        matched_idx = -1
        best_overlap = 0.0
        for idx, other in enumerate(merged_accepted):
            iou = box_iou(det.box(), other.box())
            cont = box_containment(det.box(), other.box())
            overlap = max(iou, cont)
            if (iou >= iou_threshold or cont >= containment_threshold) and overlap > best_overlap:
                best_overlap = overlap
                matched_idx = idx

        if matched_idx >= 0:
            existing = merged_accepted[matched_idx]
            existing_kpts_vis = sum(1 for pt in existing.keypoints if len(pt) >= 3 and pt[2] >= 0.25) if existing.keypoints else 0
            zoom_kpts_vis = sum(1 for pt in det.keypoints if len(pt) >= 3 and pt[2] >= 0.25) if det.keypoints else 0
            if det.conf >= existing.conf or zoom_kpts_vis > existing_kpts_vis + 2:
                det.track_id = det.track_id or existing.track_id
                det.identity = det.identity or existing.identity
                det.is_staff = det.is_staff or existing.is_staff
                merged_accepted[matched_idx] = det
        else:
            merged_accepted.append(det)

    merged_rejected = list(full_rejected)
    for det in zoom_rejected:
        if any(box_iou(det.box(), other.box()) >= iou_threshold for other in merged_accepted):
            continue
        if any(box_iou(det.box(), other.box()) >= iou_threshold for other in full_rejected):
            continue
        merged_rejected.append(det)
    return suppress_nested_boxes(merged_accepted), merged_rejected


def candidate_vehicle_bays(
    bays: list[dict],
    accepted: list[Detection],
    frame_w: int,
    frame_h: int,
    kpt_conf: float,
    occupancy_by_id: dict[str, dict] | None = None,
    only_empty: bool = False,
) -> list[dict]:
    """Vehicle bays to zoom.
    
    If only_empty is True, bays with an already-accepted person are skipped.
    In active production (only_empty=False), bays with active vehicles, open sessions,
    or configured vehicle bays are zoomed so all mechanics (foreground, occluded, or background)
    are detected simultaneously.
    """
    out: list[dict] = []
    for bay in bays or []:
        if not isinstance(bay, dict):
            continue
        bay_type = str(bay.get("type") or "vehicle_bay").strip() or "vehicle_bay"
        if bay_type != "vehicle_bay":
            continue
        if only_empty and any(detection_in_bay(det, bay, frame_w, frame_h, kpt_conf) for det in accepted):
            continue
        if occupancy_by_id is not None:
            hint = occupancy_by_id.get(str(bay.get("id") or ""), {})
            has_activity = bool(hint.get("session_open") or hint.get("vehicle_present"))
            has_person = any(detection_in_bay(det, bay, frame_w, frame_h, kpt_conf) for det in accepted)
            if not has_activity and not has_person:
                continue
        out.append(bay)
    return out


def empty_vehicle_bays(
    bays: list[dict],
    accepted: list[Detection],
    frame_w: int,
    frame_h: int,
    kpt_conf: float,
    occupancy_by_id: dict[str, dict] | None = None,
    only_empty: bool = True,
) -> list[dict]:
    """Vehicle bays with no accepted person. Kept for backwards compatibility."""
    return candidate_vehicle_bays(
        bays,
        accepted,
        frame_w,
        frame_h,
        kpt_conf,
        occupancy_by_id=occupancy_by_id,
        only_empty=only_empty,
    )


def detections_from_result(result, conf_min: float) -> list[Detection]:
    """Boxes + keypoints from a YOLO result, before kinematic filtering."""
    dets: list[Detection] = []
    if result is None or getattr(result, "boxes", None) is None:
        return dets
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        dets.append(Detection(x1, y1, x2, y2, conf, extract_keypoints(result, i)))
    return dets


def _filter_pose(
    dets: list[Detection],
    frame_h: int,
    min_height_frac: float,
    min_aspect: float,
    min_keypoints: int,
    kpt_conf: float,
) -> tuple[list[Detection], list[Detection]]:
    accepted: list[Detection] = []
    rejected: list[Detection] = []
    for det in dets:
        det.accepted = is_human_pose(
            det.x1,
            det.y1,
            det.x2,
            det.y2,
            det.keypoints,
            frame_h,
            min_height_frac=min_height_frac,
            min_aspect=min_aspect,
            min_keypoints=min_keypoints,
            kpt_conf=kpt_conf,
            box_conf=det.conf,
        )
        (accepted if det.accepted else rejected).append(det)
    return accepted, rejected


def zoom_empty_bays(
    model,
    frame,
    bays: list[dict],
    accepted: list[Detection],
    rejected: list[Detection],
    *,
    imgsz: int,
    device,
    person_conf: float,
    min_height_frac: float,
    min_aspect: float,
    min_keypoints: int,
    kpt_conf: float,
    pad: float = DEFAULT_BAY_ZOOM_PAD,
    occupancy_by_id: dict[str, dict] | None = None,
    only_empty: bool = False,
) -> tuple[list[Detection], list[Detection]]:
    """Second-pass YOLO/TinyPose on vehicle bays.
    
    Zooms vehicle bays to resolve occluded or background workers.
    Occupancy hints skip inactive empty bays when vehicles are not present.
    """
    if model is None or frame is None or not bays:
        return accepted, rejected
    frame_h, frame_w = frame.shape[:2]
    targets = candidate_vehicle_bays(
        bays,
        accepted,
        frame_w,
        frame_h,
        kpt_conf,
        occupancy_by_id=occupancy_by_id,
        only_empty=only_empty,
    )
    if not targets:
        return accepted, rejected

    zoom_accepted: list[Detection] = []
    zoom_rejected: list[Detection] = []
    for bay in targets:
        crop_xyxy = bay_crop_xyxy(bay, frame_w, frame_h, pad=pad)
        x1, y1, x2, y2 = crop_xyxy
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) >= int(frame_w * 0.95) and (y2 - y1) >= int(frame_h * 0.95):
            continue
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            continue
        zoom_conf = max(float(person_conf), BAY_ZOOM_MIN_CONF)
        try:
            result = model.predict(
                crop,
                imgsz=imgsz,
                conf=zoom_conf,
                device=device,
                verbose=False,
            )[0]
        except Exception as exc:
            print(f"[BayZoom] Prediction error on {bay.get('id')}: {exc}")
            continue
        remapped = [remap_detection(det, crop_xyxy) for det in detections_from_result(result, zoom_conf)]
        acc, rej = _filter_pose(
            remapped,
            frame_h,
            min_height_frac,
            min_aspect,
            min_keypoints,
            kpt_conf,
        )
        zoom_accepted.extend(acc)
        zoom_rejected.extend(rej)
    return merge_detections(accepted, rejected, zoom_accepted, zoom_rejected)
