"""Cross-check pose boxes against the vehicle detector already in the loop.

``yolo11n.pt`` runs every infer tick for cars and bikes (Dual Neural Network
Law). Its boxes are free evidence the pose model never sees: a "person" that
floats entirely inside a car with no hips or feet reaching past the bodywork is
an engine bay wearing a COCO skeleton.

The discriminator is where the legs end. A mechanic leaning over a fender still
stands on the floor, so their box drops to or below the vehicle's lower edge and
their hips sit outside it. A hallucination on the engine block does neither.
"""

from __future__ import annotations

from occupancy import is_sitting_pose, is_under_vehicle_pose
from person import L_ANKLE, L_HIP, R_ANKLE, R_HIP

DEFAULT_CONTAINMENT = 0.75
# The hallucination floats; a standing worker's box reaches the shop floor.
DEFAULT_FLOOR_MARGIN = 0.08
GROUNDING_POINTS = (L_HIP, R_HIP, L_ANKLE, R_ANKLE)


def box_containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0.0:
        return 0.0
    area = max(inner[2] - inner[0], 1e-6) * max(inner[3] - inner[1], 1e-6)
    return float(overlap / area)


def _grounded_outside(det, vehicle_box: tuple[float, float, float, float], kpt_conf: float) -> bool:
    """True when a visible hip or ankle reaches past the vehicle's bodywork."""
    vx1, vy1, vx2, vy2 = vehicle_box
    for idx in GROUNDING_POINTS:
        keypoints = getattr(det, "keypoints", None) or []
        if idx >= len(keypoints):
            continue
        x, y, c = keypoints[idx]
        if c < kpt_conf:
            continue
        if not (vx1 <= x <= vx2 and vy1 <= y <= vy2):
            return True
    return False


def is_vehicle_interior_ghost(
    det,
    vehicles: list,
    *,
    kpt_conf: float = 0.35,
    containment: float = DEFAULT_CONTAINMENT,
    floor_margin: float = DEFAULT_FLOOR_MARGIN,
    protected_ids: set[int] | None = None,
) -> bool:
    """A pose box swallowed by a vehicle with nothing reaching the floor."""
    if getattr(det, "is_staff", False):
        return False
    tid = getattr(det, "track_id", None)
    if protected_ids and tid is not None and int(tid) in protected_ids:
        return False
    kpts = getattr(det, "keypoints", None) or []
    if is_sitting_pose(kpts, kpt_conf):
        return False
    if is_under_vehicle_pose(kpts, kpt_conf * 0.85):
        return False
    box = det.box()
    for veh in vehicles or []:
        vbox = veh.box() if hasattr(veh, "box") else tuple(veh)
        if box_containment(box, vbox) < containment:
            continue
        veh_h = max(vbox[3] - vbox[1], 1e-6)
        if box[3] >= vbox[3] - floor_margin * veh_h:
            continue
        if _grounded_outside(det, vbox, kpt_conf):
            continue
        return True
    return False


def veto_vehicle_interior(
    accepted: list,
    vehicles: list,
    *,
    kpt_conf: float = 0.35,
    containment: float = DEFAULT_CONTAINMENT,
    floor_margin: float = DEFAULT_FLOOR_MARGIN,
    protected_ids: set[int] | None = None,
) -> tuple[list, list]:
    """Split accepted detections into (kept, vetoed) using vehicle geometry."""
    if not accepted or not vehicles:
        return list(accepted), []
    kept: list = []
    vetoed: list = []
    for det in accepted:
        if is_vehicle_interior_ghost(
            det,
            vehicles,
            kpt_conf=kpt_conf,
            containment=containment,
            floor_margin=floor_margin,
            protected_ids=protected_ids,
        ):
            det.accepted = False
            vetoed.append(det)
        else:
            kept.append(det)
    return kept, vetoed
