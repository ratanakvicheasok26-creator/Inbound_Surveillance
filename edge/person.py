"""Accept a detection as a person only when it has a plausible body skeleton.

YOLO-pose will happily emit 17 noisy keypoints on shoes, bags, jackets, and
shop-floor machinery (open engines, motorcycle frames). A connected keypoint
graph is not proof of a human: joints must spread like a body tree, hood-lean
heads sit above the shoulders, and low-confidence boxes cannot use the
upper-body shortcuts that catch far / under-car workers at high confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from occupancy import box_center_in_roi

# COCO-pose indices used by Ultralytics YOLO-pose.
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

TORSO = (NOSE, L_SHOULDER, R_SHOULDER)
TORSO_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
HEAD_POINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
FACE = HEAD_POINTS
FACE_CORE = (NOSE, L_EYE, R_EYE)
LEG_POINTS = (L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
HIP_POINTS = (L_HIP, R_HIP)

# Biologically connected bones. Disconnected floating points are not a body.
KINEMATIC_EDGES = (
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
)
TORSO_EDGES = (
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
)
LEG_EDGES = (
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
    (L_HIP, R_HIP),
)

SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

Keypoint = tuple[float, float, float]

# Hood-lean / head+shoulder shortcuts need this floor. Real workers in the
# garage CCTV frames sat at 0.61–0.84; engine/bike FPs sat at 0.36–0.47.
SHORTCUT_MIN_CONF = 0.50
CLUSTER_SPAN_FRAC = 0.32
# Laptop webcam close-ups fill the frame. An engine/bike blob in a garage
# shot is a small box with three shiny "eyes" and must not use this path.
CLOSEUP_MIN_HEIGHT_FRAC = 0.32
CLOSEUP_MIN_ASPECT = 0.28


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    keypoints: list[Keypoint] = field(default_factory=list)
    accepted: bool = False
    track_id: int | None = None
    identity: str | None = None
    identity_conf: float = 0.0
    is_staff: bool = False
    reid_feat: np.ndarray | None = None
    active_time_str: str | None = None
    bay_name: str | None = None
    coasting: bool = False
    liveness: float | None = None
    hits: int | None = None
    clutter: bool = False
    motion: float | None = None
    jitter: float | None = None
    # None until occupancy resolves the occupant; False means the time on this
    # box is provisional and has not been billed to anyone.
    verified: bool | None = None
    confirmed: bool = True

    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def in_roi(self, roi_px: tuple[int, int, int, int], kpt_conf: float = 0.4) -> bool:
        if box_center_in_roi(self.box(), roi_px):
            return True
        rx1, ry1, rx2, ry2 = roi_px
        hits = 0
        for x, y, c in self.keypoints:
            if c >= kpt_conf and rx1 <= x <= rx2 and ry1 <= y <= ry2:
                hits += 1
        return hits >= 2


def _as_xyconf(pt) -> Keypoint:
    vals = pt.tolist() if hasattr(pt, "tolist") else list(pt)
    if len(vals) >= 3:
        return float(vals[0]), float(vals[1]), float(vals[2])
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1]), 1.0
    return 0.0, 0.0, 0.0


def extract_keypoints(result, index: int) -> list[Keypoint]:
    kpts = result.keypoints
    if kpts is None:
        return []
    data = getattr(kpts, "data", None)
    if data is None:
        return []
    if index >= len(data):
        return []
    return [_as_xyconf(pt) for pt in data[index]]


def _count_visible(keypoints: list[Keypoint], indices: tuple[int, ...], kpt_conf: float) -> int:
    n = 0
    for idx in indices:
        if idx < len(keypoints) and keypoints[idx][2] >= kpt_conf:
            n += 1
    return n


def _pt(keypoints: list[Keypoint], index: int, kpt_conf: float) -> tuple[float, float] | None:
    if index >= len(keypoints):
        return None
    x, y, c = keypoints[index]
    if c < kpt_conf:
        return None
    return float(x), float(y)


def _bbox_diag(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0))


def _bone_ok(
    keypoints: list[Keypoint],
    a: int,
    b: int,
    diag: float,
    kpt_conf: float,
    min_frac: float = 0.045,
    max_frac: float = 0.90,
) -> bool:
    """True when both joints are visible and the segment length is anatomical."""
    pa = _pt(keypoints, a, kpt_conf)
    pb = _pt(keypoints, b, kpt_conf)
    if pa is None or pb is None:
        return False
    dist = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
    return min_frac * diag <= dist <= max_frac * diag


def count_valid_bones(
    keypoints: list[Keypoint],
    edges: tuple[tuple[int, int], ...],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    kpt_conf: float,
) -> int:
    diag = _bbox_diag(x1, y1, x2, y2)
    return sum(1 for a, b in edges if _bone_ok(keypoints, a, b, diag, kpt_conf))


def is_face_closeup(
    keypoints: list[Keypoint],
    kpt_conf: float,
    *,
    box: tuple[float, float, float, float] | None = None,
    frame_h: int | None = None,
) -> bool:
    """Laptop webcam: head fills the frame, shoulders are often cropped out.

    A small box in a garage shot with three shiny metal peaks is not a close-up.
    """
    if _count_visible(keypoints, (L_SHOULDER, R_SHOULDER), kpt_conf) >= 2:
        return False
    if _count_visible(keypoints, FACE, kpt_conf) < 3:
        return False
    if _count_visible(keypoints, FACE_CORE, kpt_conf) < 2:
        return False
    if box is not None and frame_h:
        height = max(box[3] - box[1], 1e-6)
        width = max(box[2] - box[0], 1e-6)
        if height < CLOSEUP_MIN_HEIGHT_FRAC * max(frame_h, 1):
            return False
        if height / width < CLOSEUP_MIN_ASPECT:
            return False
    return True


def anatomy_is_weak(keypoints: list[Keypoint], kpt_conf: float) -> bool:
    """True when the skeleton looks like clutter (no head, no torso girdle)."""
    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)
    torso_visible = _count_visible(keypoints, TORSO_POINTS, kpt_conf)
    return head_visible == 0 and torso_visible < 2


def _visible_xy(keypoints: list[Keypoint], kpt_conf: float) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y, c in keypoints if c >= kpt_conf]


def _segments_intersect(
    a: tuple[float, float] | None,
    b: tuple[float, float] | None,
    c: tuple[float, float] | None,
    d: tuple[float, float] | None,
) -> bool:
    """True when ab and cd properly cross. Shared endpoints are ignored."""
    if a is None or b is None or c is None or d is None:
        return False
    if a == c or a == d or b == c or b == d:
        return False

    def orient(p, q, r) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def keypoints_are_clustered(
    keypoints: list[Keypoint],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    kpt_conf: float,
    frame_h: int | None = None,
) -> bool:
    """True when visible joints pile up in a blob instead of spanning a body."""
    if is_face_closeup(keypoints, kpt_conf, box=(x1, y1, x2, y2), frame_h=frame_h):
        return False
    pts = _visible_xy(keypoints, kpt_conf)
    if len(pts) < 4:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    diag = _bbox_diag(x1, y1, x2, y2)
    return span < CLUSTER_SPAN_FRAC * diag


def bones_cross(keypoints: list[Keypoint], kpt_conf: float) -> bool:
    """Pipes and forks often cross; a living upper body rarely does."""
    ls = _pt(keypoints, L_SHOULDER, kpt_conf)
    le = _pt(keypoints, L_ELBOW, kpt_conf)
    lw = _pt(keypoints, L_WRIST, kpt_conf)
    rs = _pt(keypoints, R_SHOULDER, kpt_conf)
    re = _pt(keypoints, R_ELBOW, kpt_conf)
    rw = _pt(keypoints, R_WRIST, kpt_conf)
    if _segments_intersect(ls, le, rs, re):
        return True
    if _segments_intersect(ls, lw or le, rs, rw or re):
        return True
    lh = _pt(keypoints, L_HIP, kpt_conf)
    rh = _pt(keypoints, R_HIP, kpt_conf)
    return _segments_intersect(ls, lh, rs, rh)


def count_bone_crossings(keypoints: list[Keypoint], kpt_conf: float) -> int:
    crossings = 0
    ls = _pt(keypoints, L_SHOULDER, kpt_conf)
    le = _pt(keypoints, L_ELBOW, kpt_conf)
    lw = _pt(keypoints, L_WRIST, kpt_conf)
    rs = _pt(keypoints, R_SHOULDER, kpt_conf)
    re = _pt(keypoints, R_ELBOW, kpt_conf)
    rw = _pt(keypoints, R_WRIST, kpt_conf)
    lh = _pt(keypoints, L_HIP, kpt_conf)
    rh = _pt(keypoints, R_HIP, kpt_conf)
    lk = _pt(keypoints, L_KNEE, kpt_conf)
    rk = _pt(keypoints, R_KNEE, kpt_conf)
    la = _pt(keypoints, L_ANKLE, kpt_conf)
    ra = _pt(keypoints, R_ANKLE, kpt_conf)

    if _segments_intersect(ls, le, rs, re):
        crossings += 1
    if _segments_intersect(ls, lw or le, rs, rw or re):
        crossings += 1
    if _segments_intersect(le, lw, re, rw):
        crossings += 1
    if _segments_intersect(ls, lh, rs, rh):
        crossings += 1
    if _segments_intersect(lh, lk, rh, rk):
        crossings += 1
    if _segments_intersect(lk, la, rk, ra):
        crossings += 1
    return crossings


def _swap_indices(keypoints: list[Keypoint], pairs: list[tuple[int, int]]) -> list[Keypoint]:
    swapped = list(keypoints)
    for a, b in pairs:
        if a < len(swapped) and b < len(swapped):
            swapped[a], swapped[b] = swapped[b], swapped[a]
    return swapped


def resolve_bilateral_swap(keypoints: list[Keypoint], kpt_conf: float = 0.25) -> list[Keypoint]:
    """Resolve Left/Right limb swaps on crossed-arm or back-turned postures.

    If bones_cross, tests candidate swaps of shoulders-arms and/or hips-legs
    and keeps the variant with the fewest bone crossings.
    """
    if not bones_cross(keypoints, kpt_conf):
        return keypoints

    base_crossings = count_bone_crossings(keypoints, kpt_conf)
    if base_crossings == 0:
        return keypoints

    arm_pairs = [(L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST)]
    upper_pairs = [(L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST), (L_EYE, R_EYE), (L_EAR, R_EAR)]
    leg_pairs = [(L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)]
    lower_pairs = [(L_HIP, R_HIP), (L_KNEE, R_KNEE), (L_ANKLE, R_ANKLE)]
    full_pairs = upper_pairs + lower_pairs

    candidates = [
        _swap_indices(keypoints, arm_pairs),
        _swap_indices(keypoints, upper_pairs),
        _swap_indices(keypoints, leg_pairs),
        _swap_indices(keypoints, lower_pairs),
        _swap_indices(keypoints, upper_pairs + leg_pairs),
        _swap_indices(keypoints, full_pairs),
    ]

    best_kpts = keypoints
    min_crossings = base_crossings
    for cand in candidates:
        c = count_bone_crossings(cand, kpt_conf)
        if c < min_crossings:
            min_crossings = c
            best_kpts = cand
            if min_crossings == 0:
                break

    return best_kpts


def head_is_above_shoulders(keypoints: list[Keypoint], kpt_conf: float) -> bool | None:
    """Image y grows downward. None when there is not enough anatomy to judge."""
    head_ys = [
        keypoints[i][1]
        for i in HEAD_POINTS
        if i < len(keypoints) and keypoints[i][2] >= kpt_conf
    ]
    shoulder_ys = [
        keypoints[i][1]
        for i in (L_SHOULDER, R_SHOULDER)
        if i < len(keypoints) and keypoints[i][2] >= kpt_conf
    ]
    if not head_ys or not shoulder_ys:
        return None
    return (sum(head_ys) / len(head_ys)) + 4.0 <= (sum(shoulder_ys) / len(shoulder_ys))


def skeleton_is_plausible(
    keypoints: list[Keypoint],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    kpt_conf: float,
    frame_h: int | None = None,
) -> bool:
    """Reject engine/bike hallucinations: clustered joints or crossing limbs."""
    if keypoints_are_clustered(keypoints, x1, y1, x2, y2, kpt_conf, frame_h=frame_h):
        return False
    if not bones_cross(keypoints, kpt_conf):
        return True
    swapped = resolve_bilateral_swap(keypoints, kpt_conf)
    if not bones_cross(swapped, kpt_conf):
        return True
    # If both cross and joints are clustered or head is below shoulders, still reject (engine/bike fixtures)
    if head_is_above_shoulders(swapped, kpt_conf) is False or keypoints_are_clustered(
        swapped, x1, y1, x2, y2, kpt_conf, frame_h=frame_h
    ):
        return False
    # If still cross and no connected torso shortcut, reject
    torso_visible = _count_visible(swapped, TORSO_POINTS, kpt_conf)
    torso_bones = count_valid_bones(swapped, TORSO_EDGES, x1, y1, x2, y2, kpt_conf)
    if torso_visible >= 3 and torso_bones >= 2:
        return True
    return False


def is_creeper_or_underbody_pose(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    keypoints: list[Keypoint],
    frame_h: int,
    min_dim_frac: float = 0.08,
    kpt_conf: float = 0.35,
    allow_head_shoulder: bool = True,
) -> bool:
    """Worker lying horizontally under a chassis or on a creeper.

    Requires a connected torso or a connected hip–knee–ankle chain.
    Head+shoulders without hips is optional and must be gated by confidence
    so an open engine cannot use this path.
    A pair of shoes (two hallucinated ankles, no hips) is not a worker.
    """
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    max_dim = max(width, height)
    if max_dim < min_dim_frac * max(frame_h, 1):
        return False

    torso_visible = _count_visible(keypoints, TORSO_POINTS, kpt_conf)
    legs_visible = _count_visible(keypoints, LEG_POINTS, kpt_conf)
    hips_visible = _count_visible(keypoints, HIP_POINTS, kpt_conf)
    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)
    shoulders_visible = _count_visible(keypoints, (L_SHOULDER, R_SHOULDER), kpt_conf)
    torso_bones = count_valid_bones(keypoints, TORSO_EDGES, x1, y1, x2, y2, kpt_conf)
    leg_bones = count_valid_bones(keypoints, LEG_EDGES, x1, y1, x2, y2, kpt_conf)
    upper_bones = count_valid_bones(
        keypoints,
        (
            (NOSE, L_EYE),
            (NOSE, R_EYE),
            (L_SHOULDER, R_SHOULDER),
            (L_SHOULDER, L_ELBOW),
            (R_SHOULDER, R_ELBOW),
        ),
        x1,
        y1,
        x2,
        y2,
        kpt_conf,
    )

    # Standing/crouching torso plus at least one connected limb.
    if torso_visible >= 2 and torso_bones >= 1 and (legs_visible >= 1 or torso_visible >= 3):
        return True
    if torso_visible >= 3 and torso_bones >= 1:
        return True

    # Occluded under-vehicle: hips must be present and joined to knees/ankles.
    # Two disconnected shoe points never form a hip–knee bone.
    if hips_visible >= 1 and legs_visible >= 2 and leg_bones >= 2:
        return True

    # Chassis hides the legs: head + shoulders. Engines hallucinate this
    # graph, so callers must disable it for low-confidence boxes.
    if (
        allow_head_shoulder
        and head_visible >= 1
        and shoulders_visible >= 1
        and upper_bones >= 2
        and head_is_above_shoulders(keypoints, kpt_conf) is not False
    ):
        return True
    return False


def is_hood_lean_pose(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    keypoints: list[Keypoint],
    frame_h: int,
    min_dim_frac: float = 0.035,
    kpt_conf: float = 0.35,
) -> bool:
    """Mechanic leaning into an engine bay or over a bumper.

    From a high camera the box stays taller than wide while legs vanish into
    the car. Require a real head or shoulder girdle plus connected upper-body
    bones so shoes and bags still fail.
    """
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    if max(width, height) < min_dim_frac * max(frame_h, 1):
        return False
    if anatomy_is_weak(keypoints, kpt_conf):
        return False
    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)
    shoulders_visible = _count_visible(keypoints, (L_SHOULDER, R_SHOULDER), kpt_conf)
    if head_visible < 1 or shoulders_visible < 1:
        return False
    if head_is_above_shoulders(keypoints, kpt_conf) is False:
        return False
    upper_bones = count_valid_bones(
        keypoints,
        (
            (NOSE, L_EYE),
            (NOSE, R_EYE),
            (L_SHOULDER, R_SHOULDER),
            (L_SHOULDER, L_ELBOW),
            (R_SHOULDER, R_ELBOW),
            (L_SHOULDER, L_HIP),
            (R_SHOULDER, R_HIP),
        ),
        x1,
        y1,
        x2,
        y2,
        kpt_conf,
    )
    if upper_bones < 2:
        return False
    return True


def is_crouch_or_sit_pose(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    keypoints: list[Keypoint],
    frame_h: int,
    min_dim_frac: float = 0.035,
    kpt_conf: float = 0.35,
) -> bool:
    """Squat, sit, or kneel with a real torso/head — not two floating ankles."""
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    if max(width, height) < min_dim_frac * max(frame_h, 1):
        return False
    if anatomy_is_weak(keypoints, kpt_conf):
        return False
    bones = count_valid_bones(keypoints, KINEMATIC_EDGES, x1, y1, x2, y2, kpt_conf)
    if bones < 2:
        return False
    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)
    torso_visible = _count_visible(keypoints, TORSO_POINTS, kpt_conf)
    compressed = height / width <= 1.15
    if not compressed:
        return False
    return head_visible >= 1 or torso_visible >= 2


def is_upper_body_pose(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    keypoints: list[Keypoint],
    frame_h: int,
    min_dim_frac: float = 0.035,
    kpt_conf: float = 0.35,
) -> bool:
    """Chest-up or desk worker: laptop webcam, workbench, or vehicle hood.

    Head keypoints (nose, eyes, ears) and shoulders are visible, with the head
    positioned anatomically above the shoulders, without requiring hips or legs.
    """
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    if max(width, height) < min_dim_frac * max(frame_h, 1):
        return False
    if anatomy_is_weak(keypoints, kpt_conf):
        return False

    # Adaptive shoulder confidence for edge models / TinyPose where boundary joints have lower peak conf
    sho_conf = min(kpt_conf, 0.15)
    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)
    shoulders_visible = _count_visible(keypoints, (L_SHOULDER, R_SHOULDER), sho_conf)

    if head_visible >= 1 and shoulders_visible >= 1:
        if head_is_above_shoulders(keypoints, sho_conf) is False:
            return False
        diag = _bbox_diag(x1, y1, x2, y2)
        shoulder_bone = _bone_ok(keypoints, L_SHOULDER, R_SHOULDER, diag, sho_conf)
        upper_bones = count_valid_bones(
            keypoints,
            (
                (NOSE, L_EYE),
                (NOSE, R_EYE),
                (L_EYE, L_EAR),
                (R_EYE, R_EAR),
                (L_SHOULDER, R_SHOULDER),
                (L_SHOULDER, L_ELBOW),
                (R_SHOULDER, R_ELBOW),
                (L_ELBOW, L_WRIST),
                (R_ELBOW, R_WRIST),
            ),
            x1,
            y1,
            x2,
            y2,
            sho_conf,
        )
        if shoulder_bone and head_visible >= 1:
            return True
        if upper_bones >= 1 and head_visible >= 2:
            return True

    # Face close-up / desk worker where shoulders are below frame bottom
    if head_visible >= 3 and is_face_closeup(
        keypoints, kpt_conf, box=(x1, y1, x2, y2), frame_h=frame_h
    ):
        return True

    return False


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
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
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(ax2 - ax1, 1e-6) * max(ay2 - ay1, 1e-6)
    area_b = max(bx2 - bx1, 1e-6) * max(by2 - by1, 1e-6)
    return inter / min(area_a, area_b)


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


def is_human_pose(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    keypoints: list[Keypoint],
    frame_h: int,
    min_height_frac: float = 0.05,
    min_aspect: float = 0.85,
    min_keypoints: int = 3,
    kpt_conf: float = 0.35,
    box_conf: float = 1.0,
) -> bool:
    """Rigorous human pose kinematic validation.

    Rejects inanimate objects (shoes, backpacks, jackets, chairs, engines)
    whose keypoints are clustered, crossing, or lack an upper-body / hip girdle.
    """
    height = y2 - y1
    width = max(x2 - x1, 1e-6)
    allow_shortcuts = float(box_conf) >= SHORTCUT_MIN_CONF

    keypoints = resolve_bilateral_swap(keypoints, kpt_conf=kpt_conf)

    if not skeleton_is_plausible(keypoints, x1, y1, x2, y2, kpt_conf, frame_h=frame_h):
        return False

    if is_creeper_or_underbody_pose(
        x1,
        y1,
        x2,
        y2,
        keypoints,
        frame_h,
        min_height_frac * 0.7,
        kpt_conf,
        allow_head_shoulder=allow_shortcuts,
    ):
        return True

    if allow_shortcuts and is_hood_lean_pose(
        x1, y1, x2, y2, keypoints, frame_h, min_height_frac * 0.7, kpt_conf
    ):
        return True

    if is_crouch_or_sit_pose(
        x1, y1, x2, y2, keypoints, frame_h, min_height_frac * 0.7, kpt_conf
    ):
        return True

    if (
        allow_shortcuts
        and is_upper_body_pose(
            x1, y1, x2, y2, keypoints, frame_h, min_height_frac * 0.7, kpt_conf
        )
    ):
        return True

    # Crouching, kneeling, or bending worker: height is compressed but kinematic bones are solid
    effective_min_h = min_height_frac * max(frame_h, 1)
    bones = count_valid_bones(keypoints, KINEMATIC_EDGES, x1, y1, x2, y2, kpt_conf)
    torso_visible = _count_visible(keypoints, TORSO_POINTS, kpt_conf)
    legs_visible = _count_visible(keypoints, LEG_POINTS, kpt_conf)
    hips_visible = _count_visible(keypoints, HIP_POINTS, kpt_conf)
    if bones >= 2 and (torso_visible >= 2 or legs_visible >= 2):
        effective_min_h *= 0.35

    if height < effective_min_h:
        return False

    if is_face_closeup(
        keypoints, kpt_conf, box=(x1, y1, x2, y2), frame_h=frame_h
    ):
        return True

    visible_pts = [pt for pt in keypoints if pt[2] >= kpt_conf]
    if len(visible_pts) < min_keypoints:
        return False

    head_visible = _count_visible(keypoints, HEAD_POINTS, kpt_conf)

    # Anti-object: desk clutter has no head and no connected torso girdle.
    if head_visible == 0 and torso_visible < 2:
        return False
    if bones < 2:
        return False

    # Low-confidence boxes cannot use an upper-body-only standing pass —
    # that is the same graph YOLO paints on engines and bike frames.
    if not allow_shortcuts and hips_visible < 1 and legs_visible < 2:
        return False

    if (head_visible >= 1 or torso_visible >= 2) and (torso_visible >= 1 or legs_visible >= 1):
        if min_aspect > 0 and height / width < min_aspect * 0.35 and torso_visible < 2:
            return False
        if allow_shortcuts and head_is_above_shoulders(keypoints, kpt_conf) is False:
            return False
        return True

    return torso_visible >= 3 and bones >= 2


def person_detections(
    result,
    frame_h: int,
    conf_min: float = 0.25,
    min_height_frac: float = 0.05,
    min_aspect: float = 1.1,
    min_keypoints: int = 3,
    kpt_conf: float = 0.25,
    track_low_thresh: float = 0.10,
    return_low: bool = False,
) -> tuple[list[Detection], list[Detection]] | tuple[list[Detection], list[Detection], list[Detection]]:
    accepted: list[Detection] = []
    rejected: list[Detection] = []
    low_score: list[Detection] = []
    if result.boxes is None:
        if return_low:
            return accepted, rejected, low_score
        return accepted, rejected
    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf < track_low_thresh:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        keypoints = extract_keypoints(result, i)
        if conf < conf_min:
            # Low-score detection: for secondary tracker association only
            det = Detection(x1, y1, x2, y2, conf, keypoints)
            det.accepted = False
            low_score.append(det)
            continue

        keypoints = resolve_bilateral_swap(keypoints, kpt_conf=kpt_conf)
        det = Detection(x1, y1, x2, y2, conf, keypoints)
        det.accepted = is_human_pose(
            x1,
            y1,
            x2,
            y2,
            keypoints,
            frame_h,
            min_height_frac=min_height_frac,
            min_aspect=min_aspect,
            min_keypoints=min_keypoints,
            kpt_conf=kpt_conf,
            box_conf=conf,
        )
        (accepted if det.accepted else rejected).append(det)
    suppressed = suppress_nested_boxes(accepted)
    if return_low:
        return suppressed, rejected, low_score
    return suppressed, rejected


def person_detections_split(
    result,
    frame_h: int,
    conf_min: float = 0.25,
    track_low_thresh: float = 0.10,
    min_height_frac: float = 0.05,
    min_aspect: float = 1.1,
    min_keypoints: int = 3,
    kpt_conf: float = 0.25,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Return (accepted_high, rejected_high, low_score_dets)."""
    return person_detections(
        result,
        frame_h,
        conf_min=conf_min,
        min_height_frac=min_height_frac,
        min_aspect=min_aspect,
        min_keypoints=min_keypoints,
        kpt_conf=kpt_conf,
        track_low_thresh=track_low_thresh,
        return_low=True,
    )


DRAW_KPT_FLOOR = 0.25
DRAW_BBOX_MARGIN = 0.18


def draw_skeleton(
    frame,
    keypoints: list[Keypoint],
    kpt_conf: float = 0.30,
    color=(100, 240, 100),
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw only anatomically plausible bones that land inside the box.

    The accept path already runs `_bone_ok`. Drawing used to skip that check,
    so a box accepted on two good torso bones still painted YOLO's scattered
    wrist/ankle points across the shop floor.
    """
    floor = float(kpt_conf) if kpt_conf is not None else DRAW_KPT_FLOOR
    clip: tuple[float, float, float, float] | None = None
    diag = 1.0
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        diag = _bbox_diag(x1, y1, x2, y2)
        mx = DRAW_BBOX_MARGIN * max(x2 - x1, 1.0)
        my = DRAW_BBOX_MARGIN * max(y2 - y1, 1.0)
        clip = (x1 - mx, y1 - my, x2 + mx, y2 + my)

    def _in_box(x: float, y: float) -> bool:
        if clip is None:
            return True
        cx1, cy1, cx2, cy2 = clip
        return cx1 <= x <= cx2 and cy1 <= y <= cy2

    for a, b in SKELETON:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca < floor or cb < floor:
            continue
        if bbox is not None and not _bone_ok(keypoints, a, b, diag, floor):
            continue
        if not _in_box(xa, ya) or not _in_box(xb, yb):
            continue
        cv2.line(
            frame,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            color,
            2,
            cv2.LINE_AA,
        )
    for x, y, c in keypoints:
        if c < floor:
            continue
        if not _in_box(x, y):
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def draw_detection(
    frame,
    det: Detection,
    *,
    in_roi: bool,
    kpt_conf: float = 0.30,
) -> None:
    x1, y1, x2, y2 = (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
    time_badge = f" ({det.active_time_str})" if getattr(det, "active_time_str", None) else ""
    if det.accepted:
        is_confirmed = getattr(det, "confirmed", True)
        if not is_confirmed:
            color = (255, 180, 50) if in_roi else (180, 150, 100)
            label = f"[Acquiring] {det.conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            if det.keypoints:
                draw_skeleton(frame, det.keypoints, kpt_conf, color, det.box())
        else:
            color = (80, 220, 80) if in_roi else (170, 170, 170)
            if det.identity and det.is_staff:
                label = f"[Staff: {det.identity}]{time_badge} {det.conf:.2f}"
                color = (50, 240, 50) if in_roi else (100, 200, 100)
            elif det.identity:
                label = f"[{det.identity}]{time_badge} {det.conf:.2f}"
                color = (0, 165, 255) if in_roi else (170, 170, 170)
            else:
                label = f"person{time_badge} {det.conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if det.keypoints:
                draw_skeleton(frame, det.keypoints, kpt_conf, color, det.box())
    else:
        color = (120, 120, 120)
        label = f"blob {det.conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    # Render dark contrasting backdrop for label legibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_y = max(text_h + 4, y1 - 6)
    cv2.rectangle(
        frame,
        (x1, text_y - text_h - 4),
        (x1 + text_w + 6, text_y + baseline),
        (15, 15, 15),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 3, text_y - 2),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def crouch_sit_keypoints() -> list[Keypoint]:
    """Squat / sit-under-bumper: connected torso and head, compressed legs."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (100.0, 70.0, 0.9)
    pts[1] = (92.0, 62.0, 0.8)
    pts[2] = (108.0, 62.0, 0.8)
    pts[5] = (80.0, 95.0, 0.9)
    pts[6] = (120.0, 98.0, 0.9)
    pts[7] = (70.0, 118.0, 0.75)
    pts[8] = (130.0, 120.0, 0.75)
    pts[11] = (85.0, 130.0, 0.85)
    pts[12] = (118.0, 132.0, 0.85)
    pts[13] = (82.0, 142.0, 0.8)
    pts[14] = (120.0, 144.0, 0.8)
    return pts


def occluded_upper_body_keypoints() -> list[Keypoint]:
    """Head and shoulders visible; legs hidden under a chassis."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (160.0, 90.0, 0.9)
    pts[1] = (150.0, 80.0, 0.85)
    pts[2] = (170.0, 80.0, 0.85)
    pts[5] = (120.0, 115.0, 0.9)
    pts[6] = (200.0, 118.0, 0.9)
    pts[7] = (110.0, 145.0, 0.75)
    pts[8] = (210.0, 148.0, 0.75)
    return pts


def hood_lean_keypoints() -> list[Keypoint]:
    """Head and shoulders at a bumper; legs hidden in the engine bay."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (140.0, 70.0, 0.85)
    pts[1] = (132.0, 62.0, 0.70)
    pts[2] = (148.0, 62.0, 0.70)
    pts[5] = (110.0, 110.0, 0.80)
    pts[6] = (170.0, 112.0, 0.80)
    pts[7] = (100.0, 150.0, 0.55)
    pts[8] = (180.0, 148.0, 0.55)
    return pts


def standing_person_keypoints() -> list[Keypoint]:
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (100.0, 40.0, 0.9)
    pts[5] = (80.0, 80.0, 0.9)
    pts[6] = (120.0, 80.0, 0.9)
    pts[7] = (70.0, 120.0, 0.8)
    pts[8] = (130.0, 120.0, 0.8)
    pts[11] = (85.0, 150.0, 0.8)
    pts[12] = (115.0, 150.0, 0.8)
    return pts


def closeup_face_keypoints() -> list[Keypoint]:
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (100.0, 80.0, 0.9)
    pts[1] = (90.0, 70.0, 0.85)
    pts[2] = (110.0, 70.0, 0.85)
    pts[3] = (80.0, 80.0, 0.7)
    pts[4] = (120.0, 80.0, 0.7)
    return pts


def shoe_pair_keypoints() -> list[Keypoint]:
    """Hallucinated ankles/knees on a pair of boots — no hips, no torso."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[13] = (90.0, 180.0, 0.42)
    pts[14] = (130.0, 182.0, 0.40)
    pts[15] = (88.0, 210.0, 0.48)
    pts[16] = (132.0, 212.0, 0.45)
    return pts


def backpack_clutter_keypoints() -> list[Keypoint]:
    """Disconnected floating points typical of a bag or folded jacket."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[9] = (70.0, 90.0, 0.38)
    pts[13] = (95.0, 170.0, 0.36)
    pts[15] = (140.0, 200.0, 0.41)
    pts[16] = (60.0, 205.0, 0.37)
    return pts


def engine_bay_keypoints() -> list[Keypoint]:
    """YOLO hallucination on an open engine: joints piled on the block.

    Head sits below the shoulder line and the arm bones cross — the pattern
    seen on Camera 06 when the pose head maps hoses onto a COCO skeleton.
    """
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (148.0, 175.0, 0.42)
    pts[1] = (142.0, 170.0, 0.38)
    pts[2] = (154.0, 172.0, 0.36)
    pts[5] = (130.0, 150.0, 0.40)
    pts[6] = (155.0, 148.0, 0.39)
    pts[7] = (160.0, 165.0, 0.37)
    pts[8] = (125.0, 168.0, 0.36)
    pts[11] = (138.0, 162.0, 0.33)
    pts[12] = (150.0, 164.0, 0.32)
    return pts


def tinypose_inflated_engine_keypoints() -> list[Keypoint]:
    """TinyPose argmax-on-metal after the old 1/0.16 scale-up.

    Every joint looks YOLO-visible even though the crop is an engine block.
    Head sits below the shoulders; joints stay piled on the block.
    """
    pts = engine_bay_keypoints()
    return [(x, y, 0.85 if c > 0 else 0.0) for x, y, c in pts]


def motorcycle_frame_keypoints() -> list[Keypoint]:
    """YOLO hallucination on a motorcycle frame / forks as a fake torso."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (80.0, 95.0, 0.38)
    pts[1] = (76.0, 92.0, 0.34)
    pts[2] = (84.0, 93.0, 0.33)
    pts[5] = (70.0, 88.0, 0.40)
    pts[6] = (90.0, 90.0, 0.39)
    pts[7] = (92.0, 100.0, 0.35)
    pts[8] = (68.0, 102.0, 0.34)
    pts[11] = (75.0, 98.0, 0.33)
    pts[12] = (85.0, 99.0, 0.32)
    return pts
