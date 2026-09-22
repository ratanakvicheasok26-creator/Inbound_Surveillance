"""Kinematic human validation: reject shoes, bags, and other desk clutter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occupancy import partial_legs_pose_keypoints, under_vehicle_pose_keypoints
from person import (
    DRAW_KPT_FLOOR,
    Detection,
    backpack_clutter_keypoints,
    closeup_face_keypoints,
    crouch_sit_keypoints,
    draw_skeleton,
    engine_bay_keypoints,
    hood_lean_keypoints,
    is_creeper_or_underbody_pose,
    is_face_closeup,
    is_hood_lean_pose,
    is_human_pose,
    motorcycle_frame_keypoints,
    occluded_upper_body_keypoints,
    shoe_pair_keypoints,
    standing_person_keypoints,
    tinypose_inflated_engine_keypoints,
)
from corroborate import veto_vehicle_interior
from vehicle import VehicleDetection
from bay_zoom import (
    bay_crop_xyxy,
    candidate_vehicle_bays,
    empty_vehicle_bays,
    merge_detections,
    remap_detection,
    remap_keypoints,
    suppress_nested_boxes,
    zoom_empty_bays,
)


def _scale_kpts(kpts, src, dst):
    sx1, sy1, sx2, sy2 = src
    dx1, dy1, dx2, dy2 = dst
    sw = max(sx2 - sx1, 1e-6)
    sh = max(sy2 - sy1, 1e-6)
    dw = dx2 - dx1
    dh = dy2 - dy1
    return [
        (dx1 + (x - sx1) / sw * dw, dy1 + (y - sy1) / sh * dh, c)
        for x, y, c in kpts
    ]


class KinematicPoseTests(unittest.TestCase):
    def test_standing_person_is_accepted(self):
        kpts = standing_person_keypoints()
        self.assertTrue(is_human_pose(50, 10, 150, 280, kpts, 480, kpt_conf=0.35))

    def test_closeup_face_is_accepted(self):
        kpts = closeup_face_keypoints()
        self.assertTrue(is_human_pose(40, 20, 160, 200, kpts, 480, kpt_conf=0.35))

    def test_landscape_webcam_face_closeup_is_accepted(self):
        kpts = closeup_face_keypoints()
        # Laptop webcam: person fills the width, box is shorter than it is wide.
        self.assertTrue(
            is_human_pose(20, 200, 620, 470, kpts, 480, kpt_conf=0.35, box_conf=0.80)
        )

    def test_shoe_pair_on_desk_is_rejected(self):
        kpts = shoe_pair_keypoints()
        self.assertFalse(
            is_creeper_or_underbody_pose(60, 140, 200, 280, kpts, 480, kpt_conf=0.35)
        )
        self.assertFalse(is_human_pose(60, 140, 200, 280, kpts, 480, kpt_conf=0.35))

    def test_backpack_clutter_is_rejected(self):
        kpts = backpack_clutter_keypoints()
        self.assertFalse(is_human_pose(40, 40, 180, 260, kpts, 480, kpt_conf=0.35))

    def test_under_vehicle_connected_chain_is_accepted(self):
        kpts = under_vehicle_pose_keypoints()
        self.assertTrue(
            is_creeper_or_underbody_pose(40, 70, 280, 160, kpts, 480, kpt_conf=0.35)
        )
        self.assertTrue(is_human_pose(40, 70, 280, 160, kpts, 480, kpt_conf=0.35))

    def test_partial_legs_with_hips_are_accepted(self):
        kpts = partial_legs_pose_keypoints()
        self.assertTrue(
            is_creeper_or_underbody_pose(100, 80, 280, 160, kpts, 480, kpt_conf=0.35)
        )

    def test_two_floating_ankles_are_rejected(self):
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[15] = (90.0, 200.0, 0.45)
        kpts[16] = (130.0, 202.0, 0.44)
        self.assertFalse(is_human_pose(70, 160, 160, 240, kpts, 480, kpt_conf=0.35))

    def test_distant_small_person_is_accepted(self):
        frame_h = 1080
        src = (70.0, 40.0, 130.0, 150.0)
        height = 0.05 * frame_h
        width = height * (src[2] - src[0]) / (src[3] - src[1])
        box = (900.0, 980.0, 900.0 + width, 980.0 + height)
        kpts = _scale_kpts(standing_person_keypoints(), src, box)
        self.assertTrue(
            is_human_pose(
                *box,
                kpts,
                frame_h,
                min_height_frac=0.05,
                min_aspect=1.1,
                min_keypoints=4,
                kpt_conf=0.35,
            )
        )

    def test_crouch_sit_under_bumper_is_accepted(self):
        kpts = crouch_sit_keypoints()
        self.assertTrue(
            is_human_pose(60, 55, 200, 155, kpts, 480, min_height_frac=0.05, kpt_conf=0.35)
        )

    def test_occluded_head_and_shoulders_under_car_are_accepted(self):
        kpts = occluded_upper_body_keypoints()
        self.assertTrue(
            is_creeper_or_underbody_pose(40, 70, 280, 160, kpts, 480, kpt_conf=0.35)
        )
        self.assertTrue(
            is_human_pose(40, 70, 280, 160, kpts, 480, min_height_frac=0.05, kpt_conf=0.35)
        )

    def test_hood_lean_tall_box_is_accepted(self):
        kpts = hood_lean_keypoints()
        self.assertTrue(
            is_hood_lean_pose(90, 40, 190, 280, kpts, 480, kpt_conf=0.25)
        )
        self.assertTrue(
            is_human_pose(
                90, 40, 190, 280, kpts, 480, min_height_frac=0.05, min_keypoints=3, kpt_conf=0.25
            )
        )

    def test_hood_lean_does_not_accept_clutter(self):
        self.assertFalse(
            is_hood_lean_pose(60, 140, 200, 280, shoe_pair_keypoints(), 480, kpt_conf=0.25)
        )
        self.assertFalse(
            is_hood_lean_pose(40, 40, 180, 260, backpack_clutter_keypoints(), 480, kpt_conf=0.25)
        )
        self.assertFalse(
            is_human_pose(60, 140, 200, 280, shoe_pair_keypoints(), 480, kpt_conf=0.25)
        )
        self.assertFalse(
            is_human_pose(40, 40, 180, 260, backpack_clutter_keypoints(), 480, kpt_conf=0.25)
        )

    def test_engine_bay_hallucination_is_rejected(self):
        kpts = engine_bay_keypoints()
        box = (90.0, 40.0, 190.0, 280.0)
        self.assertFalse(is_hood_lean_pose(*box, kpts, 480, kpt_conf=0.25))
        for conf in (0.36, 0.47, 0.85):
            self.assertFalse(
                is_human_pose(
                    *box,
                    kpts,
                    480,
                    min_height_frac=0.05,
                    min_keypoints=3,
                    kpt_conf=0.25,
                    box_conf=conf,
                ),
                msg=f"engine skeleton should fail at conf={conf}",
            )
        # TinyPose default joint floor. Inflated argmax-on-metal must still fail.
        self.assertFalse(
            is_human_pose(
                *box,
                kpts,
                480,
                min_height_frac=0.05,
                min_keypoints=3,
                kpt_conf=0.12,
                box_conf=0.80,
            )
        )
        self.assertFalse(
            is_human_pose(
                *box,
                tinypose_inflated_engine_keypoints(),
                480,
                min_height_frac=0.05,
                min_keypoints=3,
                kpt_conf=0.12,
                box_conf=0.80,
            )
        )

    def test_garage_engine_face_blob_is_not_a_webcam_closeup(self):
        """Three shiny metal peaks in a small garage box must not use the laptop path."""
        src = (40.0, 20.0, 160.0, 200.0)
        box = (220.0, 240.0, 360.0, 330.0)
        kpts = _scale_kpts(closeup_face_keypoints(), src, box)
        self.assertFalse(is_face_closeup(kpts, 0.35, box=box, frame_h=480))
        self.assertFalse(
            is_human_pose(*box, kpts, 480, kpt_conf=0.12, box_conf=0.80)
        )
        self.assertFalse(
            is_human_pose(*box, kpts, 480, kpt_conf=0.35, box_conf=0.80)
        )

    def test_motorcycle_frame_hallucination_is_rejected(self):
        kpts = motorcycle_frame_keypoints()
        box = (40.0, 40.0, 140.0, 200.0)
        for conf in (0.38, 0.47, 0.85):
            self.assertFalse(
                is_human_pose(
                    *box,
                    kpts,
                    480,
                    min_height_frac=0.05,
                    min_keypoints=3,
                    kpt_conf=0.25,
                    box_conf=conf,
                ),
                msg=f"motorcycle skeleton should fail at conf={conf}",
            )

    def test_low_conf_upper_body_shortcut_is_blocked(self):
        kpts = hood_lean_keypoints()
        self.assertFalse(
            is_human_pose(
                90,
                40,
                190,
                280,
                kpts,
                480,
                min_height_frac=0.05,
                min_keypoints=3,
                kpt_conf=0.25,
                box_conf=0.40,
            )
        )
        self.assertTrue(
            is_human_pose(
                90,
                40,
                190,
                280,
                kpts,
                480,
                min_height_frac=0.05,
                min_keypoints=3,
                kpt_conf=0.25,
                box_conf=0.85,
            )
        )

    def test_crossed_arms_accepted_with_bilateral_swap(self):
        kpts = standing_person_keypoints()
        # Cross arms: left elbow reaches right, right elbow reaches left
        kpts[7] = (130.0, 120.0, 0.85)
        kpts[8] = (70.0, 120.0, 0.85)
        from person import bones_cross, resolve_bilateral_swap
        self.assertTrue(bones_cross(kpts, 0.35))
        swapped = resolve_bilateral_swap(kpts, 0.35)
        self.assertFalse(bones_cross(swapped, 0.35))
        self.assertTrue(is_human_pose(50, 10, 150, 280, kpts, 480, kpt_conf=0.35))

    def test_upper_body_webcam_pose_is_accepted(self):
        from person import is_upper_body_pose
        # Chest-up desk worker: head (nose, eyes, ears) + shoulders visible, hips/legs occluded
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (160.0, 80.0, 0.90)   # Nose
        kpts[1] = (150.0, 70.0, 0.85)   # Left eye
        kpts[2] = (170.0, 70.0, 0.85)   # Right eye
        kpts[3] = (140.0, 75.0, 0.80)   # Left ear
        kpts[4] = (180.0, 75.0, 0.80)   # Right ear
        kpts[5] = (110.0, 140.0, 0.88)  # Left shoulder
        kpts[6] = (210.0, 142.0, 0.88)  # Right shoulder
        self.assertTrue(is_upper_body_pose(80, 50, 240, 260, kpts, 480, kpt_conf=0.35))
        self.assertTrue(is_human_pose(80, 50, 240, 260, kpts, 480, kpt_conf=0.35, box_conf=0.75))

    def test_upper_body_pose_rejects_headless_clutter(self):
        from person import is_upper_body_pose
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[5] = (110.0, 140.0, 0.88)
        kpts[6] = (210.0, 142.0, 0.88)
        self.assertFalse(is_upper_body_pose(80, 50, 240, 260, kpts, 480, kpt_conf=0.35))


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = [_ListTensor(xyxy)]
        self.conf = [conf]


class _ListTensor:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _Keypoints:
    def __init__(self, data):
        self.data = data


class _FakeResult:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakePoseModel:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, crop, **kwargs):
        self.calls += 1
        self.last_crop_shape = getattr(crop, "shape", None)
        return [self.result]


class BayZoomTests(unittest.TestCase):
    def test_crop_remap_lands_in_frame_coords(self):
        crop = (120, 80, 400, 360)
        kpts = standing_person_keypoints()
        det = Detection(10.0, 20.0, 50.0, 160.0, 0.8, kpts)
        remapped = remap_detection(det, crop)
        self.assertAlmostEqual(remapped.x1, 130.0)
        self.assertAlmostEqual(remapped.y1, 100.0)
        self.assertAlmostEqual(remapped.x2, 170.0)
        self.assertAlmostEqual(remapped.y2, 240.0)
        shifted = remap_keypoints(kpts, crop)
        self.assertAlmostEqual(shifted[0][0], kpts[0][0] + 120.0)
        self.assertAlmostEqual(shifted[0][1], kpts[0][1] + 80.0)
        self.assertAlmostEqual(shifted[0][2], kpts[0][2])

    def test_zoom_merge_does_not_duplicate_full_frame_person(self):
        full = Detection(100.0, 80.0, 180.0, 300.0, 0.9, standing_person_keypoints())
        zoom = Detection(105.0, 85.0, 175.0, 290.0, 0.7, standing_person_keypoints())
        accepted, rejected = merge_detections([full], [], [zoom], [])
        self.assertEqual(len(accepted), 1)
        self.assertIs(accepted[0], full)
        self.assertEqual(rejected, [])

    def test_zoom_merge_keeps_new_person(self):
        full = Detection(40.0, 40.0, 90.0, 200.0, 0.9, standing_person_keypoints())
        zoom = Detection(500.0, 200.0, 580.0, 420.0, 0.8, standing_person_keypoints())
        accepted, _ = merge_detections([full], [], [zoom], [])
        self.assertEqual(len(accepted), 2)

    def test_zoom_merge_upgrades_higher_confidence_zoom_person(self):
        full = Detection(100.0, 80.0, 180.0, 300.0, 0.5, standing_person_keypoints())
        zoom = Detection(105.0, 85.0, 175.0, 290.0, 0.85, standing_person_keypoints())
        accepted, rejected = merge_detections([full], [], [zoom], [])
        self.assertEqual(len(accepted), 1)
        self.assertIs(accepted[0], zoom)
        self.assertEqual(accepted[0].conf, 0.85)

    def test_candidate_vehicle_bays_includes_occupied_bay_by_default(self):
        bays = [
            {"id": "bay_1", "name": "Lift", "type": "vehicle_bay", "roi": [0.10, 0.20, 0.35, 0.60]},
            {"id": "bay_2", "name": "Bay 2", "type": "vehicle_bay", "roi": [0.55, 0.20, 0.35, 0.60]},
        ]
        person = Detection(150.0, 300.0, 220.0, 620.0, 0.9, standing_person_keypoints())
        candidates = candidate_vehicle_bays(bays, [person], 1000, 1000, 0.25, only_empty=False)
        self.assertEqual([b["id"] for b in candidates], ["bay_1", "bay_2"])

    def test_suppress_nested_boxes_removes_contained_duplicate(self):
        full = Detection(100.0, 80.0, 200.0, 350.0, 0.8, standing_person_keypoints())
        torso = Detection(110.0, 85.0, 195.0, 220.0, 0.6, standing_person_keypoints())
        suppressed = suppress_nested_boxes([full, torso])
        self.assertEqual(len(suppressed), 1)
        self.assertIs(suppressed[0], full)

    def test_empty_vehicle_bays_skip_occupied_and_tool_area(self):
        bays = [
            {"id": "bay_1", "name": "Lift", "type": "vehicle_bay", "roi": [0.10, 0.20, 0.35, 0.60]},
            {"id": "tools", "name": "Tools", "type": "tool_area", "roi": [0.42, 0.05, 0.16, 0.20]},
            {"id": "bay_2", "name": "Bay 2", "type": "vehicle_bay", "roi": [0.55, 0.20, 0.35, 0.60]},
        ]
        person = Detection(150.0, 300.0, 220.0, 620.0, 0.9, standing_person_keypoints())
        empty = empty_vehicle_bays(bays, [person], 1000, 1000, 0.25)
        self.assertEqual([b["id"] for b in empty], ["bay_2"])

    def test_empty_vehicle_bays_honor_occupancy_hints(self):
        bays = [{"id": "bay_2", "name": "Bay 2", "type": "vehicle_bay", "roi": [0.55, 0.20, 0.35, 0.60]}]
        hints = {"bay_2": {"session_open": False, "vehicle_present": False}}
        self.assertEqual(empty_vehicle_bays(bays, [], 1000, 1000, 0.25, occupancy_by_id=hints), [])
        hints["bay_2"]["session_open"] = True
        self.assertEqual(len(empty_vehicle_bays(bays, [], 1000, 1000, 0.25, occupancy_by_id=hints)), 1)

    def test_polygon_crop_includes_pad(self):
        bay = {
            "id": "bay_2",
            "roi": [0.50, 0.20, 0.30, 0.50],
            "polygon": [[0.50, 0.20], [0.80, 0.20], [0.80, 0.70], [0.50, 0.70]],
        }
        x1, y1, x2, y2 = bay_crop_xyxy(bay, 1000, 1000, pad=0.08)
        self.assertLessEqual(x1, 500)
        self.assertLessEqual(y1, 200)
        self.assertGreaterEqual(x2, 800)
        self.assertGreaterEqual(y2, 700)

    def test_zoom_empty_bays_remaps_and_accepts_crouch(self):
        import numpy as np

        bays = [{"id": "bay_2", "name": "Bay 2", "type": "vehicle_bay", "roi": [0.55, 0.20, 0.35, 0.60]}]
        crop = bay_crop_xyxy(bays[0], 1000, 1000, pad=0.08)
        ox, oy = crop[0], crop[1]
        src = (60.0, 55.0, 200.0, 155.0)
        kpts = _scale_kpts(crouch_sit_keypoints(), (60.0, 55.0, 200.0, 155.0), src)
        result = _FakeResult(
            [_Box(src, 0.7)],
            _Keypoints([kpts]),
        )
        model = _FakePoseModel(result)
        frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
        accepted, rejected = zoom_empty_bays(
            model,
            frame,
            bays,
            [],
            [],
            imgsz=640,
            device="cpu",
            person_conf=0.25,
            min_height_frac=0.05,
            min_aspect=1.1,
            min_keypoints=3,
            kpt_conf=0.25,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertAlmostEqual(accepted[0].x1, src[0] + ox)
        self.assertAlmostEqual(accepted[0].y1, src[1] + oy)

    def test_zoom_shoe_clutter_is_rejected(self):
        import numpy as np

        kpts = shoe_pair_keypoints()
        result = _FakeResult(
            [_Box((60.0, 140.0, 200.0, 280.0), 0.6)],
            _Keypoints([kpts]),
        )
        model = _FakePoseModel(result)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bays = [{"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.10, 0.20, 0.35, 0.60]}]
        accepted, rejected = zoom_empty_bays(
            model,
            frame,
            bays,
            [],
            [],
            imgsz=640,
            device="cpu",
            person_conf=0.25,
            min_height_frac=0.05,
            min_aspect=1.1,
            min_keypoints=3,
            kpt_conf=0.25,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)

    def test_zoom_engine_hallucination_is_rejected(self):
        import numpy as np

        kpts = engine_bay_keypoints()
        result = _FakeResult(
            [_Box((90.0, 40.0, 190.0, 280.0), 0.70)],
            _Keypoints([kpts]),
        )
        model = _FakePoseModel(result)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bays = [{"id": "bay_2", "name": "Bay 2", "type": "vehicle_bay", "roi": [0.10, 0.20, 0.35, 0.60]}]
        accepted, rejected = zoom_empty_bays(
            model,
            frame,
            bays,
            [],
            [],
            imgsz=640,
            device="cpu",
            person_conf=0.25,
            min_height_frac=0.05,
            min_aspect=1.1,
            min_keypoints=3,
            kpt_conf=0.25,
        )
        self.assertEqual(accepted, [])
        self.assertGreaterEqual(len(rejected), 1)


class PersonDetectionSplitTests(unittest.TestCase):
    def test_person_detections_split_returns_low_score_boxes(self):
        from person import person_detections_split

        kpts_data = [standing_person_keypoints(), standing_person_keypoints()]
        boxes = [
            _Box((50, 10, 150, 280), 0.85),
            _Box((200, 10, 300, 280), 0.18),
        ]
        res = _FakeResult(boxes, _Keypoints(kpts_data))
        accepted, rejected, low = person_detections_split(
            res, 480, conf_min=0.25, track_low_thresh=0.10
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].conf, 0.85)
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0].conf, 0.18)
        self.assertFalse(low[0].accepted)


class SkeletonRenderTests(unittest.TestCase):
    def test_draw_skeleton_skips_out_of_box_joints(self):
        import numpy as np

        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        kpts = standing_person_keypoints()
        kpts[9] = (220.0, 20.0, 0.95)
        kpts[10] = (230.0, 18.0, 0.95)
        bbox = (50.0, 10.0, 150.0, 180.0)
        draw_skeleton(frame, kpts, kpt_conf=0.25, bbox=bbox)
        self.assertEqual(int(frame[20, 220].sum()), 0)
        self.assertEqual(int(frame[18, 230].sum()), 0)
        self.assertGreater(int(frame.sum()), 0)

    def test_draw_skeleton_floors_keypoint_confidence(self):
        import numpy as np

        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        kpts = [(0.0, 0.0, 0.0)] * 17
        weak = DRAW_KPT_FLOOR - 0.05
        kpts[5] = (80.0, 80.0, weak)
        kpts[6] = (120.0, 80.0, weak)
        kpts[11] = (85.0, 150.0, weak)
        kpts[12] = (115.0, 150.0, weak)
        draw_skeleton(frame, kpts, kpt_conf=0.25, bbox=(50.0, 10.0, 150.0, 180.0))
        self.assertEqual(int(frame.sum()), 0)


class VehicleCorroborationTests(unittest.TestCase):
    def test_engine_inside_car_is_vetoed(self):
        kpts = engine_bay_keypoints()
        det = Detection(200.0, 80.0, 280.0, 150.0, 0.75, kpts)
        det.accepted = True
        car = VehicleDetection(100.0, 50.0, 450.0, 360.0, 0.90, "car")
        kept, vetoed = veto_vehicle_interior([det], [car], kpt_conf=0.35)
        self.assertEqual(kept, [])
        self.assertEqual(len(vetoed), 1)
        self.assertFalse(vetoed[0].accepted)

    def test_hood_lean_mechanic_reaching_floor_is_kept(self):
        kpts = hood_lean_keypoints()
        det = Detection(300.0, 80.0, 400.0, 380.0, 0.85, kpts)
        det.accepted = True
        car = VehicleDetection(200.0, 50.0, 500.0, 300.0, 0.90, "car")
        kept, vetoed = veto_vehicle_interior([det], [car], kpt_conf=0.35)
        self.assertEqual(len(kept), 1)
        self.assertEqual(vetoed, [])

    def test_named_staff_inside_vehicle_is_not_vetoed(self):
        kpts = engine_bay_keypoints()
        det = Detection(200.0, 80.0, 280.0, 150.0, 0.75, kpts)
        det.accepted = True
        det.is_staff = True
        det.identity = "George"
        car = VehicleDetection(100.0, 50.0, 450.0, 360.0, 0.90, "car")
        kept, vetoed = veto_vehicle_interior([det], [car], kpt_conf=0.35)
        self.assertEqual(len(kept), 1)
        self.assertEqual(vetoed, [])

    def test_unverified_seated_box_inside_lifted_car_is_kept(self):
        from occupancy import overhead_sitting_keypoints

        kpts = overhead_sitting_keypoints()
        det = Detection(200.0, 120.0, 340.0, 250.0, 0.75, kpts)
        det.accepted = True
        det.is_staff = False
        det.identity = "Employee"
        car = VehicleDetection(180.0, 80.0, 450.0, 280.0, 0.90, "car")
        kept, vetoed = veto_vehicle_interior([det], [car], kpt_conf=0.35)
        self.assertEqual(len(kept), 1)
        self.assertEqual(vetoed, [])

    def test_protected_track_inside_vehicle_is_not_vetoed(self):
        kpts = engine_bay_keypoints()
        det = Detection(200.0, 80.0, 280.0, 150.0, 0.75, kpts)
        det.accepted = True
        det.track_id = 4
        car = VehicleDetection(100.0, 50.0, 450.0, 360.0, 0.90, "car")
        kept, vetoed = veto_vehicle_interior(
            [det], [car], kpt_conf=0.35, protected_ids={4}
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(vetoed, [])


if __name__ == "__main__":
    unittest.main()
