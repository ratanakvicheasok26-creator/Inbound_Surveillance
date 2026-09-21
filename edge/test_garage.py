"""Garage operations: multi-bay pose, attendance, Wi-Fi, scorecards, APIs."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import (
    close_bay_sessions,
    complete_vehicle_job,
    connect,
    get_daily_garage_summary,
    get_or_create_vehicle_job,
    get_vehicle_job_history,
    list_vehicle_jobs,
    record_face_clock_in,
    record_face_clock_out,
    update_technician_activity,
    update_vehicle_job_activity,
)
from occupancy import (
    DEFAULT_BAYS,
    UNKNOWN_WORKER,
    BayZoneManager,
    crouching_pose_keypoints,
    idle_standing_keypoints,
    is_phone_usage_pose,
    is_sitting_pose,
    is_under_vehicle_pose,
    is_working_pose,
    next_available_bay_name,
    normalize_bays,
    overhead_sitting_keypoints,
    partial_legs_pose_keypoints,
    point_in_polygon,
    point_in_roi,
    roi_as_polygon,
    under_vehicle_pose_keypoints,
    working_pose_keypoints,
)
from person import Detection, draw_detection, draw_skeleton, is_human_pose
import numpy as np
from report import build_garage_report, efficiency_badge
from sensors.wifi_tracker import WifiTracker, parse_arp_table, presence_status
from service_patterns import (
    KNOWLEDGE_BASE,
    calculate_performance_grade,
    evaluate_completed_vehicle_job,
)
from vehicle import VehicleDetection, extract_vehicle_detections


class _Det:
    def __init__(
        self,
        kpts,
        *,
        x1=180,
        y1=220,
        x2=320,
        y2=620,
        name="Hour-Meng",
        staff=True,
        hits=3,
        clutter=False,
        liveness=None,
        jitter=None,
        motion=None,
        track_id=None,
    ):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = 0.9
        self.keypoints = kpts
        self.accepted = True
        self.identity = name
        self.identity_conf = 0.9
        self.is_staff = staff
        self.hits = hits
        self.clutter = clutter
        self.liveness = liveness
        self.jitter = jitter
        self.motion = motion
        self.track_id = track_id

    def box(self):
        return self.x1, self.y1, self.x2, self.y2

    def in_roi(self, roi_px, kpt_conf=0.4):
        from occupancy import box_center_in_roi

        return box_center_in_roi(self.box(), roi_px)


def _shift_kpts(kpts, dx: float, dy: float):
    return [(x + dx, y + dy, c) for x, y, c in kpts]


def _det(kpts, **kwargs) -> _Det:
    return _Det(kpts, **kwargs)


def phone_usage_keypoints() -> list[tuple[float, float, float]]:
    """Synthetic: worker looking down at mobile phone with converged wrists."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (100.0, 85.0, 0.9)   # Nose (tilted down towards chest)
    pts[1] = (95.0, 75.0, 0.9)    # L Eye
    pts[2] = (105.0, 75.0, 0.9)   # R Eye
    pts[5] = (80.0, 95.0, 0.9)    # L Shoulder
    pts[6] = (120.0, 95.0, 0.9)   # R Shoulder
    pts[7] = (75.0, 130.0, 0.85)  # L Elbow
    pts[8] = (125.0, 130.0, 0.85) # R Elbow
    pts[9] = (98.0, 120.0, 0.9)   # L Wrist (converged in front of chest)
    pts[10] = (102.0, 120.0, 0.9) # R Wrist (converged in front of chest)
    pts[11] = (85.0, 180.0, 0.8)  # L Hip
    pts[12] = (115.0, 180.0, 0.8) # R Hip
    pts[13] = (85.0, 240.0, 0.8)  # L Knee
    pts[14] = (115.0, 240.0, 0.8) # R Knee
    return pts


def sitting_keypoints() -> list[tuple[float, float, float]]:
    """Synthetic: worker seated with thighs roughly horizontal."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[0] = (100.0, 60.0, 0.9)   # Nose
    pts[5] = (80.0, 90.0, 0.9)    # L Shoulder
    pts[6] = (120.0, 90.0, 0.9)   # R Shoulder
    pts[11] = (85.0, 160.0, 0.9)  # L Hip
    pts[12] = (115.0, 160.0, 0.9) # R Hip
    pts[13] = (135.0, 165.0, 0.9) # L Knee (horizontal: y~165, x~135)
    pts[14] = (155.0, 165.0, 0.9) # R Knee (horizontal: y~165, x~155)
    pts[15] = (135.0, 220.0, 0.8) # L Ankle
    pts[16] = (155.0, 220.0, 0.8) # R Ankle
    return pts


class PoseAndRoiTests(unittest.TestCase):
    def test_point_in_polygon_and_roi(self):
        roi = [0.10, 0.20, 0.35, 0.60]
        poly = roi_as_polygon(roi)
        self.assertTrue(point_in_roi(0.20, 0.40, roi))
        self.assertTrue(point_in_polygon(0.20, 0.40, poly))
        self.assertFalse(point_in_roi(0.90, 0.10, roi))
        self.assertFalse(point_in_polygon(0.90, 0.10, poly))
        self.assertTrue(point_in_polygon(0.60, 0.40, roi_as_polygon(DEFAULT_BAYS[1]["roi"])))
        self.assertTrue(point_in_roi(0.50, 0.12, DEFAULT_BAYS[2]["roi"]))

    def test_next_available_bay_name_fills_gap(self):
        remaining = [
            {"id": "bay_1", "name": "Bay 1"},
            {"id": "bay_2", "name": "Bay 2"},
            {"id": "bay_4", "name": "Bay 4"},
            {"id": "bay_5", "name": "Bay 5"},
        ]
        self.assertEqual(next_available_bay_name(remaining), "Bay 3")
        deleted = [b for b in remaining if b["id"] != "bay_4"]
        names = [b["name"] for b in deleted]
        self.assertEqual(names, ["Bay 1", "Bay 2", "Bay 5"])
        self.assertEqual(next_available_bay_name(deleted), "Bay 3")

    def test_delete_bay_keeps_sibling_metrics(self):
        manager = BayZoneManager(
            [
                {"id": "bay_1", "name": "Bay 1", "roi": [0.10, 0.20, 0.35, 0.60], "type": "vehicle_bay"},
                {"id": "bay_2", "name": "Bay 2", "roi": [0.55, 0.20, 0.35, 0.60], "type": "vehicle_bay"},
                {"id": "bay_3", "name": "Bay 3", "roi": [0.10, 0.70, 0.20, 0.20], "type": "vehicle_bay"},
            ],
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280))
        t0 = 50.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        wrench_before = {s.bay_id: s.wrench_seconds for s in manager.snapshots()}
        manager.set_bays(
            [
                {"id": "bay_1", "name": "Bay 1", "roi": [0.10, 0.20, 0.35, 0.60], "type": "vehicle_bay"},
                {"id": "bay_2", "name": "Bay 2", "roi": [0.55, 0.20, 0.35, 0.60], "type": "vehicle_bay"},
            ]
        )
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertNotIn("bay_3", snaps)
        self.assertEqual(snaps["bay_1"].name, "Bay 1")
        self.assertEqual(snaps["bay_2"].name, "Bay 2")
        self.assertGreater(snaps["bay_1"].wrench_seconds, 0)
        self.assertEqual(snaps["bay_1"].wrench_seconds, wrench_before["bay_1"])

    def test_close_bay_sessions_preserves_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "events.db")
            ended = datetime(2026, 9, 1, 8, 0, 0)
            opened = datetime(2026, 9, 1, 10, 0, 0)
            now = datetime(2026, 9, 1, 11, 0, 0)
            conn.execute(
                """
                INSERT INTO bay_sessions (bay_id, technician_name, start_time, end_time, active_duration)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("bay_3", "Hour-Meng", ended.isoformat(timespec="seconds"), ended.isoformat(timespec="seconds"), 12.0),
            )
            conn.execute(
                """
                INSERT INTO bay_sessions (bay_id, technician_name, start_time, active_duration)
                VALUES (?, ?, ?, ?)
                """,
                ("bay_3", "Hour-Meng", opened.isoformat(timespec="seconds"), 4.0),
            )
            conn.commit()
            closed = close_bay_sessions(conn, ["bay_3"], now)
            self.assertEqual(closed, 1)
            rows = list(conn.execute("SELECT * FROM bay_sessions WHERE bay_id = ? ORDER BY id", ("bay_3",)))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["end_time"], ended.isoformat(timespec="seconds"))
            self.assertEqual(rows[0]["active_duration"], 12.0)
            self.assertIsNotNone(rows[1]["end_time"])
            self.assertEqual(rows[1]["active_duration"], 4.0)
            conn.close()

    def test_normalize_bays_keeps_explicit_empty(self):
        self.assertEqual(normalize_bays([], seed_if_empty=False), [])
        seeded = normalize_bays([])
        self.assertGreaterEqual(len(seeded), 3)

    def test_pose_classification(self):
        self.assertTrue(is_working_pose(working_pose_keypoints()))
        self.assertTrue(is_working_pose(crouching_pose_keypoints()))
        self.assertFalse(is_working_pose(idle_standing_keypoints()))
        self.assertTrue(is_under_vehicle_pose(under_vehicle_pose_keypoints()))
        self.assertTrue(is_under_vehicle_pose(partial_legs_pose_keypoints()))
        self.assertFalse(is_under_vehicle_pose(idle_standing_keypoints()))
        self.assertFalse(is_under_vehicle_pose(sitting_keypoints()))
        self.assertFalse(is_under_vehicle_pose(overhead_sitting_keypoints()))
        self.assertTrue(is_sitting_pose(sitting_keypoints()))
        self.assertTrue(is_sitting_pose(overhead_sitting_keypoints()))
        self.assertFalse(is_sitting_pose(under_vehicle_pose_keypoints()))
        self.assertFalse(is_sitting_pose(partial_legs_pose_keypoints()))

    def test_under_vehicle_occlusion_and_dwell_grace_period(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=5.0,
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(under_vehicle_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0

        # 1. Mechanic starts working under vehicle (horizontal pose inside bay)
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "UNDER_VEHICLE")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreaterEqual(snaps["bay_1"].wrench_seconds, 1.9)
        self.assertGreaterEqual(snaps["bay_1"].under_vehicle_seconds, 1.9)

        # 2. Brief occlusion (ducking under the car) stays UNDER_VEHICLE
        # last seen at t0+2, now t0+5 => 3s < 5s under_car_grace
        manager.update([], 1000, 1000, t0 + 5.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "UNDER_VEHICLE")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreater(snaps["bay_1"].under_vehicle_seconds, 2.0)

        # 3. Past under_car_grace_seconds -> ON_BREAK and wrench timer pauses
        manager.update([], 1000, 1000, t0 + 8.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        wrench_at_break = snaps["bay_1"].wrench_seconds
        manager.update([], 1000, 1000, t0 + 10.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")
        self.assertAlmostEqual(snaps["bay_1"].wrench_seconds, wrench_at_break, places=1)

    def test_break_and_resume_continuity(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=5.0,
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        wrench_before_break = {s.bay_id: s.wrench_seconds for s in manager.snapshots()}["bay_1"]
        self.assertGreater(wrench_before_break, 1.5)

        # Mechanic steps away past under_car_grace_seconds (5s) -> transitions to ON_BREAK
        manager.update([], 1000, 1000, t0 + 10.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")

        # Mechanic returns to bay -> resumes WORKING without resetting wrench_seconds
        manager.update([work], 1000, 1000, t0 + 15.0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 17.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreater(snaps["bay_1"].wrench_seconds, wrench_before_break + 1.5)

    def test_working_occlusion_grace_holds_state(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=5.0,
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        wrench_before = {s.bay_id: s.wrench_seconds for s in manager.snapshots()}["bay_1"]

        manager.update([], 1000, 1000, t0 + 5.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreater(snaps["bay_1"].wrench_seconds, wrench_before)

        manager.update([], 1000, 1000, t0 + 8.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")

    def test_under_car_grace_holds_twenty_seconds_then_breaks(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=30.0,
            break_timeout_seconds=3600.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        wrench_before = {s.bay_id: s.wrench_seconds for s in manager.snapshots()}["bay_1"]

        # 20s of miss is still inside the 30s ducking grace — wrench keeps running.
        manager.update([], 1000, 1000, t0 + 22.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertTrue(snaps["bay_1"].session_open)
        self.assertGreater(snaps["bay_1"].wrench_seconds, wrench_before)

        manager.update([], 1000, 1000, t0 + 33.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")
        wrench_at_break = snaps["bay_1"].wrench_seconds
        manager.update([], 1000, 1000, t0 + 35.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")
        self.assertAlmostEqual(snaps["bay_1"].wrench_seconds, wrench_at_break, places=1)

    def test_multi_bay_states_and_wrench_dt(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            idle_stationary_seconds=0.4,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280))
        idle = _det(_shift_kpts(idle_standing_keypoints(), 500, 280), name="Sothun", x1=620, y1=220, x2=760, y2=620)
        t0 = 100.0
        manager.update([work, idle], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work, idle], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreater(snaps["bay_1"].wrench_seconds, 0.9)
        self.assertEqual(snaps["tools"].state, "EMPTY")
        manager.update([work, idle], 1000, 1000, t0 + 1.6, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertIn(snaps["bay_2"].state, ("WORKING", "IDLE"))
        self.assertGreater(snaps["bay_2"].idle_seconds + snaps["bay_2"].wrench_seconds, 0.4)
        empty = manager.update([], 1000, 1000, t0 + 3700.0, kpt_conf=0.4)
        by_id = {s.bay_id: s.state for s in empty}
        self.assertEqual(by_id["bay_1"], "EMPTY")
        self.assertEqual(by_id["bay_2"], "EMPTY")
        self.assertEqual(by_id["tools"], "EMPTY")

    def test_tool_station_time_counting(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        # Position a worker inside the Tool Station box (x=450, y=100)
        tool_worker = _det(_shift_kpts(idle_standing_keypoints(), 430, 60), name="Hour-Meng", x1=430, y1=60, x2=530, y2=220)
        t0 = 100.0

        # Worker is in Tool Station -> counts time!
        manager.update([tool_worker], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([tool_worker], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        manager.update([tool_worker], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        manager.update([tool_worker], 1000, 1000, t0 + 3.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["tools"].state, "WORKING")
        self.assertEqual(snaps["tools"].mechanic_name, "Hour-Meng")
        self.assertGreaterEqual(snaps["tools"].wrench_seconds, 2.9)

        # Worker steps away from Tool Station -> pauses time!
        manager.update([], 1000, 1000, t0 + 4.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["tools"].state, "ON_BREAK")
        self.assertAlmostEqual(snaps["tools"].wrench_seconds, 3.0, places=1)

    def test_multiple_employees_in_same_bay(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        # 2 workers inside Bay 1: Hour-Meng and Sothun
        w1 = _det(_shift_kpts(working_pose_keypoints(), 50, 250), name="Hour-Meng")
        w2 = _det(_shift_kpts(working_pose_keypoints(), 120, 300), name="Sothun")
        t0 = 100.0

        manager.update([w1, w2], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([w1, w2], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        manager.update([w1, w2], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}

        # Verify both technicians are tracked
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertIn("Hour-Meng", snaps["bay_1"].mechanic_name)
        self.assertIn("Sothun", snaps["bay_1"].mechanic_name)
        self.assertIn("Hour-Meng", snaps["bay_1"].technicians_times)
        self.assertIn("Sothun", snaps["bay_1"].technicians_times)
        self.assertGreaterEqual(snaps["bay_1"].technicians_times["Hour-Meng"], 1.9)
        self.assertGreaterEqual(snaps["bay_1"].technicians_times["Sothun"], 1.9)

        # Check the formatted badge includes both names and their individual times
        badge = snaps["bay_1"].as_dict()["badge"]
        self.assertIn("Hour-Meng", badge)
        self.assertIn("Sothun", badge)

    def test_auto_vehicle_detection_and_dynamic_bay_sync(self):
        manager = BayZoneManager(
            [],  # Empty initial manual bays
            break_timeout_seconds=60.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        # Simulate auto-detected vehicle at x=200..700, y=300..800
        car = VehicleDetection(
            x1=200.0,
            y1=300.0,
            x2=700.0,
            y2=800.0,
            conf=0.88,
            vehicle_type="car",
            vehicle_id="auto_car_1",
        )
        manager.sync_auto_vehicles([car], frame_w=1000, frame_h=1000)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertIn("auto_car_1", snaps)
        self.assertEqual(snaps["auto_car_1"].name, "Auto: Car #1")
        self.assertEqual(snaps["auto_car_1"].type, "vehicle_bay")

        # Now simulate a technician working near/under that auto-detected car
        tech = _det(_shift_kpts(working_pose_keypoints(), 350, 450), name="Hour-Meng")
        t0 = 100.0
        manager.update([tech], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([tech], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        manager.update([tech], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["auto_car_1"].state, "WORKING")
        self.assertEqual(snaps["auto_car_1"].mechanic_name, "Hour-Meng")
        self.assertGreaterEqual(snaps["auto_car_1"].wrench_seconds, 1.9)

        # Now simulate the car driving away / departing after work is done
        departed = manager.sync_auto_vehicles([], 1000, 1000, now=t0 + 20.0)
        self.assertIn("auto_car_1", departed)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["auto_car_1"].state, "EMPTY")

    def test_vehicle_service_performance_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "test_eval.db")
            job_id = "JOB-auto_car_1-20260902-01"
            now_str = datetime.now().isoformat(timespec="seconds")

            # Create a completed vehicle job with 1200s active, 200s break
            conn.execute(
                """
                INSERT INTO vehicle_jobs (
                    job_id, bay_id, vehicle_type, primary_technician,
                    status, total_active_seconds, total_break_seconds,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, "auto_car_1", "car", "Hour-Meng", "COMPLETED", 1200.0, 200.0, now_str, now_str, now_str),
            )
            # Add daily technician log
            conn.execute(
                """
                INSERT INTO daily_vehicle_job_logs (job_id, day, technician_name, active_seconds, break_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, "2026-09-02", "Hour-Meng", 1200.0, 200.0),
            )
            conn.commit()

            report = evaluate_completed_vehicle_job(conn, job_id)
            self.assertIsNotNone(report)
            self.assertEqual(report.job_id, job_id)
            self.assertEqual(report.primary_technician, "Hour-Meng")
            self.assertEqual(report.performance_grade, "A+")
            self.assertGreaterEqual(report.performance_score, 90)
            self.assertAlmostEqual(report.efficiency_pct, (1200.0 / 1400.0) * 100.0, places=1)
            self.assertIn("Hour-Meng", report.technicians_breakdown)

            # Test Knowledge Base
            self.assertIsNotNone(KNOWLEDGE_BASE.get_template("oil_change"))
            brake_tmpl = KNOWLEDGE_BASE.get_template("brake_rotor_and_pad_replacement")
            self.assertIsNotNone(brake_tmpl)
            self.assertEqual(len(brake_tmpl["stages"]), 7)
            self.assertEqual(brake_tmpl["target_minutes"], 35.0)
            self.assertIn("12mm socket & ratchet", brake_tmpl["tools_required"])
            conn.close()

    def test_parked_waiting_queue_time_and_work_transition(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        t0 = 100.0
        # 1. Car is in Bay 1 (vehicle_present=True), but no mechanic is inside (customer check-in)
        manager._bays[0].vehicle_present = True
        manager.update([], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([], 1000, 1000, t0 + 10.0, kpt_conf=0.4)
        
        bay1 = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(bay1.state, "PARKED_WAITING")
        self.assertAlmostEqual(bay1.queue_seconds, 2.0, places=1)
        self.assertEqual(bay1.wrench_seconds, 0.0) # Work timer stays frozen at 0!

        # 2. Mechanic enters and starts wrenching
        work_det = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng")
        manager.update([work_det], 1000, 1000, t0 + 11.0, kpt_conf=0.4)
        manager.update([work_det], 1000, 1000, t0 + 12.0, kpt_conf=0.4)

        bay1_active = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(bay1_active.state, "WORKING")
        self.assertGreater(bay1_active.wrench_seconds, 0.0)
        self.assertIn("Hour-Meng", bay1_active.mechanic_name)

    def test_head_turn_keeps_locked_technician_timer(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="George")
        t0 = 20.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        turned = _det(
            _shift_kpts(working_pose_keypoints(), 80, 280),
            name="Employee",
            staff=False,
        )
        manager.update([turned], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "WORKING")
        self.assertEqual(snap.mechanic_name, "George")
        self.assertIn("George", snap.technicians_times)
        self.assertNotIn("Employee", snap.technicians_times)

    def test_unverified_occupant_does_not_accrue_wrench(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        ghost = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Employee", staff=False)
        t0 = 30.0
        manager.update([ghost], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([ghost], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "WORKING")
        self.assertEqual(snap.mechanic_name, UNKNOWN_WORKER)
        self.assertEqual(snap.wrench_seconds, 0.0)
        self.assertGreater(snap.unverified_seconds, 0.0)
        self.assertNotIn(UNKNOWN_WORKER, snap.technicians_times)
        badge = snap.as_dict()["badge"]
        self.assertIn("WORKING", badge)
        self.assertIn("unverified", badge)

    def test_unverified_time_backfills_on_identity(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        ghost = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Employee", staff=False)
        t0 = 40.0
        manager.update([ghost], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([ghost], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        named = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="George", staff=True)
        manager.update([named], 1000, 1000, t0 + 3.0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.mechanic_name, "George")
        self.assertIn("George", snap.technicians_times)
        self.assertEqual(snap.unverified_seconds, 0.0)
        self.assertGreater(snap.wrench_seconds, 0.0)
        self.assertNotIn(UNKNOWN_WORKER, snap.technicians_times)
        self.assertNotIn("unverified", snap.as_dict()["badge"])

    def test_idle_state_and_idle_seconds_accumulation(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            idle_stationary_seconds=0.5,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        idle_worker = _det(_shift_kpts(idle_standing_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 50.0

        # 1. Worker enters and stands idle in Bay 1
        manager.update([idle_worker], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([idle_worker], 1000, 1000, t0 + 1.0, kpt_conf=0.4)
        manager.update([idle_worker], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "IDLE")
        self.assertGreaterEqual(snap.idle_seconds, 1.0)
        wrench_at_idle = snap.wrench_seconds

        # 2. Worker begins active wrenching
        work_worker = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng")
        manager.update([work_worker], 1000, 1000, t0 + 3.0, kpt_conf=0.4)
        manager.update([work_worker], 1000, 1000, t0 + 4.0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "WORKING")
        self.assertGreater(snap.wrench_seconds, wrench_at_idle)

    def test_detection_and_skeleton_rendering(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = working_pose_keypoints()
        det = Detection(
            x1=100.0,
            y1=100.0,
            x2=200.0,
            y2=300.0,
            conf=0.88,
            keypoints=kpts,
            accepted=True,
            identity="Hour-Meng",
            is_staff=True,
            active_time_str="12m 30s",
        )
        draw_detection(frame, det, in_roi=True, kpt_conf=0.30)
        self.assertGreater(int(frame.sum()), 0)

        # Draw rejected blob
        det_rej = Detection(x1=50.0, y1=50.0, x2=80.0, y2=80.0, conf=0.20, accepted=False)
        draw_detection(frame, det_rej, in_roi=False)



class AttendanceAndScorecardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "events.db")

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_clock_in_out_and_shift_hours(self):
        start = datetime(2026, 9, 1, 8, 5, 0)
        record_face_clock_in(self.conn, "Hour-Meng", start)
        again = record_face_clock_in(self.conn, "Hour-Meng", start + timedelta(minutes=10))
        self.assertFalse(again["created"])
        record_face_clock_out(self.conn, "Hour-Meng", start + timedelta(hours=8, minutes=5))
        summary = get_daily_garage_summary(self.conn, start.date())
        tech = summary["technicians"][0]
        self.assertEqual(tech["staff_name"], "Hour-Meng")
        self.assertAlmostEqual(tech["total_shift_minutes"], 485.0, places=0)
        self.assertFalse(tech["clocked_in"])

    def test_wrench_percent_and_telegram_scorecard(self):
        day = datetime(2026, 9, 1, 8, 0, 0)
        record_face_clock_in(self.conn, "Hour-Meng", day)
        update_technician_activity(self.conn, "Hour-Meng", "bay_1", True, 180 * 60, day + timedelta(minutes=1))
        update_technician_activity(self.conn, "Hour-Meng", "bay_1", False, 60 * 60, day + timedelta(hours=4))
        record_face_clock_out(self.conn, "Hour-Meng", day + timedelta(hours=6))
        summary = get_daily_garage_summary(
            self.conn, day.date(), open_time="08:00", close_time="18:00", bay_ids=["bay_1", "bay_2", "tools"]
        )
        tech = summary["technicians"][0]
        self.assertAlmostEqual(tech["performance_score"], 50.0, places=0)
        self.assertEqual(efficiency_badge(tech["performance_score"]), "🟡 Normal")
        text, _paths = build_garage_report(
            self.conn, day.date(), "Demo Garage", "08:00", close_time="18:00",
            bay_ids=["bay_1", "bay_2", "tools"],
        )
        self.assertIn("Demo Garage", text)
        self.assertIn("Hour-Meng", text)
        self.assertIn("Wrench", text)
        self.assertIn("bay_1", text)
        self.assertGreaterEqual(len(summary["bays"]), 3)

    def test_phone_usage_pose_and_state(self):
        self.assertTrue(is_phone_usage_pose(phone_usage_keypoints()))
        self.assertFalse(is_phone_usage_pose(working_pose_keypoints()))

        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        for b in manager._bays:
            b.phone_threshold_seconds = 1.0

        phone_worker = _det(_shift_kpts(phone_usage_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0

        manager.update([phone_worker], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([phone_worker], 1000, 1000, t0 + 1.5, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "NOT_WORKING")
        self.assertEqual(snap.not_working_reason, "PHONE")
        self.assertIn("NOT WORKING (PHONE)", snap.as_dict()["badge"])

    def test_sitting_pose_and_state(self):
        self.assertTrue(is_sitting_pose(sitting_keypoints()))
        self.assertFalse(is_sitting_pose(working_pose_keypoints()))
        self.assertFalse(is_working_pose(sitting_keypoints()))

        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        for b in manager._bays:
            b.sitting_threshold_seconds = 1.0

        sitting_worker = _det(_shift_kpts(sitting_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0

        manager.update([sitting_worker], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([sitting_worker], 1000, 1000, t0 + 1.5, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "NOT_WORKING")
        self.assertEqual(snap.not_working_reason, "SITTING")
        self.assertIn("NOT WORKING (SITTING)", snap.as_dict()["badge"])

    def test_sitting_to_work_under_vehicle_stays_working(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        for b in manager._bays:
            b.sitting_threshold_seconds = 1.0

        under_car = _det(_shift_kpts(under_vehicle_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0
        manager.update([under_car], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([under_car], 1000, 1000, t0 + 1.5, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "UNDER_VEHICLE")
        self.assertNotEqual(snap.not_working_reason, "SITTING")

    def test_crouch_to_adjust_car_stays_working(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        for b in manager._bays:
            b.sitting_threshold_seconds = 1.0

        crouch = _det(_shift_kpts(crouching_pose_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0
        manager.update([crouch], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([crouch], 1000, 1000, t0 + 1.5, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "WORKING")
        self.assertIsNone(snap.not_working_reason)

    def test_default_to_working_in_bay(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        natural_worker = _det(_shift_kpts(idle_standing_keypoints(), 80, 280), name="Hour-Meng")
        t0 = 100.0

        manager.update([natural_worker], 1000, 1000, t0, kpt_conf=0.4)
        snap = {s.bay_id: s for s in manager.snapshots()}["bay_1"]
        self.assertEqual(snap.state, "WORKING")
        self.assertEqual(snap.mechanic_name, "Hour-Meng")

    def test_bending_mechanic_aspect_ratio_not_dropped(self):
        kpts = working_pose_keypoints()
        accepted = is_human_pose(
            x1=50.0, y1=100.0, x2=210.0, y2=220.0,
            keypoints=kpts, frame_h=1000,
            min_height_frac=0.12, min_aspect=1.1, min_keypoints=4, kpt_conf=0.4
        )
        self.assertTrue(accepted)


class WifiPresenceTests(unittest.TestCase):
    def test_arp_parse_and_two_factor(self):
        table = (
            "IP address       HW type     Flags       HW address            Mask     Device\n"
            "192.168.1.42     0x1         0x2         aa:bb:cc:dd:ee:ff     *        wlan0\n"
            "host (192.168.1.50) at 11:22:33:44:55:66 [ether] on wlan0\n"
        )
        rows = parse_arp_table(table)
        macs = {row["mac"] for row in rows}
        self.assertIn("aa:bb:cc:dd:ee:ff", macs)
        self.assertIn("11:22:33:44:55:66", macs)
        tracker = WifiTracker(
            [{"name": "Hour-Meng", "mac": "AA-BB-CC-DD-EE-FF", "ip": "192.168.1.42"}],
            interval=30,
        )
        present = tracker.scan_once(table)
        self.assertTrue(present[0].connected)
        self.assertTrue(tracker.is_connected("Hour-Meng"))
        self.assertEqual(presence_status(True, True), "confirmed")
        self.assertEqual(presence_status(True, False), "face_only")
        self.assertEqual(presence_status(False, True), "wifi_only")
        self.assertEqual(presence_status(False, False), "off_site")
        gone = tracker.scan_once("192.168.1.1     0x1         0x2         00:11:22:33:44:55     *        wlan0\n")
        self.assertFalse(gone[0].connected)
        dropped = tracker.departures()
        self.assertEqual(dropped, ["Hour-Meng"])


class GarageApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from launcher import DashboardRequestHandler, GLOBAL_ENGINE, ThreadingHTTPServer

        GLOBAL_ENGINE.bay_manager.set_bays(DEFAULT_BAYS)
        GLOBAL_ENGINE.bay_telemetry = GLOBAL_ENGINE.bay_manager.telemetry()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardRequestHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _get(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        return resp.status, body

    def test_telemetry_and_scorecard_under_5ms(self):
        from launcher import GLOBAL_ENGINE

        t0 = time.perf_counter()
        payload = GLOBAL_ENGINE.garage_telemetry()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(elapsed_ms, 5.0)
        self.assertGreaterEqual(len(payload["bays"]), 3)
        ids = {b["bay_id"] for b in payload["bays"]}
        self.assertTrue({"bay_1", "bay_2", "tools"}.issubset(ids))
        for bay in payload["bays"]:
            self.assertIn(bay["state"], ("WORKING", "IDLE", "EMPTY"))

        status, body = self._get("/api/garage/telemetry")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertIn("bays", data)
        self.assertIn("fps", data)

        status, body = self._get("/api/garage/scorecard")
        self.assertEqual(status, 200)
        card = json.loads(body)
        self.assertIn("technicians", card)
        self.assertIn("bays", card)
        self.assertIn("attendance_logs", card)

    def test_workplace_visits_api_survives_worker_thread_conn(self):
        """HTTP thread must not use the camera-loop sqlite connection (check_same_thread)."""
        from launcher import GLOBAL_ENGINE

        prev_wp = GLOBAL_ENGINE.cfg.get("workplace_type")
        prev_conn = GLOBAL_ENGINE.conn
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Same-thread = True: any query from the HTTP server thread would raise.
        worker_conn = connect(Path(tmp.name) / "worker_events.db", check_same_thread=True)
        self.addCleanup(worker_conn.close)
        GLOBAL_ENGINE.conn = worker_conn
        GLOBAL_ENGINE.cfg["workplace_type"] = "massage"
        try:
            status, body = self._get("/api/workplace/visits")
            self.assertEqual(status, 200, body[:200] if body else "")
            data = json.loads(body)
            self.assertIn("today_unique", data)
            self.assertIn("today_visits", data)
            self.assertIn("week_unique", data)
            self.assertIn("week_visits", data)
            self.assertIn("open_sessions", data)
            self.assertIsInstance(data["open_sessions"], list)

            # Telemetry poll must also stay SQLite-free in massage mode.
            status, body = self._get("/api/garage/telemetry")
            self.assertEqual(status, 200, body[:200] if body else "")
            tel = json.loads(body)
            self.assertEqual(tel.get("workplace_type"), "massage")
            self.assertIn("bays", tel)
            self.assertIn("open_sessions", tel)

            # Direct call must not touch GLOBAL_ENGINE.conn either.
            live = GLOBAL_ENGINE.workplace_telemetry()
            self.assertNotIn("visits", live)
            self.assertIn("open_sessions", live)
        finally:
            GLOBAL_ENGINE.conn = prev_conn
            if prev_wp is None:
                GLOBAL_ENGINE.cfg.pop("workplace_type", None)
            else:
                GLOBAL_ENGINE.cfg["workplace_type"] = prev_wp

    def _post(self, path: str, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        conn = HTTPConnection("127.0.0.1", self.port, timeout=2)
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_vehicle_jobs_crud_and_multi_day_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = connect(Path(tmp) / "events.db")
            day1 = datetime(2026, 9, 1, 9, 0, 0)
            day2 = datetime(2026, 9, 2, 9, 0, 0)

            # 1. Create a vehicle job for Bay 1
            job_id = get_or_create_vehicle_job(
                conn,
                bay_id="bay_1",
                vehicle_type="truck",
                vehicle_label="Ford F-150 (Transmission)",
                primary_technician="Hour-Meng",
                timestamp=day1,
            )
            self.assertTrue(job_id.startswith("JOB-bay_1-20260901-"))

            # 2. Day 1: 3.5 hours active work (12,600s) + 30m break (1800s)
            update_vehicle_job_activity(
                conn,
                job_id=job_id,
                active_dt=12600.0,
                break_dt=1800.0,
                technician_name="Hour-Meng",
                timestamp=day1,
                status="WORKING",
            )

            # 3. Day 2: 4.2 hours active work (15,120s)
            update_vehicle_job_activity(
                conn,
                job_id=job_id,
                active_dt=15120.0,
                break_dt=0.0,
                technician_name="Hour-Meng",
                timestamp=day2,
                status="UNDER_VEHICLE",
            )

            # 4. Fetch job history across multiple days
            history = get_vehicle_job_history(conn, job_id)
            self.assertIsNotNone(history)
            self.assertEqual(history["bay_id"], "bay_1")
            self.assertEqual(history["vehicle_type"], "truck")
            self.assertEqual(len(history["daily_logs"]), 2)

            # Total = 12600 + 15120 = 27720s (7.7 hours)
            self.assertAlmostEqual(history["total_active_hours"], 7.7, places=1)
            self.assertEqual(history["daily_logs"][0]["day"], "2026-09-01")
            self.assertEqual(history["daily_logs"][0]["active_hours"], 3.5)
            self.assertEqual(history["daily_logs"][1]["day"], "2026-09-02")
            self.assertEqual(history["daily_logs"][1]["active_hours"], 4.2)

            # 5. Complete job
            ok = complete_vehicle_job(conn, job_id, timestamp=day2)
            self.assertTrue(ok)
            jobs = list_vehicle_jobs(conn, status="COMPLETED")
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["status"], "COMPLETED")
            self.assertIsNotNone(jobs[0]["completed_at"])
            conn.close()

    def test_vehicle_jobs_api(self):
        status, body = self._get("/api/garage/jobs")
        self.assertEqual(status, 200)
        jobs = json.loads(body)
        self.assertIsInstance(jobs, list)

        # Complete non-existent job returns ok=False
        status, body = self._post("/api/garage/jobs/complete", {"job_id": "NON-EXISTENT"})
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertFalse(data["ok"])

    def test_per_camera_bays_isolation(self):
        from launcher import _normalize_cameras, upsert_camera

        bays_cam1 = [
            {"id": "bay_a", "name": "Bay A", "roi": [0.1, 0.1, 0.3, 0.4], "type": "vehicle_bay"},
        ]
        bays_cam2 = [
            {"id": "bay_b", "name": "Bay B", "roi": [0.5, 0.2, 0.4, 0.5], "type": "vehicle_bay"},
            {"id": "tools_2", "name": "Tools 2", "roi": [0.2, 0.05, 0.2, 0.2], "type": "tool_area"},
        ]
        raw_cameras = [
            {"id": "cam-1", "name": "Camera 1", "source": "0", "bays": bays_cam1},
            {"id": "cam-2", "name": "Camera 2", "source": "1", "bays": bays_cam2},
        ]
        normalized = _normalize_cameras(raw_cameras)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(len(normalized[0]["bays"]), 1)
        self.assertEqual(normalized[0]["bays"][0]["id"], "bay_a")
        self.assertEqual(len(normalized[1]["bays"]), 2)
        self.assertEqual(normalized[1]["bays"][0]["id"], "bay_b")

        # Upsert camera without touching bays preserves existing bays
        cfg = {"cameras": normalized, "active_camera_id": "cam-1"}
        updated = upsert_camera(cfg, {"id": "cam-1", "name": "Camera 1 Renamed"})
        self.assertEqual(len(updated["bays"]), 1)
        self.assertEqual(updated["bays"][0]["id"], "bay_a")
        self.assertEqual(len(cfg["cameras"][1]["bays"]), 2)

    def test_per_camera_absent_seconds(self):
        from launcher import _normalize_cameras, upsert_camera, camera_absent_seconds

        raw_cameras = [
            {"id": "cam-1", "name": "Camera 1", "source": "0", "absent_seconds": 20},
            {"id": "cam-2", "name": "Camera 2", "source": "1", "absent_seconds": 10},
            {"id": "cam-3", "name": "Camera 3", "source": "2", "absent_seconds": 5},
        ]
        normalized = _normalize_cameras(raw_cameras, default_absent=30)
        self.assertEqual(normalized[0]["absent_seconds"], 20)
        self.assertEqual(normalized[1]["absent_seconds"], 10)
        self.assertEqual(normalized[2]["absent_seconds"], 5)

        inherited = _normalize_cameras(
            [{"id": "cam-new", "name": "New", "source": "3"}],
            default_absent=15,
        )
        self.assertEqual(inherited[0]["absent_seconds"], 15)

        cfg = {"cameras": normalized, "active_camera_id": "cam-1", "absent_seconds": 30}
        self.assertEqual(camera_absent_seconds(cfg, "cam-1"), 20)
        self.assertEqual(camera_absent_seconds(cfg, "cam-2"), 10)
        self.assertEqual(camera_absent_seconds(cfg, "cam-3"), 5)
        self.assertEqual(camera_absent_seconds(cfg), 20)

        updated = upsert_camera(cfg, {"id": "cam-2", "name": "Camera 2 Renamed", "source": "1"})
        self.assertEqual(updated["absent_seconds"], 10)
        self.assertEqual(cfg["cameras"][0]["absent_seconds"], 20)
        self.assertEqual(cfg["cameras"][2]["absent_seconds"], 5)

        changed = upsert_camera(
            cfg,
            {"id": "cam-3", "name": "Camera 3", "source": "2", "absent_seconds": 8},
        )
        self.assertEqual(changed["absent_seconds"], 8)

    def test_import_bays(self):
        from launcher import LiveStreamEngine
        from unittest.mock import patch

        engine = LiveStreamEngine()
        engine.cfg["cameras"] = [
            {
                "id": "cam-source",
                "name": "Source Cam",
                "source": "0",
                "roi": [0.1, 0.1, 0.4, 0.5],
                "bays": [
                    {"id": "b1", "name": "Bay 1", "roi": [0.1, 0.1, 0.4, 0.5], "type": "vehicle_bay"},
                    {"id": "b2", "name": "Bay 2", "roi": [0.5, 0.1, 0.4, 0.5], "type": "vehicle_bay"},
                ],
            },
            {
                "id": "cam-target",
                "name": "Target Cam",
                "source": "1",
                "roi": [0.2, 0.2, 0.3, 0.3],
                "bays": [
                    {"id": "old_bay", "name": "Old Bay", "roi": [0.2, 0.2, 0.3, 0.3], "type": "vehicle_bay"},
                ],
            },
        ]
        engine.cfg["active_camera_id"] = "cam-target"

        with patch("launcher.save_config"):
            result = engine.import_bays("cam-source", "cam-target")
        self.assertTrue(result["success"])
        self.assertEqual(len(result["bays"]), 2)
        self.assertEqual(result["bays"][0]["id"], "b1")
        self.assertEqual(result["bays"][1]["id"], "b2")

        # Source camera's bays should be untouched
        src_cam = next(c for c in engine.cfg["cameras"] if c["id"] == "cam-source")
        self.assertEqual(len(src_cam["bays"]), 2)

        # Target camera's bays now match imported bays
        tgt_cam = next(c for c in engine.cfg["cameras"] if c["id"] == "cam-target")
        self.assertEqual(len(tgt_cam["bays"]), 2)
        self.assertEqual(tgt_cam["bays"][0]["id"], "b1")

    def test_get_camera_frame_api(self):
        from launcher import LiveStreamEngine

        engine = LiveStreamEngine()
        engine.current_frame_jpeg = b"FAKE_JPEG_ACTIVE"
        engine.cfg["active_camera_id"] = "cam-active"
        engine.cfg["cameras"] = [{"id": "cam-active", "name": "Active", "source": "0"}]

        # Active camera returns current_frame_jpeg directly
        frame_bytes, ctype = engine.get_camera_frame("cam-active")
        self.assertEqual(frame_bytes, b"FAKE_JPEG_ACTIVE")
        self.assertEqual(ctype, "image/jpeg")

        # None/empty camera_id returns active camera
        frame_bytes, ctype = engine.get_camera_frame(None)
        self.assertEqual(frame_bytes, b"FAKE_JPEG_ACTIVE")

    def test_camera_stream_pool_and_concurrent_frames(self):
        from launcher import LiveStreamEngine, CameraStreamWorker

        engine = LiveStreamEngine()
        engine.current_frame_jpeg = b"AI_ACTIVE_FRAME"
        engine.cfg["active_camera_id"] = "cam-1"
        engine.cfg["cameras"] = [
            {"id": "cam-1", "name": "Camera 1", "source": "0"},
            {"id": "cam-2", "name": "Camera 2", "source": "rtsp://192.168.1.100/stream"},
        ]

        # Simulate background worker for cam-2 in the pool
        worker_2 = CameraStreamWorker("cam-2", engine.cfg["cameras"][1])
        worker_2.latest_jpeg = b"BACKGROUND_LIVE_FRAME_2"
        worker_2.latest_ts = 1234567890.0
        engine.camera_pool._workers["cam-2"] = worker_2

        # Active camera cam-1 returns AI frame
        f1, c1 = engine.get_camera_frame("cam-1")
        self.assertEqual(f1, b"AI_ACTIVE_FRAME")

        # Concurrent background camera cam-2 returns its worker frame without dropping
        f2, c2 = engine.get_camera_frame("cam-2")
        self.assertEqual(f2, b"BACKGROUND_LIVE_FRAME_2")

        # Switch active camera to cam-2
        engine.camera_pool.set_active_camera("cam-2")
        engine.cfg["active_camera_id"] = "cam-2"
        engine.current_frame_jpeg = b"AI_ACTIVE_FRAME_2"

        # Worker for cam-1 now has background frames
        worker_1 = CameraStreamWorker("cam-1", engine.cfg["cameras"][0])
        worker_1.latest_jpeg = b"BACKGROUND_LIVE_FRAME_1"
        worker_1.latest_ts = 1234567891.0
        engine.camera_pool._workers["cam-1"] = worker_1

        # Now cam-2 is active (AI spotlight), cam-1 is background live CCTV
        f2_active, _ = engine.get_camera_frame("cam-2")
        f1_bg, _ = engine.get_camera_frame("cam-1")
        self.assertEqual(f2_active, b"AI_ACTIVE_FRAME_2")
        self.assertEqual(f1_bg, b"BACKGROUND_LIVE_FRAME_1")

    def test_connect_camera_does_not_reregister_live_grid(self):
        from unittest.mock import patch
        from launcher import LiveStreamEngine, CameraStreamWorker

        engine = LiveStreamEngine()
        cam = {
            "id": "cam-rtsp",
            "name": "Bay Cam",
            "source": "rtsp://192.168.1.10/stream",
            "protocol": "rtsp",
            "enabled": True,
        }
        engine.cfg["cameras"] = [cam]
        engine.cfg["source"] = cam["source"]
        engine.cfg["active_camera_id"] = "cam-rtsp"
        engine.running = True

        worker = CameraStreamWorker("cam-rtsp", cam)
        worker.grabber.connection_state = "CONNECTED"
        worker.latest_jpeg = b"LIVE_JPEG"
        engine.camera_pool._workers["cam-rtsp"] = worker

        register_calls = []
        bind_calls = []
        switch_calls = []
        engine.register_all_cameras_in_gateway = lambda: register_calls.append("all")
        engine._bind_gateway = lambda *a, **k: bind_calls.append(a) or "cam-rtsp"
        engine.grabber.switch_source = lambda adapter: switch_calls.append(adapter)

        with patch("launcher.save_config"):
            result = engine.connect_camera({
                "camera_id": "cam-rtsp",
                "camera_name": "Bay Cam",
                "source": "rtsp://192.168.1.10/stream",
                "protocol": "rtsp",
            })

        self.assertTrue(result["success"])
        self.assertEqual(register_calls, [])
        self.assertEqual(bind_calls, [])
        self.assertEqual(switch_calls, [])
        self.assertEqual(engine.camera_pool.active_camera_id, "cam-rtsp")
        frame, _ = engine.get_camera_frame("cam-rtsp")
        self.assertEqual(frame, b"LIVE_JPEG")

    def test_toggle_camera_port_closes_and_reconnects_worker(self):
        from launcher import LiveStreamEngine, CameraStreamWorker
        engine = LiveStreamEngine()
        engine.cfg["cameras"] = [
            {"id": "cam-1", "name": "Bay 1", "source": "0", "enabled": True},
            {"id": "cam-2", "name": "Bay 2", "source": "1", "enabled": True},
        ]
        worker_1 = CameraStreamWorker("cam-1", engine.cfg["cameras"][0])
        worker_2 = CameraStreamWorker("cam-2", engine.cfg["cameras"][1])
        engine.camera_pool._workers["cam-1"] = worker_1
        engine.camera_pool._workers["cam-2"] = worker_2

        # Close port on cam-2
        res = engine.toggle_camera_port("cam-2", False)
        self.assertTrue(res["success"])
        self.assertFalse(res["enabled"])
        self.assertNotIn("cam-2", engine.camera_pool._workers)

        # Closed camera frame returns None (204)
        frame, _ = engine.get_camera_frame("cam-2")
        self.assertIsNone(frame)

        # Reopen port on cam-2
        res2 = engine.toggle_camera_port("cam-2", True)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["enabled"])
        # Pool now syncs cam-2 back
        self.assertIn("cam-2", engine.camera_pool._workers)

    def test_connect_camera_reopens_closed_port_without_fallback(self):
        from unittest.mock import patch
        from launcher import LiveStreamEngine, CameraStreamWorker

        engine = LiveStreamEngine()
        cam = {
            "id": "cam-1",
            "name": "Front Camera",
            "source": 0,
            "protocol": "webcam",
            "enabled": True,
        }
        engine.cfg["cameras"] = [cam]
        engine.cfg["source"] = 0
        engine.cfg["protocol"] = "webcam"
        engine.cfg["active_camera_id"] = "cam-1"
        engine.running = True
        engine.is_streaming = True
        engine.current_frame_jpeg = b"LIVE_BEFORE_CLOSE"

        worker = CameraStreamWorker("cam-1", cam)
        worker.latest_jpeg = b"LIVE_BEFORE_CLOSE"
        engine.camera_pool._workers["cam-1"] = worker

        fallback_ops = []
        engine._fallback_grabber.clear_source = lambda: fallback_ops.append("clear")
        engine._fallback_grabber.switch_source = lambda adapter: fallback_ops.append("switch")

        with patch("launcher.save_config"), patch.object(
            CameraStreamWorker, "start", lambda self: None
        ), patch.object(CameraStreamWorker, "stop", lambda self: None):
            closed = engine.toggle_camera_port("cam-1", False)
            self.assertTrue(closed["success"])
            self.assertFalse(closed["enabled"])
            self.assertNotIn("cam-1", engine.camera_pool._workers)
            self.assertFalse(engine.is_streaming)
            self.assertIsNone(engine.current_frame_jpeg)
            self.assertIn("clear", fallback_ops)
            frame, _ = engine.get_camera_frame("cam-1")
            self.assertIsNone(frame)

            result = engine.connect_camera({
                "camera_id": "cam-1",
                "camera_name": "Front Camera",
                "source": 0,
                "protocol": "webcam",
            })

        self.assertTrue(result["success"])
        self.assertTrue(result["cameras"][0]["enabled"])
        self.assertIn("cam-1", engine.camera_pool._workers)
        self.assertTrue(engine.is_streaming)
        self.assertGreaterEqual(fallback_ops.count("clear"), 2)
        self.assertNotIn("switch", fallback_ops)

        reopened = engine.camera_pool.get_worker("cam-1")
        reopened.latest_jpeg = b"AFTER_RECONNECT"
        frame, _ = engine.get_camera_frame("cam-1")
        self.assertEqual(frame, b"AFTER_RECONNECT")

    def test_disconnect_releases_fallback_grabber(self):
        from launcher import LiveStreamEngine

        engine = LiveStreamEngine()
        engine.is_streaming = True
        engine.current_frame_jpeg = b"LIVE"
        clears = []
        engine._fallback_grabber.clear_source = lambda: clears.append("clear")
        engine._drop_gateway_stream = lambda: None
        engine.camera_pool.stop = lambda: None

        engine.disconnect()

        self.assertEqual(clears, ["clear"])
        self.assertFalse(engine.is_streaming)
        self.assertIsNone(engine.current_frame_jpeg)
        self.assertEqual(engine.status_text, "STANDBY")

    def test_toggle_non_active_port_leaves_streaming_intact(self):
        from unittest.mock import patch
        from launcher import LiveStreamEngine, CameraStreamWorker

        engine = LiveStreamEngine()
        engine.cfg["cameras"] = [
            {"id": "cam-1", "name": "Bay 1", "source": "0", "enabled": True},
            {"id": "cam-2", "name": "Bay 2", "source": "1", "enabled": True},
        ]
        engine.cfg["active_camera_id"] = "cam-1"
        engine.is_streaming = True
        engine.current_frame_jpeg = b"ACTIVE_LIVE"
        engine.camera_pool._workers["cam-1"] = CameraStreamWorker("cam-1", engine.cfg["cameras"][0])
        engine.camera_pool._workers["cam-2"] = CameraStreamWorker("cam-2", engine.cfg["cameras"][1])
        fallback_ops = []
        engine._fallback_grabber.clear_source = lambda: fallback_ops.append("clear")

        with patch("launcher.save_config"), patch.object(
            CameraStreamWorker, "stop", lambda self: None
        ):
            res = engine.toggle_camera_port("cam-2", False)

        self.assertTrue(res["success"])
        self.assertFalse(res["enabled"])
        self.assertTrue(engine.is_streaming)
        self.assertEqual(engine.current_frame_jpeg, b"ACTIVE_LIVE")
        self.assertEqual(fallback_ops, [])
        self.assertIn("cam-1", engine.camera_pool._workers)

    def test_garage_telemetry_includes_nested_garage_object(self):
        from launcher import LiveStreamEngine
        engine = LiveStreamEngine()
        telem = engine.garage_telemetry()
        self.assertIn("garage", telem)
        self.assertIn("shop_open", telem["garage"])
        self.assertEqual(telem["garage"]["shop_open"], telem["shop_open"])
        self.assertEqual(telem["garage"]["name"], telem["garage_name"])

    def test_bay_badge_uses_active_technician_time_matching_hud(self):
        from occupancy import BaySnapshot
        snap = BaySnapshot(
            bay_id="bay_1",
            name="Lift Bay 1",
            type="vehicle_bay",
            roi=[0.1, 0.2, 0.3, 0.4],
            state="WORKING",
            mechanic_name="HourMeng",
            wrench_seconds=428.0,  # 7m 08s
            idle_seconds=0.0,
            under_vehicle_seconds=0.0,
            break_seconds=0.0,
            wrench_time_today=711.0,  # 11m 51s
            idle_time_today=0.0,
            under_vehicle_today=0.0,
            break_time_today=0.0,
            is_working=True,
            technicians_times={"HourMeng": 428.0},
        )
        badge = snap.as_dict()["badge"]
        # Badge should show HourMeng (7m 08s), not today's (11m 51s)
        self.assertIn("HourMeng (7m 08s)", badge)
        self.assertNotIn("11m 51s", badge)

    def test_standing_pose_not_flagged_as_under_vehicle(self):
        from occupancy import is_under_vehicle_pose
        # Standing human: vertical height (400px) much greater than width (40px)
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (300.0, 100.0, 0.9)  # Nose
        kpts[5] = (280.0, 150.0, 0.9)  # L_SHOULDER
        kpts[6] = (320.0, 150.0, 0.9)  # R_SHOULDER
        kpts[11] = (285.0, 300.0, 0.9)  # L_HIP
        kpts[12] = (315.0, 300.0, 0.9)  # R_HIP
        kpts[13] = (285.0, 400.0, 0.9)  # L_KNEE
        kpts[14] = (315.0, 400.0, 0.9)  # R_KNEE
        kpts[15] = (285.0, 500.0, 0.9)  # L_ANKLE
        kpts[16] = (315.0, 500.0, 0.9)  # R_ANKLE
        self.assertFalse(is_under_vehicle_pose(kpts))

    def test_creeper_pose_flagged_as_under_vehicle(self):
        from occupancy import is_under_vehicle_pose
        # Lying horizontally on a creeper under vehicle (horizontal span > vertical)
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (100.0, 300.0, 0.9)  # Nose
        kpts[5] = (180.0, 295.0, 0.9)  # L_SHOULDER
        kpts[6] = (180.0, 305.0, 0.9)  # R_SHOULDER
        kpts[11] = (350.0, 295.0, 0.9)  # L_HIP
        kpts[12] = (350.0, 305.0, 0.9)  # R_HIP
        kpts[13] = (450.0, 295.0, 0.9)  # L_KNEE
        kpts[14] = (450.0, 305.0, 0.9)  # R_KNEE
        kpts[15] = (550.0, 295.0, 0.9)  # L_ANKLE
        kpts[16] = (550.0, 305.0, 0.9)  # R_ANKLE
        self.assertTrue(is_under_vehicle_pose(kpts))

    def test_wrenching_pose_not_flagged_as_phone_usage(self):
        from occupancy import is_phone_usage_pose
        # Mechanic working with two hands (wrists separated > 0.22 torso, not holding phone)
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (300.0, 130.0, 0.9)  # Nose
        kpts[5] = (250.0, 150.0, 0.9)  # L_SHOULDER
        kpts[6] = (350.0, 150.0, 0.9)  # R_SHOULDER
        kpts[7] = (220.0, 230.0, 0.9)  # L_ELBOW
        kpts[8] = (380.0, 230.0, 0.9)  # R_ELBOW
        kpts[9] = (260.0, 240.0, 0.9)  # L_WRIST
        kpts[10] = (340.0, 240.0, 0.9)  # R_WRIST (distance = 80px, torso = 200px -> 0.40 torso)
        kpts[11] = (260.0, 350.0, 0.9)  # L_HIP
        kpts[12] = (340.0, 350.0, 0.9)  # R_HIP
        self.assertFalse(is_phone_usage_pose(kpts))

    def test_out_of_roi_triggers_ghost_alert(self):
        from occupancy import BayZoneManager, GhostCounter
        from person import Detection
        bays_cfg = [{"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.2, 0.2, 0.4, 0.6]}]
        mgr = BayZoneManager(
            bays_cfg,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
            under_car_grace_seconds=0.0,
        )
        ghost = GhostCounter(absent_seconds=5.0, cooldown_seconds=10.0)

        # Step 1: Worker inside bay at t=0
        det_in_bay = Detection(
            x1=160, y1=120, x2=240, y2=260, conf=0.9, accepted=True,
            identity="Alice", is_staff=True, keypoints=[(200.0, 150.0, 0.9)] * 17
        )
        snaps = mgr.update([det_in_bay], 640, 480, now=0.0)
        any_occupied = any(
            getattr(s, "person_present", False)
            or s.state in ("WORKING", "UNDER_VEHICLE", "IDLE", "NOT_WORKING")
            for s in snaps
        )
        self.assertTrue(any_occupied)
        st = ghost.update(any_occupied, 0.0)
        self.assertTrue(st.occupied)
        self.assertFalse(st.should_alert)

        # Step 2: Worker steps out of bay at t=1.0 (empty bay or person elsewhere)
        snaps = mgr.update([], 640, 480, now=1.0)
        self.assertEqual(snaps[0].state, "ON_BREAK")
        self.assertFalse(snaps[0].person_present)
        any_occupied = any(
            getattr(s, "person_present", False)
            or s.state in ("WORKING", "UNDER_VEHICLE", "IDLE", "NOT_WORKING")
            for s in snaps
        )
        self.assertFalse(any_occupied)
        st = ghost.update(any_occupied, 1.0)
        self.assertFalse(st.occupied)
        self.assertFalse(st.should_alert)

        # Step 3: Worker stays out of bay past absent_seconds (at t=7.0, elapsed=6.0s >= 5.0s)
        snaps = mgr.update([], 640, 480, now=7.0)
        any_occupied = any(
            getattr(s, "person_present", False)
            or s.state in ("WORKING", "UNDER_VEHICLE", "IDLE", "NOT_WORKING")
            for s in snaps
        )
        self.assertFalse(any_occupied)
        st = ghost.update(any_occupied, 7.0)
        self.assertFalse(st.occupied)
        self.assertTrue(st.should_alert)

    def test_sync_auto_vehicles_guards_against_phantom_bays(self):
        from occupancy import BayZoneManager
        from vehicle import VehicleDetection
        bays_cfg = [{"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.1, 0.1, 0.4, 0.4]}]
        mgr = BayZoneManager(bays_cfg, auto_create_bays=False)
        self.assertEqual(len(mgr._bays), 1)

        car = VehicleDetection(x1=80, y1=60, x2=200, y2=180, conf=0.85, vehicle_type="car", vehicle_id="car_1")
        # With auto_create=False, should NOT create a new bay
        departed = mgr.sync_auto_vehicles([car], 640, 480, now=0.0, auto_create=False)
        self.assertEqual(len(mgr._bays), 1)
        self.assertTrue(mgr._bays[0].vehicle_present)

    def test_three_tier_clutter_rejection_and_creeper_whitelist(self):
        from person import backpack_clutter_keypoints
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        t0 = 100.0

        # 1. Shoes on floor inside Bay 1 ROI (only ankles visible, no hips/knees)
        shoes_kpts = [(0.0, 0.0, 0.0)] * 17
        shoes_kpts[15] = (250.0, 400.0, 0.9)
        shoes_kpts[16] = (290.0, 400.0, 0.9)
        shoes_det = _det(shoes_kpts, x1=220, y1=380, x2=320, y2=420, staff=False)

        # 2. Backpack clutter inside Bay 1 ROI
        bp_det = _det(backpack_clutter_keypoints(), x1=200, y1=300, x2=300, y2=400, staff=False)

        # 3. Completely inanimate / frozen clutter (motion=0, jitter=0, liveness=0)
        inanimate_det = _det(
            idle_standing_keypoints(),
            x1=200, y1=250, x2=300, y2=450,
            staff=False, motion=0.0, jitter=0.0, liveness=0.0, clutter=True
        )

        # 4. Unconfirmed track (hits=1)
        unconfirmed_det = _det(
            working_pose_keypoints(),
            x1=200, y1=250, x2=300, y2=450,
            staff=False, hits=1
        )

        for clutter_item in (shoes_det, bp_det, inanimate_det, unconfirmed_det):
            manager.update([clutter_item], 1000, 1000, t0, kpt_conf=0.4)
            manager.update([clutter_item], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
            snaps = {s.bay_id: s for s in manager.snapshots()}
            self.assertEqual(snaps["bay_1"].state, "EMPTY")
            self.assertEqual(snaps["bay_1"].unverified_seconds, 0.0)
            self.assertEqual(snaps["bay_1"].wrench_seconds, 0.0)
            self.assertFalse(snaps["bay_1"].session_open)

        # 5. Whitelist: Real mechanic lying on creeper under vehicle
        creeper_det = _det(
            _shift_kpts(under_vehicle_pose_keypoints(), 100, 150),
            name="Hour-Meng",
            staff=True,
            hits=3,
        )
        manager.update([creeper_det], 1000, 1000, t0 + 5.0, kpt_conf=0.4)
        manager.update([creeper_det], 1000, 1000, t0 + 7.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "UNDER_VEHICLE")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertGreater(snaps["bay_1"].wrench_seconds, 1.5)
        self.assertGreater(snaps["bay_1"].under_vehicle_seconds, 1.5)

        # 6. Whitelist: Upper body / desk worker leaning over hood or bench
        upper_kpts = [(0.0, 0.0, 0.0)] * 17
        upper_kpts[0] = (250.0, 300.0, 0.9)   # Nose
        upper_kpts[1] = (240.0, 290.0, 0.85)  # L Eye
        upper_kpts[2] = (260.0, 290.0, 0.85)  # R Eye
        upper_kpts[5] = (200.0, 350.0, 0.88)  # L Shoulder
        upper_kpts[6] = (300.0, 352.0, 0.88)  # R Shoulder
        upper_det = _det(
            upper_kpts,
            x1=180, y1=280, x2=320, y2=450,
            name="George",
            staff=True,
            hits=3,
        )
        manager.update([upper_det], 1000, 1000, t0 + 10.0, kpt_conf=0.4)
        manager.update([upper_det], 1000, 1000, t0 + 12.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].mechanic_name, "George")

    def test_thirty_second_bay_occlusion_hysteresis(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=30.0,
            break_timeout_seconds=3600.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
        )
        work = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng", staff=True, track_id=1)
        t0 = 100.0

        # Mechanic working in Bay 1
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        wrench_before = snaps["bay_1"].wrench_seconds

        # Occluded under vehicle for 25 seconds (no detections in frame)
        manager.update([], 1000, 1000, t0 + 27.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertTrue(snaps["bay_1"].session_open)
        self.assertGreater(snaps["bay_1"].wrench_seconds, wrench_before)

        # Re-emergence at 28 seconds without face detection (just body)
        reemerged = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Employee", staff=False, track_id=1)
        manager.update([reemerged], 1000, 1000, t0 + 29.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")  # Kept locked identity!

        # Now mechanic stays missing past 30s grace (at t0 + 65.0) -> transitions to ON_BREAK
        manager.update([], 1000, 1000, t0 + 65.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "ON_BREAK")

    def test_polygon_exit_short_circuits_dwell_timer_immediately(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            under_car_grace_seconds=30.0,
            break_timeout_seconds=3600.0,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
            presence_grace_seconds=0.01,
        )
        work_in_bay = _det(_shift_kpts(working_pose_keypoints(), 80, 280), name="Hour-Meng", staff=True, track_id=5)
        t0 = 100.0

        # Mechanic working in Bay 1 (x ~ 260, y ~ 500)
        manager.update([work_in_bay], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work_in_bay], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertTrue(snaps["bay_1"].session_open)

        # Mechanic explicitly steps OUTSIDE Bay 1 (x ~ 800, outside Bay 1 ROI)
        work_outside = _det(
            _shift_kpts(working_pose_keypoints(), 600, 280),
            x1=700, y1=220, x2=850, y2=620,
            name="Hour-Meng",
            staff=True,
            track_id=5,
        )
        # Should short-circuit 30s dwell timer immediately!
        manager.update([work_outside], 1000, 1000, t0 + 3.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "EMPTY")
        self.assertFalse(snaps["bay_1"].session_open)
        self.assertIsNone(snaps["bay_1"].mechanic_name)

    def test_camera06_sit_down_stays_in_bay_not_under_vehicle(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
            presence_grace_seconds=8.0,
        )
        for b in manager._bays:
            b.sitting_threshold_seconds = 1.0

        t0 = 100.0
        standing = _det(
            _shift_kpts(working_pose_keypoints(), 80, 280),
            name="Hour-Meng",
            staff=True,
            track_id=1,
        )
        manager.update([standing], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([standing], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "WORKING")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")

        sitting = _det(
            _shift_kpts(overhead_sitting_keypoints(), 80, 280),
            x1=180, y1=400, x2=320, y2=620,
            name="Employee",
            staff=False,
            track_id=99,
        )
        manager.update([sitting], 1000, 1000, t0 + 2.5, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertNotEqual(snaps["bay_1"].state, "UNDER_VEHICLE")
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")
        self.assertTrue(snaps["bay_1"].session_open)

        manager.update([sitting], 1000, 1000, t0 + 4.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "NOT_WORKING")
        self.assertEqual(snaps["bay_1"].not_working_reason, "SITTING")
        self.assertTrue(snaps["bay_1"].session_open)
        self.assertEqual(snaps["bay_1"].mechanic_name, "Hour-Meng")

        edge = _det(
            _shift_kpts(overhead_sitting_keypoints(), -90, 200),
            x1=0, y1=250, x2=199, y2=780,
            name="Employee",
            staff=False,
            track_id=99,
        )
        manager.update([edge], 1000, 1000, t0 + 6.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertTrue(snaps["bay_1"].session_open)
        self.assertNotEqual(snaps["bay_1"].state, "EMPTY")

        still = _det(
            _shift_kpts(overhead_sitting_keypoints(), 80, 280),
            x1=180, y1=400, x2=320, y2=620,
            name="Employee",
            staff=False,
            track_id=99,
            liveness=0.0,
            jitter=0.0,
            motion=0.0,
        )
        manager.update([still], 1000, 1000, t0 + 7.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertTrue(snaps["bay_1"].session_open)
        self.assertEqual(snaps["bay_1"].state, "NOT_WORKING")

    def test_walk_out_after_presence_grace_closes_session(self):
        manager = BayZoneManager(
            DEFAULT_BAYS,
            occupy_confirm_seconds=0.01,
            occupy_clear_seconds=0.01,
            presence_grace_seconds=0.01,
        )
        work = _det(
            _shift_kpts(working_pose_keypoints(), 80, 280),
            name="Hour-Meng",
            staff=True,
            track_id=5,
        )
        t0 = 100.0
        manager.update([work], 1000, 1000, t0, kpt_conf=0.4)
        manager.update([work], 1000, 1000, t0 + 2.0, kpt_conf=0.4)
        outside = _det(
            _shift_kpts(working_pose_keypoints(), 600, 280),
            x1=700, y1=220, x2=850, y2=620,
            name="Hour-Meng",
            staff=True,
            track_id=5,
        )
        manager.update([outside], 1000, 1000, t0 + 3.0, kpt_conf=0.4)
        snaps = {s.bay_id: s for s in manager.snapshots()}
        self.assertEqual(snaps["bay_1"].state, "EMPTY")
        self.assertFalse(snaps["bay_1"].session_open)


if __name__ == "__main__":
    unittest.main()


