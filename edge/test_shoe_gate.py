"""Unit tests for the spa shoe-change / flip-flop customer monitor (shoe_gate.ShoeChangeMonitor)."""

import unittest
from datetime import datetime

from shoe_gate import ShoeChangeMonitor, reach_down_distance

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

_NAME = {
    "NOSE": NOSE, "L_EYE": L_EYE, "R_EYE": R_EYE, "L_EAR": L_EAR, "R_EAR": R_EAR,
    "L_SHOULDER": L_SHOULDER, "R_SHOULDER": R_SHOULDER,
    "L_ELBOW": L_ELBOW, "R_ELBOW": R_ELBOW,
    "L_WRIST": L_WRIST, "R_WRIST": R_WRIST,
    "L_HIP": L_HIP, "R_HIP": R_HIP,
    "L_KNEE": L_KNEE, "R_KNEE": R_KNEE,
    "L_ANKLE": L_ANKLE, "R_ANKLE": R_ANKLE,
}


def _kpts(**kwargs):
    pts = [(0.0, 0.0, 0.0)] * 17
    default = {
        NOSE: (480.0, 700.0, 0.9),
        L_SHOULDER: (450.0, 745.0, 0.9),
        R_SHOULDER: (520.0, 748.0, 0.9),
        L_ELBOW: (470.0, 805.0, 0.85),
        R_ELBOW: (502.0, 812.0, 0.85),
        L_HIP: (455.0, 830.0, 0.9),
        R_HIP: (505.0, 828.0, 0.9),
        L_KNEE: (470.0, 920.0, 0.85),
        R_KNEE: (500.0, 918.0, 0.85),
        L_ANKLE: (480.0, 985.0, 0.9),
        R_ANKLE: (505.0, 992.0, 0.9),
    }
    for name, val in kwargs.items():
        default[_NAME[name]] = val
    for idx, val in default.items():
        pts[idx] = val
    return pts


def _reach_down_kpts():
    """Visitor bent to their feet: wrists 45px from ankles, hands below hips."""
    return _kpts(L_WRIST=(480.0, 940.0, 0.9), R_WRIST=(505.0, 950.0, 0.9))


def _standing_kpts():
    """Standing upright: wrists at hip level, far from ankles (~120px)."""
    return _kpts(L_WRIST=(470.0, 860.0, 0.9), R_WRIST=(510.0, 862.0, 0.9))


def _headless_kpts():
    """A shoe rack / flip-flop bucket that pose models hallucinate a foot onto."""
    pts = [(0.0, 0.0, 0.0)] * 17
    pts[L_ANKLE] = (480.0, 985.0, 0.5)
    pts[R_ANKLE] = (505.0, 992.0, 0.5)
    pts[L_WRIST] = (480.0, 940.0, 0.4)
    pts[R_WRIST] = (505.0, 950.0, 0.4)
    return pts


class _FakeDet:
    def __init__(self, cx, cy=0.7, width=0.12, track_id=1, w=960, h=1080, keypoints=None):
        self.x1 = (cx - width / 2) * w
        self.x2 = (cx + width / 2) * w
        self.y1 = cy * h
        self.y2 = min(h, (cy + 0.5) * h)
        self.track_id = track_id
        self.keypoints = keypoints if keypoints is not None else _standing_kpts()
        self.is_customer = False


class ShoeGateTest(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 960, 1080
        self.t = 1000.0
        self.stamp = datetime(2026, 9, 23, 10, 30, 0)

    def _mon(self, **over):
        cfg = {
            "zone": [0.40, 0.40, 0.70, 1.0],
            "wrist_ankle_dist": 0.06,
            "confirm_seconds": 1.5,
            "cooldown_seconds": 30.0,
            "leave_seconds": 4.0,
            "kpt_conf": 0.30,
            "require_person": True,
            "proofs": False,
        }
        cfg.update(over)
        return ShoeChangeMonitor(cfg)

    def _tick(self, mon, dets, dt=1.0):
        self.t += dt
        self.stamp = datetime.fromtimestamp(self.t)
        return mon.update(dets, self.w, self.h, self.t, frame=None, stamp=self.stamp)

    # ------------------------------------------------------------- firing
    def test_reach_down_inside_zone_fires_after_confirm(self):
        mon = self._mon(confirm_seconds=1.5)
        det = _FakeDet(cx=0.55, keypoints=_reach_down_kpts())
        self.assertEqual(self._tick(mon, [det]), [])
        self.assertEqual(self._tick(mon, [det]), [])          # accum = 1.0s
        ev = self._tick(mon, [det])                           # accum = 2.0s
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].kind, "customer")
        self.assertEqual(ev[0].via, "shoe_change")
        self.assertEqual(ev[0].track_id, 1)
        self.assertTrue(det.is_customer)
        self.assertEqual(mon.flagged_count(), 1)

    def test_standing_inside_zone_never_fires(self):
        mon = self._mon()
        det = _FakeDet(cx=0.55, keypoints=_standing_kpts())
        for _ in range(10):
            self._tick(mon, [det])
        self.assertEqual(mon.recent(), [])
        self.assertEqual(mon.flagged_count(), 0)
        self.assertFalse(det.is_customer)

    def test_reach_down_outside_zone_never_fires(self):
        mon = self._mon()
        det = _FakeDet(cx=0.20, keypoints=_reach_down_kpts())  # outside zone
        for _ in range(10):
            self._tick(mon, [det])
        self.assertEqual(mon.recent(), [])
        self.assertEqual(mon.flagged_count(), 0)

    def test_clutter_without_person_rejected(self):
        mon = self._mon()
        det = _FakeDet(cx=0.55, keypoints=_headless_kpts())    # no head / shoulders
        for _ in range(10):
            self._tick(mon, [det])
        self.assertEqual(mon.recent(), [])
        self.assertEqual(mon.flagged_count(), 0)

    # ------------------------------------------------------------- cooldown
    def test_cooldown_suppresses_duplicate_fire(self):
        mon = self._mon(confirm_seconds=1.5, cooldown_seconds=30.0)
        det = _FakeDet(cx=0.55, keypoints=_reach_down_kpts(), track_id=1)
        for _ in range(3):
            self._tick(mon, [det])
        self.assertEqual(len(mon.recent()), 1)
        # keep crouching at the rack -> must NOT fire again
        for _ in range(6):
            self._tick(mon, [det])
        self.assertEqual(len(mon.recent()), 1)

    def test_fresh_track_at_other_spot_fires_independently(self):
        mon = self._mon(confirm_seconds=1.5, relink_distance=0.05, relink_seconds=8.0)
        for tid, cx in ((1, 0.55), (2, 0.64)):      # two visitors, different spots
            det = _FakeDet(cx=cx, keypoints=_reach_down_kpts(), track_id=tid)
            for _ in range(3):
                self._tick(mon, [det])
        self.assertEqual(len(mon.recent()), 2)

    def test_tracker_flicker_same_spot_suppressed(self):
        """Tracker re-labels the same crouching visitor with a new ID -> no double count."""
        mon = self._mon(confirm_seconds=1.5, relink_distance=0.10, relink_seconds=8.0)
        det1 = _FakeDet(cx=0.55, keypoints=_reach_down_kpts(), track_id=1)
        for _ in range(3):
            self._tick(mon, [det1])
        self.assertEqual(len(mon.recent()), 1)
        det2 = _FakeDet(cx=0.55, keypoints=_reach_down_kpts(), track_id=2)  # re-ID flicker
        for _ in range(6):
            self._tick(mon, [det2])
        self.assertEqual(len(mon.recent()), 1)

    # ---------------------------------------------------------------- config
    def test_custom_zone_controls_trigger(self):
        mon = self._mon(zone=[0.10, 0.10, 0.35, 0.90])
        det = _FakeDet(cx=0.20, keypoints=_reach_down_kpts())
        for _ in range(3):
            self._tick(mon, [det])
        self.assertEqual(len(mon.recent()), 1)

    # ---------------------------------------------------------------- hygiene
    def test_bad_track_ids_ignored(self):
        mon = self._mon()
        for tid in (0, -1):
            det = _FakeDet(cx=0.55, track_id=tid, keypoints=_reach_down_kpts())
            for _ in range(3):
                self._tick(mon, [det])
        self.assertEqual(mon.recent(), [])
        self.assertEqual(mon.flagged_count(), 0)

    def test_tracks_purge_after_leave_seconds(self):
        mon = self._mon(leave_seconds=4.0)
        det = _FakeDet(cx=0.55, keypoints=_reach_down_kpts())
        self._tick(mon, [det])
        self.assertIn(1, mon.tracks)
        self._tick(mon, [], dt=8.0)
        self.assertNotIn(1, mon.tracks)


class ReachDownDistanceTest(unittest.TestCase):
    def test_bent_pose_scored_below_default_threshold(self):
        det = _FakeDet(cx=0.55, keypoints=_reach_down_kpts(), w=960, h=1080)
        d = reach_down_distance(det, 1080, 960, 0.30)
        self.assertIsNotNone(d)
        self.assertLessEqual(d, 0.06)

    def test_standing_pose_scored_above_default_threshold(self):
        det = _FakeDet(cx=0.55, keypoints=_standing_kpts(), w=960, h=1080)
        d = reach_down_distance(det, 1080, 960, 0.30)
        self.assertIsNotNone(d)
        self.assertGreater(d, 0.06)

    def test_missing_wrists_or_ankles_returns_none(self):
        pts = [(0.0, 0.0, 0.0)] * 17
        pts[NOSE] = (480.0, 700.0, 0.9)
        det = _FakeDet(cx=0.55, keypoints=pts, w=960, h=1080)
        self.assertIsNone(reach_down_distance(det, 1080, 960, 0.30))


if __name__ == "__main__":
    unittest.main()