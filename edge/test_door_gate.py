"""Unit tests for the spa lobby door traffic monitor (door_gate.DoorTrafficMonitor)."""

import unittest
from datetime import datetime

from door_gate import DoorTrafficMonitor


class _FakeDet:
    def __init__(self, cx, cy=0.7, width=0.12, track_id=1, w=960, h=1080):
        self.x1 = (cx - width / 2) * w
        self.x2 = (cx + width / 2) * w
        self.y1 = cy * h
        self.y2 = (cy + 0.5) * h
        self.track_id = track_id


class DoorGateTest(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 960, 1080
        self.t = 1000.0
        self.stamp = datetime(2026, 9, 23, 10, 30, 0)

    def _mon(self, **over):
        cfg = {
            "line_x": 0.55,
            "zone": [0.50, 1.0],
            "enter_direction": "right_to_left",
            "confirm_obs": 3,
            "cooldown_seconds": 12.0,
            "leave_seconds": 5.0,
            "relink_seconds": 30.0,
            "relink_distance": 0.28,
            "motion_confirm": False,
            "max_sessions": 2,
            "proofs": False,
        }
        cfg.update(over)
        return DoorTrafficMonitor(cfg)

    def _tick(self, mon, dets, dt=1.0):
        self.t += dt
        self.stamp = datetime.fromtimestamp(self.t)
        return mon.update(dets, self.w, self.h, self.t, frame=None, stamp=self.stamp)

    # ------------------------------------------------------------ appear/exit
    def test_appear_enter_then_disappear_exit_sessions_with_dwell(self):
        mon = self._mon()
        det = _FakeDet(cx=0.62, track_id=1)
        ev = self._tick(mon, [det])
        self.assertEqual(ev, [])                      # needs 2 obs in zone
        ev = self._tick(mon, [det])
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].kind, "enter")
        self.assertEqual(ev[0].via, "appear")
        sid = ev[0].session_id
        self.assertIsNotNone(sid)

        for _ in range(5):                            # customer stays in lobby
            self._tick(mon, [det])

        ev = self._tick(mon, [], dt=8.0)              # vanished -> exit
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].kind, "exit")
        self.assertEqual(ev[0].via, "disappear")
        self.assertEqual(ev[0].session_id, sid)
        self.assertGreater(ev[0].dwell_s, 0)

        sess = mon.sessions_meta()[-1]
        self.assertEqual(sess["session_id"], sid)
        self.assertIsNotNone(sess["dwell_s"])

    def test_new_track_outside_zone_does_not_enter(self):
        mon = self._mon()
        det = _FakeDet(cx=0.30, track_id=1)           # deep in the lobby
        self._tick(mon, [det])
        self._tick(mon, [det])
        self.assertEqual(mon.recent(), [])
        self.assertEqual(sum(1 for s in mon.sessions if not s.closed), 0)

    # ------------------------------------------------------------- tripwire
    def test_tripwire_enter_and_exit_by_direction(self):
        # Door-side zone made invisible so only the tripwire decides.
        mon = self._mon(zone=[0.99, 1.0], cooldown_seconds=0.0)
        for cx in (0.70, 0.66, 0.62, 0.58):
            self._tick(mon, [_FakeDet(cx=cx, track_id=1)])
        ev = self._tick(mon, [_FakeDet(cx=0.54, track_id=1)])  # R -> L crossing
        enters = [e for e in ev if e.kind == "enter"]
        self.assertEqual(len(enters), 1)
        self.assertEqual(enters[0].via, "tripwire")

        all_events = []
        for cx in (0.56, 0.60, 0.64, 0.68):                    # walk back out
            all_events.extend(self._tick(mon, [_FakeDet(cx=cx, track_id=1)]))
        exits = [e for e in all_events if e.kind == "exit"]
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0].via, "tripwire")
        self.assertEqual(enters[0].session_id, exits[0].session_id)

    # ------------------------------------------------- cooldown + relink
    def test_cooldown_suppresses_flicker_reenter(self):
        mon = self._mon(cooldown_seconds=12.0)
        det1 = _FakeDet(cx=0.60, track_id=1)
        self._tick(mon, [det1])
        ev = self._tick(mon, [det1])
        self.assertEqual(len(ev), 1)                  # first enter

        det2 = _FakeDet(cx=0.60, track_id=2)          # tracker re-ids same person
        ev = self._tick(mon, [det2])
        self.assertTrue(all(e.kind != "enter" for e in ev))   # suppressed

    def test_relink_reuses_open_session_instead_of_new_one(self):
        mon = self._mon(cooldown_seconds=0.0)
        det1 = _FakeDet(cx=0.60, track_id=1)
        self._tick(mon, [det1])
        first = self._tick(mon, [det1])
        sid1 = first[0].session_id

        det2 = _FakeDet(cx=0.62, track_id=2)          # flicker, same spot
        ev = self._tick(mon, [det2])
        self.assertTrue(all(e.kind != "enter" for e in ev))
        open_sessions = [s for s in mon.sessions if not s.closed]
        self.assertEqual(len(open_sessions), 1)
        self.assertEqual(open_sessions[0].session_id, sid1)

    # ------------------------------------------------------- max sessions
    def test_max_sessions_forces_oldest_close(self):
        mon = self._mon(cooldown_seconds=0.0, max_sessions=2)
        # two customers far enough apart not to be relinked (spa groups)
        for tid, cx in ((1, 0.60), (2, 0.95)):
            det = _FakeDet(cx=cx, track_id=tid)
            self._tick(mon, [det])
            self._tick(mon, [det])
        open_sessions = [s for s in mon.sessions if not s.closed]
        self.assertEqual(len(open_sessions), 2)
        # a third customer arrives after the relink window -> must cap at 2,
        # force-closing the oldest (customer A) session
        self.t += 31
        dets = [_FakeDet(cx=0.60, track_id=1), _FakeDet(cx=0.95, track_id=2)]
        for _ in range(2):
            dets.append(_FakeDet(cx=0.97, track_id=3))
            self._tick(mon, dets)
        open_sessions = [s for s in mon.sessions if not s.closed]
        self.assertEqual(len(open_sessions), 2)       # capped
        self.assertTrue(all(s.session_id != 1 for s in open_sessions))  # A closed

    # ------------------------------------------------------------ no doors
    def test_disabled_tracks_purge_without_events(self):
        mon = self._mon()
        det = _FakeDet(cx=0.30, track_id=1)
        for _ in range(3):
            self._tick(mon, [det])
        self._tick(mon, [], dt=8.0)
        self.assertEqual(mon.recent(), [])

    def test_bad_track_ids_ignored(self):
        mon = self._mon()
        ev = self._tick(mon, [_FakeDet(cx=0.6, track_id=0)])
        self.assertEqual(ev, [])
        ev = self._tick(mon, [_FakeDet(cx=0.6, track_id=-1)])
        self.assertEqual(ev, [])

    def test_out_of_frame_boxes_ignored(self):
        mon = self._mon()
        off = _FakeDet(cx=-0.8, track_id=5)          # extrapolated off left edge
        off.x1 = -900; off.x2 = -100
        self._tick(mon, [off])
        self._tick(mon, [off])
        self.assertEqual(mon.recent(), [])
        self.assertNotIn(5, mon.tracks)


if __name__ == "__main__":
    unittest.main()