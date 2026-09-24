"""Customer visit monitor: anonymous re-ID uniqueness and session counting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from db import connect, customer_visit_counts
from person import Detection
from workplaces import parse_workplace_id
from workplaces.customer_visits import AnonymousVisitorGallery, CustomerVisitMonitor


def _feat(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=32).astype(np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-6)


def _det(x: float, y: float, feat: np.ndarray, track_id: int = 1) -> Detection:
    return Detection(
        x1=x,
        y1=y,
        x2=x + 40,
        y2=y + 80,
        conf=0.9,
        track_id=track_id,
        reid_feat=feat,
    )


class AnonymousGalleryTests(unittest.TestCase):
    def test_same_embedding_reuses_id(self) -> None:
        gallery = AnonymousVisitorGallery(threshold=0.5)
        feat = _feat(7)
        first = gallery.match_or_enroll(feat)
        second = gallery.match_or_enroll(feat)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("visitor_"))

    def test_different_embeddings_get_new_ids(self) -> None:
        gallery = AnonymousVisitorGallery(threshold=0.99)
        a = gallery.match_or_enroll(_feat(1))
        b = gallery.match_or_enroll(_feat(99))
        self.assertNotEqual(a, b)

    def test_empty_feat_does_not_enroll(self) -> None:
        gallery = AnonymousVisitorGallery(threshold=0.5)
        self.assertIsNone(gallery.match_or_enroll(None))
        self.assertIsNone(gallery.match_or_enroll(np.zeros(32, dtype=np.float32)))
        self.assertEqual(gallery.entries, {})

    def test_angle_change_still_matches_best_of_bucket(self) -> None:
        gallery = AnonymousVisitorGallery(threshold=0.55)
        base = _feat(7)
        sid = gallery.match_or_enroll(base)
        # Add a drifted view of the same person (coat / angle).
        drifted = base * 0.7 + _feat(8) * 0.3
        drifted = drifted / max(float(np.linalg.norm(drifted)), 1e-6)
        gallery.enroll(drifted, sid)
        again = gallery.match_or_enroll(drifted)
        self.assertEqual(again, sid)


class CustomerVisitMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "events.db", check_same_thread=False)
        self.monitor = CustomerVisitMonitor(
            [
                {"id": "entrance", "name": "Entrance", "roi": [0.0, 0.0, 0.5, 1.0], "type": "entrance"},
                {"id": "room_1", "name": "Room 1", "roi": [0.5, 0.0, 0.5, 1.0], "type": "treatment_room"},
            ],
            confirm_seconds=0.01,
            clear_seconds=0.01,
            grace_seconds=0.05,
            match_threshold=0.5,
            conn=self.conn,
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def _dwell(self, dets, now: float) -> None:
        """Two ticks so OccupancyGate confirm fills (first tick has dt=0)."""
        self.monitor.update(dets, 100, 100, now=now)
        self.monitor.update(dets, 100, 100, now=now + 0.05)

    def test_two_visits_one_unique_same_day(self) -> None:
        import time

        feat = _feat(3)
        now = time.time()
        self._dwell([_det(10, 10, feat)], now)
        self.monitor.update([], 100, 100, now=now + 1.0)
        self._dwell([_det(10, 10, feat)], now + 2.0)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_visits"], 2)
        self.assertEqual(counts["today_unique"], 1)
        self.assertEqual(counts["week_unique"], 1)
        self.assertEqual(counts["week_visits"], 2)

    def test_massage_zones_not_coerced_to_bays(self) -> None:
        kinds = {z["type"] for z in self.monitor.configs()}
        self.assertIn("entrance", kinds)
        self.assertIn("treatment_room", kinds)
        self.assertNotIn("vehicle_bay", kinds)

    def test_reception_does_not_start_visit(self) -> None:
        import time

        self.monitor.set_zones(
            [
                {"id": "reception", "name": "Reception", "roi": [0.0, 0.0, 0.5, 1.0], "type": "reception"},
                {"id": "entrance", "name": "Entrance", "roi": [0.5, 0.0, 0.5, 1.0], "type": "entrance"},
            ]
        )
        feat = _feat(11)
        now = time.time()
        snaps = self.monitor.update([_det(10, 10, feat)], 100, 100, now=now)
        self.monitor.update([_det(10, 10, feat)], 100, 100, now=now + 0.05)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_visits"], 0)
        self.assertEqual(snaps[0].zone_kind, "reception")
        self.assertEqual(snaps[0].visitor_ids, [])
        self.assertEqual(snaps[0].open_visit_count, 0)

    def test_new_person_enrolls_visitor_id(self) -> None:
        feat = _feat(21)
        det = _det(10, 10, feat, track_id=3)
        self.monitor.update([det], 100, 100, now=1.0)
        self.assertTrue(str(det.identity).startswith("visitor_"))
        self.assertFalse(det.is_staff)

    def test_empty_reid_frames_do_not_inflate_uniques(self) -> None:
        """ReID is throttled (~every 30 frames). Empty feats must not mint visitor_* ids."""
        import time

        feat = _feat(42)
        now = time.time()
        self._dwell([_det(10, 10, feat, track_id=7)], now)
        for i in range(50):
            empty = Detection(
                x1=10, y1=10, x2=50, y2=90, conf=0.9, track_id=7, reid_feat=None
            )
            self.monitor.update([empty], 100, 100, now=now + 0.1 + i * 0.04)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_visits"], 1)
        self.assertEqual(counts["today_unique"], 1)
        self.assertEqual(len(self.monitor.gallery.entries), 1)

    def test_track_binding_survives_appearance_drift(self) -> None:
        import time

        now = time.time()
        first = _det(10, 10, _feat(1), track_id=9)
        self._dwell([first], now)
        drifted = _det(10, 10, _feat(99), track_id=9)  # different emb, same track
        self.monitor.update([drifted], 100, 100, now=now + 0.2)
        self.assertEqual(drifted.identity, first.identity)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_unique"], 1)
        self.assertEqual(counts["today_visits"], 1)

    def test_workplace_parser(self) -> None:
        self.assertEqual(parse_workplace_id("massage"), "massage")
        self.assertEqual(parse_workplace_id("unknown"), "garage")

    def test_garage_workplace_does_not_seed_massage_rooms(self) -> None:
        monitor = CustomerVisitMonitor([], workplace="garage")
        kinds = {z.get("type") for z in monitor.zones}
        self.assertNotIn("treatment_room", kinds)
        self.assertNotIn("entrance", kinds)
        self.assertEqual(monitor.workplace, "garage")

    def test_track_stitching_prevents_id_churn_on_angle_change_track_loss(self) -> None:
        """When turning drops a Kalman track (track_id 10 -> track_id 11 nearby), stitch identity."""
        import time
        from workplaces.customer_visits import AnonymousVisitorGallery

        now = time.time()
        base_feat = _feat(50)
        det1 = _det(15, 15, base_feat, track_id=10)
        self._dwell([det1], now)
        orig_id = det1.identity

        # Frame dropped / Kalman track lost for 0.5s, then new track_id 11 appears at (18, 16)
        # with a slightly drifted feature (e.g. angle change)
        drifted_feat = base_feat * 0.75 + _feat(51) * 0.25
        drifted_feat = drifted_feat / max(float(np.linalg.norm(drifted_feat)), 1e-6)
        det2 = _det(18, 16, drifted_feat, track_id=11)

        # Update empty for 0.1s to simulate momentary occlusion / track death
        self.monitor.update([], 100, 100, now=now + 0.1)
        # Now det2 arrives
        self.monitor.update([det2], 100, 100, now=now + 0.3)
        self.assertEqual(det2.identity, orig_id)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_unique"], 1)

    def test_reset_all_customer_visits(self) -> None:
        import time
        from db import reset_all_customer_visits, get_detailed_visits_report

        now = time.time()
        self._dwell([_det(10, 10, _feat(12), track_id=1)], now)
        counts = customer_visit_counts(self.conn)
        self.assertGreater(counts["today_visits"], 0)

        # Reset in-memory and database
        self.monitor.reset()
        reset_all_customer_visits(self.conn)
        counts_after = customer_visit_counts(self.conn)
        self.assertEqual(counts_after["today_visits"], 0)
        self.assertEqual(counts_after["today_unique"], 0)
        self.assertEqual(len(self.monitor.gallery.entries), 0)
        self.assertEqual(len(self.monitor._open), 0)

        report = get_detailed_visits_report(self.conn)
        self.assertEqual(report["summary"]["total_unique"], 0)
        self.assertEqual(len(report["visitors"]), 0)

    def test_detailed_visits_report(self) -> None:
        import time
        from db import get_detailed_visits_report, update_subject_alias

        now = time.time()
        feat = _feat(77)
        self._dwell([_det(10, 10, feat, track_id=20)], now)
        # Close visit
        self.monitor.update([], 100, 100, now=now + 1.0)
        self.monitor.update([], 100, 100, now=now + 2.0)

        report = get_detailed_visits_report(self.conn)
        self.assertEqual(report["summary"]["today_unique"], 1)
        self.assertEqual(report["summary"]["today_visits"], 1)
        self.assertEqual(len(report["visitors"]), 1)
        sub_id = report["visitors"][0]["subject_id"]

        # Test alias update
        update_subject_alias(self.conn, sub_id, "VIP Guest")
        report2 = get_detailed_visits_report(self.conn)
        self.assertEqual(report2["visitors"][0]["alias"], "VIP Guest")

    def test_merge_customer_subjects_in_db(self) -> None:
        from db import (
            get_detailed_visits_report,
            merge_customer_subjects,
            start_customer_visit,
            update_subject_alias,
            upsert_anonymous_subject,
        )

        upsert_anonymous_subject(self.conn, "subj_a", 1000.0)
        update_subject_alias(self.conn, "subj_a", "Alice")
        start_customer_visit(self.conn, "v1", "subj_a", "entrance", 1000.0)
        start_customer_visit(self.conn, "v2", "subj_a", "entrance", 1050.0)

        upsert_anonymous_subject(self.conn, "subj_b", 2000.0)
        start_customer_visit(self.conn, "v3", "subj_b", "waiting", 2000.0)

        ok = merge_customer_subjects(self.conn, "subj_a", "subj_b")
        self.assertTrue(ok)

        # Verify visits re-linked
        rows = self.conn.execute("SELECT id, subject_id FROM customer_visits WHERE subject_id = 'subj_b'").fetchall()
        self.assertEqual(len(rows), 3)

        # Verify source deleted
        src_row = self.conn.execute("SELECT id FROM anonymous_subjects WHERE id = 'subj_a'").fetchone()
        self.assertIsNone(src_row)

        # Verify target has combined metadata (Alice alias inherited)
        tgt_row = self.conn.execute("SELECT alias, first_seen_at, last_seen_at FROM anonymous_subjects WHERE id = 'subj_b'").fetchone()
        self.assertEqual(tgt_row["alias"], "Alice")

    def test_merge_gallery_embeddings(self) -> None:
        gallery = AnonymousVisitorGallery(threshold=0.5)
        feat_a1 = _feat(10)
        feat_a2 = _feat(11)
        feat_b1 = _feat(20)

        sid_a = gallery.enroll(feat_a1, "visitor_a")
        gallery.enroll(feat_a2, sid_a)
        sid_b = gallery.enroll(feat_b1, "visitor_b")

        self.assertIn("visitor_a", gallery.entries)
        self.assertIn("visitor_b", gallery.entries)
        self.assertEqual(len(gallery.entries["visitor_a"].embeddings), 2)
        self.assertEqual(len(gallery.entries["visitor_b"].embeddings), 1)

        gallery.merge_entries("visitor_a", "visitor_b")
        self.assertNotIn("visitor_a", gallery.entries)
        self.assertIn("visitor_b", gallery.entries)
        self.assertEqual(len(gallery.entries["visitor_b"].embeddings), 3)

    def test_merge_in_monitor(self) -> None:
        feat_a = _feat(10)
        feat_b = _feat(20)
        now = 1000.0

        # Create active sessions
        self._dwell([_det(10, 10, feat_a, track_id=1)], now)
        self._dwell([_det(60, 60, feat_b, track_id=2)], now)

        sessions = self.monitor.open_sessions()
        sub_ids = {s["subject_id"] for s in sessions}
        self.assertEqual(len(sub_ids), 2)

        s_list = list(sub_ids)
        src, tgt = s_list[0], s_list[1]
        self.monitor.merge_subjects(src, tgt)

        sessions_after = self.monitor.open_sessions()
        sub_ids_after = {s["subject_id"] for s in sessions_after}
        self.assertNotIn(src, sub_ids_after)
        self.assertIn(tgt, sub_ids_after)

    def test_get_visits_calendar_summary(self) -> None:
        from db import get_visits_calendar_summary, start_customer_visit

        # Two visits on 2026-09-20, one on 2026-09-21
        self.conn.execute("INSERT INTO customer_visits (id, subject_id, zone_id, started_at) VALUES ('c1', 's1', 'z1', '2026-09-20T10:00:00')")
        self.conn.execute("INSERT INTO customer_visits (id, subject_id, zone_id, started_at) VALUES ('c2', 's2', 'z1', '2026-09-20T14:00:00')")
        self.conn.execute("INSERT INTO customer_visits (id, subject_id, zone_id, started_at) VALUES ('c3', 's1', 'z1', '2026-09-21T09:00:00')")
        self.conn.commit()

        summary = get_visits_calendar_summary(self.conn)
        dates = {r["date"]: r for r in summary}
        self.assertIn("2026-09-20", dates)
        self.assertEqual(dates["2026-09-20"]["total_visits"], 2)
        self.assertEqual(dates["2026-09-20"]["unique_visitors"], 2)
        self.assertIn("2026-09-21", dates)
        self.assertEqual(dates["2026-09-21"]["total_visits"], 1)

        # Filter by month prefix
        month_summary = get_visits_calendar_summary(self.conn, month_prefix="2026-09")
        self.assertEqual(len(month_summary), 2)
        empty_month = get_visits_calendar_summary(self.conn, month_prefix="2025-01")
        self.assertEqual(len(empty_month), 0)

    def test_massage_never_writes_visitor_avatars(self) -> None:
        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)
        subject = "visitor_privacy"
        self.monitor.store_customer_avatars = False
        self.monitor.workplace = "massage"
        det = _det(40, 40, _feat(3), track_id=9)
        self.monitor._maybe_save_avatar(subject, det, frame, 1000.0)
        self.assertNotIn(subject, self.monitor._saved_avatars)

        # Massage hard-gates even if store_customer_avatars is True.
        self.monitor.workplace = "massage"
        self.monitor.store_customer_avatars = True
        self.monitor._maybe_save_avatar(subject, det, frame, 1001.0)
        self.assertNotIn(subject, self.monitor._saved_avatars)

    def test_wait_sla_alerts_once_with_cooldown(self) -> None:
        class _Bot:
            def __init__(self) -> None:
                self.enabled = True
                self.messages: list[str] = []

            def send_message(self, text: str) -> bool:
                self.messages.append(text)
                return True

        bot = _Bot()
        monitor = CustomerVisitMonitor(
            [
                {"id": "waiting", "name": "Waiting", "roi": [0.0, 0.0, 1.0, 1.0], "type": "waiting"},
                {"id": "reception", "name": "Reception", "roi": [0.0, 0.0, 0.01, 0.01], "type": "reception"},
            ],
            confirm_seconds=0.01,
            clear_seconds=0.01,
            grace_seconds=0.05,
            match_threshold=0.5,
            conn=self.conn,
            workplace="massage",
            store_customer_avatars=False,
            branch_id="champei-pp-01",
            camera_role="front_desk",
            telegram_out=bot,
            wait_sla_seconds=180.0,
            wait_sla_cooldown_seconds=600.0,
        )
        feat = _feat(42)
        t0 = 5_000.0
        monitor.update([_det(80, 80, feat, track_id=1)], 200, 200, now=t0)
        self.assertEqual(bot.messages, [])
        monitor.update([_det(80, 80, feat, track_id=1)], 200, 200, now=t0 + 179.0)
        self.assertEqual(bot.messages, [])
        monitor.update([_det(80, 80, feat, track_id=1)], 200, 200, now=t0 + 181.0)
        self.assertEqual(len(bot.messages), 1)
        self.assertIn("champei-pp-01", bot.messages[0])
        self.assertIn("WAIT BOTTLENECK", bot.messages[0])
        # Cooldown: same subject should not spam immediately.
        monitor.update([_det(80, 80, feat, track_id=1)], 200, 200, now=t0 + 200.0)
        self.assertEqual(len(bot.messages), 1)
        row = self.conn.execute(
            "SELECT event_type, branch_id, camera_role FROM events WHERE event_type = 'wait_bottleneck'"
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["branch_id"], "champei-pp-01")
        self.assertEqual(row["camera_role"], "front_desk")


if __name__ == "__main__":
    unittest.main()
