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

    def test_two_visits_one_unique_same_day(self) -> None:
        import time

        feat = _feat(3)
        now = time.time()
        self.monitor.update([_det(10, 10, feat)], 100, 100, now=now)
        self.monitor.update([], 100, 100, now=now + 1.0)
        self.monitor.update([_det(10, 10, feat)], 100, 100, now=now + 2.0)
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

    def test_workplace_parser(self) -> None:
        self.assertEqual(parse_workplace_id("massage"), "massage")
        self.assertEqual(parse_workplace_id("unknown"), "garage")


if __name__ == "__main__":
    unittest.main()
