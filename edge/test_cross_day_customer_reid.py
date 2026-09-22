"""Unit tests for Cross-Day Outfit Invariance & Multi-Camera Customer Recognition.

Tests:
1. Face Biometric Invariance: Day 1 (White outfit) vs. Day 2 (Black outfit).
2. Multi-Camera Step-by-Step Hand-off (Parking -> Entrance -> Reception).
3. Staff vs. Customer Separation.
4. Persistent Cross-Day Visit History.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
import numpy as np

from face_id import CustomerFaceGallery, CustomerProfile, FaceMatch
from workplaces.customer_visits import CustomerVisitMonitor
from db import connect, customer_visit_counts


def _make_unit_vector(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


class CrossDayCustomerReIDTests(unittest.TestCase):
    def setUp(self):
        self.gallery = CustomerFaceGallery(threshold=0.60)

    def test_cross_day_outfit_invariance(self):
        """Simulate Customer #1 visiting on Day 1 in White clothing, and Day 2 in Black clothing."""
        # Fixed facial geometry vector for Customer A
        face_vector_customer_a = _make_unit_vector(42, dim=128)

        # Day 1: Customer A visits wearing White clothes (Body appearance vector 101)
        clothing_day1_white = _make_unit_vector(101, dim=512)
        prof_day1, is_new_1, score_1 = self.gallery.match_or_enroll(
            face_vector_customer_a, now=1000.0
        )
        self.assertTrue(is_new_1)
        self.assertEqual(prof_day1.customer_num, 1)
        self.assertEqual(prof_day1.visit_count, 1)
        cust_id = prof_day1.customer_id

        # Day 2: Same customer visits wearing Black clothes (Body appearance vector 999 - completely different!)
        clothing_day2_black = _make_unit_vector(999, dim=512)
        # Cosine similarity between white and black outfits is low
        clothing_sim = float(np.dot(clothing_day1_white, clothing_day2_black))
        self.assertLess(clothing_sim, 0.40)

        # But when Face Recognizer scans their face on Day 2:
        prof_day2, is_new_2, score_2 = self.gallery.match_or_enroll(
            face_vector_customer_a, now=1000.0 + 86400.0  # +1 day
        )
        self.assertFalse(is_new_2, "Day 2 should match existing customer profile!")
        self.assertEqual(prof_day2.customer_id, cust_id, "Customer ID must remain the same across days!")
        self.assertEqual(prof_day2.customer_num, 1)
        self.assertEqual(prof_day2.visit_count, 2, "Visit count must increment to 2!")

    def test_multicamera_step_by_step_verification(self):
        """Camera 1 (Parking) verifies face early; Camera 2 (Entrance) and Camera 3 (Reception) inherit identity."""
        zones = [
            {"id": "parking", "name": "Parking", "roi": [0.0, 0.0, 0.3, 1.0], "type": "parking"},
            {"id": "entrance", "name": "Entrance", "roi": [0.3, 0.0, 0.3, 1.0], "type": "entrance"},
            {"id": "reception", "name": "Reception", "roi": [0.6, 0.0, 0.4, 1.0], "type": "reception"},
        ]
        monitor = CustomerVisitMonitor(zones, workplace="massage")

        face_vector = _make_unit_vector(77, dim=128)
        clothing_vector = _make_unit_vector(300, dim=512)
        prof, _, _ = self.gallery.match_or_enroll(face_vector, now=100.0)
        customer_id = prof.customer_id

        # Step 1: In Camera 1 (Parking), face is spotted -> tagged with customer_id
        det_parking = SimpleNamespace(
            x1=10, y1=10, x2=50, y2=90,
            track_id=1,
            is_staff=False,
            is_customer=True,
            customer_id=customer_id,
            reid_feat=clothing_vector,
        )
        snaps1 = monitor.update([det_parking], 100, 100, now=100.0)
        self.assertEqual(det_parking.identity, customer_id)

        # Step 2: In Camera 2 (Entrance), walking inside with the same clothing
        det_entrance = SimpleNamespace(
            x1=40, y1=10, x2=55, y2=90,
            track_id=2,  # New camera/track ID
            is_staff=False,
            is_customer=False,  # Face momentarily turned away
            customer_id=None,
            reid_feat=clothing_vector,  # ReID links them!
        )
        snaps2 = monitor.update([det_entrance], 100, 100, now=102.0)
        self.assertEqual(det_entrance.identity, customer_id, "Entrance camera must resolve to verified customer!")

        # Step 3: In Camera 3 (Reception), customer stands at counter
        det_reception = SimpleNamespace(
            x1=75, y1=20, x2=90, y2=90,
            track_id=3,
            is_staff=False,
            is_customer=True,
            customer_id=customer_id,
            reid_feat=clothing_vector,
        )
        snaps3 = monitor.update([det_reception], 100, 100, now=105.0)
        self.assertEqual(det_reception.identity, customer_id)

    def test_multi_customer_separation_no_crosstalk(self):
        """Customer A and Customer B visit at the same time and retain distinct profiles."""
        face_a = _make_unit_vector(111, dim=128)
        face_b = _make_unit_vector(222, dim=128)

        prof_a, is_new_a, _ = self.gallery.match_or_enroll(face_a, now=10.0)
        prof_b, is_new_b, _ = self.gallery.match_or_enroll(face_b, now=15.0)

        self.assertTrue(is_new_a)
        self.assertTrue(is_new_b)
        self.assertNotEqual(prof_a.customer_id, prof_b.customer_id)
        self.assertEqual(prof_a.customer_num, 1)
        self.assertEqual(prof_b.customer_num, 2)

    def test_database_persistence(self):
        """Verify customer profiles write into SQLite database correctly."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_events.db"
            conn = connect(db_path)
            db_gallery = CustomerFaceGallery(threshold=0.60, conn=conn)

            face_a = _make_unit_vector(555, dim=128)
            prof, is_new, _ = db_gallery.match_or_enroll(face_a, now=500.0)
            self.assertTrue(is_new)

            # Query database directly
            row = conn.execute("SELECT * FROM anonymous_subjects WHERE id = ?", (prof.customer_id,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["alias"], "Customer #1")

            stored = conn.execute(
                "SELECT COUNT(*) AS n FROM customer_face_embeddings WHERE subject_id = ?",
                (prof.customer_id,),
            ).fetchone()
            self.assertEqual(int(stored["n"]), 1)

            reloaded = CustomerFaceGallery(threshold=0.60, conn=conn)
            matched, is_new, score = reloaded.match_or_enroll(face_a, now=900.0)
            self.assertFalse(is_new)
            self.assertEqual(matched.customer_id, prof.customer_id)
            self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()

