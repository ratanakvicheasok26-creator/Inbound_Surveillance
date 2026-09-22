"""Staff memory: counter dwell + VLM verify + appearance rematch."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_auditor import TokenSaverGate
from db import connect, customer_visit_counts, list_staff_memory
from person import Detection
from workplaces.customer_visits import CustomerVisitMonitor
from workplaces.staff_memory import StaffMemory, StaffVerdict, StaffVLMClient


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


def _zones() -> list[dict]:
    return [
        {"id": "reception", "name": "Reception", "roi": [0.0, 0.0, 0.5, 1.0], "type": "reception"},
        {"id": "waiting", "name": "Waiting", "roi": [0.5, 0.0, 0.5, 1.0], "type": "waiting"},
    ]


class MockStaffVLM(StaffVLMClient):
    def __init__(self, is_staff: bool = True) -> None:
        super().__init__(api_key="")
        self.is_staff_verdict = is_staff
        self.calls = 0

    def classify_counter(self, *args, **kwargs) -> StaffVerdict:
        self.calls += 1
        role = "staff" if self.is_staff_verdict else "customer"
        return StaffVerdict(
            subject_id=str(kwargs.get("subject_id") or args[2] if len(args) > 2 else ""),
            role=role,
            confidence=0.97,
            explanation="mock",
            is_staff=self.is_staff_verdict,
        )


class StaffMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "events.db", check_same_thread=False)
        self.frame = np.zeros((80, 80, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def _monitor(self, vlm: StaffVLMClient, dwell: float = 0.0) -> CustomerVisitMonitor:
        memory = StaffMemory(
            _zones(),
            conn=self.conn,
            vlm_client=vlm,
            gate=TokenSaverGate(duration_threshold=dwell, cooldown_seconds=30.0, grace_seconds=0.5),
            inline=True,
        )
        return CustomerVisitMonitor(
            _zones(),
            confirm_seconds=0.01,
            clear_seconds=0.01,
            grace_seconds=0.05,
            match_threshold=0.5,
            conn=self.conn,
            staff_memory=memory,
        )

    def test_new_person_enrolls_visitor(self) -> None:
        monitor = self._monitor(MockStaffVLM(is_staff=False), dwell=0.0)
        det = _det(10, 10, _feat(4))
        monitor.update([det], 100, 100, now=1.0, frame=self.frame)
        self.assertTrue(str(det.identity).startswith("visitor_"))
        self.assertFalse(det.is_staff)

    def test_staff_verdict_excludes_visits(self) -> None:
        vlm = MockStaffVLM(is_staff=True)
        monitor = self._monitor(vlm, dwell=0.0)
        feat = _feat(8)
        now = time.time()
        det = _det(10, 10, feat)
        monitor.update([det], 100, 100, now=now, frame=self.frame)
        self.assertTrue(det.is_staff)
        self.assertTrue(str(det.identity).startswith("staff_"))
        later = _det(60, 10, feat, track_id=2)
        monitor.update([later], 100, 100, now=now + 1.0, frame=self.frame)
        self.assertTrue(later.is_staff)
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_visits"], 0)
        self.assertGreaterEqual(vlm.calls, 1)

    def test_brief_walkthrough_does_not_enroll(self) -> None:
        vlm = MockStaffVLM(is_staff=True)
        monitor = self._monitor(vlm, dwell=3.0)
        feat = _feat(12)
        now = time.time()
        det = _det(10, 10, feat)
        monitor.update([det], 100, 100, now=now, frame=self.frame)
        monitor.update([det], 100, 100, now=now + 0.4, frame=self.frame)
        self.assertFalse(det.is_staff)
        self.assertEqual(vlm.calls, 0)
        waiting = _det(60, 10, feat)
        # Two ticks fill visit confirm hysteresis (first OccupancyGate sample has dt=0).
        monitor.update([waiting], 100, 100, now=now + 1.0, frame=self.frame)
        monitor.update([waiting], 100, 100, now=now + 1.05, frame=self.frame)
        self.assertTrue(str(waiting.identity).startswith("visitor_"))
        counts = customer_visit_counts(self.conn)
        self.assertEqual(counts["today_visits"], 1)

    def test_remembered_embedding_rematches_later(self) -> None:
        feat = _feat(33)
        first = StaffMemory(
            _zones(),
            conn=self.conn,
            vlm_client=MockStaffVLM(is_staff=True),
            gate=TokenSaverGate(duration_threshold=0.0, cooldown_seconds=1.0),
            inline=True,
        )
        staff_id = first.remember(feat, now=1.0)
        self.assertIsNotNone(staff_id)
        self.assertTrue(str(staff_id).startswith("staff_"))
        rows = list_staff_memory(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], staff_id)

        later = StaffMemory(
            _zones(),
            conn=self.conn,
            vlm_client=MockStaffVLM(is_staff=False),
            inline=True,
        )
        self.assertEqual(later.match(feat), staff_id)
        monitor = CustomerVisitMonitor(
            _zones(),
            confirm_seconds=0.01,
            clear_seconds=0.01,
            grace_seconds=0.05,
            match_threshold=0.5,
            conn=self.conn,
            staff_memory=later,
        )
        det = _det(60, 10, feat)
        monitor.update([det], 100, 100, now=time.time(), frame=self.frame)
        self.assertEqual(det.identity, staff_id)
        self.assertTrue(det.is_staff)
        self.assertEqual(customer_visit_counts(self.conn)["today_visits"], 0)

    def test_simulated_audit_without_key_is_customer(self) -> None:
        client = StaffVLMClient(api_key="")
        verdict = client.classify_counter("p", "c", "visitor_ab")
        self.assertFalse(verdict.is_staff)
        self.assertEqual(verdict.role, "customer")

    def test_garage_workplace_keeps_vehicle_bays(self) -> None:
        memory = StaffMemory(
            [{"id": "bay-1", "name": "Bay 1", "roi": [0.1, 0.2, 0.3, 0.4], "type": "vehicle_bay"}],
            workplace="garage",
            inline=True,
        )
        self.assertEqual(memory.zones[0]["type"], "vehicle_bay")
        self.assertEqual(memory.workplace, "garage")


if __name__ == "__main__":
    unittest.main()
