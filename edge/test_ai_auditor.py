"""Tests for Tier 2 Vision AI Auditor, Token Gating, and Bay Integration."""

from __future__ import annotations

import base64
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_auditor import (
    AIAuditVerdict,
    AIAuditorQueue,
    FireworksVLMClient,
    TokenSaverGate,
    extract_dual_crops,
)
from db import connect, get_recent_ai_audits, record_ai_audit_verdict
from occupancy import BayZoneManager, is_phone_usage_pose, is_sitting_pose
from person import Detection


class TestAIAuditor(unittest.TestCase):
    def test_token_saver_gate_duration_and_cooldown(self):
        gate = TokenSaverGate(duration_threshold=5.0, cooldown_seconds=180.0)
        bay_id = "bay_1"

        # Frame at t=0: Pattern starts, but duration < 5s -> Should NOT trigger
        self.assertFalse(gate.evaluate(bay_id, pattern_matched=True, now=100.0))

        # Frame at t=3s: Duration is 3s < 5s -> Should NOT trigger
        self.assertFalse(gate.evaluate(bay_id, pattern_matched=True, now=103.0))

        # Frame at t=5.1s: Duration >= 5s -> Should TRIGGER
        self.assertTrue(gate.evaluate(bay_id, pattern_matched=True, now=105.1))

        # Frame at t=6.0s: In cooldown (< 180s) -> Should NOT trigger again
        self.assertFalse(gate.evaluate(bay_id, pattern_matched=True, now=106.0))
        self.assertFalse(gate.evaluate(bay_id, pattern_matched=True, now=200.0))

        # Frame at t=286s (181s after last audit): Duration >= 5s -> Should TRIGGER again
        self.assertFalse(gate.evaluate(bay_id, pattern_matched=True, now=280.0))  # Start new pattern
        self.assertTrue(gate.evaluate(bay_id, pattern_matched=True, now=286.0))  # 6s later

    def test_dual_crop_extraction(self):
        # Create a dummy 640x480 frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 400), (255, 255, 255), -1)

        # 17 COCO keypoints with wrists near center
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[9] = (200.0, 250.0, 0.9)   # L_WRIST
        kpts[10] = (220.0, 250.0, 0.9)  # R_WRIST

        b64_h, b64_c, hand_img, context_img = extract_dual_crops(
            frame, keypoints=kpts, bbox=(100, 100, 300, 400)
        )

        self.assertTrue(len(b64_h) > 50)
        self.assertTrue(len(b64_c) > 50)
        self.assertGreater(hand_img.shape[0], 10)
        self.assertGreater(context_img.shape[0], 10)

    def test_db_audit_logging(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = Path(tf.name)
        conn = connect(db_path)

        record_ai_audit_verdict(
            conn,
            bay_id="Bay-1",
            technician_name="Alice",
            action="DIAGNOSTIC_TOOL",
            category="ACTIVE_WORK",
            confidence=0.95,
            explanation="Technician is using an OBD-II scanner.",
        )

        records = get_recent_ai_audits(conn, bay_id="Bay-1")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["action"], "DIAGNOSTIC_TOOL")
        self.assertEqual(records[0]["category"], "ACTIVE_WORK")
        self.assertAlmostEqual(records[0]["confidence"], 0.95)
        conn.close()

    def test_bay_zone_manager_with_ai_auditor_override(self):
        # Mock VLM client returning diagnostic tool (work activity)
        class MockDiagnosticVLM(FireworksVLMClient):
            def audit_activity(self, *args, **kwargs):
                return AIAuditVerdict(
                    bay_id="bay_1",
                    technician_id="Bob",
                    action="DIAGNOSTIC_TOOL",
                    category="ACTIVE_WORK",
                    confidence=0.98,
                    explanation="Using OBD tool.",
                    is_work_activity=True,
                )

        gate = TokenSaverGate(duration_threshold=0.1, cooldown_seconds=10.0)
        auditor = AIAuditorQueue(vlm_client=MockDiagnosticVLM(), gate=gate)

        bays_cfg = [
            {"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.0, 0.0, 1.0, 1.0]}
        ]
        mgr = BayZoneManager(bays_cfg, ai_auditor=auditor)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a detection with pose holding scanner / phone near face
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (300.0, 150.0, 0.9)  # NOSE
        kpts[5] = (280.0, 200.0, 0.9)  # L_SHOULDER
        kpts[6] = (320.0, 200.0, 0.9)  # R_SHOULDER
        kpts[9] = (295.0, 155.0, 0.9)  # L_WRIST near face
        kpts[11] = (285.0, 300.0, 0.9)  # L_HIP
        kpts[12] = (315.0, 300.0, 0.9)  # R_HIP

        det = Detection(
            x1=150, y1=100, x2=350, y2=400, conf=0.9, accepted=True,
            identity="Bob", is_staff=True, keypoints=kpts
        )

        # Update manager at t=0 (pattern starts)
        mgr.update([det], 640, 480, now=0.0, frame=frame)
        # Update manager at t=2.5s (phone_elapsed >= 2.0s triggers audit)
        mgr.update([det], 640, 480, now=2.5, frame=frame)
        time.sleep(0.1)  # Allow async worker to process

        mgr.update([det], 640, 480, now=3.0, frame=frame)
        snaps = mgr.snapshots()
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].state, "WORKING")
        self.assertIsNotNone(snaps[0].ai_verdict)
        self.assertTrue(snaps[0].ai_verdict["is_work_activity"])
        self.assertIn("[AI Verified]", snaps[0].as_dict()["badge"])

        auditor.shutdown()

    def test_bay_timeline_memory(self):
        bays_cfg = [
            {"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.0, 0.0, 1.0, 1.0]}
        ]
        mgr = BayZoneManager(bays_cfg)
        bay = mgr._bays[0]

        # Log some events
        bay.log_event(100.0, "Technician Bob entered bay")
        bay.log_event(105.0, "State transitioned to WORKING")
        bay.log_event(110.0, "State transitioned to NOT_WORKING (PHONE)")

        self.assertEqual(len(bay.timeline), 3)
        self.assertIn("Technician Bob entered bay", bay.timeline[0][1])
        self.assertIn("NOT_WORKING (PHONE)", bay.timeline[2][1])

    def test_offline_fallback_simulation(self):
        client = FireworksVLMClient(api_key="")
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        b64_h, b64_c, _, _ = extract_dual_crops(frame)
        verdict = client.audit_activity(b64_h, b64_c, "bay_1", "Bob", "recent timeline")
        self.assertTrue(verdict.is_work_activity)
        self.assertEqual(verdict.action, "DIAGNOSTIC_TOOL")
        self.assertEqual(verdict.category, "ACTIVE_WORK")
        self.assertGreaterEqual(verdict.confidence, 0.9)

    def test_phone_distraction_verdict_does_not_override(self):
        class MockDistractionVLM(FireworksVLMClient):
            def audit_activity(self, *args, **kwargs):
                return AIAuditVerdict(
                    bay_id="bay_1",
                    technician_id="Bob",
                    action="PHONE_USAGE",
                    category="DISTRACTION",
                    confidence=0.99,
                    explanation="Personal phone scrolling detected.",
                    is_work_activity=False,
                )

        gate = TokenSaverGate(duration_threshold=0.0, cooldown_seconds=10.0)
        auditor = AIAuditorQueue(vlm_client=MockDistractionVLM(), gate=gate)

        bays_cfg = [
            {"id": "bay_1", "name": "Bay 1", "type": "vehicle_bay", "roi": [0.0, 0.0, 1.0, 1.0]}
        ]
        mgr = BayZoneManager(bays_cfg, ai_auditor=auditor)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Person in bay holding phone to ear
        kpts = [(0.0, 0.0, 0.0)] * 17
        kpts[0] = (300.0, 150.0, 0.9)  # NOSE
        kpts[5] = (280.0, 200.0, 0.9)  # L_SHOULDER
        kpts[6] = (320.0, 200.0, 0.9)  # R_SHOULDER
        kpts[9] = (295.0, 155.0, 0.9)  # L_WRIST near ear
        kpts[11] = (285.0, 300.0, 0.9)  # L_HIP
        kpts[12] = (315.0, 300.0, 0.9)  # R_HIP

        det = Detection(
            x1=200, y1=100, x2=400, y2=450, conf=0.9, accepted=True,
            identity="Bob", is_staff=True, keypoints=kpts
        )

        # Update at t=0
        mgr.update([det], 640, 480, now=0.0, frame=frame)
        # Update at t=3s (trigger pattern matched >= 2s)
        mgr.update([det], 640, 480, now=3.0, frame=frame)
        time.sleep(0.1)

        # Advance past 12s phone threshold
        mgr.update([det], 640, 480, now=13.0, frame=frame)
        snaps = mgr.snapshots()
        self.assertEqual(len(snaps), 1)
        # Should be NOT_WORKING because AI confirmed distraction
        self.assertEqual(snaps[0].state, "NOT_WORKING")
        self.assertEqual(snaps[0].not_working_reason, "PHONE")

        auditor.shutdown()

    def test_default_model_and_config(self):
        client = FireworksVLMClient(api_key="test-key")
        self.assertEqual(client.model, "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        self.assertEqual(client.timeout, 25.0)

    def test_http_error_detail_extraction(self):
        from unittest.mock import MagicMock, patch
        import requests

        client = FireworksVLMClient(api_key="fake-key")
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {"message": "Model not found, inaccessible, and/or not deployed"}
        }
        http_err = requests.exceptions.HTTPError(
            "404 Client Error: Not Found for url: https://api.fireworks.ai/inference/v1/chat/completions",
            response=mock_response,
        )

        with patch("requests.post", side_effect=http_err):
            verdict = client.audit_activity("h", "c", "bay_1", "Bob")
            self.assertEqual(verdict.action, "UNKNOWN")
            self.assertEqual(verdict.confidence, 0.0)
            self.assertIn("Model not found, inaccessible, and/or not deployed", verdict.explanation)


if __name__ == "__main__":
    unittest.main()
