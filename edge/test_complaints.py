"""Comprehensive Unit & Integration Test Suite for Audio Customer Complaint Feature."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from audio_source import LaptopMicrophoneSource
from complaint_auditor import (
    ComplaintAnalysis,
    OllamaComplaintAuditor,
    _has_khmer_script,
    _khmer_coverage,
)
from complaint_service import ComplaintMonitoringService, ComplaintRecord
from db import (
    connect,
    get_recent_complaints,
    insert_customer_complaint,
    update_complaint_telegram_status,
)
from speech_pipeline import KhmerSTTService, KhmerTranslationService
from telegram_out import TelegramOut
from vad import SileroVAD, SpeechSegment


class TestComplaintFeature(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        self.conn = connect(self.db_path, check_same_thread=False)

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_database_crud(self):
        """Test SQLite customer_complaints table insertion, retrieval, and telegram update."""
        ok = insert_customer_complaint(
            conn=self.conn,
            complaint_id="CMP-000001",
            audio_path="/path/to/complaint.wav",
            khmer_transcript="ខ្ញុំមិនពេញចិត្តនឹងការជួសជុលទេ",
            english_transcript="I am not satisfied with the repair.",
            category="repair_quality",
            severity="high",
            summary="Customer dissatisfied with repair quality.",
            customer_id="cust_123",
            camera_id="cam-1",
            audio_source="laptop_microphone",
            screenshot_path="/path/to/still.jpg",
            is_complaint=True,
            telegram_sent=False,
        )
        self.assertTrue(ok)

        complaints = get_recent_complaints(self.conn, limit=10)
        self.assertEqual(len(complaints), 1)
        c = complaints[0]
        self.assertEqual(c["complaint_id"], "CMP-000001")
        self.assertEqual(c["khmer_transcript"], "ខ្ញុំមិនពេញចិត្តនឹងការជួសជុលទេ")
        self.assertEqual(c["category"], "repair_quality")
        self.assertEqual(c["severity"], "high")
        self.assertEqual(c["telegram_sent"], 0)

        # Update telegram status
        up_ok = update_complaint_telegram_status(self.conn, "CMP-000001", sent=True)
        self.assertTrue(up_ok)

        updated = get_recent_complaints(self.conn, limit=10)[0]
        self.assertEqual(updated["telegram_sent"], 1)

    def test_silero_vad_silence_filtering(self):
        """Verify that silence chunks do not trigger a speech segment."""
        vad = SileroVAD(sample_rate=16000, min_speech_duration_seconds=0.3)
        # Feed 10 silent chunks (each 512 samples of 0.0)
        silent_chunk = np.zeros(512, dtype=np.float32)
        t = time.time()
        segment = None
        for _ in range(10):
            segment = vad.process_chunk(silent_chunk, t)
            t += 0.032
        self.assertIsNone(segment, "Silence should never produce a speech segment")

    def test_silero_vad_speech_segmentation(self):
        """Verify that speech followed by silence produces a finalized SpeechSegment."""
        vad = SileroVAD(
            sample_rate=16000,
            min_speech_duration_seconds=0.1,
        )
        # Force the deterministic energy/rule fallback path for this unit test.
        vad._is_ready = False
        vad._model = None
        vad._iterator = None
        # Mock calculate_speech_prob: 5 frames of speech (1.0), then 5 frames of silence (0.0)
        probs = [1.0] * 8 + [0.0] * 8
        chunk = np.ones(512, dtype=np.float32) * 0.1
        t = time.time()
        segment = None
        with patch.object(vad, "calculate_speech_prob", side_effect=probs):
            for _ in range(16):
                res = vad.process_chunk(chunk, t)
                if res is not None:
                    segment = res
                t += 0.032

        self.assertIsNotNone(segment, "Speech followed by post-roll silence must yield a segment")
        self.assertGreater(segment.duration_seconds, 0.1)

    def test_ollama_complaint_auditor_json_parsing(self):
        """Test that OllamaComplaintAuditor correctly parses structured complaint JSON."""
        auditor = OllamaComplaintAuditor()
        sample_json = json.dumps({
            "is_complaint": True,
            "category": "repair_quality",
            "severity": "high",
            "summary": "Brake noise persists after service."
        })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": sample_json}

        with patch("requests.post", return_value=mock_resp):
            verdict: ComplaintAnalysis = auditor.analyze(
                khmer_text="សំឡេងហ្វ្រាំងនៅតែលាន់",
                english_text="Brake noise is still loud.",
            )
            self.assertTrue(verdict.is_complaint)
            self.assertEqual(verdict.category, "repair_quality")
            self.assertEqual(verdict.severity, "high")
            self.assertIn("Brake", verdict.summary)

    def test_ollama_normal_conversation(self):
        """Test that normal polite dialogue is flagged as is_complaint=False."""
        auditor = OllamaComplaintAuditor()
        sample_json = json.dumps({
            "is_complaint": False,
            "category": "none",
            "severity": "none",
            "summary": "Customer thanked the technician."
        })

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": sample_json}

        with patch("requests.post", return_value=mock_resp):
            verdict: ComplaintAnalysis = auditor.analyze(
                khmer_text="អរគុណច្រើនបង",
                english_text="Thank you very much, brother.",
            )
            self.assertFalse(verdict.is_complaint)

    def test_telegram_out_complaint_alert(self):
        """Test TelegramOut 3-part complaint dispatching."""
        bot = TelegramOut(token="123456:ABC-DEF", chat_id="12345678")
        complaint_data = {
            "complaint_id": "CMP-999999",
            "category": "pricing",
            "severity": "medium",
            "khmer_transcript": "ថ្លៃពេកហើយបង",
            "english_transcript": "It is too expensive, brother.",
            "summary": "Customer disputed the service cost.",
            "timestamp": "2026-09-17 14:30:00",
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.ok = True
            mock_post.return_value.status_code = 200

            # Mock temp audio and image file
            audio_file = Path(self.temp_dir.name) / "test.wav"
            audio_file.write_bytes(b"RIFFdummywavcontent")
            photo_file = Path(self.temp_dir.name) / "test.jpg"
            photo_file.write_bytes(b"\xFF\xD8\xFFdummyjpg")

            ok = bot.send_complaint_alert(complaint_data, audio_path=audio_file, photo_path=photo_file)
            self.assertTrue(ok)
            # Should have called sendMessage, sendVoice, sendPhoto
            self.assertEqual(mock_post.call_count, 3)

    def test_end_to_end_complaint_service(self):
        """Test end-to-end speech processing in ComplaintMonitoringService."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        service = ComplaintMonitoringService(
            db_conn=self.conn,
            telegram_out=None,
            get_camera_frame_fn=lambda: mock_frame,
            enabled=True,
        )

        # Mock STT, Translator, and Auditor
        service.stt.transcribe = MagicMock(return_value="ខ្ញុំមិនពេញចិត្តទេ ឡានខូចដដែល")
        service.translator.translate_km_to_en = MagicMock(return_value="I am not satisfied, the car is still broken.")
        service.auditor.analyze = MagicMock(return_value=ComplaintAnalysis(
            is_complaint=True,
            category="repair_quality",
            severity="critical",
            summary="Vehicle still broken after repair.",
        ))

        fake_audio = np.zeros(16000 * 2, dtype=np.float32)
        segment = SpeechSegment(
            audio=fake_audio,
            sample_rate=16000,
            start_time=time.time() - 2.0,
            end_time=time.time(),
            duration_seconds=2.0,
        )

        record: ComplaintRecord = service.process_speech_segment(segment)
        self.assertIsNotNone(record)
        self.assertTrue(record.is_complaint)
        self.assertEqual(record.severity, "critical")
        self.assertEqual(record.category, "repair_quality")
        self.assertTrue(Path(record.audio_path).exists())
        self.assertTrue(Path(record.screenshot_path).exists())

        # Verify record in database
        db_records = get_recent_complaints(self.conn, limit=5)
        self.assertEqual(len(db_records), 1)
        self.assertEqual(db_records[0]["complaint_id"], record.complaint_id)

    def test_khmer_script_gate(self):
        """Khmer script validation must accept Khmer text and reject Thai/Latin garbage."""
        self.assertTrue(_has_khmer_script("ឡាន់នៅតែក៏ខ្វក់"))
        self.assertFalse(_has_khmer_script("ร้านได้ใต้กว่า"))
        self.assertFalse(_has_khmer_script("The car is still dirty"))
        self.assertGreaterEqual(_khmer_coverage("ឡាន់នៅតែក៏ខ្វក់"), 0.9)
        self.assertLess(_khmer_coverage("ร้านได้ใต้กว่า"), 0.5)
        self.assertLess(_khmer_coverage("The car is still dirty"), 0.5)

    def test_duplicate_complaint_suppression(self):
        """Identical (looped / replayed) complaint audio must be deduplicated."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        service = ComplaintMonitoringService(
            db_conn=self.conn,
            telegram_out=None,
            get_camera_frame_fn=lambda: mock_frame,
            enabled=True,
            dedup_window_seconds=60.0,
        )
        service.stt.transcribe = MagicMock(return_value="ឡាន់នៅតែក៏ខ្វក់")
        service.translator.translate_km_to_en = MagicMock(return_value="The car is still dirty.")
        service.auditor.analyze = MagicMock(return_value=ComplaintAnalysis(
            is_complaint=True,
            category="cleanliness",
            severity="medium",
            summary="The customer is dissatisfied with the cleanliness of the car.",
        ))

        audio = (np.sin(np.linspace(0, 2 * np.pi * 200, 16000 * 2)) * 0.3).astype(np.float32)
        segment = SpeechSegment(
            audio=audio,
            sample_rate=16000,
            start_time=time.time() - 2.0,
            end_time=time.time(),
            duration_seconds=2.0,
        )

        first = service.process_speech_segment(segment)
        self.assertIsNotNone(first, "First complaint must be processed.")

        second = service.process_speech_segment(
            SpeechSegment(
                audio=audio.copy(),
                sample_rate=16000,
                start_time=time.time() - 2.0,
                end_time=time.time(),
                duration_seconds=2.0,
            )
        )
        self.assertIsNone(second, "Identical replayed complaint must be suppressed.")

        db_records = get_recent_complaints(self.conn, limit=5)
        self.assertEqual(len(db_records), 1)

    def test_telegram_dispatch_requires_severity(self):
        """A 'none' severity result must not be dispatched to Telegram."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_tg = MagicMock()
        mock_tg.enabled = True
        service = ComplaintMonitoringService(
            db_conn=self.conn,
            telegram_out=mock_tg,
            get_camera_frame_fn=lambda: mock_frame,
            enabled=True,
        )
        service.stt.transcribe = MagicMock(return_value="អរគុណច្រើនបង")
        service.translator.translate_km_to_en = MagicMock(return_value="Thank you very much, brother.")
        service.auditor.analyze = MagicMock(return_value=ComplaintAnalysis(
            is_complaint=False,
            category="inquiry",
            severity="none",
            summary="Normal customer dialogue.",
        ))

        audio = np.zeros(16000, dtype=np.float32)
        segment = SpeechSegment(
            audio=audio,
            sample_rate=16000,
            start_time=time.time() - 1.0,
            end_time=time.time(),
            duration_seconds=1.0,
        )
        record = service.process_speech_segment(segment)
        self.assertIsNotNone(record)
        self.assertFalse(record.is_complaint)
        mock_tg.send_complaint_alert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
