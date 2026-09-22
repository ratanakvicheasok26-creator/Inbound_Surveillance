"""Integration test: Verify runtime identity enrollment while streaming."""

import os
import shutil
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_id import try_create_face_recognizer, FaceRecognizer
from launcher import LiveStreamEngine


def _face_cfg() -> dict:
    return {
        "enable_face_id": True,
        "faces_dir": str(ROOT / "faces"),
        "face_match_threshold": 0.40,
    }


def test_transient_miss_is_retried():
    """A first-pass failed extract must not be cached as None forever."""
    rec = try_create_face_recognizer(_face_cfg())
    assert rec is not None
    test_dir = ROOT / "faces" / "RetryStaff"
    source_photo = list((ROOT / "faces" / "HourMeng").glob("*.jpg"))[0]
    orig = rec.extract_embedding_from_image
    calls = {"n": 0}

    def flaky(img, detector=None, recognizer=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return orig(img, detector=detector, recognizer=recognizer)

    rec.extract_embedding_from_image = flaky
    try:
        os.makedirs(test_dir, exist_ok=True)
        shutil.copy(source_photo, test_dir / "ref.jpg")
        rec.reload_enrolled_faces()
        assert "RetryStaff" in rec.known_embeddings, "First extract miss was not retried in-reload"
        retry_keys = [k for k in rec._embedding_cache if "RetryStaff" in k[0]]
        assert retry_keys, "RetryStaff photo was not cached"
        assert all(rec._embedding_cache[k] is not None for k in retry_keys)
        print("[OK] Transient extract miss was retried instead of cached")
    finally:
        rec.extract_embedding_from_image = orig
        if test_dir.exists():
            shutil.rmtree(test_dir)
        rec.reload_enrolled_faces()
        assert "RetryStaff" not in rec.known_embeddings


def test_live_reload():
    cfg = _face_cfg()
    rec = try_create_face_recognizer(cfg)
    assert rec is not None, "Failed to create FaceRecognizer"
    initial_members = set(rec.known_embeddings.keys())
    print(f"Initial enrolled members: {initial_members}")
    assert "TestStaff" not in initial_members

    # Start a background streaming worker simulating 20 FPS face recognition
    stop_event = threading.Event()
    worker_errors = []
    frames_processed = [0]

    dummy_crop = np.zeros((200, 150, 3), dtype=np.uint8)

    def worker():
        while not stop_event.is_set():
            try:
                rec.recognize_in_crop(dummy_crop)
                frames_processed[0] += 1
                time.sleep(0.01)
            except Exception as e:
                worker_errors.append(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    test_dir = ROOT / "faces" / "TestStaff"
    source_photo = list((ROOT / "faces" / "HourMeng").glob("*.jpg"))[0]

    try:
        # 1. Add new identity
        os.makedirs(test_dir, exist_ok=True)
        shutil.copy(source_photo, test_dir / "ref.jpg")
        print(f"Created temporary identity: {test_dir}")

        # 2. Reload enrolled faces while worker is running
        count = rec.reload_enrolled_faces()
        print(f"Reload returned count: {count}")
        assert "TestStaff" in rec.known_embeddings, "TestStaff was not picked up!"
        print("[OK] TestStaff was successfully enrolled during live worker execution!")

        # 3. Test caching: second reload should be instant (cached)
        t0 = time.perf_counter()
        rec.reload_enrolled_faces()
        t_reload = (time.perf_counter() - t0) * 1000
        print(f"Cached reload took: {t_reload:.2f} ms")
        assert t_reload < 100, f"Reload was too slow: {t_reload}ms"

        # 4. Now test LiveStreamEngine.reload_face_id
        engine = LiveStreamEngine()
        engine.cfg = cfg
        engine.face_rec = rec
        engine_res = engine.reload_face_id()
        assert engine_res == count
        assert engine.force_infer is True
        print(f"[OK] LiveStreamEngine.reload_face_id succeeded, enrolled {engine_res} staff!")

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("Cleaned up temporary identity directory")

        # Reload after cleanup
        rec.reload_enrolled_faces()
        assert "TestStaff" not in rec.known_embeddings, "TestStaff should have been removed"
        print("[OK] Cleanup verified; TestStaff removed from memory")

        stop_event.set()
        t.join()

    print(f"Worker processed {frames_processed[0]} frames during test with {len(worker_errors)} errors.")
    assert len(worker_errors) == 0, f"Worker encountered errors: {worker_errors}"
    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test_transient_miss_is_retried()
    test_live_reload()
