"""Multi-object tracking and body ReID identity persistence."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person import (
    Detection,
    backpack_clutter_keypoints,
    engine_bay_keypoints,
    standing_person_keypoints,
)
from reid import BodyReIDExtractor, appearance_embedding
from runtime import (
    DEFAULT_KPT_CONF,
    EDGE_WEIGHTS,
    person_weights_name,
    resolve_kpt_conf,
    resolve_runtime,
)
from tracker import PersonTracker, Track


def _det(
    *,
    x1=80,
    y1=40,
    x2=160,
    y2=280,
    name=None,
    staff=False,
    feat=None,
) -> Detection:
    det = Detection(x1, y1, x2, y2, 0.9, standing_person_keypoints())
    det.accepted = True
    det.identity = name
    det.is_staff = staff
    det.identity_conf = 0.9 if staff else 0.0
    det.reid_feat = feat
    return det


class RuntimeProfileTests(unittest.TestCase):
    def test_cpu_profile(self):
        profile = resolve_runtime({"runtime": "cpu", "weights": "yolo11n-pose.pt"})
        self.assertEqual(profile.name, "cpu")
        self.assertEqual(profile.yolo_device, "cpu")
        self.assertTrue(profile.reid_enabled)
        self.assertGreaterEqual(profile.track_min_hits, 2)

    def test_person_weights_never_use_detect_checkpoint(self):
        self.assertIn("pose", EDGE_WEIGHTS)
        self.assertEqual(person_weights_name("yolo11n_improved.pt"), "yolo11n-pose.pt")
        self.assertEqual(person_weights_name("yolo11n.pt"), "yolo11n-pose.pt")
        self.assertEqual(person_weights_name("yolo11s-pose.pt"), "yolo11s-pose.pt")
        profile = resolve_runtime({"runtime": "cpu", "weights": "yolo11n_improved.pt"})
        self.assertEqual(profile.weights_name, "yolo11n-pose.pt")

    def test_default_kpt_conf_matches_anatomy_helpers(self):
        self.assertGreaterEqual(DEFAULT_KPT_CONF, 0.35)
        self.assertGreaterEqual(resolve_kpt_conf({}), 0.35)

    def test_tinypose_default_kpt_conf_is_below_yolo_floor(self):
        from runtime import DEFAULT_TINYPOSE_KPT_CONF

        self.assertLessEqual(DEFAULT_TINYPOSE_KPT_CONF, 0.15)
        self.assertLessEqual(resolve_kpt_conf({"pose_engine": "tinypose"}), 0.15)
        self.assertGreaterEqual(resolve_kpt_conf({"pose_engine": "tinypose", "kpt_conf": 0.35}), 0.35)

    def test_cuda_falls_back_without_gpu(self):
        profile = resolve_runtime({"runtime": "cuda"})
        self.assertIn(profile.name, ("cuda", "cpu"))
        if profile.name == "cpu":
            self.assertEqual(profile.yolo_device, "cpu")


class TrackerIdentityTests(unittest.TestCase):
    def test_identity_survives_face_miss(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        confirmed = []
        for _ in range(3):
            confirmed = tracker.update([_det(name="George", staff=True)])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].identity, "George")
        self.assertTrue(confirmed[0].is_staff)
        track_id = confirmed[0].track_id

        turned = tracker.update([_det(name="Employee", staff=False)])
        self.assertEqual(len(turned), 1)
        self.assertEqual(turned[0].identity, "George")
        self.assertTrue(turned[0].is_staff)
        self.assertEqual(turned[0].track_id, track_id)

    def test_two_people_keep_separate_ids(self):
        tracker = PersonTracker(max_age=10, min_hits=2, iou_threshold=0.3)
        for _ in range(2):
            tracker.update(
                [
                    _det(x1=80, y1=40, x2=160, y2=280, name="George", staff=True),
                    _det(x1=400, y1=40, x2=480, y2=280, name="Alex", staff=True),
                ]
            )
        out = tracker.update(
            [
                _det(x1=82, y1=42, x2=162, y2=278, name="Employee", staff=False),
                _det(x1=398, y1=38, x2=482, y2=282, name="Employee", staff=False),
            ]
        )
        names = sorted(d.identity for d in out)
        self.assertEqual(names, ["Alex", "George"])
        self.assertEqual(len({d.track_id for d in out}), 2)

    def test_unconfirmed_tracks_are_withheld(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        out = tracker.update([_det(name="George", staff=True)])
        self.assertEqual(out, [])
        missed = tracker.update([])
        self.assertEqual(missed, [])

    def test_unconfirmed_tracks_returned_when_requested(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        # Frame 1: new track, hits=1, return_unconfirmed=True
        out = tracker.update([_det(name="George", staff=True)], return_unconfirmed=True)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].confirmed)
        self.assertEqual(out[0].hits, 1)

        # Frame 2: hits=2, still unconfirmed
        out = tracker.update([_det(name="George", staff=True)], return_unconfirmed=True)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].confirmed)
        self.assertEqual(out[0].hits, 2)

        # Frame 3: hits=3, now confirmed!
        out = tracker.update([_det(name="George", staff=True)], return_unconfirmed=True)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)

    def test_confirmed_track_coasts_on_miss(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        confirmed = []
        for _ in range(3):
            confirmed = tracker.update([_det(name="George", staff=True)])
        self.assertEqual(len(confirmed), 1)
        track_id = confirmed[0].track_id
        coasted = tracker.update([])
        self.assertEqual(len(coasted), 1)
        self.assertEqual(coasted[0].identity, "George")
        self.assertTrue(coasted[0].is_staff)
        self.assertEqual(coasted[0].track_id, track_id)
        for _ in range(11):
            coasted = tracker.update([])
        self.assertEqual(coasted, [])

    def test_coasted_track_keeps_skeleton(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        confirmed = []
        for _ in range(3):
            confirmed = tracker.update([_det(name="George", staff=True)])
        self.assertTrue(confirmed[0].keypoints)
        coasted = tracker.update([])
        self.assertEqual(len(coasted), 1)
        self.assertTrue(coasted[0].coasting)
        self.assertTrue(len(coasted[0].keypoints) > 0)
        self.assertEqual(coasted[0].identity, "George")

    def test_static_unknown_person_is_kept_tracked(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3, static_hits=20)
        out = []
        for _ in range(25):
            det = _det(name=None, staff=False)
            det.conf = 0.75
            out = tracker.update([det])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, 1)
        self.assertFalse(out[0].coasting)

    def test_static_weak_anatomy_is_clutter(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3, static_hits=10)
        out = []
        for _ in range(15):
            det = Detection(80, 40, 160, 280, 0.40, backpack_clutter_keypoints())
            det.accepted = True
            out = tracker.update([det])
        self.assertEqual(out, [])
        self.assertTrue(any(t.clutter for t in tracker.tracks))

    def test_frozen_engine_skeleton_is_clutter_when_liveness_is_dead(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3, static_hits=10)
        kpts = [(x, y, 0.85 if c > 0 else 0.0) for x, y, c in engine_bay_keypoints()]
        out = []
        for _ in range(20):
            det = Detection(90, 40, 190, 280, 0.75, kpts)
            det.accepted = True
            det.liveness = 0.4
            out = tracker.update([det])
        self.assertTrue(any(t.clutter for t in tracker.tracks))
        self.assertEqual(out, [])

    def test_moving_high_conf_unknown_is_kept(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3, static_hits=20)
        out = []
        for i in range(8):
            det = _det(x1=80 + i * 12, y1=40, x2=160 + i * 12, y2=280, name=None, staff=False)
            det.conf = 0.75
            out = tracker.update([det])
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0].coasting)

    def test_static_staff_is_not_cluttered_by_low_conf(self):
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3, static_hits=20)
        out = []
        for _ in range(8):
            det = _det(name="George", staff=True)
            det.conf = 0.40
            out = tracker.update([det])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].identity, "George")

    def test_shrunk_box_stays_on_same_track(self):
        tracker = PersonTracker(max_age=10, min_hits=2, iou_threshold=0.3)
        out = []
        for _ in range(2):
            out = tracker.update([_det(x1=80, y1=40, x2=160, y2=280, name="George", staff=True)])
        track_id = out[0].track_id
        shrunk = tracker.update([_det(x1=100, y1=180, x2=150, y2=250, name="Employee", staff=False)])
        self.assertEqual(len(shrunk), 1)
        self.assertEqual(shrunk[0].track_id, track_id)
        self.assertEqual(shrunk[0].identity, "George")

    def test_stand_to_crouch_id_stability(self):
        tracker = PersonTracker(max_age=30, min_hits=2, iou_threshold=0.3)
        out = []
        for _ in range(3):
            out = tracker.update([_det(x1=100, y1=40, x2=180, y2=280, name="George", staff=True)])
        self.assertEqual(len(out), 1)
        track_id = out[0].track_id

        # Morph from standing (80x240) to crouching (100x70) with low IoU
        crouched_det = Detection(90, 210, 190, 280, 0.85, standing_person_keypoints())
        crouched_det.accepted = True
        crouch_out = tracker.update([crouched_det])
        self.assertEqual(len(crouch_out), 1)
        self.assertEqual(crouch_out[0].track_id, track_id)
        self.assertEqual(crouch_out[0].identity, "George")

    def test_stand_to_sit_keeps_track_id(self):
        tracker = PersonTracker(max_age=30, min_hits=2, iou_threshold=0.3)
        out = []
        for _ in range(3):
            out = tracker.update([_det(x1=80, y1=40, x2=160, y2=280, name="George", staff=True)])
        track_id = out[0].track_id
        seated = Detection(90, 150, 210, 280, 0.85, standing_person_keypoints())
        seated.accepted = True
        seated.identity = "Employee"
        seated.is_staff = False
        seated_out = tracker.update([seated])
        self.assertEqual(len(seated_out), 1)
        self.assertEqual(seated_out[0].track_id, track_id)
        self.assertEqual(seated_out[0].identity, "George")

    def test_low_conf_preserves_existing_track(self):
        tracker = PersonTracker(max_age=10, min_hits=2, iou_threshold=0.3, low_iou_threshold=0.15)
        out = []
        for _ in range(3):
            out = tracker.update([_det(x1=100, y1=40, x2=180, y2=280, name="George", staff=True)])
        self.assertEqual(len(out), 1)
        track_id = out[0].track_id

        # Low confidence detection (conf=0.18) matching via ByteTrack secondary association
        low_det = Detection(102, 42, 178, 278, 0.18, standing_person_keypoints())
        low_out = tracker.update([], low_detections=[low_det])
        self.assertEqual(len(low_out), 1)
        self.assertEqual(low_out[0].track_id, track_id)
        self.assertEqual(low_out[0].identity, "George")
        self.assertFalse(low_out[0].coasting)

    def test_low_conf_does_not_spawn_track(self):
        tracker = PersonTracker(max_age=10, min_hits=2, iou_threshold=0.3)
        low_det = Detection(100, 40, 180, 280, 0.18, standing_person_keypoints())
        out = tracker.update([], low_detections=[low_det])
        self.assertEqual(out, [])
        self.assertEqual(len(tracker.tracks), 0)

    def test_confirmed_track_with_weak_anatomy_not_killed(self):
        tracker = PersonTracker(max_age=30, min_hits=2, iou_threshold=0.3, static_hits=10)
        out = []
        for _ in range(3):
            out = tracker.update([_det(name=None, staff=False)])
        self.assertEqual(len(out), 1)
        track_id = out[0].track_id

        # Person leans in / occluded - weak anatomy for 15 frames, but living person is not clutter
        for _ in range(15):
            det = Detection(80, 40, 160, 280, 0.70, backpack_clutter_keypoints())
            det.accepted = True
            # give slight motion so not frozen inanimate
            det.liveness = 5.0
            out = tracker.update([det])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, track_id)
        self.assertFalse(tracker.tracks[0].clutter)


class ReIDEmbeddingTests(unittest.TestCase):
    def test_fallback_embedding_is_stable_and_discriminative(self):
        extractor = BodyReIDExtractor(model_path=None)
        red = np.zeros((240, 120, 3), dtype=np.uint8)
        red[20:220, 20:100] = (40, 40, 200)
        blue = np.zeros((240, 120, 3), dtype=np.uint8)
        blue[20:220, 20:100] = (200, 40, 40)
        feat_a = extractor.extract(red, (20, 20, 100, 220))
        feat_b = extractor.extract(red, (22, 18, 98, 218))
        feat_c = extractor.extract(blue, (20, 20, 100, 220))
        self.assertEqual(feat_a.shape[0], 512)
        self.assertGreater(BodyReIDExtractor.cosine_similarity(feat_a, feat_b), 0.90)
        self.assertGreater(
            BodyReIDExtractor.cosine_similarity(feat_a, feat_b),
            BodyReIDExtractor.cosine_similarity(feat_a, feat_c),
        )

    def test_gallery_rebinds_after_new_track(self):
        tracker = PersonTracker(max_age=10, min_hits=2, iou_threshold=0.3, reid_threshold=0.50)
        extractor = BodyReIDExtractor(model_path=None)
        red = np.zeros((240, 120, 3), dtype=np.uint8)
        red[20:220, 20:100] = (30, 80, 210)
        feat = extractor.extract(red, (20, 20, 100, 220))
        tracker.update([_det(name="George", staff=True, feat=feat)])
        tracker.update([_det(name="George", staff=True, feat=feat)])
        tracker.reset()
        tracker.gallery.remember("George", feat)
        revived = None
        for _ in range(2):
            revived = tracker.update(
                [_det(x1=300, y1=40, x2=380, y2=280, name="Employee", staff=False, feat=feat)]
            )
        self.assertEqual(len(revived), 1)
        self.assertEqual(revived[0].identity, "George")
        self.assertTrue(revived[0].is_staff)

    def test_appearance_embedding_rejects_empty_crop(self):
        zeros = appearance_embedding(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(float(np.linalg.norm(zeros)), 0.0)

    def test_persistent_gallery_survives_tracker_reset(self):
        from reid import PersistentReIDGallery
        shared_gallery = PersistentReIDGallery()
        tracker = PersonTracker(max_age=10, min_hits=2, gallery=shared_gallery, camera_id="cam-1")
        dummy_feat = np.ones(512, dtype=np.float32)
        dummy_feat /= np.linalg.norm(dummy_feat)
        # Remember George with valid upright bbox
        shared_gallery.remember("George", dummy_feat, bbox=(10, 10, 60, 150), face_conf=0.95, is_staff=True)
        self.assertIn("George", shared_gallery.embeddings)
        # Tracker reset with clear_gallery=False keeps embeddings
        tracker.reset(clear_gallery=False)
        self.assertIn("George", tracker.gallery.embeddings)
        # Explicit clear_gallery=True wipes them
        tracker.reset(clear_gallery=True)
        self.assertNotIn("George", tracker.gallery.embeddings)

    def test_spatial_exclusivity_blocks_cross_camera_identity_theft(self):
        from reid import PersistentReIDGallery
        gallery = PersistentReIDGallery(exclusivity_timeout=5.0)
        feat = np.ones(512, dtype=np.float32)
        feat /= np.linalg.norm(feat)
        t0 = 1000.0
        # Camera 1 enrolls and claims George
        gallery.remember("George", feat, bbox=(10, 10, 60, 150), face_conf=0.90, camera_id="cam-1", now=t0)
        # Camera 1 queries George -> match found
        name1, score1 = gallery.match(feat, threshold=0.50, camera_id="cam-1", now=t0 + 1.0)
        self.assertEqual(name1, "George")
        # Camera 2 queries George while active on Camera 1 -> blocked by spatial exclusivity!
        name2, score2 = gallery.match(feat, threshold=0.50, camera_id="cam-2", now=t0 + 1.0)
        self.assertIsNone(name2)
        # After 5.1 seconds of inactivity on Camera 1, Camera 2 can claim George
        name2_after, score2_after = gallery.match(feat, threshold=0.50, camera_id="cam-2", now=t0 + 5.1)
        self.assertEqual(name2_after, "George")

    def test_anti_poisoning_rejects_horizontal_or_low_confidence_crops(self):
        from reid import PersistentReIDGallery
        gallery = PersistentReIDGallery()
        feat = np.ones(512, dtype=np.float32)
        feat /= np.linalg.norm(feat)
        # 1. Low face confidence (< 0.70) rejected
        ok = gallery.remember("George", feat, bbox=(10, 10, 50, 120), face_conf=0.65, is_staff=True)
        self.assertFalse(ok)
        self.assertNotIn("George", gallery.embeddings)
        # 2. Horizontal / non-upright crop (w=100, h=40 -> aspect=0.4 < 1.0) rejected
        ok = gallery.remember("George", feat, bbox=(10, 10, 110, 50), face_conf=0.95, is_staff=True)
        self.assertFalse(ok)
        self.assertNotIn("George", gallery.embeddings)
        # 3. Tiny degenerate crop (w=10, h=15) rejected
        ok = gallery.remember("George", feat, bbox=(10, 10, 20, 25), face_conf=0.95, is_staff=True)
        self.assertFalse(ok)
        self.assertNotIn("George", gallery.embeddings)
        # 4. Valid upright worker crop accepted
        ok = gallery.remember("George", feat, bbox=(10, 10, 60, 160), face_conf=0.85, is_staff=True)
        self.assertTrue(ok)
        self.assertIn("George", gallery.embeddings)

    def test_gallery_capped_at_max_embeddings_with_fifo(self):
        from reid import PersistentReIDGallery
        gallery = PersistentReIDGallery(max_per_name=4)
        for i in range(6):
            vec = np.zeros(512, dtype=np.float32)
            vec[i] = 1.0
            gallery.remember("George", vec, bbox=(10, 10, 60, 160), face_conf=0.90, is_staff=True)
        # Must be capped at 4
        self.assertEqual(len(gallery.embeddings["George"]), 4)
        # Oldest embeddings (0 and 1) should have been evicted by FIFO
        self.assertEqual(gallery.embeddings["George"][0][2], 1.0)
        self.assertEqual(gallery.embeddings["George"][-1][5], 1.0)

    def test_should_extract_reid_throttling_and_gating(self):
        from tracker import should_extract_reid
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        # 1. Degenerate crop: w < 20 or h < 40
        det_skinny = Detection(10, 10, 25, 100, 0.9)  # w = 15 < 20
        self.assertFalse(should_extract_reid(det_skinny, tracker))
        det_short = Detection(10, 10, 50, 30, 0.9)   # h = 20 < 40
        self.assertFalse(should_extract_reid(det_short, tracker))

        # 2. Unmatched detection (no tracks yet) -> True
        det_valid = Detection(10, 10, 60, 150, 0.9)
        self.assertTrue(should_extract_reid(det_valid, tracker))

        # Create confirmed track with 3 hits
        for _ in range(3):
            tracker.update([det_valid])
        self.assertEqual(len(tracker.tracks), 1)
        trk = tracker.tracks[0]
        self.assertEqual(trk.hits, 3)

        # Trk has no reid_features -> True to capture initial profile
        self.assertTrue(should_extract_reid(det_valid, tracker))

        # Now simulate track has extracted reid_features
        trk.last_reid_frame = tracker.frame_count
        trk.reid_features.append(np.ones(512, dtype=np.float32))

        # Next frame (interval < 30) -> should NOT extract (throttled)
        tracker.frame_count += 1
        self.assertFalse(should_extract_reid(det_valid, tracker, min_interval=30))

        # After 30 frames elapsed -> should extract periodically
        tracker.frame_count += 35
        self.assertTrue(should_extract_reid(det_valid, tracker, min_interval=30))

        # Spatial match dropped (new detection far away) -> True (for recovery)
        det_distant = Detection(300, 300, 350, 450, 0.9)
        self.assertTrue(should_extract_reid(det_distant, tracker))

    def test_pipeline_reid_call_count_throttled(self):
        from unittest.mock import MagicMock
        from tracker import run_identity_pipeline
        tracker = PersonTracker(max_age=10, min_hits=3, iou_threshold=0.3)
        mock_reid = MagicMock()
        mock_reid.extract.return_value = np.ones(512, dtype=np.float32)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Simulate 90 frames with 2 tracks
        for _ in range(90):
            d1 = Detection(50, 50, 150, 300, 0.9, standing_person_keypoints())
            d2 = Detection(350, 50, 450, 300, 0.9, standing_person_keypoints())
            run_identity_pipeline(frame, [d1, d2], tracker, reid=mock_reid, reid_interval=30)

        # In 90 frames with 2 tracks:
        # Without throttling: 90 * 2 = 180 calls!
        # With throttling (interval=30): ~3-4 calls per track = 6-8 calls total.
        self.assertLessEqual(mock_reid.extract.call_count, 8)
        self.assertGreaterEqual(mock_reid.extract.call_count, 2)


class LivenessAndNegativesTests(unittest.TestCase):
    def test_box_motion_energy_separates_static_from_moving(self):
        from liveness import DEAD_MOTION_ENERGY, box_motion_energy, to_probe_gray

        still = np.zeros((120, 120, 3), dtype=np.uint8)
        still[20:80, 20:80] = 80
        moved = still.copy()
        moved[20:80, 20:80] = 200
        gray_a = to_probe_gray(still)
        gray_b = to_probe_gray(moved)
        self.assertLess(box_motion_energy(gray_a, gray_a, (20, 20, 80, 80)), DEAD_MOTION_ENERGY)
        self.assertGreater(box_motion_energy(gray_b, gray_a, (20, 20, 80, 80)), DEAD_MOTION_ENERGY)

    def test_bank_hard_negatives_writes_crop(self):
        from negatives import bank_hard_negatives

        tmp = tempfile.TemporaryDirectory()
        try:
            frame = np.zeros((120, 120, 3), dtype=np.uint8)
            frame[20:80, 20:80] = 80
            trk = Track(track_id=7, bbox=(20, 20, 80, 80), conf=0.75)
            paths = bank_hard_negatives(frame, [trk], Path(tmp.name))
            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].is_file())
            self.assertTrue(paths[0].with_suffix(".json").is_file())
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
