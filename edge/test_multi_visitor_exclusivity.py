"""Unit tests for multi-visitor frame exclusivity and anti-poisoning in CustomerVisitMonitor.

Verifies:
1. Two or more people visible in the exact same frame NEVER share the same visitor ID.
2. Moderate feature similarity does not collapse simultaneous visitors into one ID.
3. Track continuity: Each person keeps their unique ID across successive frames.
4. Clutter and unconfirmed detections are rejected and never minted as visitors.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from db import connect
from workplaces.customer_visits import (
    AnonymousVisitorGallery,
    CustomerVisitMonitor,
    REID_MATCH_THRESHOLD,
)
from person import Detection


def _make_unit_vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-6)


def _make_det(
    x1: float,
    y1: float,
    feat: np.ndarray,
    track_id: int,
    conf: float = 0.85,
    clutter: bool = False,
    confirmed: bool = True,
    hits: int = 5,
) -> Detection:
    det = Detection(
        x1=x1,
        y1=y1,
        x2=x1 + 50,
        y2=y1 + 100,
        conf=conf,
        track_id=track_id,
    )
    det.reid_feat = feat
    det.clutter = clutter
    det.confirmed = confirmed
    det.hits = hits
    return det


class MultiVisitorExclusivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / "events.db", check_same_thread=False)
        self.zones = [
            {"id": "entrance", "name": "Entrance", "roi": [0.0, 0.0, 0.5, 1.0], "type": "entrance"},
            {"id": "store_aisle", "name": "Store Aisle", "roi": [0.5, 0.0, 0.5, 1.0], "type": "treatment_room"},
        ]
        self.monitor = CustomerVisitMonitor(
            self.zones,
            confirm_seconds=0.1,
            clear_seconds=0.5,
            grace_seconds=1.0,
            conn=self.conn,
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_simultaneous_detections_never_share_same_id(self) -> None:
        """Multiple people in the same camera frame must each receive a distinct visitor ID."""
        base_feat = _make_unit_vec(100)
        # Create 5 detections with very high similarity (simulating similar clothing / retail lighting)
        detections = []
        for i in range(5):
            # Drifted slightly so similarity is high (> 0.75)
            drift = base_feat * 0.85 + _make_unit_vec(101 + i) * 0.15
            drift = drift / float(np.linalg.norm(drift))
            det = _make_det(
                x1=50.0 + i * 150.0,
                y1=100.0,
                feat=drift,
                track_id=i + 1,
            )
            detections.append(det)

        self.monitor.update(detections, frame_w=1000, frame_h=600, now=100.0)

        assigned_ids = [det.identity for det in detections]
        # 1. All detections must have an assigned ID
        for aid in assigned_ids:
            self.assertIsNotNone(aid)
            self.assertTrue(str(aid).startswith("visitor_"))

        # 2. Every single ID in the frame must be unique (zero duplicates)
        unique_ids = set(assigned_ids)
        self.assertEqual(
            len(unique_ids),
            len(detections),
            f"Expected {len(detections)} distinct IDs in the same frame, but got {len(unique_ids)}: {assigned_ids}",
        )

    def test_track_continuity_across_frames(self) -> None:
        """Each person retains their unique sticky ID across sequential video frames."""
        feats = [_make_unit_vec(200 + i) for i in range(3)]
        
        # Frame 1
        dets_f1 = [
            _make_det(100.0 * (i + 1), 100.0, feats[i], track_id=i + 1)
            for i in range(3)
        ]
        self.monitor.update(dets_f1, frame_w=1000, frame_h=600, now=100.0)
        initial_ids = [d.identity for d in dets_f1]
        self.assertEqual(len(set(initial_ids)), 3)

        # Frame 2 (+ 0.1s later, slight motion)
        dets_f2 = [
            _make_det(100.0 * (i + 1) + 5, 102.0, feats[i], track_id=i + 1)
            for i in range(3)
        ]
        self.monitor.update(dets_f2, frame_w=1000, frame_h=600, now=100.1)
        subsequent_ids = [d.identity for d in dets_f2]

        self.assertEqual(
            initial_ids,
            subsequent_ids,
            "Track identities must remain stable across consecutive video frames",
        )

    def test_clutter_detections_are_ignored(self) -> None:
        """Inanimate clutter (hanging clothes, shelves) is rejected and not assigned visitor IDs."""
        feat_clutter = _make_unit_vec(300)
        det_clutter = _make_det(
            x1=200.0,
            y1=200.0,
            feat=feat_clutter,
            track_id=99,
            clutter=True,  # Inanimate static clutter
        )
        det_person = _make_det(
            x1=100.0,
            y1=100.0,
            feat=_make_unit_vec(301),
            track_id=1,
            clutter=False,
        )

        self.monitor.update([det_clutter, det_person], frame_w=1000, frame_h=600, now=100.0)

        # Clutter must not receive a visitor identity
        self.assertIsNone(det_clutter.identity)
        # Real person must be identified
        self.assertIsNotNone(det_person.identity)
        self.assertTrue(str(det_person.identity).startswith("visitor_"))

    def test_unconfirmed_detections_do_not_mint_visitors(self) -> None:
        """Single-frame unconfirmed detections (e.g. background blips) do not mint visitor IDs."""
        det_unconfirmed = _make_det(
            x1=300.0,
            y1=300.0,
            feat=_make_unit_vec(400),
            track_id=50,
            confirmed=False,
            hits=1,
        )

        self.monitor.update([det_unconfirmed], frame_w=1000, frame_h=600, now=100.0)
        self.assertIsNone(det_unconfirmed.identity)
        self.assertEqual(len(self.monitor.open_sessions()), 0)


if __name__ == "__main__":
    unittest.main()
