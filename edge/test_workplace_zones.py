"""Pipeline zones on the live camera and workplace-aware empty-camera seeding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workplaces import DEFAULT_MASSAGE_ZONES, parse_zone_kind


def _massage_graph() -> dict:
    return {
        "workplace_type": "massage",
        "nodes": [
            {"id": "cam", "data": {"kind": "camera"}},
            {"id": "person", "data": {"kind": "personDetect"}},
            {"id": "reid", "data": {"kind": "anonymousReid"}},
            {"id": "ent", "data": {"kind": "zone", "zoneKind": "entrance", "zoneName": "Entrance"}},
            {"id": "wait", "data": {"kind": "zone", "zoneKind": "waiting", "zoneName": "Waiting"}},
            {
                "id": "room",
                "data": {"kind": "zone", "zoneKind": "treatment_room", "zoneName": "Treatment Room 1"},
            },
            {"id": "desk", "data": {"kind": "zone", "zoneKind": "reception", "zoneName": "Reception"}},
            {"id": "visits", "data": {"kind": "customerVisits"}},
            {"id": "persist", "data": {"kind": "persist"}},
        ],
        "edges": [
            {"id": "e1", "source": "cam", "target": "person", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e2", "source": "cam", "target": "ent", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e3", "source": "cam", "target": "wait", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e4", "source": "cam", "target": "room", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e5", "source": "cam", "target": "desk", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e6", "source": "person", "target": "reid", "sourceHandle": "detections", "targetHandle": "detections"},
            {"id": "e7", "source": "reid", "target": "visits", "sourceHandle": "identity", "targetHandle": "identity"},
            {"id": "e8", "source": "ent", "target": "visits", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e9", "source": "wait", "target": "visits", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e10", "source": "room", "target": "visits", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e11", "source": "desk", "target": "visits", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e12", "source": "visits", "target": "persist", "sourceHandle": "events", "targetHandle": "events"},
        ],
    }


class WorkplaceZoneSeedTests(unittest.TestCase):
    def test_empty_camera_seeds_massage_defaults(self) -> None:
        from workplaces import normalize_workplace_zones

        seeded = normalize_workplace_zones(
            "massage",
            None,
            fallback_roi=[0.30, 0.20, 0.40, 0.60],
            seed_if_empty=True,
        )
        kinds = {z["type"] for z in seeded}
        names = {z["name"] for z in seeded}
        self.assertIn("entrance", kinds)
        self.assertIn("waiting", kinds)
        self.assertIn("treatment_room", kinds)
        self.assertIn("reception", kinds)
        self.assertNotIn("vehicle_bay", kinds)
        self.assertIn("Reception", names)
        self.assertEqual(parse_zone_kind("reception", "massage"), "reception")
        self.assertEqual(len(seeded), len(DEFAULT_MASSAGE_ZONES))

    def test_empty_camera_without_workplace_still_garage(self) -> None:
        from occupancy import normalize_bays

        seeded = normalize_bays(None, seed_if_empty=True)
        kinds = {z.get("type") for z in seeded}
        self.assertIn("vehicle_bay", kinds)


class ApplyPipelineOverlayTests(unittest.TestCase):
    def test_apply_pipeline_writes_zones_onto_active_camera(self) -> None:
        from launcher import LiveStreamEngine

        engine = LiveStreamEngine()
        engine.cfg["workplace_type"] = "massage"
        engine.cfg["active_camera_id"] = "cam-1"
        engine.cfg["cameras"] = [
            {
                "id": "cam-1",
                "name": "Front",
                "source": "0",
                "roi": [0.30, 0.20, 0.40, 0.60],
                "bays": [
                    {
                        "id": "bay_1",
                        "name": "Bay 1",
                        "roi": [0.10, 0.20, 0.35, 0.60],
                        "type": "vehicle_bay",
                    }
                ],
            }
        ]
        engine.cfg["bays"] = list(engine.cfg["cameras"][0]["bays"])

        with patch("launcher.save_config"), patch.object(engine, "_sync_pipeline_models"):
            result = engine.apply_pipeline(_massage_graph())
        self.assertTrue(result["ok"])
        cam = next(c for c in engine.cfg["cameras"] if c["id"] == "cam-1")
        names = {b["name"] for b in cam["bays"]}
        kinds = {b["type"] for b in cam["bays"]}
        self.assertIn("Entrance", names)
        self.assertIn("Reception", names)
        self.assertNotIn("Bay 1", names)
        self.assertIn("reception", kinds)
        self.assertEqual(engine.cfg["bays"], cam["bays"])
        self.assertEqual(result["bays"], cam["bays"])
        self.assertEqual(result["active_camera_id"], "cam-1")

    def test_apply_pipeline_keeps_matching_polygon(self) -> None:
        from launcher import LiveStreamEngine

        polygon = [[0.08, 0.12], [0.30, 0.12], [0.30, 0.62], [0.08, 0.62]]
        engine = LiveStreamEngine()
        engine.cfg["workplace_type"] = "massage"
        engine.cfg["active_camera_id"] = "cam-1"
        engine.cfg["cameras"] = [
            {
                "id": "cam-1",
                "name": "Front",
                "source": "0",
                "bays": [
                    {
                        "id": "ent",
                        "name": "Entrance",
                        "roi": [0.05, 0.15, 0.25, 0.70],
                        "type": "entrance",
                        "polygon": polygon,
                    }
                ],
            }
        ]
        engine.cfg["bays"] = list(engine.cfg["cameras"][0]["bays"])

        with patch("launcher.save_config"), patch.object(engine, "_sync_pipeline_models"):
            result = engine.apply_pipeline(_massage_graph())
        self.assertTrue(result["ok"])
        cam = next(c for c in engine.cfg["cameras"] if c["id"] == "cam-1")
        entrance = next(b for b in cam["bays"] if b["id"] == "ent" or b["name"] == "Entrance")
        self.assertEqual(entrance["polygon"], polygon)
        self.assertEqual(entrance["roi"], [0.05, 0.15, 0.25, 0.70])


if __name__ == "__main__":
    unittest.main()
