"""Pipeline graph compiler: typed ports, cycles, workplace flags."""

from __future__ import annotations

import unittest

from graph.compile import (
    compile_to_config,
    complaint_monitoring_wanted,
    pipeline_runtime_flags,
    validate_graph,
)


def _garage_graph() -> dict:
    return {
        "workplace_type": "garage",
        "nodes": [
            {"id": "cam", "data": {"kind": "camera"}},
            {"id": "person", "data": {"kind": "personDetect"}},
            {"id": "pose", "data": {"kind": "pose"}},
            {"id": "face", "data": {"kind": "faceId"}},
            {"id": "vehicle", "data": {"kind": "vehicleDetect"}},
            {"id": "zone1", "data": {"kind": "zone", "zoneKind": "vehicle_bay", "zoneName": "Lift Bay 1"}},
            {"id": "labor", "data": {"kind": "employeeLabor"}},
            {"id": "persist", "data": {"kind": "persist"}},
        ],
        "edges": [
            {"id": "e1", "source": "cam", "target": "person", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e2", "source": "cam", "target": "vehicle", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e3", "source": "cam", "target": "zone1", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e4", "source": "person", "target": "pose", "sourceHandle": "detections", "targetHandle": "detections"},
            {"id": "e5", "source": "pose", "target": "face", "sourceHandle": "detections", "targetHandle": "detections"},
            {"id": "e6", "source": "face", "target": "labor", "sourceHandle": "identity", "targetHandle": "identity"},
            {"id": "e7", "source": "zone1", "target": "labor", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e8", "source": "labor", "target": "persist", "sourceHandle": "events", "targetHandle": "events"},
        ],
    }


def _massage_graph() -> dict:
    return {
        "workplace_type": "massage",
        "nodes": [
            {"id": "cam", "data": {"kind": "camera"}},
            {"id": "person", "data": {"kind": "personDetect"}},
            {"id": "reid", "data": {"kind": "anonymousReid"}},
            {"id": "ent", "data": {"kind": "zone", "zoneKind": "entrance", "zoneName": "Entrance"}},
            {"id": "visits", "data": {"kind": "customerVisits"}},
            {"id": "persist", "data": {"kind": "persist"}},
        ],
        "edges": [
            {"id": "e1", "source": "cam", "target": "person", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e2", "source": "cam", "target": "ent", "sourceHandle": "frames", "targetHandle": "frames"},
            {"id": "e3", "source": "person", "target": "reid", "sourceHandle": "detections", "targetHandle": "detections"},
            {"id": "e4", "source": "reid", "target": "visits", "sourceHandle": "identity", "targetHandle": "identity"},
            {"id": "e5", "source": "ent", "target": "visits", "sourceHandle": "zones", "targetHandle": "zones"},
            {"id": "e6", "source": "visits", "target": "persist", "sourceHandle": "events", "targetHandle": "events"},
        ],
    }


class GraphCompileTests(unittest.TestCase):
    def test_garage_template_enables_vehicle(self) -> None:
        cfg = compile_to_config(_garage_graph())
        self.assertTrue(cfg["vehicle_detect"])
        self.assertTrue(cfg["person_detect"])
        self.assertTrue(cfg["face_id_enabled"])
        self.assertTrue(cfg["employee_labor"])
        self.assertFalse(cfg["customer_visits"])
        self.assertEqual(cfg["workplace_type"], "garage")

    def test_removing_vehicle_disables_yolo_vehicle(self) -> None:
        graph = _garage_graph()
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "vehicle"]
        graph["edges"] = [e for e in graph["edges"] if e["target"] != "vehicle"]
        cfg = compile_to_config(graph)
        self.assertFalse(cfg["vehicle_detect"])
        self.assertTrue(cfg["person_detect"])

    def test_cycle_is_invalid(self) -> None:
        graph = {
            "nodes": [
                {"id": "cam", "data": {"kind": "camera"}},
                {"id": "a", "data": {"kind": "personDetect"}},
                {"id": "b", "data": {"kind": "pose"}},
            ],
            "edges": [
                {"id": "ca", "source": "cam", "target": "a", "sourceHandle": "frames", "targetHandle": "frames"},
                {"id": "ab", "source": "a", "target": "b", "sourceHandle": "detections", "targetHandle": "detections"},
                {"id": "ba", "source": "b", "target": "a", "sourceHandle": "detections", "targetHandle": "detections"},
            ],
        }
        errors = validate_graph(graph)
        self.assertTrue(any("cycle" in err.lower() for err in errors))
        with self.assertRaises(ValueError):
            compile_to_config(graph)

    def test_missing_camera_is_invalid(self) -> None:
        errors = validate_graph({"nodes": [{"id": "p", "data": {"kind": "personDetect"}}], "edges": []})
        self.assertTrue(any("Camera" in err for err in errors))

    def test_port_mismatch_is_invalid(self) -> None:
        graph = {
            "nodes": [
                {"id": "cam", "data": {"kind": "camera"}},
                {"id": "labor", "data": {"kind": "employeeLabor"}},
            ],
            "edges": [
                {"id": "bad", "source": "cam", "target": "labor", "sourceHandle": "frames", "targetHandle": "identity"},
            ],
        }
        errors = validate_graph(graph)
        self.assertTrue(any("cannot connect" in err for err in errors))

    def test_monitor_requires_identity_and_zones(self) -> None:
        graph = {
            "workplace_type": "garage",
            "nodes": [
                {"id": "cam", "data": {"kind": "camera"}},
                {"id": "labor", "data": {"kind": "employeeLabor", "label": "Employee labor"}},
            ],
            "edges": [],
        }
        errors = validate_graph(graph)
        self.assertTrue(any("identity" in err.lower() for err in errors))
        self.assertTrue(any("zone" in err.lower() for err in errors))

    def test_massage_reception_zone_kind(self) -> None:
        graph = _massage_graph()
        graph["nodes"].append(
            {
                "id": "desk",
                "data": {"kind": "zone", "zoneKind": "reception", "zoneName": "Reception"},
            }
        )
        graph["edges"].append(
            {
                "id": "e-desk",
                "source": "cam",
                "target": "desk",
                "sourceHandle": "frames",
                "targetHandle": "frames",
            }
        )
        cfg = compile_to_config(graph)
        kinds = {z["type"] for z in cfg["bays"]}
        self.assertIn("reception", kinds)
        desk = next(z for z in cfg["bays"] if z["id"] == "desk")
        self.assertEqual(desk["roi"], [0.38, 0.06, 0.14, 0.14])

    def test_massage_template_sets_visits(self) -> None:
        cfg = compile_to_config(_massage_graph())
        self.assertEqual(cfg["workplace_type"], "massage")
        self.assertTrue(cfg["reid_enabled"])
        self.assertTrue(cfg["customer_visits"])
        self.assertFalse(cfg["vehicle_detect"])
        self.assertFalse(cfg["face_id_enabled"])
        self.assertTrue(cfg["persist_events"])
        self.assertFalse(cfg["complaint_intake"])

    def test_massage_complaint_node_enables_intake(self) -> None:
        graph = _massage_graph()
        graph["nodes"].append({"id": "complaint", "data": {"kind": "complaintIntake"}})
        graph["edges"].append(
            {
                "id": "e7",
                "source": "visits",
                "target": "complaint",
                "sourceHandle": "events",
                "targetHandle": "events",
            }
        )
        cfg = compile_to_config(graph)
        self.assertTrue(cfg["complaint_intake"])

    def test_alert_rule_does_not_overwrite_absent_seconds(self) -> None:
        graph = _garage_graph()
        graph["nodes"].append({"id": "alert", "data": {"kind": "alertRule", "cooldownSec": 20}})
        graph["edges"].append(
            {
                "id": "e9",
                "source": "labor",
                "target": "alert",
                "sourceHandle": "events",
                "targetHandle": "events",
            }
        )
        cfg = compile_to_config(graph)
        self.assertNotIn("absent_seconds", cfg)
        self.assertEqual(cfg["cooldown_seconds"], 20)
        self.assertTrue(cfg["alerts_enabled"])

    def test_runtime_flags_without_graph_keep_defaults(self) -> None:
        garage = pipeline_runtime_flags({"workplace_type": "garage"})
        self.assertTrue(garage["valid"])
        self.assertTrue(garage["vehicle_detect"])
        self.assertTrue(garage["employee_labor"])
        self.assertFalse(garage["customer_visits"])
        self.assertFalse(garage["complaint_intake"])
        massage = pipeline_runtime_flags({"workplace_type": "massage"})
        self.assertFalse(massage["vehicle_detect"])
        self.assertTrue(massage["customer_visits"])
        self.assertFalse(massage["face_id"])
        self.assertTrue(massage["complaint_intake"])

    def test_runtime_flags_follow_compiled_graph(self) -> None:
        graph = _garage_graph()
        graph["nodes"] = [n for n in graph["nodes"] if n["id"] != "vehicle"]
        graph["edges"] = [e for e in graph["edges"] if e["target"] != "vehicle"]
        compiled = compile_to_config(graph)
        compiled["pipeline_graph"] = graph
        flags = pipeline_runtime_flags(compiled)
        self.assertTrue(flags["valid"])
        self.assertFalse(flags["vehicle_detect"])
        self.assertTrue(flags["face_id"])

    def test_complaint_monitoring_defaults_and_override(self) -> None:
        self.assertFalse(complaint_monitoring_wanted({"workplace_type": "garage"}))
        self.assertTrue(complaint_monitoring_wanted({"workplace_type": "massage"}))
        self.assertTrue(
            complaint_monitoring_wanted(
                {"workplace_type": "garage", "complaint_monitoring": {"enabled": True}}
            )
        )
        self.assertFalse(
            complaint_monitoring_wanted(
                {"workplace_type": "massage", "complaint_monitoring": {"enabled": False}}
            )
        )

    def test_sidecar_build_requires_graph_and_workplace_modules(self) -> None:
        from build_sidecar import REQUIRED_PYZ_MODULES

        for name in ("graph", "graph.compile", "workplaces", "workplaces.customer_visits", "workplaces.staff_memory", "complaint_service"):
            self.assertIn(name, REQUIRED_PYZ_MODULES)

    def test_pipeline_graph_asset_is_bundled(self) -> None:
        from pathlib import Path

        from paths import get_resource_path

        asset = get_resource_path(str(Path("static") / "pipeline-graph.js"))
        self.assertTrue(asset.is_file(), asset)


if __name__ == "__main__":
    unittest.main()
