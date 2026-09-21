"""Compile a pipeline graph IR into engine config flags.

Invalid graphs must not start inference. This module is the only place the
node editor is allowed to change runtime behavior.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from workplaces import DEFAULT_RECEPTION_ROI, parse_workplace_id, parse_zone_kind

NODE_PORTS: dict[str, dict[str, tuple[str, ...]]] = {
    "camera": {"inputs": (), "outputs": ("frames",)},
    "personDetect": {"inputs": ("frames",), "outputs": ("detections",)},
    "pose": {"inputs": ("detections",), "outputs": ("detections",)},
    "faceId": {"inputs": ("detections",), "outputs": ("identity",)},
    "anonymousReid": {"inputs": ("detections",), "outputs": ("identity",)},
    "vehicleDetect": {"inputs": ("frames",), "outputs": ("detections",)},
    "zone": {"inputs": ("frames",), "outputs": ("zones",)},
    "employeeLabor": {"inputs": ("identity", "zones"), "outputs": ("events",)},
    "customerVisits": {"inputs": ("identity", "zones"), "outputs": ("events",)},
    "alertRule": {"inputs": ("events",), "outputs": ("alerts",)},
    "telegram": {"inputs": ("alerts",), "outputs": ()},
    "persist": {"inputs": ("events",), "outputs": ()},
    "complaintIntake": {"inputs": ("events",), "outputs": ()},
}

MONITOR_KINDS = {"employeeLabor", "customerVisits"}
DEFAULT_ZONE_ROI = [0.2, 0.2, 0.3, 0.4]


def _kind(node: dict[str, Any]) -> str:
    data = node.get("data") if isinstance(node.get("data"), dict) else {}
    return str(data.get("kind") or node.get("kind") or "")


def _data(node: dict[str, Any]) -> dict[str, Any]:
    return node.get("data") if isinstance(node.get("data"), dict) else {}


def validate_graph(graph: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["Graph is missing."]
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    by_id = {str(n.get("id")): n for n in nodes if isinstance(n, dict) and n.get("id")}
    if not any(_kind(n) == "camera" for n in by_id.values()):
        errors.append("Graph needs a Camera source.")

    incoming: dict[str, int] = {nid: 0 for nid in by_id}
    incoming_ports: dict[str, set[str]] = {nid: set() for nid in by_id}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src, dst = str(edge.get("source") or ""), str(edge.get("target") or "")
        if src not in by_id or dst not in by_id:
            errors.append(f"Edge {edge.get('id')} points at a missing node.")
            continue
        outgoing[src].append(dst)
        incoming[dst] = incoming.get(dst, 0) + 1
        src_kind = _kind(by_id[src])
        dst_kind = _kind(by_id[dst])
        src_port = str(edge.get("sourceHandle") or (NODE_PORTS.get(src_kind, {}).get("outputs") or ("",))[0])
        dst_port = str(edge.get("targetHandle") or (NODE_PORTS.get(dst_kind, {}).get("inputs") or ("",))[0])
        incoming_ports.setdefault(dst, set()).add(dst_port)
        if src_port and dst_port and src_port != dst_port:
            errors.append(f"{src_kind} {src_port} cannot connect to {dst_kind} {dst_port}.")

    for nid, node in by_id.items():
        kind = _kind(node)
        if kind not in MONITOR_KINDS:
            continue
        ports = incoming_ports.get(nid) or set()
        label = str(_data(node).get("label") or kind)
        if "identity" not in ports:
            errors.append(f"{label} needs an identity input (Face ID or anonymous re-ID).")
        if "zones" not in ports:
            errors.append(f"{label} needs a zone input.")

    queue = deque([nid for nid, deg in incoming.items() if deg == 0])
    seen = 0
    indeg = dict(incoming)
    while queue:
        nid = queue.popleft()
        seen += 1
        for nxt in outgoing.get(nid, []):
            indeg[nxt] = indeg.get(nxt, 1) - 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if by_id and seen != len(by_id):
        errors.append("Graph has a cycle. Disconnect a loop before deploying.")
    return errors


def compile_to_config(graph: dict[str, Any]) -> dict[str, Any]:
    errors = validate_graph(graph)
    if errors:
        raise ValueError("; ".join(errors))
    nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict)]
    kinds = {_kind(n) for n in nodes}
    workplace = str(graph.get("workplace_type") or "")
    if "customerVisits" in kinds:
        workplace = "massage"
    elif "employeeLabor" in kinds:
        workplace = "garage"
    elif workplace not in ("garage", "massage"):
        workplace = "garage"

    zones: list[dict[str, Any]] = []
    cooldown = None
    for node in nodes:
        kind = _kind(node)
        data = _data(node)
        if kind == "zone":
            roi = data.get("roi")
            zone_kind = parse_zone_kind(data.get("zoneKind"), workplace)
            placeholder = list(DEFAULT_RECEPTION_ROI) if zone_kind == "reception" else list(DEFAULT_ZONE_ROI)
            zones.append(
                {
                    "id": str(node.get("id")),
                    "name": str(data.get("zoneName") or data.get("label") or node.get("id")),
                    "type": zone_kind,
                    "roi": list(roi) if isinstance(roi, list) and len(roi) == 4 else placeholder,
                }
            )
        if kind == "alertRule" and data.get("cooldownSec") is not None:
            try:
                cooldown = max(1, int(data.get("cooldownSec")))
            except (TypeError, ValueError):
                cooldown = None

    patch: dict[str, Any] = {
        "workplace_type": workplace,
        "person_detect": "personDetect" in kinds,
        "pose_enabled": "pose" in kinds,
        "vehicle_detect": "vehicleDetect" in kinds,
        "face_id_enabled": "faceId" in kinds,
        "enable_face_id": "faceId" in kinds,
        "reid_enabled": "anonymousReid" in kinds or "faceId" in kinds,
        "enable_reid": "anonymousReid" in kinds or "faceId" in kinds,
        "employee_labor": "employeeLabor" in kinds,
        "customer_visits": "customerVisits" in kinds,
        "alerts_enabled": "alertRule" in kinds,
        "telegram_dispatch": "telegram" in kinds,
        "persist_events": "persist" in kinds,
        "complaint_intake": "complaintIntake" in kinds,
        "pipeline_valid": True,
    }
    if cooldown is not None:
        patch["cooldown_seconds"] = cooldown
    if zones:
        patch["bays"] = zones
    return patch


def default_runtime_flags(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or {}
    workplace = parse_workplace_id(cfg.get("workplace_type"))
    vehicle = cfg.get("vehicle_detect")
    if vehicle is None:
        vehicle = workplace != "massage"
    face = cfg.get("enable_face_id")
    if face is None:
        face = workplace != "massage"
    return {
        "valid": True,
        "person_detect": True,
        "pose_enabled": True,
        "vehicle_detect": bool(vehicle),
        "face_id": bool(face),
        "reid": bool(cfg.get("enable_reid", True)),
        "employee_labor": workplace == "garage",
        "customer_visits": workplace == "massage",
        "alerts": True,
        "telegram": True,
        "persist": True,
        "complaint_intake": workplace == "massage",
    }


def pipeline_runtime_flags(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flags the inference loop reads. No graph → current garage/massage defaults."""
    cfg = cfg or {}
    defaults = default_runtime_flags(cfg)
    graph = cfg.get("pipeline_graph")
    has_graph = isinstance(graph, dict) and bool(graph.get("nodes"))
    if not has_graph:
        return defaults
    if cfg.get("pipeline_valid") is False:
        return {**defaults, "valid": False}
    errors = validate_graph(graph)
    if errors:
        return {**defaults, "valid": False, "errors": errors}
    return {
        "valid": True,
        "person_detect": bool(cfg.get("person_detect", True)),
        "pose_enabled": bool(cfg.get("pose_enabled", True)),
        "vehicle_detect": bool(cfg.get("vehicle_detect", defaults["vehicle_detect"])),
        "face_id": bool(cfg.get("face_id_enabled", cfg.get("enable_face_id", defaults["face_id"]))),
        "reid": bool(cfg.get("reid_enabled", cfg.get("enable_reid", True))),
        "employee_labor": bool(cfg.get("employee_labor", defaults["employee_labor"])),
        "customer_visits": bool(cfg.get("customer_visits", defaults["customer_visits"])),
        "alerts": bool(cfg.get("alerts_enabled", True)),
        "telegram": bool(cfg.get("telegram_dispatch", True)),
        "persist": bool(cfg.get("persist_events", True)),
        "complaint_intake": bool(cfg.get("complaint_intake", defaults["complaint_intake"])),
    }


def complaint_monitoring_wanted(cfg: dict[str, Any] | None = None) -> bool:
    """Whether the audio complaint pipeline should run.

    Explicit ``complaint_monitoring.enabled`` wins. Otherwise massage
    workplaces and graphs that include a Complaint intake node turn it on.
    """
    cfg = cfg or {}
    cmp_cfg = cfg.get("complaint_monitoring") if isinstance(cfg.get("complaint_monitoring"), dict) else {}
    if "enabled" in cmp_cfg:
        return bool(cmp_cfg.get("enabled"))
    return bool(pipeline_runtime_flags(cfg).get("complaint_intake"))
