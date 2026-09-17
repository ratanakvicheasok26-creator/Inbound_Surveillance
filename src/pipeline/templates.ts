import type { PipelineGraph, PipelineNodeData, PipelineNodeKind } from "./ir";

function node(
  id: string,
  kind: PipelineNodeKind,
  label: string,
  x: number,
  y: number,
  extra: Partial<PipelineNodeData> = {},
) {
  return {
    id,
    type: "pipeline" as const,
    position: { x, y },
    data: { kind, label, ...extra },
  };
}

function edge(source: string, target: string, sourceHandle: string, targetHandle: string) {
  return {
    id: `${source}-${target}-${sourceHandle}`,
    source,
    target,
    sourceHandle,
    targetHandle,
  };
}

export const GARAGE_TEMPLATE: PipelineGraph = {
  workplace_type: "garage",
  nodes: [
    node("cam", "camera", "Camera", 40, 180, { cameraName: "Lift Bay 1", protocol: "webcam" }),
    node("person", "personDetect", "Person detect", 280, 80),
    node("pose", "pose", "Pose", 520, 80),
    node("face", "faceId", "Face ID (staff)", 760, 40),
    node("vehicle", "vehicleDetect", "Vehicle detect", 280, 280),
    node("zone1", "zone", "Lift Bay 1", 520, 280, {
      zoneKind: "vehicle_bay",
      zoneName: "Lift Bay 1",
      roi: [0.1, 0.2, 0.35, 0.6],
    }),
    node("labor", "employeeLabor", "Employee labor", 1000, 160),
    node("persist", "persist", "Persist", 1240, 80),
    node("alert", "alertRule", "Alert rules", 1240, 200, { cooldownSec: 30 }),
    node("tg", "telegram", "Telegram", 1480, 200),
  ],
  edges: [
    edge("cam", "person", "frames", "frames"),
    edge("cam", "vehicle", "frames", "frames"),
    edge("cam", "zone1", "frames", "frames"),
    edge("person", "pose", "detections", "detections"),
    edge("pose", "face", "detections", "detections"),
    edge("face", "labor", "identity", "identity"),
    edge("zone1", "labor", "zones", "zones"),
    edge("labor", "persist", "events", "events"),
    edge("labor", "alert", "events", "events"),
    edge("alert", "tg", "alerts", "alerts"),
  ],
};

export const MASSAGE_TEMPLATE: PipelineGraph = {
  workplace_type: "massage",
  nodes: [
    node("cam", "camera", "Camera", 40, 160, { cameraName: "Front desk", protocol: "webcam" }),
    node("person", "personDetect", "Person detect", 280, 80),
    node("reid", "anonymousReid", "Anonymous re-ID", 520, 80),
    node("ent", "zone", "Entrance", 280, 280, { zoneKind: "entrance", zoneName: "Entrance", roi: [0.05, 0.15, 0.25, 0.7] }),
    node("wait", "zone", "Waiting", 520, 280, { zoneKind: "waiting", zoneName: "Waiting", roi: [0.35, 0.2, 0.28, 0.55] }),
    node("room", "zone", "Treatment Room 1", 760, 280, {
      zoneKind: "treatment_room",
      zoneName: "Treatment Room 1",
      roi: [0.68, 0.18, 0.28, 0.62],
    }),
    node("visits", "customerVisits", "Customer visits", 1000, 140),
    node("persist", "persist", "Persist", 1240, 60),
    node("complaint", "complaintIntake", "Complaint intake", 1240, 220),
  ],
  edges: [
    edge("cam", "person", "frames", "frames"),
    edge("cam", "ent", "frames", "frames"),
    edge("cam", "wait", "frames", "frames"),
    edge("cam", "room", "frames", "frames"),
    edge("person", "reid", "detections", "detections"),
    edge("reid", "visits", "identity", "identity"),
    edge("ent", "visits", "zones", "zones"),
    edge("wait", "visits", "zones", "zones"),
    edge("room", "visits", "zones", "zones"),
    edge("visits", "persist", "events", "events"),
    edge("visits", "complaint", "events", "events"),
  ],
};

export function templateFor(workplace: string): PipelineGraph {
  return workplace === "massage" ? structuredClone(MASSAGE_TEMPLATE) : structuredClone(GARAGE_TEMPLATE);
}
