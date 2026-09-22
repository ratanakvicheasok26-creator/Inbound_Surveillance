import type { WorkplaceId } from "../workplaces";
import type { PipelineNodeData, PipelineNodeGroup, PipelineNodeKind } from "./ir";

export type NodeCatalogEntry = {
  kind: PipelineNodeKind;
  label: string;
  group: PipelineNodeGroup;
  summary: string;
};

export const NODE_CATALOG: Record<PipelineNodeKind, NodeCatalogEntry> = {
  camera: { kind: "camera", label: "Camera", group: "source", summary: "Frame ingest" },
  personDetect: { kind: "personDetect", label: "Person detect", group: "vision", summary: "YOLO / RTMPose people" },
  pose: { kind: "pose", label: "Pose", group: "vision", summary: "Skeleton keypoints" },
  faceId: { kind: "faceId", label: "Face ID", group: "vision", summary: "Staff enrollment only" },
  anonymousReid: { kind: "anonymousReid", label: "Anonymous re-ID", group: "vision", summary: "Body crop → OSNet (HSV fallback) → local gallery. New person gets a visitor_id. No names." },
  vehicleDetect: { kind: "vehicleDetect", label: "Vehicle detect", group: "vision", summary: "Car / van YOLO" },
  zone: { kind: "zone", label: "Zone", group: "space", summary: "Named ROI" },
  employeeLabor: { kind: "employeeLabor", label: "Employee labor", group: "monitor", summary: "Bay wrench-time" },
  customerVisits: { kind: "customerVisits", label: "Customer visits", group: "monitor", summary: "SQLite customer_visits + anonymous_subjects; today/week counts. Staff excluded." },
  alertRule: { kind: "alertRule", label: "Alert rules", group: "output", summary: "Cooldown + promote" },
  telegram: { kind: "telegram", label: "Telegram", group: "output", summary: "Photo dispatch" },
  persist: { kind: "persist", label: "Persist", group: "output", summary: "SQLite / sync" },
  complaintIntake: { kind: "complaintIntake", label: "Complaint intake", group: "output", summary: "Structure stub" },
};

export const GROUP_ORDER: PipelineNodeGroup[] = ["source", "vision", "space", "monitor", "output"];

export const GROUP_LABELS: Record<PipelineNodeGroup, string> = {
  source: "Sources",
  vision: "Vision",
  space: "Space",
  monitor: "Monitors",
  output: "Outputs",
};

export const GROUP_ACCENT: Record<PipelineNodeGroup, string> = {
  source: "#00ff66",
  vision: "#22d3ee",
  space: "#fbbf24",
  monitor: "#fb923c",
  output: "#f87171",
};

export function defaultNodeData(kind: PipelineNodeKind, workplace: WorkplaceId = "garage"): PipelineNodeData {
  const meta = NODE_CATALOG[kind];
  const data: PipelineNodeData = { kind, label: meta.label };
  if (kind === "zone") {
    data.zoneKind = workplace === "massage" ? "treatment_room" : "vehicle_bay";
    data.zoneName = workplace === "massage" ? "Treatment Room" : "Lift Bay";
    data.roi = [0.2, 0.2, 0.3, 0.4];
  }
  if (kind === "camera") {
    data.cameraName = "Camera";
    data.protocol = "webcam";
  }
  if (kind === "alertRule") {
    data.cooldownSec = 30;
  }
  return data;
}
