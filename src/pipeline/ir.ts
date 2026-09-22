import type { WorkplaceId } from "../workplaces";

export type PipelineNodeKind =
  | "camera"
  | "personDetect"
  | "pose"
  | "faceId"
  | "anonymousReid"
  | "vehicleDetect"
  | "zone"
  | "employeeLabor"
  | "customerVisits"
  | "alertRule"
  | "telegram"
  | "persist"
  | "complaintIntake";

export type PipelinePort = "frames" | "detections" | "identity" | "zones" | "events" | "alerts";

export type PipelineNodeGroup = "source" | "vision" | "space" | "monitor" | "output";

export type PipelineNodeData = {
  kind: PipelineNodeKind;
  label: string;
  zoneKind?: string;
  zoneName?: string;
  roi?: number[];
  cameraName?: string;
  protocol?: string;
  sourceUrl?: string;
  cooldownSec?: number;
};

export type PipelineNode = {
  id: string;
  type: "pipeline";
  position: { x: number; y: number };
  data: PipelineNodeData;
};

export type PipelineEdge = {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
};

export type PipelineGraph = {
  workplace_type: WorkplaceId;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
};

export const NODE_PORTS: Record<PipelineNodeKind, { inputs: PipelinePort[]; outputs: PipelinePort[] }> = {
  camera: { inputs: [], outputs: ["frames"] },
  personDetect: { inputs: ["frames"], outputs: ["detections"] },
  pose: { inputs: ["detections"], outputs: ["detections"] },
  faceId: { inputs: ["detections"], outputs: ["identity"] },
  anonymousReid: { inputs: ["detections"], outputs: ["identity"] },
  vehicleDetect: { inputs: ["frames"], outputs: ["detections"] },
  zone: { inputs: ["frames"], outputs: ["zones"] },
  employeeLabor: { inputs: ["identity", "zones"], outputs: ["events"] },
  customerVisits: { inputs: ["identity", "zones"], outputs: ["events"] },
  alertRule: { inputs: ["events"], outputs: ["alerts"] },
  telegram: { inputs: ["alerts"], outputs: [] },
  persist: { inputs: ["events"], outputs: [] },
  complaintIntake: { inputs: ["events"], outputs: [] },
};

export const PORT_COLORS: Record<PipelinePort, string> = {
  frames: "#e879f9",
  detections: "#22d3ee",
  identity: "#fbbf24",
  zones: "#4ade80",
  events: "#fb923c",
  alerts: "#f87171",
};

export function isPipelinePort(value: string | null | undefined): value is PipelinePort {
  return value === "frames" || value === "detections" || value === "identity" || value === "zones" || value === "events" || value === "alerts";
}

export function portsCompatible(sourcePort?: string | null, targetPort?: string | null): boolean {
  return Boolean(sourcePort && targetPort && sourcePort === targetPort);
}
