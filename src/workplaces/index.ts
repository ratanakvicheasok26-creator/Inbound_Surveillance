export type WorkplaceId = "garage" | "massage";

export type SubjectKind = "employee" | "customer";

export type MonitorId = "employee_labor" | "customer_visits";

export type ZoneKind =
  | "vehicle_bay"
  | "tool_area"
  | "entrance"
  | "waiting"
  | "treatment_room";

export type ConsoleViewId =
  | "live"
  | "rules"
  | "cases"
  | "alerts"
  | "visits"
  | "complaints"
  | "pipeline"
  | "scan-and-go"
  | "bot";

export type ZoneKindOption = {
  id: ZoneKind;
  label: string;
};

export type WorkplaceTab = {
  id: ConsoleViewId;
  label: string;
};

export type WorkplaceProfile = {
  id: WorkplaceId;
  label: string;
  subjectKind: SubjectKind;
  monitorId: MonitorId;
  unknownPersonLabel: string;
  zoneNoun: string;
  zoneNounPlural: string;
  zoneKinds: ZoneKindOption[];
  defaultZoneKind: ZoneKind;
  includeCrewSetup: boolean;
  roiNameLabel: string;
  roiPlaceholder: string;
  cameraPlaceholder: string;
  emailPlaceholder: string;
  pipelineIdle: string;
  tabs: WorkplaceTab[];
};

const SHARED_TABS: WorkplaceTab[] = [
  { id: "live", label: "Live" },
  { id: "rules", label: "Rules" },
  { id: "cases", label: "Cases" },
  { id: "alerts", label: "Alerts" },
];

const TAIL_TABS: WorkplaceTab[] = [
  { id: "complaints", label: "Complaints" },
  { id: "pipeline", label: "Pipeline" },
  { id: "scan-and-go", label: "Scan & Go" },
  { id: "bot", label: "Telegram" },
];

export const WORKPLACE_IDS: WorkplaceId[] = ["garage", "massage"];

export const GARAGE_ZONE_KINDS: ZoneKindOption[] = [
  { id: "vehicle_bay", label: "Vehicle bay" },
  { id: "tool_area", label: "Tool area" },
];

export const MASSAGE_ZONE_KINDS: ZoneKindOption[] = [
  { id: "entrance", label: "Entrance" },
  { id: "waiting", label: "Waiting" },
  { id: "treatment_room", label: "Treatment room" },
];

export const ALL_ZONE_KINDS: ZoneKind[] = [
  ...GARAGE_ZONE_KINDS.map((item) => item.id),
  ...MASSAGE_ZONE_KINDS.map((item) => item.id),
];

export const WORKPLACES: Record<WorkplaceId, WorkplaceProfile> = {
  garage: {
    id: "garage",
    label: "Garage",
    subjectKind: "employee",
    monitorId: "employee_labor",
    unknownPersonLabel: "Employee",
    zoneNoun: "bay",
    zoneNounPlural: "bays",
    zoneKinds: GARAGE_ZONE_KINDS,
    defaultZoneKind: "vehicle_bay",
    includeCrewSetup: true,
    roiNameLabel: "Bay name",
    roiPlaceholder: "Lift Bay 1",
    cameraPlaceholder: "Lift Bay 1",
    emailPlaceholder: "you@garage.com",
    pipelineIdle: "YOLO detections",
    tabs: [...SHARED_TABS, ...TAIL_TABS],
  },
  massage: {
    id: "massage",
    label: "Massage",
    subjectKind: "customer",
    monitorId: "customer_visits",
    unknownPersonLabel: "Visitor",
    zoneNoun: "zone",
    zoneNounPlural: "zones",
    zoneKinds: MASSAGE_ZONE_KINDS,
    defaultZoneKind: "treatment_room",
    includeCrewSetup: false,
    roiNameLabel: "Zone name",
    roiPlaceholder: "Treatment Room 1",
    cameraPlaceholder: "Front desk",
    emailPlaceholder: "you@shop.com",
    pipelineIdle: "Anonymous re-ID",
    tabs: [
      ...SHARED_TABS,
      { id: "visits", label: "Visits" },
      ...TAIL_TABS,
    ],
  },
};

export function parseWorkplaceId(value: string | null | undefined): WorkplaceId {
  return value === "massage" ? "massage" : "garage";
}

export function workplaceOf(value: string | null | undefined): WorkplaceProfile {
  return WORKPLACES[parseWorkplaceId(value)];
}

export function isZoneKind(value: string | null | undefined): value is ZoneKind {
  return ALL_ZONE_KINDS.includes(value as ZoneKind);
}

export function parseZoneKind(value: string | null | undefined, workplace: WorkplaceId): ZoneKind {
  if (isZoneKind(value)) return value;
  return WORKPLACES[workplace].defaultZoneKind;
}

export function nextAvailableZoneName(
  workplace: WorkplaceId,
  zones: { name?: string }[],
): string {
  const profile = WORKPLACES[workplace];
  const prefix = workplace === "garage" ? "Bay" : "Room";
  const used = new Set<number>();
  for (const zone of zones) {
    const match = String(zone.name || "").match(new RegExp(`${prefix}\\s*(\\d+)`, "i"));
    if (match) used.add(Number.parseInt(match[1], 10));
  }
  let num = 1;
  while (used.has(num)) num += 1;
  return workplace === "garage" ? `${prefix} ${num}` : `${profile.roiPlaceholder.replace(/\s*\d+$/, "")} ${num}`;
}
