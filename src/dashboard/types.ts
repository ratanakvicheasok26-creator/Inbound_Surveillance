export type ObjectClass = "human" | "vehicle";

export type TimeWindow = "all" | "after-hours";

export type CaseKind = "missing" | "watchlist";

export type CaseStatus =
  | "searching"
  | "match_pending"
  | "resolved"
  | "expired"
  | "in_progress";

export type WatchlistCategory = "shoplifter" | "banned" | "wanted";

export type TelegramState = "queued" | "sent" | "failed";

export type OperatorAction = "verify" | "intercom" | "siren" | null;

export type MediaKind = "still" | "gif";

export type TelegramDirection = "in" | "out";

export type TelegramMethod = "sendMessage" | "getUpdates" | "getMe" | "ticket" | "status";

export type ViewId =
  | "live"
  | "rules"
  | "cases"
  | "alerts"
  | "visits"
  | "complaints"
  | "pipeline"
  | "bot"
  | "scan-and-go";

export type CameraProtocol = "webcam" | "rtsp" | "phone" | "onvif" | "tapo" | "webrtc";

export type BBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type Camera = {
  id: string;
  name: string;
  zone: string;
  rtspLabel: string;
  protocol?: CameraProtocol | string;
  vendor?: string;
  source?: string;
  mainSource?: string;
  username?: string;
};

export type DiscoveredDevice = {
  ip: string;
  port: number;
  service_type: string;
  name: string;
  xaddrs: string[];
  manufacturer: string;
  model: string;
};

export type DiscoveryStatus = "idle" | "scanning" | "done" | "error";

export type DiscoveryResults = {
  status: DiscoveryStatus;
  devices: DiscoveredDevice[];
  error?: string;
  started_at?: string | number;
};

export type EngineBay = {
  bay_id?: string;
  id?: string;
  name?: string;
  type?: string;
  zone_kind?: string;
  roi?: number[];
  state?: string;
};

export type VisitCounts = {
  today_unique: number;
  today_visits: number;
  week_unique: number;
  week_visits: number;
  open_sessions?: Array<{
    subject_id: string;
    zone_id: string;
    zone_name?: string;
    started_at?: string;
  }>;
};

export type EngineTelemetry = {
  protocol?: string;
  resolution?: string;
  fps?: number;
  ingest_fps?: number;
  infer_ms?: number;
  main_stream?: boolean | string | null;
  connection?: string;
  status?: string;
  error?: string;
  width?: number;
  height?: number;
  bays?: EngineBay[];
  zones?: EngineBay[];
  workplace_type?: string;
  visits?: VisitCounts;
};

export type Detection = {
  id: string;
  cameraId: string;
  objectClass: ObjectClass;
  confidence: number;
  bbox: BBox;
  ts: number;
  roiHit: boolean;
  afterHours: boolean;
  caseId?: string;
};

export type RuleConfig = {
  roiEnabled: boolean;
  timeWindow: TimeWindow;
  cooldownSec: number;
  objectTypes: ObjectClass[];
};

export type Alert = {
  id: string;
  detectionId: string;
  cameraId: string;
  caseId?: string;
  reason: string;
  ts: number;
  mediaKind: MediaKind;
  confidence: number;
  telegramState: TelegramState;
  operatorAction: OperatorAction;
  dismissed: boolean;
};

export type MissingFields = {
  name: string;
  age: string;
  clothing: string;
  lastSeenZone: string;
  reporterName: string;
  reporterPhone: string;
  ephemeral: boolean;
};

export type WatchlistFields = {
  alias: string;
  category: WatchlistCategory;
  notes: string;
  shareAcrossVenues: boolean;
};

export type CaseRecord = {
  id: string;
  ticketId: string;
  kind: CaseKind;
  status: CaseStatus;
  photo: string;
  createdAt: number;
  expiredAt?: number;
  source: "web" | "telegram";
  missing?: MissingFields;
  watchlist?: WatchlistFields;
};

export type TelegramMessage = {
  id: string;
  direction: TelegramDirection;
  method: TelegramMethod;
  body: string;
  ts: number;
  ticketId?: string;
  alertId?: string;
  json?: string;
};

export type OpsState = {
  venue: string;
  cameras: Camera[];
  detections: Detection[];
  rules: RuleConfig;
  alerts: Alert[];
  cases: CaseRecord[];
  messages: TelegramMessage[];
  scanning: boolean;
  toast: string | null;
  ticketSeq: number;
  alertSeq: number;
  detectionSeq: number;
};

export type RuleHoldReason =
  | "object-type"
  | "roi"
  | "time-window"
  | "cooldown"
  | null;
