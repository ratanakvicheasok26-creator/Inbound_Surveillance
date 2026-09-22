import { createContext, createElement, useContext, useEffect, useMemo, useReducer, type ReactNode } from "react";
import { padSeq, portraitDataUri, uid } from "./ids";
import { holdReason, lastAlertByCamera } from "./rules";
import { seedState } from "./seed";
import { persistableCameras } from "./account";
import type {
  Alert,
  Camera,
  CaseRecord,
  Detection,
  ObjectClass,
  OperatorAction,
  OpsState,
  RuleConfig,
  TelegramMessage,
} from "./types";

const STORAGE_KEY = "inbound-ops-v1";

type Action =
  | { type: "SET_RULES"; rules: RuleConfig }
  | { type: "ADD_CASE"; record: CaseRecord }
  | { type: "RESOLVE_CASE"; id: string }
  | { type: "SET_SCANNING"; value: boolean }
  | { type: "DEMO_MATCH"; caseId: string }
  | { type: "OPERATOR"; alertId: string; action: OperatorAction }
  | { type: "DISMISS_ALERT"; alertId: string }
  | { type: "ADD_MESSAGES"; messages: TelegramMessage[] }
  | { type: "TOAST"; text: string | null }
  | { type: "HYDRATE_ACCOUNT"; venue: string; cameras: Camera[] };

function loadState(): OpsState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return structuredClone(seedState);
    const parsed = JSON.parse(raw) as OpsState;
    if (!parsed.cameras?.length || !parsed.rules) return structuredClone(seedState);
    return { ...parsed, scanning: false, toast: null };
  } catch {
    return structuredClone(seedState);
  }
}

function promoteEligible(state: OpsState): OpsState {
  const last = lastAlertByCamera(state.alerts);
  const existing = new Set(state.alerts.map((alert) => alert.detectionId));
  let { alertSeq } = state;
  const newAlerts: Alert[] = [];
  const newMessages: TelegramMessage[] = [];

  for (const detection of state.detections) {
    if (existing.has(detection.id)) continue;
    const hold = holdReason(detection, state.rules, last.get(detection.cameraId));
    if (hold) continue;
    alertSeq += 1;
    const alert = packageAlert(state, detection, alertSeq);
    newAlerts.push(alert);
    newMessages.push(outboundAlertMessage(state, alert, detection));
    last.set(detection.cameraId, alert.ts);
    existing.add(detection.id);
  }

  if (!newAlerts.length) return state;

  const caseUpdates = new Map(newAlerts.filter((a) => a.caseId).map((a) => [a.caseId!, "match_pending" as const]));
  return {
    ...state,
    alertSeq,
    alerts: [...newAlerts, ...state.alerts],
    messages: [...state.messages, ...newMessages],
    cases: state.cases.map((record) =>
      caseUpdates.has(record.id) ? { ...record, status: "match_pending" } : record,
    ),
  };
}

function packageAlert(state: OpsState, detection: Detection, seq: number): Alert {
  const record = state.cases.find((item) => item.id === detection.caseId);
  const reason = record
    ? record.kind === "missing"
      ? "Possible match — Safety Wedge. Human verification required."
      : "Watchlist match — Loss Prevention. Human verification required."
    : detection.cameraId === "cam-24"
      ? "Restricted backroom — human in ROI."
      : "Rule engine: important detection.";

  return {
    id: `ALT-${seq}`,
    detectionId: detection.id,
    cameraId: detection.cameraId,
    caseId: detection.caseId,
    reason,
    ts: Date.now(),
    mediaKind: detection.cameraId === "cam-24" ? "gif" : "still",
    confidence: detection.confidence,
    telegramState: "sent",
    operatorAction: null,
    dismissed: false,
  };
}

function outboundAlertMessage(state: OpsState, alert: Alert, detection: Detection): TelegramMessage {
  const camera = state.cameras.find((item) => item.id === alert.cameraId);
  const record = state.cases.find((item) => item.id === alert.caseId);
  const who =
    record?.kind === "missing"
      ? record.missing?.name
      : record?.kind === "watchlist"
        ? record.watchlist?.alias
        : null;
  const body = [
    `ALERT ${camera?.name ?? alert.cameraId} · ${camera?.zone ?? ""}`.trim(),
    alert.reason,
    who ? `${who} · confidence ${alert.confidence.toFixed(2)}` : `confidence ${alert.confidence.toFixed(2)}`,
    "HTTPS POST → Telegram Bot API sendMessage",
  ].join("\n");

  return {
    id: uid("msg"),
    direction: "out",
    method: "sendMessage",
    body,
    ts: alert.ts,
    alertId: alert.id,
    ticketId: record?.ticketId,
  };
}

function demoDetection(state: OpsState, record: CaseRecord, seq: number): Detection {
  const missingCam = "cam-12";
  const watchCam = "cam-07";
  return {
    id: `DET-${padSeq(seq)}`,
    cameraId: record.kind === "missing" ? missingCam : watchCam,
    objectClass: "human",
    confidence: record.kind === "missing" ? 0.87 : 0.83,
    bbox: record.kind === "missing" ? { x: 36, y: 20, w: 30, h: 60 } : { x: 42, y: 24, w: 26, h: 56 },
    ts: Date.now(),
    roiHit: true,
    afterHours: false,
    caseId: record.id,
  };
}

function reducer(state: OpsState, action: Action): OpsState {
  switch (action.type) {
    case "SET_RULES":
      return promoteEligible({ ...state, rules: action.rules });
    case "ADD_CASE":
      return { ...state, cases: [action.record, ...state.cases], ticketSeq: state.ticketSeq + 1 };
    case "RESOLVE_CASE": {
      const record = state.cases.find((item) => item.id === action.id);
      if (!record) return state;
      const nextStatus = record.kind === "missing" && record.missing?.ephemeral ? "expired" : "resolved";
      return {
        ...state,
        cases: state.cases.map((item) =>
          item.id === action.id
            ? { ...item, status: nextStatus, expiredAt: Date.now() }
            : item,
        ),
        toast:
          nextStatus === "expired"
            ? `${record.ticketId} resolved. Ephemeral search data expired.`
            : `${record.ticketId} resolved.`,
      };
    }
    case "SET_SCANNING":
      return { ...state, scanning: action.value };
    case "DEMO_MATCH": {
      const record = state.cases.find((item) => item.id === action.caseId);
      if (!record) return { ...state, scanning: false };
      const detectionSeq = state.detectionSeq + 1;
      const detection = demoDetection(state, record, detectionSeq);
      const withDetection: OpsState = {
        ...state,
        scanning: false,
        detectionSeq,
        detections: [detection, ...state.detections],
      };
      const last = lastAlertByCamera(withDetection.alerts);
      const hold = holdReason(detection, withDetection.rules, last.get(detection.cameraId));
      if (hold) {
        return {
          ...withDetection,
          toast: `Detection ${detection.id} held by rules (${hold}).`,
        };
      }
      const promoted = promoteEligible(withDetection);
      return {
        ...promoted,
        toast: `${detection.id} packaged · HTTPS POST sendMessage`,
      };
    }
    case "OPERATOR": {
      const alert = state.alerts.find((item) => item.id === action.alertId);
      if (!alert || !action.action) return state;
      const camera = state.cameras.find((item) => item.id === alert.cameraId);
      const verb =
        action.action === "verify"
          ? "Verified. Human confirmation recorded."
          : action.action === "intercom"
            ? `Intercom queued for ${camera?.name ?? alert.cameraId} (demo — not connected).`
            : `Remote siren queued for ${camera?.name ?? alert.cameraId} (demo — not connected).`;
      const followUp: TelegramMessage = {
        id: uid("msg"),
        direction: "out",
        method: "sendMessage",
        body: `Operator ${action.action} · ${alert.id} · ${camera?.name ?? alert.cameraId}`,
        ts: Date.now(),
        alertId: alert.id,
      };
      return {
        ...state,
        alerts: state.alerts.map((item) =>
          item.id === action.alertId ? { ...item, operatorAction: action.action } : item,
        ),
        messages: [...state.messages, followUp],
        toast: verb,
      };
    }
    case "DISMISS_ALERT":
      return {
        ...state,
        alerts: state.alerts.map((item) =>
          item.id === action.alertId ? { ...item, dismissed: true, operatorAction: null } : item,
        ),
        toast: `${action.alertId} dismissed. No dispatch.`,
      };
    case "ADD_MESSAGES":
      return { ...state, messages: [...state.messages, ...action.messages] };
    case "TOAST":
      return { ...state, toast: action.text };
    case "HYDRATE_ACCOUNT": {
      const venue = action.venue || state.venue;
      const same =
        venue === state.venue &&
        action.cameras.length === state.cameras.length &&
        action.cameras.every(
          (cam, index) =>
            cam.id === state.cameras[index]?.id &&
            cam.name === state.cameras[index]?.name &&
            cam.zone === state.cameras[index]?.zone &&
            cam.rtspLabel === state.cameras[index]?.rtspLabel &&
            cam.protocol === state.cameras[index]?.protocol,
        );
      if (same) return state;
      return { ...state, venue, cameras: action.cameras };
    }
    default:
      return state;
  }
}

type StoreApi = {
  state: OpsState;
  hydrateAccount: (venue: string, cameras: Camera[]) => void;
  setRules: (rules: RuleConfig) => void;
  submitMissing: (fields: NonNullable<CaseRecord["missing"]>, photo: string, source: CaseRecord["source"]) => string;
  submitWatchlist: (fields: NonNullable<CaseRecord["watchlist"]>, photo: string) => string;
  resolveCase: (id: string) => void;
  operator: (alertId: string, action: Exclude<OperatorAction, null>) => void;
  dismissAlert: (alertId: string) => void;
  telegramSubmitTicket: () => void;
  telegramCheckStatus: () => void;
  clearToast: () => void;
};

const StoreContext = createContext<StoreApi | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, loadState);

  useEffect(() => {
    const persist: OpsState = {
      ...state,
      toast: null,
      scanning: state.scanning,
      cameras: persistableCameras(state.cameras),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(persist));
  }, [state]);

  useEffect(() => {
    if (!state.toast) return;
    const timer = window.setTimeout(() => dispatch({ type: "TOAST", text: null }), 4200);
    return () => window.clearTimeout(timer);
  }, [state.toast]);

  const api = useMemo<StoreApi>(() => {
    const queueMatch = (caseId: string) => {
      dispatch({ type: "SET_SCANNING", value: true });
      window.setTimeout(() => dispatch({ type: "DEMO_MATCH", caseId }), 2000);
    };

    return {
      state,
      hydrateAccount: (venue, cameras) => dispatch({ type: "HYDRATE_ACCOUNT", venue, cameras }),
      setRules: (rules) => dispatch({ type: "SET_RULES", rules }),
      submitMissing: (fields, photo, source) => {
        const ticketId = `TKT-${state.ticketSeq + 1}`;
        const id = uid("case");
        const record: CaseRecord = {
          id,
          ticketId,
          kind: "missing",
          status: "searching",
          photo,
          createdAt: Date.now(),
          source,
          missing: fields,
        };
        dispatch({ type: "ADD_CASE", record });
        if (source === "web") {
          dispatch({
            type: "ADD_MESSAGES",
            messages: [
              {
                id: uid("msg"),
                direction: "in",
                method: "ticket",
                body: `Info desk submitted ${fields.name}, age ${fields.age}, ${fields.clothing}. Last seen ${fields.lastSeenZone}.`,
                ts: Date.now(),
                ticketId,
              },
              {
                id: uid("msg"),
                direction: "out",
                method: "ticket",
                body: "Ticket opened. Ephemeral search started on connected cameras.",
                ts: Date.now(),
                ticketId,
                json: JSON.stringify(
                  { status: "success", data: { ticket_id: ticketId, status: "In Progress" } },
                  null,
                  2,
                ),
              },
            ],
          });
        }
        queueMatch(id);
        return ticketId;
      },
      submitWatchlist: (fields, photo) => {
        const ticketId = `TKT-${state.ticketSeq + 1}`;
        const id = uid("case");
        const record: CaseRecord = {
          id,
          ticketId,
          kind: "watchlist",
          status: "searching",
          photo,
          createdAt: Date.now(),
          source: "web",
          watchlist: fields,
        };
        dispatch({ type: "ADD_CASE", record });
        queueMatch(id);
        return ticketId;
      },
      resolveCase: (id) => dispatch({ type: "RESOLVE_CASE", id }),
      operator: (alertId, action) => dispatch({ type: "OPERATOR", alertId, action }),
      dismissAlert: (alertId) => dispatch({ type: "DISMISS_ALERT", alertId }),
      telegramSubmitTicket: () => {
        const ticketId = `TKT-${state.ticketSeq + 1}`;
        const id = uid("case");
        const fields = {
          name: "Vicheka Lim",
          age: "5",
          clothing: "Green polo, white sneakers",
          lastSeenZone: "Food court",
          reporterName: "Parent via Telegram",
          reporterPhone: "+855 10 882 441",
          ephemeral: true,
        };
        const record: CaseRecord = {
          id,
          ticketId,
          kind: "missing",
          status: "searching",
          photo: portraitDataUri(fields.name),
          createdAt: Date.now(),
          source: "telegram",
          missing: fields,
        };
        dispatch({
          type: "ADD_MESSAGES",
          messages: [
            {
              id: uid("msg"),
              direction: "in",
              method: "ticket",
              body: "Submit Ticket: Vicheka Lim, age 5, green polo, last seen food court.",
              ts: Date.now(),
              ticketId,
            },
          ],
        });
        dispatch({ type: "ADD_CASE", record });
        dispatch({
          type: "ADD_MESSAGES",
          messages: [
            {
              id: uid("msg"),
              direction: "out",
              method: "ticket",
              body: "Ticket opened. Searching connected cameras. Potential matches require human verification.",
              ts: Date.now(),
              ticketId,
              json: JSON.stringify(
                { status: "success", data: { ticket_id: ticketId, status: "In Progress" } },
                null,
                2,
              ),
            },
          ],
        });
        queueMatch(id);
      },
      telegramCheckStatus: () => {
        const latest = state.cases[0];
        const payload = latest
          ? { status: "success", data: { ticket_id: latest.ticketId, status: latest.status } }
          : { status: "success", data: { ticket_id: null, status: "No tickets" } };
        dispatch({
          type: "ADD_MESSAGES",
          messages: [
            {
              id: uid("msg"),
              direction: "in",
              method: "status",
              body: "Check Status",
              ts: Date.now(),
              ticketId: latest?.ticketId,
            },
            {
              id: uid("msg"),
              direction: "out",
              method: "status",
              body: latest
                ? `${latest.ticketId} is ${latest.status.replace("_", " ")}.`
                : "No tickets on file.",
              ts: Date.now(),
              ticketId: latest?.ticketId,
              json: JSON.stringify(payload, null, 2),
            },
          ],
        });
      },
      clearToast: () => dispatch({ type: "TOAST", text: null }),
    };
  }, [state]);

  return createElement(StoreContext.Provider, { value: api }, children);
}

export function useOps(): StoreApi {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error("useOps must be used inside StoreProvider");
  return ctx;
}

export function toggleObjectType(rules: RuleConfig, value: ObjectClass): RuleConfig {
  const has = rules.objectTypes.includes(value);
  const objectTypes = has
    ? rules.objectTypes.filter((item) => item !== value)
    : [...rules.objectTypes, value];
  return { ...rules, objectTypes: objectTypes.length ? objectTypes : rules.objectTypes };
}
