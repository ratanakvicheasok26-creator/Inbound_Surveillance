import type { Camera, CaseRecord } from "./types";

export function cameraById(cameras: Camera[], id: string): Camera | undefined {
  return cameras.find((camera) => camera.id === id);
}

export function caseLabel(record: CaseRecord): string {
  if (record.kind === "missing") return record.missing?.name ?? record.ticketId;
  return record.watchlist?.alias ?? record.ticketId;
}

export function formatClock(ts: number): string {
  return new Date(ts).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatRelative(ts: number, now = Date.now()): string {
  const delta = Math.max(0, now - ts);
  const sec = Math.round(delta / 1000);
  if (sec < 60) return `${sec}s ago`;
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  return `${hr}h ago`;
}

export function confidenceLabel(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function statusLabel(status: CaseRecord["status"]): string {
  switch (status) {
    case "searching":
      return "Searching";
    case "match_pending":
      return "Match pending";
    case "resolved":
      return "Resolved";
    case "expired":
      return "Expired";
    case "in_progress":
      return "In progress";
    default:
      return status;
  }
}
