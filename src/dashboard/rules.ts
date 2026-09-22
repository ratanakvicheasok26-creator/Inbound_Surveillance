import type { Detection, RuleConfig, RuleHoldReason } from "./types";

export function holdReason(
  detection: Detection,
  rules: RuleConfig,
  lastAlertTsForCamera: number | undefined,
  now = Date.now(),
): RuleHoldReason {
  if (!rules.objectTypes.includes(detection.objectClass)) return "object-type";
  if (rules.roiEnabled && !detection.roiHit) return "roi";
  if (rules.timeWindow === "after-hours" && !detection.afterHours) return "time-window";
  if (
    lastAlertTsForCamera !== undefined &&
    now - lastAlertTsForCamera < rules.cooldownSec * 1000
  ) {
    return "cooldown";
  }
  return null;
}

export function lastAlertByCamera(
  alerts: { cameraId: string; ts: number; dismissed: boolean }[],
): Map<string, number> {
  const map = new Map<string, number>();
  for (const alert of alerts) {
    if (alert.dismissed) continue;
    const prev = map.get(alert.cameraId);
    if (prev === undefined || alert.ts > prev) map.set(alert.cameraId, alert.ts);
  }
  return map;
}

export function holdLabel(reason: RuleHoldReason): string {
  switch (reason) {
    case "object-type":
      return "Held · object type";
    case "roi":
      return "Held · ROI";
    case "time-window":
      return "Held · time window";
    case "cooldown":
      return "Held · cooldown";
    default:
      return "Promoted";
  }
}
