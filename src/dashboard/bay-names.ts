import { parseWorkplaceId, parseZoneKind, type WorkplaceId, type ZoneKind } from "../workplaces";

export type StationBay = {
  id: string;
  name: string;
  type: ZoneKind | string;
  roi: number[];
};

export function nextAvailableBayName(bays: { name?: string }[]): string {
  const used = new Set<number>();
  for (const bay of bays) {
    const match = String(bay.name || "").match(/Bay\s*(\d+)/i);
    if (match) used.add(Number.parseInt(match[1], 10));
  }
  let num = 1;
  while (used.has(num)) num += 1;
  return `Bay ${num}`;
}

export function engineBayToStation(
  row: Record<string, unknown>,
  workplace: WorkplaceId | string = "garage",
): StationBay | null {
  const id = String(row.bay_id || row.id || "").trim();
  if (!id) return null;
  const roiRaw = Array.isArray(row.roi) ? row.roi.map(Number) : [0.3, 0.2, 0.2, 0.3];
  const workplaceId = parseWorkplaceId(String(workplace));
  const rawType = String(row.type || row.zone_kind || "");
  return {
    id,
    name: String(row.name || id),
    type: parseZoneKind(rawType, workplaceId),
    roi: roiRaw.length === 4 ? roiRaw : [0.3, 0.2, 0.2, 0.3],
  };
}
