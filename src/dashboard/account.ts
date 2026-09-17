import type { User } from "@supabase/supabase-js";
import { ACCOUNT_BUCKET, supabase } from "../lib/supabase";
import type { Database, Json } from "../lib/database.types";
import { templateFor } from "../pipeline/templates";
import { parseWorkplaceId, parseZoneKind, type WorkplaceId, type ZoneKind } from "../workplaces";
import type { Camera, CameraProtocol } from "./types";

export const PROTOCOLS: CameraProtocol[] = ["rtsp", "onvif", "tapo", "phone", "webcam", "webrtc"];

export type ProfileRow = Database["public"]["Tables"]["profiles"]["Row"];
export type CameraRow = Database["public"]["Tables"]["cameras"]["Row"];
export type RoiRow = Database["public"]["Tables"]["roi_bays"]["Row"];
export type CrewRow = Database["public"]["Tables"]["crew_identities"]["Row"];
export type ComplaintRow = Database["public"]["Tables"]["complaints"]["Row"];
export type PipelineGraphRow = Database["public"]["Tables"]["pipeline_graphs"]["Row"];
export type CustomerVisitRow = Database["public"]["Tables"]["customer_visits"]["Row"];

export type CrewWithPhoto = CrewRow & { photoUrl: string | null };

export type AccountSnapshot = {
  profile: ProfileRow;
  cameras: CameraRow[];
  rois: RoiRow[];
  crews: CrewWithPhoto[];
  complaints: ComplaintRow[];
  visits: CustomerVisitRow[];
  pipelineGraph: PipelineGraphRow | null;
  avatarUrl: string | null;
};

function throwIf(error: { message: string } | null) {
  if (error) throw new Error(error.message);
}

function must<T>(data: T | null, error: { message: string } | null, fallback = "No data returned."): T {
  throwIf(error);
  if (data == null) throw new Error(fallback);
  return data;
}

export function initials(name: string, email?: string | null): string {
  const source = name.trim() || email?.split("@")[0] || "OP";
  const parts = source.split(/\s+/).filter(Boolean);
  const letters = (parts.length > 1 ? parts[0][0] + parts[1][0] : source.slice(0, 2)).toUpperCase();
  return letters || "OP";
}

export function redactUrl(value: string | undefined): string {
  if (!value) return "";
  if (/^\d+$/.test(value)) return value;
  try {
    const withProto = value.includes("://") ? value : `rtsp://${value}`;
    const url = new URL(withProto);
    url.username = "";
    url.password = "";
    const stripped = url.toString().replace(/\/$/, "");
    return value.includes("://") ? stripped : stripped.replace(/^rtsp:\/\//, "");
  } catch {
    return value.replace(/\/\/([^/@]+)@/, "//");
  }
}

export function hostFromUrl(value: string): string {
  const clean = redactUrl(value);
  if (!clean) return "no url";
  if (/^\d+$/.test(clean)) return `webcam ${clean}`;
  try {
    const withProto = clean.includes("://") ? clean : `rtsp://${clean}`;
    const url = new URL(withProto);
    return url.host || clean;
  } catch {
    return clean;
  }
}

export function persistableCameras(cameras: Camera[]): Camera[] {
  return cameras.map((cam) => ({
    id: cam.id,
    name: cam.name,
    zone: cam.zone,
    rtspLabel: cam.rtspLabel,
    protocol: cam.protocol,
    vendor: cam.vendor,
  }));
}

export function cameraToOps(row: CameraRow): Camera {
  const host = hostFromUrl(row.source_url);
  return {
    id: row.id,
    name: row.name,
    zone: row.zone || row.name,
    rtspLabel: `${row.protocol} · ${host}`,
    protocol: row.protocol,
    vendor: row.vendor || undefined,
    source: redactUrl(row.source_url),
    mainSource: redactUrl(row.main_source_url) || undefined,
    username: row.username || undefined,
  };
}

export async function signedPath(path: string | null | undefined): Promise<string | null> {
  if (!path) return null;
  const { data, error } = await supabase.storage.from(ACCOUNT_BUCKET).createSignedUrl(path, 60 * 60);
  if (error) return null;
  return data.signedUrl;
}

function extOf(file: File): string {
  const fromName = file.name.split(".").pop()?.toLowerCase();
  if (fromName && /^[a-z0-9]+$/.test(fromName)) return fromName;
  if (file.type === "image/png") return "png";
  if (file.type === "image/webp") return "webp";
  if (file.type === "image/gif") return "gif";
  return "jpg";
}

async function uploadPrivate(userId: string, folder: string, file: File, previousPath?: string | null) {
  const path = `${userId}/${folder}/${crypto.randomUUID()}.${extOf(file)}`;
  const { error } = await supabase.storage.from(ACCOUNT_BUCKET).upload(path, file, {
    upsert: true,
    contentType: file.type || "image/jpeg",
  });
  throwIf(error);
  if (previousPath && previousPath !== path) {
    await supabase.storage.from(ACCOUNT_BUCKET).remove([previousPath]);
  }
  return path;
}

export async function ensureProfile(user: User): Promise<ProfileRow> {
  const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).maybeSingle();
  throwIf(error);
  if (data) return data;
  const fallbackName =
    (typeof user.user_metadata?.display_name === "string" && user.user_metadata.display_name) ||
    user.email?.split("@")[0] ||
    "Operator";
  const fallbackVenue =
    (typeof user.user_metadata?.venue_name === "string" && user.user_metadata.venue_name) || "";
  const fallbackWorkplace = parseWorkplaceId(
    typeof user.user_metadata?.workplace_type === "string" ? user.user_metadata.workplace_type : "garage",
  );
  const inserted = await supabase
    .from("profiles")
    .upsert({
      id: user.id,
      display_name: fallbackName,
      venue_name: fallbackVenue,
      workplace_type: fallbackWorkplace,
    })
    .select("*")
    .single();
  throwIf(inserted.error);
  return must(inserted.data, inserted.error, "Could not create profile.");
}

export async function loadAccount(user: User): Promise<AccountSnapshot> {
  const profile = await ensureProfile(user);
  const [camerasRes, roisRes, crewsRes, complaintsRes, visitsRes, graphRes, avatarUrl] = await Promise.all([
    supabase.from("cameras").select("*").eq("user_id", user.id).order("created_at", { ascending: true }),
    supabase.from("roi_bays").select("*").eq("user_id", user.id).order("sort_order", { ascending: true }),
    supabase.from("crew_identities").select("*").eq("user_id", user.id).order("created_at", { ascending: true }),
    supabase.from("complaints").select("*").eq("user_id", user.id).order("created_at", { ascending: false }),
    supabase.from("customer_visits").select("*").eq("user_id", user.id).order("started_at", { ascending: false }).limit(200),
    supabase.from("pipeline_graphs").select("*").eq("user_id", user.id).eq("is_active", true).maybeSingle(),
    signedPath(profile.avatar_path),
  ]);
  throwIf(camerasRes.error);
  throwIf(roisRes.error);
  throwIf(crewsRes.error);
  throwIf(complaintsRes.error);
  throwIf(visitsRes.error);
  if (graphRes.error && graphRes.error.code !== "PGRST116") throwIf(graphRes.error);
  let pipelineGraph = graphRes.data || null;
  if (!pipelineGraph) {
    pipelineGraph = await savePipelineGraph(
      user.id,
      parseWorkplaceId(profile.workplace_type),
      templateFor(profile.workplace_type) as unknown as Json,
    );
  }
  const crews = await Promise.all(
    (crewsRes.data || []).map(async (crew) => ({
      ...crew,
      photoUrl: await signedPath(crew.photo_path),
    })),
  );
  return {
    profile,
    cameras: camerasRes.data || [],
    rois: roisRes.data || [],
    crews,
    complaints: complaintsRes.data || [],
    visits: visitsRes.data || [],
    pipelineGraph,
    avatarUrl,
  };
}

export async function updateProfile(
  userId: string,
  patch: Pick<ProfileRow, "display_name" | "venue_name"> & {
    setup_completed?: boolean;
    workplace_type?: WorkplaceId;
  },
) {
  const { data, error } = await supabase.from("profiles").update(patch).eq("id", userId).select("*").single();
  return must(data, error, "Could not update profile.");
}

export async function uploadAvatar(user: User, file: File, previousPath?: string | null) {
  const path = await uploadPrivate(user.id, "avatar", file, previousPath);
  const { data, error } = await supabase
    .from("profiles")
    .update({ avatar_path: path })
    .eq("id", user.id)
    .select("*")
    .single();
  return { profile: must(data, error, "Could not save avatar."), avatarUrl: await signedPath(path) };
}

export type CameraInput = {
  id?: string;
  name: string;
  zone?: string;
  protocol: CameraProtocol | string;
  source_url: string;
  main_source_url?: string;
  username?: string;
  password?: string;
  vendor?: string;
  external_id?: string | null;
};

function asProtocol(value: string): CameraRow["protocol"] {
  return (PROTOCOLS as string[]).includes(value) ? (value as CameraRow["protocol"]) : "rtsp";
}

export async function upsertCamera(userId: string, input: CameraInput) {
  const payload = {
    user_id: userId,
    name: input.name.trim(),
    zone: (input.zone || "").trim(),
    protocol: asProtocol(String(input.protocol || "rtsp")),
    source_url: input.source_url.trim(),
    main_source_url: (input.main_source_url || "").trim(),
    username: (input.username || "").trim(),
    vendor: (input.vendor || "").trim(),
    external_id: input.external_id || null,
    ...(input.password != null && input.password !== "" ? { password: input.password } : {}),
  };
  const query = input.id
    ? supabase.from("cameras").update(payload).eq("id", input.id).eq("user_id", userId)
    : supabase.from("cameras").insert(payload);
  const { data, error } = await query.select("*").single();
  return must(data, error, "Could not save camera.");
}

export async function deleteCamera(userId: string, id: string) {
  const { error } = await supabase.from("cameras").delete().eq("id", id).eq("user_id", userId);
  throwIf(error);
}

export type RoiInput = {
  id?: string;
  name: string;
  bay_type?: RoiRow["bay_type"];
  zone_kind?: ZoneKind;
  roi: number[];
  camera_id?: string | null;
  external_id?: string | null;
  sort_order?: number;
};

export async function replaceRois(userId: string, bays: RoiInput[], workplace: WorkplaceId = "garage") {
  const { error: delError } = await supabase.from("roi_bays").delete().eq("user_id", userId);
  throwIf(delError);
  if (!bays.length) return [] as RoiRow[];
  const { data, error } = await supabase
    .from("roi_bays")
    .insert(
      bays.map((bay, index) => {
        const kind = parseZoneKind(bay.zone_kind || bay.bay_type, workplace);
        return {
          user_id: userId,
          name: bay.name.trim(),
          bay_type: kind,
          zone_kind: kind,
          roi: bay.roi.slice(0, 4),
          camera_id: bay.camera_id || null,
          external_id: bay.external_id || null,
          sort_order: bay.sort_order ?? index,
        };
      }),
    )
    .select("*");
  throwIf(error);
  return data || [];
}

export async function upsertCrew(userId: string, input: { id?: string; display_name: string; role?: string }) {
  const payload = {
    user_id: userId,
    display_name: input.display_name.trim(),
    role: (input.role || "technician").trim() || "technician",
  };
  const query = input.id
    ? supabase.from("crew_identities").update(payload).eq("id", input.id).eq("user_id", userId)
    : supabase.from("crew_identities").insert(payload);
  const { data, error } = await query.select("*").single();
  return must(data, error, "Could not save crew identity.");
}

export async function uploadCrewPhoto(user: User, crew: CrewRow, file: File) {
  const path = await uploadPrivate(user.id, `crew/${crew.id}`, file, crew.photo_path);
  const { data, error } = await supabase
    .from("crew_identities")
    .update({ photo_path: path })
    .eq("id", crew.id)
    .eq("user_id", user.id)
    .select("*")
    .single();
  return { crew: must(data, error, "Could not save crew photo."), photoUrl: await signedPath(path) };
}

export async function deleteCrew(userId: string, crew: CrewRow) {
  if (crew.photo_path) {
    await supabase.storage.from(ACCOUNT_BUCKET).remove([crew.photo_path]);
  }
  const { error } = await supabase.from("crew_identities").delete().eq("id", crew.id).eq("user_id", userId);
  throwIf(error);
}

export async function listComplaints(userId: string) {
  const { data, error } = await supabase
    .from("complaints")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });
  throwIf(error);
  return data || [];
}

export async function importComplaint(
  userId: string,
  input: {
    body?: string;
    channel?: ComplaintRow["channel"];
    status?: ComplaintRow["status"];
    external_ref?: string | null;
    payload?: Json;
  },
) {
  const { data, error } = await supabase
    .from("complaints")
    .insert({
      user_id: userId,
      body: (input.body || "").trim(),
      channel: input.channel || "import",
      status: input.status || "open",
      external_ref: input.external_ref || null,
      payload: input.payload ?? {},
    })
    .select("*")
    .single();
  return must(data, error, "Could not import complaint.");
}

export async function savePipelineGraph(userId: string, workplace: WorkplaceId, graph: Json) {
  const existing = await supabase
    .from("pipeline_graphs")
    .select("id")
    .eq("user_id", userId)
    .eq("is_active", true)
    .maybeSingle();
  throwIf(existing.error);
  if (existing.data?.id) {
    const { data, error } = await supabase
      .from("pipeline_graphs")
      .update({ graph, workplace_type: workplace, is_active: true })
      .eq("id", existing.data.id)
      .eq("user_id", userId)
      .select("*")
      .single();
    return must(data, error, "Could not save pipeline.");
  }
  const { data, error } = await supabase
    .from("pipeline_graphs")
    .insert({
      user_id: userId,
      workplace_type: workplace,
      graph,
      is_active: true,
    })
    .select("*")
    .single();
  return must(data, error, "Could not save pipeline.");
}
