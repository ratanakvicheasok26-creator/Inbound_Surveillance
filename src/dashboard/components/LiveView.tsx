import { useEffect, useMemo, useState } from "react";
import { engineBaseUrl } from "../../engine-url";
import { engineApi } from "../../lib/engineApi";
import { engineBayToStation, type StationBay } from "../bay-names";
import { cameraById, confidenceLabel, formatRelative } from "../format";
import { holdLabel, holdReason, lastAlertByCamera } from "../rules";
import { PROTOCOLS } from "../account";
import { useAccount } from "../auth";
import { useOps } from "../store";
import type { CameraProtocol, Detection, EngineTelemetry } from "../types";
import { workplaceOf, parseZoneKind } from "../../workplaces";
import { BayContextMenu } from "./BayContextMenu";
import { DiscoveryModal } from "./DiscoveryModal";

function Scene({ detection }: { detection: Detection }) {
  const { state } = useOps();
  const last = lastAlertByCamera(state.alerts);
  const hold = holdReason(detection, state.rules, last.get(detection.cameraId));
  const camera = cameraById(state.cameras, detection.cameraId);

  return (
    <article className="frame">
      <div className="frame__scene">
        <div
          className={`bbox${hold ? " is-held" : ""}`}
          style={{
            left: `${detection.bbox.x}%`,
            top: `${detection.bbox.y}%`,
            width: `${detection.bbox.w}%`,
            height: `${detection.bbox.h}%`,
          }}
        />
        <span className="frame__tag">
          {camera?.name} · {camera?.protocol ? `${camera.protocol} · ` : ""}{camera?.rtspLabel}
        </span>
      </div>
      <div className="frame__meta">
        <strong>
          {detection.objectClass} · {confidenceLabel(detection.confidence)}
        </strong>
        <span>
          {camera?.zone} · {formatRelative(detection.ts)} · {detection.id}
        </span>
        <span className={`pill${hold ? " is-held" : ""}`}>{holdLabel(hold)}</span>
      </div>
    </article>
  );
}

function formatIngest(tel: EngineTelemetry | null): string {
  if (!tel) return "— fps";
  const raw = tel.ingest_fps != null ? tel.ingest_fps : tel.fps;
  if (raw == null || Number.isNaN(Number(raw))) return "— fps";
  return `${Number(raw).toFixed(1)} fps`;
}

function formatInfer(tel: EngineTelemetry | null): string {
  if (tel?.infer_ms == null || Number.isNaN(Number(tel.infer_ms))) return "— ms";
  return `${Math.round(Number(tel.infer_ms))} ms`;
}

function formatResolution(tel: EngineTelemetry | null): string {
  if (!tel) return "—";
  if (tel.resolution) return tel.resolution;
  if (tel.width && tel.height) return `${tel.width}x${tel.height}`;
  return "—";
}

function asProtocol(value: string): CameraProtocol {
  return (PROTOCOLS as string[]).includes(value) ? (value as CameraProtocol) : "rtsp";
}

export function LiveView() {
  const { state } = useOps();
  const { saveCamera, saveRois, snapshot } = useAccount();
  const workplace = workplaceOf(snapshot?.profile.workplace_type);
  const [engineLive, setEngineLive] = useState(false);
  const [scanOpen, setScanOpen] = useState(false);
  const [telemetry, setTelemetry] = useState<EngineTelemetry | null>(null);
  const [selectedBayId, setSelectedBayId] = useState("");
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const detections = [...state.detections].sort((a, b) => b.ts - a.ts);
  const engine = engineBaseUrl();
  const [liveFrame, setLiveFrame] = useState("");
  const [showRoi, setShowRoi] = useState(() => {
    try {
      return localStorage.getItem("dashboard_roi_visible") !== "false";
    } catch {
      return true;
    }
  });
  const bays = useMemo(() => {
    const rows = telemetry?.zones || telemetry?.bays || [];
    return rows
      .map((row) => engineBayToStation(row as Record<string, unknown>, workplace.id))
      .filter((row): row is StationBay => row != null);
  }, [telemetry, workplace.id]);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      const controller = new AbortController();
      const abortTimer = window.setTimeout(() => controller.abort(), 1200);
      try {
        const res = await engineApi(`/api/telemetry`, { cache: "no-store", signal: controller.signal });
        if (cancelled) return;
        if (!res.ok) {
          setEngineLive(false);
          return;
        }
        const data = (await res.json()) as EngineTelemetry;
        setEngineLive(true);
        setTelemetry(data);
      } catch {
        if (!cancelled) setEngineLive(false);
      } finally {
        window.clearTimeout(abortTimer);
      }
    }
    void tick();
    const interval = window.setInterval(() => void tick(), 1000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [engine]);

  useEffect(() => {
    if (!engineLive) return;
    let cancelled = false;
    let current = "";
    async function pump() {
      while (!cancelled) {
        try {
          const res = await engineApi(`/api/frame.jpeg?t=${Date.now()}`, { cache: "no-store" });
          if (cancelled) break;
          if (res.ok) {
            const blob = await res.blob();
            if (cancelled) break;
            const next = URL.createObjectURL(blob);
            setLiveFrame(next);
            if (current) URL.revokeObjectURL(current);
            current = next;
          }
        } catch {
          /* retry */
        }
        await new Promise((resolve) => window.setTimeout(resolve, 80));
      }
      if (current) URL.revokeObjectURL(current);
    }
    void pump();
    return () => {
      cancelled = true;
    };
  }, [engine, engineLive]);

  useEffect(() => {
    function onKey(ev: KeyboardEvent) {
      if (ev.key === "Escape") {
        setMenu(null);
        return;
      }
      if (ev.key !== "Delete" && ev.key !== "Backspace") return;
      const tag = ((ev.target as HTMLElement | null)?.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if ((ev.target as HTMLElement | null)?.isContentEditable) return;
      if (!selectedBayId) return;
      ev.preventDefault();
      void deleteBay(selectedBayId);
    }
    function onDown(ev: MouseEvent) {
      const target = ev.target as HTMLElement | null;
      if (target?.closest?.("#bay-context-menu")) return;
      setMenu(null);
    }
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onDown);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onDown);
    };
  }, [selectedBayId, bays, engine]);

  async function persistBays(next: StationBay[]) {
    try {
      const res = await engineApi(`/api/workplace/zones`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bays: next }),
      });
      const data = (await res.json()) as { bays?: Record<string, unknown>[] };
      if (Array.isArray(data.bays)) {
        setTelemetry((prev) => ({
          ...(prev || {}),
          bays: data.bays,
        }));
      }
    } catch {
      /* Engine may be offline; still persist ROI privately. */
    }
    await saveRois(
      next.map((bay, index) => ({
        name: bay.name,
        bay_type: parseZoneKind(String(bay.type), workplace.id),
        zone_kind: parseZoneKind(String(bay.type), workplace.id),
        roi: bay.roi,
        external_id: bay.id,
        sort_order: index,
      })),
    );
  }

  async function deleteBay(bayId: string) {
    setMenu(null);
    const bay = bays.find((item) => item.id === bayId);
    if (!bay) return;
    if (!window.confirm(`Delete "${bay.name}"? Past reports keep their history.`)) return;
    const next = bays.filter((item) => item.id !== bayId);
    setSelectedBayId(next[0]?.id || "");
    await persistBays(next);
  }

  function renameBay() {
    setMenu(null);
    const bay = bays.find((item) => item.id === selectedBayId);
    if (!bay) return;
    const label = window.prompt(`Rename ${workplace.zoneNoun}`, bay.name);
    if (label == null) return;
    const nextName = label.trim();
    if (!nextName) return;
    void persistBays(
      bays.map((item) =>
        item.id === bay.id
          ? { ...item, name: nextName, type: /tool/i.test(nextName) ? "tool_area" : item.type }
          : item,
      ),
    );
  }

  function toggleBayType() {
    setMenu(null);
    const bay = bays.find((item) => item.id === selectedBayId);
    if (!bay) return;
    const kinds = workplace.zoneKinds.map((item) => item.id);
    const current = kinds.indexOf(bay.type as (typeof kinds)[number]);
    const nextKind = kinds[(current + 1 + kinds.length) % kinds.length] || workplace.defaultZoneKind;
    void persistBays(
      bays.map((item) => (item.id === bay.id ? { ...item, type: nextKind } : item)),
    );
  }

  const proto = (telemetry?.protocol || "").trim();
  const hasMain = Boolean(telemetry?.main_stream);

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Live inference</h2>
          <p>
            Mocked RTSP ingest at 2 fps. YOLO returns class, bounding box, and confidence.
            Only detections that pass the rule engine are packaged as alerts.
          </p>
        </div>
        <div className="live-badges">
          {engineLive ? (
            <>
              <span className="live-badge is-proto">{proto ? proto.toUpperCase() : "ENGINE"}</span>
              <span className="live-badge">{formatResolution(telemetry)}</span>
              <span className="live-badge">{formatIngest(telemetry)}</span>
              <span className="live-badge">{formatInfer(telemetry)}</span>
              {hasMain ? <span className="live-badge">MAIN</span> : null}
            </>
          ) : null}
          <button
            className={`btn btn--sm ${showRoi ? "btn--primary" : "btn--ghost"}`}
            type="button"
            title="Toggle ROI hotspot overlays"
            onClick={() => {
              const next = !showRoi;
              setShowRoi(next);
              try {
                localStorage.setItem("dashboard_roi_visible", String(next));
              } catch {}
            }}
          >
            {showRoi ? "ROI: ON" : "ROI: OFF"}
          </button>
          <button className="btn btn--ghost btn--sm" type="button" onClick={() => setScanOpen(true)}>
            Scan network
          </button>
        </div>
      </header>
      <div className="grid-detect">
        {engineLive ? (
          <article className="frame">
            <div className="frame__scene">
              {liveFrame ? (
                <img src={liveFrame} alt="Live camera engine" />
              ) : (
                <div className="frame__tag">Waiting for frames…</div>
              )}
              {showRoi &&
                bays.map((bay) => {
                  const [x, y, w, h] = bay.roi;
                  return (
                    <button
                      key={bay.id}
                      type="button"
                      className={`roi-hotspot${bay.id === selectedBayId ? " is-selected" : ""}`}
                      style={{
                        left: `${x * 100}%`,
                        top: `${y * 100}%`,
                        width: `${w * 100}%`,
                        height: `${h * 100}%`,
                      }}
                      onClick={() => setSelectedBayId(bay.id)}
                      onContextMenu={(event) => {
                        event.preventDefault();
                        setSelectedBayId(bay.id);
                        setMenu({ x: event.clientX, y: event.clientY });
                      }}
                      aria-label={bay.name}
                      title={bay.name}
                    />
                  );
                })}
              <span className="frame__tag">
                Edge engine · {engine}
                {proto ? ` · ${proto}` : ""}
              </span>
            </div>
            <div className="frame__meta">
              <strong>Live JPEG frames from the local camera engine</strong>
              <span>
                {formatResolution(telemetry)} · ingest {formatIngest(telemetry)} · infer {formatInfer(telemetry)}
              </span>
            </div>
          </article>
        ) : null}
        {detections.map((detection) => (
          <Scene key={detection.id} detection={detection} />
        ))}
      </div>
      <DiscoveryModal
        open={scanOpen}
        engineBase={engine}
        onClose={() => setScanOpen(false)}
        onConnected={(info) => {
          void saveCamera({
            name: info.name,
            protocol: asProtocol(info.protocol),
            source_url: info.source,
            username: info.username,
            password: info.password,
            vendor: info.vendor,
          });
        }}
      />
      <BayContextMenu
        open={Boolean(menu)}
        x={menu?.x || 0}
        y={menu?.y || 0}
        zoneNoun={workplace.zoneNoun}
        toggleLabel={workplace.zoneKinds.map((item) => item.label).join(" / ")}
        onRename={renameBay}
        onToggleType={toggleBayType}
        onDelete={() => void deleteBay(selectedBayId)}
      />
    </section>
  );
}
