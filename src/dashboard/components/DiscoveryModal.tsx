import { useEffect, useState } from "react";
import { ScanAndGoLoader } from "../../components/ui/scan-and-go-loader";
import type {
  CameraProtocol,
  DiscoveredDevice,
  DiscoveryResults,
  DiscoveryStatus,
} from "../types";

export type DiscoveryModalProps = {
  open: boolean;
  engineBase: string;
  onClose: () => void;
  onConnected?: (info: {
    name: string;
    source: string;
    protocol: string;
    vendor?: string;
    username?: string;
    password?: string;
  }) => void;
};

function protocolFromDevice(dev: DiscoveredDevice): CameraProtocol {
  const st = String(dev.service_type || "").toLowerCase();
  const brand = `${dev.manufacturer} ${dev.model}`.toLowerCase();
  if (st === "webcam") return "webcam";
  if (st === "onvif") return "onvif";
  if (st === "tapo" || brand.includes("tapo") || brand.includes("tp-link") || brand.includes("tplink")) {
    return "tapo";
  }
  if (st === "webrtc" || st === "whep" || st === "whip") return "webrtc";
  if (st === "http" || st === "phone") return "phone";
  return "rtsp";
}

function vendorFromDevice(dev: DiscoveredDevice): string {
  const brand = `${dev.manufacturer} ${dev.model}`.toLowerCase();
  if (brand.includes("hik")) return "hikvision";
  if (brand.includes("dahua")) return "dahua";
  if (brand.includes("ip webcam")) return "ipwebcam";
  return "generic";
}

function webcamIndex(dev: DiscoveredDevice): string {
  const xaddrs = (dev.xaddrs || []).filter(Boolean);
  if (xaddrs.length && /^\d+$/.test(String(xaddrs[0]))) return String(xaddrs[0]);
  if (dev.port != null && String(dev.port) !== "" && Number(dev.port) >= 0 && Number(dev.port) < 64) {
    return String(dev.port);
  }
  if (/^\d+$/.test(String(dev.ip || ""))) return String(dev.ip);
  const fromName = String(dev.name || "").match(/(\d+)/);
  return fromName ? fromName[1] : "0";
}

function sourceFromDevice(dev: DiscoveredDevice): string {
  const protocol = protocolFromDevice(dev);
  const xaddrs = (dev.xaddrs || []).filter(Boolean);
  if (protocol === "webcam") return webcamIndex(dev);
  if (xaddrs.length) return xaddrs[0];
  const host = dev.ip || "";
  const port = dev.port;
  if (protocol === "onvif") {
    const p = port && Number(port) !== 80 ? `:${port}` : "";
    return `http://${host}${p}/onvif/device_service`;
  }
  if (protocol === "tapo") return host;
  if (protocol === "webrtc") return host ? `http://${host}${port ? `:${port}` : ""}/` : "";
  if (protocol === "phone") return `http://${host}:${port || 8080}/video`;
  return `rtsp://${host}:${port || 554}`;
}

function deviceLabel(dev: DiscoveredDevice): string {
  return (dev.name || "").trim() || (dev.ip ? `Camera ${dev.ip}` : "Discovered camera");
}

function brandLabel(dev: DiscoveredDevice): string {
  return [dev.manufacturer, dev.model].filter(Boolean).join(" ") || "—";
}

function normalizeDevice(raw: Record<string, unknown>): DiscoveredDevice {
  const xaddrs = Array.isArray(raw.xaddrs) ? raw.xaddrs.map((item) => String(item)) : [];
  return {
    ip: String(raw.ip ?? ""),
    port: Number(raw.port ?? 0) || 0,
    service_type: String(raw.service_type ?? ""),
    name: String(raw.name ?? ""),
    xaddrs,
    manufacturer: String(raw.manufacturer ?? ""),
    model: String(raw.model ?? ""),
  };
}

function normalizeStatus(raw: unknown): DiscoveryStatus {
  const status = String(raw || "idle");
  if (status === "empty") return "idle";
  if (status === "idle" || status === "scanning" || status === "done" || status === "error") {
    return status;
  }
  return "idle";
}

function parseResults(raw: Record<string, unknown>): DiscoveryResults {
  const devices = Array.isArray(raw.devices)
    ? raw.devices
        .filter((item): item is Record<string, unknown> => !!item && typeof item === "object")
        .map(normalizeDevice)
    : [];
  return {
    status: normalizeStatus(raw.status),
    devices,
    error: raw.error != null ? String(raw.error) : undefined,
    started_at: raw.started_at as string | number | undefined,
  };
}

export function DiscoveryModal({ open, engineBase, onClose, onConnected }: DiscoveryModalProps) {
  const [status, setStatus] = useState<DiscoveryStatus>("idle");
  const [devices, setDevices] = useState<DiscoveredDevice[]>([]);
  const [error, setError] = useState("");
  const [connecting, setConnecting] = useState(false);
  const [pending, setPending] = useState<DiscoveredDevice | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [connectError, setConnectError] = useState("");

  useEffect(() => {
    if (!open) {
      setPending(null);
      setUsername("");
      setPassword("");
      setConnectError("");
      return;
    }
    let cancelled = false;
    let timer = 0;
    const base = engineBase.replace(/\/$/, "");

    async function poll() {
      try {
        const res = await fetch(`${base}/api/discovery/results`, { cache: "no-store" });
        if (cancelled) return;
        if (!res.ok) {
          setStatus("error");
          setError(res.status === 404 ? "Discovery API is not available yet." : `Scan results failed (${res.status}).`);
          return;
        }
        const data = parseResults((await res.json()) as Record<string, unknown>);
        setStatus(data.status);
        setDevices(data.devices);
        setError(data.error || "");
        if (data.status !== "done" && data.status !== "error") {
          timer = window.setTimeout(poll, 700);
        }
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Scan failed.");
      }
    }

    async function start() {
      setStatus("scanning");
      setDevices([]);
      setError("");
      try {
        const res = await fetch(`${base}/api/discovery/scan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const data = (await res.json().catch(() => ({}))) as { success?: boolean; error?: string };
        if (cancelled) return;
        if (!res.ok || data.success === false) {
          setStatus("error");
          setError(
            data.error
              || (res.status === 404 ? "Discovery API is not available yet." : "Could not start scan."),
          );
          return;
        }
        poll();
      } catch (err) {
        if (cancelled) return;
        setStatus("error");
        setError(err instanceof Error ? err.message : "Scan failed.");
      }
    }

    start();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [open, engineBase]);

  if (!open) return null;

  function beginConnect(dev: DiscoveredDevice) {
    setConnectError("");
    if (protocolFromDevice(dev) === "webcam") {
      void submitConnect(dev, "", "");
      return;
    }
    setPending(dev);
    setUsername("");
    setPassword("");
  }

  async function submitConnect(dev: DiscoveredDevice, user: string, pass: string) {
    const protocol = protocolFromDevice(dev);
    const source = sourceFromDevice(dev);
    const name = deviceLabel(dev);
    const payload = {
      name,
      source,
      protocol,
      vendor: vendorFromDevice(dev),
      username: user,
      password: pass,
      main_source: "",
      xaddrs: dev.xaddrs || [],
      credentials: { username: user, password: pass },
    };
    setConnecting(true);
    setConnectError("");
    const base = engineBase.replace(/\/$/, "");
    try {
      const saveRes = await fetch(`${base}/api/cameras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const saveData = (await saveRes.json().catch(() => ({}))) as {
        success?: boolean;
        error?: string;
        camera?: { id?: string };
        active_camera_id?: string;
      };
      if (!saveData.success) {
        setConnectError(saveData.error || "Could not save camera.");
        setConnecting(false);
        return;
      }
      const cameraId = saveData.camera?.id || saveData.active_camera_id || "";
      const connectRes = await fetch(`${base}/api/connect-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: payload.source,
          main_source: payload.main_source,
          camera_id: cameraId,
          camera_name: payload.name,
          protocol: payload.protocol,
          vendor: payload.vendor,
          username: payload.username,
          password: payload.password,
          credentials: payload.credentials,
          xaddrs: payload.xaddrs,
        }),
      });
      const connectData = (await connectRes.json().catch(() => ({}))) as { success?: boolean; error?: string };
      if (connectData.success === false) {
        setConnectError(connectData.error || "Could not connect stream.");
        setConnecting(false);
        return;
      }
      onConnected?.({
        name,
        source,
        protocol,
        vendor: payload.vendor,
        username: user,
        password: pass,
      });
      onClose();
    } catch (err) {
      setConnectError(err instanceof Error ? err.message : "Connect failed.");
    } finally {
      setConnecting(false);
    }
  }

  const statusText =
    status === "scanning"
      ? devices.length
        ? `Scanning… ${devices.length} device${devices.length === 1 ? "" : "s"} so far.`
        : "Scanning the network…"
      : status === "error"
        ? error || "Scan failed."
        : status === "done"
          ? devices.length
            ? `${devices.length} device${devices.length === 1 ? "" : "s"} found.`
            : "No cameras found on this network."
          : "Not scanned yet. Scan the LAN for ONVIF, RTSP, and local webcams.";

  return (
    <div className="modal-scrim" role="presentation" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-labelledby="discovery-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="modal__head">
          <div>
            <h3 id="discovery-title">Scan network</h3>
            <p>ONVIF, RTSP, HTTP, and local webcams. Connect saves the camera then opens the stream.</p>
          </div>
          <button className="btn btn--ghost btn--sm" type="button" onClick={onClose}>
            Close
          </button>
        </header>
        {status === "scanning" && (
          <div style={{ display: "flex", justifyContent: "center", margin: "16px 0" }}>
            <ScanAndGoLoader size="sm" showTelemetry={false} statusLabel="SCANNING NETWORK FOR CAMERAS..." />
          </div>
        )}
        <p className={`discover-status${status === "error" ? " is-err" : ""}`}>{statusText}</p>
        {pending ? (
          <form
            className="discover-creds"
            onSubmit={(event) => {
              event.preventDefault();
              void submitConnect(pending, username, password);
            }}
          >
            <h4>{deviceLabel(pending)}</h4>
            <p>
              {pending.ip || "local"} · {brandLabel(pending)} · {pending.service_type || "unknown"}
            </p>
            <label className="field">
              <span>Username</span>
              <input
                autoComplete="off"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="camera login"
              />
            </label>
            <label className="field">
              <span>Password</span>
              <input
                type="password"
                autoComplete="off"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="camera password"
              />
            </label>
            {connectError ? <p className="discover-status is-err">{connectError}</p> : null}
            <div className="actions">
              <button className="btn btn--ghost btn--sm" type="button" onClick={() => setPending(null)} disabled={connecting}>
                Back
              </button>
              <button className="btn btn--primary btn--sm" type="submit" disabled={connecting}>
                {connecting ? "Connecting…" : "Connect"}
              </button>
            </div>
          </form>
        ) : (
          <div className="discover-list">
            {devices.map((dev, index) => (
              <article className="discover-row" key={`${dev.ip}-${dev.port}-${dev.service_type}-${index}`}>
                <div>
                  <h4>{deviceLabel(dev)}</h4>
                  <p>
                    {dev.ip || "local"}
                    {dev.port ? `:${dev.port}` : ""} · {brandLabel(dev)} · {dev.service_type || "unknown"}
                  </p>
                </div>
                <button className="btn btn--primary btn--sm" type="button" disabled={connecting} onClick={() => beginConnect(dev)}>
                  Connect
                </button>
              </article>
            ))}
            {connectError ? <p className="discover-status is-err">{connectError}</p> : null}
          </div>
        )}
      </div>
    </div>
  );
}
