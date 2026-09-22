export const DEFAULT_ENGINE_PORT = 8765;

type EngineWindow = Window & {
  __INBOUND_ENGINE_PORT__?: number;
};

export function enginePort(): number {
  const injected = (window as EngineWindow).__INBOUND_ENGINE_PORT__;
  if (typeof injected === "number" && Number.isFinite(injected) && injected > 0) {
    return injected;
  }
  const fromEnv = import.meta.env.VITE_ENGINE_PORT;
  if (fromEnv) {
    const parsed = Number(fromEnv);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return DEFAULT_ENGINE_PORT;
}

export function engineBaseUrl(port = enginePort()): string {
  const explicit = import.meta.env.VITE_ENGINE_URL;
  if (explicit) return String(explicit).replace(/\/$/, "");
  return `http://127.0.0.1:${port}`;
}
