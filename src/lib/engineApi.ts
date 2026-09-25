import { engineBaseUrl } from "../engine-url";

let currentToken: string | null = null;

function persistedToken(): string | null {
  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith("sb-") || !key.endsWith("-auth-token")) continue;
      const raw = JSON.parse(localStorage.getItem(key) || "null");
      const token = raw?.access_token ?? raw?.currentSession?.access_token;
      if (typeof token === "string" && token.length > 0) return token;
    }
  } catch {
    // ignore
  }
  return null;
}

export function setEngineAuthToken(token: string | null): void {
  currentToken = token;
}

export function engineAuthToken(): string | null {
  return currentToken ?? persistedToken();
}

export async function engineApi(
  path: string,
  init: RequestInit = {},
  base: string = engineBaseUrl(),
): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = engineAuthToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${base.replace(/\/$/, "")}${path}`, { ...init, headers });
}