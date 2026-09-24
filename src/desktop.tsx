import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { engineBaseUrl, enginePort } from "./engine-url";

interface MissingRuntime {
  id: string;
  title: string;
  description: string;
  install_hint: string;
  install_command: string;
  can_install: boolean;
}

interface EngineFailure {
  exit_code?: number | null;
  error_summary: string;
  log_path: string;
  is_dll_error: boolean;
  missing_runtime?: MissingRuntime | null;
}

const status = document.getElementById("status");
const pulseDot = document.querySelector(".pulse-dot") as HTMLElement | null;
const errorPanel = document.getElementById("error-panel") as HTMLElement | null;
const errorTitle = document.getElementById("error-title");
const errorDesc = document.getElementById("error-desc");
const vcredistBox = document.getElementById("vcredist-box") as HTMLElement | null;
const runtimeTitle = document.getElementById("runtime-title");
const runtimeDesc = document.getElementById("runtime-desc");
const runtimeHint = document.getElementById("runtime-hint");
const runtimeCommand = document.getElementById("runtime-command");
const btnInstallRuntime = document.getElementById("btn-install-runtime") as HTMLButtonElement | null;
const btnCopyCommand = document.getElementById("btn-copy-command") as HTMLButtonElement | null;
const errorLog = document.getElementById("error-log");
const errorPath = document.getElementById("error-path");
const btnRetry = document.getElementById("btn-retry");
const btnOpenLog = document.getElementById("btn-open-log");
const btnCopyError = document.getElementById("btn-copy-error");

let redirected = false;
let pollTimer: number | null = null;
let lastFailure: EngineFailure | null = null;

function setStatus(text: string, isError = false) {
  if (status) {
    status.textContent = text;
    status.style.color = isError ? "#f87171" : "var(--color-surveillance-green)";
  }
}

function openHub(port: number) {
  if (redirected) return;
  redirected = true;
  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
  const url = `${engineBaseUrl(port)}/`;
  setStatus(`Connecting to camera hub on port ${port}…`);
  window.location.replace(url);
}

function showEngineFailure(failure: EngineFailure) {
  if (redirected) return;
  lastFailure = failure;

  if (pollTimer) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }

  setStatus("Engine process stopped", true);
  if (pulseDot) pulseDot.style.display = "none";

  if (errorPanel) {
    errorPanel.style.display = "block";
  }

  if (errorTitle) {
    errorTitle.textContent =
      failure.exit_code !== undefined && failure.exit_code !== null
        ? `Engine Exited (Code ${failure.exit_code})`
        : "Engine Startup Failed";
  }

  const missing = failure.missing_runtime ?? null;

  if (errorDesc) {
    errorDesc.textContent = missing
      ? missing.description
      : failure.is_dll_error
        ? "A required system C++ library is missing from this Windows installation."
        : "The camera engine exited unexpectedly during startup.";
  }

  if (vcredistBox) {
    vcredistBox.style.display = missing || failure.is_dll_error ? "block" : "none";
  }

  if (runtimeTitle) {
    runtimeTitle.textContent = missing?.title || "Missing Visual C++ runtime";
  }
  if (runtimeDesc) {
    runtimeDesc.textContent =
      missing?.description ||
      "PyTorch and OpenCV require Microsoft Visual C++ 2015–2022 Redistributable (x64).";
  }
  if (runtimeHint) {
    runtimeHint.textContent = missing?.install_hint || "";
  }

  const command = missing?.install_command?.trim() || "";
  if (runtimeCommand) {
    runtimeCommand.textContent = command;
    runtimeCommand.style.display = command ? "block" : "none";
  }

  if (btnInstallRuntime) {
    const canInstall = Boolean(missing?.can_install || failure.is_dll_error);
    btnInstallRuntime.style.display = canInstall ? "inline-flex" : "none";
    btnInstallRuntime.disabled = false;
    btnInstallRuntime.textContent = "Install now";
  }
  if (btnCopyCommand) {
    btnCopyCommand.style.display = command ? "inline-flex" : "none";
    btnCopyCommand.textContent = "Copy install command";
  }

  if (errorLog) {
    errorLog.textContent = failure.error_summary || "No error output recorded.";
  }

  if (errorPath) {
    errorPath.textContent = failure.log_path ? `Log: ${failure.log_path}` : "Diagnostic log written to disk & Desktop.";
    void invoke<string[]>("get_diagnostic_logs")
      .then((logs) => {
        if (logs && logs.length > 0 && errorPath) {
          errorPath.innerHTML = `<strong>Diagnostic Logs:</strong><br/>${logs.map((p) => `• ${p}`).join("<br/>")}<br/><em>Crash report also saved to Desktop as INBOUND_CRASH_REPORT.txt</em>`;
        }
      })
      .catch(() => {});
  }
}


async function probe(port: number): Promise<boolean> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 800);
  try {
    const res = await fetch(`${engineBaseUrl(port)}/api/public-config`, {
      signal: controller.signal,
      cache: "no-store",
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

function startPolling() {
  if (pollTimer) window.clearInterval(pollTimer);
  const started = Date.now();
  pollTimer = window.setInterval(async () => {
    if (redirected) {
      if (pollTimer) window.clearInterval(pollTimer);
      return;
    }
    if (await probe(enginePort())) {
      if (pollTimer) window.clearInterval(pollTimer);
      openHub(enginePort());
      return;
    }
    if (Date.now() - started > 90_000) {
      if (pollTimer) window.clearInterval(pollTimer);
      showEngineFailure({
        exit_code: null,
        error_summary: "Engine probe timed out after 90 seconds. Port 8765 did not respond.",
        log_path: "",
        is_dll_error: false,
      });
    }
  }, 500);
}

function initErrorActions() {
  btnRetry?.addEventListener("click", async () => {
    if (errorPanel) errorPanel.style.display = "none";
    if (pulseDot) pulseDot.style.display = "block";
    setStatus("Restarting camera engine…");
    try {
      await invoke("retry_engine");
    } catch (_) {}
    startPolling();
  });

  btnInstallRuntime?.addEventListener("click", async () => {
    if (!btnInstallRuntime) return;
    btnInstallRuntime.disabled = true;
    btnInstallRuntime.textContent = "Installing…";
    setStatus("Installing the missing Visual C++ runtime…");
    try {
      const message = await invoke<string>("install_bundled_runtime");
      btnInstallRuntime.textContent = "Installed";
      setStatus(message || "Runtime installed. Restarting engine…");
      if (errorPanel) errorPanel.style.display = "none";
      if (pulseDot) pulseDot.style.display = "block";
      try {
        await invoke("retry_engine");
      } catch (_) {}
      startPolling();
    } catch (err) {
      btnInstallRuntime.disabled = false;
      btnInstallRuntime.textContent = "Install now";
      setStatus(String(err), true);
    }
  });

  btnCopyCommand?.addEventListener("click", async () => {
    const command = lastFailure?.missing_runtime?.install_command?.trim();
    if (!command) return;
    try {
      await navigator.clipboard.writeText(command);
      if (btnCopyCommand) btnCopyCommand.textContent = "Copied!";
      window.setTimeout(() => {
        if (btnCopyCommand) btnCopyCommand.textContent = "Copy install command";
      }, 2000);
    } catch (_) {}
  });

  btnOpenLog?.addEventListener("click", async () => {
    try {
      await invoke("open_engine_log");
    } catch (_) {}
  });

  btnCopyError?.addEventListener("click", async () => {
    const missing = lastFailure?.missing_runtime;
    const text = [
      `Exit Code: ${lastFailure?.exit_code ?? "N/A"}`,
      `Log Path: ${lastFailure?.log_path ?? "N/A"}`,
      missing ? `Missing: ${missing.title}` : "",
      missing?.install_command ? `Install: ${missing.install_command}` : "",
      `Error Summary:\n${lastFailure?.error_summary ?? "N/A"}`,
    ]
      .filter(Boolean)
      .join("\n\n");
    try {
      await navigator.clipboard.writeText(text);
      if (btnCopyError) btnCopyError.textContent = "Copied!";
      window.setTimeout(() => {
        if (btnCopyError) btnCopyError.textContent = "Copy Details";
      }, 2000);
    } catch (_) {}
  });
}

async function boot() {
  initErrorActions();

  // Expose hook on window so Rust eval can also trigger it directly
  (window as unknown as { __onEngineFailed: (f: EngineFailure) => void }).__onEngineFailed =
    showEngineFailure;

  window.addEventListener(
    "inbound-engine-failed",
    ((event: CustomEvent<EngineFailure>) => {
      if (event.detail) showEngineFailure(event.detail);
    }) as EventListener
  );

  window.addEventListener("inbound-engine-ready", ((event: CustomEvent<number>) => {
    if (typeof event.detail === "number") openHub(event.detail);
  }) as EventListener);

  try {
    await listen<EngineFailure>("engine-failed", (event) => {
      showEngineFailure(event.payload);
    });
  } catch {
    // Running outside Tauri
  }

  try {
    await listen<number>("engine-ready", (event) => {
      openHub(event.payload);
    });
  } catch {
    // Running outside Tauri
  }

  try {
    type EngineStatusResponse =
      | { status: "Starting" }
      | { status: "Ready"; data: { port: number } }
      | { status: "Failed"; data: EngineFailure };

    const engineState = await invoke<EngineStatusResponse>("get_engine_status");
    if (engineState.status === "Ready" && engineState.data?.port) {
      openHub(engineState.data.port);
      return;
    }
    if (engineState.status === "Failed" && engineState.data) {
      showEngineFailure(engineState.data);
      return;
    }
  } catch {
    // Running outside Tauri or older backend
  }

  const initial = enginePort();
  if (await probe(initial)) {
    openHub(initial);
    return;
  }


  setStatus("Starting local camera engine…");
  startPolling();
}

void boot();
