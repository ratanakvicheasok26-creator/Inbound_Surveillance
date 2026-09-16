use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

#[cfg(target_os = "windows")]
use std::sync::atomic::AtomicBool;

#[cfg(unix)]
use std::time::Duration;

use tauri::{Emitter, Manager, RunEvent};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

struct EngineProcess(Mutex<Option<CommandChild>>);

/// Sidecar PID kept outside Tauri managed state so we can still kill it
/// after `RunEvent::Exit` has already dropped app state.
static ENGINE_PID: AtomicU32 = AtomicU32::new(0);

#[cfg(target_os = "windows")]
static VC_REDIST_ATTEMPTED: AtomicBool = AtomicBool::new(false);

const ENGINE_PORT: &str = "8765";
const READY_MARKER: &str = "[INBOUND_SERVER_READY]";

#[derive(Clone, serde::Serialize)]
struct MissingRuntime {
    id: String,
    title: String,
    description: String,
    install_hint: String,
    install_command: String,
    can_install: bool,
}

#[derive(Clone, serde::Serialize)]
struct EngineFailure {
    exit_code: Option<i32>,
    error_summary: String,
    log_path: String,
    is_dll_error: bool,
    missing_runtime: Option<MissingRuntime>,
}

fn apply_linux_webkit_workarounds() {
    #[cfg(target_os = "linux")]
    {
        // Intel Iris Xe + WebKitGTK 2.42+/2.52: DMABUF/EGL compositing tears
        // frames and can take down WebKitWebProcess. Set only when unset so
        // operators can override from the environment.
        for (key, value) in [
            ("WEBKIT_DISABLE_DMABUF_RENDERER", "1"),
            ("WEBKIT_DISABLE_COMPOSITING_MODE", "1"),
        ] {
            if std::env::var_os(key).is_none() {
                #[allow(unused_unsafe)]
                unsafe {
                    std::env::set_var(key, value);
                }
            }
        }
    }
}

fn parse_ready_port(line: &str) -> Option<u16> {
    let idx = line.find(READY_MARKER)?;
    let rest = line[idx + READY_MARKER.len()..].trim();
    for part in rest.split_whitespace() {
        if let Some(value) = part.strip_prefix("port=") {
            return value.parse().ok();
        }
    }
    None
}

fn engine_log_path(app: &tauri::AppHandle) -> std::path::PathBuf {
    let dir = app
        .path()
        .app_log_dir()
        .or_else(|_| app.path().app_data_dir())
        .unwrap_or_else(|_| std::env::temp_dir().join("inbound-surveillance"));
    let _ = std::fs::create_dir_all(&dir);
    dir.join("engine.log")
}

fn append_to_engine_log(log_path: &std::path::Path, text: &str) {
    use std::io::Write;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let _ = writeln!(file, "{text}");
    }
}

fn check_is_dll_error(text: &str, exit_code: Option<i32>) -> bool {
    if let Some(code) = exit_code {
        if code == -1073741515 || code == 3221225781u32 as i32 {
            return true;
        }
    }
    let lower = text.to_lowercase();
    lower.contains("dll load failed")
        || lower.contains("specified module could not be found")
        || lower.contains("vcruntime")
        || lower.contains("msvcp")
        || lower.contains("c_extension")
        || lower.contains("torch_python")
        || lower.contains("fbgemm.dll")
        || lower.contains("import _c")
}

fn parse_missing_shared_library(text: &str) -> Option<String> {
    let markers = [
        "error while loading shared libraries: ",
        "importerror: ",
        "oserror: ",
    ];
    let lower = text.to_lowercase();
    let original = text;
    for marker in markers {
        if let Some(idx) = lower.find(marker) {
            let rest = original[idx + marker.len()..].trim();
            let name = rest
                .split(|c: char| c == ':' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .trim();
            if name.contains(".so") {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn linux_package_for(lib: &str) -> &'static str {
    let base = lib.split(".so").next().unwrap_or(lib);
    match base {
        "libgomp" => "libgomp1",
        "libGL" | "libOpenGL" | "libGLX" | "libEGL" => "libgl1",
        "libglib-2.0" | "libgthread-2.0" | "libgobject-2.0" => "libglib2.0-0",
        "libgtk-3" => "libgtk-3-0",
        "libwebkit2gtk-4.1" | "libwebkit2gtk-4.0" => "libwebkit2gtk-4.1-0",
        "libfuse" => "libfuse2",
        "libstdc++" => "libstdc++6",
        "libSM" => "libsm6",
        "libXext" => "libxext6",
        "libXrender" => "libxrender1",
        _ => "",
    }
}

fn classify_missing_runtime(text: &str, exit_code: Option<i32>) -> Option<MissingRuntime> {
    if check_is_dll_error(text, exit_code) {
        return Some(MissingRuntime {
            id: "vcredist".into(),
            title: "Missing Visual C++ runtime".into(),
            description: "The camera engine needs Microsoft Visual C++ 2015-2022 (x64). The installer that came with this app can install it for you. No extra download.".into(),
            install_hint: "Click Install now. Windows may ask for permission, then click Retry Engine.".into(),
            install_command: String::new(),
            can_install: true,
        });
    }

    if let Some(lib) = parse_missing_shared_library(text) {
        let pkg = linux_package_for(&lib);
        let install_command = if pkg.is_empty() {
            format!("sudo apt install {lib}")
        } else {
            format!("sudo apt install {pkg}")
        };
        let package_name = if pkg.is_empty() { lib.as_str() } else { pkg };
        return Some(MissingRuntime {
            id: "shared_library".into(),
            title: format!("Missing library: {lib}"),
            description: format!(
                "Linux is missing `{lib}` (`{package_name}`). Installing the .deb with apt pulls this in automatically; AppImage users can install just this package."
            ),
            install_hint: "Copy the command below, run it in a terminal, then click Retry Engine.".into(),
            install_command,
            can_install: false,
        });
    }

    None
}

fn engine_failure(
    exit_code: Option<i32>,
    error_summary: String,
    log_path: String,
) -> EngineFailure {
    let missing_runtime = classify_missing_runtime(&error_summary, exit_code);
    let is_dll_error = missing_runtime
        .as_ref()
        .map(|item| item.id == "vcredist")
        .unwrap_or(false);
    EngineFailure {
        exit_code,
        error_summary,
        log_path,
        is_dll_error,
        missing_runtime,
    }
}

#[cfg(target_os = "windows")]
fn bundled_vc_redist_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(path) = app
        .path()
        .resolve("vc_redist.x64.exe", tauri::path::BaseDirectory::Resource)
    {
        candidates.push(path);
    }
    if let Ok(path) = app.path().resolve(
        "resources/vc_redist.x64.exe",
        tauri::path::BaseDirectory::Resource,
    ) {
        candidates.push(path);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("resources").join("vc_redist.x64.exe"));
            candidates.push(dir.join("vc_redist.x64.exe"));
        }
    }
    candidates.into_iter().find(|path| path.is_file())
}

#[cfg(target_os = "windows")]
fn notify_boot_status(app: &tauri::AppHandle, text: &str) {
    if let Some(window) = app.get_webview_window("main") {
        let escaped = text
            .replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('\n', " ");
        let _ = window.eval(&format!(
            "const el=document.getElementById('status'); if(el) el.textContent='{escaped}';"
        ));
    }
}

#[cfg(target_os = "windows")]
fn vcredist_present() -> bool {
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
    let system32 = std::path::Path::new(&system_root).join("System32");
    ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"]
        .iter()
        .all(|name| system32.join(name).is_file())
}

#[cfg(target_os = "windows")]
fn run_bundled_vc_redist(app: &tauri::AppHandle) -> Result<String, String> {
    let path = bundled_vc_redist_path(app).ok_or_else(|| {
        "The Visual C++ installer was not bundled with this app. Reinstall using the Setup.exe from the latest build."
            .to_string()
    })?;
    let path_str = path.to_string_lossy().replace('\'', "''");
    let status = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &format!(
                "Start-Process -FilePath '{path_str}' -ArgumentList '/install /passive /norestart' -Verb RunAs -Wait"
            ),
        ])
        .status()
        .map_err(|err| format!("Could not start Visual C++ installer: {err}"))?;
    if status.success() {
        Ok("Visual C++ runtime installed. Click Retry Engine.".into())
    } else {
        Err(format!(
            "Visual C++ installer exited with {}. If you cancelled the Windows permission prompt, click Install now again.",
            status.code().unwrap_or(-1)
        ))
    }
}

#[cfg(target_os = "windows")]
fn ensure_windows_runtimes(app: &tauri::AppHandle) {
    if vcredist_present() {
        return;
    }
    if VC_REDIST_ATTEMPTED.swap(true, Ordering::SeqCst) {
        return;
    }
    log::info!("Visual C++ runtime missing; launching bundled installer");
    notify_boot_status(
        app,
        "Installing the missing Visual C++ runtime. Windows may ask for permission…",
    );
    match run_bundled_vc_redist(app) {
        Ok(msg) => {
            log::info!("{msg}");
            notify_boot_status(app, "Visual C++ runtime installed. Starting camera engine…");
        }
        Err(err) => {
            log::warn!("automatic VC++ install failed: {err}");
            notify_boot_status(
                app,
                "Could not auto-install Visual C++. Starting engine anyway…",
            );
        }
    }
}

fn notify_engine_ready(app: &tauri::AppHandle, port: u16) {
    let _ = app.emit("engine-ready", port);
    if let Some(window) = app.get_webview_window("main") {
        let script = format!(
            "window.__INBOUND_ENGINE_PORT__={port};\
             window.dispatchEvent(new CustomEvent('inbound-engine-ready',{{detail:{port}}}));\
             window.location.replace('http://127.0.0.1:{port}/');"
        );
        let _ = window.eval(&script);
    }
}

fn notify_engine_failed(app: &tauri::AppHandle, failure: EngineFailure) {
    let _ = app.emit("engine-failed", &failure);
    if let Some(window) = app.get_webview_window("main") {
        if let Ok(json) = serde_json::to_string(&failure) {
            let escaped = json.replace('\\', "\\\\").replace('\'', "\\'");
            let script = format!(
                "window.dispatchEvent(new CustomEvent('inbound-engine-failed', {{ detail: JSON.parse('{escaped}') }}));\
                 if (window.__onEngineFailed) {{ window.__onEngineFailed(JSON.parse('{escaped}')); }}"
            );
            let _ = window.eval(&script);
        }
    }
}

fn run_silent(program: &str, args: &[&str]) {
    let _ = std::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(unix)]
fn collect_matching_pids(pattern: &str) -> Vec<u32> {
    let output = std::process::Command::new("pgrep")
        .args(["-f", pattern])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

fn kill_pid_tree(pid: u32) {
    if pid == 0 {
        return;
    }
    log::info!("stopping inbound-engine process tree pid={pid}");
    #[cfg(windows)]
    {
        run_silent(
            "taskkill",
            &["/PID", &pid.to_string(), "/T", "/F"],
        );
    }
    #[cfg(unix)]
    {
        let pid_s = pid.to_string();
        // SIGTERM first so launcher.py can stop Telegram polling and cameras.
        run_silent("kill", &["-TERM", &pid_s]);
        run_silent("pkill", &["-TERM", "-P", &pid_s]);
        std::thread::sleep(Duration::from_millis(400));
        run_silent("kill", &["-KILL", &pid_s]);
        run_silent("pkill", &["-KILL", "-P", &pid_s]);
    }
}

fn kill_stale_engines() {
    #[cfg(windows)]
    {
        run_silent("taskkill", &["/IM", "inbound-engine.exe", "/F", "/T"]);
    }
    #[cfg(unix)]
    {
        let mut pids = collect_matching_pids("inbound-engine --port");
        pids.extend(collect_matching_pids("inbound-go2rtc"));
        pids.sort_unstable();
        pids.dedup();
        for pid in pids {
            if pid == std::process::id() {
                continue;
            }
            kill_pid_tree(pid);
        }
    }
}

fn kill_engine(app: &tauri::AppHandle) {
    let mut pid = ENGINE_PID.swap(0, Ordering::SeqCst);
    if let Some(state) = app.try_state::<EngineProcess>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(child) = guard.take() {
                pid = child.pid();
                // SIGTERM the tree before plugin kill (which is SIGKILL and
                // can orphan go2rtc + leave Telegram polling alive).
                kill_pid_tree(pid);
                let _ = child.kill();
                kill_stale_engines();
                return;
            }
        }
    }
    if pid != 0 {
        kill_pid_tree(pid);
    }
    kill_stale_engines();
}

fn spawn_engine(app: tauri::AppHandle) {
    let log_file = engine_log_path(&app);
    let log_path_str = log_file.to_string_lossy().to_string();
    kill_stale_engines();
    #[cfg(target_os = "windows")]
    ensure_windows_runtimes(&app);

    let command = match app.shell().sidecar("inbound-engine") {
        Ok(cmd) => cmd.args(["--port", ENGINE_PORT, "--no-browser"]),
        Err(err) => {
            log::warn!(
                "inbound-engine sidecar is not available ({err}); \
                 start `python edge/launcher.py --no-browser` for local development"
            );
            if cfg!(debug_assertions) {
                notify_engine_ready(&app, 8765);
            } else {
                notify_engine_failed(
                    &app,
                    engine_failure(
                        None,
                        format!("Sidecar executable not found: {err}"),
                        log_path_str,
                    ),
                );
            }
            return;
        }
    };

    match command.spawn() {
        Ok((mut rx, child)) => {
            let spawned_pid = child.pid();
            ENGINE_PID.store(spawned_pid, Ordering::SeqCst);
            if let Ok(mut guard) = app.state::<EngineProcess>().0.lock() {
                *guard = Some(child);
            }
            let log_file_cloned = log_file.clone();
            let log_path_str_cloned = log_path_str.clone();

            tauri::async_runtime::spawn(async move {
                let mut stdout_buf = String::new();
                let mut notified = false;
                let mut recent_lines: VecDeque<String> = VecDeque::with_capacity(60);

                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => {
                            let chunk = String::from_utf8_lossy(&bytes);
                            let trimmed = chunk.trim();
                            if !trimmed.is_empty() {
                                log::info!("[engine] {trimmed}");
                                for line in trimmed.lines() {
                                    if recent_lines.len() >= 50 {
                                        recent_lines.pop_front();
                                    }
                                    recent_lines.push_back(line.to_string());
                                    append_to_engine_log(&log_file_cloned, line);
                                }
                            }
                            stdout_buf.push_str(&chunk);
                            stdout_buf.push('\n');
                            if !notified {
                                if let Some(port) = parse_ready_port(&stdout_buf) {
                                    notified = true;
                                    notify_engine_ready(&app, port);
                                }
                            }
                        }
                        CommandEvent::Stderr(bytes) => {
                            let chunk = String::from_utf8_lossy(&bytes);
                            let trimmed = chunk.trim();
                            if !trimmed.is_empty() {
                                log::warn!("[engine:err] {trimmed}");
                                for line in trimmed.lines() {
                                    if recent_lines.len() >= 50 {
                                        recent_lines.pop_front();
                                    }
                                    recent_lines.push_back(line.to_string());
                                    append_to_engine_log(&log_file_cloned, &format!("[ERR] {line}"));
                                }
                            }
                            if !notified {
                                if let Some(port) = parse_ready_port(trimmed) {
                                    notified = true;
                                    notify_engine_ready(&app, port);
                                }
                            }
                        }
                        CommandEvent::Terminated(payload) => {
                            let _ = ENGINE_PID.compare_exchange(
                                spawned_pid,
                                0,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            );
                            log::info!("inbound-engine exited: {payload:?}");
                            if !notified {
                                let exit_code = payload.code;
                                let summary = if !recent_lines.is_empty() {
                                    recent_lines.iter().cloned().collect::<Vec<_>>().join("\n")
                                } else {
                                    format!("Process exited prematurely with exit code {exit_code:?}")
                                };
                                notify_engine_failed(
                                    &app,
                                    engine_failure(
                                        exit_code,
                                        summary,
                                        log_path_str_cloned.clone(),
                                    ),
                                );
                            }
                            break;
                        }
                        CommandEvent::Error(message) => {
                            log::error!("inbound-engine error: {message}");
                            append_to_engine_log(&log_file_cloned, &format!("[FATAL] {message}"));
                            if !notified {
                                notify_engine_failed(
                                    &app,
                                    engine_failure(
                                        None,
                                        format!("Engine error: {message}"),
                                        log_path_str_cloned.clone(),
                                    ),
                                );
                            }
                        }
                        _ => {}
                    }
                }
            });
        }
        Err(err) => {
            log::error!(
                "failed to spawn inbound-engine ({err}); \
                 start `python edge/launcher.py --no-browser` for local development"
            );
            if cfg!(debug_assertions) {
                notify_engine_ready(&app, 8765);
            } else {
                notify_engine_failed(
                    &app,
                    engine_failure(
                        None,
                        format!("Failed to spawn engine: {err}"),
                        log_path_str,
                    ),
                );
            }
        }
    }
}

#[tauri::command]
fn retry_engine(app: tauri::AppHandle) {
    kill_engine(&app);
    spawn_engine(app);
}

#[tauri::command]
fn install_bundled_runtime(app: tauri::AppHandle) -> Result<String, String> {
    #[cfg(not(target_os = "windows"))]
    {
        let _ = app;
        return Err(
            "This computer needs a system package. Copy the command on the boot screen and run it in a terminal."
                .into(),
        );
    }

    #[cfg(target_os = "windows")]
    {
        run_bundled_vc_redist(&app)
    }
}

#[tauri::command]
fn open_engine_log(app: tauri::AppHandle) -> Result<(), String> {
    let path = engine_log_path(&app);
    if let Some(parent) = path.parent() {
        #[cfg(target_os = "windows")]
        {
            let _ = std::process::Command::new("explorer").arg(parent).spawn();
        }
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("open").arg(parent).spawn();
        }
        #[cfg(target_os = "linux")]
        {
            let _ = std::process::Command::new("xdg-open").arg(parent).spawn();
        }
    }
    Ok(())
}

fn navigation_guard_plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("navigation-guard")
        .on_navigation(|_webview, url| {
            let scheme = url.scheme();
            if scheme == "tauri" || scheme == "data" || scheme == "about" {
                return true;
            }
            if (scheme == "http" || scheme == "https")
                && (url.host_str() == Some("127.0.0.1") || url.host_str() == Some("localhost"))
            {
                return true;
            }
            // Open external URLs and custom URI schemes with the system handler, never navigating the webview
            let target = url.as_str().to_string();
            #[cfg(target_os = "windows")]
            {
                let _ = std::process::Command::new("rundll32")
                    .args(["url.dll,FileProtocolHandler", &target])
                    .spawn();
            }
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("open").arg(&target).spawn();
            }
            #[cfg(target_os = "linux")]
            {
                let _ = std::process::Command::new("xdg-open").arg(&target).spawn();
            }
            false
        })
        .build()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    apply_linux_webkit_workarounds();
    tauri::Builder::default()
        .plugin(navigation_guard_plugin())
        .plugin(tauri_plugin_shell::init())
        .manage(EngineProcess(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            retry_engine,
            open_engine_log,
            install_bundled_runtime
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            if let Some(window) = app.get_webview_window("main") {
                let icon = tauri::include_image!("icons/128x128.png");
                let _ = window.set_icon(icon);
            }
            spawn_engine(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(
                event,
                tauri::WindowEvent::Destroyed | tauri::WindowEvent::CloseRequested { .. }
            ) {
                kill_engine(window.app_handle());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while running tauri application")
        .run(|app, event| {
            if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
                kill_engine(app);
            }
        });
}

