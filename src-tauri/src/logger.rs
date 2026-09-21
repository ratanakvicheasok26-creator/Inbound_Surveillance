use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

static LOGGER: OnceLock<DiagnosticLogger> = OnceLock::new();

pub struct DiagnosticLogger {
    log_paths: Vec<PathBuf>,
    file_handles: Mutex<Vec<File>>,
    desktop_report_path: PathBuf,
}

impl DiagnosticLogger {
    pub fn new() -> Self {
        let mut log_paths = Vec::new();
        let mut file_handles = Vec::new();

        // 1. Check if we can write next to the executable (portable log)
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let portable_log = parent.join("inbound-surveillance.log");
                if let Ok(file) = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&portable_log)
                {
                    log_paths.push(portable_log);
                    file_handles.push(file);
                }
            }
        }

        // 2. Standard AppData / User Local log path
        let app_data_log = get_appdata_log_path();
        if let Some(parent) = app_data_log.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(file) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&app_data_log)
        {
            if !log_paths.contains(&app_data_log) {
                log_paths.push(app_data_log);
                file_handles.push(file);
            }
        }

        // 3. Desktop crash report path
        let desktop_report_path = get_desktop_path().join("INBOUND_CRASH_REPORT.txt");

        Self {
            log_paths,
            file_handles: Mutex::new(file_handles),
            desktop_report_path,
        }
    }

    pub fn write_entry(&self, level: &str, tag: &str, message: &str) {
        let now = chrono_now_string();
        let formatted = format!("[{now}] [{level}] [{tag}] {message}\n");

        // Print to standard console if available
        print!("{formatted}");

        if let Ok(mut handles) = self.file_handles.lock() {
            for handle in handles.iter_mut() {
                let _ = handle.write_all(formatted.as_bytes());
                let _ = handle.flush();
            }
        }
    }

    pub fn primary_log_path(&self) -> PathBuf {
        self.log_paths
            .first()
            .cloned()
            .unwrap_or_else(|| get_appdata_log_path())
    }

    pub fn all_log_paths(&self) -> Vec<PathBuf> {
        self.log_paths.clone()
    }

    pub fn desktop_report_path(&self) -> PathBuf {
        self.desktop_report_path.clone()
    }
}

pub fn init() -> &'static DiagnosticLogger {
    LOGGER.get_or_init(|| {
        let logger = DiagnosticLogger::new();
        logger.write_entry("INFO", "BOOT", "=== Inbound Surveillance Diagnostic Logger Initialized ===");
        logger
    })
}

pub fn get() -> &'static DiagnosticLogger {
    init()
}

pub fn log_info(tag: &str, message: &str) {
    get().write_entry("INFO", tag, message);
}

pub fn log_warn(tag: &str, message: &str) {
    get().write_entry("WARN", tag, message);
}

pub fn log_error(tag: &str, message: &str) {
    get().write_entry("ERROR", tag, message);
}

fn chrono_now_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let millis = duration.subsec_millis();
    format!("{secs}.{millis:03}")
}

fn get_appdata_log_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(appdata) = std::env::var("APPDATA") {
            return PathBuf::from(appdata)
                .join("Inbound Surveillance")
                .join("logs")
                .join("startup.log");
        }
        if let Ok(localappdata) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(localappdata)
                .join("Inbound Surveillance")
                .join("logs")
                .join("startup.log");
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("Inbound Surveillance")
                .join("logs")
                .join("startup.log");
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("inbound-surveillance")
                .join("logs")
                .join("startup.log");
        }
    }
    std::env::temp_dir()
        .join("inbound-surveillance")
        .join("startup.log")
}

fn get_desktop_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(profile) = std::env::var("USERPROFILE") {
            let desktop = PathBuf::from(profile).join("Desktop");
            if desktop.exists() {
                return desktop;
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            let desktop = PathBuf::from(home).join("Desktop");
            if desktop.exists() {
                return desktop;
            }
        }
    }
    #[cfg(unix)]
    {
        if let Ok(home) = std::env::var("HOME") {
            let desktop = PathBuf::from(home).join("Desktop");
            if desktop.exists() {
                return desktop;
            }
        }
    }
    std::env::temp_dir()
}

pub fn log_system_diagnostics(app_handle: Option<&tauri::AppHandle>) {
    log_info("SYS", &format!("OS: {} ({})", std::env::consts::OS, std::env::consts::ARCH));
    log_info("SYS", &format!("Process ID: {}", std::process::id()));
    if let Ok(exe) = std::env::current_exe() {
        log_info("SYS", &format!("Executable Path: {}", exe.display()));
    }
    if let Ok(cwd) = std::env::current_dir() {
        log_info("SYS", &format!("Working Directory: {}", cwd.display()));
    }

    #[cfg(target_os = "windows")]
    {
        inspect_windows_environment();
    }

    if let Some(app) = app_handle {
        inspect_bundled_resources(app);
    }
}

#[cfg(target_os = "windows")]
fn inspect_windows_environment() {
    let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
    let system32 = std::path::Path::new(&system_root).join("System32");


    // 1. Check Visual C++ Runtime DLLs
    let vc_dlls = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"];
    let mut missing_vc = Vec::new();
    for dll in &vc_dlls {
        let p = system32.join(dll);
        if p.is_file() {
            log_info("SYS:VC", &format!("Found system DLL: {}", p.display()));
        } else {
            missing_vc.push(*dll);
            log_warn("SYS:VC", &format!("Missing system DLL: {}", p.display()));
        }
    }
    if missing_vc.is_empty() {
        log_info("SYS:VC", "Visual C++ 2015-2022 x64 runtime verified in System32");
    } else {
        log_warn("SYS:VC", &format!("Visual C++ runtime incomplete. Missing: {:?}", missing_vc));
    }

    // 2. Check WebView2 Runtime Installation
    inspect_webview2_installation();
}

#[cfg(target_os = "windows")]
fn inspect_webview2_installation() {
    let pf86 = std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".into());
    let pf = std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into());
    let wv_dirs = [
        PathBuf::from(&pf86).join("Microsoft").join("EdgeWebView").join("Application"),
        PathBuf::from(&pf).join("Microsoft").join("EdgeWebView").join("Application"),
    ];

    let mut found_version = None;
    for dir in &wv_dirs {
        if dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                        found_version = Some(name);
                        break;
                    }
                }
            }
        }
        if found_version.is_some() {
            break;
        }
    }

    if let Some(ver) = found_version {
        log_info("SYS:WEBVIEW2", &format!("Microsoft Edge WebView2 runtime detected: version {ver}"));
    } else {
        log_warn(
            "SYS:WEBVIEW2",
            "Microsoft Edge WebView2 runtime directory not found in standard Program Files locations.",
        );
    }
}

fn inspect_bundled_resources(app: &tauri::AppHandle) {
    use tauri::Manager;

    if let Ok(res_dir) = app.path().resource_dir() {
        log_info("RES", &format!("Resource Directory: {}", res_dir.display()));
    }

    let sidecar_candidates = [
        "inbound-engine",
        "inbound-engine-x86_64-pc-windows-msvc.exe",
        "inbound-engine-x86_64-unknown-linux-gnu",
        "inbound-engine.exe",
    ];
    for name in &sidecar_candidates {
        if let Ok(path) = app.path().resolve(name, tauri::path::BaseDirectory::Resource) {
            if path.is_file() {
                let size_mb = path.metadata().map(|m| m.len() as f64 / 1_048_576.0).unwrap_or(0.0);
                log_info("RES:SIDECAR", &format!("Found sidecar candidate {name} at {} ({size_mb:.1} MB)", path.display()));
            }
        }
    }
}

pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let location = info.location().map_or_else(
            || "unknown location".to_string(),
            |loc| format!("{}:{}:{}", loc.file(), loc.line(), loc.column()),
        );
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic payload".to_string()
        };

        let message = format!("PANIC at {location}: {payload}");
        log_error("PANIC", &message);

        write_emergency_crash_report(&message);

        #[cfg(target_os = "windows")]
        {
            show_native_message_box(
                "Inbound Surveillance - Fatal Crash",
                &format!(
                    "A fatal error occurred during startup:\n\n{message}\n\nA crash report has been saved to your Desktop:\nINBOUND_CRASH_REPORT.txt\n\nPlease share this file with support.",
                ),
            );
        }
    }));
}

pub fn write_emergency_crash_report(error_summary: &str) {
    let logger = get();
    let report_path = logger.desktop_report_path();

    let mut content = String::new();
    content.push_str("=================================================================\n");
    content.push_str("          INBOUND SURVEILLANCE - STARTUP CRASH REPORT           \n");
    content.push_str("=================================================================\n\n");
    content.push_str(&format!("Timestamp: {}\n", chrono_now_string()));
    content.push_str(&format!("Error Summary:\n{}\n\n", error_summary));

    content.push_str("-----------------------------------------------------------------\n");
    content.push_str("System Diagnostics:\n");
    content.push_str(&format!("OS: {} ({})\n", std::env::consts::OS, std::env::consts::ARCH));
    if let Ok(exe) = std::env::current_exe() {
        content.push_str(&format!("Exe Path: {}\n", exe.display()));
    }
    if let Ok(cwd) = std::env::current_dir() {
        content.push_str(&format!("Cwd: {}\n", cwd.display()));
    }
    content.push_str("\nActive Log Files:\n");
    for p in logger.all_log_paths() {
        content.push_str(&format!("  - {}\n", p.display()));
    }

    content.push_str("\n-----------------------------------------------------------------\n");
    content.push_str("Troubleshooting Recommendations:\n");
    content.push_str("1. Missing Visual C++ Runtime: Run the bundled vc_redist.x64.exe installer\n");
    content.push_str("   or install from: https://aka.ms/vs/17/release/vc_redist.x64.exe\n");
    content.push_str("2. Missing WebView2: Ensure Microsoft Edge WebView2 is installed from:\n");
    content.push_str("   https://developer.microsoft.com/en-us/microsoft-edge/webview2/\n");
    content.push_str("3. Run in Debug Mode: Double-click run-debug.bat or run with --console\n");
    content.push_str("   to see live diagnostic output in a command prompt terminal.\n");
    content.push_str("=================================================================\n");

    let _ = std::fs::write(&report_path, content);
    log_info("CRASH_REPORT", &format!("Wrote emergency crash report to {}", report_path.display()));
}

#[cfg(target_os = "windows")]
extern "system" {
    fn AllocConsole() -> i32;
    fn MessageBoxW(
        hwnd: *mut std::ffi::c_void,
        lp_text: *const u16,
        lp_caption: *const u16,
        u_type: u32,
    ) -> i32;
}

#[cfg(target_os = "windows")]
pub fn attach_console_if_requested() {
    let args: Vec<String> = std::env::args().collect();
    let requested = args.iter().any(|arg| arg == "--console" || arg == "--debug" || arg == "-d");
    let debug_file_present = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|dir| dir.join("debug.txt").exists()))
        .unwrap_or(false);

    if requested || debug_file_present {
        unsafe {
            if AllocConsole() != 0 {
                log_info("CONSOLE", "Allocated Windows Command Prompt console for live debugging.");
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn attach_console_if_requested() {}

#[cfg(target_os = "windows")]
pub fn show_native_message_box(title: &str, message: &str) {
    use std::os::windows::ffi::OsStrExt;
    let wide_title: Vec<u16> = std::ffi::OsStr::new(title)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let wide_msg: Vec<u16> = std::ffi::OsStr::new(message)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        // MB_OK (0x0) | MB_ICONERROR (0x10) | MB_SYSTEMMODAL (0x1000) = 0x1010
        MessageBoxW(std::ptr::null_mut(), wide_msg.as_ptr(), wide_title.as_ptr(), 0x1010);
    }
}

#[cfg(not(target_os = "windows"))]
pub fn show_native_message_box(_title: &str, _message: &str) {}
