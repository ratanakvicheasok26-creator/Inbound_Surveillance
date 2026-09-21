// Prevents additional console window on Windows in release unless explicitly requested via --console
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    app_lib::logger::attach_console_if_requested();
    app_lib::logger::install_panic_hook();
    app_lib::logger::log_info("MAIN", "Inbound Surveillance process entry point reached");
    app_lib::run();
}

