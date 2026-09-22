#!/usr/bin/env node
/**
 * Launch `tauri` with Linux WebKitGTK workarounds in the parent environment.
 * src-tauri/src/lib.rs also sets these before the webview is created.
 */
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

if (process.platform === "linux") {
  process.env.WEBKIT_DISABLE_DMABUF_RENDERER ||= "1";
  process.env.WEBKIT_DISABLE_COMPOSITING_MODE ||= "1";
}

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const binName = process.platform === "win32" ? "tauri.cmd" : "tauri";
const bin = path.join(repo, "node_modules", ".bin", binName);
const args = process.argv.slice(2);
const child = spawn(bin, args, {
  stdio: "inherit",
  env: process.env,
  shell: process.platform === "win32",
  cwd: repo,
});
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 1);
});
