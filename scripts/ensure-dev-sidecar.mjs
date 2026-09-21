#!/usr/bin/env node
/**
 * Create a local inbound-engine sidecar so `tauri dev` can start.
 *
 * Production builds use the PyInstaller binary from `npm run sidecar`.
 * For day-to-day `npm run desktop:dev`, this writes a small wrapper that
 * execs the project venv (or python3) with absolute paths, so it still
 * works after Tauri copies the file into target/debug/.
 */
import { execSync } from "node:child_process";
import { chmodSync, closeSync, copyFileSync, existsSync, mkdirSync, openSync, readFileSync, readSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const binaries = path.join(repo, "src-tauri", "binaries");
function findPython() {
  const candidates = [
    path.join(repo, "edge", ".venv", "bin", "python"),
    path.join(repo, "edge", ".venv", "Scripts", "python.exe"),
    path.join(repo, "edge", ".venv", "python.exe"),
  ];
  for (const c of candidates) {
    if (existsSync(c)) return c;
  }
  return process.platform === "win32" ? "python" : "python3";
}
const python = findPython();

function syncLinuxDesktopIcons() {
  if (process.platform !== "linux") return;
  const home = process.env.HOME;
  if (!home) return;
  const iconsHicolor = path.join(home, ".local", "share", "icons", "hicolor");
  const appsDir = path.join(home, ".local", "share", "applications");
  const iconSrc = path.join(repo, "src-tauri", "icons");

  const sizeMap = {
    "32x32": "32x32.png",
    "64x64": "64x64.png",
    "128x128": "128x128.png",
    "256x256": "128x128@2x.png",
    "512x512": "icon.png",
  };

  try {
    for (const [dirName, srcFile] of Object.entries(sizeMap)) {
      const srcPath = path.join(iconSrc, srcFile);
      if (!existsSync(srcPath)) continue;
      const targetDir = path.join(iconsHicolor, dirName, "apps");
      mkdirSync(targetDir, { recursive: true });
      copyFileSync(srcPath, path.join(targetDir, "inbound-surveillance.png"));
    }

    mkdirSync(appsDir, { recursive: true });
    // Clean up duplicate old file if present
    const legacyDesktopFile = path.join(appsDir, "Inbound Surveillance.desktop");
    if (existsSync(legacyDesktopFile)) {
      try { import("node:fs").then(fs => fs.unlinkSync(legacyDesktopFile)); } catch (_) {}
    }

    try { execSync(`gtk-update-icon-cache -f -t ${JSON.stringify(iconsHicolor)}`, { stdio: "ignore" }); } catch (_) {}
    try { execSync(`update-desktop-database ${JSON.stringify(appsDir)}`, { stdio: "ignore" }); } catch (_) {}
  } catch (_) {}
}

function ensureIcons() {
  const possibleSources = [
    path.join(repo, "inb_surveillance-removebg-preview.png"),
    path.join(repo, "inb_surveillance.png"),
    path.join(repo, "inb_surveillance.jpg"),
    path.join(repo, "INB Surveillance.jpg"),
    path.join(repo, "INB Surveillance.png"),
    path.join(repo, "app-icon.png"),
  ];
  const source = possibleSources.find((p) => existsSync(p));
  if (!source) return;

  const inbSurveillance = path.join(repo, "INB Surveillance.jpg");
  const inbSurveillanceLower = path.join(repo, "inb_surveillance.jpg");
  if (existsSync(inbSurveillance) && !existsSync(inbSurveillanceLower)) {
    try { copyFileSync(inbSurveillance, inbSurveillanceLower); } catch (_) {}
  } else if (existsSync(inbSurveillanceLower) && !existsSync(inbSurveillance)) {
    try { copyFileSync(inbSurveillanceLower, inbSurveillance); } catch (_) {}
  }

  const iconPng = path.join(repo, "src-tauri", "icons", "icon.png");
  const sourceMtime = statSync(source).mtimeMs;
  const iconMtime = existsSync(iconPng) ? statSync(iconPng).mtimeMs : 0;

  if (!existsSync(iconPng) || sourceMtime > iconMtime) {
    console.log(`[icons] Detected launcher image: ${path.basename(source)}`);
    console.log(`[icons] Compiling Tauri icon bundle...`);
    try {
      execSync(`npx tauri icon ${JSON.stringify(source)}`, {
        cwd: repo,
        stdio: "inherit",
      });
      console.log(`[icons] Successfully compiled Tauri icons.`);
    } catch (err) {
      console.warn(`[icons] Warning: failed to run tauri icon: ${err.message}`);
    }
  }

  syncLinuxDesktopIcons();
}

function hostTriple() {
  const fromEnv = process.env.TAURI_ENV_TARGET_TRIPLE?.trim();
  if (fromEnv) return fromEnv;
  return execSync("rustc --print host-tuple", { encoding: "utf8" }).trim();
}

function isFrozenBinary(file) {
  if (!existsSync(file)) return false;
  try {
    const fd = openSync(file, "r");
    const buf = Buffer.alloc(4);
    readSync(fd, buf, 0, 4, 0);
    closeSync(fd);
    const elf = buf[0] === 0x7f && buf[1] === 0x45 && buf[2] === 0x4c && buf[3] === 0x46;
    const pe = buf[0] === 0x4d && buf[1] === 0x5a;
    const macho = buf[0] === 0xcf && buf[1] === 0xfa;
    return elf || pe || macho;
  } catch {
    return false;
  }
}

ensureIcons();

if (process.platform === "linux") {
  // Applied here for documentation; run-tauri.mjs and src-tauri/src/lib.rs
  // are what the WebKitGTK process actually inherits.
  process.env.WEBKIT_DISABLE_DMABUF_RENDERER ||= "1";
  process.env.WEBKIT_DISABLE_COMPOSITING_MODE ||= "1";
}

const launcher = path.join(repo, "edge", "launcher.py");
const isProduction = process.argv.includes("--require-frozen") || process.argv.includes("--production");

const triple = hostTriple();
const ext = process.platform === "win32" || triple.includes("windows") ? ".exe" : "";
const dest = path.join(binaries, `inbound-engine-${triple}${ext}`);

mkdirSync(binaries, { recursive: true });

function shouldRebuildSidecar(destFile) {
  if (process.argv.includes("--rebuild") || process.argv.includes("--force")) {
    return true;
  }
  if (!isFrozenBinary(destFile)) {
    return true;
  }
  try {
    const destMtime = statSync(destFile).mtimeMs;
    const watchPaths = [
      path.join(repo, "edge", "launcher.py"),
      path.join(repo, "edge", "db.py"),
      path.join(repo, "edge", "ai_auditor.py"),
      path.join(repo, "edge", "occupancy.py"),
      path.join(repo, "edge", "runtime.py"),
      path.join(repo, "edge", "hub.html"),
      path.join(repo, "edge", "static", "visits-reporting.js"),
      path.join(repo, "edge", "inbound-engine.spec"),
      path.join(repo, "edge", "build_sidecar.py"),
      path.join(repo, "edge", "static", "supabase.js"),
      path.join(repo, ".env"),
      path.join(repo, "edge", ".env"),
    ];
    for (const wp of watchPaths) {
      if (existsSync(wp) && statSync(wp).mtimeMs > destMtime) {
        console.log(`[build] Source file modified: ${path.relative(repo, wp)} (newer than frozen sidecar)`);
        return true;
      }
    }
  } catch (_) {
    return true;
  }
  return false;
}

const needsBuild = shouldRebuildSidecar(dest);

if (!needsBuild && isFrozenBinary(dest)) {
  console.log(`Using up-to-date frozen sidecar: ${dest}`);
  process.exit(0);
}

if (isProduction || needsBuild) {
  console.log(`[build] Standalone sidecar rebuild required.`);
  console.log(`[build] Building standalone sidecar using ${python} edge/build_sidecar.py...`);
  const buildSidecarPy = path.join(repo, "edge", "build_sidecar.py");
  try {
    execSync(`${JSON.stringify(python)} ${JSON.stringify(buildSidecarPy)} --target ${triple}`, {
      cwd: repo,
      stdio: "inherit",
    });
  } catch (err) {
    console.error(`\n[build] ERROR: Failed to build standalone sidecar binary: ${err.message}`);
    console.error(`[build] Please ensure 'pyinstaller' is installed in your Python environment:`);
    console.error(`        ${python} -m pip install pyinstaller\n`);
    process.exit(1);
  }

  if (!isFrozenBinary(dest)) {
    console.error(`\n[build] ERROR: Sidecar build did not produce a valid frozen binary at: ${dest}`);
    console.error(`[build] Aborting production desktop build to prevent shipping broken binaries.\n`);
    process.exit(1);
  }

  console.log(`[build] Successfully built frozen sidecar: ${dest}`);
  process.exit(0);
}

if (process.platform === "win32") {
  console.warn(`[dev] Notice: On Windows, please run 'npm run sidecar' to build the engine executable.`);
  process.exit(0);
}

const script = `#!/usr/bin/env bash
set -euo pipefail
exec ${JSON.stringify(python)} ${JSON.stringify(launcher)} "$@"
`;

writeFileSync(dest, script, { mode: 0o755 });
chmodSync(dest, 0o755);
console.log(`Dev sidecar wrapper -> ${dest}`);
console.log(`Engine: ${python} ${launcher}`);
