/**
 * Download the Microsoft Visual C++ 2015–2022 Redistributable (x64)
 * into src-tauri/resources so the Windows installer can install it for the user.
 *
 * Safe to re-run: skips download when a large enough copy already exists.
 */
import fs from "node:fs";
import https from "node:https";
import path from "node:path";
import { fileURLToPath } from "node:url";

const VC_REDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe";
const MIN_BYTES = 8 * 1024 * 1024;

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const destDir = path.join(repo, "src-tauri", "resources");
const dest = path.join(destDir, "vc_redist.x64.exe");

function download(url, file, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 8) {
      reject(new Error(`Too many redirects fetching ${url}`));
      return;
    }
    https
      .get(url, (res) => {
        const status = res.statusCode ?? 0;
        const location = res.headers.location;
        if (status >= 300 && status < 400 && location) {
          res.resume();
          download(location, file, redirects + 1).then(resolve, reject);
          return;
        }
        if (status !== 200) {
          res.resume();
          reject(new Error(`Failed to download VC++ redistributable: HTTP ${status}`));
          return;
        }
        const stream = fs.createWriteStream(file);
        res.pipe(stream);
        stream.on("finish", () => stream.close(() => resolve(file)));
        stream.on("error", reject);
      })
      .on("error", reject);
  });
}

fs.mkdirSync(destDir, { recursive: true });

if (fs.existsSync(dest) && fs.statSync(dest).size >= MIN_BYTES) {
  console.log(`Using existing Visual C++ Redistributable: ${dest}`);
  process.exit(0);
}

console.log(`Downloading Visual C++ Redistributable to ${dest}`);
const tmp = `${dest}.partial`;
try {
  await download(VC_REDIST_URL, tmp);
  const size = fs.statSync(tmp).size;
  if (size < MIN_BYTES) {
    throw new Error(`Downloaded vc_redist.x64.exe is too small (${size} bytes)`);
  }
  fs.renameSync(tmp, dest);
  console.log(`Saved ${dest} (${size} bytes)`);
} catch (err) {
  try {
    fs.unlinkSync(tmp);
  } catch {
    // ignore cleanup errors
  }
  console.error(err instanceof Error ? err.message : err);
  process.exit(1);
}
