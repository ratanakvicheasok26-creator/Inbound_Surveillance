import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const root = path.dirname(fileURLToPath(import.meta.url));
const tauriDebug = !!process.env.TAURI_ENV_DEBUG;

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  clearScreen: false,

  resolve: {
    alias: {
      "@": path.resolve(root, "src"),
    },
  },
  server: {
    port: 1420,
    strictPort: true,
    // Vite watches the repo root. edge/.venv (torch, protobuf headers, …)
    // has tens of thousands of files and blows the inotify watcher limit.
    watch: {
      ignored: [
        "**/.venv/**",
        "**/site-packages/**",
        "**/src-tauri/target/**",
        "**/dist-sidecar/**",
        "**/build/**",
        "**/*.pt",
        "**/*.onnx",
      ],
    },
  },
  build: {
    outDir: "dist",
    target: process.env.TAURI_ENV_PLATFORM == "windows" ? "chrome105" : "safari13",
    minify: !tauriDebug,
    sourcemap: tauriDebug,
    rollupOptions: {
      input: {
        main: path.resolve(root, "index.html"),
        dashboard: path.resolve(root, "dashboard.html"),
        desktop: path.resolve(root, "desktop.html"),
      },
    },
  },
});
