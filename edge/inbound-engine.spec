# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Inbound Surveillance sidecar binary.

The frozen executable is named ``inbound-engine``. ``build_sidecar.py``
renames it with the Tauri target triple and copies it to
``src-tauri/binaries/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.building.api import EXE, PYZ
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_all, collect_submodules

EDGE = Path(SPECPATH).resolve()

datas: list[tuple[str, str]] = []
binaries: list = []
hiddenimports = [
    "ultralytics",
    "torch",
    "torchvision",
    "cv2",
    "onnxruntime",
    "yaml",
    "requests",
    "requests.auth",
    "sqlite3",
    "PIL",
    "paths",
    "db",
    "occupancy",
    "workplaces",
    "workplaces.customer_visits",
    "graph",
    "graph.compile",
    "person",
    "face_id",
    "tracker",
    "reid",
    "runtime",
    "proof",
    "report",
    "roi_edit",
    "telegram_out",
    "capture",
    "adapters",
    "adapters.base",
    "adapters.webcam",
    "adapters.rtsp",
    "adapters.phone_http",
    "adapters.gateway",
    "adapters.onvif",
    "adapters.tapo",
    "adapters.webrtc",
    "discovery",
    "discovery.scanner",
    "zeroconf",
    "media",
    "media.client",
    "media.go2rtc",
    "sensors",
    "sensors.wifi_tracker",
    "ai_auditor",
    "telegram_link",
    "service_patterns",
    "vehicle",
    "bay_zoom",
    "liveness",
    "corroborate",
    "negatives",
    "one_euro",
    "tinypose",
    "rtmpose",
]


def _local_module_name(py: Path) -> str | None:
    rel = py.relative_to(EDGE)
    if rel.parts[0] in {"tools", "bin", "static", "faces", "models", "videos"}:
        return None
    if py.name.startswith("test_") or py.name == "build_sidecar.py":
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


# Always collect first-party modules from disk. A manual hiddenimports list
# is not enough: PyInstaller can still drop a sibling module (this shipped a
# Windows build where launcher imported ai_auditor but the PYZ lacked it).
for py in sorted(EDGE.rglob("*.py")):
    name = _local_module_name(py)
    if not name:
        continue
    if name not in hiddenimports:
        hiddenimports.append(name)
    dest = "." if py.parent == EDGE else py.relative_to(EDGE).parent.as_posix()
    datas.append((str(py), dest))

for name in (
    "yolo11n_improved.pt",
    "yolo11n-pose.pt",
    "yolo11n.pt",
    "yolo11n-pose.onnx",
    "yolo11n.onnx",
    "config.example.yaml",
    "hub.html",
    "inb_surveillance.png",
    "inb_surveillance-removebg-preview.png",
):
    src = EDGE / name
    if src.exists():
        datas.append((str(src), "."))
    repo_src = EDGE.parent / name
    if repo_src.exists() and (str(repo_src), ".") not in datas:
        datas.append((str(repo_src), "."))

for ov_dir in list(EDGE.glob("*_openvino_model")) + list(EDGE.parent.glob("*_openvino_model")):
    if ov_dir.is_dir():
        datas.append((str(ov_dir), ov_dir.name))

REPO = EDGE.parent
for env_candidate in (EDGE / ".env", REPO / ".env"):
    if env_candidate.is_file():
        datas.append((str(env_candidate), "."))
        break

faces = EDGE / "faces"
if faces.is_dir():
    datas.append((str(faces), "faces"))
static_dir = EDGE / "static"
if static_dir.is_dir():
    datas.append((str(static_dir), "static"))
models = EDGE / "models"
if models.is_dir():
    for onnx in models.glob("*.onnx"):
        datas.append((str(onnx), "models"))
videos = EDGE / "videos"
if videos.is_dir():
    datas.append((str(videos), "videos"))

go2rtc_name = "go2rtc.exe" if sys.platform == "win32" else "go2rtc"
go2rtc_bin = EDGE / "bin" / go2rtc_name
if go2rtc_bin.is_file():
    binaries.append((str(go2rtc_bin), "bin"))


def _msvc_runtime_binaries() -> list[tuple[str, str]]:
    """Ship the Visual C++ runtime next to the sidecar so Windows users
    do not need a separate system-wide redistributable for the camera engine.
    """
    if sys.platform != "win32":
        return []
    roots = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
        Path(sys.base_prefix),
        Path(sys.base_prefix) / "Library" / "bin",
    ]
    names = (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "msvcp140_atomic_wait.dll",
        "concrt140.dll",
        "vcomp140.dll",
    )
    found: list[tuple[str, str]] = []
    for name in names:
        for folder in roots:
            src = folder / name
            if src.is_file():
                found.append((str(src), "."))
                break
    return found


binaries += _msvc_runtime_binaries()

pkgs_to_collect = ["ultralytics", "torch", "torchvision", "cv2", "PIL", "yaml", "requests", "onnxruntime", "rtmlib"]
try:
    import openvino
    pkgs_to_collect.append("openvino")
except Exception:
    pass

for pkg in pkgs_to_collect:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    except Exception:
        try:
            pkg_hidden = collect_submodules(pkg)
        except Exception:
            pkg_hidden = []
        pkg_datas, pkg_binaries = [], []
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    [str(EDGE / "launcher.py")],
    pathex=[str(EDGE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "pytest", "IPython"],
    noarchive=False,
)

if sys.platform == "win32":
    bundled_names = set()
    for item in a.binaries:
        name = item[0] if isinstance(item, (tuple, list)) else getattr(item, "name", "")
        bundled_names.add(Path(str(name)).name.lower())
    extra = []
    for src, _dest in _msvc_runtime_binaries():
        dll = Path(src).name
        if dll.lower() not in bundled_names:
            extra.append((dll, src, "BINARY"))
    if extra:
        a.binaries += extra
        print("Forced MSVC runtime DLLs into sidecar:", [item[0] for item in extra], flush=True)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="inbound-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
