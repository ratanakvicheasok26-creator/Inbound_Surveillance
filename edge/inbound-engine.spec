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
    "workplaces.staff_memory",
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
    "audio_source",
    "complaint_auditor",
    "complaint_service",
    "speech_pipeline",
    "vad",
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

# Privacy and archive guardrails. The frozen app reads secrets, enrolled
# faces, and clip files from the writable data dir (AppData on Windows).
# Packing them into the one-file sidecar both leaks customer data and can
# push the CArchive past PyInstaller's 4 GiB limit.
# 1. NEVER package .env (Telegram tokens, Supabase secrets, API keys).
# 2. NEVER package edge/faces (enrolled photos; runtime seeds AppData).
# 3. NEVER package edge/videos (test clips; runtime loads AppData / camera).

static_dir = EDGE / "static"
if static_dir.is_dir():
    datas.append((str(static_dir), "static"))

# Active weights only. RTMPose is the default pose path; TinyPose and YOLO11
# ONNX stay because this branch still selects them via pose_engine / runtime.
PRODUCTION_MODELS = (
    "face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx",
    "osnet_x0_25_market1501.onnx",
    "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx",
    "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
    "yolo11n.onnx",
    "yolo11n-pose.onnx",
    "tinypose_256_192.onnx",
    "picodet_s_320_lcnet_pedestrian.onnx",
)
models = EDGE / "models"
if models.is_dir():
    for m in PRODUCTION_MODELS:
        onnx = models / m
        if onnx.is_file():
            datas.append((str(onnx), "models"))

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

# torchvision is unused. Blind collect_all("openvino") is not: rtmpose's
# Intel path calls core.read_model() on ONNX, which needs the ONNX frontend
# and the CPU plugin. Unused framework frontends and the GPU plugin are
# stripped below so they cannot blow the 4 GiB one-file CArchive.
pkgs_to_collect = ["ultralytics", "torch", "cv2", "PIL", "yaml", "requests", "onnxruntime", "rtmlib"]
try:
    import openvino  # noqa: F401
    pkgs_to_collect.append("openvino")
except Exception:
    pass
# Native audio libs only. Do not collect_all(silero_vad): it pulls torchaudio
# and, with CUDA Torch, nvidia/* wheels that blow past PyInstaller's 4 GiB
# one-file CArchive limit (struct.error: 'I' format requires <= 4294967295).
for optional_pkg in ("soundfile", "sounddevice"):
    try:
        __import__(optional_pkg)
        pkgs_to_collect.append(optional_pkg)
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
    excludes=[
        "tkinter",
        "matplotlib",
        "matplotlib.tests",
        "pytest",
        "IPython",
        "torchaudio",
        "torchvision",
        "triton",
        "polars",
        "_polars_runtime_32",
        "_polars_runtime_64",
        "_polars_runtime_compat",
        "scipy.spatial.tests",
        "scipy.sparse.tests",
    ],
    noarchive=False,
)

_BUNDLE_BLOAT = (
    "nvidia",
    "torchaudio",
    "torchvision",
    "triton",
    "libtorch_cuda",
    "cudnn",
    "cublas",
    "cusparse",
    "cusolver",
    "cufft",
    "curand",
    "nccl",
    "nvrtc",
    "nvjitlink",
    "nvtx",
    "nvshmem",
    "cusparselt",
    "cufile",
    "cuda_runtime",
    "cuda-cupti",
    "polars",
    "_polars_runtime",
    # Keep openvino_onnx_frontend and the CPU plugin. RTMPose reads ONNX
    # through OpenVINO on Intel; TinyPose weights stay in PRODUCTION_MODELS.
    "openvino_tensorflow_frontend",
    "openvino_paddle_frontend",
    "openvino_pytorch_frontend",
    "openvino_jax_frontend",
    "openvino_intel_gpu_plugin",
    "torch/bin/test",
    "torch/include",
    "torch/share",
)


def _keep_collected(item) -> bool:
    dest = ""
    src = ""
    if isinstance(item, (tuple, list)) and item:
        dest = str(item[0])
        src = str(item[1]) if len(item) > 1 else ""
    else:
        dest = str(getattr(item, "name", item) or "")
        src = str(getattr(item, "path", "") or "")
    text = f"{dest} {src}".replace("\\", "/").lower()
    return not any(marker in text for marker in _BUNDLE_BLOAT)


_before_binaries, _before_datas = len(a.binaries), len(a.datas)
a.binaries = [item for item in a.binaries if _keep_collected(item)]
a.datas = [item for item in a.datas if _keep_collected(item)]
print(
    f"Stripped CUDA/audio bloat: binaries {_before_binaries}->{len(a.binaries)} "
    f"datas {_before_datas}->{len(a.datas)}",
    flush=True,
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
