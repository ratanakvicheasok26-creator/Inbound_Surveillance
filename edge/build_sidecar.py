#!/usr/bin/env python3
"""Build the inbound-engine sidecar and install it for Tauri.

Usage (from the repository root or this directory):

    python edge/build_sidecar.py
    python edge/build_sidecar.py --target x86_64-unknown-linux-gnu

The resulting binary is copied to::

    src-tauri/binaries/inbound-engine-<target-triple>[.exe]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

EDGE = Path(__file__).resolve().parent
REPO = EDGE.parent
SPEC = EDGE / "inbound-engine.spec"
WEIGHTS = EDGE / "yolo11n-pose.pt"
VEHICLE_WEIGHTS = EDGE / "yolo11n.pt"
BINARIES = REPO / "src-tauri" / "binaries"


def _run(cmd: list[str], **kwargs) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def detect_target_triple() -> str:
    env = os.environ.get("TAURI_ENV_TARGET_TRIPLE", "").strip()
    if env:
        return env
    rustc = shutil.which("rustc")
    if rustc:
        try:
            out = subprocess.check_output(
                [rustc, "--print", "host-tuple"], text=True
            ).strip()
            if out:
                return out
        except (subprocess.CalledProcessError, OSError):
            pass
    # Fallback for hosts without rustc (should not happen in CI).
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc"
    if sys.platform == "darwin":
        import platform

        return (
            "aarch64-apple-darwin"
            if platform.machine() == "arm64"
            else "x86_64-apple-darwin"
        )
    return "x86_64-unknown-linux-gnu"


def ensure_weights() -> None:
    improved = EDGE / "yolo11n_improved.pt"
    if improved.exists() and improved.stat().st_size > 1_000_000:
        print(f"Using improved weights: {improved}", flush=True)

    for model_name, target_file in [
        ("yolo11n-pose.pt", WEIGHTS),
        ("yolo11n.pt", VEHICLE_WEIGHTS),
    ]:
        if target_file.exists() and target_file.stat().st_size > 1_000_000:
            print(f"Using existing weights: {target_file}", flush=True)
            continue
        print(f"Downloading {model_name} via ultralytics…", flush=True)
        from ultralytics import YOLO

        cwd = os.getcwd()
        os.chdir(EDGE)
        try:
            YOLO(model_name)
        finally:
            os.chdir(cwd)
        if not target_file.exists():
            fallback = Path.cwd() / model_name
            if fallback.exists():
                shutil.copy2(fallback, target_file)
        if not target_file.exists():
            raise SystemExit(
                f"{model_name} was not downloaded. Place the weights in edge/ and retry."
            )


def ensure_exported_weights(weights_dir: Path | None = None, force_reexport: bool = False) -> None:
    """Exports YOLO models strictly at 640x640 (pose) and 512x512 (vehicles).
    Guarantees proposal grid reduction from 18,900 down to 8,400 boxes.
    """
    from ultralytics import YOLO

    if weights_dir is None:
        weights_dir = EDGE
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pose Model: Strict 640x640
    pose_pt = weights_dir / "yolo11n-pose.pt"
    if not pose_pt.exists() and (EDGE / "yolo11n-pose.pt").exists():
        pose_pt = EDGE / "yolo11n-pose.pt"
    pose_onnx = weights_dir / "yolo11n-pose.onnx"
    edge_pose_onnx = EDGE / "yolo11n-pose.onnx"
    if pose_pt.exists() and (not pose_onnx.exists() or not edge_pose_onnx.exists() or force_reexport):
        print("[BUILD] Exporting yolo11n-pose to ONNX/OpenVINO at 640x640...", flush=True)
        model = YOLO(str(pose_pt))
        model.export(
            format="onnx",
            imgsz=640,
            half=False,
            dynamic=False,
            simplify=True,
        )
        if pose_pt.parent != weights_dir and (pose_pt.with_suffix(".onnx")).exists():
            shutil.copy2(pose_pt.with_suffix(".onnx"), pose_onnx)
        if pose_pt.parent != EDGE and (pose_pt.with_suffix(".onnx")).exists():
            shutil.copy2(pose_pt.with_suffix(".onnx"), edge_pose_onnx)
        try:
            import openvino

            model.export(
                format="openvino",
                imgsz=640,
                half=True,
                dynamic=False,
            )
            print("[BUILD] Exported yolo11n-pose to OpenVINO FP16 (640x640)", flush=True)
        except ImportError:
            print("[BUILD] OpenVINO not installed; ONNX export retained.", flush=True)
        except Exception as ex:
            print(f"[BUILD] OpenVINO export failed for yolo11n-pose: {ex}", flush=True)

    # 2. Vehicle Model: Strict 512x512
    veh_pt = weights_dir / "yolo11n.pt"
    if not veh_pt.exists() and (EDGE / "yolo11n.pt").exists():
        veh_pt = EDGE / "yolo11n.pt"
    veh_onnx = weights_dir / "yolo11n.onnx"
    edge_veh_onnx = EDGE / "yolo11n.onnx"
    if veh_pt.exists() and (not veh_onnx.exists() or not edge_veh_onnx.exists() or force_reexport):
        print("[BUILD] Exporting yolo11n to ONNX/OpenVINO at 512x512...", flush=True)
        model = YOLO(str(veh_pt))
        model.export(
            format="onnx",
            imgsz=512,
            half=False,
            dynamic=False,
            simplify=True,
        )
        if veh_pt.parent != weights_dir and (veh_pt.with_suffix(".onnx")).exists():
            shutil.copy2(veh_pt.with_suffix(".onnx"), veh_onnx)
        if veh_pt.parent != EDGE and (veh_pt.with_suffix(".onnx")).exists():
            shutil.copy2(veh_pt.with_suffix(".onnx"), edge_veh_onnx)
        try:
            import openvino

            model.export(
                format="openvino",
                imgsz=512,
                half=True,
                dynamic=False,
            )
            print("[BUILD] Exported yolo11n to OpenVINO FP16 (512x512)", flush=True)
        except ImportError:
            pass
        except Exception as ex:
            print(f"[BUILD] OpenVINO export failed for yolo11n: {ex}", flush=True)


def ensure_osnet() -> None:
    """Ensure OSNet person re-identification ONNX model is available."""
    models_dir = EDGE / "models"
    try:
        from reid import ensure_reid_model

        path = ensure_reid_model(models_dir, download=True)
        if path and path.exists():
            print(f"Bundled OSNet ReID model: {path}", flush=True)
    except Exception as ex:
        print(f"OSNet ReID check skipped: {ex}", flush=True)


def exe_name() -> str:
    return "inbound-engine.exe" if sys.platform == "win32" else "inbound-engine"


REQUIRED_PYZ_MODULES = (
    "ai_auditor",
    "occupancy",
    "corroborate",
    "telegram_link",
    "vehicle",
    "paths",
    "db",
    "face_id",
    "tinypose",
    "rtmpose",
    "one_euro",
    "adapters.video_file",
    "workplaces",
    "workplaces.customer_visits",
    "graph",
    "graph.compile",
)


def _assert_modules_bundled(work: Path) -> None:
    """Fail the sidecar build if PyInstaller dropped a first-party module."""
    engine_dir = work / "inbound-engine"
    toc_blobs: list[str] = []
    for name in ("PYZ-00.toc", "PKG-00.toc", "Analysis-00.toc"):
        path = engine_dir / name
        if path.is_file():
            toc_blobs.append(path.read_text(encoding="utf-8", errors="replace"))
    blob = "\n".join(toc_blobs)
    if not blob:
        raise SystemExit(f"PyInstaller did not write analysis files under {engine_dir}")

    missing: list[str] = []
    for module in REQUIRED_PYZ_MODULES:
        as_py = module.replace(".", "/") + ".py"
        if (
            f"'{module}'" not in blob
            and f'"{module}"' not in blob
            and as_py not in blob
            and module.replace(".", os.sep) + ".py" not in blob
        ):
            missing.append(module)
    if missing:
        raise SystemExit(
            "PyInstaller bundle is missing required engine modules: "
            + ", ".join(missing)
        )
    print("Verified first-party modules in sidecar bundle.", flush=True)


def sidecar_name(target: str) -> str:
    ext = ".exe" if sys.platform == "win32" or target.endswith("windows-msvc") else ""
    return f"inbound-engine-{target}{ext}"


def go2rtc_sidecar_name(target: str) -> str:
    ext = ".exe" if sys.platform == "win32" or target.endswith("windows-msvc") else ""
    return f"go2rtc-{target}{ext}"


def ensure_go2rtc() -> Path:
    """Download go2rtc into edge/bin/ so PyInstaller and Tauri can bundle it."""
    sys.path.insert(0, str(EDGE))
    from media.go2rtc import binary_filename, ensure_binary

    dest = EDGE / "bin" / binary_filename()
    path = ensure_binary(dest)
    print(f"go2rtc binary: {path}", flush=True)
    return path


def verify_dry_run() -> bool:
    """Validate all required models, runtime libraries, and modules before packaging."""
    print("=== Inbound Sidecar Build: Dry-Run Inspection ===", flush=True)
    all_ok = True

    # 1. Models & Resolution Lock
    pose_onnx = EDGE / "yolo11n-pose.onnx"
    veh_onnx = EDGE / "yolo11n.onnx"
    osnet_onnx = EDGE / "models" / "osnet_x0_25_market1501.onnx"

    import onnxruntime as ort

    print("\n[1/4] Verifying models & resolution lock...")
    if not pose_onnx.exists():
        print(f"  FAIL: Missing {pose_onnx}", flush=True)
        all_ok = False
    else:
        try:
            sess = ort.InferenceSession(str(pose_onnx), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            if list(shape[-2:]) == [640, 640]:
                print(f"  PASS: yolo11n-pose.onnx locked at 640x640 (shape: {shape})", flush=True)
            else:
                print(f"  FAIL: yolo11n-pose.onnx shape is {shape}, expected [..., 640, 640]", flush=True)
                all_ok = False
        except Exception as ex:
            print(f"  FAIL: Error inspecting yolo11n-pose.onnx: {ex}", flush=True)
            all_ok = False

    if not veh_onnx.exists():
        print(f"  FAIL: Missing {veh_onnx}", flush=True)
        all_ok = False
    else:
        try:
            sess = ort.InferenceSession(str(veh_onnx), providers=["CPUExecutionProvider"])
            shape = sess.get_inputs()[0].shape
            if list(shape[-2:]) == [512, 512]:
                print(f"  PASS: yolo11n.onnx locked at 512x512 (shape: {shape})", flush=True)
            else:
                print(f"  FAIL: yolo11n.onnx shape is {shape}, expected [..., 512, 512]", flush=True)
                all_ok = False
        except Exception as ex:
            print(f"  FAIL: Error inspecting yolo11n.onnx: {ex}", flush=True)
            all_ok = False

    if not osnet_onnx.exists():
        print(f"  FAIL: Missing OSNet ReID model at {osnet_onnx}", flush=True)
        all_ok = False
    else:
        print(f"  PASS: OSNet ReID model present ({osnet_onnx.name}, {osnet_onnx.stat().st_size // 1024} KB)", flush=True)

    # 2. First-party modules
    print("\n[2/4] Verifying required modules...")
    modules_to_check = (
        "one_euro",
        "tracker",
        "reid",
        "occupancy",
        "corroborate",
        "ai_auditor",
        "person",
        "tinypose",
        "rtmpose",
        "vehicle",
        "bay_zoom",
        "runtime",
    )
    for mod in modules_to_check:
        mod_path = EDGE / f"{mod}.py"
        if mod_path.exists():
            print(f"  PASS: Module {mod}.py present", flush=True)
        else:
            print(f"  FAIL: Module {mod}.py missing at {mod_path}", flush=True)
            all_ok = False

    # 3. Runtime execution providers
    print("\n[3/4] Verifying runtime libraries...")
    providers = ort.get_available_providers()
    print(f"  PASS: ONNX Runtime available (providers: {providers})", flush=True)

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"  PASS: OpenVINO runtime available (devices: {devices})", flush=True)
    except Exception as ex:
        print(f"  INFO: OpenVINO runtime not loaded: {ex}", flush=True)

    # 4. PyInstaller environment
    print("\n[4/4] Verifying packaging spec & PyInstaller...")
    if not SPEC.exists():
        print(f"  FAIL: Spec file missing at {SPEC}", flush=True)
        all_ok = False
    else:
        print(f"  PASS: Spec file present ({SPEC.name})", flush=True)

    try:
        import PyInstaller
        print(f"  PASS: PyInstaller available (version: {PyInstaller.__version__})", flush=True)
    except ImportError:
        print("  FAIL: PyInstaller not installed in environment", flush=True)
        all_ok = False

    print("\n=== Dry-Run Inspection Result ===", flush=True)
    if all_ok:
        print("All required models, modules, and packaging dependencies verified successfully!\n", flush=True)
    else:
        print("Dry-run inspection failed with missing or invalid dependencies!\n", flush=True)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the inbound-engine Tauri sidecar")
    parser.add_argument(
        "--target",
        default="",
        help="Rust target triple (default: rustc --print host-tuple)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download YOLO weights if missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate models, runtime libraries, and modules without running full PyInstaller build",
    )
    args = parser.parse_args()

    if args.dry_run:
        ok = verify_dry_run()
        sys.exit(0 if ok else 1)

    if not args.skip_download:
        ensure_weights()
        ensure_exported_weights()
        ensure_osnet()
    elif not WEIGHTS.exists():
        print("WARNING: edge/yolo11n-pose.pt is missing; the sidecar will download at runtime.", flush=True)

    go2rtc_path = ensure_go2rtc()

    dist = REPO / "dist-sidecar"
    work = REPO / "build" / "pyinstaller"
    dist.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--distpath",
            str(dist),
            "--workpath",
            str(work),
            str(SPEC),
        ],
        cwd=str(REPO),
    )

    _assert_modules_bundled(work)

    built = dist / exe_name()
    if not built.exists():
        raise SystemExit(f"PyInstaller did not produce {built}")

    target = args.target.strip() or detect_target_triple()
    BINARIES.mkdir(parents=True, exist_ok=True)
    dest = BINARIES / sidecar_name(target)
    shutil.copy2(built, dest)
    try:
        dest.chmod(dest.stat().st_mode | 0o111)
    except OSError:
        pass
    print(f"Sidecar installed: {dest}", flush=True)

    go2rtc_dest = BINARIES / go2rtc_sidecar_name(target)
    shutil.copy2(go2rtc_path, go2rtc_dest)
    try:
        go2rtc_dest.chmod(go2rtc_dest.stat().st_mode | 0o111)
    except OSError:
        pass
    print(f"go2rtc installed: {go2rtc_dest}", flush=True)


if __name__ == "__main__":
    main()
