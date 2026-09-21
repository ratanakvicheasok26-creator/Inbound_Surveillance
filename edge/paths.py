"""Resolve bundled assets vs writable data for source and PyInstaller runs."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

INBOUND_APP_VERSION = "0.1.3"
INBOUND_BUILD_ID = os.environ.get("INBOUND_BUILD_ID", "").strip() or INBOUND_APP_VERSION


def _meipass() -> Path | None:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return None


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False) or _meipass() is not None)


def resource_dir() -> Path:
    frozen = _meipass()
    if frozen is not None:
        return frozen
    return Path(__file__).resolve().parent


def get_resource_path(relative_path: str) -> Path:
    """Locate a bundled file (model, hub.html, example config).

    PyInstaller extracts datas into ``sys._MEIPASS``. Source checkouts
    resolve relative to this module (the ``edge/`` directory).
    """
    return resource_dir() / relative_path


def data_dir() -> Path:
    """Writable config, SQLite, and proof stills.

    Frozen binaries cannot persist files inside the extract dir, so the
    platform application-data folder is used. Source checkouts keep files
    next to the Python modules. Override with ``INBOUND_DATA_DIR``.
    """
    override = os.environ.get("INBOUND_DATA_DIR", "").strip()
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    if is_frozen():
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
            path = base / "Inbound Surveillance"
        elif sys.platform == "darwin":
            path = Path.home() / "Library" / "Application Support" / "Inbound Surveillance"
        else:
            xdg = os.environ.get("XDG_DATA_HOME")
            base = Path(xdg) if xdg else Path.home() / ".local" / "share"
            path = base / "inbound-surveillance"
        path.mkdir(parents=True, exist_ok=True)
        return path

    path = Path(__file__).resolve().parent
    path.mkdir(parents=True, exist_ok=True)
    return path


from datetime import datetime


def diagnostic_log_paths() -> list[Path]:
    paths: list[Path] = []
    try:
        exe_dir = Path(sys.executable).parent
        paths.append(exe_dir / "inbound-surveillance.log")
    except Exception:
        pass

    try:
        dd = data_dir()
        paths.append(dd / "logs" / "startup.log")
        paths.append(dd / "startup.log")
    except Exception:
        pass
    return paths


def append_to_diagnostic_log(text: str) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for path in diagnostic_log_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{now_str}] [PYTHON] {text}\n")
        except Exception:
            pass


def log_boot_banner() -> None:
    """Print a one-line identity so Windows engine.log proves which build ran."""
    banner = (
        f"[INBOUND_BOOT] build={INBOUND_BUILD_ID} version={INBOUND_APP_VERSION} "
        f"platform={sys.platform} frozen={int(is_frozen())} "
        f"resource={resource_dir()} data={data_dir()} python={sys.executable}"
    )
    print(banner, flush=True)
    append_to_diagnostic_log(banner)


def fatal_boot(exc: BaseException) -> None:
    """Log a startup crash and exit. Used by the frozen Windows sidecar."""
    msg = f"[FATAL] Engine startup failed: {exc}"
    print(msg, file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    append_to_diagnostic_log(f"{msg}\n{tb}")

    # Also write to Desktop crash report if possible
    try:
        desktop_candidates = []
        if sys.platform == "win32":
            userprofile = os.environ.get("USERPROFILE")
            if userprofile:
                desktop_candidates.append(Path(userprofile) / "Desktop")
        desktop_candidates.append(Path.home() / "Desktop")
        for desk in desktop_candidates:
            if desk.exists():
                report = desk / "INBOUND_CRASH_REPORT.txt"
                with open(report, "a", encoding="utf-8", errors="replace") as f:
                    f.write(f"\n[CRASH REPORT FROM PYTHON ENGINE]\nTimestamp: {datetime.now()}\n{msg}\n{tb}\n")
                break
    except Exception:
        pass

    sys.exit(1)

