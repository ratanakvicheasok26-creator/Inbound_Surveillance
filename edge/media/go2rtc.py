"""Lifecycle manager for an embedded go2rtc sidecar process."""

from __future__ import annotations

import atexit
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from media.client import Go2RtcClient
from paths import data_dir, is_frozen, resource_dir

GO2RTC_VERSION = "1.9.14"
GITHUB_RELEASE = f"https://github.com/AlexxIT/go2rtc/releases/download/v{GO2RTC_VERSION}"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 1984
DEFAULT_RTSP_PORT = 8554
DEFAULT_WEBRTC_PORT = 8555

_STREAM_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")

# GitHub asset name + whether the download is a zip archive.
_RELEASE_ASSETS: dict[str, tuple[str, bool]] = {
    "linux-x86_64": ("go2rtc_linux_amd64", False),
    "linux-aarch64": ("go2rtc_linux_arm64", False),
    "linux-arm64": ("go2rtc_linux_arm64", False),
    "windows-x86_64": ("go2rtc_win64.zip", True),
    "windows-arm64": ("go2rtc_win_arm64.zip", True),
    "darwin-arm64": ("go2rtc_mac_arm64.zip", True),
    "darwin-x86_64": ("go2rtc_mac_amd64.zip", True),
}


def platform_tag() -> str:
    """Return ``linux-x86_64``, ``windows-x86_64``, ``darwin-arm64``, …"""
    system = sys.platform
    machine = platform.machine().lower()
    if system.startswith("linux"):
        os_name = "linux"
    elif system == "win32":
        os_name = "windows"
    elif system == "darwin":
        os_name = "darwin"
    else:
        os_name = system
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "arm64" if os_name == "darwin" else "aarch64"
    elif machine in ("i386", "i686", "x86"):
        arch = "x86"
    else:
        arch = machine
    return f"{os_name}-{arch}"


def binary_filename() -> str:
    return "go2rtc.exe" if sys.platform == "win32" else "go2rtc"


def sanitize_stream_id(value: str | None) -> str:
    text = _STREAM_ID_RE.sub("-", str(value or "live")).strip("-")
    return text or "live"


def tauri_sidecar_name(target_triple: str) -> str:
    ext = ".exe" if sys.platform == "win32" or "windows" in target_triple else ""
    return f"go2rtc-{target_triple}{ext}"


def _chmod_exec(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | 0o111)
    except OSError:
        pass


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return path.stat().st_size > 100_000


def _writable_bin_dir() -> Path:
    if is_frozen():
        return data_dir() / "bin"
    return resource_dir() / "bin"


def candidate_binary_paths() -> list[Path]:
    """Search order for a local go2rtc binary (dev, frozen, Tauri)."""
    name = binary_filename()
    seen: set[Path] = set()
    out: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(path)

    res = resource_dir()
    add(res / "bin" / name)
    add(res / name)
    add(_writable_bin_dir() / name)

    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        add(exe_dir / name)
        add(exe_dir / "bin" / name)
    else:
        binaries = res.parent / "src-tauri" / "binaries"
        add(binaries / name)
        tag = platform_tag()
        add(binaries / f"go2rtc-{tag}")
        add(binaries / f"go2rtc-{tag}.exe")
        triple = os.environ.get("TAURI_ENV_TARGET_TRIPLE", "").strip()
        if triple:
            add(binaries / tauri_sidecar_name(triple))
        if binaries.is_dir():
            for extra in binaries.glob("go2rtc-*"):
                add(extra)

    which = shutil.which("go2rtc") or shutil.which("go2rtc.exe")
    if which:
        add(Path(which))
    return out


def find_local_binary() -> Path | None:
    for path in candidate_binary_paths():
        if _is_executable_file(path):
            return path
    return None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    headers = {"User-Agent": "InboundSurveillance/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)
    tmp.replace(dest)


def _extract_from_zip(archive: Path, dest: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        pick = None
        for name in names:
            base = name.rstrip("/").rsplit("/", 1)[-1]
            if base.startswith("go2rtc") and not name.endswith("/"):
                pick = name
                break
        if pick is None:
            files = [n for n in names if not n.endswith("/")]
            if not files:
                raise RuntimeError(f"go2rtc zip {archive} is empty")
            pick = files[0]
        dest.write_bytes(zf.read(pick))


def download_binary(dest: Path | None = None) -> Path:
    """Fetch the official go2rtc release for this OS/arch into ``edge/bin/``."""
    tag = platform_tag()
    asset = _RELEASE_ASSETS.get(tag)
    if asset is None:
        raise RuntimeError(
            f"No go2rtc release asset for platform '{tag}'. "
            "Place a binary at edge/bin/go2rtc."
        )
    filename, is_zip = asset
    if dest is None:
        dest = _writable_bin_dir() / binary_filename()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GITHUB_RELEASE}/{filename}"
    print(f"[go2rtc] downloading {url}", flush=True)
    if is_zip:
        with tempfile.TemporaryDirectory(prefix="go2rtc-dl-") as tmp:
            archive = Path(tmp) / filename
            _download(url, archive)
            _extract_from_zip(archive, dest)
    else:
        _download(url, dest)
    _chmod_exec(dest)
    if not _is_executable_file(dest):
        raise RuntimeError(f"Downloaded go2rtc binary looks invalid: {dest}")
    return dest


def ensure_binary(dest: Path | None = None) -> Path:
    """Return a usable go2rtc path, downloading the official release if needed."""
    found = find_local_binary()
    if found is not None:
        _chmod_exec(found)
        if dest is not None:
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if found.resolve() != dest.resolve():
                shutil.copy2(found, dest)
                _chmod_exec(dest)
                return dest
        return found
    return download_binary(dest)


def _kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F", "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _is_runnable(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    if sys.platform == "win32":
        return True
    return os.access(path, os.X_OK)


def ffmpeg_candidate_paths() -> list[str]:
    """Ordered ffmpeg lookup. Unix paths are skipped on Windows and vice versa."""
    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    candidates: list[str | None] = [
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe") if sys.platform == "win32" else None,
    ]
    try:
        res = resource_dir()
        data = data_dir()
        candidates.extend(
            [
                str(res / "bin" / ffmpeg_name),
                str(res / ffmpeg_name),
                str(data / "bin" / ffmpeg_name),
            ]
        )
    except Exception:
        pass
    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local = os.environ.get("LOCALAPPDATA", "")
        candidates.extend(
            [
                str(Path(pf) / "ffmpeg" / "bin" / "ffmpeg.exe"),
                str(Path(pf86) / "ffmpeg" / "bin" / "ffmpeg.exe"),
            ]
        )
        if local:
            candidates.append(str(Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"))
        candidates.append(str(Path.home() / "scoop" / "shims" / "ffmpeg.exe"))
    else:
        candidates.extend(
            [
                str(Path.home() / ".local" / "bin" / "ffmpeg"),
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
            ]
        )
    out: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        out.append(cand)
    return out


def find_ffmpeg_binary() -> str | None:
    """Find a usable ffmpeg executable for go2rtc transcoding and frame capture."""
    for cand in ffmpeg_candidate_paths():
        if _is_runnable(cand):
            return cand
    return None


def yaml_quoted_path(path: str | Path) -> str:
    """Slash-normalize a filesystem path for YAML and go2rtc configs."""
    return str(path).replace("\\", "/")


_CONFIG_TEMPLATE = """\
api:
  listen: "{host}:{api_port}"
  origin: "*"
rtsp:
  listen: "{host}:{rtsp_port}"
webrtc:
  listen: "{host}:{webrtc_port}"
log:
  level: info
{ffmpeg_section}"""


class Go2RtcManager:
    """Spawn go2rtc on isolated localhost ports and tear it down cleanly."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        api_port: int = DEFAULT_API_PORT,
        rtsp_port: int = DEFAULT_RTSP_PORT,
        webrtc_port: int = DEFAULT_WEBRTC_PORT,
        state_dir: Path | None = None,
        binary: Path | None = None,
    ):
        self.host = host
        self.api_port = int(api_port)
        self.rtsp_port = int(rtsp_port)
        self.webrtc_port = int(webrtc_port)
        self._state_dir = Path(state_dir) if state_dir else None
        self._binary = Path(binary) if binary else None
        self._proc: subprocess.Popen | None = None
        self._owned = False
        self._config_path: Path | None = None
        self._work_dir: Path | None = None
        self._log_file: Any = None
        self._client: Go2RtcClient | None = None
        self._atexit_registered = False

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.api_port}"

    @property
    def client(self) -> Go2RtcClient:
        if self._client is None:
            self._client = Go2RtcClient(self.api_base)
        return self._client

    @property
    def pid(self) -> int | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        return None

    def is_running(self) -> bool:
        return self.pid is not None or self.is_ready()

    def is_ready(self) -> bool:
        return self.client.is_ready()

    def rtsp_url(self, stream_id: str) -> str:
        return f"rtsp://{self.host}:{self.rtsp_port}/{sanitize_stream_id(stream_id)}"

    def mjpeg_url(self, stream_id: str) -> str:
        sid = sanitize_stream_id(stream_id)
        return f"{self.api_base}/api/stream.mjpeg?src={quote(sid, safe='')}"

    def ws_url(self, stream_id: str) -> str:
        return self.client.ws_url(sanitize_stream_id(stream_id))

    def consumer_url(self, stream_id: str, source: Any) -> str:
        """Local URL the Python grabber should open for this upstream source."""
        from adapters.base import protocol_from_source

        sid = sanitize_stream_id(stream_id)
        if protocol_from_source(source) == "phone":
            return self.mjpeg_url(sid)
        return self.rtsp_url(sid)

    def status(self, stream_id: str | None = None) -> dict[str, Any]:
        ready = self.is_ready()
        sid = sanitize_stream_id(stream_id) if stream_id else None
        payload: dict[str, Any] = {
            "ready": ready,
            "pid": self.pid,
            "owned": self._owned,
            "api": self.api_base,
            "rtsp": f"rtsp://{self.host}:{self.rtsp_port}",
            "webrtc": f"{self.host}:{self.webrtc_port}",
        }
        if sid:
            payload["stream_id"] = sid
            payload["ws"] = self.ws_url(sid)
            payload["mjpeg"] = self.mjpeg_url(sid)
            payload["rtsp_url"] = self.rtsp_url(sid)
        return payload

    def _state_root(self) -> Path:
        if self._state_dir is not None:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            return self._state_dir
        path = data_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _pid_path(self) -> Path:
        return self._state_root() / "go2rtc.pid"

    def _write_config(self) -> Path:
        work = Path(tempfile.mkdtemp(prefix="inbound-go2rtc-"))
        self._work_dir = work
        config = work / "go2rtc.yaml"
        ffmpeg_bin = find_ffmpeg_binary()
        if ffmpeg_bin:
            # YAML double-quoted strings treat `\` as escape, so Windows
            # `C:\ffmpeg\bin\ffmpeg.exe` would become garbage. Forward slashes
            # are accepted by CreateProcess and go2rtc on Windows.
            ffmpeg_yaml = yaml_quoted_path(ffmpeg_bin)
            ffmpeg_section = f'ffmpeg:\n  bin: "{ffmpeg_yaml}"\n'
        else:
            ffmpeg_section = ""
        config.write_text(
            _CONFIG_TEMPLATE.format(
                host=self.host,
                api_port=self.api_port,
                rtsp_port=self.rtsp_port,
                webrtc_port=self.webrtc_port,
                ffmpeg_section=ffmpeg_section,
            ),
            encoding="utf-8",
        )
        self._config_path = config
        return config

    def _kill_stale(self) -> None:
        pid_path = self._pid_path()
        if not pid_path.exists():
            return
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = 0
        if pid:
            print(f"[go2rtc] stopping leftover pid={pid}", flush=True)
            _kill_pid(pid)
        try:
            pid_path.unlink()
        except OSError:
            pass

    def _spawn(self, binary: Path, config: Path) -> subprocess.Popen:
        log_path = self._state_root() / "go2rtc.log"
        self._log_file = log_path.open("ab", buffering=0)
        kwargs: dict[str, Any] = {
            "stdout": self._log_file,
            "stderr": subprocess.STDOUT,
            "cwd": str(config.parent),
        }
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            kwargs["creationflags"] = flags | no_window
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen([str(binary), "-c", str(config)], **kwargs)

    def start(self, timeout: float = 8.0) -> bool:
        """Launch go2rtc (or attach if this instance is already up)."""
        if self._proc is not None and self._proc.poll() is None and self.is_ready():
            return True

        self._kill_stale()

        if self.is_ready():
            # Default ports already serve go2rtc (another copy). Reuse it.
            print(f"[go2rtc] reusing API at {self.api_base}", flush=True)
            self._owned = False
            return True

        binary = self._binary or ensure_binary()
        config = self._write_config()
        try:
            self._proc = self._spawn(binary, config)
        except OSError as exc:
            print(f"[go2rtc] failed to spawn {binary}: {exc}", flush=True)
            self._close_log()
            return False

        self._owned = True
        try:
            self._pid_path().write_text(str(self._proc.pid), encoding="utf-8")
        except OSError:
            pass

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

        if self.wait_ready(timeout=timeout):
            print(
                f"[go2rtc] ready pid={self._proc.pid} api={self.api_base} "
                f"rtsp=:{self.rtsp_port} webrtc=:{self.webrtc_port}",
                flush=True,
            )
            return True

        tail = self._read_log_tail()
        print(f"[go2rtc] did not become ready on {self.api_base}", flush=True)
        if tail:
            print(tail, flush=True)
        self.stop()
        return False

    def wait_ready(self, timeout: float = 8.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                return False
            if self.is_ready():
                return True
            time.sleep(0.1)
        return self.is_ready()

    def wait_port(self, port: int, timeout: float = 8.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if _port_open(self.host, port):
                return True
            time.sleep(0.05)
        return _port_open(self.host, port)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        owned = self._owned
        self._owned = False
        if proc is not None and owned and proc.poll() is None:
            self._terminate(proc)
        elif owned:
            pid_path = self._pid_path()
            if pid_path.exists():
                try:
                    _kill_pid(int(pid_path.read_text(encoding="utf-8").strip()))
                except (ValueError, OSError):
                    pass
        try:
            self._pid_path().unlink()
        except OSError:
            pass
        if self._client is not None:
            self._client.close()
            self._client = None
        self._close_log()
        work = self._work_dir
        self._work_dir = None
        self._config_path = None
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)

    def _terminate(self, proc: subprocess.Popen) -> None:
        pid = proc.pid
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    proc.terminate()
            try:
                proc.wait(timeout=3.0)
                return
            except subprocess.TimeoutExpired:
                pass
            if sys.platform == "win32":
                _kill_pid(pid)
                proc.wait(timeout=2.0)
            else:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _close_log(self) -> None:
        handle = self._log_file
        self._log_file = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _read_log_tail(self, nbytes: int = 4000) -> str:
        log_path = self._state_root() / "go2rtc.log"
        try:
            data = log_path.read_bytes()[-nbytes:]
            return data.decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def __enter__(self) -> "Go2RtcManager":
        if not self.start():
            raise RuntimeError("go2rtc failed to start")
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
