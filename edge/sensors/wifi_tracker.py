"""Local Wi-Fi presence via ARP / neighbor tables.

Owners register mechanic phone MAC addresses or static IPs. A background
thread reads the host ARP table (never probes foreign networks) and marks
those devices as on the garage LAN. Combined with Face ID this is two-factor
on-site confirmation.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAC_RE = re.compile(r"(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DEFAULT_INTERVAL = 45.0


def normalize_mac(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", ":")
    hexes = re.findall(r"[0-9a-f]{2}", text)
    if len(hexes) != 6:
        return ""
    return ":".join(hexes)


def normalize_ip(value: str | None) -> str:
    text = str(value or "").strip()
    match = IP_RE.search(text)
    return match.group(0) if match else ""


def normalize_wifi_devices(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("staff_name") or "").strip()
        mac = normalize_mac(item.get("mac") or item.get("mac_address"))
        ip = normalize_ip(item.get("ip") or item.get("static_ip"))
        if not name or (not mac and not ip):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "mac": mac, "ip": ip})
    return out


def presence_status(face_on_site: bool, wifi_connected: bool) -> str:
    """Two-factor on-site check: Face ID + garage Wi-Fi."""
    if face_on_site and wifi_connected:
        return "confirmed"
    if face_on_site:
        return "face_only"
    if wifi_connected:
        return "wifi_only"
    return "off_site"


def parse_arp_table(text: str) -> list[dict[str, str]]:
    """Parse `arp -a`, `ip neigh`, or /proc/net/arp style dumps."""
    neighbors: list[dict[str, str]] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("address") or line.lower().startswith("ip address"):
            continue
        mac = ""
        mac_match = MAC_RE.search(line.replace("-", ":"))
        if mac_match:
            mac = normalize_mac(mac_match.group(0))
        # /proc/net/arp columns: IP HW-type Flags MAC Mask Device
        parts = line.split()
        ip = ""
        if parts and IP_RE.fullmatch(parts[0]):
            ip = parts[0]
            if len(parts) >= 4:
                mac = normalize_mac(parts[3]) or mac
        if not ip:
            ip_match = IP_RE.search(line)
            ip = ip_match.group(0) if ip_match else ""
        if not mac and not ip:
            continue
        neighbors.append({"ip": ip, "mac": mac})
    return neighbors


def read_proc_arp(path: Path | str = "/proc/net/arp") -> str:
    if sys.platform == "win32":
        return ""
    proc = Path(path)
    try:
        return proc.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def run_arp_command() -> str:
    chunks: list[str] = []
    for argv in (("arp", "-a"), ("ip", "neigh")):
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
        if result.stdout:
            chunks.append(result.stdout)
    return "\n".join(chunks)


def load_neighbors() -> list[dict[str, str]]:
    text = read_proc_arp()
    if not text:
        text = run_arp_command()
    return parse_arp_table(text)


@dataclass
class DevicePresence:
    name: str
    mac: str
    ip: str
    connected: bool
    matched_ip: str = ""
    matched_mac: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mac": self.mac,
            "ip": self.ip,
            "connected": self.connected,
            "matched_ip": self.matched_ip,
            "matched_mac": self.matched_mac,
        }


class WifiTracker:
    """Non-blocking ARP watcher for registered employee phones."""

    def __init__(
        self,
        devices: Any | None = None,
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self.interval = max(15.0, min(120.0, float(interval)))
        self._lock = threading.Lock()
        self._devices = normalize_wifi_devices(devices)
        self._present: dict[str, DevicePresence] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._was_connected: dict[str, bool] = {}
        self._last_departures: list[str] = []

    def set_devices(self, devices: Any) -> list[dict[str, str]]:
        with self._lock:
            self._devices = normalize_wifi_devices(devices)
            return [dict(d) for d in self._devices]

    def devices(self) -> list[dict[str, str]]:
        with self._lock:
            return [dict(d) for d in self._devices]

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="WifiTracker",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None

    def _loop(self) -> None:
        self.scan_once()
        while not self._stop.wait(self.interval):
            try:
                self.scan_once()
            except Exception as exc:
                print(f"[wifi] scan failed: {exc}", flush=True)

    def scan_once(self, table_text: str | None = None) -> list[DevicePresence]:
        neighbors = parse_arp_table(table_text) if table_text is not None else load_neighbors()
        macs = {row["mac"]: row for row in neighbors if row.get("mac")}
        ips = {row["ip"]: row for row in neighbors if row.get("ip")}
        found: dict[str, DevicePresence] = {}
        with self._lock:
            devices = list(self._devices)
        for device in devices:
            hit = None
            if device["mac"] and device["mac"] in macs:
                hit = macs[device["mac"]]
            elif device["ip"] and device["ip"] in ips:
                hit = ips[device["ip"]]
            presence = DevicePresence(
                name=device["name"],
                mac=device["mac"],
                ip=device["ip"],
                connected=hit is not None,
                matched_ip=(hit or {}).get("ip", ""),
                matched_mac=(hit or {}).get("mac", ""),
            )
            found[device["name"]] = presence
        dropped: list[str] = []
        with self._lock:
            for name, presence in found.items():
                was = self._was_connected.get(name, False)
                if was and not presence.connected:
                    dropped.append(name)
                self._was_connected[name] = presence.connected
            self._present = found
            self._last_departures = dropped
        return list(found.values())

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [p.as_dict() for p in self._present.values()]

    def is_connected(self, name: str) -> bool:
        key = str(name or "").strip()
        with self._lock:
            row = self._present.get(key)
            if row is not None:
                return row.connected
            for presence in self._present.values():
                if presence.name.lower() == key.lower():
                    return presence.connected
        return False

    def departures(self) -> list[str]:
        """Names that dropped off garage Wi-Fi since the previous snapshot."""
        with self._lock:
            out = list(getattr(self, "_last_departures", []))
            self._last_departures = []
            return out
