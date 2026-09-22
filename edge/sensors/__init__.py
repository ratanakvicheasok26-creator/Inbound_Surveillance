"""Local shop-floor sensors (Wi-Fi presence, etc.)."""

from sensors.wifi_tracker import WifiTracker, normalize_mac, presence_status

__all__ = ["WifiTracker", "normalize_mac", "presence_status"]
