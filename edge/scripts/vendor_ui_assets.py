#!/usr/bin/env python3
"""Download hub/desktop CSS fonts and the Tailwind Play compiler for offline use."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

HUB_CSS = (
    "https://fonts.googleapis.com/css2"
    "?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,100..700,0..1,0"
    "&family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500"
    "&display=swap"
)
WEB_CSS = (
    "https://fonts.googleapis.com/css2"
    "?family=Outfit:wght@400;500;600;700;800"
    "&family=Plus+Jakarta+Sans:wght@400;500;600;700"
    "&display=swap"
)
TAILWIND = "https://cdn.tailwindcss.com?plugins=forms,container-queries"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def vendor_css(css_url: str, css_path: Path, font_dir: Path, url_prefix: str) -> None:
    font_dir.mkdir(parents=True, exist_ok=True)
    css = fetch(css_url).decode("utf-8")
    for raw in re.findall(r"url\(([^)]+)\)", css):
        url = raw.strip().strip("'\"")
        if not url.startswith("http"):
            continue
        fname = urlparse(url).path.rsplit("/", 1)[-1]
        dest = font_dir / fname
        if not dest.exists():
            dest.write_bytes(fetch(url))
        css = css.replace(url, url_prefix + fname)
    css_path.parent.mkdir(parents=True, exist_ok=True)
    css_path.write_text(css, encoding="utf-8")


def main() -> None:
    tailwind = ROOT / "edge" / "static" / "vendor" / "tailwind.js"
    tailwind.parent.mkdir(parents=True, exist_ok=True)
    tailwind.write_bytes(fetch(TAILWIND))
    vendor_css(
        HUB_CSS,
        ROOT / "edge" / "static" / "vendor" / "fonts.css",
        ROOT / "edge" / "static" / "vendor" / "fonts",
        "/static/vendor/fonts/",
    )
    vendor_css(
        WEB_CSS,
        ROOT / "public" / "vendor" / "fonts.css",
        ROOT / "public" / "vendor" / "fonts",
        "/vendor/fonts/",
    )
    print("vendored Tailwind + fonts")


if __name__ == "__main__":
    main()
