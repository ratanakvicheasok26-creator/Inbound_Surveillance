#!/usr/bin/env python3
"""Create a clean, production-ready ZIP file for customer download.

Excludes developer artifacts (.git, virtual environments, caches, node_modules, temp media)
and packages everything required for 1-click installation on customer laptops.
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = ROOT_DIR / "Champei_Spa_Intelligence.zip"

EXCLUDE_DIRS = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    ".tempmediaStorage",
    ".user_uploaded",
    "test_tracking_outputs",
    "artifacts",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".log",
}


def make_customer_zip():
    print(f"[Packaging] Scanning project directory: {ROOT_DIR}")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    included_count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".venv")]

            rel_root = Path(root).relative_to(ROOT_DIR)

            # Skip root output zip itself if in walking path
            if rel_root.as_posix().startswith("Champei_Spa_Intelligence.zip"):
                continue

            for file in files:
                file_path = Path(root) / file
                if file_path == OUTPUT_ZIP:
                    continue
                if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                    continue
                if file.startswith(".DS_Store"):
                    continue

                arcname = (rel_root / file).as_posix()
                zf.write(file_path, arcname)
                included_count += 1

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"\n[Packaging] SUCCESS!")
    print(f"📦 Created Release ZIP: {OUTPUT_ZIP.name} ({size_mb:.2f} MB)")
    print(f"📄 Total Files Included: {included_count}")
    print(f"📁 Path: {OUTPUT_ZIP.resolve()}")


if __name__ == "__main__":
    make_customer_zip()
