#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " 🌿 Starting Champei Spa Intelligence System..."
echo "=========================================================="

if [ -f "edge/.venv/bin/activate" ]; then
    source edge/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

cd edge
python3 run_champei.py
