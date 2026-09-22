#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Stopping Inbound Virtual Camera Streamer Container..."
docker compose down
echo "Container stopped."
