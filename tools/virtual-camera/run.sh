#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

mkdir -p "$DIR/videos"

echo "========================================================="
echo "  Starting Inbound Virtual Camera Streamer Container...  "
echo "========================================================="

docker compose up --build -d

echo ""
echo "Container started successfully!"
echo "========================================================="
echo "  Web Management UI : http://localhost:8090"
echo "  RTSP Stream Feed  : rtsp://127.0.0.1:8556/garage"
echo "========================================================="
echo "To view live logs: docker compose logs -f"
echo "To stop container: ./stop.sh"
echo "========================================================="
