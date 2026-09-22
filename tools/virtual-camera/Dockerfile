FROM python:3.11-slim-bookworm

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Install system dependencies: FFmpeg, curl, tar, and OpenCV graphics libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    tar \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install MediaMTX binary (detects architecture: amd64 or arm64)
ARG MEDIAMTX_VERSION=v1.9.3
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    case "${ARCH}" in \
        amd64) MTX_ARCH="linux_amd64" ;; \
        arm64) MTX_ARCH="linux_arm64" ;; \
        *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
    esac; \
    curl -fsSL "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_${MTX_ARCH}.tar.gz" \
    | tar -xz -C /usr/local/bin mediamtx; \
    chmod +x /usr/local/bin/mediamtx

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY mediamtx.yml .
COPY stream_manager.py .
COPY generate_test_clip.py .
COPY app.py .
COPY static ./static

# Ensure videos directory exists
RUN mkdir -p /app/videos

# Environment defaults
ENV VIDEOS_DIR=/app/videos \
    MEDIAMTX_PATH=/usr/local/bin/mediamtx \
    RTSP_INTERNAL_PORT=8554 \
    EXTERNAL_RTSP_PORT=8556 \
    EXTERNAL_WEBRTC_PORT=8889 \
    EXTERNAL_HOST=localhost

# Expose Web UI (8090), RTSP (8554 internal), WebRTC (8889), HLS (8888)
EXPOSE 8090 8554 8889 8888

# Launch FastAPI web application (lifespan supervises MediaMTX & streams)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
