// Virtual Camera Streamer Dashboard Client

let streams = [];
let videos = [];
let activeDownloads = {};

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initDropzone();
  fetchStreams();
  fetchVideos();
  fetchDownloads();

  // Polling intervals
  setInterval(fetchStreams, 3000);
  setInterval(fetchDownloads, 2000);
});

// Fetch Active Streams
async function fetchStreams() {
  try {
    const res = await fetch("/api/streams");
    if (!res.ok) return;
    const data = await res.json();
    streams = data.streams || [];
    
    // Update header port
    if (data.rtsp_port) {
      document.getElementById("nav-rtsp-port").textContent = data.rtsp_port;
      const guideUrl = document.getElementById("guide-rtsp-url");
      if (guideUrl) {
        guideUrl.textContent = `rtsp://${data.external_host || "127.0.0.1"}:${data.rtsp_port}/garage`;
      }
    }

    renderStreams();
  } catch (err) {
    console.error("Failed to load streams:", err);
  }
}

// Render active stream cards
function renderStreams() {
  const container = document.getElementById("streams-container");
  const badge = document.getElementById("active-streams-badge");

  const liveCount = streams.filter(s => s.status === "streaming").length;
  badge.textContent = `${liveCount} Live`;
  badge.className = liveCount > 0
    ? "text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full"
    : "text-xs font-mono bg-slate-800 text-slate-400 border border-slate-700 px-2 py-0.5 rounded-full";

  if (streams.length === 0) {
    container.innerHTML = `
      <div class="col-span-full bg-[#111726]/60 border border-dashed border-slate-800 rounded-xl p-8 text-center">
        <p class="text-sm text-slate-400">No active camera channels. Upload a video and click <strong>Stream</strong> below to start.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = streams.map(s => {
    const isLive = s.status === "streaming";
    return `
      <div class="bg-[#111726] border ${isLive ? 'border-emerald-500/30' : 'border-slate-800'} rounded-xl p-5 shadow-xl relative overflow-hidden flex flex-col justify-between">
        ${isLive ? '<div class="camera-scanline"></div>' : ''}
        
        <div>
          <!-- Header info -->
          <div class="flex items-center justify-between mb-3">
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-full ${isLive ? 'bg-emerald-400 shadow-sm shadow-emerald-400 animate-pulse' : 'bg-rose-500'}"></span>
              <h4 class="text-sm font-bold text-white uppercase tracking-wider font-mono">CHANNEL: ${s.channel}</h4>
            </div>
            <span class="text-[11px] font-mono px-2 py-0.5 rounded ${isLive ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400'}">
              ${isLive ? `LIVE (${s.uptime_seconds}s)` : (s.error ? 'ERROR' : 'STOPPED')}
            </span>
          </div>

          <!-- Video preview simulator placeholder or player -->
          <div class="relative bg-slate-950 rounded-lg overflow-hidden border border-slate-800 aspect-video flex flex-col items-center justify-center p-4 mb-4">
            <div class="absolute top-2 left-2 flex items-center gap-1.5 bg-black/60 backdrop-blur px-2 py-0.5 rounded text-[10px] font-mono text-emerald-400">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
              REC • RTSP STREAMING
            </div>
            <div class="text-center px-4">
              <svg class="w-12 h-12 text-slate-700 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
              </svg>
              <p class="text-xs font-mono text-slate-300 font-semibold truncate max-w-[320px]">${s.video_file || 'No video loaded'}</p>
              <p class="text-[11px] text-slate-500 mt-1">Continuous Real-Time Pacing (1x)</p>
            </div>
          </div>

          <!-- Stream URL Display -->
          <div class="space-y-1.5 mb-4">
            <label class="text-[10px] font-bold uppercase tracking-wider text-slate-400">RTSP Feed URL</label>
            <div class="flex items-center gap-2">
              <input type="text" readonly value="${s.rtsp_url}" 
                class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-emerald-400 font-mono select-all">
              <button onclick="copyToClipboard('${s.rtsp_url}', this)" 
                class="shrink-0 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium px-3 py-1.5 rounded-lg transition flex items-center gap-1 cursor-pointer">
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                </svg>
                Copy
              </button>
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="pt-3 border-t border-slate-800/80 flex items-center justify-between">
          <div class="flex items-center gap-2">
            <select id="switch-video-${s.channel}" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded px-2 py-1 outline-none">
              ${videos.map(v => `<option value="${v.name}" ${v.name === s.video_file ? 'selected' : ''}>${v.name}</option>`).join('')}
            </select>
            <button onclick="switchStreamVideo('${s.channel}')" class="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs px-2.5 py-1 rounded transition cursor-pointer">
              Switch
            </button>
          </div>

          <div>
            ${isLive 
              ? `<button onclick="stopStream('${s.channel}')" class="bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs px-3 py-1 rounded transition cursor-pointer">Stop</button>`
              : `<button onclick="startStream('${s.channel}', '${s.video_file}')" class="bg-emerald-600 hover:bg-emerald-500 text-white text-xs px-3 py-1 rounded transition cursor-pointer">Start</button>`
            }
          </div>
        </div>

      </div>
    `;
  }).join('');
}

// Fetch Video Library
async function fetchVideos() {
  try {
    const res = await fetch("/api/videos");
    if (!res.ok) return;
    const data = await res.json();
    videos = data.videos || [];
    renderVideos();
  } catch (err) {
    console.error("Failed to load videos:", err);
  }
}

// Render Video List Table
function renderVideos() {
  const container = document.getElementById("video-list-container");
  if (videos.length === 0) {
    container.innerHTML = `
      <div class="py-8 text-center">
        <p class="text-xs text-slate-400">No videos in library yet. Upload a video file or download a link above.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = videos.map(v => `
    <div class="py-3 px-2 flex items-center justify-between hover:bg-slate-900/40 rounded-lg transition">
      <div class="flex items-center gap-3 min-w-0 pr-4">
        <div class="w-8 h-8 rounded bg-slate-800 flex items-center justify-center shrink-0 text-slate-400">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
          </svg>
        </div>
        <div class="min-w-0">
          <p class="text-xs font-semibold text-slate-200 truncate">${v.name}</p>
          <p class="text-[10px] text-slate-500 font-mono">${v.size_mb} MB • ${v.modified_at}</p>
        </div>
      </div>

      <div class="flex items-center gap-2 shrink-0">
        <button onclick="startStream('garage', '${v.name}')" 
          class="bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 text-xs px-2.5 py-1 rounded transition cursor-pointer flex items-center gap-1">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
          </svg>
          Stream
        </button>
        <button onclick="deleteVideo('${v.name}')" 
          class="text-slate-500 hover:text-rose-400 text-xs p-1 rounded transition cursor-pointer" title="Delete video">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
          </svg>
        </button>
      </div>
    </div>
  `).join('');
}

// Start Stream Action
async function startStream(channel, videoFile) {
  try {
    const res = await fetch("/api/streams/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel, video: videoFile })
    });
    const data = await res.json();
    if (!res.ok) {
      alert("Error starting stream: " + (data.detail || "Unknown error"));
    }
    fetchStreams();
  } catch (err) {
    alert("Connection error: " + err.message);
  }
}

// Stop Stream Action
async function stopStream(channel) {
  try {
    await fetch("/api/streams/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ channel })
    });
    fetchStreams();
  } catch (err) {
    console.error("Stop stream error:", err);
  }
}

// Switch video on channel
function switchStreamVideo(channel) {
  const select = document.getElementById(`switch-video-${channel}`);
  if (select && select.value) {
    startStream(channel, select.value);
  }
}

// Delete Video Action
async function deleteVideo(filename) {
  if (!confirm(`Delete ${filename}?`)) return;
  try {
    await fetch(`/api/videos/${encodeURIComponent(filename)}`, { method: "DELETE" });
    fetchVideos();
    fetchStreams();
  } catch (err) {
    console.error("Delete failed:", err);
  }
}

// Generate Quick Demo Clip
async function generateSample() {
  try {
    const res = await fetch("/api/generate-sample", { method: "POST" });
    const data = await res.json();
    if (data.success && data.filename) {
      fetchVideos();
      startStream("garage", data.filename);
    }
  } catch (err) {
    alert("Sample generation failed: " + err.message);
  }
}

// Drag and drop video upload
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("border-cyan-500", "bg-slate-900/80");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("border-cyan-500", "bg-slate-900/80");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("border-cyan-500", "bg-slate-900/80");
    if (e.dataTransfer.files.length > 0) {
      uploadFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      uploadFile(fileInput.files[0]);
    }
  });
}

// Upload file with XHR progress
function uploadFile(file) {
  const card = document.getElementById("upload-progress-card");
  const nameLabel = document.getElementById("upload-filename");
  const pctLabel = document.getElementById("upload-pct");
  const bar = document.getElementById("upload-bar");

  card.classList.remove("hidden");
  nameLabel.textContent = file.name;
  pctLabel.textContent = "0%";
  bar.style.width = "0%";

  const formData = new FormData();
  formData.append("file", file);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      pctLabel.textContent = `${pct}%`;
      bar.style.width = `${pct}%`;
    }
  };

  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      const res = JSON.parse(xhr.responseText);
      pctLabel.textContent = "Complete!";
      setTimeout(() => card.classList.add("hidden"), 2000);
      fetchVideos();
      // Auto stream the newly uploaded video
      if (res.filename) {
        startStream("garage", res.filename);
      }
    } else {
      pctLabel.textContent = "Failed";
      alert("Upload failed: " + xhr.responseText);
    }
  };

  xhr.onerror = () => {
    pctLabel.textContent = "Network Error";
  };

  xhr.send(formData);
}

// Online video URL downloader
async function handleUrlSubmit(e) {
  e.preventDefault();
  const input = document.getElementById("online-url-input");
  const btn = document.getElementById("btn-download-url");
  const url = input.value.trim();
  if (!url) return;

  btn.disabled = true;
  btn.innerHTML = `<svg class="animate-spin w-4 h-4 mr-2" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path></svg> Starting download...`;

  try {
    const res = await fetch("/api/download-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    if (res.ok && data.task_id) {
      input.value = "";
      fetchDownloads();
    } else {
      alert("Failed: " + (data.detail || "Could not start download"));
    }
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
      </svg>
      Download & Add to Library
    `;
  }
}

// Fetch active background downloads
async function fetchDownloads() {
  try {
    const res = await fetch("/api/downloads");
    if (!res.ok) return;
    const data = await res.json();
    const list = data.downloads || [];
    renderDownloads(list);
  } catch (err) {
    console.error("Downloads fetch error:", err);
  }
}

// Render downloads status list
function renderDownloads(list) {
  const container = document.getElementById("downloads-list");
  const active = list.filter(d => d.status === "downloading" || (Date.now() / 1000 - d.started_at < 15));

  if (active.length === 0) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = active.map(d => `
    <div class="bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-xs font-mono">
      <div class="flex items-center justify-between mb-1">
        <span class="text-slate-300 truncate max-w-[200px]">${d.filename || d.url}</span>
        <span class="${d.status === 'completed' ? 'text-emerald-400' : d.status === 'error' ? 'text-rose-400' : 'text-purple-400'} font-semibold">
          ${d.status === 'completed' ? 'Done' : d.status === 'error' ? 'Failed' : `${d.progress}%`}
        </span>
      </div>
      <div class="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
        <div class="bg-purple-500 h-full transition-all duration-200" style="width: ${d.progress}%"></div>
      </div>
      ${d.status === 'completed' ? `<p class="text-[10px] text-emerald-400 mt-1">Ready! Auto-added to library.</p>` : ''}
      ${d.error ? `<p class="text-[10px] text-rose-400 mt-1 truncate">${d.error}</p>` : ''}
    </div>
  `).join('');

  // If newly completed, refresh library
  if (list.some(d => d.status === "completed" && !activeDownloads[d.id])) {
    list.forEach(d => { activeDownloads[d.id] = true; });
    fetchVideos();
  }
}

// Copy URL helper with feedback
function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.innerHTML;
    btn.innerHTML = `<span class="text-emerald-300 font-bold">Copied!</span>`;
    setTimeout(() => { btn.innerHTML = original; }, 1800);
  }).catch(() => {
    prompt("Copy Stream URL:", text);
  });
}

function copyGuideUrl() {
  const text = document.getElementById("guide-rtsp-url").textContent;
  navigator.clipboard.writeText(text).then(() => {
    alert("Copied to clipboard: " + text);
  });
}
