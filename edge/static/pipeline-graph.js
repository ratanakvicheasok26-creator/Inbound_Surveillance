/* Hub pipeline graph. Same IR the edge compiler consumes via POST /api/pipeline. */
(function (global) {
  const PORT_COLORS = {
    frames: "#e879f9",
    detections: "#22d3ee",
    identity: "#fbbf24",
    zones: "#4ade80",
    events: "#fb923c",
    alerts: "#f87171",
  };
  const NODE_PORTS = {
    camera: { inputs: [], outputs: ["frames"] },
    personDetect: { inputs: ["frames"], outputs: ["detections"] },
    pose: { inputs: ["detections"], outputs: ["detections"] },
    faceId: { inputs: ["detections"], outputs: ["identity"] },
    anonymousReid: { inputs: ["detections"], outputs: ["identity"] },
    vehicleDetect: { inputs: ["frames"], outputs: ["detections"] },
    zone: { inputs: ["frames"], outputs: ["zones"] },
    employeeLabor: { inputs: ["identity", "zones"], outputs: ["events"] },
    customerVisits: { inputs: ["identity", "zones"], outputs: ["events"] },
    alertRule: { inputs: ["events"], outputs: ["alerts"] },
    telegram: { inputs: ["alerts"], outputs: [] },
    persist: { inputs: ["events"], outputs: [] },
    complaintIntake: { inputs: ["events"], outputs: [] },
  };
  const CATALOG = {
    camera: { label: "Camera", group: "source", summary: "Video source for this graph. Detection runs on every camera in the Camera List.", accent: "#00ff66" },
    personDetect: { label: "Person detect", group: "vision", summary: "YOLO / RTMPose people", accent: "#22d3ee" },
    pose: { label: "Pose", group: "vision", summary: "Skeleton keypoints", accent: "#22d3ee" },
    faceId: { label: "Face ID", group: "vision", summary: "Staff enrollment only", accent: "#22d3ee" },
    anonymousReid: { label: "Anonymous re-ID", group: "vision", summary: "Body crop → OSNet (HSV fallback) → local gallery. New person gets a visitor_id. No names.", accent: "#22d3ee" },
    vehicleDetect: { label: "Vehicle detect", group: "vision", summary: "Car / van YOLO", accent: "#22d3ee" },
    zone: { label: "Zone", group: "space", summary: "Named ROI", accent: "#fbbf24" },
    employeeLabor: { label: "Employee labor", group: "monitor", summary: "Bay wrench-time", accent: "#fb923c" },
    customerVisits: { label: "Customer visits", group: "monitor", summary: "SQLite customer_visits + anonymous_subjects; today/week counts. Staff excluded.", accent: "#fb923c" },
    alertRule: { label: "Alert rules", group: "output", summary: "Idle-bay wait per camera", accent: "#f87171" },
    telegram: { label: "Telegram", group: "output", summary: "Photo dispatch", accent: "#f87171" },
    persist: { label: "Persist", group: "output", summary: "SQLite / sync", accent: "#f87171" },
    complaintIntake: { label: "Complaint intake", group: "output", summary: "Khmer STT + Telegram", accent: "#f87171" },
  };
  const GROUPS = [
    ["source", "Sources"],
    ["vision", "Vision"],
    ["space", "Space"],
    ["monitor", "Monitors"],
    ["output", "Outputs"],
  ];
  const MONITORS = { employeeLabor: true, customerVisits: true };
  const NODE_W = 208;
  const HEAD_H = 30;
  const PORT_ROW = 22;
  const PREVIEW_MIN_SCALE = 0.35;
  const PREVIEW_MAX_SCALE = 2.5;
  const DRAFT_KEY = "inbound.pipeline.draft";
  const LAYOUTS_KEY = "inbound.pipeline.layouts";
  const NODE_MIME = "application/x-inbound-node";
  const NODE_DRAG_PREFIX = "inbound-node:";

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function node(id, kind, label, x, y, extra) {
    return { id, type: "pipeline", position: { x, y }, data: Object.assign({ kind, label }, extra || {}) };
  }

  function edge(source, target, port) {
    return { id: source + "-" + target + "-" + port, source, target, sourceHandle: port, targetHandle: port };
  }

  function garageTemplate() {
    return {
      workplace_type: "garage",
      nodes: [
        node("cam", "camera", "Camera", 40, 180, { cameraName: "Lift Bay 1", protocol: "webcam" }),
        node("person", "personDetect", "Person detect", 280, 80),
        node("pose", "pose", "Pose", 520, 80),
        node("face", "faceId", "Face ID (staff)", 760, 40),
        node("vehicle", "vehicleDetect", "Vehicle detect", 280, 280),
        node("zone1", "zone", "Lift Bay 1", 520, 280, { zoneKind: "vehicle_bay", zoneName: "Lift Bay 1", roi: [0.1, 0.2, 0.35, 0.6] }),
        node("labor", "employeeLabor", "Employee labor", 1000, 160),
        node("persist", "persist", "Persist", 1240, 80),
        node("alert", "alertRule", "Alert rules", 1240, 200, { cooldownSec: 30 }),
        node("tg", "telegram", "Telegram", 1480, 200),
      ],
      edges: [
        edge("cam", "person", "frames"),
        edge("cam", "vehicle", "frames"),
        edge("cam", "zone1", "frames"),
        edge("person", "pose", "detections"),
        edge("pose", "face", "detections"),
        edge("face", "labor", "identity"),
        edge("zone1", "labor", "zones"),
        edge("labor", "persist", "events"),
        edge("labor", "alert", "events"),
        edge("alert", "tg", "alerts"),
      ],
    };
  }

  function massageTemplate() {
    return {
      workplace_type: "massage",
      nodes: [
        node("cam", "camera", "Camera", 40, 160, { cameraName: "Front desk", protocol: "webcam" }),
        node("person", "personDetect", "Person detect", 280, 80),
        node("reid", "anonymousReid", "Anonymous re-ID", 520, 80),
        node("ent", "zone", "Entrance", 280, 280, { zoneKind: "entrance", zoneName: "Entrance", roi: [0.05, 0.15, 0.25, 0.7] }),
        node("wait", "zone", "Waiting", 520, 280, { zoneKind: "waiting", zoneName: "Waiting", roi: [0.35, 0.2, 0.28, 0.55] }),
        node("room", "zone", "Treatment Room 1", 760, 280, { zoneKind: "treatment_room", zoneName: "Treatment Room 1", roi: [0.68, 0.18, 0.28, 0.62] }),
        node("reception", "zone", "Reception", 40, 320, { zoneKind: "reception", zoneName: "Reception", roi: [0.38, 0.06, 0.14, 0.14] }),
        node("visits", "customerVisits", "Customer visits", 1000, 140),
        node("persist", "persist", "Persist", 1240, 60),
        node("complaint", "complaintIntake", "Complaint intake", 1240, 220),
      ],
      edges: [
        edge("cam", "person", "frames"),
        edge("cam", "ent", "frames"),
        edge("cam", "wait", "frames"),
        edge("cam", "room", "frames"),
        edge("cam", "reception", "frames"),
        edge("person", "reid", "detections"),
        edge("reid", "visits", "identity"),
        edge("ent", "visits", "zones"),
        edge("wait", "visits", "zones"),
        edge("room", "visits", "zones"),
        edge("visits", "persist", "events"),
        edge("visits", "complaint", "events"),
      ],
    };
  }

  function templateFor(workplace) {
    return workplace === "massage" ? massageTemplate() : garageTemplate();
  }

  function builtinLayouts() {
    return [
      { id: "garage", name: "Garage", builtin: true, graph: garageTemplate() },
      { id: "massage", name: "Massage", builtin: true, graph: massageTemplate() },
    ];
  }

  function readCustomLayouts() {
    try {
      const raw = localStorage.getItem(LAYOUTS_KEY);
      if (!raw) return [];
      const rows = JSON.parse(raw);
      if (!Array.isArray(rows)) return [];
      return rows.filter((row) => row && row.id && row.name && row.graph && Array.isArray(row.graph.nodes)).map((row) => ({
        id: String(row.id),
        name: String(row.name),
        builtin: false,
        graph: row.graph,
      }));
    } catch (err) {
      return [];
    }
  }

  function writeCustomLayouts(rows) {
    try {
      localStorage.setItem(LAYOUTS_KEY, JSON.stringify(rows.map((row) => ({
        id: row.id,
        name: row.name,
        graph: row.graph,
      }))));
    } catch (err) {
      /* private mode */
    }
  }

  function allLayouts() {
    return builtinLayouts().concat(readCustomLayouts());
  }

  function layoutById(id) {
    return allLayouts().find((row) => row.id === id) || null;
  }

  function escXml(value) {
    return String(value || "").replace(/[<>&"]/g, (ch) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[ch]));
  }

  function defaultData(kind, workplace) {
    const meta = CATALOG[kind];
    const data = { kind, label: meta.label };
    if (kind === "zone") {
      data.zoneKind = workplace === "massage" ? "treatment_room" : "vehicle_bay";
      data.zoneName = workplace === "massage" ? "Treatment Room" : "Lift Bay";
      data.roi = [0.2, 0.2, 0.3, 0.4];
    }
    if (kind === "camera") {
      data.cameraName = "Camera";
      data.protocol = "webcam";
    }
    if (kind === "alertRule") data.cooldownSec = 30;
    return data;
  }

  function nodeHeight(kind) {
    const ports = NODE_PORTS[kind] || { inputs: [], outputs: [] };
    const rows = Math.max(ports.inputs.length, ports.outputs.length, 1);
    let h = HEAD_H + 26 + rows * PORT_ROW + 8;
    if (kind === "alertRule") {
      const extra = Math.max(0, cameraAlertParts().length - 1);
      h += Math.min(48, extra * 14);
    }
    return h;
  }

  function validate(graph) {
    const errors = [];
    const nodes = graph.nodes || [];
    const edges = graph.edges || [];
    const byId = {};
    nodes.forEach((n) => { byId[n.id] = n; });
    if (!nodes.some((n) => n.data && n.data.kind === "camera")) errors.push("Graph needs a Camera source.");
    const incoming = {};
    const incomingPorts = {};
    const outgoing = {};
    nodes.forEach((n) => {
      incoming[n.id] = 0;
      incomingPorts[n.id] = new Set();
      outgoing[n.id] = [];
    });
    edges.forEach((e) => {
      if (!byId[e.source] || !byId[e.target]) {
        errors.push("An edge points at a missing node.");
        return;
      }
      outgoing[e.source].push(e.target);
      incoming[e.target] += 1;
      const srcKind = byId[e.source].data.kind;
      const dstKind = byId[e.target].data.kind;
      const srcPort = e.sourceHandle || (NODE_PORTS[srcKind] && NODE_PORTS[srcKind].outputs[0]);
      const dstPort = e.targetHandle || (NODE_PORTS[dstKind] && NODE_PORTS[dstKind].inputs[0]);
      incomingPorts[e.target].add(dstPort);
      if (srcPort && dstPort && srcPort !== dstPort) {
        errors.push(srcKind + " " + srcPort + " cannot connect to " + dstKind + " " + dstPort + ".");
      }
    });
    nodes.forEach((n) => {
      if (!MONITORS[n.data.kind]) return;
      const ports = incomingPorts[n.id];
      if (!ports.has("identity")) errors.push((n.data.label || n.data.kind) + " needs an identity input.");
      if (!ports.has("zones")) errors.push((n.data.label || n.data.kind) + " needs a zone input.");
    });
    const queue = nodes.filter((n) => incoming[n.id] === 0).map((n) => n.id);
    let seen = 0;
    const indeg = Object.assign({}, incoming);
    while (queue.length) {
      const id = queue.shift();
      seen += 1;
      (outgoing[id] || []).forEach((next) => {
        indeg[next] -= 1;
        if (indeg[next] === 0) queue.push(next);
      });
    }
    if (nodes.length && seen !== nodes.length) errors.push("Graph has a cycle. Disconnect a loop before deploying.");
    return errors;
  }

  const state = {
    graph: garageTemplate(),
    snapshot: "",
    selectedId: null,
    view: { x: 24, y: 24, scale: 0.82 },
    connecting: null,
    drag: null,
    pan: null,
    mounted: false,
    ready: false,
    cameras: [],
    layoutPickId: "garage",
    paletteDragKind: null,
    previewView: { x: 0, y: 0, scale: 1 },
    previewPan: null,
    previewSize: { w: 1, h: 1 },
    onAbsentSecondsChange: null,
    onDeploy: null,
  };

  function els() {
    return {
      status: document.getElementById("pipe-status"),
      errors: document.getElementById("pipe-errors"),
      notice: document.getElementById("pipe-notice"),
      palette: document.getElementById("pipe-palette"),
      stage: document.getElementById("pipe-stage"),
      world: document.getElementById("pipe-world"),
      wires: document.getElementById("pipe-wires"),
      nodes: document.getElementById("pipe-nodes"),
      inspector: document.getElementById("pipe-inspector"),
      rubber: document.getElementById("pipe-rubber"),
      unsaved: document.getElementById("pipe-unsaved"),
      expand: document.getElementById("pipe-expand"),
      layoutModal: document.getElementById("pipe-layout-modal"),
      layoutList: document.getElementById("pipe-layout-list"),
      layoutPreview: document.getElementById("pipe-layout-preview"),
    };
  }

  function fingerprint(graph) {
    return JSON.stringify(graph || {});
  }

  function isDirty() {
    return fingerprint(state.graph) !== state.snapshot;
  }

  function cameraNames() {
    const cams = Array.isArray(state.cameras) ? state.cameras : [];
    return cams.map((c) => String((c && (c.name || c.id)) || "").trim()).filter(Boolean);
  }

  function clampAbsentSeconds(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 10;
    return Math.max(5, Math.min(600, Math.round(n)));
  }

  function cameraAlertParts() {
    const cams = Array.isArray(state.cameras) ? state.cameras : [];
    return cams.map((c) => ({
      id: String((c && c.id) || "").trim(),
      name: String((c && (c.name || c.id)) || "").trim() || "Camera",
      sec: clampAbsentSeconds(c && c.absent_seconds),
    })).filter((row) => row.id);
  }

  function readDraft() {
    try {
      const raw = sessionStorage.getItem(DRAFT_KEY);
      if (!raw) return null;
      const graph = JSON.parse(raw);
      if (graph && Array.isArray(graph.nodes) && graph.nodes.length) return graph;
    } catch (err) {
      return null;
    }
    return null;
  }

  function syncDraft() {
    const dirty = isDirty();
    const badge = els().unsaved;
    if (badge) badge.classList.toggle("hidden", !dirty);
    try {
      if (dirty) sessionStorage.setItem(DRAFT_KEY, JSON.stringify(state.graph));
      else sessionStorage.removeItem(DRAFT_KEY);
    } catch (err) {
      /* private mode */
    }
  }

  function markSnapshot() {
    state.snapshot = fingerprint(state.graph);
    syncDraft();
  }

  function setFullscreen(on) {
    document.body.classList.toggle("pipe-is-fullscreen", !!on);
    const btn = els().expand;
    if (!btn) return;
    const icon = btn.querySelector(".material-symbols-outlined");
    btn.title = on ? "Exit fullscreen" : "Fullscreen";
    btn.setAttribute("aria-label", on ? "Exit fullscreen" : "Enter fullscreen");
    if (icon) icon.textContent = on ? "fullscreen_exit" : "fullscreen";
  }

  function applyPreviewView() {
    const box = els().layoutPreview;
    if (!box) return;
    const world = box.querySelector(".pipe-layout-world");
    if (!world) return;
    const v = state.previewView;
    world.style.transform = "translate(" + v.x + "px," + v.y + "px) scale(" + v.scale + ")";
    box.classList.toggle("is-panning", !!state.previewPan);
  }

  function fitPreviewView() {
    const box = els().layoutPreview;
    if (!box) return;
    const rect = box.getBoundingClientRect();
    const w = state.previewSize.w || 1;
    const h = state.previewSize.h || 1;
    if (rect.width < 8 || rect.height < 8) return;
    const scale = Math.min(PREVIEW_MAX_SCALE, Math.max(PREVIEW_MIN_SCALE, Math.min(rect.width / w, rect.height / h)));
    state.previewView.scale = scale;
    state.previewView.x = (rect.width - w * scale) / 2;
    state.previewView.y = (rect.height - h * scale) / 2;
    applyPreviewView();
  }

  function previewToWorld(clientX, clientY) {
    const box = els().layoutPreview;
    if (!box) return { x: 0, y: 0 };
    const rect = box.getBoundingClientRect();
    const v = state.previewView;
    return {
      x: (clientX - rect.left - v.x) / v.scale,
      y: (clientY - rect.top - v.y) / v.scale,
    };
  }

  function startPreviewPan(ev) {
    if (ev.button !== 0 && ev.button !== 1) return;
    const box = els().layoutPreview;
    if (!box || !box.contains(ev.target)) return;
    ev.preventDefault();
    state.previewPan = { x: ev.clientX - state.previewView.x, y: ev.clientY - state.previewView.y };
    box.classList.add("is-panning");
    if (box.setPointerCapture) box.setPointerCapture(ev.pointerId);
  }

  function onPreviewPointerMove(ev) {
    if (!state.previewPan) return;
    state.previewView.x = ev.clientX - state.previewPan.x;
    state.previewView.y = ev.clientY - state.previewPan.y;
    applyPreviewView();
  }

  function onPreviewPointerUp() {
    state.previewPan = null;
    applyPreviewView();
  }

  function onPreviewWheel(ev) {
    ev.preventDefault();
    const box = els().layoutPreview;
    if (!box) return;
    const before = previewToWorld(ev.clientX, ev.clientY);
    const next = Math.min(PREVIEW_MAX_SCALE, Math.max(PREVIEW_MIN_SCALE, state.previewView.scale * (ev.deltaY > 0 ? 0.92 : 1.08)));
    state.previewView.scale = next;
    const rect = box.getBoundingClientRect();
    state.previewView.x = ev.clientX - rect.left - before.x * next;
    state.previewView.y = ev.clientY - rect.top - before.y * next;
    applyPreviewView();
  }

  function applyView() {
    const world = els().world;
    if (!world) return;
    const v = state.view;
    world.style.transform = "translate(" + v.x + "px," + v.y + "px) scale(" + v.scale + ")";
  }

  function portCenter(node, port, dir) {
    const spec = NODE_PORTS[node.data.kind] || { inputs: [], outputs: [] };
    const ports = spec[dir === "in" ? "inputs" : "outputs"];
    const idx = Math.max(0, ports.indexOf(port));
    const y = node.position.y + HEAD_H + 22 + idx * PORT_ROW + 8;
    const x = dir === "in" ? node.position.x : node.position.x + NODE_W;
    return { x, y };
  }

  function bezier(a, b) {
    const dx = Math.max(48, Math.abs(b.x - a.x) * 0.45);
    return "M " + a.x + " " + a.y + " C " + (a.x + dx) + " " + a.y + ", " + (b.x - dx) + " " + b.y + ", " + b.x + " " + b.y;
  }

  function drawWires() {
    const svg = els().wires;
    if (!svg) return;
    const byId = {};
    state.graph.nodes.forEach((n) => { byId[n.id] = n; });
    let html = "";
    state.graph.edges.forEach((e) => {
      const src = byId[e.source];
      const dst = byId[e.target];
      if (!src || !dst) return;
      const port = e.sourceHandle || e.targetHandle || "frames";
      const a = portCenter(src, port, "out");
      const b = portCenter(dst, port, "in");
      html += '<path d="' + bezier(a, b) + '" fill="none" stroke="' + (PORT_COLORS[port] || "#888") + '" stroke-width="2.4"></path>';
    });
    svg.innerHTML = html + '<path id="pipe-rubber" d="" fill="none" stroke="#00FF66" stroke-width="2" stroke-dasharray="6 4"></path>';
  }

  function subtitle(data) {
    if (data.kind === "zone") return data.zoneName || data.zoneKind || CATALOG[data.kind].summary;
    if (data.kind === "camera") {
      const names = cameraNames();
      return names.length ? "All cameras · " + names.join(" · ") : "All cameras";
    }
    if (data.kind === "alertRule") {
      const parts = cameraAlertParts();
      if (!parts.length) return "No cameras";
      return parts.map((row) => row.name + " " + row.sec + "s").join(" · ");
    }
    return CATALOG[data.kind].summary;
  }

  function drawNodes() {
    const root = els().nodes;
    if (!root) return;
    root.innerHTML = "";
    state.graph.nodes.forEach((n) => {
      const meta = CATALOG[n.data.kind];
      const ports = NODE_PORTS[n.data.kind];
      const el = document.createElement("div");
      el.className = "pipe-node" + (n.id === state.selectedId ? " is-selected" : "");
      el.style.left = n.position.x + "px";
      el.style.top = n.position.y + "px";
      el.style.width = NODE_W + "px";
      el.style.minHeight = nodeHeight(n.data.kind) + "px";
      el.style.borderColor = meta.accent;
      el.dataset.id = n.id;

      const head = document.createElement("header");
      head.style.background = meta.accent + "22";
      head.style.color = meta.accent;
      head.textContent = meta.label;
      head.addEventListener("pointerdown", (ev) => startNodeDrag(ev, n.id));
      el.appendChild(head);

      const sub = document.createElement("p");
      sub.className = "pipe-node__sub";
      sub.textContent = subtitle(n.data);
      el.appendChild(sub);

      const io = document.createElement("div");
      io.className = "pipe-node__io";
      const inn = document.createElement("div");
      ports.inputs.forEach((port) => inn.appendChild(socketEl(n.id, port, "in")));
      const out = document.createElement("div");
      ports.outputs.forEach((port) => out.appendChild(socketEl(n.id, port, "out")));
      io.appendChild(inn);
      io.appendChild(out);
      el.appendChild(io);

      el.addEventListener("pointerdown", (ev) => {
        if (ev.target.closest(".pipe-socket")) return;
        ev.stopPropagation();
        selectNode(n.id);
      });
      root.appendChild(el);
    });
  }

  function socketEl(nodeId, port, dir) {
    const wrap = document.createElement("div");
    wrap.className = "pipe-socket pipe-socket--" + dir;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pipe-dot";
    btn.style.background = PORT_COLORS[port];
    btn.dataset.node = nodeId;
    btn.dataset.port = port;
    btn.dataset.dir = dir;
    btn.title = port;
    btn.addEventListener("pointerdown", (ev) => startConnect(ev, nodeId, port, dir));
    const label = document.createElement("span");
    label.style.color = PORT_COLORS[port];
    label.textContent = port;
    if (dir === "in") {
      wrap.appendChild(btn);
      wrap.appendChild(label);
    } else {
      wrap.appendChild(label);
      wrap.appendChild(btn);
    }
    return wrap;
  }

  function selectNode(id) {
    state.selectedId = id;
    drawNodes();
    renderInspector();
  }

  function renderInspector() {
    const box = els().inspector;
    if (!box) return;
    const node = state.graph.nodes.find((n) => n.id === state.selectedId);
    box.innerHTML = "";
    const title = document.createElement("h3");
    if (!node) {
      title.textContent = "Inspector";
      box.appendChild(title);
      const p = document.createElement("p");
      p.textContent = "Select a node. Drag matching colors to wire the pipeline. Deploy applies it to this engine.";
      box.appendChild(p);
      return;
    }
    const meta = CATALOG[node.data.kind];
    title.textContent = meta.label;
    box.appendChild(title);
    const hint = document.createElement("p");
    hint.textContent = meta.summary;
    box.appendChild(hint);

    function field(label, input) {
      const wrap = document.createElement("label");
      wrap.className = "pipe-field";
      const span = document.createElement("span");
      span.textContent = label;
      wrap.appendChild(span);
      wrap.appendChild(input);
      box.appendChild(wrap);
    }

    if (node.data.kind === "camera") {
      const list = document.createElement("ul");
      list.className = "pipe-cam-list";
      const names = cameraNames();
      if (!names.length) {
        const empty = document.createElement("p");
        empty.textContent = "No cameras yet. Add them in the left sidebar.";
        box.appendChild(empty);
      } else {
        names.forEach((name) => {
          const li = document.createElement("li");
          li.textContent = name;
          list.appendChild(li);
        });
        box.appendChild(list);
      }
    }

    const titleIn = document.createElement("input");
    titleIn.value = node.data.label || "";
    titleIn.addEventListener("input", () => { node.data.label = titleIn.value; drawNodes(); syncDraft(); });
    field("Title", titleIn);

    if (node.data.kind === "zone") {
      const nameIn = document.createElement("input");
      nameIn.value = node.data.zoneName || "";
      nameIn.addEventListener("input", () => {
        node.data.zoneName = nameIn.value;
        node.data.label = nameIn.value || node.data.label;
        drawNodes();
        syncDraft();
      });
      field("Zone name", nameIn);
      const kind = document.createElement("select");
      const workplace = state.graph.workplace_type === "massage" ? "massage" : "garage";
      const kinds = workplace === "massage"
        ? [["entrance", "Entrance"], ["waiting", "Waiting"], ["treatment_room", "Treatment room"], ["reception", "Reception"]]
        : [["vehicle_bay", "Vehicle bay"], ["tool_area", "Tool area"]];
      kinds.forEach(([id, label]) => {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = label;
        if ((node.data.zoneKind || kinds[0][0]) === id) opt.selected = true;
        kind.appendChild(opt);
      });
      kind.addEventListener("change", () => { node.data.zoneKind = kind.value; drawNodes(); syncDraft(); });
      field("Kind", kind);
    }

    if (node.data.kind === "alertRule") {
      const parts = cameraAlertParts();
      if (!parts.length) {
        const empty = document.createElement("p");
        empty.textContent = "Add cameras in the left sidebar to set idle-bay alerts.";
        box.appendChild(empty);
      } else {
        parts.forEach((row) => {
          const input = document.createElement("input");
          input.type = "number";
          input.min = "5";
          input.max = "600";
          input.step = "5";
          input.value = String(row.sec);
          input.addEventListener("change", () => {
            const sec = clampAbsentSeconds(input.value);
            input.value = String(sec);
            const cam = (state.cameras || []).find((c) => String(c.id) === row.id);
            if (cam) cam.absent_seconds = sec;
            drawNodes();
            if (typeof state.onAbsentSecondsChange === "function") {
              state.onAbsentSecondsChange(row.id, sec);
            }
          });
          field(row.name + " alert seconds", input);
        });
      }
    }

    const del = document.createElement("button");
    del.type = "button";
    del.className = "pipe-danger";
    del.textContent = "Delete node";
    del.addEventListener("click", confirmRemoveSelected);
    box.appendChild(del);
  }

  function renderPalette() {
    const root = els().palette;
    if (!root || root.dataset.ready) return;
    GROUPS.forEach(([id, label]) => {
      const h = document.createElement("h3");
      h.textContent = label;
      root.appendChild(h);
      Object.keys(CATALOG).forEach((kind) => {
        if (CATALOG[kind].group !== id) return;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.draggable = true;
        btn.dataset.kind = kind;
        btn.innerHTML = "<strong></strong><span></span>";
        btn.querySelector("strong").textContent = CATALOG[kind].label;
        btn.querySelector("span").textContent = CATALOG[kind].summary;
        btn.addEventListener("dragstart", (ev) => {
          state.paletteDragKind = kind;
          ev.dataTransfer.setData(NODE_MIME, kind);
          ev.dataTransfer.setData("text/plain", NODE_DRAG_PREFIX + kind);
          ev.dataTransfer.effectAllowed = "copy";
        });
        btn.addEventListener("dragend", () => {
          const stage = els().stage;
          if (stage) stage.classList.remove("is-drop");
          window.setTimeout(() => { state.paletteDragKind = null; }, 0);
        });
        btn.addEventListener("click", () => {
          if (state.paletteDragKind) return;
          addNode(kind);
        });
        root.appendChild(btn);
      });
    });
    root.dataset.ready = "1";
  }

  function viewportCenter() {
    const stage = els().stage;
    if (!stage) return { x: 80, y: 70 };
    const rect = stage.getBoundingClientRect();
    return stageToWorld(rect.left + rect.width / 2, rect.top + rect.height / 2);
  }

  function addNode(kind, position) {
    if (!CATALOG[kind]) return;
    const id = kind + "-" + Math.random().toString(36).slice(2, 8);
    let pos;
    if (position) {
      pos = { x: position.x, y: position.y };
    } else {
      const center = viewportCenter();
      pos = { x: center.x - NODE_W / 2, y: center.y - nodeHeight(kind) / 2 };
    }
    state.graph.nodes.push({
      id,
      type: "pipeline",
      position: pos,
      data: defaultData(kind, state.graph.workplace_type),
    });
    state.selectedId = id;
    refresh();
  }

  function selectedNodeLabel() {
    const node = state.graph.nodes.find((n) => n.id === state.selectedId);
    if (!node || !node.data) return "";
    return node.data.label || (CATALOG[node.data.kind] && CATALOG[node.data.kind].label) || node.data.kind;
  }

  function confirmRemoveSelected() {
    if (!state.selectedId) return;
    const label = selectedNodeLabel();
    if (!confirm('Delete "' + label + '"? This removes it from the canvas. Deployed detection is unchanged until you Deploy.')) return;
    removeSelected();
  }

  function removeSelected() {
    if (!state.selectedId) return;
    const id = state.selectedId;
    state.graph.nodes = state.graph.nodes.filter((n) => n.id !== id);
    state.graph.edges = state.graph.edges.filter((e) => e.source !== id && e.target !== id);
    state.selectedId = null;
    refresh();
  }

  function refresh() {
    drawNodes();
    drawWires();
    renderInspector();
    const errors = validate(state.graph);
    const status = els().status;
    const errBox = els().errors;
    if (status) {
      status.textContent = errors.length ? errors.length + " issue" + (errors.length === 1 ? "" : "s") : "Valid";
      status.className = "pipe-status " + (errors.length ? "is-bad" : "is-ok");
    }
    if (errBox) {
      errBox.innerHTML = errors.map((e) => "<li>" + e.replace(/[<>]/g, "") + "</li>").join("");
      errBox.classList.toggle("hidden", !errors.length);
    }
    syncDraft();
  }

  function stageToWorld(clientX, clientY) {
    const stage = els().stage;
    const rect = stage.getBoundingClientRect();
    const v = state.view;
    return {
      x: (clientX - rect.left - v.x) / v.scale,
      y: (clientY - rect.top - v.y) / v.scale,
    };
  }

  function startNodeDrag(ev, id) {
    ev.preventDefault();
    ev.stopPropagation();
    selectNode(id);
    const node = state.graph.nodes.find((n) => n.id === id);
    state.drag = {
      id,
      dx: stageToWorld(ev.clientX, ev.clientY).x - node.position.x,
      dy: stageToWorld(ev.clientX, ev.clientY).y - node.position.y,
    };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  }

  function startConnect(ev, nodeId, port, dir) {
    ev.preventDefault();
    ev.stopPropagation();
    if (dir !== "out") {
      selectNode(nodeId);
      return;
    }
    state.connecting = { nodeId, port, dir };
    ev.currentTarget.setPointerCapture(ev.pointerId);
  }

  function onPointerMove(ev) {
    if (state.drag) {
      const node = state.graph.nodes.find((n) => n.id === state.drag.id);
      const pt = stageToWorld(ev.clientX, ev.clientY);
      node.position.x = pt.x - state.drag.dx;
      node.position.y = pt.y - state.drag.dy;
      const el = els().nodes.querySelector('[data-id="' + node.id + '"]');
      if (el) {
        el.style.left = node.position.x + "px";
        el.style.top = node.position.y + "px";
      }
      drawWires();
      return;
    }
    if (state.pan) {
      state.view.x = ev.clientX - state.pan.x;
      state.view.y = ev.clientY - state.pan.y;
      applyView();
      return;
    }
    if (state.connecting) {
      const src = state.graph.nodes.find((n) => n.id === state.connecting.nodeId);
      const a = portCenter(src, state.connecting.port, "out");
      const b = stageToWorld(ev.clientX, ev.clientY);
      const rubber = document.getElementById("pipe-rubber");
      if (rubber) rubber.setAttribute("d", bezier(a, b));
    }
  }

  function onPointerUp(ev) {
    if (state.connecting) {
      const hit = document.elementFromPoint(ev.clientX, ev.clientY);
      const dot = hit && hit.closest ? hit.closest(".pipe-dot") : null;
      if (dot && dot.dataset.dir === "in" && dot.dataset.port === state.connecting.port && dot.dataset.node !== state.connecting.nodeId) {
        const exists = state.graph.edges.some((e) =>
          e.source === state.connecting.nodeId && e.target === dot.dataset.node && e.sourceHandle === state.connecting.port
        );
        if (!exists) {
          state.graph.edges.push(edge(state.connecting.nodeId, dot.dataset.node, state.connecting.port));
        }
      }
      state.connecting = null;
      refresh();
    }
    if (state.drag) syncDraft();
    state.drag = null;
    state.pan = null;
  }

  function startPan(ev) {
    if (ev.target.closest && ev.target.closest("#pipe-expand")) return;
    if (ev.target !== els().stage && ev.target !== els().world && ev.target !== els().wires && ev.target.id !== "pipe-nodes") return;
    if (ev.button !== 0 && ev.button !== 1) return;
    state.pan = { x: ev.clientX - state.view.x, y: ev.clientY - state.view.y };
    state.selectedId = ev.target.closest && ev.target.closest(".pipe-node") ? state.selectedId : null;
    if (!ev.target.closest(".pipe-node")) renderInspector();
  }

  function onWheel(ev) {
    ev.preventDefault();
    const before = stageToWorld(ev.clientX, ev.clientY);
    const next = Math.min(1.6, Math.max(0.35, state.view.scale * (ev.deltaY > 0 ? 0.92 : 1.08)));
    state.view.scale = next;
    const stage = els().stage.getBoundingClientRect();
    state.view.x = ev.clientX - stage.left - before.x * next;
    state.view.y = ev.clientY - stage.top - before.y * next;
    applyView();
  }

  function setNotice(text, bad) {
    const el = els().notice;
    if (!el) return;
    el.textContent = text || "";
    el.className = "text-sm " + (bad ? "text-error" : "text-primary");
    el.classList.toggle("hidden", !text);
  }

  async function loadFromEngine(workplace) {
    let graph = null;
    try {
      const res = await fetch("/api/pipeline");
      const data = await res.json();
      if (data && data.graph && data.graph.nodes && data.graph.nodes.length) graph = data.graph;
    } catch (err) {
      graph = null;
    }
    if (!graph) {
      try {
        const cfg = await fetch("/api/config").then((r) => r.json());
        workplace = workplace || cfg.workplace_type || "garage";
      } catch (err) {
        workplace = workplace || "garage";
      }
      graph = templateFor(workplace);
    }
    state.snapshot = fingerprint(graph);
    const draft = readDraft();
    state.graph = draft || graph;
    state.selectedId = null;
    state.ready = true;
    refresh();
    applyView();
  }

  function isNodeDrag(ev) {
    if (state.paletteDragKind && CATALOG[state.paletteDragKind]) return true;
    const types = ev.dataTransfer && ev.dataTransfer.types;
    if (!types) return false;
    const list = typeof types.contains === "function" ? types : { contains: (name) => Array.from(types).indexOf(name) >= 0 };
    return list.contains(NODE_MIME) || list.contains("text/plain");
  }

  function kindFromDrop(ev) {
    const mime = ev.dataTransfer.getData(NODE_MIME);
    if (CATALOG[mime]) return mime;
    const plain = ev.dataTransfer.getData("text/plain") || "";
    if (plain.indexOf(NODE_DRAG_PREFIX) === 0) {
      const kind = plain.slice(NODE_DRAG_PREFIX.length);
      if (CATALOG[kind]) return kind;
    }
    if (state.paletteDragKind && CATALOG[state.paletteDragKind]) return state.paletteDragKind;
    return "";
  }

  function onStageDragOver(ev) {
    if (!isNodeDrag(ev)) return;
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "copy";
    els().stage.classList.add("is-drop");
  }

  function onStageDragLeave(ev) {
    const stage = els().stage;
    if (!stage) return;
    if (ev.relatedTarget && stage.contains(ev.relatedTarget)) return;
    stage.classList.remove("is-drop");
  }

  function onStageDrop(ev) {
    const kind = kindFromDrop(ev);
    if (!kind) return;
    ev.preventDefault();
    ev.stopPropagation();
    const stage = els().stage;
    if (stage) stage.classList.remove("is-drop");
    const pt = stageToWorld(ev.clientX, ev.clientY);
    addNode(kind, { x: pt.x - NODE_W / 2, y: pt.y - 16 });
    state.paletteDragKind = null;
  }

  function renderLayoutPreview(graph) {
    const box = els().layoutPreview;
    if (!box) return;
    const nodes = (graph && graph.nodes) || [];
    if (!nodes.length) {
      box.innerHTML = '<p style="color:#8E9297;padding:16px;margin:0;">This layout has no nodes.</p>';
      return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    nodes.forEach((n) => {
      const h = nodeHeight((n.data && n.data.kind) || "camera");
      minX = Math.min(minX, n.position.x);
      minY = Math.min(minY, n.position.y);
      maxX = Math.max(maxX, n.position.x + NODE_W);
      maxY = Math.max(maxY, n.position.y + h);
    });
    const pad = 36;
    const width = Math.max(1, maxX - minX + pad * 2);
    const height = Math.max(1, maxY - minY + pad * 2);
    const byId = {};
    nodes.forEach((n) => { byId[n.id] = n; });
    let html = "";
    (graph.edges || []).forEach((e) => {
      const src = byId[e.source];
      const dst = byId[e.target];
      if (!src || !dst) return;
      const port = e.sourceHandle || e.targetHandle || "frames";
      const a = portCenter(src, port, "out");
      const b = portCenter(dst, port, "in");
      html += '<path d="' + bezier(a, b) + '" fill="none" stroke="' + (PORT_COLORS[port] || "#888") + '" stroke-width="3"></path>';
    });
    nodes.forEach((n) => {
      const kind = n.data && n.data.kind;
      const meta = CATALOG[kind] || { label: kind, accent: "#888" };
      const h = nodeHeight(kind);
      html += '<rect x="' + n.position.x + '" y="' + n.position.y + '" width="' + NODE_W + '" height="' + h + '" rx="10" fill="#121212" stroke="' + meta.accent + '" stroke-width="2"></rect>';
      html += '<rect x="' + n.position.x + '" y="' + n.position.y + '" width="' + NODE_W + '" height="28" rx="10" fill="' + meta.accent + '22"></rect>';
      html += '<text x="' + (n.position.x + 10) + '" y="' + (n.position.y + 19) + '" fill="' + meta.accent + '" font-size="11" font-family="ui-monospace, monospace" font-weight="700">' + escXml((meta.label || "").toUpperCase()) + "</text>";
      html += '<text x="' + (n.position.x + 10) + '" y="' + (n.position.y + 46) + '" fill="#8E9297" font-size="11" font-family="sans-serif">' + escXml(n.data.label || meta.label) + "</text>";
    });
    box.innerHTML = '<div class="pipe-layout-world"><svg width="' + width + '" height="' + height + '" viewBox="' + (minX - pad) + " " + (minY - pad) + " " + width + " " + height + '" preserveAspectRatio="xMinYMin meet" xmlns="http://www.w3.org/2000/svg">' + html + "</svg></div>";
    state.previewSize = { w: width, h: height };
    requestAnimationFrame(fitPreviewView);
  }

  function renderLayoutPicker() {
    const list = els().layoutList;
    if (!list) return;
    const layouts = allLayouts();
    if (!layoutById(state.layoutPickId) && layouts[0]) state.layoutPickId = layouts[0].id;
    list.innerHTML = "";
    layouts.forEach((row) => {
      const item = document.createElement("div");
      item.setAttribute("role", "button");
      item.tabIndex = 0;
      item.className = "pipe-layout-item" + (row.id === state.layoutPickId ? " is-on" : "");
      item.dataset.id = row.id;
      const meta = document.createElement("div");
      meta.className = "pipe-layout-item__meta";
      const name = document.createElement("strong");
      name.textContent = row.name;
      const hint = document.createElement("span");
      hint.textContent = row.builtin ? "Built-in" : (row.graph.nodes || []).length + " nodes";
      meta.appendChild(name);
      meta.appendChild(hint);
      item.appendChild(meta);
      if (!row.builtin) {
        const del = document.createElement("button");
        del.type = "button";
        del.className = "pipe-layout-item__del";
        del.title = "Delete layout";
        del.setAttribute("aria-label", "Delete layout " + row.name);
        del.textContent = "×";
        del.addEventListener("click", (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          deleteCustomLayout(row.id);
        });
        item.appendChild(del);
      }
      const pick = () => {
        state.layoutPickId = row.id;
        renderLayoutPicker();
      };
      item.addEventListener("click", pick);
      item.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          pick();
        }
      });
      list.appendChild(item);
    });
    const selected = layoutById(state.layoutPickId);
    renderLayoutPreview(selected && selected.graph);
  }

  function openLayoutPicker() {
    const modal = els().layoutModal;
    if (!modal) return;
    const workplace = state.graph.workplace_type === "massage" ? "massage" : "garage";
    state.layoutPickId = workplace;
    modal.classList.remove("hidden");
    renderLayoutPicker();
  }

  function closeLayoutPicker() {
    const modal = els().layoutModal;
    if (modal) modal.classList.add("hidden");
  }

  function applyPickedLayout() {
    const entry = layoutById(state.layoutPickId);
    if (!entry) return;
    state.graph = clone(entry.graph);
    state.selectedId = null;
    closeLayoutPicker();
    setNotice(entry.name + " layout loaded. Deploy to apply it.");
    refresh();
  }

  function saveCurrentAsLayout() {
    const suggestion = state.graph.workplace_type === "massage" ? "My massage layout" : "My garage layout";
    const name = window.prompt("Name this layout", suggestion);
    if (name == null) return;
    const trimmed = name.trim();
    if (!trimmed) {
      setNotice("Layout name cannot be empty.", true);
      return;
    }
    const rows = readCustomLayouts();
    const existing = rows.find((row) => row.name.toLowerCase() === trimmed.toLowerCase());
    const entry = {
      id: existing ? existing.id : "custom-" + Date.now().toString(36),
      name: trimmed,
      graph: clone(state.graph),
    };
    const next = existing
      ? rows.map((row) => (row.id === existing.id ? entry : row))
      : rows.concat(entry);
    writeCustomLayouts(next);
    state.layoutPickId = entry.id;
    setNotice("Saved layout “" + trimmed + "”.");
    if (els().layoutModal && !els().layoutModal.classList.contains("hidden")) renderLayoutPicker();
  }

  function deleteCustomLayout(id) {
    const rows = readCustomLayouts();
    const item = rows.find((row) => row.id === id);
    if (!item) return;
    if (!confirm('Delete layout "' + item.name + '"?')) return;
    writeCustomLayouts(rows.filter((row) => row.id !== id));
    if (state.layoutPickId === id) {
      const workplace = state.graph.workplace_type === "massage" ? "massage" : "garage";
      state.layoutPickId = workplace;
    }
    renderLayoutPicker();
  }

  async function deploy() {
    const errors = validate(state.graph);
    if (errors.length) {
      setNotice(errors.join(" "), true);
      refresh();
      return;
    }
    setNotice("Deploying…");
    try {
      const res = await fetch("/api/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.graph),
      });
      const data = await res.json();
      if (!res.ok || data.ok === false) {
        setNotice((data.errors || [data.error || "Engine rejected the graph"]).join(" "), true);
        return;
      }
      markSnapshot();
      setNotice("Pipeline deployed. This graph now controls detection on this engine.");
      if (typeof state.onDeploy === "function") state.onDeploy(data);
    } catch (err) {
      setNotice(err.message || "Could not deploy pipeline.", true);
    }
  }

  function bindOnce() {
    if (state.mounted) return;
    const stage = els().stage;
    if (!stage) return;
    renderPalette();
    stage.addEventListener("pointerdown", startPan);
    stage.addEventListener("pointermove", onPointerMove);
    stage.addEventListener("pointerup", onPointerUp);
    stage.addEventListener("wheel", onWheel, { passive: false });
    stage.addEventListener("dragover", onStageDragOver, true);
    stage.addEventListener("dragleave", onStageDragLeave);
    stage.addEventListener("drop", onStageDrop, true);
    const preview = els().layoutPreview;
    if (preview) {
      preview.addEventListener("pointerdown", startPreviewPan);
      preview.addEventListener("pointermove", onPreviewPointerMove);
      preview.addEventListener("pointerup", onPreviewPointerUp);
      preview.addEventListener("pointercancel", onPreviewPointerUp);
      preview.addEventListener("wheel", onPreviewWheel, { passive: false });
      preview.addEventListener("dblclick", (ev) => {
        ev.preventDefault();
        fitPreviewView();
      });
    }
    document.addEventListener("keydown", (ev) => {
      const workspace = document.getElementById("pipeline-workspace");
      if (!workspace || workspace.classList.contains("hidden")) return;
      const modal = els().layoutModal;
      const modalOpen = modal && !modal.classList.contains("hidden");
      if (ev.key === "Escape") {
        if (modalOpen) {
          ev.preventDefault();
          ev.stopPropagation();
          closeLayoutPicker();
          return;
        }
        if (document.body.classList.contains("pipe-is-fullscreen")) {
          ev.preventDefault();
          setFullscreen(false);
        }
        return;
      }
      if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "TEXTAREA")) return;
      if (ev.key === "Backspace" || ev.key === "Delete") {
        ev.preventDefault();
        ev.stopPropagation();
        if (ev.stopImmediatePropagation) ev.stopImmediatePropagation();
        if (!modalOpen) confirmRemoveSelected();
      }
    }, true);
    const loadBtn = document.getElementById("pipe-load-layout");
    const saveBtn = document.getElementById("pipe-save-layout");
    const deployBtn = document.getElementById("pipe-deploy");
    const expandBtn = els().expand;
    const modal = els().layoutModal;
    if (loadBtn) loadBtn.addEventListener("click", openLayoutPicker);
    if (saveBtn) saveBtn.addEventListener("click", saveCurrentAsLayout);
    if (deployBtn) deployBtn.addEventListener("click", () => { void deploy(); });
    if (expandBtn) {
      expandBtn.addEventListener("pointerdown", (ev) => ev.stopPropagation());
      expandBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        setFullscreen(!document.body.classList.contains("pipe-is-fullscreen"));
      });
    }
    if (modal) {
      modal.addEventListener("click", (ev) => {
        if (ev.target === modal) closeLayoutPicker();
      });
      const cancel = document.getElementById("pipe-layout-cancel");
      const apply = document.getElementById("pipe-layout-apply");
      const saveInModal = document.getElementById("pipe-layout-save");
      if (cancel) cancel.addEventListener("click", closeLayoutPicker);
      if (apply) apply.addEventListener("click", applyPickedLayout);
      if (saveInModal) saveInModal.addEventListener("click", saveCurrentAsLayout);
    }
    state.mounted = true;
  }

  global.PipelineGraph = {
    open: function (workplace, cameraList) {
      bindOnce();
      if (Array.isArray(cameraList)) state.cameras = cameraList;
      if (state.ready) {
        refresh();
        applyView();
        return;
      }
      void loadFromEngine(workplace);
    },
    setCameras: function (cameraList) {
      if (!Array.isArray(cameraList)) return;
      state.cameras = cameraList;
      if (state.ready) refresh();
    },
    onAbsentSecondsChange: function (fn) {
      state.onAbsentSecondsChange = typeof fn === "function" ? fn : null;
    },
    onDeploy: function (fn) {
      state.onDeploy = typeof fn === "function" ? fn : null;
    },
    setFullscreen: setFullscreen,
  };
})(window);
