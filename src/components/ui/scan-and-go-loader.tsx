import React, { useEffect, useState, useMemo } from "react";
import "../../theme/digital-overwatch.css";

export type BusinessSector =
  | "facility"
  | "inventory"
  | "occupancy"
  | "edge_node"
  | "verified";

export interface SectorMeta {
  id: BusinessSector;
  title: string;
  category: string;
  telemetryCode: string;
  detail: string;
}

export const BUSINESS_SECTORS: Record<BusinessSector, SectorMeta> = {
  facility: {
    id: "facility",
    title: "COMMERCIAL VENUE",
    category: "Real Estate & Retail",
    telemetryCode: "LOC: SECTOR-01 // STOREFRONT",
    detail: "Scanning entrance gates & perimeter clearance",
  },
  inventory: {
    id: "inventory",
    title: "ASSET LOGISTICS",
    category: "Supply Chain & Storage",
    telemetryCode: "SKU: 884-INB // OPTICAL TRACK",
    detail: "Verifying parcel barcodes & bay clearance",
  },
  occupancy: {
    id: "occupancy",
    title: "CROWD OCCUPANCY",
    category: "Venue Safety & Flow",
    telemetryCode: "TGT: DENSITY // 98.4% CONF",
    detail: "Tracking pedestrian lanes & dwell thresholds",
  },
  edge_node: {
    id: "edge_node",
    title: "EDGE AI COMPUTE",
    category: "Infrastructure & Telemetry",
    telemetryCode: "NODE: RTSP-ONVIF // 30 FPS",
    detail: "Synchronizing YOLO weights & RTSP feed",
  },
  verified: {
    id: "verified",
    title: "ANOMALY RESOLVED",
    category: "Safety & Compliance",
    telemetryCode: "STATUS: MATRIX SECURE [OK]",
    detail: "0 Active breaches // Telegram alert idle",
  },
};

export interface ScanAndGoLoaderProps {
  /** Square size in px or preset */
  size?: "sm" | "md" | "lg" | "xl" | number;
  /** Active icon subset to cycle */
  sectors?: BusinessSector[];
  /** Milliseconds per sector in cycle (default 2400ms) */
  cycleDurationMs?: number;
  /** Custom overlay text */
  statusLabel?: string;
  /** Display cyber HUD coordinates and telemetry data */
  showTelemetry?: boolean;
  /** Show manual play/pause and sector jump bar */
  showControls?: boolean;
  /** Extra container className */
  className?: string;
}

export function ScanAndGoLoader({
  size = "md",
  sectors = ["facility", "inventory", "occupancy", "edge_node", "verified"],
  cycleDurationMs = 2400,
  statusLabel,
  showTelemetry = true,
  showControls = false,
  className = "",
}: ScanAndGoLoaderProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [scanTick, setScanTick] = useState(0);

  const activeSectors = useMemo(() => {
    return sectors.length > 0
      ? sectors
      : (["facility", "inventory", "occupancy", "edge_node", "verified"] as BusinessSector[]);
  }, [sectors]);

  // Handle cycle timer
  useEffect(() => {
    if (!isPlaying || activeSectors.length <= 1) return;

    const timer = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % activeSectors.length);
      setScanTick((t) => (t + 1) % 9999);
    }, cycleDurationMs);

    return () => clearInterval(timer);
  }, [isPlaying, activeSectors.length, cycleDurationMs]);

  const currentSectorKey = activeSectors[currentIndex] || "facility";
  const currentSector = BUSINESS_SECTORS[currentSectorKey] || BUSINESS_SECTORS.facility;

  // Numeric pixel dimension
  const pixelSize = useMemo(() => {
    if (typeof size === "number") return size;
    switch (size) {
      case "sm":
        return 180;
      case "md":
        return 280;
      case "lg":
        return 380;
      case "xl":
        return 480;
      default:
        return 280;
    }
  }, [size]);

  return (
    <div
      className={`scan-and-go-container theme-digital-overwatch ${className}`}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--color-bg-base)",
        borderRadius: "16px",
        padding: showTelemetry ? "24px" : "12px",
        border: "1px solid var(--border-stealth)",
        boxShadow: "var(--glow-stealth), inset 0 0 40px rgba(0, 0, 0, 0.95)",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
        color: "var(--color-crisp-white)",
        userSelect: "none",
        maxWidth: "100%",
      }}
    >
      {/* Top HUD Telemetry Bar */}
      {showTelemetry && (
        <div
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "0.72rem",
            letterSpacing: "0.08em",
            color: "var(--color-text-muted)",
            borderBottom: "1px solid var(--border-subtle)",
            paddingBottom: "10px",
            marginBottom: "16px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: "var(--color-surveillance-green)",
                boxShadow: "var(--glow-laser)",
                display: "inline-block",
                animation: "telemetry-flicker 2s infinite",
              }}
            />
            <span style={{ color: "var(--color-surveillance-green)", fontWeight: 700 }}>
              INB-AGENTIC-VISION
            </span>
          </div>
          <div style={{ color: "var(--color-text-muted)" }}>
            SCAN_SEQ: #{String(scanTick).padStart(4, "0")}
          </div>
        </div>
      )}

      {/* Main SVG Mascot & HUD Iris Canvas */}
      <div
        style={{
          position: "relative",
          width: pixelSize,
          height: pixelSize,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg
          viewBox="0 0 300 300"
          width="100%"
          height="100%"
          style={{
            overflow: "visible",
            filter: "drop-shadow(0 0 16px rgba(11, 131, 58, 0.25))",
          }}
        >
          <defs>
            {/* Stealth Green Trailing Beam Gradient */}
            <linearGradient id="scanBeamGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#00FF66" stopOpacity="0.8" />
              <stop offset="30%" stopColor="#0B833A" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#0B833A" stopOpacity="0" />
            </linearGradient>

            {/* Subtle Ocular Lens Flare Filter */}
            <filter id="surveillanceGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="blur1" />
              <feGaussianBlur in="SourceGraphic" stdDeviation="8" result="blur2" />
              <feMerge>
                <feMergeNode in="blur2" />
                <feMergeNode in="blur1" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <filter id="stealthTrail" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* ======================================================== */}
          {/* LAYER 1: MONKEY MASCOT SILHOUETTE (Crisp White & Black)   */}
          {/* ======================================================== */}
          <g id="mascot-base">
            {/* Outer Cyber Ear-Muffs / Headset Chassis */}
            <path
              d="M 45 140 C 35 125 38 95 55 85 C 70 75 75 90 75 110"
              fill="#121212"
              stroke="#FAFAFA"
              strokeWidth="3.5"
              strokeLinecap="round"
            />
            <path
              d="M 255 140 C 265 125 262 95 245 85 C 230 75 225 90 225 110"
              fill="#121212"
              stroke="#FAFAFA"
              strokeWidth="3.5"
              strokeLinecap="round"
            />
            {/* Headset Over-Arch */}
            <path
              d="M 55 90 C 80 40 220 40 245 90"
              fill="none"
              stroke="#FAFAFA"
              strokeWidth="4"
              strokeLinecap="round"
            />
            <path
              d="M 70 85 C 100 48 200 48 230 85"
              fill="none"
              stroke="#0B833A"
              strokeWidth="2"
              strokeDasharray="4 6"
            />

            {/* Monkey Head Silhouette (Crisp White outline + Absolute Black fill) */}
            <path
              d="M 80 110 C 80 65 220 65 220 110 C 235 155 220 220 150 225 C 80 220 65 155 80 110 Z"
              fill="#000000"
              stroke="#FAFAFA"
              strokeWidth="3.5"
              strokeLinejoin="round"
            />

            {/* Brow & Cyber Crest Plate */}
            <path
              d="M 95 105 L 150 120 L 205 105 L 195 90 L 105 90 Z"
              fill="#121212"
              stroke="#FAFAFA"
              strokeWidth="2"
            />

            {/* Monkey Muzzle & Jaw Contour */}
            <path
              d="M 105 175 C 105 155 120 150 150 150 C 180 150 195 155 195 175 C 195 210 175 218 150 218 C 125 218 105 210 105 175 Z"
              fill="#121212"
              stroke="#FAFAFA"
              strokeWidth="2.5"
            />

            {/* Nostril Optical Vents */}
            <ellipse cx="140" cy="180" rx="3" ry="4.5" fill="#FAFAFA" transform="rotate(-15 140 180)" />
            <ellipse cx="160" cy="180" rx="3" ry="4.5" fill="#FAFAFA" transform="rotate(15 160 180)" />

            {/* Mechanical Jaw Seam */}
            <path
              d="M 135 200 Q 150 206 165 200"
              stroke="#00FF66"
              strokeWidth="2"
              strokeLinecap="round"
              fill="none"
            />
          </g>

          {/* ======================================================== */}
          {/* LAYER 2: ACTIVE SURVEILLANCE OPTICAL IRIS (Green/Stealth) */}
          {/* ======================================================== */}
          <g id="optical-eye-reticle" transform="translate(150, 125)">
            {/* Stealth Green Ocular Outer Aura */}
            <circle
              cx="0"
              cy="0"
              r="48"
              fill="rgba(11, 131, 58, 0.15)"
              stroke="#0B833A"
              strokeWidth="1.5"
              strokeDasharray="6 4"
            />

            {/* Rotating Outer HUD Caliper Ring (Clockwise) */}
            <g
              style={{
                transformOrigin: "0 0",
                animation: "iris-rotate-cw 12s linear infinite",
              }}
            >
              <circle
                cx="0"
                cy="0"
                r="42"
                fill="none"
                stroke="#00FF66"
                strokeWidth="1.5"
                strokeDasharray="28 14 8 14"
                opacity="0.9"
              />
              <circle cx="0" cy="-42" r="2.5" fill="#00FF66" />
              <circle cx="0" cy="42" r="2.5" fill="#00FF66" />
            </g>

            {/* Counter-Rotating Aperture Calipers (CCW) */}
            <g
              style={{
                transformOrigin: "0 0",
                animation: "iris-rotate-ccw 8s linear infinite",
              }}
            >
              <circle
                cx="0"
                cy="0"
                r="36"
                fill="none"
                stroke="#0B833A"
                strokeWidth="2"
                strokeDasharray="6 12 18 12"
              />
              {/* Aperture HUD Ticks */}
              <line x1="-36" y1="0" x2="-30" y2="0" stroke="#00FF66" strokeWidth="1.5" />
              <line x1="36" y1="0" x2="30" y2="0" stroke="#00FF66" strokeWidth="1.5" />
              <line x1="0" y1="-36" x2="0" y2="-30" stroke="#00FF66" strokeWidth="1.5" />
              <line x1="0" y1="36" x2="0" y2="30" stroke="#00FF66" strokeWidth="1.5" />
            </g>

            {/* Inner Lens Glass Chamber */}
            <circle
              cx="0"
              cy="0"
              r="28"
              fill="#000000"
              stroke="#00FF66"
              strokeWidth="2"
              style={{
                animation: "iris-pulse-glow 3s ease-in-out infinite",
              }}
            />

            {/* 4 Corner Targeting Lock Brackets [ ] */}
            <g
              style={{
                transformOrigin: "0 0",
                animation: "hud-bracket-pulse 2s ease-in-out infinite",
              }}
            >
              {/* Top-Left */}
              <path d="M -24 -16 L -24 -24 L -16 -24" fill="none" stroke="#00FF66" strokeWidth="2" />
              {/* Top-Right */}
              <path d="M 16 -24 L 24 -24 L 24 -16" fill="none" stroke="#00FF66" strokeWidth="2" />
              {/* Bottom-Left */}
              <path d="M -24 16 L -24 24 L -16 24" fill="none" stroke="#00FF66" strokeWidth="2" />
              {/* Bottom-Right */}
              <path d="M 16 24 L 24 24 L 24 16" fill="none" stroke="#00FF66" strokeWidth="2" />
            </g>

            {/* ======================================================== */}
            {/* LAYER 3: DYNAMIC GENERAL BUSINESS ICONS CYCLING IN RETICLE*/}
            {/* ======================================================== */}
            <g
              key={currentSectorKey}
              style={{
                animation: "icon-fade-in-scale 2.4s cubic-bezier(0.16, 1, 0.3, 1) infinite",
              }}
            >
              {/* 1. Commercial Venue / Storefront Icon */}
              {currentSectorKey === "facility" && (
                <g id="icon-facility" stroke="#FAFAFA" strokeWidth="1.8" fill="none">
                  {/* Building Base */}
                  <path d="M -14 16 L -14 -8 L 14 -8 L 14 16 Z" fill="#121212" />
                  {/* Roof Top Awning / Surveillance Mast */}
                  <path d="M -18 -8 L 0 -17 L 18 -8 Z" fill="#121212" stroke="#00FF66" />
                  {/* Windows / Storefront Glass */}
                  <line x1="-8" y1="-2" x2="-2" y2="-2" stroke="#00FF66" />
                  <line x1="2" y1="-2" x2="8" y2="-2" stroke="#00FF66" />
                  <line x1="-8" y1="4" x2="-2" y2="4" stroke="#00FF66" />
                  <line x1="2" y1="4" x2="8" y2="4" stroke="#00FF66" />
                  {/* Entrance Doorway */}
                  <rect x="-4" y="9" width="8" height="7" stroke="#FAFAFA" fill="#00FF66" fillOpacity="0.3" />
                  {/* Roof Camera Node */}
                  <circle cx="0" cy="-17" r="2.5" fill="#00FF66" stroke="#FAFAFA" strokeWidth="1" />
                </g>
              )}

              {/* 2. Asset / Inventory / Package Box Icon */}
              {currentSectorKey === "inventory" && (
                <g id="icon-inventory" stroke="#FAFAFA" strokeWidth="1.8" fill="none">
                  {/* Isometric Box Faces */}
                  <path d="M 0 -14 L 14 -6 L 0 2 L -14 -6 Z" fill="#1A1A1A" stroke="#00FF66" />
                  <path d="M -14 -6 L 0 2 L 0 16 L -14 8 Z" fill="#121212" />
                  <path d="M 14 -6 L 0 2 L 0 16 L 14 8 Z" fill="#121212" />
                  {/* Barcode / Tracking Lines on Front Face */}
                  <line x1="-10" y1="2" x2="-7" y2="4" stroke="#00FF66" strokeWidth="1.4" />
                  <line x1="-5" y1="5" x2="-3" y2="6" stroke="#00FF66" strokeWidth="1.4" />
                  <line x1="-1" y1="7" x2="0" y2="8" stroke="#00FF66" strokeWidth="1.4" />
                  {/* RFID Tracking Beacon */}
                  <circle cx="7" cy="6" r="2" fill="#00FF66" />
                </g>
              )}

              {/* 3. Crowd Occupancy / Pedestrian Flow Icon */}
              {currentSectorKey === "occupancy" && (
                <g id="icon-occupancy" stroke="#FAFAFA" strokeWidth="1.8" fill="none">
                  {/* Agentic Vision Bounding Box */}
                  <rect
                    x="-15"
                    y="-16"
                    width="30"
                    height="32"
                    stroke="#00FF66"
                    strokeWidth="1.2"
                    strokeDasharray="4 2"
                    fill="rgba(0, 255, 102, 0.08)"
                  />
                  {/* Person Head */}
                  <circle cx="0" cy="-8" r="4.5" fill="#FAFAFA" stroke="#00FF66" />
                  {/* Person Torso */}
                  <path
                    d="M -7 9 C -7 2 -4 -1 0 -1 C 4 -1 7 2 7 9"
                    fill="#121212"
                    stroke="#FAFAFA"
                  />
                  {/* Tracking Crosshair Dot */}
                  <circle cx="0" cy="4" r="1.5" fill="#00FF66" />
                  <text
                    x="-13"
                    y="-18"
                    fontSize="5"
                    fill="#00FF66"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    PERSON
                  </text>
                </g>
              )}

              {/* 4. Edge Server / Cloud Node Icon */}
              {currentSectorKey === "edge_node" && (
                <g id="icon-edge-node" stroke="#FAFAFA" strokeWidth="1.8" fill="none">
                  {/* Rack Outer Chassis */}
                  <rect x="-13" y="-15" width="26" height="30" rx="2" fill="#121212" stroke="#FAFAFA" />
                  {/* Blade 1 */}
                  <line x1="-13" y1="-6" x2="13" y2="-6" stroke="#FAFAFA" strokeWidth="1.2" />
                  <circle cx="-8" cy="-10" r="1.5" fill="#00FF66" />
                  <line x1="-3" y1="-10" x2="8" y2="-10" stroke="#00FF66" strokeWidth="1.2" />
                  {/* Blade 2 */}
                  <line x1="-13" y1="4" x2="13" y2="4" stroke="#FAFAFA" strokeWidth="1.2" />
                  <circle cx="-8" cy="-1" r="1.5" fill="#00FF66" />
                  <line x1="-3" y1="-1" x2="8" y2="-1" stroke="#00FF66" strokeWidth="1.2" />
                  {/* Blade 3 */}
                  <circle cx="-8" cy="9" r="1.5" fill="#00FF66" />
                  <line x1="-3" y1="9" x2="8" y2="9" stroke="#00FF66" strokeWidth="1.2" />
                </g>
              )}

              {/* 5. Verified Shield / Anomaly Free Icon */}
              {currentSectorKey === "verified" && (
                <g id="icon-verified" stroke="#00FF66" strokeWidth="2" fill="none">
                  {/* Shield Perimeter */}
                  <path
                    d="M -13 -13 L 0 -17 L 13 -13 C 13 4 8 13 0 17 C -8 13 -13 4 -13 -13 Z"
                    fill="#121212"
                    stroke="#00FF66"
                  />
                  {/* Verified Crisp Checkmark */}
                  <path
                    d="M -6 -1 L -2 4 L 7 -5"
                    stroke="#FAFAFA"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </g>
              )}
            </g>

            {/* ======================================================== */}
            {/* LAYER 4: ACTIVE SCANNING LASER BEAM (Surveillance Green)  */}
            {/* ======================================================== */}
            <g clipPath="url(#lensClip)">
              <clipPath id="lensClip">
                <circle cx="0" cy="0" r="28" />
              </clipPath>
              <g
                style={{
                  animation: "scan-laser-sweep 1.8s ease-in-out infinite alternate",
                }}
              >
                {/* Horizontal Laser Line */}
                <line
                  x1="-35"
                  y1="-28"
                  x2="35"
                  y2="-28"
                  stroke="#00FF66"
                  strokeWidth="2.5"
                  filter="url(#surveillanceGlow)"
                />
                {/* Trailing Stealth Shadow Gradient */}
                <rect
                  x="-35"
                  y="-42"
                  width="70"
                  height="14"
                  fill="url(#scanBeamGrad)"
                />
              </g>
            </g>
          </g>
        </svg>
      </div>

      {/* Dynamic Status / Sector Readout Panel */}
      <div
        style={{
          marginTop: "14px",
          textAlign: "center",
          width: "100%",
        }}
      >
        <div
          style={{
            fontSize: "0.85rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
            color: "var(--color-crisp-white)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "8px",
          }}
        >
          <span style={{ color: "var(--color-surveillance-green)" }}>▶</span>
          <span>{statusLabel || currentSector.title}</span>
        </div>

        {showTelemetry && (
          <div
            style={{
              fontSize: "0.7rem",
              color: "var(--color-stealth-green)",
              fontWeight: 600,
              marginTop: "4px",
              letterSpacing: "0.04em",
            }}
          >
            {currentSector.telemetryCode}
          </div>
        )}

        {showTelemetry && (
          <div
            style={{
              fontSize: "0.68rem",
              color: "var(--color-text-muted)",
              marginTop: "2px",
            }}
          >
            {currentSector.detail}
          </div>
        )}
      </div>

      {/* Optional Interactive Control Bar */}
      {showControls && (
        <div
          style={{
            marginTop: "16px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
            background: "var(--color-surface-charcoal)",
            padding: "6px 12px",
            borderRadius: "20px",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <button
            type="button"
            onClick={() => setIsPlaying((p) => !p)}
            style={{
              background: "transparent",
              border: "1px solid var(--border-subtle)",
              color: isPlaying ? "var(--color-surveillance-green)" : "var(--color-text-muted)",
              borderRadius: "4px",
              padding: "2px 8px",
              fontSize: "0.7rem",
              cursor: "pointer",
            }}
          >
            {isPlaying ? "PAUSE" : "SCAN"}
          </button>
          <div style={{ display: "flex", gap: "4px" }}>
            {activeSectors.map((sectorKey, idx) => (
              <button
                key={sectorKey}
                type="button"
                onClick={() => {
                  setCurrentIndex(idx);
                  setIsPlaying(false);
                }}
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  border: "none",
                  padding: 0,
                  background:
                    idx === currentIndex
                      ? "var(--color-surveillance-green)"
                      : "var(--color-stealth-green)",
                  opacity: idx === currentIndex ? 1 : 0.4,
                  cursor: "pointer",
                  boxShadow: idx === currentIndex ? "var(--glow-laser)" : "none",
                }}
                title={BUSINESS_SECTORS[sectorKey]?.title}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
