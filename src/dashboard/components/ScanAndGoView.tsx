import { useState } from "react";
import {
  ScanAndGoLoader,
  BUSINESS_SECTORS,
  type BusinessSector,
} from "../../components/ui/scan-and-go-loader";
import { DIGITAL_OVERWATCH_PALETTE } from "../../theme/tokens";

export function ScanAndGoView() {
  const [size, setSize] = useState<"sm" | "md" | "lg" | "xl">("md");
  const [selectedSectors, setSelectedSectors] = useState<BusinessSector[]>([
    "facility",
    "inventory",
    "occupancy",
    "edge_node",
    "verified",
  ]);
  const [speed, setSpeed] = useState<number>(2400);
  const [showTelemetry, setShowTelemetry] = useState<boolean>(true);
  const [showControls, setShowControls] = useState<boolean>(true);
  const [customStatus, setCustomStatus] = useState<string>("");

  const toggleSector = (sec: BusinessSector) => {
    setSelectedSectors((prev) =>
      prev.includes(sec)
        ? prev.length > 1
          ? prev.filter((s) => s !== sec)
          : prev
        : [...prev, sec]
    );
  };

  return (
    <section className="panel" style={{ background: "var(--color-bg-base)" }}>
      <header className="panel__head">
        <div>
          <h2>"Digital Overwatch" & Scan and Go Loader</h2>
          <p>
            Constrained high-contrast palette with Crisp White & Absolute Black mascot,
            Surveillance Green active optical iris, Stealth Green trailing glow shadows,
            and 5 general business scanning targets.
          </p>
        </div>
      </header>

      {/* Palette Color Swatches */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "12px",
          padding: "16px 20px",
          borderBottom: "1px solid var(--border-subtle)",
        }}
      >
        {Object.entries(DIGITAL_OVERWATCH_PALETTE).map(([key, color]) => (
          <div
            key={key}
            style={{
              background: "var(--color-surface-charcoal)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "10px",
              padding: "12px",
              display: "flex",
              flexDirection: "column",
              gap: "6px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span
                style={{
                  width: "20px",
                  height: "20px",
                  borderRadius: "6px",
                  background: color.hex,
                  border: "1px solid rgba(255,255,255,0.2)",
                  boxShadow:
                    color.hex === "#00FF66"
                      ? "0 0 10px rgba(0,255,102,0.6)"
                      : color.hex === "#0B833A"
                      ? "0 0 10px rgba(11,131,58,0.5)"
                      : "none",
                }}
              />
              <strong style={{ fontSize: "0.85rem", color: "#FAFAFA" }}>
                {key.replace(/([A-Z])/g, " $1").trim()}
              </strong>
            </div>
            <code style={{ fontSize: "0.75rem", color: "var(--color-surveillance-green)" }}>
              {color.hex} · rgb({color.rgbString})
            </code>
            <p style={{ fontSize: "0.7rem", color: "var(--color-text-muted)", margin: 0 }}>
              {color.usage}
            </p>
          </div>
        ))}
      </div>

      {/* Interactive Playground & Stage */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr minmax(280px, 340px)",
          gap: "24px",
          padding: "24px 20px",
          alignItems: "start",
        }}
      >
        {/* Visual Stage */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            background: "radial-gradient(circle at center, #0b1c11 0%, #000000 70%)",
            border: "1px solid var(--border-stealth)",
            borderRadius: "16px",
            padding: "40px 20px",
            minHeight: "440px",
            boxShadow: "var(--glow-stealth)",
          }}
        >
          <ScanAndGoLoader
            size={size}
            sectors={selectedSectors}
            cycleDurationMs={speed}
            statusLabel={customStatus || undefined}
            showTelemetry={showTelemetry}
            showControls={showControls}
          />
        </div>

        {/* Configuration Controls */}
        <div
          style={{
            background: "var(--color-surface-charcoal)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "14px",
            padding: "20px",
            display: "flex",
            flexDirection: "column",
            gap: "18px",
          }}
        >
          <h3 style={{ fontSize: "0.95rem", color: "#FAFAFA", margin: 0 }}>
            Loader Controls & Parameters
          </h3>

          {/* Size Preset */}
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", display: "block", marginBottom: "6px" }}>
              Component Size:
            </label>
            <div style={{ display: "flex", gap: "6px" }}>
              {(["sm", "md", "lg", "xl"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSize(s)}
                  style={{
                    flex: 1,
                    background: size === s ? "var(--color-surveillance-green)" : "var(--color-surface-elevated)",
                    color: size === s ? "#000000" : "#FAFAFA",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "6px",
                    padding: "6px 0",
                    fontWeight: 700,
                    fontSize: "0.75rem",
                    cursor: "pointer",
                  }}
                >
                  {s.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Active Business Sectors */}
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", display: "block", marginBottom: "8px" }}>
              Scanning Target Icons ({selectedSectors.length}/5):
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              {(Object.keys(BUSINESS_SECTORS) as BusinessSector[]).map((secKey) => {
                const sec = BUSINESS_SECTORS[secKey];
                const active = selectedSectors.includes(secKey);
                return (
                  <button
                    key={secKey}
                    type="button"
                    onClick={() => toggleSector(secKey)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      background: active ? "rgba(0, 255, 102, 0.08)" : "var(--color-surface-elevated)",
                      border: `1px solid ${active ? "var(--color-surveillance-green)" : "var(--border-subtle)"}`,
                      borderRadius: "6px",
                      padding: "8px 10px",
                      cursor: "pointer",
                      textAlign: "left",
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: "0.75rem", color: active ? "#00FF66" : "#FAFAFA", display: "block" }}>
                        {sec.title}
                      </strong>
                      <span style={{ fontSize: "0.68rem", color: "var(--color-text-muted)" }}>
                        {sec.category}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.75rem", color: active ? "#00FF66" : "#666" }}>
                      {active ? "✓" : "+"}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Cycle Speed */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
              <label style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
                Cycle Speed:
              </label>
              <span style={{ fontSize: "0.75rem", color: "var(--color-surveillance-green)", fontFamily: "monospace" }}>
                {(speed / 1000).toFixed(1)}s / target
              </span>
            </div>
            <input
              type="range"
              min="1000"
              max="5000"
              step="200"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              style={{ width: "100%", accentColor: "var(--color-surveillance-green)" }}
            />
          </div>

          {/* Custom Status Label */}
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", display: "block", marginBottom: "6px" }}>
              Custom Overlay Label:
            </label>
            <input
              type="text"
              placeholder="e.g. INGESTING RTSP STREAMS..."
              value={customStatus}
              onChange={(e) => setCustomStatus(e.target.value)}
              style={{
                width: "100%",
                background: "var(--color-surface-elevated)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "6px",
                padding: "6px 10px",
                color: "#FAFAFA",
                fontSize: "0.75rem",
                boxSizing: "border-box",
              }}
            />
          </div>

          {/* Toggles */}
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem", color: "#FAFAFA", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={showTelemetry}
                onChange={(e) => setShowTelemetry(e.target.checked)}
                style={{ accentColor: "var(--color-surveillance-green)" }}
              />
              Show Telemetry & HUD Readout
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem", color: "#FAFAFA", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={showControls}
                onChange={(e) => setShowControls(e.target.checked)}
                style={{ accentColor: "var(--color-surveillance-green)" }}
              />
              Show Manual Step & Pause Bar
            </label>
          </div>
        </div>
      </div>
    </section>
  );
}
