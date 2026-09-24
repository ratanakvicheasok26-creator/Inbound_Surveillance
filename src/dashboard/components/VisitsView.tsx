import { useEffect, useState } from "react";
import { engineBaseUrl } from "../../engine-url";
import { engineApi } from "../../lib/engineApi";
import type { DetailedVisitReport, DetailedVisitSummary, VisitorProfile, OpenSessionItem } from "../types";

function defaultSummary(): DetailedVisitSummary {
  return {
    total_unique: 0,
    today_unique: 0,
    week_unique: 0,
    today_visits: 0,
    week_visits: 0,
    total_visits: 0,
    avg_dwell_seconds: 0,
    returning_rate: 0,
    active_now_count: 0,
  };
}

function defaultReport(): DetailedVisitReport {
  return {
    summary: defaultSummary(),
    visitors: [],
    recent_visits: [],
    open_sessions: [],
  };
}

function formatDwell(seconds?: number | null): string {
  if (seconds == null || isNaN(seconds) || seconds <= 0) return "—";
  const s = Math.round(Number(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const remS = s % 60;
  if (h > 0) return `${h}h ${m}m ${remS}s`;
  if (m > 0) return `${m}m ${remS}s`;
  return `${remS}s`;
}

function formatTime(isoStr?: string | null): string {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return isoStr;
  }
}

function formatDate(isoStr?: string | null): string {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    if (isNaN(d.getTime())) return isoStr;
    return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + formatTime(isoStr);
  } catch {
    return isoStr;
  }
}

export function VisitsView() {
  const [report, setReport] = useState<DetailedVisitReport>(defaultReport());
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"recent" | "frequency" | "dwell">("recent");
  const [feedback, setFeedback] = useState<string | null>(null);
  const engine = engineBaseUrl();

  async function loadData() {
    try {
      const res = await engineApi(`/api/workplace/visits/detailed?range=today`, { cache: "no-store" });
      if (res.ok) {
        const data = (await res.json()) as DetailedVisitReport;
        if (data && typeof data === "object") {
          setReport({
            summary: { ...defaultSummary(), ...(data.summary || {}) },
            visitors: Array.isArray(data.visitors) ? data.visitors : [],
            recent_visits: Array.isArray(data.recent_visits) ? data.recent_visits : [],
            open_sessions: Array.isArray(data.open_sessions) ? data.open_sessions : [],
          });
        }
      }
    } catch (err: any) {
      console.warn("[VisitsView] Failed to fetch detailed visits:", err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadData();
    const timer = window.setInterval(() => void loadData(), 5000);
    return () => window.clearInterval(timer);
  }, [engine]);

  async function handleReset() {
    if (!confirm("⚠️ ARE YOU SURE YOU WANT TO RESET ALL VISITS?\n\nThis will permanently delete all visitor attendance logs, stay histories, and anonymous profiles from the system.")) {
      return;
    }
    try {
      const res = await engineApi(`/api/workplace/visits/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.ok) {
        setFeedback("All customer visits successfully reset.");
        void loadData();
      } else {
        alert("Failed to reset: " + (data.error || "Unknown error"));
      }
    } catch (err: any) {
      alert("Error resetting visits: " + err.message);
    }
  }

  async function handleEditAlias(subjectId: string, currentAlias?: string | null) {
    const next = prompt("Enter a customer name or label (e.g. 'Alice Regular', 'John VIP'):", currentAlias || "");
    if (next === null) return;
    try {
      const res = await engineApi(`/api/workplace/visits/alias`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject_id: subjectId, alias: next.trim() }),
      });
      const data = await res.json();
      if (data.ok) {
        void loadData();
      } else {
        alert("Failed to update alias: " + (data.error || "Unknown error"));
      }
    } catch (err: any) {
      alert("Error updating alias: " + err.message);
    }
  }

  const { summary, visitors, open_sessions } = report;

  const filteredVisitors = visitors.filter((v: VisitorProfile) => {
    if (!search) return true;
    const q = search.toLowerCase();
    const sid = (v.subject_id || "").toLowerCase();
    const alias = (v.alias || "").toLowerCase();
    return sid.includes(q) || alias.includes(q);
  });

  if (sortBy === "recent") {
    filteredVisitors.sort((a, b) => (b.last_seen_at || "").localeCompare(a.last_seen_at || ""));
  } else if (sortBy === "frequency") {
    filteredVisitors.sort((a, b) => (b.total_visits || 0) - (a.total_visits || 0));
  } else if (sortBy === "dwell") {
    filteredVisitors.sort((a, b) => (b.avg_dwell_seconds || 0) - (a.avg_dwell_seconds || 0));
  }

  return (
    <section className="panel space-y-4">
      <header className="panel__head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <h2>Customer Attendance & Visits</h2>
          <p>
            Multi-angle appearance matching, dwell times, and individual customer profiles.
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <button
            type="button"
            className="btn"
            style={{ border: "1px solid rgba(255, 85, 85, 0.4)", color: "#ff5555", background: "rgba(255, 85, 85, 0.08)", fontSize: "12px", padding: "6px 12px" }}
            onClick={handleReset}
          >
            Reset All Visits
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: "12px", padding: "6px 12px" }}
            onClick={() => { setLoading(true); void loadData(); }}
          >
            Refresh
          </button>
        </div>
      </header>

      {feedback && (
        <div style={{ padding: "8px 14px", borderRadius: "8px", background: "rgba(0, 255, 102, 0.1)", border: "1px solid rgba(0, 255, 102, 0.3)", color: "var(--color-surveillance-green)", fontSize: "12px" }}>
          {feedback}
        </div>
      )}

      {/* 5-Card Summary Grid */}
      <div className="visit-stats" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))" }}>
        <article>
          <span>Today Unique</span>
          <strong>{summary.today_unique}</strong>
          <small style={{ color: "var(--muted)", fontSize: "11px" }}>{summary.today_visits} visits total</small>
        </article>
        <article>
          <span>7-Day Unique</span>
          <strong>{summary.week_unique}</strong>
          <small style={{ color: "var(--muted)", fontSize: "11px" }}>{summary.week_visits} past 7d</small>
        </article>
        <article>
          <span>Average Stay</span>
          <strong>{formatDwell(summary.avg_dwell_seconds)}</strong>
          <small style={{ color: "var(--muted)", fontSize: "11px" }}>Completed visits</small>
        </article>
        <article>
          <span>Returning Rate</span>
          <strong>{summary.returning_rate}%</strong>
          <small style={{ color: "var(--muted)", fontSize: "11px" }}>Multiple visits</small>
        </article>
        <article style={{ border: "1px solid rgba(0, 255, 102, 0.35)", background: "rgba(0, 255, 102, 0.04)" }}>
          <span style={{ color: "var(--color-surveillance-green)" }}>In Venue Now</span>
          <strong>{open_sessions.length}</strong>
          <small style={{ color: "var(--color-surveillance-green)", opacity: 0.8, fontSize: "11px" }}>Active inside zones</small>
        </article>
      </div>

      {/* Active In-Venue Visitors Cards */}
      {open_sessions.length > 0 && (
        <div style={{ padding: "16px", borderRadius: "12px", background: "var(--bg-3)", border: "1px solid rgba(0, 255, 102, 0.25)" }}>
          <h3 style={{ fontSize: "13px", fontWeight: 600, color: "var(--color-surveillance-green)", marginBottom: "12px", display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--color-surveillance-green)", display: "inline-block" }}></span>
            Currently Inside Venue ({open_sessions.length})
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "10px" }}>
            {open_sessions.map((session: OpenSessionItem) => {
              const sid = session.subject_id || "visitor";
              const avatarUrl = session.avatar_url || `${engine}/api/workplace/avatar/${sid}`;
              return (
                <div key={`${sid}-${session.zone_id}`} style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px", borderRadius: "8px", background: "var(--bg-2)", border: "1px solid var(--line)" }}>
                  <img
                    src={avatarUrl}
                    alt="avatar"
                    style={{ width: "42px", height: "42px", borderRadius: "8px", objectFit: "cover", background: "#1a1a1a", border: "1px solid var(--line)" }}
                    onError={(e) => { (e.target as HTMLElement).style.display = "none"; }}
                  />
                  <div style={{ minWidth: 0, flex: 1 }}>
                    <div style={{ fontWeight: 600, fontSize: "13px", color: "var(--color-crisp-white)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {sid}
                    </div>
                    <div style={{ fontSize: "11px", color: "var(--muted)" }}>In {session.zone_name || session.zone_id}</div>
                    <div style={{ fontSize: "11px", color: "var(--color-surveillance-green)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>
                      ⏱️ {formatDwell(session.dwell_seconds)}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Directory Filter & Search Bar */}
      <div style={{ padding: "16px", borderRadius: "12px", background: "var(--bg-3)", border: "1px solid var(--line)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px", marginBottom: "14px" }}>
          <h3 style={{ fontSize: "14px", fontWeight: 600, margin: 0 }}>Customer Profiles & Visit History</h3>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <input
              type="text"
              placeholder="Search visitor ID or name..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{ padding: "6px 10px", borderRadius: "6px", fontSize: "12px", background: "var(--bg-2)", border: "1px solid var(--line)", color: "#fff", width: "180px" }}
            />
            <select
              value={sortBy}
              onChange={(e: any) => setSortBy(e.target.value)}
              style={{ padding: "6px 10px", borderRadius: "6px", fontSize: "12px", background: "var(--bg-2)", border: "1px solid var(--line)", color: "#fff" }}
            >
              <option value="recent">Sort: Most Recent</option>
              <option value="frequency">Sort: Most Frequent</option>
              <option value="dwell">Sort: Longest Stay</option>
            </select>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: "32px", textAlign: "center", color: "var(--muted)", fontSize: "13px" }}>
            Loading customer visits report…
          </div>
        ) : filteredVisitors.length === 0 ? (
          <div style={{ padding: "32px", textAlign: "center", color: "var(--muted)", fontSize: "13px" }}>
            <p>No visitor records found.</p>
            <p style={{ fontSize: "11px", marginTop: "4px" }}>Visitors will automatically appear here once detected on camera.</p>
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--line)", color: "var(--muted)" }}>
                  <th style={{ padding: "8px 10px", fontWeight: 500 }}>Customer Profile</th>
                  <th style={{ padding: "8px 10px", fontWeight: 500 }}>Status</th>
                  <th style={{ padding: "8px 10px", fontWeight: 500 }}>Visits</th>
                  <th style={{ padding: "8px 10px", fontWeight: 500 }}>Average Stay</th>
                  <th style={{ padding: "8px 10px", fontWeight: 500 }}>Last Seen</th>
                  <th style={{ padding: "8px 10px", fontWeight: 500, textAlign: "right" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredVisitors.map((v: VisitorProfile) => {
                  const sid = v.subject_id || "visitor";
                  const avatarUrl = v.avatar_path || `${engine}/api/workplace/avatar/${sid}`;
                  const isOnline = Boolean(v.is_active_now);
                  return (
                    <tr key={sid} style={{ borderBottom: "1px solid rgba(255, 255, 255, 0.05)" }}>
                      <td style={{ padding: "10px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                          <img
                            src={avatarUrl}
                            alt="avatar"
                            style={{ width: "36px", height: "36px", borderRadius: "8px", objectFit: "cover", background: "#1a1a1a", border: "1px solid var(--line)" }}
                            onError={(e) => { (e.target as HTMLElement).style.display = "none"; }}
                          />
                          <div>
                            <div style={{ fontWeight: 600, color: "var(--color-crisp-white)" }}>
                              {v.alias || sid}
                              {v.alias && <span style={{ fontSize: "10px", color: "var(--muted)", marginLeft: "4px", fontFamily: "var(--font-mono)" }}>({sid})</span>}
                            </div>
                            <div style={{ fontSize: "10px", color: "var(--muted)" }}>
                              {v.first_seen_at ? `Enrolled ${formatDate(v.first_seen_at)}` : "Active"}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: "10px" }}>
                        {isOnline ? (
                          <span style={{ padding: "2px 8px", borderRadius: "4px", fontSize: "11px", background: "rgba(0, 255, 102, 0.15)", color: "var(--color-surveillance-green)", border: "1px solid rgba(0, 255, 102, 0.3)" }}>
                            In Venue
                          </span>
                        ) : (
                          <span style={{ padding: "2px 8px", borderRadius: "4px", fontSize: "11px", background: "var(--bg-2)", color: "var(--muted)" }}>
                            Departed
                          </span>
                        )}
                      </td>
                      <td style={{ padding: "10px" }}>
                        <div style={{ fontWeight: 600, fontFamily: "var(--font-mono)" }}>{v.total_visits} visits</div>
                        <div style={{ fontSize: "11px", color: "var(--muted)" }}>{v.today_visits} today · {v.week_visits} wk</div>
                      </td>
                      <td style={{ padding: "10px", fontFamily: "var(--font-mono)" }}>
                        <div>Avg: {formatDwell(v.avg_dwell_seconds)}</div>
                        <div style={{ fontSize: "10px", color: "var(--muted)" }}>Last: {formatDwell(v.last_dwell_seconds)}</div>
                      </td>
                      <td style={{ padding: "10px", fontSize: "11px", color: "var(--muted)" }}>
                        {formatDate(v.last_seen_at)}
                      </td>
                      <td style={{ padding: "10px", textAlign: "right" }}>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ fontSize: "11px", padding: "4px 8px" }}
                          onClick={() => handleEditAlias(sid, v.alias)}
                        >
                          {v.alias ? "Edit Name" : "+ Set Name"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  );
}

