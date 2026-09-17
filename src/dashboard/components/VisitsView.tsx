import { useEffect, useState } from "react";
import { engineBaseUrl } from "../../engine-url";
import { useAccount } from "../auth";
import type { VisitCounts } from "../types";

function emptyCounts(): VisitCounts {
  return { today_unique: 0, today_visits: 0, week_unique: 0, week_visits: 0, open_sessions: [] };
}

export function VisitsView() {
  const { snapshot } = useAccount();
  const [counts, setCounts] = useState<VisitCounts>(emptyCounts());
  const engine = engineBaseUrl();

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const res = await fetch(`${engine}/api/workplace/visits`, { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as VisitCounts;
        if (!cancelled) setCounts({ ...emptyCounts(), ...data });
      } catch {
        /* engine offline — fall back to supabase rows */
        if (cancelled) return;
        const visits = snapshot?.visits || [];
        const now = Date.now();
        const dayStart = new Date();
        dayStart.setHours(0, 0, 0, 0);
        const weekStart = now - 7 * 24 * 60 * 60 * 1000;
        const today = visits.filter((row) => Date.parse(row.started_at) >= dayStart.getTime());
        const week = visits.filter((row) => Date.parse(row.started_at) >= weekStart);
        setCounts({
          today_visits: today.length,
          today_unique: new Set(today.map((row) => row.subject_id)).size,
          week_visits: week.length,
          week_unique: new Set(week.map((row) => row.subject_id)).size,
          open_sessions: visits
            .filter((row) => !row.ended_at)
            .map((row) => ({
              subject_id: row.subject_id || "visitor",
              zone_id: row.zone_id,
              started_at: row.started_at,
            })),
        });
      }
    }
    void tick();
    const id = window.setInterval(() => void tick(), 4000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [engine, snapshot?.visits]);

  const recent = snapshot?.visits || [];

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Customer visits</h2>
          <p>
            Anonymous appearance re-ID. Same visitor twice in one day is two visits and one unique.
            No names or face enrollment.
          </p>
        </div>
      </header>
      <div className="visit-stats">
        <article>
          <span>Today unique</span>
          <strong>{counts.today_unique}</strong>
        </article>
        <article>
          <span>Today visits</span>
          <strong>{counts.today_visits}</strong>
        </article>
        <article>
          <span>Week unique</span>
          <strong>{counts.week_unique}</strong>
        </article>
        <article>
          <span>Week visits</span>
          <strong>{counts.week_visits}</strong>
        </article>
      </div>
      <div className="list">
        {(counts.open_sessions || []).length ? (
          counts.open_sessions?.map((session) => (
            <article className="alert" key={`${session.subject_id}-${session.zone_id}-${session.started_at}`}>
              <div>
                <h3>{session.subject_id}</h3>
                <p className="mono">
                  Open in {session.zone_name || session.zone_id}
                  {session.started_at ? ` · ${session.started_at}` : ""}
                </p>
              </div>
            </article>
          ))
        ) : (
          <p className="mono">No open sessions.</p>
        )}
        {recent.slice(0, 12).map((row) => (
          <article className="alert" key={row.id}>
            <div>
              <h3>{row.subject_id || "visitor"}</h3>
              <p className="mono">
                {row.zone_id} · {row.started_at}
                {row.ended_at ? ` → ${row.ended_at}` : " (open)"}
              </p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
