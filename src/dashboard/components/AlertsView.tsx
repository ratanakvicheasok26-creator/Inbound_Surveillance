import { cameraById, caseLabel, confidenceLabel, formatClock, formatRelative } from "../format";
import { useOps } from "../store";

export function AlertsView() {
  const { state, operator, dismissAlert } = useOps();
  const alerts = [...state.alerts].sort((a, b) => b.ts - a.ts);

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Packaged alerts</h2>
          <p>
            Still or GIF plus metadata (timestamp, camera ID, reason). Dispatch is a mocked
            HTTPS POST to the Telegram Bot API. Matches always wait for a human.
          </p>
        </div>
      </header>
      <div className="list">
        {alerts.map((alert) => {
          const camera = cameraById(state.cameras, alert.cameraId);
          const record = state.cases.find((item) => item.id === alert.caseId);
          const muted = alert.dismissed;
          return (
            <article className="alert" key={alert.id} style={muted ? { opacity: 0.5 } : undefined}>
              {record?.photo ? <img src={record.photo} alt="" /> : <div className="thumb" />}
              <div>
                <h3>{alert.id} · {camera?.name}</h3>
                <p className="mono">
                  {formatClock(alert.ts)} · {camera?.zone} · {alert.mediaKind} ·{" "}
                  {confidenceLabel(alert.confidence)} · Telegram {alert.telegramState}
                </p>
                <p>{alert.reason}</p>
                {record && <p>{caseLabel(record)} · {record.ticketId}</p>}
                {alert.operatorAction && (
                  <p className="mono">Operator: {alert.operatorAction}</p>
                )}
                <p className="mono">{formatRelative(alert.ts)}</p>
                {!muted && (
                  <div className="actions">
                    <button className="btn btn--primary btn--sm" type="button" onClick={() => operator(alert.id, "verify")}>
                      Verify
                    </button>
                    <button className="btn btn--ghost btn--sm" type="button" onClick={() => operator(alert.id, "intercom")}>
                      Intercom
                    </button>
                    <button className="btn btn--warn btn--sm" type="button" onClick={() => operator(alert.id, "siren")}>
                      Siren
                    </button>
                    <button className="btn btn--danger btn--sm" type="button" onClick={() => dismissAlert(alert.id)}>
                      Dismiss
                    </button>
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
