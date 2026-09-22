import { useAccount } from "../auth";

const STATUS_LABEL: Record<string, string> = {
  open: "Open",
  in_progress: "In progress",
  resolved: "Resolved",
};

export function ComplaintsView() {
  const { snapshot } = useAccount();
  const rows = snapshot?.complaints || [];

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Complaints</h2>
          <p>
            Import pending — waiting on the capture team. This console only stores the record
            once it arrives (`channel`, `body`, `external_ref`, `payload`).
          </p>
        </div>
      </header>
      <div className="list">
        {rows.length ? (
          rows.map((row) => (
            <article className="alert" key={row.id}>
              <div>
                <h3>
                  {STATUS_LABEL[row.status] || row.status} · {row.channel}
                </h3>
                <p>{row.body || "No body yet."}</p>
                <p className="mono">
                  {row.external_ref ? `ref ${row.external_ref} · ` : ""}
                  {row.created_at}
                </p>
              </div>
            </article>
          ))
        ) : (
          <p className="mono">No imported complaints yet.</p>
        )}
      </div>
    </section>
  );
}
