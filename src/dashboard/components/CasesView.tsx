import { useState, type FormEvent } from "react";
import { caseLabel, formatRelative, statusLabel } from "../format";
import { portraitDataUri } from "../ids";
import { useOps } from "../store";
import type { CaseKind, CaseRecord, WatchlistCategory } from "../types";

function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function MissingForm() {
  const { submitMissing } = useOps();
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const name = String(data.get("name") ?? "").trim();
    if (!name) return;
    setBusy(true);
    try {
      const file = data.get("photo");
      const photo =
        file instanceof File && file.size > 0 ? await readFile(file) : portraitDataUri(name);
      submitMissing(
        {
          name,
          age: String(data.get("age") ?? ""),
          clothing: String(data.get("clothing") ?? ""),
          lastSeenZone: String(data.get("zone") ?? ""),
          reporterName: String(data.get("reporter") ?? ""),
          reporterPhone: String(data.get("phone") ?? ""),
          ephemeral: data.get("ephemeral") === "on",
        },
        photo,
        "web",
      );
      form.reset();
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="form" onSubmit={onSubmit}>
      <h3>Report missing person</h3>
      <p className="mono">Safety Wedge · info desk</p>
      <label className="field">
        <span>Photograph</span>
        <input name="photo" type="file" accept="image/*" />
      </label>
      <label className="field">
        <span>Name</span>
        <input name="name" required placeholder="Given name" />
      </label>
      <div className="row-2">
        <label className="field">
          <span>Age</span>
          <input name="age" placeholder="6" />
        </label>
        <label className="field">
          <span>Last seen zone</span>
          <input name="zone" placeholder="Food court" />
        </label>
      </div>
      <label className="field">
        <span>Clothing / features</span>
        <textarea name="clothing" placeholder="Yellow dress, red backpack" />
      </label>
      <div className="row-2">
        <label className="field">
          <span>Reporter</span>
          <input name="reporter" placeholder="Parent name" />
        </label>
        <label className="field">
          <span>Phone</span>
          <input name="phone" placeholder="+855" />
        </label>
      </div>
      <label className="checks">
        <input name="ephemeral" type="checkbox" defaultChecked />
        Ephemeral search — expire embeddings when resolved
      </label>
      <button className="btn btn--primary" type="submit" disabled={busy}>
        {busy ? "Submitting" : "Start search"}
      </button>
    </form>
  );
}

function WatchlistForm() {
  const { submitWatchlist } = useOps();
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const alias = String(data.get("alias") ?? "").trim();
    if (!alias) return;
    setBusy(true);
    try {
      const file = data.get("photo");
      const photo =
        file instanceof File && file.size > 0 ? await readFile(file) : portraitDataUri(alias);
      submitWatchlist(
        {
          alias,
          category: String(data.get("category") ?? "shoplifter") as WatchlistCategory,
          notes: String(data.get("notes") ?? ""),
          shareAcrossVenues: data.get("share") === "on",
        },
        photo,
      );
      form.reset();
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="form" onSubmit={onSubmit}>
      <h3>Add watchlist entry</h3>
      <p className="mono">Loss Prevention · participating locations</p>
      <label className="field">
        <span>Photograph</span>
        <input name="photo" type="file" accept="image/*" />
      </label>
      <label className="field">
        <span>Name / alias</span>
        <input name="alias" required placeholder="Alias" />
      </label>
      <label className="field">
        <span>Category</span>
        <select name="category" defaultValue="shoplifter">
          <option value="shoplifter">Shoplifter</option>
          <option value="banned">Banned</option>
          <option value="wanted">Wanted</option>
        </select>
      </label>
      <label className="field">
        <span>Notes</span>
        <textarea name="notes" placeholder="Last incident, store, description" />
      </label>
      <label className="checks">
        <input name="share" type="checkbox" defaultChecked />
        Share with participating locations
      </label>
      <button className="btn btn--primary" type="submit" disabled={busy}>
        {busy ? "Submitting" : "Add to watchlist"}
      </button>
    </form>
  );
}

function CaseCard({ record }: { record: CaseRecord }) {
  const { resolveCase } = useOps();
  const open = record.status === "searching" || record.status === "match_pending" || record.status === "in_progress";
  const detail =
    record.kind === "missing"
      ? `${record.missing?.age ? `Age ${record.missing.age} · ` : ""}${record.missing?.clothing ?? ""} · ${record.missing?.lastSeenZone ?? ""}`
      : `${record.watchlist?.category} · ${record.watchlist?.notes ?? ""}`;

  return (
    <article className="case">
      <img src={record.photo} alt="" />
      <div>
        <h3>{caseLabel(record)}</h3>
        <p className="case__id">
          {record.ticketId} · {statusLabel(record.status)} · {formatRelative(record.createdAt)}
        </p>
        <p>{detail}</p>
        {open && (
          <div className="actions">
            <button className="btn btn--ghost btn--sm" type="button" onClick={() => resolveCase(record.id)}>
              Resolve / expire
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

export function CasesView() {
  const { state } = useOps();
  const [kind, setKind] = useState<CaseKind>("missing");
  const rows = state.cases.filter((record) => record.kind === kind);

  return (
    <section className="panel">
      <header className="panel__head">
        <div>
          <h2>Customer DB</h2>
          <p>
            Web forms and Telegram tickets write to the same case store. Mentors can replace
            localStorage with the backend later.
          </p>
        </div>
      </header>
      <div className="subtabs">
        <button type="button" className={kind === "missing" ? "is-on" : ""} onClick={() => setKind("missing")}>
          Missing
        </button>
        <button type="button" className={kind === "watchlist" ? "is-on" : ""} onClick={() => setKind("watchlist")}>
          Watchlist
        </button>
      </div>
      <div className="split">
        {kind === "missing" ? <MissingForm /> : <WatchlistForm />}
        <div className="list">
          {rows.map((record) => (
            <CaseCard key={record.id} record={record} />
          ))}
        </div>
      </div>
    </section>
  );
}
