# Massage Workplace Operating Rules

These laws apply when `workplace_type = massage`. Garage wrench-time laws in `garage_system_laws.md` do not apply here.

---

## 1. Subject Law

- The monitor tracks **customers**, not employees.
- Unassigned / unmatched persons are labeled **`Visitor`** (never `Employee`, never a guessed name).
- Do not enroll customer faces. Identity is anonymous appearance re-ID only.

---

## 2. Anonymous Re-ID Law

- Match appearance embeddings to a local gallery. On match, reuse the same opaque `visitor_<id>`.
- Store no customer name, photo, or Face ID enrollment.
- `local_track_key` in persistence is an opaque hash, not a display name.

---

## 3. Visit Session Law

- Entering a zone (entrance / waiting / treatment_room) starts a visit session after occupancy confirm hysteresis.
- Leaving a zone plus a grace window ends the session. Do not reset unique-visitor identity when they walk between rooms in the same day.
- Unique counts:
  - **Today**: distinct subjects with at least one visit starting today.
  - **Week**: distinct subjects with at least one visit in the rolling 7 days.
- Visit events (session rows) and unique subjects are different metrics. Same person twice in one day = 2 visits, 1 unique.

---

## 4. Zone Kind Law

Allowed zone kinds: `entrance`, `waiting`, `treatment_room`.
Do not coerce these into `vehicle_bay` / `tool_area`.

---

## 5. Complaints Law

- The edge/dashboard only persist complaint **structure** (`status`, `channel`, `body`, `external_ref`, `payload`).
- Capture UI is owned by another team. Pull completed records via `external_ref` / ingest contract.
