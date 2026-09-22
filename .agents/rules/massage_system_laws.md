# Massage Workplace Operating Rules

> **SUPREME LAW REFERENCE:** All work on this system is governed by [SYSTEM_ARCHITECTURE_LAWS.md](file:///home/george/Documents/Inbound-Surveillance/.agents/rules/SYSTEM_ARCHITECTURE_LAWS.md).  
> **LAW 0 (STOP AND REPORT):** If an edit risks frame drops, tracking loss, or regressions, STOP and warn the user before writing code.

These laws apply when `workplace_type = massage`. Garage wrench-time laws in `garage_system_laws.md` do not apply here.

---

## 1. Subject Law

- The monitor tracks **customers**, not employees.
- Unassigned / unmatched persons are labeled **`Visitor`** (never `Employee`, never a guessed name).
- Do not enroll customer faces. Identity is anonymous appearance re-ID only.
- Staff may be remembered from **AI-verified counter presence** (dwell in the `reception` ROI, then VLM staff-behind-counter vs customer-at-desk). Stored identity is an opaque `staff_<id>` plus appearance embedding only — no staff name or photo. Later frames rematch by cosine and live-label **Staff**. Still no customer Face ID.

---

## 2. Anonymous Re-ID Law

- Match appearance embeddings to a local gallery. On match, reuse the same opaque `visitor_<id>`.
- Store no customer name, photo, or Face ID enrollment.
- `local_track_key` in persistence is an opaque hash, not a display name.

---

## 3. Visit Session Law

- Entering a visit zone (`entrance` / `waiting` / `treatment_room`) starts a visit session after occupancy confirm hysteresis.
- The `reception` (counter) ROI is **not** a visit zone. It is used only to gate staff verification.
- Remembered staff (`is_staff` / `staff_<id>`) are excluded from customer visit counts.
- Leaving a zone plus a grace window ends the session. Do not reset unique-visitor identity when they walk between rooms in the same day.
- Unique counts:
  - **Today**: distinct subjects with at least one visit starting today.
  - **Week**: distinct subjects with at least one visit in the rolling 7 days.
- Visit events (session rows) and unique subjects are different metrics. Same person twice in one day = 2 visits, 1 unique.

---

## 4. Zone Kind Law

Allowed zone kinds: `entrance`, `waiting`, `treatment_room`, `reception`.
Do not coerce these into `vehicle_bay` / `tool_area`.
`reception` is a tight counter ROI (operator-drawn behind the desk). Occupancy may show a person there; visit sessions do not start from it.

---

## 5. Complaints Law

- The edge/dashboard only persist complaint **structure** (`status`, `channel`, `body`, `external_ref`, `payload`).
- Capture UI is owned by another team. Pull completed records via `external_ref` / ingest contract.
