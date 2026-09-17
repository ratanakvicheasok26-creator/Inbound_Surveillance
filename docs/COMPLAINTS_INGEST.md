# Complaints ingest contract

The capture product is owned by another team. Inbound Surveillance only **stores** completed complaints and lists them in the operator console.

Do not send customer photos or names. Anonymous `subject_id` / `visit_id` are optional links to edge visit rows.

## Insert (preferred)

Authenticated Supabase insert into `public.complaints` (owner-only RLS: `user_id = auth.uid()`).

```json
{
  "user_id": "<operator uuid>",
  "channel": "in_app | telegram | import",
  "status": "open | in_progress | resolved",
  "body": "Customer-facing complaint text",
  "external_ref": "capture-system-id",
  "subject_id": null,
  "visit_id": null,
  "payload": {}
}
```

`external_ref` is unique per operator when set, so retries can upsert on `(user_id, external_ref)`.

## HTTP shape (future Edge Function)

`POST /functions/v1/complaints-ingest`

```json
{
  "channel": "import",
  "body": "…",
  "external_ref": "abc-123",
  "payload": {}
}
```

Authorization: operator JWT. The function copies `auth.uid()` into `user_id`.
