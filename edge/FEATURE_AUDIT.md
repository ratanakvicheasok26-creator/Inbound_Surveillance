# Champei Feature Audit Diagnostic

Mode: **READ-ONLY** inspection (no vision/tracker/workplace edits).

## Live constants

- `ARRIVAL_COOLDOWN_SECONDS` = `3600`
- `EARLY_DEPARTURE_SECONDS` = `1200`
- `STAFF_EVENTS` = `['early_departure', 'guest_arrival', 'vip_arrival', 'wait_bottleneck']`
- `OWNER_EVENTS` = `['daily_scorecard', 'early_departure', 'silent_churn']`

## Schema notes

- `customer_visits`: `subject_id`, `started_at`, `ended_at`, `duration_seconds`, `status`, `zone_id`
  - `status` via `sessions.ensure_session_columns`; `db.connect` migrates `duration_seconds`
- `events`: `ts`, `event_type`, `branch_id` (+ `camera_role` migrate)
- `visitor_meta`: `visitor_id`, `alias`, `vip_tier`, `notes` (registry-owned, not in `db.connect`)
- Adjacent: `staff_memory`, `anonymous_subjects`

## SECTION 1: Active Feature Matrix

| Feature | Module.Function | DB Tables/Cols | Event + Role | Trigger | Status |
| --- | --- | --- | --- | --- | --- |
| VIP Handshake & Cooldown | `visitor_registry.notify_guest_arrival` | `visitor_meta(alias, vip_tier)` | vip_arrival / Staff | Named alias + cooldown (3600s) | WARNING — Helper ready, pending live launcher wiring |
| Guest Arrival (Unnamed) | `visitor_registry.notify_guest_arrival` | `visitor_meta` | guest_arrival / Staff | Visit count check + cooldown (3600s) | WARNING — Helper ready, pending live launcher wiring |
| Wait SLA Bottleneck | `customer_visits.py (adjacent) + scorecard` | `events(event_type='wait_bottleneck')` | wait_bottleneck / Staff | 180s dwell in waiting/reception zone | WARNING — Live path uses send_message, bypasses send_alert staff route |
| Session Duration | `sessions.record_session_completion / db.end_customer_visit` | `customer_visits(duration_seconds, ended_at, status)` | None (Internal DB metric) | Duration calculated on session close | WARNING — Live end path skips status & early-departure helper |
| Early Departure Anomaly | `sessions.record_session_completion` | `customer_visits` | early_departure / Dual (Staff + Owner) | < 1200s (<20m) | WARNING — Active in helper, pending live exit trigger |
| Lobby Bounce / Walk-Away | `sessions.record_walk_away` | `events(event_type='walk_away')` | Internal DB only (Scorecard) | Explicit helper invocation | UNCOVERED — No vision bounce detector wired |
| Daily Ops Scorecard | `scorecard.build_daily_scorecard / send_daily_scorecard` | `customer_visits + events` | daily_scorecard / Owner | Scheduled daily at 21:00 or manual CLI | OK — Background scheduler in `run_champei` (21:00 local) + `/scorecard` for owner |
| Silent Churn Watchlist | `churn.compute_churn_risks / send_weekly_churn` | `customer_visits + visitor_meta` | silent_churn / Owner | Days overdue > max(cadence * 2.0, 21d) | WARNING — Ready via CLI / `/churn` (owner); no weekly auto-scheduler yet |

## SECTION 2: Camera & Spatial Coverage (Entrance CCTV View)

### Zone A (Glass Door & Welcome Mat)

- **Coverage:** Entry/Exit tripwires, Lobby Bounces, Group Clustering.
- **Diagnostic:** Walk-away detection is UNCOVERED. Visit start relies on generalized workplace zones rather than shoe-swap gating.

### Zone B (Wooden Changing Bench)

- **Coverage:** Seating dwell time, Wait SLA (>180s), Shoe-untie posture.
- **Diagnostic:** Wait SLA is OK if zone kinds include waiting/reception. Shoe-swap posture detection is UNCOVERED in hospitality helpers.

### Zone C (Shoe Cubby & Hallway Threshold)

- **Coverage:** Footwear storage motion, Inward hallway crossing.
- **Diagnostic:** `shoe_lounge` label is used in notification text, but zone kind does not exist in standard massage `VISIT_ZONE_KINDS`. Status: WARNING (Naming mismatch).

## SECTION 3: Edge Case & Vulnerability Scan

| Risk | Status | Current Mitigation | Remaining Gap |
| --- | --- | --- | --- |
| Staff vs. Customer Confusion | WARNING | staff_memory and is_staff flags in workplace monitor. | Therapists attending guests at the bench or rack outside the reception desk ROI may enroll as customers if not filtered by uniform color or staff ID. |
| Temporary Exits (Smoke / Phone Calls) | UNCOVERED | Live end_customer_visit uses ~8s track loss. | No 5–10 minute re-entry grace period in record_session_completion. A short exit can split 1 treatment into 2 visits or fire a false early departure alert. |
| Multi-Guest Occlusion at Bench | WARNING | DeepSORT tracking bounding boxes. | Two guests sitting closely or swapping shoes at the same time can cause bounding-box collisions or Re-ID vector swaps. |
| Slipper Re-Identification | WARNING | OSNet full-body feature extraction. | Guests enter in dark shoes and return in white spa slippers. The tracker must prioritize torso/upper-body embeddings to avoid identity fragmentation. |
| Entourage / Non-Paying Seating | WARNING | None. | Bench dwell time fires the Wait SLA bottleneck alert even if the person is a driver or bodyguard who never intended to take a service. |

## SECTION 4: Gap & Recommendation Engine

1. Arrival Cooldown (3600s): Evaluate lowering to 900s–1800s (15–30m) to accommodate same-day repeat visits or quick add-on services.
2. Early Departure Threshold (1200s): Champei's minimum menu service is 60 minutes ($17–$27). Consider raising the anomaly threshold to 1800s–2400s (30–40m) to catch shortened 1-hour sessions.
3. Re-Entry Grace Window: Add a 5–10 minute buffer before finalizing record_session_completion to handle mid-service cigarette or phone breaks.
4. Live Pipeline Wiring: Connect notify_guest_arrival, record_session_completion, and record_walk_away directly to CustomerVisitMonitor state transitions. Route live SLA alerts through TelegramOut.send_alert("wait_bottleneck", ...).
5. Staff Greeting Latency: Track elapsed seconds from glass door opening until a therapist arrives in the lounge.
6. Entourage Filter: Require verified shoe-swap posture before enrolling a visitor into the wait SLA counter.
7. Zone Alignment: Add shoe_lounge to workplace configuration files to match notification metadata.

## SECTION 5: Summary Tally

**OK: 2 | WARNING: 11 | UNCOVERED: 4**

## SECTION 6: Security Architecture Note: Telegram RBAC Scope

- **Owner Role:** Strictly bound to the Owner's private 1-on-1 chat (`TELEGRAM_OWNER_CHAT_ID`). All management reports (`/scorecard`, `/churn`) fail-closed outside this chat.
- **Staff Role:** Bound to the shared Staff Group Chat (`TELEGRAM_STAFF_CHAT_ID`). Access control is managed natively via Telegram group membership. Any member present in the staff group can execute operational updates (`/name`).
- **Risk Assessment:** Accepted. Eliminates manual user-ID provisioning on the edge laptop while keeping sensitive analytics isolated from group members.

See also [`DEPLOY_CHAMPEI.md`](DEPLOY_CHAMPEI.md) for laptop install, `.env` lockdown, Task Scheduler, and Tailscale.
