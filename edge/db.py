from __future__ import annotations

import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any


def connect(db_path: Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            abs_path TEXT
        )
        """
    )
    try:
        cols_events = {r["name"] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
        if "branch_id" not in cols_events:
            conn.execute("ALTER TABLE events ADD COLUMN branch_id TEXT")
        if "camera_role" not in cols_events:
            conn.execute("ALTER TABLE events ADD COLUMN camera_role TEXT")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS minutes (
            minute TEXT PRIMARY KEY,
            max_persons INTEGER NOT NULL,
            occupied_frames INTEGER NOT NULL,
            total_frames INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS technician_shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT NOT NULL,
            staff_name TEXT NOT NULL,
            clock_in_time TEXT,
            clock_out_time TEXT,
            total_shift_minutes REAL NOT NULL DEFAULT 0,
            wrench_minutes REAL NOT NULL DEFAULT 0,
            idle_minutes REAL NOT NULL DEFAULT 0,
            wifi_active_minutes REAL NOT NULL DEFAULT 0,
            performance_score REAL NOT NULL DEFAULT 0,
            UNIQUE(day, staff_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bay_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bay_id TEXT NOT NULL,
            technician_name TEXT,
            start_time TEXT NOT NULL,
            end_time TEXT,
            active_duration REAL NOT NULL DEFAULT 0,
            idle_duration REAL NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            bay_id TEXT NOT NULL,
            vehicle_type TEXT NOT NULL DEFAULT 'vehicle',
            vehicle_label TEXT,
            primary_technician TEXT,
            status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
            total_active_seconds REAL NOT NULL DEFAULT 0,
            total_break_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_vehicle_job_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            day TEXT NOT NULL,
            technician_name TEXT,
            active_seconds REAL NOT NULL DEFAULT 0,
            break_seconds REAL NOT NULL DEFAULT 0,
            UNIQUE(job_id, day, technician_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_job_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL,
            vehicle_type TEXT NOT NULL,
            primary_technician TEXT,
            technicians_json TEXT,
            total_wrench_seconds REAL NOT NULL DEFAULT 0,
            total_break_seconds REAL NOT NULL DEFAULT 0,
            efficiency_pct REAL NOT NULL DEFAULT 100.0,
            performance_grade TEXT NOT NULL,
            performance_score INTEGER NOT NULL DEFAULT 100,
            summary_notes TEXT,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            bay_id TEXT NOT NULL,
            technician_name TEXT,
            action TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            explanation TEXT,
            crop_path TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anonymous_subjects (
            id TEXT PRIMARY KEY,
            local_track_key TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_visits (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL,
            zone_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            source TEXT NOT NULL DEFAULT 'edge'
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS customer_visits_started_idx
        ON customer_visits (started_at)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS customer_visits_subject_idx
        ON customer_visits (subject_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS anon_subjects_last_seen_idx
        ON anonymous_subjects (last_seen_at)
        """
    )
    try:
        cols_anon = {r["name"] for r in conn.execute("PRAGMA table_info(anonymous_subjects)").fetchall()}
        if "alias" not in cols_anon:
            conn.execute("ALTER TABLE anonymous_subjects ADD COLUMN alias TEXT")
        if "avatar_path" not in cols_anon:
            conn.execute("ALTER TABLE anonymous_subjects ADD COLUMN avatar_path TEXT")
        cols_vis = {r["name"] for r in conn.execute("PRAGMA table_info(customer_visits)").fetchall()}
        if "duration_seconds" not in cols_vis:
            conn.execute("ALTER TABLE customer_visits ADD COLUMN duration_seconds REAL")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS staff_memory (
            id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_face_embeddings (
            subject_id TEXT NOT NULL,
            slot INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            PRIMARY KEY (subject_id, slot)
        )
        """
    )
    try:
        cols_anon = {r["name"] for r in conn.execute("PRAGMA table_info(anonymous_subjects)").fetchall()}
        if "visit_count" not in cols_anon:
            conn.execute("ALTER TABLE anonymous_subjects ADD COLUMN visit_count INTEGER NOT NULL DEFAULT 1")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            customer_id TEXT,
            camera_id TEXT,
            audio_source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            screenshot_path TEXT,
            khmer_transcript TEXT NOT NULL,
            english_transcript TEXT NOT NULL,
            is_complaint INTEGER NOT NULL DEFAULT 1,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            summary TEXT NOT NULL,
            telegram_sent INTEGER NOT NULL DEFAULT 0,
            telegram_error TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn



def insert_event(
    conn: sqlite3.Connection,
    event_type: str,
    ts: datetime,
    abs_path: str | None = None,
    *,
    branch_id: str | None = None,
    camera_role: str | None = None,
) -> None:
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
    if "branch_id" in cols and "camera_role" in cols:
        conn.execute(
            "INSERT INTO events (ts, event_type, abs_path, branch_id, camera_role) VALUES (?, ?, ?, ?, ?)",
            (
                ts.isoformat(timespec="seconds"),
                event_type,
                abs_path,
                (branch_id or None),
                (camera_role or None),
            ),
        )
    else:
        conn.execute(
            "INSERT INTO events (ts, event_type, abs_path) VALUES (?, ?, ?)",
            (ts.isoformat(timespec="seconds"), event_type, abs_path),
        )
    conn.commit()


def has_opened_today(conn: sqlite3.Connection, day: date) -> bool:
    prefix = day.isoformat()
    row = conn.execute(
        "SELECT 1 FROM events WHERE event_type = 'opened' AND ts LIKE ? LIMIT 1",
        (f"{prefix}%",),
    ).fetchone()
    return row is not None


def upsert_minute(
    conn: sqlite3.Connection,
    minute: str,
    person_count: int,
    occupied: bool,
) -> None:
    row = conn.execute("SELECT * FROM minutes WHERE minute = ?", (minute,)).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO minutes (minute, max_persons, occupied_frames, total_frames)
            VALUES (?, ?, ?, 1)
            """,
            (minute, person_count, 1 if occupied else 0),
        )
    else:
        conn.execute(
            """
            UPDATE minutes
            SET max_persons = MAX(max_persons, ?),
                occupied_frames = occupied_frames + ?,
                total_frames = total_frames + 1
            WHERE minute = ?
            """,
            (person_count, 1 if occupied else 0, minute),
        )
    conn.commit()


def day_events(conn: sqlite3.Connection, day: date) -> list[sqlite3.Row]:
    prefix = day.isoformat()
    return list(
        conn.execute(
            "SELECT * FROM events WHERE ts LIKE ? ORDER BY ts ASC",
            (f"{prefix}%",),
        )
    )


def day_minutes(conn: sqlite3.Connection, day: date) -> list[sqlite3.Row]:
    prefix = day.isoformat()
    return list(
        conn.execute(
            "SELECT * FROM minutes WHERE minute LIKE ? ORDER BY minute ASC",
            (f"{prefix}%",),
        )
    )


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _shift_minutes(clock_in: str | None, clock_out: str | None, now: datetime) -> float:
    start = _parse_ts(clock_in)
    if start is None:
        return 0.0
    end = _parse_ts(clock_out) or now
    return max(0.0, (end - start).total_seconds() / 60.0)


def _performance_score(wrench_minutes: float, shift_minutes: float) -> float:
    if shift_minutes <= 0:
        return 0.0
    return round(100.0 * wrench_minutes / shift_minutes, 1)


def _refresh_shift_row(conn: sqlite3.Connection, day: date, name: str, now: datetime) -> None:
    row = conn.execute(
        "SELECT * FROM technician_shifts WHERE day = ? AND staff_name = ?",
        (day.isoformat(), name),
    ).fetchone()
    if row is None:
        return
    total = _shift_minutes(row["clock_in_time"], row["clock_out_time"], now)
    score = _performance_score(float(row["wrench_minutes"]), total)
    conn.execute(
        """
        UPDATE technician_shifts
        SET total_shift_minutes = ?, performance_score = ?
        WHERE day = ? AND staff_name = ?
        """,
        (round(total, 3), score, day.isoformat(), name),
    )


def record_face_clock_in(
    conn: sqlite3.Connection,
    name: str,
    timestamp: datetime,
) -> dict[str, Any]:
    """First Face ID sighting of the day. Returning from a break reopens the shift."""
    staff = str(name or "").strip()
    if not staff:
        return {}
    day = timestamp.date()
    row = conn.execute(
        "SELECT * FROM technician_shifts WHERE day = ? AND staff_name = ?",
        (day.isoformat(), staff),
    ).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO technician_shifts (
                day, staff_name, clock_in_time, total_shift_minutes
            ) VALUES (?, ?, ?, 0)
            """,
            (day.isoformat(), staff, _iso(timestamp)),
        )
        insert_event(conn, "clock_in", timestamp, staff)
        conn.commit()
        return {"name": staff, "clock_in_time": _iso(timestamp), "created": True}
    if row["clock_out_time"]:
        conn.execute(
            """
            UPDATE technician_shifts
            SET clock_out_time = NULL
            WHERE day = ? AND staff_name = ?
            """,
            (day.isoformat(), staff),
        )
        insert_event(conn, "clock_in", timestamp, staff)
    _refresh_shift_row(conn, day, staff, timestamp)
    conn.commit()
    return {
        "name": staff,
        "clock_in_time": row["clock_in_time"],
        "created": False,
    }


def record_face_clock_out(
    conn: sqlite3.Connection,
    name: str,
    timestamp: datetime,
) -> dict[str, Any]:
    staff = str(name or "").strip()
    if not staff:
        return {}
    day = timestamp.date()
    row = conn.execute(
        "SELECT * FROM technician_shifts WHERE day = ? AND staff_name = ?",
        (day.isoformat(), staff),
    ).fetchone()
    if row is None or not row["clock_in_time"]:
        return {}
    if row["clock_out_time"]:
        return {"name": staff, "clock_out_time": row["clock_out_time"]}
    conn.execute(
        """
        UPDATE technician_shifts
        SET clock_out_time = ?
        WHERE day = ? AND staff_name = ?
        """,
        (_iso(timestamp), day.isoformat(), staff),
    )
    _refresh_shift_row(conn, day, staff, timestamp)
    insert_event(conn, "clock_out", timestamp, staff)
    conn.commit()
    return {"name": staff, "clock_out_time": _iso(timestamp)}


def add_wifi_minutes(
    conn: sqlite3.Connection,
    name: str,
    dt_seconds: float,
    timestamp: datetime,
) -> None:
    staff = str(name or "").strip()
    if not staff or dt_seconds <= 0:
        return
    conn.execute(
        """
        UPDATE technician_shifts
        SET wifi_active_minutes = wifi_active_minutes + ?
        WHERE day = ? AND staff_name = ?
        """,
        (dt_seconds / 60.0, timestamp.date().isoformat(), staff),
    )
    _refresh_shift_row(conn, timestamp.date(), staff, timestamp)
    conn.commit()


def update_technician_activity(
    conn: sqlite3.Connection,
    name: str | None,
    bay_id: str,
    is_working: bool,
    dt_seconds: float,
    timestamp: datetime | None = None,
) -> None:
    """Increment wrench/idle seconds for a mechanic and the open bay session."""
    stamp = timestamp or datetime.now()
    dt = max(0.0, float(dt_seconds))
    if dt <= 0 or not bay_id:
        return
    staff = str(name or "").strip() or None
    if staff:
        record_face_clock_in(conn, staff, stamp)
        field = "wrench_minutes" if is_working else "idle_minutes"
        conn.execute(
            f"""
            UPDATE technician_shifts
            SET {field} = {field} + ?
            WHERE day = ? AND staff_name = ?
            """,
            (dt / 60.0, stamp.date().isoformat(), staff),
        )
        _refresh_shift_row(conn, stamp.date(), staff, stamp)

    open_row = conn.execute(
        """
        SELECT * FROM bay_sessions
        WHERE bay_id = ? AND end_time IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (bay_id,),
    ).fetchone()
    needs_new = (
        open_row is None
        or (open_row["technician_name"] or None) != staff
    )
    if needs_new:
        if open_row is not None:
            conn.execute(
                "UPDATE bay_sessions SET end_time = ? WHERE id = ?",
                (_iso(stamp), open_row["id"]),
            )
        conn.execute(
            """
            INSERT INTO bay_sessions (
                bay_id, technician_name, start_time, active_duration, idle_duration
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                bay_id,
                staff,
                _iso(stamp),
                dt if is_working else 0.0,
                0.0 if is_working else dt,
            ),
        )
    else:
        field = "active_duration" if is_working else "idle_duration"
        conn.execute(
            f"UPDATE bay_sessions SET {field} = {field} + ? WHERE id = ?",
            (dt, open_row["id"]),
        )
    conn.commit()


def close_empty_bays(
    conn: sqlite3.Connection,
    occupied_bay_ids: set[str],
    timestamp: datetime,
) -> None:
    rows = conn.execute("SELECT id, bay_id FROM bay_sessions WHERE end_time IS NULL").fetchall()
    for row in rows:
        if row["bay_id"] not in occupied_bay_ids:
            conn.execute(
                "UPDATE bay_sessions SET end_time = ? WHERE id = ?",
                (_iso(timestamp), row["id"]),
            )
    conn.commit()


def close_bay_sessions(
    conn: sqlite3.Connection,
    bay_ids: list[str] | set[str],
    timestamp: datetime | None = None,
) -> int:
    """End open sessions for removed bays. Past (already ended) rows stay intact."""
    ids = [str(bay_id).strip() for bay_id in bay_ids if str(bay_id).strip()]
    if not ids:
        return 0
    stamp = _iso(timestamp or datetime.now())
    closed = 0
    for bay_id in ids:
        cur = conn.execute(
            "UPDATE bay_sessions SET end_time = ? WHERE bay_id = ? AND end_time IS NULL",
            (stamp, bay_id),
        )
        closed += int(cur.rowcount or 0)
    conn.commit()
    return closed


def _parse_clock(value: str | None, default: str) -> time:
    text = str(value or default)
    try:
        hour, minute = text.split(":")
        return time(int(hour), int(minute))
    except (TypeError, ValueError):
        return time(8, 0)


def get_daily_garage_summary(
    conn: sqlite3.Connection,
    day: date,
    *,
    open_time: str = "08:00",
    close_time: str = "18:00",
    bay_ids: list[str] | None = None,
) -> dict[str, Any]:
    now = datetime.now()
    techs_by_name: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT * FROM technician_shifts WHERE day = ? ORDER BY clock_in_time ASC",
        (day.isoformat(),),
    ):
        raw_name = str(row["staff_name"] or "").strip()
        if not raw_name:
            continue
        key = raw_name.lower()
        shift_min = _shift_minutes(row["clock_in_time"], row["clock_out_time"], now)
        wrench = float(row["wrench_minutes"] or 0)
        idle = float(row["idle_minutes"] or 0)
        wifi = float(row["wifi_active_minutes"] or 0)
        is_clocked_in = row["clock_in_time"] is not None and not row["clock_out_time"]

        if key not in techs_by_name:
            techs_by_name[key] = {
                "staff_name": raw_name,
                "clock_in_time": row["clock_in_time"],
                "clock_out_time": row["clock_out_time"],
                "total_shift_minutes": shift_min,
                "wrench_minutes": wrench,
                "idle_minutes": idle,
                "wifi_active_minutes": wifi,
                "clocked_in": is_clocked_in,
            }
        else:
            agg = techs_by_name[key]
            agg["total_shift_minutes"] += shift_min
            agg["wrench_minutes"] += wrench
            agg["idle_minutes"] += idle
            agg["wifi_active_minutes"] += wifi
            if is_clocked_in:
                agg["clocked_in"] = True
                agg["clock_out_time"] = None
            elif not agg["clocked_in"]:
                if row["clock_out_time"] and (not agg["clock_out_time"] or row["clock_out_time"] > agg["clock_out_time"]):
                    agg["clock_out_time"] = row["clock_out_time"]
            if row["clock_in_time"] and (not agg["clock_in_time"] or row["clock_in_time"] < agg["clock_in_time"]):
                agg["clock_in_time"] = row["clock_in_time"]

    technicians: list[dict[str, Any]] = []
    for agg in techs_by_name.values():
        total = agg["total_shift_minutes"]
        wrench = agg["wrench_minutes"]
        score = _performance_score(wrench, total)
        technicians.append(
            {
                "staff_name": agg["staff_name"],
                "clock_in_time": agg["clock_in_time"],
                "clock_out_time": agg["clock_out_time"],
                "total_shift_minutes": round(total, 2),
                "wrench_minutes": round(wrench, 2),
                "idle_minutes": round(agg["idle_minutes"], 2),
                "wifi_active_minutes": round(agg["wifi_active_minutes"], 2),
                "performance_score": score,
                "clocked_in": agg["clocked_in"],
            }
        )

    prefix = day.isoformat()
    bay_rows = list(
        conn.execute(
            """
            SELECT bay_id,
                   SUM(active_duration) AS active_duration,
                   SUM(idle_duration) AS idle_duration,
                   COUNT(*) AS session_count
            FROM bay_sessions
            WHERE start_time LIKE ?
            GROUP BY bay_id
            """,
            (f"{prefix}%",),
        )
    )
    by_id = {row["bay_id"]: row for row in bay_rows}
    ids = list(bay_ids or []) or list(by_id.keys())
    open_t = _parse_clock(open_time, "08:00")
    close_t = _parse_clock(close_time, "18:00")
    window_start = datetime.combine(day, open_t)
    window_end = datetime.combine(day, close_t)
    if now.date() == day:
        elapsed = max(0.0, (min(now, window_end) - window_start).total_seconds())
    else:
        elapsed = max(0.0, (window_end - window_start).total_seconds())
    operating_hours = max(0.01, elapsed / 3600.0)

    bays: list[dict[str, Any]] = []
    used_seconds = 0.0
    for bay_id in ids:
        row = by_id.get(bay_id)
        active = float(row["active_duration"] if row else 0) or 0.0
        idle = float(row["idle_duration"] if row else 0) or 0.0
        occupied = active + idle
        used_seconds += occupied
        util = 0.0 if elapsed <= 0 else 100.0 * occupied / elapsed
        bays.append(
            {
                "bay_id": bay_id,
                "active_seconds": round(active, 2),
                "idle_seconds": round(idle, 2),
                "active_duration": round(active, 2),
                "idle_duration": round(idle, 2),
                "utilization_pct": round(util, 1),
                "session_count": int(row["session_count"]) if row else 0,
            }
        )

    n_bays = max(1, len(bays) if bays else 1)
    shop_util = 0.0 if elapsed <= 0 else 100.0 * used_seconds / (elapsed * n_bays)
    total_shift_hours = sum(t["total_shift_minutes"] for t in technicians) / 60.0

    job_rows = list(
        conn.execute(
            """
            SELECT j.*, l.active_seconds AS active_today, l.break_seconds AS break_today
            FROM vehicle_jobs j
            LEFT JOIN daily_vehicle_job_logs l
                   ON j.job_id = l.job_id AND l.day = ?
            WHERE j.created_at LIKE ? OR l.day = ? OR j.status != 'COMPLETED'
            ORDER BY j.updated_at DESC
            """,
            (day.isoformat(), f"{prefix}%", day.isoformat()),
        )
    )
    jobs: list[dict[str, Any]] = []
    for r in job_rows:
        jobs.append(
            {
                "job_id": r["job_id"],
                "bay_id": r["bay_id"],
                "vehicle_type": r["vehicle_type"],
                "vehicle_label": r["vehicle_label"] or r["job_id"],
                "primary_technician": r["primary_technician"],
                "status": r["status"],
                "total_active_seconds": round(float(r["total_active_seconds"] or 0), 2),
                "total_break_seconds": round(float(r["total_break_seconds"] or 0), 2),
                "active_today_seconds": round(float(r["active_today"] or 0), 2),
                "break_today_seconds": round(float(r["break_today"] or 0), 2),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "completed_at": r["completed_at"],
            }
        )

    return {
        "date": day.isoformat(),
        "technicians": technicians,
        "bays": bays,
        "jobs": jobs,
        "shop": {
            "operating_hours": round(operating_hours, 2),
            "total_shift_hours": round(total_shift_hours, 2),
            "utilization_pct": round(shop_util, 1),
            "open_time": open_time,
            "close_time": close_time,
            "active_jobs_count": sum(1 for j in jobs if j["status"] != "COMPLETED"),
        },
    }


def get_or_create_vehicle_job(
    conn: sqlite3.Connection | None,
    bay_id: str,
    vehicle_type: str = "vehicle",
    vehicle_label: str | None = None,
    primary_technician: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Return active job for bay, or create a new vehicle repair job."""
    if conn is None:
        return ""
    now = timestamp or datetime.now()
    row = conn.execute(
        "SELECT job_id FROM vehicle_jobs WHERE bay_id = ? AND status != 'COMPLETED' ORDER BY updated_at DESC LIMIT 1",
        (bay_id,),
    ).fetchone()
    if row:
        return row["job_id"]

    day_str = now.strftime("%Y%m%d")
    count_row = conn.execute(
        "SELECT COUNT(*) AS c FROM vehicle_jobs WHERE job_id LIKE ?",
        (f"JOB-{bay_id}-{day_str}-%",),
    ).fetchone()
    seq = (int(count_row["c"]) if count_row else 0) + 1
    job_id = f"JOB-{bay_id}-{day_str}-{seq:02d}"
    label = vehicle_label or f"{vehicle_type.capitalize()} in {bay_id.replace('_', ' ').title()}"
    iso_now = _iso(now)
    conn.execute(
        """
        INSERT INTO vehicle_jobs (
            job_id, bay_id, vehicle_type, vehicle_label, primary_technician,
            status, total_active_seconds, total_break_seconds, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, 'IN_PROGRESS', 0, 0, ?, ?)
        """,
        (job_id, bay_id, vehicle_type, label, primary_technician, iso_now, iso_now),
    )
    conn.commit()
    return job_id


def update_vehicle_job_activity(
    conn: sqlite3.Connection | None,
    job_id: str,
    active_dt: float,
    break_dt: float = 0.0,
    technician_name: str | None = None,
    timestamp: datetime | None = None,
    status: str = "IN_PROGRESS",
) -> None:
    if conn is None or not job_id:
        return
    now = timestamp or datetime.now()
    day_str = now.date().isoformat()
    iso_now = _iso(now)

    conn.execute(
        """
        UPDATE vehicle_jobs
        SET total_active_seconds = total_active_seconds + ?,
            total_break_seconds = total_break_seconds + ?,
            primary_technician = COALESCE(?, primary_technician),
            status = ?,
            updated_at = ?
        WHERE job_id = ?
        """,
        (max(0.0, active_dt), max(0.0, break_dt), technician_name, status, iso_now, job_id),
    )

    # Upsert daily log for multi-day reporting
    tech_key = technician_name or "Unassigned"
    conn.execute(
        """
        INSERT INTO daily_vehicle_job_logs (job_id, day, technician_name, active_seconds, break_seconds)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(job_id, day, technician_name) DO UPDATE SET
            active_seconds = active_seconds + excluded.active_seconds,
            break_seconds = break_seconds + excluded.break_seconds
        """,
        (job_id, day_str, tech_key, max(0.0, active_dt), max(0.0, break_dt)),
    )
    conn.commit()


def complete_vehicle_job(
    conn: sqlite3.Connection | None,
    job_id: str,
    timestamp: datetime | None = None,
) -> bool:
    if conn is None or not job_id:
        return False
    now = timestamp or datetime.now()
    cur = conn.execute(
        """
        UPDATE vehicle_jobs
        SET status = 'COMPLETED',
            completed_at = ?,
            updated_at = ?
        WHERE job_id = ?
        """,
        (_iso(now), _iso(now), job_id),
    )
    conn.commit()
    return bool(cur.rowcount and cur.rowcount > 0)


def list_vehicle_jobs(conn: sqlite3.Connection | None, status: str | None = None) -> list[dict[str, Any]]:
    if conn is None:
        return []
    query = "SELECT * FROM vehicle_jobs"
    params: tuple[Any, ...] = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY updated_at DESC"
    rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_vehicle_job_history(conn: sqlite3.Connection | None, job_id: str) -> dict[str, Any] | None:
    if conn is None or not job_id:
        return None
    job = conn.execute("SELECT * FROM vehicle_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if not job:
        return None
    daily_rows = conn.execute(
        "SELECT * FROM daily_vehicle_job_logs WHERE job_id = ? ORDER BY day ASC",
        (job_id,),
    ).fetchall()
    daily_logs = [
        {
            "day": r["day"],
            "technician_name": r["technician_name"],
            "active_seconds": round(float(r["active_seconds"] or 0), 2),
            "break_seconds": round(float(r["break_seconds"] or 0), 2),
            "active_hours": round(float(r["active_seconds"] or 0) / 3600.0, 2),
        }
        for r in daily_rows
    ]
    res = dict(job)
    res["daily_logs"] = daily_logs
    res["total_active_hours"] = round(float(job["total_active_seconds"] or 0) / 3600.0, 2)
    return res



def record_ai_audit_verdict(
    conn: sqlite3.Connection | None,
    bay_id: str,
    technician_name: str | None,
    action: str,
    category: str,
    confidence: float,
    explanation: str = "",
    crop_path: str | None = None,
    ts: str | None = None,
) -> None:
    if conn is None:
        return
    if ts is None:
        ts = datetime.now().isoformat()
    conn.execute(
        """
        INSERT INTO ai_audit_events (ts, bay_id, technician_name, action, category, confidence, explanation, crop_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, bay_id, technician_name or "unknown", action, category, float(confidence), explanation, crop_path),
    )
    conn.commit()


def get_recent_ai_audits(conn: sqlite3.Connection | None, bay_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    if conn is None:
        return []
    query = "SELECT * FROM ai_audit_events"
    params: list[Any] = []
    if bay_id:
        query += " WHERE bay_id = ?"
        params.append(bay_id)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def _as_datetime(value: float | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromtimestamp(float(value))


def upsert_anonymous_subject(
    conn: sqlite3.Connection | None,
    subject_id: str,
    timestamp: float | datetime,
) -> None:
    if conn is None or not subject_id:
        return
    stamp = _iso(_as_datetime(timestamp))
    row = conn.execute("SELECT id FROM anonymous_subjects WHERE id = ?", (subject_id,)).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO anonymous_subjects (id, local_track_key, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?)
            """,
            (subject_id, subject_id, stamp, stamp),
        )
    else:
        conn.execute(
            "UPDATE anonymous_subjects SET last_seen_at = ? WHERE id = ?",
            (stamp, subject_id),
        )
    conn.commit()


def start_customer_visit(
    conn: sqlite3.Connection | None,
    visit_id: str,
    subject_id: str,
    zone_id: str,
    timestamp: float | datetime,
) -> None:
    if conn is None or not visit_id:
        return
    stamp = _iso(_as_datetime(timestamp))
    conn.execute(
        """
        INSERT OR IGNORE INTO customer_visits (id, subject_id, zone_id, started_at, source)
        VALUES (?, ?, ?, ?, 'edge')
        """,
        (visit_id, subject_id, zone_id, stamp),
    )
    conn.commit()


def end_customer_visit(
    conn: sqlite3.Connection | None,
    visit_id: str,
    timestamp: float | datetime,
) -> None:
    if conn is None or not visit_id:
        return
    stamp_dt = _as_datetime(timestamp)
    stamp = _iso(stamp_dt)
    duration: float | None = None
    try:
        row = conn.execute("SELECT started_at FROM customer_visits WHERE id = ?", (visit_id,)).fetchone()
        if row and row["started_at"]:
            try:
                start_dt = datetime.fromisoformat(row["started_at"])
                duration = max(0.0, (stamp_dt - start_dt).total_seconds())
            except Exception:
                pass
    except Exception:
        pass

    if duration is not None:
        conn.execute(
            "UPDATE customer_visits SET ended_at = ?, duration_seconds = ? WHERE id = ? AND ended_at IS NULL",
            (stamp, round(duration, 1), visit_id),
        )
    else:
        conn.execute(
            "UPDATE customer_visits SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
            (stamp, visit_id),
        )
    conn.commit()


def reset_all_customer_visits(conn: sqlite3.Connection | None) -> None:
    """Reset every single customer visit and anonymous subject in the database."""
    if conn is None:
        return
    conn.execute("DELETE FROM customer_visits")
    conn.execute("DELETE FROM anonymous_subjects")
    conn.commit()


def update_subject_alias(
    conn: sqlite3.Connection | None,
    subject_id: str,
    alias: str | None,
) -> None:
    """Assign a human-friendly nickname or alias to an anonymous customer."""
    if conn is None or not subject_id:
        return
    val = alias.strip() if alias and alias.strip() else None
    conn.execute(
        "UPDATE anonymous_subjects SET alias = ? WHERE id = ?",
        (val, subject_id),
    )
    conn.commit()


def save_subject_avatar(
    conn: sqlite3.Connection | None,
    subject_id: str,
    avatar_path: str,
) -> None:
    """Store the local thumbnail avatar path for a customer."""
    if conn is None or not subject_id:
        return
    conn.execute(
        "UPDATE anonymous_subjects SET avatar_path = ? WHERE id = ?",
        (avatar_path, subject_id),
    )
    conn.commit()


def merge_customer_subjects(
    conn: sqlite3.Connection | None,
    source_id: str,
    target_id: str,
) -> bool:
    """Merge source_id visitor profile into target_id visitor profile.
    
    Combines visit history, dwell times, first/last seen stamps, aliases, and deletes source_id.
    """
    if conn is None or not source_id or not target_id or source_id == target_id:
        return False
    try:
        # Re-link all visits from source to target
        conn.execute(
            "UPDATE customer_visits SET subject_id = ? WHERE subject_id = ?",
            (target_id, source_id),
        )
        # Fetch metadata from both subjects
        src_row = conn.execute(
            "SELECT alias, avatar_path, first_seen_at, last_seen_at FROM anonymous_subjects WHERE id = ?",
            (source_id,),
        ).fetchone()
        tgt_row = conn.execute(
            "SELECT alias, avatar_path, first_seen_at, last_seen_at FROM anonymous_subjects WHERE id = ?",
            (target_id,),
        ).fetchone()

        if tgt_row is not None:
            src_alias = (src_row["alias"] if src_row else None) or ""
            tgt_alias = tgt_row["alias"] or ""
            alias = tgt_alias if tgt_alias.strip() else src_alias

            src_avatar = (src_row["avatar_path"] if src_row else None) or ""
            tgt_avatar = tgt_row["avatar_path"] or ""
            avatar = tgt_avatar if tgt_avatar.strip() else src_avatar

            dates_first = [d for d in [tgt_row["first_seen_at"], src_row["first_seen_at"] if src_row else None] if d]
            first_seen = min(dates_first) if dates_first else tgt_row["first_seen_at"]

            dates_last = [d for d in [tgt_row["last_seen_at"], src_row["last_seen_at"] if src_row else None] if d]
            last_seen = max(dates_last) if dates_last else tgt_row["last_seen_at"]

            conn.execute(
                """
                UPDATE anonymous_subjects
                SET alias = ?, avatar_path = ?, first_seen_at = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (alias or None, avatar or None, first_seen, last_seen, target_id),
            )
        elif src_row is not None:
            conn.execute(
                "UPDATE anonymous_subjects SET id = ? WHERE id = ?",
                (target_id, source_id),
            )

        # Delete source subject
        conn.execute("DELETE FROM anonymous_subjects WHERE id = ?", (source_id,))
        conn.commit()
        return True
    except Exception as exc:
        print(f"[merge_customer_subjects] Error: {exc}", flush=True)
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def get_visits_calendar_summary(
    conn: sqlite3.Connection | None,
    month_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Return visit counts and unique customers aggregated by date (YYYY-MM-DD)."""
    if conn is None:
        return []
    try:
        sql = """
            SELECT 
                SUBSTR(started_at, 1, 10) AS visit_date,
                COUNT(*) AS total_visits,
                COUNT(DISTINCT subject_id) AS unique_visitors,
                ROUND(AVG(CASE WHEN duration_seconds IS NOT NULL AND duration_seconds > 0 THEN duration_seconds ELSE NULL END), 1) AS avg_dwell_seconds
            FROM customer_visits
            WHERE started_at IS NOT NULL
        """
        params: list[Any] = []
        if month_prefix:
            sql += " AND started_at LIKE ?"
            params.append(f"{month_prefix}%")
        sql += " GROUP BY SUBSTR(started_at, 1, 10) ORDER BY visit_date DESC"
        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "date": r["visit_date"],
                "total_visits": int(r["total_visits"] or 0),
                "unique_visitors": int(r["unique_visitors"] or 0),
                "avg_dwell_seconds": float(r["avg_dwell_seconds"] or 0.0),
            }
            for r in rows
            if r["visit_date"]
        ]
    except Exception as exc:
        print(f"[get_visits_calendar_summary] Error: {exc}", flush=True)
        return []


def get_detailed_visits_report(
    conn: sqlite3.Connection | None,
    filter_range: str = "today",
    date_filter: str | None = None,
    now: datetime | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """In-depth customer report with avatars, frequencies, stay durations, and journey logs."""
    empty: dict[str, Any] = {
        "summary": {
            "total_unique": 0,
            "today_unique": 0,
            "week_unique": 0,
            "today_visits": 0,
            "week_visits": 0,
            "total_visits": 0,
            "avg_dwell_seconds": 0.0,
            "returning_rate": 0.0,
            "active_now_count": 0,
        },
        "visitors": [],
        "recent_visits": [],
    }
    if conn is None:
        return empty

    try:
        stamp = now or datetime.now()
        today_str = stamp.date().isoformat()
        week_start_str = (stamp - timedelta(days=7)).isoformat(timespec="seconds")

        t_uniq_row = conn.execute("SELECT COUNT(DISTINCT subject_id) FROM customer_visits WHERE started_at LIKE ?", (f"{today_str}%",)).fetchone()
        t_uniq = int(t_uniq_row[0] if t_uniq_row and t_uniq_row[0] else 0)

        t_vis_row = conn.execute("SELECT COUNT(*) FROM customer_visits WHERE started_at LIKE ?", (f"{today_str}%",)).fetchone()
        t_vis = int(t_vis_row[0] if t_vis_row and t_vis_row[0] else 0)

        w_uniq_row = conn.execute("SELECT COUNT(DISTINCT subject_id) FROM customer_visits WHERE started_at >= ?", (week_start_str,)).fetchone()
        w_uniq = int(w_uniq_row[0] if w_uniq_row and w_uniq_row[0] else 0)

        w_vis_row = conn.execute("SELECT COUNT(*) FROM customer_visits WHERE started_at >= ?", (week_start_str,)).fetchone()
        w_vis = int(w_vis_row[0] if w_vis_row and w_vis_row[0] else 0)

        tot_uniq_row = conn.execute("SELECT COUNT(DISTINCT subject_id) FROM customer_visits").fetchone()
        tot_uniq = int(tot_uniq_row[0] if tot_uniq_row and tot_uniq_row[0] else 0)

        tot_vis_row = conn.execute("SELECT COUNT(*) FROM customer_visits").fetchone()
        tot_vis = int(tot_vis_row[0] if tot_vis_row and tot_vis_row[0] else 0)

        avg_row = conn.execute("SELECT AVG(duration_seconds) FROM customer_visits WHERE duration_seconds IS NOT NULL AND duration_seconds > 0").fetchone()
        avg_dwell = float(avg_row[0]) if (avg_row and avg_row[0] is not None) else 0.0

        mult_visits_row = conn.execute(
            "SELECT COUNT(*) FROM (SELECT subject_id FROM customer_visits GROUP BY subject_id HAVING COUNT(*) > 1)"
        ).fetchone()
        mult_visits = int(mult_visits_row[0] if mult_visits_row and mult_visits_row[0] else 0)
        returning_rate = round((mult_visits / max(tot_uniq, 1)) * 100.0, 1) if tot_uniq > 0 else 0.0

        date_uniq = 0
        date_vis = 0
        if date_filter:
            d_u_row = conn.execute("SELECT COUNT(DISTINCT subject_id) FROM customer_visits WHERE started_at LIKE ?", (f"{date_filter}%",)).fetchone()
            date_uniq = int(d_u_row[0] if d_u_row and d_u_row[0] else 0)
            d_v_row = conn.execute("SELECT COUNT(*) FROM customer_visits WHERE started_at LIKE ?", (f"{date_filter}%",)).fetchone()
            date_vis = int(d_v_row[0] if d_v_row and d_v_row[0] else 0)

        limit_val = max(1, min(int(limit or 100), 500))
        where_clause = ""
        subject_params: list[Any] = [f"{today_str}%", week_start_str]
        if date_filter:
            where_clause = "WHERE s.id IN (SELECT DISTINCT subject_id FROM customer_visits WHERE started_at LIKE ?)"
            subject_params.append(f"{date_filter}%")
        subject_params.append(limit_val)

        subject_rows = conn.execute(
            f"""
            SELECT 
                s.id AS subject_id,
                s.alias,
                s.avatar_path,
                s.first_seen_at,
                s.last_seen_at,
                COUNT(v.id) AS total_visits,
                SUM(CASE WHEN v.started_at LIKE ? THEN 1 ELSE 0 END) AS today_visits,
                SUM(CASE WHEN v.started_at >= ? THEN 1 ELSE 0 END) AS week_visits,
                AVG(CASE WHEN v.duration_seconds IS NOT NULL THEN v.duration_seconds ELSE NULL END) AS avg_dwell,
                SUM(CASE WHEN v.duration_seconds IS NOT NULL THEN v.duration_seconds ELSE 0 END) AS total_dwell
            FROM anonymous_subjects s
            LEFT JOIN customer_visits v ON s.id = v.subject_id
            {where_clause}
            GROUP BY s.id
            ORDER BY s.last_seen_at DESC
            LIMIT ?
            """,
            subject_params,
        ).fetchall()

        visitors: list[dict[str, Any]] = []
        if subject_rows:
            subject_ids = [r["subject_id"] for r in subject_rows]
            placeholders = ",".join("?" for _ in subject_ids)
            recent_rows = conn.execute(
                f"""
                SELECT id, subject_id, zone_id, started_at, ended_at, duration_seconds
                FROM customer_visits
                WHERE subject_id IN ({placeholders})
                ORDER BY started_at DESC
                """,
                subject_ids,
            ).fetchall()
            recent_by_subj: dict[str, list[dict[str, Any]]] = {}
            for row in recent_rows:
                sid = row["subject_id"]
                if sid not in recent_by_subj:
                    recent_by_subj[sid] = []
                if len(recent_by_subj[sid]) < 5:
                    recent_by_subj[sid].append(dict(row))

            for r in subject_rows:
                sid = r["subject_id"]
                last_v = recent_by_subj.get(sid, [])
                last_dwell = None
                if last_v and last_v[0].get("duration_seconds") is not None:
                    last_dwell = float(last_v[0]["duration_seconds"])

                visitors.append({
                    "subject_id": sid,
                    "alias": r["alias"],
                    "avatar_path": r["avatar_path"],
                    "first_seen_at": r["first_seen_at"],
                    "last_seen_at": r["last_seen_at"],
                    "total_visits": int(r["total_visits"] or 0),
                    "today_visits": int(r["today_visits"] or 0),
                    "week_visits": int(r["week_visits"] or 0),
                    "avg_dwell_seconds": round(float(r["avg_dwell"] or 0.0), 1),
                    "total_dwell_seconds": round(float(r["total_dwell"] or 0.0), 1),
                    "last_dwell_seconds": last_dwell,
                    "recent_visits": last_v,
                })

        recent_where = ""
        recent_params: list[Any] = []
        if date_filter:
            recent_where = "WHERE v.started_at LIKE ?"
            recent_params.append(f"{date_filter}%")

        recents = conn.execute(
            f"""
            SELECT v.id, v.subject_id, s.alias, s.avatar_path, v.zone_id, v.started_at, v.ended_at, v.duration_seconds
            FROM customer_visits v
            LEFT JOIN anonymous_subjects s ON v.subject_id = s.id
            {recent_where}
            ORDER BY v.started_at DESC
            LIMIT 30
            """,
            recent_params,
        ).fetchall()

        return {
            "summary": {
                "total_unique": tot_uniq,
                "today_unique": t_uniq,
                "week_unique": w_uniq,
                "today_visits": t_vis,
                "week_visits": w_vis,
                "total_visits": tot_vis,
                "avg_dwell_seconds": round(avg_dwell, 1),
                "returning_rate": returning_rate,
                "active_now_count": 0,
                "date_filter": date_filter,
                "date_unique": date_uniq,
                "date_visits": date_vis,
            },
            "visitors": visitors,
            "recent_visits": [dict(r) for r in recents],
        }
    except Exception as exc:
        print(f"[get_detailed_visits_report] {exc}", flush=True)
        return empty


def customer_visit_counts(
    conn: sqlite3.Connection | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    empty = {
        "today_unique": 0,
        "today_visits": 0,
        "week_unique": 0,
        "week_visits": 0,
        "recent": [],
    }
    if conn is None:
        return empty
    stamp = now or datetime.now()
    today = stamp.date().isoformat()
    week_start = (stamp - timedelta(days=7)).isoformat(timespec="seconds")
    today_visits = conn.execute(
        "SELECT COUNT(*) AS n FROM customer_visits WHERE started_at LIKE ?",
        (f"{today}%",),
    ).fetchone()
    today_unique = conn.execute(
        "SELECT COUNT(DISTINCT subject_id) AS n FROM customer_visits WHERE started_at LIKE ?",
        (f"{today}%",),
    ).fetchone()
    week_visits = conn.execute(
        "SELECT COUNT(*) AS n FROM customer_visits WHERE started_at >= ?",
        (week_start,),
    ).fetchone()
    week_unique = conn.execute(
        "SELECT COUNT(DISTINCT subject_id) AS n FROM customer_visits WHERE started_at >= ?",
        (week_start,),
    ).fetchone()
    recent = conn.execute(
        """
        SELECT id, subject_id, zone_id, started_at, ended_at
        FROM customer_visits
        ORDER BY started_at DESC
        LIMIT 50
        """
    ).fetchall()
    return {
        "today_unique": int(today_unique["n"] if today_unique else 0),
        "today_visits": int(today_visits["n"] if today_visits else 0),
        "week_unique": int(week_unique["n"] if week_unique else 0),
        "week_visits": int(week_visits["n"] if week_visits else 0),
        "recent": [dict(r) for r in recent],
    }


def replace_customer_face_embeddings(
    conn: sqlite3.Connection | None,
    subject_id: str,
    embeddings: list[Any],
    *,
    alias: str | None = None,
    visit_count: int | None = None,
    timestamp: float | datetime | None = None,
) -> None:
    """Persist customer face embeddings so identity survives process restart."""
    if conn is None or not subject_id:
        return
    now = timestamp if timestamp is not None else datetime.now()
    stamp = _iso(_as_datetime(now))
    upsert_anonymous_subject(conn, subject_id, now)
    if alias:
        conn.execute("UPDATE anonymous_subjects SET alias = ? WHERE id = ?", (alias, subject_id))
    if visit_count is not None:
        try:
            conn.execute(
                "UPDATE anonymous_subjects SET visit_count = ? WHERE id = ?",
                (int(visit_count), subject_id),
            )
        except sqlite3.OperationalError:
            pass
    conn.execute("DELETE FROM customer_face_embeddings WHERE subject_id = ?", (subject_id,))
    for slot, embedding in enumerate(embeddings or []):
        if embedding is None:
            continue
        if not hasattr(embedding, "tobytes"):
            continue
        arr = embedding.astype("float32")
        blob = bytes(arr.tobytes())
        dim = int(getattr(arr, "size", 0) or 0)
        if not blob or dim <= 0:
            continue
        conn.execute(
            """
            INSERT INTO customer_face_embeddings (subject_id, slot, embedding, dim)
            VALUES (?, ?, ?, ?)
            """,
            (subject_id, slot, blob, dim),
        )
    conn.execute(
        "UPDATE anonymous_subjects SET last_seen_at = ? WHERE id = ?",
        (stamp, subject_id),
    )
    conn.commit()


def list_customer_face_embeddings(conn: sqlite3.Connection | None) -> dict[str, list[Any]]:
    """Return subject_id -> list of L2 face embedding arrays."""
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            """
            SELECT subject_id, slot, embedding, dim
            FROM customer_face_embeddings
            ORDER BY subject_id, slot
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, list[Any]] = {}
    for row in rows:
        blob = bytes(row["embedding"] or b"")
        dim = int(row["dim"] or 0)
        if not blob:
            continue
        try:
            import numpy as np

            embedding = np.frombuffer(blob, dtype=np.float32).copy()
            if dim and embedding.size != dim:
                embedding = embedding[:dim] if embedding.size > dim else embedding
        except Exception:
            continue
        out.setdefault(str(row["subject_id"]), []).append(embedding)
    return out


def upsert_staff_memory(
    conn: sqlite3.Connection | None,
    staff_id: str,
    embedding: Any,
    timestamp: float | datetime,
) -> None:
    """Persist an opaque staff id + appearance embedding. No name or photo."""
    if conn is None or not staff_id:
        return
    if hasattr(embedding, "tobytes"):
        blob = bytes(embedding.astype("float32").tobytes())  # type: ignore[union-attr]
        dim = int(getattr(embedding, "size", 0) or 0)
    elif isinstance(embedding, (bytes, bytearray)):
        blob = bytes(embedding)
        dim = max(0, len(blob) // 4)
    else:
        return
    if not blob:
        return
    stamp = _iso(_as_datetime(timestamp))
    row = conn.execute("SELECT id FROM staff_memory WHERE id = ?", (staff_id,)).fetchone()
    if row is None:
        conn.execute(
            """
            INSERT INTO staff_memory (id, embedding, dim, created_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (staff_id, blob, dim, stamp, stamp),
        )
    else:
        conn.execute(
            """
            UPDATE staff_memory
            SET embedding = ?, dim = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (blob, dim, stamp, staff_id),
        )
    conn.commit()


def list_staff_memory(conn: sqlite3.Connection | None) -> list[dict[str, Any]]:
    if conn is None:
        return []
    rows = conn.execute(
        "SELECT id, embedding, dim, created_at, last_seen_at FROM staff_memory"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        blob = bytes(row["embedding"] or b"")
        dim = int(row["dim"] or 0)
        embedding = None
        if blob:
            try:
                import numpy as np

                embedding = np.frombuffer(blob, dtype=np.float32).copy()
                if dim and embedding.size != dim:
                    embedding = embedding[:dim] if embedding.size > dim else embedding
            except Exception:
                embedding = None
        out.append(
            {
                "id": str(row["id"]),
                "embedding": embedding,
                "dim": dim,
                "created_at": row["created_at"],
                "last_seen_at": row["last_seen_at"],
            }
        )
    return out


def insert_customer_complaint(
    conn: sqlite3.Connection | None,
    complaint_id: str,
    audio_path: str,
    khmer_transcript: str,
    english_transcript: str,
    category: str,
    severity: str,
    summary: str,
    customer_id: str | None = None,
    camera_id: str | None = None,
    audio_source: str = "laptop_microphone",
    screenshot_path: str | None = None,
    timestamp: str | None = None,
    is_complaint: bool = True,
    telegram_sent: bool = False,
    telegram_error: str | None = None,
) -> bool:
    """Insert a customer complaint record into SQLite database."""
    if conn is None:
        return False
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    now_str = datetime.now().isoformat()
    try:
        conn.execute(
            """
            INSERT INTO customer_complaints (
                complaint_id, customer_id, camera_id, audio_source,
                timestamp, audio_path, screenshot_path,
                khmer_transcript, english_transcript,
                is_complaint, category, severity, summary,
                telegram_sent, telegram_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                complaint_id,
                customer_id,
                camera_id,
                audio_source,
                timestamp,
                audio_path,
                screenshot_path,
                khmer_transcript,
                english_transcript,
                1 if is_complaint else 0,
                category,
                severity,
                summary,
                1 if telegram_sent else 0,
                telegram_error,
                now_str,
            ),
        )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[db] insert_customer_complaint error: {exc}")
        return False


def get_recent_complaints(
    conn: sqlite3.Connection | None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve recent customer complaints from SQLite database."""
    if conn is None:
        return []
    try:
        rows = conn.execute(
            """
            SELECT * FROM customer_complaints
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[db] get_recent_complaints error: {exc}")
        return []


def update_complaint_telegram_status(
    conn: sqlite3.Connection | None,
    complaint_id: str,
    sent: bool,
    error: str | None = None,
) -> bool:
    """Update Telegram dispatch status on an existing complaint."""
    if conn is None or not complaint_id:
        return False
    try:
        conn.execute(
            """
            UPDATE customer_complaints
            SET telegram_sent = ?, telegram_error = ?
            WHERE complaint_id = ?
            """,
            (1 if sent else 0, error, complaint_id),
        )
        conn.commit()
        return True
    except Exception as exc:
        print(f"[db] update_complaint_telegram_status error: {exc}")
        return False
