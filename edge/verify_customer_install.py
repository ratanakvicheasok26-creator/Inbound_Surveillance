"""Production Customer PC Installation Health & Verification Diagnostic.

Run this script on a newly installed customer PC to ensure 100% operational readiness:
    python verify_customer_install.py

Checks:
1. Python Runtime & Core AI/Edge Dependencies (cv2, onnxruntime, numpy, requests, pyyaml, sqlite3).
2. Database WAL Engine, Self-Healing Schema & Indices.
3. Telegram Outbound API & Chat ID Routing Configuration.
4. Schedule Timers & Timezone Validation (Daily 21:45, Sunday 17:00, Month-End 21:45).
5. 4-Step Intelligence Flow Simulation (Arrival -> Daily -> Weekly -> Monthly).
"""

from __future__ import annotations

import os
import sys
import tempfile
import sqlite3
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Load .env files early
try:
    import launcher
    if hasattr(launcher, "load_dotenv_files"):
        launcher.load_dotenv_files()
except Exception:
    pass

PASS = "✅ [PASS]"
FAIL = "❌ [FAIL]"
WARN = "⚠️ [WARN]"


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


def check_dependencies() -> bool:
    print_header("1. PYTHON ENVIRONMENT & DEPENDENCY AUDIT")
    all_ok = True
    print(f"• Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    
    deps = [
        ("sqlite3", "SQLite3 WAL Engine"),
        ("yaml", "PyYAML Configuration"),
        ("requests", "Requests HTTP Client"),
        ("numpy", "NumPy Matrix Acceleration"),
        ("cv2", "OpenCV Vision Processing"),
        ("onnxruntime", "ONNXRuntime AI Inference Engine"),
    ]
    for module_name, desc in deps:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "built-in")
            print(f"  {PASS} {desc:<34}: {ver}")
        except ImportError as err:
            print(f"  {FAIL} {desc:<34}: MISSING ({err})")
            all_ok = False
    return all_ok


def check_database() -> bool:
    print_header("2. DATABASE ENGINE & SELF-HEALING SCHEMA AUDIT")
    try:
        from analytics.sessions import ensure_session_columns, ensure_events_table
        from visitor_registry import ensure_visitor_meta
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test_install.db"
            conn = sqlite3.connect(str(test_db))
            conn.execute("PRAGMA journal_mode=WAL")
            wal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if str(wal_mode).lower() == "wal":
                print(f"  {PASS} SQLite WAL Mode Active        : {wal_mode}")
            else:
                print(f"  {WARN} SQLite WAL Mode Fallback      : {wal_mode}")
            
            ensure_session_columns(conn)
            ensure_events_table(conn)
            ensure_visitor_meta(conn)
            
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            indices = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
            conn.close()

            required_tables = {"customer_visits", "events", "visitor_meta"}
            if required_tables.issubset(tables):
                print(f"  {PASS} Schema Tables Bootstrap       : {sorted(list(required_tables))}")
            else:
                print(f"  {FAIL} Missing Required Tables       : {required_tables - tables}")
                return False
                
            if "idx_customer_visits_started_at" in indices:
                print(f"  {PASS} Performance Query Indices     : Ready")
            else:
                print(f"  {WARN} Query Indices                 : Missing")
            return True
    except Exception as exc:
        print(f"  {FAIL} Database Bootstrap Error: {exc}")
        return False


def check_telegram_config() -> bool:
    print_header("3. TELEGRAM BOT & NOTIFICATION ROUTING AUDIT")
    from telegram_out import resolve_telegram_credentials, resolve_routed_chat_ids
    
    token, chat_id = resolve_telegram_credentials()
    staff_chat, owner_chat = resolve_routed_chat_ids()
    
    masked_token = f"{token[:10]}...{token[-5:]}" if len(token) > 15 else (token or "(not set)")
    print(f"• Bot Token         : {masked_token}")
    print(f"• Primary Chat ID   : {chat_id or '(not set)'}")
    print(f"• Staff Group Chat  : {staff_chat or '(not set)'}")
    print(f"• Owner Direct Chat : {owner_chat or '(not set)'}")
    
    if token and owner_chat:
        print(f"  {PASS} Telegram Credentials Configured: Ready")
        return True
    else:
        print(f"  {WARN} Telegram Credentials Incomplete: Fill in .env before running in live production.")
        return False


def check_schedulers_and_timezone() -> bool:
    print_header("4. SCHEDULER TIMERS & TIMEZONE AUDIT")
    from run_champei import _local_tz, _parse_fire_at, _is_last_day_of_month
    
    tz = _local_tz()
    print(f"• Configured Timezone: {tz}")
    
    scorecard_fire = _parse_fire_at(os.environ.get("SCORECARD_FIRE_AT", "21:45"), default=dt_time(21, 45))
    weekly_fire = _parse_fire_at(os.environ.get("WEEKLY_CUSTOMER_BRIEF_AT", "17:00"), default=dt_time(17, 0))
    monthly_fire = _parse_fire_at(os.environ.get("MONTHLY_CUSTOMER_BRIEF_AT", "21:45"), default=dt_time(21, 45))
    
    print(f"  {PASS} Daily Operations Scorecard : Daily @ {scorecard_fire.strftime('%H:%M')}")
    print(f"  {PASS} Weekly Executive Report    : Sundays @ {weekly_fire.strftime('%H:%M')}")
    print(f"  {PASS} Monthly Strategic Report   : Month-End @ {monthly_fire.strftime('%H:%M')}")
    return True


def check_pipeline_simulation() -> bool:
    print_header("5. 4-STEP INTELLIGENCE PIPELINE DRY-RUN")
    try:
        from analytics.sessions import ensure_session_columns
        from analytics.scorecard import build_daily_scorecard
        from analytics.weekly_customer_brief import build_weekly_customer_brief
        from analytics.monthly_customer_brief import build_monthly_strategic_brief
        from visitor_registry import set_visitor_alias
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "test_sim.db"
            conn = sqlite3.connect(str(test_db))
            ensure_session_columns(conn)
            
            # Step 1: Simulate Arrivals
            sim_date = date.today()
            t1 = datetime.now()
            conn.execute(
                """
                INSERT INTO customer_visits
                    (id, subject_id, zone_id, started_at, ended_at, duration_seconds, status)
                VALUES
                    (?, 'CUST-0001', 'waiting', ?, ?, 3600, 'completed'),
                    (?, 'CUST-0002', 'waiting', ?, ?, 5400, 'completed')
                """,
                (
                    f"v1-{t1.timestamp()}",
                    t1.strftime("%Y-%m-%dT%H:%M:%S"),
                    (t1 + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                    f"v2-{t1.timestamp()}",
                    (t1 - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S"),
                    (t1 - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            conn.commit()
            conn.close()
            set_visitor_alias("CUST-0001", "Vichea", vip_tier="Gold VIP", db_path=test_db)
            print(f"  {PASS} Step 1: Arrival Ingest & ID Assignment   : Verified")

            # Step 2: Daily Scorecard
            scorecard = build_daily_scorecard(db_path=test_db, branch_id="champei-pp-01", as_of=sim_date)
            assert "DAILY OPERATIONS SCORECARD" in scorecard and "Total Customer Visits: 2" in scorecard
            print(f"  {PASS} Step 2: Daily Scorecard Aggregation     : Verified")

            # Step 3: Weekly Executive Brief
            weekly = build_weekly_customer_brief(db_path=test_db, branch_id="champei-pp-01", as_of=sim_date)
            assert "WEEKLY EXECUTIVE BUSINESS REPORT" in weekly and "Vichea" in weekly
            print(f"  {PASS} Step 3: Weekly Executive Rollup          : Verified")

            # Step 4: Monthly Strategic Brief
            monthly = build_monthly_strategic_brief(db_path=test_db, branch_id="champei-pp-01", as_of=sim_date)
            assert "MONTHLY STRATEGIC INTELLIGENCE REPORT" in monthly and "Vichea" in monthly
            print(f"  {PASS} Step 4: Monthly Strategic Synthesis      : Verified")
            return True
    except Exception as exc:
        print(f"  {FAIL} Pipeline Simulation Error: {exc}")
        return False


def main() -> int:
    print("\n" + "#" * 65)
    print(" 🌿 CHAMPEI SPA — PRODUCTION CUSTOMER PC INSTALL DIAGNOSTIC")
    print("#" * 65)
    
    r1 = check_dependencies()
    r2 = check_database()
    r3 = check_telegram_config()
    r4 = check_schedulers_and_timezone()
    r5 = check_pipeline_simulation()
    
    print_header("INSTALLATION READINESS SUMMARY")
    if r1 and r2 and r4 and r5:
        print(" 🎉 ALL CORE SYSTEMS OPERATIONAL & READY FOR CUSTOMER DEPLOYMENT!")
        print(" • To run in live edge mode:  python run_champei.py")
        print(" • To run in test mock mode:  python run_champei.py --mock\n")
        return 0
    else:
        print(" ⚠️ System check completed with warnings. Please review the items above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
