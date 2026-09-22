"""Vehicle Service Lifecycle & Employee Performance Evaluation Engine.

Manages the complete lifecycle of a vehicle in the service bay:
1. ARRIVAL: Car enters bay, identified & job opened.
2. IN_SERVICE: Active wrench time, pose tracking, multi-technician logging.
3. DEPARTURE: Employee drives car out of bay, stopping the timer.
4. PERFORMANCE EVALUATION: Instant performance grading (efficiency %, focus score, breakdown).
5. KNOWLEDGE BASE: Stores repair patterns and standard service benchmarks from training videos/transcripts.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ServiceStage:
    ARRIVAL: str = "ARRIVAL"
    IN_SERVICE: str = "IN_SERVICE"
    ON_BREAK: str = "ON_BREAK"
    DEPARTED: str = "DEPARTED"
    COMPLETED: str = "COMPLETED"


@dataclass
class ServicePerformanceReport:
    job_id: str
    bay_id: str
    vehicle_type: str
    primary_technician: str
    technicians_breakdown: dict[str, dict[str, Any]]
    total_wrench_seconds: float
    total_break_seconds: float
    total_dwell_seconds: float
    efficiency_pct: float
    performance_grade: str
    performance_score: int
    badge: str
    summary_notes: str
    started_at: str
    completed_at: str
    queue_seconds: float = 0.0
    total_bay_time_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "bay_id": self.bay_id,
            "vehicle_type": self.vehicle_type,
            "primary_technician": self.primary_technician,
            "technicians": self.technicians_breakdown,
            "queue_seconds": round(self.queue_seconds, 1),
            "queue_minutes": round(self.queue_seconds / 60.0, 1),
            "total_wrench_seconds": round(self.total_wrench_seconds, 1),
            "total_wrench_minutes": round(self.total_wrench_seconds / 60.0, 1),
            "total_break_seconds": round(self.total_break_seconds, 1),
            "total_break_minutes": round(self.total_break_seconds / 60.0, 1),
            "total_dwell_seconds": round(self.total_dwell_seconds, 1),
            "total_bay_time_seconds": round(self.total_bay_time_seconds or (self.queue_seconds + self.total_dwell_seconds), 1),
            "efficiency_pct": round(self.efficiency_pct, 1),
            "performance_grade": self.performance_grade,
            "performance_score": self.performance_score,
            "badge": self.badge,
            "summary_notes": self.summary_notes,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


def calculate_performance_grade(
    wrench_seconds: float,
    break_seconds: float,
) -> tuple[str, int, str, str]:
    """
    Calculate performance grade (A+, A, B, C), numeric score (0-100),
    badge, and summary notes for the service performed.
    """
    total = max(1.0, wrench_seconds + break_seconds)
    eff_pct = (wrench_seconds / total) * 100.0

    if eff_pct >= 85.0:
        grade = "A+"
        score = 98
        badge = "⭐ Elite Focus"
        notes = "Exceptional active wrench focus with minimal idle/break delay."
    elif eff_pct >= 75.0:
        grade = "A"
        score = 88
        badge = "🟢 High Efficiency"
        notes = "Fast and focused service within standard efficiency targets."
    elif eff_pct >= 60.0:
        grade = "B"
        score = 75
        badge = "🟡 Standard Service"
        notes = "Normal repair workflow with standard tool and parts retrieval breaks."
    else:
        grade = "C"
        score = 55
        badge = "⚠️ Extended Break"
        notes = "Break duration exceeded 40% of total service time. Review parts staging."

    return grade, score, badge, notes


def evaluate_completed_vehicle_job(
    conn: sqlite3.Connection | None,
    job_id: str,
) -> ServicePerformanceReport | None:
    """
    Evaluate a completed vehicle job, compute efficiency and technician
    contributions, and store the evaluation scorecard in SQLite.
    """
    if conn is None:
        return None

    row = conn.execute(
        """
        SELECT job_id, bay_id, vehicle_type, vehicle_label, primary_technician,
               total_active_seconds, total_break_seconds, created_at, completed_at
        FROM vehicle_jobs
        WHERE job_id = ?
        """,
        (job_id,),
    ).fetchone()

    if not row:
        return None

    jid, bay_id, vtype, vlabel, primary_tech, active_sec, break_sec, created_at, completed_at = row
    active_sec = float(active_sec or 0)
    break_sec = float(break_sec or 0)
    completed_at = completed_at or datetime.now().isoformat(timespec="seconds")
    created_at = created_at or completed_at

    # Fetch technician breakdown from daily_vehicle_job_logs
    tech_rows = conn.execute(
        """
        SELECT technician_name, SUM(active_seconds), SUM(break_seconds)
        FROM daily_vehicle_job_logs
        WHERE job_id = ?
        GROUP BY technician_name
        """,
        (job_id,),
    ).fetchall()

    tech_breakdown: dict[str, dict[str, Any]] = {}
    for t_name, t_act, t_brk in tech_rows:
        t_name = t_name or primary_tech or "Employee"
        t_act = float(t_act or 0)
        t_brk = float(t_brk or 0)
        t_total = max(1.0, t_act + t_brk)
        tech_breakdown[t_name] = {
            "active_seconds": round(t_act, 1),
            "break_seconds": round(t_brk, 1),
            "contribution_pct": round((t_act / max(0.1, active_sec)) * 100.0, 1) if active_sec > 0 else 100.0,
            "focus_pct": round((t_act / t_total) * 100.0, 1),
        }

    grade, score, badge, notes = calculate_performance_grade(active_sec, break_sec)
    total_dwell = active_sec + break_sec
    eff_pct = (active_sec / max(1.0, total_dwell)) * 100.0

    report = ServicePerformanceReport(
        job_id=jid,
        bay_id=bay_id,
        vehicle_type=vtype or "car",
        primary_technician=primary_tech or "Employee",
        technicians_breakdown=tech_breakdown,
        total_wrench_seconds=active_sec,
        total_break_seconds=break_sec,
        total_dwell_seconds=total_dwell,
        efficiency_pct=eff_pct,
        performance_grade=grade,
        performance_score=score,
        badge=badge,
        summary_notes=notes,
        started_at=created_at,
        completed_at=completed_at,
        queue_seconds=0.0,
        total_bay_time_seconds=total_dwell,
    )

    # Store evaluation scorecard in DB
    try:
        conn.execute(
            """
            INSERT INTO vehicle_job_evaluations (
                job_id, vehicle_type, primary_technician, technicians_json,
                total_wrench_seconds, total_break_seconds, efficiency_pct,
                performance_grade, performance_score, summary_notes,
                started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                technicians_json=excluded.technicians_json,
                total_wrench_seconds=excluded.total_wrench_seconds,
                total_break_seconds=excluded.total_break_seconds,
                efficiency_pct=excluded.efficiency_pct,
                performance_grade=excluded.performance_grade,
                performance_score=excluded.performance_score,
                summary_notes=excluded.summary_notes,
                completed_at=excluded.completed_at
            """,
            (
                report.job_id,
                report.vehicle_type,
                report.primary_technician,
                json.dumps(report.technicians_breakdown),
                report.total_wrench_seconds,
                report.total_break_seconds,
                report.efficiency_pct,
                report.performance_grade,
                report.performance_score,
                report.summary_notes,
                report.started_at,
                report.completed_at,
            ),
        )
        conn.commit()
    except Exception as e:
        print(f"[ServicePatterns] Error storing job evaluation: {e}")

    return report


# --- Knowledge Base for Training Videos & Transcripts ---
class ServiceKnowledgeBase:
    """Stores service patterns, steps, and target time benchmarks learned from transcripts."""

    def __init__(self):
        self.service_templates: dict[str, dict[str, Any]] = {
            "brake_rotor_and_pad_replacement": {
                "name": "Brake Rotor & Pad Replacement (ChrisFix Standard)",
                "category": "Braking System",
                "target_minutes": 35.0,
                "learned_from": "ChrisFix YouTube Master Tutorial (Disc Brakes)",
                "stages": [
                    {
                        "step": 1,
                        "title": "Caliper Guide Pin Removal & Clamshell Opening",
                        "timestamp": "0:41",
                        "tool": "12mm socket, Flathead screwdriver",
                        "action": "Unscrew top guide pin and pry caliper open like a clamshell to extract worn brake pads."
                    },
                    {
                        "step": 2,
                        "title": "Caliper Bracket & Rotor Removal",
                        "timestamp": "1:27",
                        "tool": "Breaker bar, Ratchet, Caliper hook/stand",
                        "action": "Loosen top & bottom bracket bolts from knuckle. Secure caliper safely to avoid brake line tension. Slide old rotor off."
                    },
                    {
                        "step": 3,
                        "title": "New Rotor Mounting & Bracket Torquing",
                        "timestamp": "2:38",
                        "tool": "Brake cleaner, Loctite Threadlocker, Torque wrench (85 ft-lbs)",
                        "action": "Spray protective oil off new rotor with brake clean. Seat with temporary lug nut. Apply threadlocker and torque bracket bolts to 85 ft-lbs."
                    },
                    {
                        "step": 4,
                        "title": "Master Cylinder Pressure Relief & Piston Compression",
                        "timestamp": "3:48",
                        "tool": "Brake pad tool / C-clamp, Old brake pad",
                        "action": "Open master cylinder cap under hood. Use old brake pad with compressor tool to seat both pistons fully inward."
                    },
                    {
                        "step": 5,
                        "title": "Guide Pin & Boot Silicone Lubrication",
                        "timestamp": "5:10",
                        "tool": "High-temperature silicone lubricant",
                        "action": "Clean guide pins and boots. Apply high-temp silicone (avoids petroleum swelling) to ensure smooth slide travel."
                    },
                    {
                        "step": 6,
                        "title": "Wear Indicator Verification & Pad Installation",
                        "timestamp": "6:44",
                        "tool": "Brake contact grease",
                        "action": "Verify squealer wear indicator clips. Apply light grease only to metal contact points. Slide new pads into caliper clips."
                    },
                    {
                        "step": 7,
                        "title": "Caliper Closure, Guide Pin Torquing & Cap Re-seal",
                        "timestamp": "7:49",
                        "tool": "Torque wrench (20-25 ft-lbs)",
                        "action": "Close caliper clamshell over thick new pads. Tighten pin bolt to 25 ft-lbs. Re-seal master cylinder cap."
                    }
                ],
                "tools_required": [
                    "12mm socket & ratchet",
                    "Breaker bar",
                    "Torque wrench (85 ft-lbs bracket, 25 ft-lbs pin)",
                    "Piston compressor tool",
                    "Flathead screwdriver",
                    "Brake clean spray",
                    "High-temp silicone lubricant",
                    "Loctite Threadlocker"
                ],
                "quality_checkpoints": [
                    "Brake line was never subjected to hanging tension",
                    "Caliper bracket bolts torqued to 85 ft-lbs with Loctite",
                    "Both caliper pistons compressed completely flush",
                    "Silicone used on guide pin boots instead of petroleum grease",
                    "Wear indicator clip attached on inboard pad",
                    "Master cylinder cap sealed after piston compression"
                ]
            },
            "oil_change": {
                "name": "Standard Engine Oil & Filter Service",
                "category": "Routine Maintenance",
                "target_minutes": 25.0,
                "learned_from": "Standard Quick-Lube Procedure",
                "stages": [
                    {"step": 1, "title": "Lift & Underbody Access", "tool": "Lift / Jack", "action": "Position car on lift and position oil drain container under pan."},
                    {"step": 2, "title": "Drain Engine Oil", "tool": "14mm socket / wrench", "action": "Unscrew oil pan drain bolt and let oil drain completely."},
                    {"step": 3, "title": "Replace Oil Filter", "tool": "Oil filter wrench", "action": "Remove old filter, lubricate new rubber gasket with fresh oil, hand tighten."},
                    {"step": 4, "title": "Torque Drain Plug & Refill", "tool": "Torque wrench (25-30 ft-lbs), Funnel", "action": "Reinstall crush washer, torque plug, fill engine with specified oil grade."},
                    {"step": 5, "title": "Level & Leak Inspection", "tool": "Dipstick", "action": "Run engine 30s, verify zero leaks, check dipstick level."}
                ],
                "tools_required": ["Drain pan", "Socket set", "Oil filter wrench", "Funnel", "Torque wrench"]
            },
            "car_tuning": {
                "name": "ECU Stage 1/2 Tuning & Dyno Diagnostic",
                "category": "Performance Tuning",
                "target_minutes": 60.0,
                "learned_from": "Dyno Performance Workshop Workflow",
                "stages": [
                    {"step": 1, "title": "Dyno Strapping & Safety", "tool": "Ratchet straps, Chocks", "action": "Secure vehicle to chassis dyno rollers and connect exhaust extraction."},
                    {"step": 2, "title": "OBD2 & Wideband Sensor Connection", "tool": "OBD2 cable, Wideband O2", "action": "Connect diagnostic laptop and auxiliary wideband lambda sensor."},
                    {"step": 3, "title": "Baseline Dyno Pull", "tool": "Dyno software", "action": "Log baseline horsepower, torque, air-fuel ratio, and boost pressure."},
                    {"step": 4, "title": "ECU Flash & Fuel/Ignition Mapping", "tool": "ECU flashing interface", "action": "Upload revised engine maps adjusting timing and target AFR."},
                    {"step": 5, "title": "Verification Pull & Data Review", "tool": "Data logger", "action": "Full throttle pull, verify knock sensors, finalize dyno graph."}
                ],
                "tools_required": ["OBD2 interface", "Tuning laptop", "Chassis dyno", "Wideband O2 sensor"]
            }
        }

    def register_pattern_from_transcript(self, pattern_key: str, data: dict[str, Any]) -> None:
        """Register a new service pattern learned from video transcripts."""
        self.service_templates[pattern_key] = data
        print(f"[KnowledgeBase] Learned new service pattern: {data.get('name', pattern_key)}")

    def get_template(self, pattern_key: str) -> dict[str, Any] | None:
        return self.service_templates.get(pattern_key)


KNOWLEDGE_BASE = ServiceKnowledgeBase()
