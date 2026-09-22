import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from edge.db import (
    connect,
    record_face_clock_in,
    record_face_clock_out,
    update_technician_activity,
    get_daily_garage_summary,
)


class TestAttendanceDeduplication(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.db_path = Path(self._tmp.name)
        self.conn = connect(self.db_path)
        self.day = datetime(2026, 9, 5, 8, 0, 0)

    def tearDown(self):
        self.conn.close()
        if self.db_path.exists():
            self.db_path.unlink()

    def test_single_employee_multiple_clockins_deduped(self):
        # Insert first shift for Hour-Meng (morning: 08:00 - 12:00)
        record_face_clock_in(self.conn, "Hour-Meng", self.day)
        update_technician_activity(self.conn, "Hour-Meng", "bay_1", True, 180 * 60, self.day + timedelta(minutes=1))
        record_face_clock_out(self.conn, "Hour-Meng", self.day + timedelta(hours=4))

        # Insert a separate shift row for Hour-Meng directly in technician_shifts
        # with slight whitespace/casing variation in staff_name to simulate separate clock-in
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO technician_shifts (
                day, staff_name, clock_in_time, clock_out_time,
                total_shift_minutes, wrench_minutes, idle_minutes, wifi_active_minutes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.day.date().isoformat(),
                "hour-meng ",
                (self.day + timedelta(hours=5)).isoformat(),
                (self.day + timedelta(hours=9)).isoformat(),
                240.0,
                120.0,
                120.0,
                200.0,
            ),
        )
        self.conn.commit()

        # Add a second technician (Sothunly)
        sothun_start = self.day + timedelta(minutes=30)
        record_face_clock_in(self.conn, "Sothunly", sothun_start)
        update_technician_activity(self.conn, "Sothunly", "bay_2", True, 240 * 60, sothun_start + timedelta(minutes=1))
        record_face_clock_out(self.conn, "Sothunly", sothun_start + timedelta(hours=8))

        # Query daily garage summary
        summary = get_daily_garage_summary(self.conn, self.day.date())
        techs = summary.get("technicians", [])

        # Verify only 2 technician summaries exist, NOT 3
        self.assertEqual(len(techs), 2)

        hour_meng = next((t for t in techs if t["staff_name"].lower() == "hour-meng"), None)
        self.assertIsNotNone(hour_meng)
        # Shift minutes must be sum of both shifts: 240 + 240 = 480 min
        self.assertAlmostEqual(hour_meng["total_shift_minutes"], 480.0, places=0)
        # Wrench minutes must be sum of both shifts: 180 + 120 = 300 min
        self.assertAlmostEqual(hour_meng["wrench_minutes"], 300.0, places=0)
        # Earliest clock-in must be 08:00
        self.assertIn("08:00", hour_meng["clock_in_time"])
        # Latest clock-out must be 17:00
        self.assertIn("17:00", hour_meng["clock_out_time"])

        sothunly = next((t for t in techs if t["staff_name"].lower() == "sothunly"), None)
        self.assertIsNotNone(sothunly)
        self.assertAlmostEqual(sothunly["wrench_minutes"], 240.0, places=0)


if __name__ == "__main__":
    unittest.main()
