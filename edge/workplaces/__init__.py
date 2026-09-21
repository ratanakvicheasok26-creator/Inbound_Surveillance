"""Workplace contracts shared by the edge engine.

Garage remains one profile, not the whole product. Massage uses a customer-visit
monitor with anonymous re-ID. Occupancy / wrench-time stays in occupancy.py.
"""

from __future__ import annotations

from typing import Any

WORKPLACE_IDS = ("garage", "massage")
SUBJECT_KINDS = ("employee", "customer")
MONITOR_IDS = ("employee_labor", "customer_visits")
GARAGE_ZONE_KINDS = ("vehicle_bay", "tool_area")
MASSAGE_ZONE_KINDS = ("entrance", "waiting", "treatment_room", "reception")
VISIT_ZONE_KINDS = ("entrance", "waiting", "treatment_room")
ALL_ZONE_KINDS = GARAGE_ZONE_KINDS + MASSAGE_ZONE_KINDS
STAFF_LABEL = "Staff"
STAFF_ID_PREFIX = "staff_"
DEFAULT_RECEPTION_ROI = [0.38, 0.06, 0.14, 0.14]

UNKNOWN_PERSON_LABEL = {
    "garage": "Employee",
    "massage": "Visitor",
}

MONITOR_FOR_WORKPLACE = {
    "garage": "employee_labor",
    "massage": "customer_visits",
}

DEFAULT_MASSAGE_ZONES: list[dict[str, Any]] = [
    {"id": "entrance", "name": "Entrance", "roi": [0.05, 0.15, 0.25, 0.70], "type": "entrance"},
    {"id": "waiting", "name": "Waiting", "roi": [0.35, 0.20, 0.28, 0.55], "type": "waiting"},
    {"id": "room_1", "name": "Treatment Room 1", "roi": [0.68, 0.18, 0.28, 0.62], "type": "treatment_room"},
    {
        "id": "reception",
        "name": "Reception",
        "roi": list(DEFAULT_RECEPTION_ROI),
        "type": "reception",
    },
]


def is_visit_zone_kind(kind: object) -> bool:
    return str(kind or "") in VISIT_ZONE_KINDS


def is_staff_id(value: object) -> bool:
    return str(value or "").startswith(STAFF_ID_PREFIX)


def parse_workplace_id(raw: object) -> str:
    value = str(raw or "").strip().lower()
    return value if value in WORKPLACE_IDS else "garage"


def unknown_person_label(workplace: object) -> str:
    return UNKNOWN_PERSON_LABEL[parse_workplace_id(workplace)]


def zone_kinds_for(workplace: object) -> tuple[str, ...]:
    return MASSAGE_ZONE_KINDS if parse_workplace_id(workplace) == "massage" else GARAGE_ZONE_KINDS


def default_zone_kind(workplace: object) -> str:
    return "treatment_room" if parse_workplace_id(workplace) == "massage" else "vehicle_bay"


def parse_zone_kind(raw: object, workplace: object) -> str:
    value = str(raw or "").strip()
    allowed = zone_kinds_for(workplace)
    if value in allowed:
        return value
    if value in ALL_ZONE_KINDS:
        return value
    return default_zone_kind(workplace)


def normalize_workplace_zones(
    workplace: object,
    raw: object,
    fallback_roi: list[float] | None = None,
    *,
    seed_if_empty: bool = True,
) -> list[dict[str, Any]]:
    """Normalize zone geometry for the active workplace.

    Garage delegates to occupancy.normalize_bays so existing garage tests keep
    their seeded Lift Bay defaults. Massage uses entrance/waiting/room/reception kinds.
    """
    workplace_id = parse_workplace_id(workplace)
    if workplace_id == "garage":
        from occupancy import normalize_bays

        return normalize_bays(raw, fallback_roi=fallback_roi, seed_if_empty=seed_if_empty)

    from occupancy import clamp_roi

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            zone_id = str(item.get("id") or "").strip()
            if not zone_id or zone_id in seen:
                continue
            seen.add(zone_id)
            out.append(
                {
                    "id": zone_id,
                    "name": str(item.get("name") or zone_id).strip() or zone_id,
                    "roi": clamp_roi(list(item.get("roi") or [0.30, 0.20, 0.40, 0.60])),
                    "type": parse_zone_kind(item.get("type") or item.get("zone_kind"), workplace_id),
                    "polygon": item.get("polygon"),
                }
            )
    if out or not seed_if_empty:
        return out
    seeded = [{**z, "roi": list(z["roi"])} for z in DEFAULT_MASSAGE_ZONES]
    if fallback_roi:
        seeded[0] = {**seeded[0], "roi": clamp_roi(fallback_roi)}
    return seeded
