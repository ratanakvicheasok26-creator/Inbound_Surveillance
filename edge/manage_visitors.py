#!/usr/bin/env python3
"""CLI for Champei visitor alias metadata (visitor_meta SQLite store).

Examples:
  python manage_visitors.py --set visitor_a1f3 --name "Lok Chumteav Sophy" --tier Gold --notes "Prefers Room 3"
  python manage_visitors.py --get visitor_a1f3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage Champei visitor aliases (VIP tier + notes) in local SQLite."
    )
    parser.add_argument(
        "--db",
        default="",
        help="Optional path to events.db (default: edge data_dir/events.db)",
    )
    parser.add_argument("--set", dest="set_id", default="", help="Visitor id to upsert")
    parser.add_argument("--name", default="", help="Human alias / display name")
    parser.add_argument("--tier", default="Standard", help="VIP tier (default Standard)")
    parser.add_argument("--notes", default="", help="Operator notes")
    parser.add_argument("--get", dest="get_id", default="", help="Visitor id to look up")
    args = parser.parse_args(argv)

    db_path = Path(args.db).expanduser() if args.db else None

    from visitor_registry import get_visitor_display_name, set_visitor_alias

    if args.set_id:
        if not args.name.strip():
            print("error: --name is required with --set", file=sys.stderr)
            return 2
        row = set_visitor_alias(
            args.set_id,
            args.name,
            vip_tier=args.tier,
            notes=args.notes,
            db_path=db_path,
        )
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0

    if args.get_id:
        row = get_visitor_display_name(args.get_id, db_path=db_path)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
