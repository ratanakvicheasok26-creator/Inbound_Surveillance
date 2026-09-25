"""Champei hospitality analytics: daily scorecard, weekly brief, churn, sessions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_daily_scorecard",
    "send_daily_scorecard",
    "compute_customer_activity_brief",
    "format_customer_activity_brief",
    "build_weekly_customer_brief",
    "send_weekly_customer_brief",
    "compute_churn_risks",
    "send_weekly_churn",
    "record_session_completion",
    "record_walk_away",
]


def __getattr__(name: str) -> Any:
    if name in ("build_daily_scorecard", "send_daily_scorecard"):
        from analytics.scorecard import build_daily_scorecard, send_daily_scorecard

        return {
            "build_daily_scorecard": build_daily_scorecard,
            "send_daily_scorecard": send_daily_scorecard,
        }[name]
    if name in (
        "compute_customer_activity_brief",
        "format_customer_activity_brief",
        "build_weekly_customer_brief",
        "send_weekly_customer_brief",
    ):
        from analytics.weekly_customer_brief import (
            build_weekly_customer_brief,
            compute_customer_activity_brief,
            format_customer_activity_brief,
            send_weekly_customer_brief,
        )

        return {
            "compute_customer_activity_brief": compute_customer_activity_brief,
            "format_customer_activity_brief": format_customer_activity_brief,
            "build_weekly_customer_brief": build_weekly_customer_brief,
            "send_weekly_customer_brief": send_weekly_customer_brief,
        }[name]
    if name in ("compute_churn_risks", "send_weekly_churn"):
        from analytics.churn import compute_churn_risks, send_weekly_churn

        return {
            "compute_churn_risks": compute_churn_risks,
            "send_weekly_churn": send_weekly_churn,
        }[name]
    if name in ("record_session_completion", "record_walk_away"):
        from analytics.sessions import record_session_completion, record_walk_away

        return {
            "record_session_completion": record_session_completion,
            "record_walk_away": record_walk_away,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
