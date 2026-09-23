"""Champei hospitality analytics: daily scorecard, silent churn, sessions."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_daily_scorecard",
    "send_daily_scorecard",
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
