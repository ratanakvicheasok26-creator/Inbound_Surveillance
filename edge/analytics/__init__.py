"""Champei hospitality analytics: daily scorecard and silent churn."""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_daily_scorecard",
    "send_daily_scorecard",
    "compute_churn_risks",
    "send_weekly_churn",
]


def __getattr__(name: str) -> Any:
    if name in ("build_daily_scorecard", "send_daily_scorecard"):
        from analytics.scorecard import build_daily_scorecard, send_daily_scorecard

        mapping = {
            "build_daily_scorecard": build_daily_scorecard,
            "send_daily_scorecard": send_daily_scorecard,
        }
        return mapping[name]
    if name in ("compute_churn_risks", "send_weekly_churn"):
        from analytics.churn import compute_churn_risks, send_weekly_churn

        mapping = {
            "compute_churn_risks": compute_churn_risks,
            "send_weekly_churn": send_weekly_churn,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
