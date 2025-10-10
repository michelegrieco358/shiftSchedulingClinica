from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd


def build_calendar(
    start_date: date, end_date: date, holidays_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    prev_week_start = start_date - timedelta(days=(start_date.isoweekday() - 1))
    month_start = start_date.replace(day=1)
    history_anchor = start_date - timedelta(days=10)
    # Cover both the initial portion of the calendar month and at least the
    # preceding ten days of history required for rest-related checks.
    extended_anchor = min(month_start, history_anchor)
    cal_start = min(prev_week_start, extended_anchor)
    rows = []
    d = cal_start
    while d <= end_date:
        dow_iso = d.isoweekday()
        week_start_date = d - timedelta(days=dow_iso - 1)
        rows.append(
            {
                "data": d.isoformat(),
                "dow_iso": dow_iso,
                "week_start_date": week_start_date.isoformat(),
                "week_id": week_start_date.isoformat(),
                "is_in_horizon": (start_date <= d <= end_date),
            }
        )
        d += timedelta(days=1)
    cal = pd.DataFrame(rows)
    wk_map = {ws: i for i, ws in enumerate(sorted(cal["week_start_date"].unique()))}
    cal["week_idx"] = cal["week_start_date"].map(wk_map)
    cal["cal_start"] = cal_start.isoformat()
    cal["data_dt"] = pd.to_datetime(cal["data"], format="%Y-%m-%d")
    cal["week_start_date_dt"] = pd.to_datetime(cal["week_start_date"], format="%Y-%m-%d")
    cal["cal_start_dt"] = pd.to_datetime(cal_start)
    cal["is_weekend"] = cal["dow_iso"].isin([6, 7])

    holiday_desc_col = pd.Series([""] * len(cal), index=cal.index)
    if holidays_df is not None and not holidays_df.empty:
        holidays_unique = holidays_df.drop_duplicates(subset=["date"], keep="first")
        holiday_map = holidays_unique.set_index("date")["name"]
        holiday_desc_col = cal["data_dt"].map(holiday_map).fillna("")
    cal["holiday_desc"] = holiday_desc_col.astype(str)
    cal["is_weekday_holiday"] = cal["holiday_desc"].str.strip().ne("") & ~cal["is_weekend"]
    return cal


def enrich_shift_slots_calendar(
    shift_slots: pd.DataFrame, holidays: pd.DataFrame
) -> pd.DataFrame:
    """Restituisce una copia degli slot arricchita con informazioni di calendario."""

    required = {"start_dt", "end_dt"}
    missing = required - set(shift_slots.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Colonne mancanti in shift_slots: {missing_str}")

    enriched = shift_slots.copy()

    if enriched.empty:
        enriched["date"] = pd.Series(dtype="object")
        enriched["end_date"] = pd.Series(dtype="object")
        enriched["weekday"] = pd.Series(dtype="Int64")
        enriched["iso_year"] = pd.Series(dtype="Int64")
        enriched["iso_week"] = pd.Series(dtype="Int64")
        enriched["week_id"] = pd.Series(dtype="Int64")
        enriched["duration_hours"] = pd.Series(dtype="float64")
        enriched["is_weekend"] = pd.Series(dtype="boolean")
        enriched["is_holiday"] = pd.Series(dtype="boolean")
        enriched["is_weekend_or_holiday"] = pd.Series(dtype="boolean")
    else:
        enriched["date"] = enriched["start_dt"].dt.date
        enriched["end_date"] = enriched["end_dt"].dt.date

        isocalendar_df = enriched["start_dt"].dt.isocalendar()
        enriched["iso_year"] = isocalendar_df["year"].astype(int)
        enriched["iso_week"] = isocalendar_df["week"].astype(int)
        enriched["weekday"] = enriched["start_dt"].dt.weekday
        enriched["week_id"] = enriched["iso_year"] * 100 + enriched["iso_week"]

        duration = (enriched["end_dt"] - enriched["start_dt"]).dt.total_seconds() / 3600.0
        enriched["duration_hours"] = duration

        enriched["is_weekend"] = enriched["weekday"] >= 5

        holiday_dates: set[date] = set()
        if not holidays.empty and "date" in holidays.columns:
            normalized_holidays = pd.to_datetime(holidays["date"], errors="coerce")
            normalized_holidays = normalized_holidays.dropna()
            if normalized_holidays.dt.tz is not None:
                normalized_holidays = normalized_holidays.dt.tz_convert(None)
            holiday_dates = {dt.date() for dt in normalized_holidays.dt.normalize()}

        enriched["is_holiday"] = enriched["date"].isin(holiday_dates)
        enriched["is_weekend_or_holiday"] = enriched["is_weekend"] | enriched["is_holiday"]

    preferred = [
        col for col in ["slot_id", "reparto_id", "date", "shift_code"] if col in enriched.columns
    ]
    remaining = [c for c in enriched.columns if c not in preferred]
    ordered_cols = preferred + remaining
    return enriched[ordered_cols]


def attach_calendar(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    return df.merge(
        cal[
            [
                "data",
                "dow_iso",
                "week_start_date",
                "week_start_date_dt",
                "week_id",
                "week_idx",
                "is_in_horizon",
                "is_weekend",
                "is_weekday_holiday",
                "holiday_desc",
            ]
        ],
        on="data",
        how="left",
    )
