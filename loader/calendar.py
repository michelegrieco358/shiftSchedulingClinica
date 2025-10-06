from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd


def build_calendar(
    start_date: date, end_date: date, holidays_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    prev_week_start = start_date - timedelta(days=(start_date.isoweekday() - 1))
    six_days_before = start_date - timedelta(days=6)
    cal_start = min(prev_week_start, six_days_before)
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
        holidays_unique = holidays_df.drop_duplicates(subset=["data"], keep="first")
        cal = cal.merge(
            holidays_unique[["data", "descrizione"]].rename(
                columns={"descrizione": "holiday_desc"}
            ),
            on="data",
            how="left",
        )
        holiday_desc_col = cal["holiday_desc"].fillna("")
    cal["holiday_desc"] = holiday_desc_col.astype(str)
    cal["is_weekday_holiday"] = cal["holiday_desc"].str.strip().ne("")
    return cal


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
