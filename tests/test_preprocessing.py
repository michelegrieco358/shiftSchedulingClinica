from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from loader.calendar import build_calendar
from src.preprocessing import build_all


def _make_employees() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E1", "E2"],
            "reparto_id": ["CARD", "CARD"],
            "role": ["INFERMIERE", "INFERMIERE"],
        }
    )


def _make_slots(slot_date: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slot_id": [1],
            "reparto_id": ["CARD"],
            "shift_code": ["M"],
            "coverage_code": ["BASE"],
            "date": [slot_date],
            "duration_min": [420],
            "start_datetime": [pd.Timestamp(f"{slot_date} 07:00:00")],
            "end_datetime": [pd.Timestamp(f"{slot_date} 14:00:00")],
            "is_night": [False],
        }
    )


def _make_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data": [
                "2025-11-03",
                "2025-11-04",
                "2025-11-05",
                "2025-11-06",
            ],
            "data_dt": pd.to_datetime(
                [
                    "2025-11-03",
                    "2025-11-04",
                    "2025-11-05",
                    "2025-11-06",
                ]
            ),
            "employee_id": ["E1", "E1", "E1", "E2"],
            "turno": ["R", "P", "M", "N"],
            "shift_duration_min": [0, 450, 420, 630],
            "shift_start_dt": [
                pd.NaT,
                pd.Timestamp("2025-11-04 14:00:00"),
                pd.Timestamp("2025-11-05 07:00:00"),
                pd.Timestamp("2025-11-06 21:00:00"),
            ],
            "shift_end_dt": [
                pd.NaT,
                pd.Timestamp("2025-11-04 21:30:00"),
                pd.Timestamp("2025-11-05 14:00:00"),
                pd.Timestamp("2025-11-07 07:30:00"),
            ],
        }
    )


def _make_leaves() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E1"],
            "data": ["2025-11-02"],
            "data_dt": pd.to_datetime(["2025-11-02"]),
            "is_absent": [True],
            "absence_hours_h": [7.5],
        }
    )


def _make_cfg(start: date, end: date) -> dict:
    return {
        "horizon": {"start_date": start.isoformat(), "end_date": end.isoformat()},
        "shift_types": {"night_codes": ["N"]},
        "rest_rules": {"min_between_shifts_h": 11},
    }


def test_month_to_date_summary_populated_for_partial_month() -> None:
    start = date(2025, 11, 15)
    end = date(2025, 11, 19)
    cfg = _make_cfg(start, end)
    calendar_df = build_calendar(start, end)

    dfs = {
        "employees_df": _make_employees(),
        "shift_slots_df": _make_slots(start.isoformat()),
        "calendar_df": calendar_df,
        "history_df": _make_history(),
        "leaves_days_df": _make_leaves(),
    }

    bundle = build_all(dfs, cfg)
    summary = bundle["history_month_to_date"].sort_values("employee_id").reset_index(drop=True)

    assert not summary.empty
    assert (summary["window_start_date"] == pd.Timestamp("2025-11-01")).all()
    assert (summary["window_end_date"] == pd.Timestamp("2025-11-14")).all()

    e1 = summary.loc[summary["employee_id"] == "E1"].iloc[0]
    assert e1["hours_worked_h"] == pytest.approx(14.5)
    assert e1["absence_hours_h"] == pytest.approx(7.5)
    assert e1["hours_with_leaves_h"] == pytest.approx(22.0)
    assert e1["night_shifts_count"] == 0
    assert e1["rest11_exceptions_count"] == 1

    e2 = summary.loc[summary["employee_id"] == "E2"].iloc[0]
    assert e2["hours_worked_h"] == pytest.approx(10.5)
    assert e2["absence_hours_h"] == pytest.approx(0.0)
    assert e2["night_shifts_count"] == 1
    assert e2["rest11_exceptions_count"] == 0


def test_month_to_date_summary_empty_when_horizon_starts_month() -> None:
    start = date(2025, 11, 1)
    end = date(2025, 11, 5)
    cfg = _make_cfg(start, end)
    calendar_df = build_calendar(start, end)

    dfs = {
        "employees_df": _make_employees(),
        "shift_slots_df": _make_slots(start.isoformat()),
        "calendar_df": calendar_df,
        "history_df": _make_history(),
        "leaves_days_df": _make_leaves(),
    }

    bundle = build_all(dfs, cfg)
    summary = bundle["history_month_to_date"]

    assert summary.empty
