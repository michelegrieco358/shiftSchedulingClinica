from datetime import date, timedelta

import pandas as pd
from loader.calendar import build_calendar
from ortools.sat.python import cp_model

from src.model import ModelContext, build_model


SUMMARY_COLUMNS = [
    "employee_id",
    "window_start_date",
    "window_end_date",
    "hours_worked_h",
    "absence_hours_h",
    "hours_with_leaves_h",
    "night_shifts_count",
    "rest11_exceptions_count",
]


def _make_month_summary(horizon_start: date, night_count: int) -> pd.DataFrame:
    if night_count <= 0:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    month_start = horizon_start.replace(day=1)
    window_end = horizon_start - timedelta(days=1)
    return pd.DataFrame(
        {
            "employee_id": ["E1"],
            "window_start_date": [pd.Timestamp(month_start)],
            "window_end_date": [pd.Timestamp(window_end)],
            "hours_worked_h": [0.0],
            "absence_hours_h": [0.0],
            "hours_with_leaves_h": [0.0],
            "night_shifts_count": [night_count],
            "rest11_exceptions_count": [0],
        }
    )


def _make_context(
    *,
    horizon_start: date,
    horizon_end: date,
    slot_days: list[str],
    week_limit: int | None,
    month_limit: int | None,
    consecutive_limit: int | None = None,
    penalty_weight: float | None = None,
    history_night_days: list[str] | None = None,
    month_history_count: int = 0,
) -> ModelContext:
    employees = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "role": ["INFERMIERE"],
            "ore_dovute_mese_h": [pd.NA],
            "max_week_hours_h": [pd.NA],
            "max_month_hours_h": [pd.NA],
            "max_nights_week": [week_limit],
            "max_nights_month": [month_limit],
            "max_consecutive_nights": [consecutive_limit],
        }
    )

    slot_ids = list(range(1, len(slot_days) + 1))
    slot_dates = [pd.to_datetime(day).date() for day in slot_days]
    slots = pd.DataFrame(
        {
            "slot_id": slot_ids,
            "shift_code": ["N"] * len(slot_ids),
            "date": slot_dates,
            "duration_min": [480] * len(slot_ids),
            "is_night": [True] * len(slot_ids),
        }
    )

    calendar_df = build_calendar(horizon_start, horizon_end)
    calendar_dates = (
        pd.to_datetime(calendar_df["data"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
        .dt.date
        .tolist()
    )
    calendar_dates = sorted(dict.fromkeys(calendar_dates))

    did_of = {day: idx for idx, day in enumerate(calendar_dates)}
    date_of = {idx: day for day, idx in did_of.items()}

    eid_of = {"E1": 0}
    emp_of = {0: "E1"}
    sid_of = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}
    slot_of = {idx: slot_id for slot_id, idx in sid_of.items()}
    eligible_eids = {sid_of[slot_id]: [0] for slot_id in slot_ids}
    slot_date2 = {
        sid_of[slot_id]: did_of[slot_date]
        for slot_id, slot_date in zip(slot_ids, slot_dates, strict=False)
    }
    slot_duration_min = {
        slot_id: duration
        for slot_id, duration in zip(slot_ids, [480] * len(slot_ids), strict=False)
    }

    history_rows = []
    for day in history_night_days or []:
        history_rows.append({"data": day, "employee_id": "E1", "turno": "N"})
    history_df = pd.DataFrame(history_rows) if history_rows else pd.DataFrame(columns=["data", "employee_id", "turno"])

    summary_df = _make_month_summary(horizon_start, month_history_count)

    night_cfg: dict[str, object] = {"can_work_night": True}
    if penalty_weight is not None:
        night_cfg["extra_consecutive_penalty_weight"] = penalty_weight

    cfg = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "shift_types": {"night_codes": ["N"]},
        "night": night_cfg,
    }

    empty_df = pd.DataFrame()

    bundle = {
        "eid_of": eid_of,
        "emp_of": emp_of,
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": 1,
        "num_slots": len(slot_ids),
        "num_days": len(did_of),
        "eligible_eids": eligible_eids,
        "slot_date2": slot_date2,
        "slot_duration_min": slot_duration_min,
        "history_month_to_date": summary_df,
    }

    preassignments = pd.DataFrame(columns=["employee_id", "data", "state_code"])

    return ModelContext(
        cfg=cfg,
        employees=employees,
        slots=slots,
        coverage_roles=empty_df,
        coverage_totals=empty_df,
        slot_requirements=empty_df,
        availability=empty_df,
        leaves=empty_df,
        history=history_df,
        locks_must=empty_df,
        locks_forbid=empty_df,
        gap_pairs=empty_df,
        calendars=calendar_df,
        preassignments=preassignments,
        bundle=bundle,
    )


def test_weekly_night_limit_blocks_overassignment() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 6),
        horizon_end=date(2025, 1, 8),
        slot_days=["2025-01-06", "2025-01-07", "2025-01-08"],
        week_limit=2,
        month_limit=None,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)
    model.Add(artifacts.assign_vars[(0, 2)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_weekly_night_limit_includes_history_before_horizon() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 8),
        horizon_end=date(2025, 1, 9),
        slot_days=["2025-01-08"],
        week_limit=2,
        month_limit=None,
        history_night_days=["2025-01-06", "2025-01-07"],
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_night_limit_counts_month_history() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 16),
        horizon_end=date(2025, 1, 18),
        slot_days=["2025-01-16", "2025-01-17"],
        week_limit=None,
        month_limit=3,
        month_history_count=2,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_consecutive_night_limit_blocks_long_streak() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 4),
        slot_days=["2025-01-01", "2025-01-02", "2025-01-03"],
        week_limit=None,
        month_limit=None,
        consecutive_limit=2,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)
    model.Add(artifacts.assign_vars[(0, 2)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_consecutive_night_limit_considers_history() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 3),
        horizon_end=date(2025, 1, 4),
        slot_days=["2025-01-03"],
        week_limit=None,
        month_limit=None,
        consecutive_limit=2,
        history_night_days=["2025-01-01", "2025-01-02"],
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_consecutive_night_penalty_counts_extra_nights() -> None:
    context = _make_context(
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 4),
        slot_days=["2025-01-01", "2025-01-02", "2025-01-03"],
        week_limit=None,
        month_limit=None,
        consecutive_limit=4,
        penalty_weight=1.0,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)
    model.Add(artifacts.assign_vars[(0, 2)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    penalties = artifacts.consecutive_night_penalties
    assert penalties
    slot_date2 = context.bundle["slot_date2"]
    day_indices = [day_idx for _, day_idx in sorted(slot_date2.items())]
    assert solver.Value(penalties[(0, day_indices[0])]) == 0
    assert solver.Value(penalties[(0, day_indices[1])]) == 1
    assert solver.Value(penalties[(0, day_indices[2])]) == 2

    total_var = artifacts.consecutive_night_penalty_totals[0]
    assert solver.Value(total_var) == 3

    weight = artifacts.consecutive_night_penalty_weight
    assert weight == 1.0
