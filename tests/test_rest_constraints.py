from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from ortools.sat.python import cp_model

from loader.calendar import build_calendar
from src.model import ModelContext, build_model


def _make_rest_context(
    *,
    horizon_start: date,
    horizon_end: date,
    slot_days: Iterable[str],
    gap_hours: Iterable[float],
    rest_threshold: float,
    monthly_limit: int | None,
    consecutive_limit: int | None,
    weekly_min: int | None = None,
    biweekly_min: int | None = None,
    employee_weekly_override: int | None = None,
    employee_biweekly_override: int | None = None,
    history_rows: Iterable[dict[str, object]] | None = None,
) -> ModelContext:
    slot_days = list(slot_days)
    gap_hours = list(gap_hours)
    slot_ids = list(range(1, len(slot_days) + 1))
    slot_dates = [pd.to_datetime(day).date() for day in slot_days]

    employees = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "role": ["INFERMIERE"],
            "rest11h_max_monthly_exceptions": [monthly_limit],
            "rest11h_max_consecutive_exceptions": [consecutive_limit],
        }
    )

    if employee_weekly_override is not None:
        employees["weekly_rest_min_days"] = [employee_weekly_override]
    if employee_biweekly_override is not None:
        employees["biweekly_rest_min_days"] = [employee_biweekly_override]

    slots = pd.DataFrame(
        {
            "slot_id": slot_ids,
            "shift_code": ["M"] * len(slot_ids),
            "reparto_id": ["DEP"] * len(slot_ids),
            "date": slot_dates,
            "duration_min": [480] * len(slot_ids),
        }
    )

    gap_rows = []
    for idx, gap in enumerate(gap_hours, start=1):
        if idx >= len(slot_ids):
            break
        gap_rows.append(
            {
                "reparto_id": "DEP",
                "s1_id": slot_ids[idx - 1],
                "s2_id": slot_ids[idx],
                "gap_hours": gap,
            }
        )

    gap_pairs = pd.DataFrame(gap_rows)

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
        sid_of[slot_id]: did_of[slot_dates[i]]
        for i, slot_id in enumerate(slot_ids)
    }
    slot_duration_min = {slot_id: 480 for slot_id in slot_ids}

    defaults_block = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "rest_rules": {"min_between_shifts_h": rest_threshold},
        "defaults": {
            "rest11h": {
                "max_monthly_exceptions": monthly_limit,
                "max_consecutive_exceptions": consecutive_limit,
            },
            "weekly_rest_min_days": weekly_min if weekly_min is not None else 0,
            "biweekly_rest_min_days": biweekly_min if biweekly_min is not None else 0,
        },
    }

    cfg = defaults_block

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
        "history_month_to_date": pd.DataFrame(
            columns=[
                "employee_id",
                "window_start_date",
                "window_end_date",
                "hours_worked_h",
                "absence_hours_h",
                "hours_with_leaves_h",
                "night_shifts_count",
                "rest11_exceptions_count",
            ]
        ),
    }

    history_df = pd.DataFrame(history_rows) if history_rows else empty_df
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
        gap_pairs=gap_pairs,
        calendars=calendar_df,
        preassignments=preassignments,
        bundle=bundle,
    )


def test_rest_violation_allows_assignment_within_limits() -> None:
    context = _make_rest_context(
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 3),
        slot_days=["2025-01-01", "2025-01-02"],
        gap_hours=[8.0],
        rest_threshold=11.0,
        monthly_limit=2,
        consecutive_limit=2,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    x = artifacts.assign_vars

    model.Add(x[(0, sid_of[1])] == 1)
    model.Add(x[(0, sid_of[2])] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    violation_var = artifacts.rest_violation_pairs[(0, sid_of[1], sid_of[2])]
    assert solver.Value(violation_var) == 1


def test_rest_monthly_limit_blocks_excess_exceptions() -> None:
    context = _make_rest_context(
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 3),
        slot_days=["2025-01-01", "2025-01-02"],
        gap_hours=[8.0],
        rest_threshold=11.0,
        monthly_limit=0,
        consecutive_limit=5,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    x = artifacts.assign_vars

    model.Add(x[(0, sid_of[1])] == 1)
    model.Add(x[(0, sid_of[2])] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_rest_consecutive_limit_blocks_back_to_back_exceptions() -> None:
    context = _make_rest_context(
        horizon_start=date(2025, 1, 1),
        horizon_end=date(2025, 1, 4),
        slot_days=["2025-01-01", "2025-01-02", "2025-01-03"],
        gap_hours=[8.0, 8.0],
        rest_threshold=11.0,
        monthly_limit=3,
        consecutive_limit=1,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    x = artifacts.assign_vars

    model.Add(x[(0, sid_of[1])] == 1)
    model.Add(x[(0, sid_of[2])] == 1)
    model.Add(x[(0, sid_of[3])] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_weekly_rest_violation_is_soft() -> None:
    days = [date(2025, 1, 1) + timedelta(days=offset) for offset in range(14)]
    slot_days = [day.isoformat() for day in days]
    history_rows = [
        {"employee_id": "E1", "turno": "R", "data": "2024-12-30"},
        {"employee_id": "E1", "turno": "R", "data": "2024-12-31"},
    ]

    context = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0] * len(slot_days),
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=1,
        biweekly_min=2,
        history_rows=history_rows,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    day_index = artifacts.day_index
    state_vars = artifacts.state_vars
    assign_vars = artifacts.assign_vars

    rest_days = {days[7], days[12]}

    for idx, current_day in enumerate(days):
        slot_idx = sid_of[idx + 1]
        day_idx = day_index[current_day]
        if current_day in rest_days:
            model.Add(assign_vars[(0, slot_idx)] == 0)
            model.Add(state_vars[(0, day_idx, "R")] == 1)
        elif idx <= 6:
            model.Add(assign_vars[(0, slot_idx)] == 1)
            model.Add(state_vars[(0, day_idx, "M")] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    first_week_end = day_index[date(2025, 1, 7)]
    violation_var = artifacts.weekly_rest_violations[(0, first_week_end)]
    assert solver.Value(violation_var) == 1

    second_week_end = day_index[date(2025, 1, 14)]
    violation_second = artifacts.weekly_rest_violations[(0, second_week_end)]
    assert solver.Value(violation_second) == 0


def test_biweekly_rest_constraint_blocks_missing_rest() -> None:
    days = [date(2025, 2, 1) + timedelta(days=offset) for offset in range(14)]
    slot_days = [day.isoformat() for day in days]

    context = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0] * len(slot_days),
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=1,
        biweekly_min=2,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    day_index = artifacts.day_index
    state_vars = artifacts.state_vars
    assign_vars = artifacts.assign_vars

    rest_day = days[5]

    for idx, current_day in enumerate(days):
        slot_idx = sid_of[idx + 1]
        day_idx = day_index[current_day]
        if current_day == rest_day:
            model.Add(assign_vars[(0, slot_idx)] == 0)
            model.Add(state_vars[(0, day_idx, "R")] == 1)
        else:
            model.Add(assign_vars[(0, slot_idx)] == 1)
            model.Add(state_vars[(0, day_idx, "M")] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_holiday_days_count_as_rest() -> None:
    days = [date(2025, 3, 1) + timedelta(days=offset) for offset in range(14)]
    slot_days = [day.isoformat() for day in days]

    context = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0] * len(slot_days),
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=1,
        biweekly_min=2,
        history_rows=[
            {"employee_id": "E1", "turno": "R", "data": "2025-02-24"},
            {"employee_id": "E1", "turno": "R", "data": "2025-02-25"},
        ],
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    day_index = artifacts.day_index
    state_vars = artifacts.state_vars
    assign_vars = artifacts.assign_vars

    holiday_days = {days[2], days[9]}

    for idx, current_day in enumerate(days):
        slot_idx = sid_of[idx + 1]
        day_idx = day_index[current_day]
        if current_day in holiday_days:
            model.Add(assign_vars[(0, slot_idx)] == 0)
            model.Add(state_vars[(0, day_idx, "F")] == 1)
        elif idx <= 6:
            model.Add(assign_vars[(0, slot_idx)] == 1)
            model.Add(state_vars[(0, day_idx, "M")] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    first_week_end = day_index[date(2025, 3, 7)]
    second_week_end = day_index[date(2025, 3, 14)]

    assert solver.Value(artifacts.weekly_rest_violations[(0, first_week_end)]) == 0
    assert solver.Value(artifacts.weekly_rest_violations[(0, second_week_end)]) == 0


def test_biweekly_rest_constraint_waits_for_full_window_without_history() -> None:
    days = [date(2025, 4, 1) + timedelta(days=offset) for offset in range(3)]
    slot_days = [day.isoformat() for day in days]

    context = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0] * len(slot_days),
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=0,
        biweekly_min=2,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    day_index = artifacts.day_index
    state_vars = artifacts.state_vars
    assign_vars = artifacts.assign_vars

    for idx, current_day in enumerate(days):
        slot_idx = sid_of[idx + 1]
        day_idx = day_index[current_day]
        model.Add(assign_vars[(0, slot_idx)] == 1)
        model.Add(state_vars[(0, day_idx, "M")] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_biweekly_rest_constraint_uses_history_window() -> None:
    horizon_start = date(2025, 5, 15)
    days = [horizon_start]
    slot_days = [day.isoformat() for day in days]

    history_rows = []
    for offset in range(1, 14):
        past_day = horizon_start - timedelta(days=offset)
        shift_code = "R" if offset in {3, 11} else "M"
        history_rows.append(
            {"employee_id": "E1", "turno": shift_code, "data": past_day.isoformat()}
        )

    history_rows_insufficient = []
    for offset in range(1, 14):
        past_day = horizon_start - timedelta(days=offset)
        shift_code = "R" if offset == 3 else "M"
        history_rows_insufficient.append(
            {"employee_id": "E1", "turno": shift_code, "data": past_day.isoformat()}
        )

    context = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0],
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=0,
        biweekly_min=2,
        history_rows=history_rows,
    )

    artifacts = build_model(context)
    model = artifacts.model

    sid_of = artifacts.slot_index
    day_index = artifacts.day_index
    state_vars = artifacts.state_vars
    assign_vars = artifacts.assign_vars

    slot_idx = sid_of[1]
    day_idx = day_index[days[0]]
    model.Add(assign_vars[(0, slot_idx)] == 1)
    model.Add(state_vars[(0, day_idx, "M")] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    context_insufficient = _make_rest_context(
        horizon_start=days[0],
        horizon_end=days[-1],
        slot_days=slot_days,
        gap_hours=[24.0],
        rest_threshold=11.0,
        monthly_limit=5,
        consecutive_limit=5,
        weekly_min=0,
        biweekly_min=2,
        history_rows=history_rows_insufficient,
    )

    artifacts_insufficient = build_model(context_insufficient)
    model_insufficient = artifacts_insufficient.model

    sid_of_insufficient = artifacts_insufficient.slot_index
    day_index_insufficient = artifacts_insufficient.day_index
    state_vars_insufficient = artifacts_insufficient.state_vars
    assign_vars_insufficient = artifacts_insufficient.assign_vars

    slot_idx_insufficient = sid_of_insufficient[1]
    day_idx_insufficient = day_index_insufficient[days[0]]
    model_insufficient.Add(assign_vars_insufficient[(0, slot_idx_insufficient)] == 1)
    model_insufficient.Add(state_vars_insufficient[(0, day_idx_insufficient, "M")] == 1)

    solver = cp_model.CpSolver()
    status_insufficient = solver.Solve(model_insufficient)

    assert status_insufficient == cp_model.INFEASIBLE
