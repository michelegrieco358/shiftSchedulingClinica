from __future__ import annotations

from datetime import date

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


def _make_history_summary(hours_with_leaves: float, month_start: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "employee_id": ["E1"],
            "window_start_date": [pd.Timestamp(month_start)],
            "window_end_date": [pd.Timestamp(month_start) + pd.Timedelta(days=13)],
            "hours_worked_h": [hours_with_leaves],
            "absence_hours_h": [0.0],
            "hours_with_leaves_h": [hours_with_leaves],
            "night_shifts_count": [0],
            "rest11_exceptions_count": [0],
        }
    )


def _make_context(
    *,
    slot_specs: list[tuple[str, int]],
    due_hours: float,
    horizon_start: date,
    horizon_end: date,
    max_week_hours: float | None = None,
    max_month_hours: float | None = None,
    history_entries: list[tuple[str, int]] | None = None,
    history_summary: pd.DataFrame | None = None,
) -> ModelContext:
    employees_dict: dict[str, list] = {
        "employee_id": ["E1"],
        "role": ["INFERMIERE"],
        "ore_dovute_mese_h": [due_hours],
    }
    employees_dict["max_week_hours_h"] = [max_week_hours]
    employees_dict["max_month_hours_h"] = [max_month_hours]
    employees = pd.DataFrame(employees_dict)

    slot_ids = list(range(1, len(slot_specs) + 1))
    slot_dates = [pd.to_datetime(day).date() for day, _ in slot_specs]
    durations = [minutes for _, minutes in slot_specs]

    slots = pd.DataFrame(
        {
            "slot_id": slot_ids,
            "shift_code": ["M"] * len(slot_ids),
            "date": slot_dates,
            "duration_min": durations,
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

    slot_date2 = {sid_of[slot_id]: did_of[slot_date] for slot_id, slot_date in zip(slot_ids, slot_dates, strict=False)}
    slot_duration_min = {slot_id: duration for slot_id, duration in zip(slot_ids, durations, strict=False)}

    eligible_eids = {sid_of[slot_id]: [0] for slot_id in slot_ids}

    history_df = pd.DataFrame(
        history_entries,
        columns=["data", "shift_duration_min"],
    ) if history_entries else pd.DataFrame(columns=["data", "shift_duration_min"])
    if not history_df.empty:
        history_df["employee_id"] = "E1"

    summary_df = history_summary if history_summary is not None else pd.DataFrame(columns=SUMMARY_COLUMNS)

    cfg = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "defaults": {
            "contract_hours_by_role_h": {"INFERMIERE": due_hours},
            "absences": {"count_as_worked_hours": True},
        },
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
        preassign_must=empty_df,
        preassign_forbid=empty_df,
        gap_pairs=empty_df,
        calendars=calendar_df,
        bundle=bundle,
    )


def test_weekly_hours_cap_blocks_overassignment() -> None:
    context = _make_context(
        slot_specs=[("2025-01-06", 360), ("2025-01-07", 360)],
        due_hours=168,
        horizon_start=date(2025, 1, 6),
        horizon_end=date(2025, 1, 7),
        max_week_hours=10,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_weekly_hours_cap_includes_history_before_horizon() -> None:
    context = _make_context(
        slot_specs=[("2025-01-08", 300)],
        due_hours=168,
        horizon_start=date(2025, 1, 8),
        horizon_end=date(2025, 1, 9),
        max_week_hours=10,
        history_entries=[("2025-01-06", 360)],
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_hours_cap_blocks_overassignment() -> None:
    context = _make_context(
        slot_specs=[("2025-01-02", 360), ("2025-01-03", 360), ("2025-01-04", 360)],
        due_hours=168,
        horizon_start=date(2025, 1, 2),
        horizon_end=date(2025, 1, 4),
        max_month_hours=10,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)
    model.Add(artifacts.assign_vars[(0, 2)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    assert status == cp_model.INFEASIBLE


def test_monthly_due_slack_reflects_difference() -> None:
    context = _make_context(
        slot_specs=[("2025-01-02", 180), ("2025-01-03", 120)],
        due_hours=10,
        horizon_start=date(2025, 1, 2),
        horizon_end=date(2025, 1, 3),
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    key = next(iter(artifacts.monthly_hour_deviation.keys()))
    deviation = artifacts.monthly_hour_deviation[key]
    balance = artifacts.monthly_hour_balance[key]

    assert solver.Value(deviation) == 300
    assert solver.Value(balance) == -300


def test_monthly_due_includes_history_summary() -> None:
    summary = _make_history_summary(hours_with_leaves=6.0, month_start="2025-01-01")
    context = _make_context(
        slot_specs=[("2025-01-16", 240), ("2025-01-17", 240)],
        due_hours=20,
        horizon_start=date(2025, 1, 16),
        horizon_end=date(2025, 1, 17),
        history_summary=summary,
    )

    artifacts = build_model(context)
    model = artifacts.model
    model.Add(artifacts.assign_vars[(0, 0)] == 1)
    model.Add(artifacts.assign_vars[(0, 1)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    key = next(iter(artifacts.monthly_hour_deviation.keys()))
    deviation = artifacts.monthly_hour_deviation[key]
    balance = artifacts.monthly_hour_balance[key]

    # History contributes 360 minutes, plan 480 minutes => deficit 360 minutes
    assert solver.Value(deviation) == 360
    assert solver.Value(balance) == -360
