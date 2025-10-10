from datetime import date
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

    cfg = {
        "horizon": {
            "start_date": horizon_start.isoformat(),
            "end_date": horizon_end.isoformat(),
        },
        "rest_rules": {"min_between_shifts_h": rest_threshold},
        "defaults": {
            "rest11h": {
                "max_monthly_exceptions": monthly_limit,
                "max_consecutive_exceptions": consecutive_limit,
            }
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

    return ModelContext(
        cfg=cfg,
        employees=employees,
        slots=slots,
        coverage_roles=empty_df,
        coverage_totals=empty_df,
        slot_requirements=empty_df,
        availability=empty_df,
        leaves=empty_df,
        history=empty_df,
        preassign_must=empty_df,
        preassign_forbid=empty_df,
        gap_pairs=gap_pairs,
        calendars=calendar_df,
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
