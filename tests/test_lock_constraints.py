from __future__ import annotations

from datetime import date

import pandas as pd
from ortools.sat.python import cp_model

from src.model import ModelContext, build_model


def _make_lock_context(
    *,
    locks_must: pd.DataFrame | None = None,
    locks_forbid: pd.DataFrame | None = None,
) -> ModelContext:
    employees = pd.DataFrame({"employee_id": ["E1"], "role": ["INFERMIERE"]})
    slots = pd.DataFrame(
        {
            "slot_id": [1],
            "shift_code": ["M"],
            "date": [date(2025, 1, 1)],
        }
    )
    calendar = pd.DataFrame({"data": [pd.Timestamp("2025-01-01")]})

    did_of = {date(2025, 1, 1): 0}
    date_of = {0: date(2025, 1, 1)}
    sid_of = {1: 0}
    slot_of = {0: 1}
    slot_date2 = {0: 0}

    bundle = {
        "eid_of": {"E1": 0},
        "emp_of": {0: "E1"},
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": 1,
        "num_slots": 1,
        "num_days": 1,
        "eligible_eids": {0: [0]},
        "slot_date2": slot_date2,
    }

    empty = pd.DataFrame()
    locks_must_df = (
        locks_must.copy()
        if locks_must is not None
        else pd.DataFrame(columns=["employee_id", "slot_id"])
    )
    locks_forbid_df = (
        locks_forbid.copy()
        if locks_forbid is not None
        else pd.DataFrame(columns=["employee_id", "slot_id"])
    )

    return ModelContext(
        cfg={},
        employees=employees,
        slots=slots,
        coverage_roles=empty,
        coverage_totals=empty,
        slot_requirements=empty,
        availability=empty,
        leaves=empty,
        history=empty,
        locks_must=locks_must_df,
        locks_forbid=locks_forbid_df,
        gap_pairs=empty,
        calendars=calendar,
        bundle=bundle,
    )


def test_must_lock_forces_assignment() -> None:
    context = _make_lock_context(
        locks_must=pd.DataFrame({"employee_id": ["E1"], "slot_id": [1]})
    )
    artifacts = build_model(context)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    assign_var = artifacts.assign_vars[(0, 0)]
    assert solver.Value(assign_var) == 1

    artifacts.model.Add(assign_var == 0)
    solver2 = cp_model.CpSolver()
    infeasible_status = solver2.Solve(artifacts.model)
    assert infeasible_status == cp_model.INFEASIBLE


def test_forbid_lock_blocks_assignment() -> None:
    context = _make_lock_context(
        locks_forbid=pd.DataFrame({"employee_id": ["E1"], "slot_id": [1]})
    )
    artifacts = build_model(context)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    assign_var = artifacts.assign_vars[(0, 0)]
    assert solver.Value(assign_var) == 0

    artifacts.model.Add(assign_var == 1)
    solver2 = cp_model.CpSolver()
    infeasible_status = solver2.Solve(artifacts.model)
    assert infeasible_status == cp_model.INFEASIBLE

