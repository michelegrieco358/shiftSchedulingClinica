from __future__ import annotations

from datetime import date

import pandas as pd
from ortools.sat.python import cp_model

from src.model import ModelContext, build_model


def _make_lock_context(
    *,
    locks_must: pd.DataFrame | None = None,
    locks_forbid: pd.DataFrame | None = None,
    slots: pd.DataFrame | None = None,
) -> ModelContext:
    employees = pd.DataFrame({"employee_id": ["E1"], "role": ["INFERMIERE"]})
    slots_df = (
        slots
        if slots is not None
        else pd.DataFrame(
            {
                "slot_id": [1],
                "shift_code": ["M"],
                "date": [date(2025, 1, 1)],
            }
        )
    )
    calendar = pd.DataFrame({"data": slots_df["date"].apply(pd.Timestamp)})

    did_of = {d: idx for idx, d in enumerate(sorted({d for d in slots_df["date"]}))}
    date_of = {idx: d for d, idx in did_of.items()}
    sid_of = {slot_id: idx for idx, slot_id in enumerate(slots_df["slot_id"]) }
    slot_of = {idx: slot_id for slot_id, idx in sid_of.items()}
    slot_date2 = {sid_of[row.slot_id]: did_of[row.date] for row in slots_df.itertuples(index=False)}

    bundle = {
        "eid_of": {"E1": 0},
        "emp_of": {0: "E1"},
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": 1,
        "num_slots": len(slots_df),
        "num_days": len(did_of),
        "eligible_eids": {sid_of[row.slot_id]: [0] for row in slots_df.itertuples(index=False)},
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

    preassignments = pd.DataFrame(columns=["employee_id", "data", "state_code"])

    return ModelContext(
        cfg={},
        employees=employees,
        slots=slots_df,
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
        preassignments=preassignments,
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


def test_must_lock_outside_horizon_is_ignored() -> None:
    context = _make_lock_context(
        locks_must=pd.DataFrame({"employee_id": ["E1"], "slot_id": [99]}),
    )

    artifacts = build_model(context)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_forbid_lock_outside_horizon_is_ignored() -> None:
    context = _make_lock_context(
        locks_forbid=pd.DataFrame({"employee_id": ["E1"], "slot_id": [99]}),
    )

    artifacts = build_model(context)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

