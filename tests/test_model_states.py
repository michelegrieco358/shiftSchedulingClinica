from datetime import date

import pytest

cp_model = pytest.importorskip("ortools.sat.python.cp_model")

import pandas as pd

from src.model import ModelArtifacts, ModelContext, build_model


def _make_basic_context(leaves: pd.DataFrame) -> ModelContext:
    employees = pd.DataFrame({"employee_id": ["E1"], "role": ["INFERMIERE"]})
    slots = pd.DataFrame({"slot_id": [1]})
    calendar = pd.DataFrame({"data": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]})

    bundle = {
        "eid_of": {"E1": 0},
        "emp_of": {0: "E1"},
        "sid_of": {1: 0},
        "slot_of": {0: 1},
        "did_of": {date(2025, 1, 1): 0, date(2025, 1, 2): 1},
        "date_of": {0: date(2025, 1, 1), 1: date(2025, 1, 2)},
        "num_employees": 1,
        "num_slots": 1,
        "num_days": 2,
        "eligible_eids": {0: [0]},
    }

    empty = pd.DataFrame()

    return ModelContext(
        cfg={},
        employees=employees,
        slots=slots,
        coverage_roles=empty,
        coverage_totals=empty,
        slot_requirements=empty,
        availability=empty,
        leaves=leaves,
        history=empty,
        preassign_must=empty,
        preassign_forbid=empty,
        gap_pairs=empty,
        calendars=calendar,
        bundle=bundle,
    )


def _solve_model(artifacts: ModelArtifacts) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return solver


def test_daily_state_sum_equals_one() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    context = _make_basic_context(leaves)
    artifacts = build_model(context)

    solver = _solve_model(artifacts)

    state_codes = artifacts.state_codes
    assert state_codes

    for day_idx in range(2):
        total = sum(
            solver.Value(artifacts.state_vars[(0, day_idx, state)]) for state in state_codes
        )
        assert total == 1


def test_absence_forces_absence_state() -> None:
    leaves = pd.DataFrame(
        {"employee_id": ["E1"], "date": [pd.Timestamp("2025-01-01")]}
    )
    context = _make_basic_context(leaves)
    artifacts = build_model(context)

    solver = _solve_model(artifacts)

    assert "F" in artifacts.state_codes
    assert solver.Value(artifacts.state_vars[(0, 0, "F")]) == 1
