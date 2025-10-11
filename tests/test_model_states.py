from datetime import date

import pytest

cp_model = pytest.importorskip("ortools.sat.python.cp_model")

import pandas as pd

from src.model import ModelArtifacts, ModelContext, build_model


def _make_basic_context(
    leaves: pd.DataFrame,
    *,
    slots: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    calendar_dates: list[pd.Timestamp] | None = None,
) -> ModelContext:
    employees = pd.DataFrame({"employee_id": ["E1"], "role": ["INFERMIERE"]})

    if calendar_dates is None:
        calendar_dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]
    calendar = pd.DataFrame({"data": calendar_dates})

    if slots is None:
        slots = pd.DataFrame(
            {
                "slot_id": [1],
                "shift_code": ["M"],
                "date": [calendar_dates[0]],
            }
        )
    else:
        slots = slots.copy()

    slots["shift_code"] = slots["shift_code"].astype(str).str.strip().str.upper()
    slots["date"] = pd.to_datetime(slots["date"]).dt.date

    did_of = {pd.to_datetime(ts).date(): idx for idx, ts in enumerate(calendar_dates)}
    date_of = {idx: pd.to_datetime(ts).date() for idx, ts in enumerate(calendar_dates)}

    sid_of = {int(slot_id): idx for idx, slot_id in enumerate(slots["slot_id"].tolist())}
    slot_of = {idx: slot_id for slot_id, idx in sid_of.items()}

    slot_date2 = {}
    for slot_id, slot_date in zip(slots["slot_id"], slots["date"], strict=False):
        slot_idx = sid_of[int(slot_id)]
        slot_date2[slot_idx] = did_of[pd.to_datetime(slot_date).date()]

    bundle = {
        "eid_of": {"E1": 0},
        "emp_of": {0: "E1"},
        "sid_of": sid_of,
        "slot_of": slot_of,
        "did_of": did_of,
        "date_of": date_of,
        "num_employees": 1,
        "num_slots": len(sid_of),
        "num_days": len(did_of),
        "eligible_eids": {idx: [0] for idx in sid_of.values()},
        "slot_date2": slot_date2,
    }

    empty = pd.DataFrame()
    history_df = history if history is not None else empty

    return ModelContext(
        cfg={},
        employees=employees,
        slots=slots,
        coverage_roles=empty,
        coverage_totals=empty,
        slot_requirements=empty,
        availability=empty,
        leaves=leaves,
        history=history_df,
        locks_must=empty,
        locks_forbid=empty,
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


def test_assignment_implies_matching_state() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    context = _make_basic_context(leaves)
    artifacts = build_model(context)

    assign_var = artifacts.assign_vars[(0, 0)]
    artifacts.model.Add(assign_var == 1)

    solver = _solve_model(artifacts)

    assert solver.Value(artifacts.state_vars[(0, 0, "M")]) == 1


def test_state_requires_assignment() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    context = _make_basic_context(leaves)
    artifacts = build_model(context)

    artifacts.model.Add(artifacts.state_vars[(0, 0, "M")] == 1)

    solver = _solve_model(artifacts)

    assert solver.Value(artifacts.assign_vars[(0, 0)]) == 1


def test_state_zero_if_no_matching_slot() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    slots = pd.DataFrame(
        {
            "slot_id": [1],
            "shift_code": ["M"],
            "date": [pd.Timestamp("2025-01-01")],
        }
    )
    context = _make_basic_context(leaves, slots=slots)
    artifacts = build_model(context)

    solver = _solve_model(artifacts)

    assert solver.Value(artifacts.state_vars[(0, 0, "P")]) == 0


def test_sn_requires_previous_night_history() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    history = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "turno": ["N"],
            "data": ["2024-12-31"],
        }
    )
    context = _make_basic_context(
        leaves,
        history=history,
        calendar_dates=[pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
        slots=pd.DataFrame(
            {
                "slot_id": [1],
                "shift_code": ["M"],
                "date": [pd.Timestamp("2025-01-02")],
            }
        ),
    )
    artifacts = build_model(context)

    artifacts.model.Add(artifacts.state_vars[(0, 0, "SN")] == 1)

    solver = _solve_model(artifacts)

    assert solver.Value(artifacts.state_vars[(0, 0, "SN")]) == 1


def test_sn_forbidden_without_previous_night() -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    slots = pd.DataFrame(
        {
            "slot_id": [1],
            "shift_code": ["N"],
            "date": [pd.Timestamp("2025-01-01")],
        }
    )
    context = _make_basic_context(leaves, slots=slots)
    artifacts = build_model(context)

    artifacts.model.Add(artifacts.state_vars[(0, 1, "SN")] == 1)
    artifacts.model.Add(artifacts.state_vars[(0, 0, "N")] == 0)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)

    assert status == cp_model.INFEASIBLE


@pytest.mark.parametrize("state_code", ["R", "M", "F"])
def test_day_after_night_forbids_states(state_code: str) -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    slots = pd.DataFrame(
        {
            "slot_id": [10, 11],
            "shift_code": ["N", "M"],
            "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
        }
    )
    context = _make_basic_context(leaves, slots=slots)
    artifacts = build_model(context)

    n_slot_idx = context.bundle["sid_of"][10]
    artifacts.model.Add(artifacts.assign_vars[(0, n_slot_idx)] == 1)
    day_idx = context.bundle["did_of"][pd.Timestamp("2025-01-02").date()]

    artifacts.model.Add(artifacts.state_vars[(0, day_idx, state_code)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)

    assert status == cp_model.INFEASIBLE


@pytest.mark.parametrize("state_code", ["R", "M", "F"])
def test_history_night_limits_first_day_states(state_code: str) -> None:
    leaves = pd.DataFrame(columns=["employee_id", "date"])
    history = pd.DataFrame(
        {
            "employee_id": ["E1"],
            "turno": ["N"],
            "data": [pd.Timestamp("2024-12-31")],
        }
    )
    context = _make_basic_context(
        leaves,
        history=history,
        calendar_dates=[pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")],
        slots=pd.DataFrame(
            {
                "slot_id": [1],
                "shift_code": ["M"],
                "date": [pd.Timestamp("2025-01-02")],
            }
        ),
    )
    artifacts = build_model(context)

    artifacts.model.Add(artifacts.state_vars[(0, 0, state_code)] == 1)

    solver = cp_model.CpSolver()
    status = solver.Solve(artifacts.model)

    assert status == cp_model.INFEASIBLE
