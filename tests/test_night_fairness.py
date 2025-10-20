from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.model import (
    NIGHT_FAIRNESS_OBJECTIVE_SCALE,
    ModelContext,
    _build_night_fairness_objective_terms,
)

cp_model = pytest.importorskip("ortools.sat.python.cp_model")


def _build_context(fairness_weight: float) -> ModelContext:
    employees = pd.DataFrame(
        {
            "employee_id": ["E1", "E2"],
            "role": ["INFERMIERE", "INFERMIERE"],
            "reparto_id": ["D1", "D1"],
            "night_weight": [1.0, 1.0],
            "can_work_night": [True, True],
        }
    )

    slots = pd.DataFrame(
        {
            "slot_id": [1],
            "shift_code": ["N"],
            "reparto_id": ["D1"],
            "coverage_code": ["BASE"],
            "date": [date(2025, 1, 1)],
            "duration_min": [480],
            "is_night": [True],
        }
    )

    empty = pd.DataFrame()
    cfg = {
        "horizon": {
            "start_date": "2025-01-01",
            "end_date": "2025-01-01",
        },
        "shift_types": {"night_codes": ["N"]},
        "fairness": {"night_penalty_weight": fairness_weight},
    }

    eid_of = {"E1": 0, "E2": 1}
    bundle = {
        "eid_of": eid_of,
        "emp_of": {0: "E1", 1: "E2"},
        "sid_of": {1: 0},
        "slot_of": {0: 1},
        "did_of": {date(2025, 1, 1): 0},
        "date_of": {0: date(2025, 1, 1)},
        "slot_date2": {0: 0},
        "slot_reparto": {0: "D1"},
        "num_employees": 2,
        "num_slots": 1,
        "num_days": 1,
    }

    return ModelContext(
        cfg=cfg,
        employees=employees,
        slots=slots,
        coverage_roles=empty,
        coverage_totals=empty,
        slot_requirements=empty,
        availability=empty,
        leaves=empty,
        history=empty,
        locks_must=empty,
        locks_forbid=empty,
        gap_pairs=empty,
        calendars=empty,
        preassignments=empty,
        bundle=bundle,
    )


def _objective_terms_for_weight(weight: float) -> list[tuple[str, int]]:
    context = _build_context(weight)
    model = cp_model.CpModel()
    night_vars = {
        (0, 0): model.NewBoolVar("night_e0_d0"),
        (1, 0): model.NewBoolVar("night_e1_d0"),
    }
    night_info = {"night_by_day": night_vars}

    terms, _metadata = _build_night_fairness_objective_terms(
        model, context, context.bundle, night_info
    )
    if not terms:
        return []

    model.Minimize(sum(terms))
    proto = model.Proto()
    var_names = {idx: var.name for idx, var in enumerate(proto.variables)}
    return [
        (var_names.get(var_idx, ""), coeff)
        for var_idx, coeff in zip(proto.objective.vars, proto.objective.coeffs, strict=False)
        if "night_fair_dev" in var_names.get(var_idx, "")
    ]


def test_night_fairness_adds_weighted_objective_terms() -> None:
    fairness_terms = _objective_terms_for_weight(1.75)
    assert fairness_terms
    expected_coeff = int(round(1.75 * NIGHT_FAIRNESS_OBJECTIVE_SCALE))
    assert expected_coeff > 0
    assert all(coeff == expected_coeff for _, coeff in fairness_terms)
    assert {name for name, _ in fairness_terms} == {
        "night_fair_dev_units_e0_D1",
        "night_fair_dev_units_e1_D1",
    }


def test_night_fairness_skipped_with_zero_weight() -> None:
    fairness_terms = _objective_terms_for_weight(0.0)
    assert fairness_terms == []
